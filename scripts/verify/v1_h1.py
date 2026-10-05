"""V1 (development images only): H1 decision under the WAM paper's conditions.

Images: COCO val2017 sorted by name, indices 0-499 (development split of configs/protocol_v5_draft.json).
Scene (WAM Sec. 5.5 / App. D.6): 256x256, k = 1..5 messages, equal 10% squares in a checkerboard, the same image fed to
WAM.embed once per message and the watermarked areas pasted on the original. Extraction: WAM.detect, tau 0.5,
DBSCAN eps 1 / min_samples 1000 (exact GPU equivalent). Transforms: all App. D.4 evaluation values with WAM's own
classes, the App. E.3 combination (JPEG 80 + brightness 1.5 + crop 0.5) and the Sec. 5.5 combinations.
Random transforms (crop location, perspective corners) are seeded per (k, batch, transform) so every message set and
checkpoint sees the same realisation. Ground-truth masks go through the same transform object and are resized to
256x256 (nearest) when the size changes; a message whose visible area is < 1000 px (WAM's min_samples) is excluded
from the denominator and counted separately.

Message sets: RAW (WAM as is), BCH16 (extended BCH(32,16), d 8), BCH21 (extended BCH(32,21), d 6),
RND16 (65,536 distinct uniform random 32-bit words), REP16 (16 bits repeated twice).
Decoders: WAM = DBSCAN centroid (RAW); hard = nearest codeword to the centroid; soft = argmax_c c . l over the codebook,
l = mean bit logits of the DBSCAN cluster. Assignment: largest pixel overlap (WAM Sec. 5.5).

H1 decision (fixed before running, as stated to the user), wam_coco.pth, hflip + contrast 1.5, k = 1..5 pooled,
image-paired bootstrap 10,000, 98.75% two-sided percentile interval:
  FAIL           lower bound of BCH16-soft minus WAM <= 0
  KEEP_NARROWED  lower bound > 0, but the BCH16-soft minus RND16-soft interval contains 0
  KEEP           lower bound > 0 and lower bound of BCH16-soft minus RND16-soft > 0
wam_mit.pth and all other transforms are reported only."""
import gzip
import hashlib
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_h1 as base  # noqa: E402  (sys.path / cwd for the WAM repository)
from h1_pretest import bch_codebook  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402
from notebooks.inference_utils import load_model_from_checkpoint  # noqa: E402
from watermark_anything.augmentation.geometric import Crop, HorizontalFlip, Perspective, Resize, Rotate  # noqa: E402
from watermark_anything.augmentation.valuemetric import (JPEG, Brightness, Contrast, GaussianBlur, Hue,  # noqa: E402
                                                         MedianFilter, Saturation)
from watermark_anything.data.transforms import default_transform  # noqa: E402
from wamset.config import progress, stable_seed  # noqa: E402
from wamset.dbscan_gpu import dbscan_bits  # noqa: E402

OUT = Path("/workspace/wam/artifacts/v1")
CKPTS = {"wam_coco": "https://dl.fbaipublicfiles.com/watermark_anything/wam_coco.pth",
         "wam_mit": "https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth"}
DEV = (0, 500)
BATCH, SEED, MIN_VISIBLE = 25, 20261007, 1000
BOOT = {"repeats": 10000, "seed": 20261006, "level": 0.9875}
DECISION = {"checkpoint": "wam_coco", "condition": "hflip_contrast1.5"}
SETS = ["RAW", "BCH16", "BCH21", "RND16", "REP16"]
HF, CO, BR, SA, HU, GB, MF, JP = HorizontalFlip(), Contrast(), Brightness(), Saturation(), Hue(), GaussianBlur(), MedianFilter(), JPEG()
CR, RS, RO, PE = Crop(), Resize(), Rotate(), Perspective()
CONDITIONS = {
    "none": [], "hflip": [(HF, None)],
    "brightness1.5": [(BR, 1.5)], "brightness2.0": [(BR, 2.0)], "contrast1.5": [(CO, 1.5)], "contrast2.0": [(CO, 2.0)],
    "hue-0.1": [(HU, -0.1)], "hue0.1": [(HU, 0.1)], "saturation1.5": [(SA, 1.5)], "saturation2.0": [(SA, 2.0)],
    "blur3": [(GB, 3)], "blur17": [(GB, 17)], "median3": [(MF, 3)], "median7": [(MF, 7)],
    "jpeg80": [(JP, 80)], "jpeg50": [(JP, 50)],
    "crop0.33": [(CR, 0.33)], "crop0.5": [(CR, 0.5)], "resize0.5": [(RS, 0.5)],
    "rotate-10": [(RO, -10)], "rotate10": [(RO, 10)], "perspective0.1": [(PE, 0.1)], "perspective0.5": [(PE, 0.5)],
    "combination": [(JP, 80), (BR, 1.5), (CR, 0.5)],
    "hflip_contrast1.5": [(HF, None), (CO, 1.5)],
    "hflip_contrast1.5_jpeg80": [(HF, None), (CO, 1.5), (JP, 80)],
}


def log(msg):
    print(f"[V1] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint(name):
    path = base.REPO / "checkpoints" / f"{name}.pth"
    if not path.is_file():
        log(f"downloading {name}.pth")
        tmp = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(CKPTS[name], tmp)
        tmp.replace(path)
    return path, sha256(path)


def codebooks():
    books = {}
    for name, t in (("BCH16", 3), ("BCH21", 2)):
        k, d, bits = bch_codebook(t)
        assert (name, k, d) in (("BCH16", 16, 8), ("BCH21", 21, 6))
        books[name] = bits
    rng = np.random.default_rng(SEED)
    words = set()
    while len(words) < 1 << 16:
        words.update(int(x) for x in rng.integers(0, 1 << 32, size=(1 << 16) - len(words), dtype=np.uint64))
    rnd = np.array(sorted(words), dtype=np.uint64)[: 1 << 16]
    rng.shuffle(rnd)
    books["RND16"] = ((rnd[:, None] >> np.arange(32, dtype=np.uint64)) & 1).astype(np.uint8)
    info = ((np.arange(1 << 16)[:, None] >> np.arange(16)) & 1).astype(np.uint8)
    books["REP16"] = np.concatenate([info, info], 1)
    return books


def min_distance_sample(bits, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    a = torch.as_tensor(bits[rng.choice(len(bits), n, replace=False)]).float() * 2 - 1
    b = torch.as_tensor(bits).float() * 2 - 1
    best = 32
    for s in range(0, n, 500):
        d = (32 - a[s:s + 500] @ b.T) / 2
        d[d == 0] = 99
        best = min(best, int(d.min()))
    return best


def apply(img, gts, ops, seed):
    torch.manual_seed(seed)
    for op, v in ops:
        img, gts = op(img, gts, v) if v is not None else op(img, gts)
    if gts.shape[-1] != base.SIZE or gts.shape[-2] != base.SIZE:
        gts = TF.interpolate(gts, size=(base.SIZE, base.SIZE), mode="nearest")
    return img, gts > .5


def evaluate(pred, gts, kind, ids, msgs, book):
    """Per-message outcomes for one image. gts (k, H, W) bool after the transform."""
    k = gts.shape[0]
    visible = gts.flatten(1).sum(1)
    centers, labels = dbscan_bits(pred, 1000, 1.0)
    lab = torch.as_tensor(labels, device=pred.device)
    logits = pred[1:]
    units = []
    for cid, cen in centers.items():
        region = lab == cid
        ov = (gts & region[None]).flatten(1).sum(1)
        j = int(ov.argmax()) if int(ov.max()) > 0 else -1
        cen = torch.as_tensor(cen, device=pred.device).float()
        if kind == "RAW":
            units.append((j, {"exact": j >= 0 and bool(torch.equal(cen, msgs[j])), "bit_acc": None if j < 0 else float((cen == msgs[j]).float().mean())}))
        else:
            l = logits[:, region].mean(1)
            hid = int((book @ (cen * 2 - 1)).argmax())
            sid = int((book @ l).argmax())
            units.append((j, {"hard": j >= 0 and hid == ids[j], "soft": j >= 0 and sid == ids[j],
                              "bit_acc": None if j < 0 else float((cen == msgs[j]).float().mean())}))
    out = []
    for j in range(k):
        mine = [u for jj, u in units if jj == j]
        rec = {"visible": int(visible[j]), "eligible": int(visible[j]) >= MIN_VISIBLE, "found": bool(mine),
               "bit_acc": [u["bit_acc"] for u in mine]}
        if kind == "RAW":
            rec["exact"] = any(u["exact"] for u in mine)
        else:
            rec["hard"] = any(u["hard"] for u in mine)
            rec["soft"] = any(u["soft"] for u in mine)
        out.append(rec)
    return {"messages": out, "units": len(units), "spurious": sum(1 for jj, _ in units if jj < 0)}


def run_checkpoint(name, imgs, books_gpu, sets, masks):
    path, digest = checkpoint(name)
    wam = load_model_from_checkpoint(str(base.REPO / "checkpoints" / "params.json"), str(path)).to("cuda:0").eval()
    log(f"{name}: sha256 {digest[:16]}..., loaded")
    records = []
    n = len(imgs)
    jobs = [(kind, s) for kind in SETS for s in range(0, n, BATCH)]
    for kind, s in progress(jobs, desc=f"V1 {name} (set x batch)", unit="batch"):
        ids_all, msgs_all = sets[kind]
        x = imgs[s:s + BATCH].cuda()
        msgs = msgs_all[s:s + BATCH].cuda()
        with torch.no_grad():
            wm = [wam.embed(x, msgs[:, i])["imgs_w"] for i in range(5)]
            for k in range(1, 6):
                multi = x.clone()
                for i in range(k):
                    m = masks[i][None, None]
                    multi = wm[i] * m + multi * (1 - m)
                for cname, ops in CONDITIONS.items():
                    gt0 = masks[:k][None].expand(len(x), -1, -1, -1).clone()
                    img, gt = apply(multi.clone(), gt0, ops, stable_seed(SEED, "v1", k, s, cname))
                    preds = wam.detect(img)["preds"].float()
                    for b in range(len(x)):
                        ids = None if ids_all is None else [int(v) for v in ids_all[s + b, :k]]
                        r = evaluate(preds[b], gt[b], kind, ids, msgs[b, :k], books_gpu.get(kind))
                        records.append({"ckpt": name, "set": kind, "condition": cname, "image": DEV[0] + s + b, "k": k, **r})
    del wam
    torch.cuda.empty_cache()
    return records, digest


def image_rates(group, kind, cond, field):
    acc = defaultdict(lambda: [0, 0])
    for r in group[(kind, cond)]:
        for m in r["messages"]:
            if m["eligible"]:
                acc[r["image"]][0] += m[field]
                acc[r["image"]][1] += 1
    return acc


def paired(group, cond, a, b):
    ra, rb = image_rates(group, a[0], cond, a[1]), image_rates(group, b[0], cond, b[1])
    keys = sorted(set(ra) & set(rb))
    d = np.array([ra[i][0] / ra[i][1] - rb[i][0] / rb[i][1] for i in keys if ra[i][1] and rb[i][1]])
    rng = np.random.default_rng(BOOT["seed"])
    boot = d[rng.integers(0, len(d), (BOOT["repeats"], len(d)))].mean(1)
    lo, hi = np.quantile(boot, [(1 - BOOT["level"]) / 2, 1 - (1 - BOOT["level"]) / 2])
    return {"diff": float(d.mean()), "ci": [float(lo), float(hi)], "images": int(len(d))}


def summarize(records):
    group = defaultdict(list)
    for r in records:
        group[(r["set"], r["condition"])].append(r)
    out = {}
    for cname in CONDITIONS:
        row = {}
        for kind in SETS:
            rr = group[(kind, cname)]
            msgs = [m for r in rr for m in r["messages"] if m["eligible"]]
            row[f"{kind}_eligible"] = len(msgs)
            row[f"{kind}_excluded"] = sum(1 for r in rr for m in r["messages"] if not m["eligible"])
            row[f"{kind}_spurious"] = sum(r["spurious"] for r in rr)
            if kind == "RAW":
                row["WAM_exact"] = float(np.mean([m["exact"] for m in msgs]))
                accs = [a for m in msgs for a in m["bit_acc"]]
                row["WAM_bitacc_found_units"] = float(np.mean(accs)) if accs else None
                row["WAM_found"] = float(np.mean([m["found"] for m in msgs]))
            else:
                row[f"{kind}_hard"] = float(np.mean([m["hard"] for m in msgs]))
                row[f"{kind}_soft"] = float(np.mean([m["soft"] for m in msgs]))
        row["H1_BCH16s_minus_WAM"] = paired(group, cname, ("BCH16", "soft"), ("RAW", "exact"))
        row["BCH16s_minus_RND16s"] = paired(group, cname, ("BCH16", "soft"), ("RND16", "soft"))
        row["BCH16s_minus_REP16s"] = paired(group, cname, ("BCH16", "soft"), ("REP16", "soft"))
        out[cname] = row
    return out


def print_table(name, summ):
    log(f"{name}: message exact recovery (eligible messages, k = 1..5 pooled) and paired differences (98.75%)")
    print(f"{'condition':26s} {'excl':>5s} {'WAM':>6s} {'BCH16s':>6s} {'BCH21s':>6s} {'RND16s':>6s} {'REP16s':>6s} {'BCH16h':>6s} | "
          f"{'BCH16s-WAM':>24s} | {'BCH16s-RND16s':>24s} | {'WAM bitacc(found)':>17s}", flush=True)
    for c, r in summ.items():
        h, m = r["H1_BCH16s_minus_WAM"], r["BCH16s_minus_RND16s"]
        print(f"{c:26s} {r['RAW_excluded']:5d} {r['WAM_exact']:6.3f} {r['BCH16_soft']:6.3f} {r['BCH21_soft']:6.3f} {r['RND16_soft']:6.3f} "
              f"{r['REP16_soft']:6.3f} {r['BCH16_hard']:6.3f} | {h['diff']:+.3f} [{h['ci'][0]:+.3f},{h['ci'][1]:+.3f}] | "
              f"{m['diff']:+.3f} [{m['ci'][0]:+.3f},{m['ci'][1]:+.3f}] | {r['WAM_bitacc_found_units'] or 0:17.3f}", flush=True)


def verdict(summ):
    r = summ[DECISION["condition"]]
    h, m = r["H1_BCH16s_minus_WAM"], r["BCH16s_minus_RND16s"]
    if h["ci"][0] <= 0:
        v = "FAIL"
    elif m["ci"][0] > 0:
        v = "KEEP"
    else:
        v = "KEEP_NARROWED"
    return {"verdict": v, "BCH16s_minus_WAM": h, "BCH16s_minus_RND16s": m, **DECISION}


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    books = codebooks()
    for n, b in books.items():
        log(f"{n}: {len(b)} codewords, sampled minimum distance {min_distance_sample(b)}")
    books_gpu = {n: torch.as_tensor(b, device="cuda:0").float() * 2 - 1 for n, b in books.items()}
    files = sorted(base.COCO.glob("*.jpg"))[DEV[0]:DEV[1]]
    resize = transforms.Resize((base.SIZE, base.SIZE))
    imgs = torch.stack([default_transform(resize(Image.open(f).convert("RGB"))) for f in files])
    masks = base.checkerboard_masks(torch.device("cuda:0"))
    gen = torch.Generator().manual_seed(SEED)
    n = len(files)
    sets = {"RAW": (None, torch.randint(0, 2, (n, 5, 32), generator=gen).float())}
    for kind in SETS[1:]:
        ids = torch.randint(0, len(books[kind]), (n, 5), generator=gen)
        sets[kind] = (ids, torch.as_tensor(books[kind])[ids].float())
    result = {"images": [f.name for f in files], "decision_rule": __doc__.split("H1 decision")[1].strip(), "checkpoints": {}}
    for name in ("wam_coco", "wam_mit"):
        records, digest = run_checkpoint(name, imgs, books_gpu, sets, masks)
        with gzip.open(OUT / f"v1_records_{name}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        summ = summarize(records)
        result["checkpoints"][name] = {"sha256": digest, "summary": summ}
        if name == DECISION["checkpoint"]:
            result["H1"] = verdict(summ)
        result["seconds"] = time.perf_counter() - t0
        (OUT / "v1_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print_table(name, summ)
        if name == DECISION["checkpoint"]:
            log(f"H1 VERDICT ({name}, {DECISION['condition']}): {json.dumps(result['H1'])}  ({result['seconds']:.0f}s)")
    log(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
