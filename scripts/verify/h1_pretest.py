"""H1 pre-test: can an error-correcting code on WAM's 32 bits raise message-level recovery under the WAM paper's
multi-watermark conditions? (WAM Sec. 5.5 setup; evaluation transforms of App. D.4 that keep the geometry.)

Setup identical to scripts/verify/oracle_h1.py (same 200 COCO val2017 images, 256x256, k = 1..5 messages, equal 10%
squares in a checkerboard, same-image embedding, tau 0.5, DBSCAN eps 1 / min_samples 1000, WAM's own augmentation
modules, wam_mit.pth). Three message sets are embedded separately:
  raw  : random 32-bit messages (WAM as is)
  c16  : codewords of the extended BCH(32,16), d = 8 (3-bit correction), random 16-bit IDs
  c21  : codewords of the extended BCH(32,21), d = 6 (2-bit correction), random 21-bit IDs
Decoding units per image: WAM DBSCAN clusters, connected components of WAM's detected mask (>= 1000 px, M1), and the
ground-truth areas (oracle). raw is decoded as in WAM (cluster centroid; oracle with WAM's msg_predict_inference).
Codes are decoded by maximum likelihood over the whole codebook: hard (centroid bits) and soft (mean bit logits of the
unit's pixels, correlation with +-1 codewords). A message is recovered when a unit assigned to it (largest pixel
overlap, WAM Sec. 5.5) decodes exactly its 32 bits (raw) or its ID (codes).

Decision rule (fixed before running, as stated to the user): under hflip + contrast 1.5, the code upper bound
UB = share of raw messages whose oracle bit errors are <= 3 must exceed WAM's raw exact recovery by >= 0.15.
The realised gains of the embedded codes are reported alongside."""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_h1 as base  # noqa: E402  (sets sys.path / cwd for the WAM repository)
from PIL import Image  # noqa: E402
from scipy.ndimage import label as cc_label  # noqa: E402
from torchvision import transforms  # noqa: E402
from notebooks.inference_utils import load_model_from_checkpoint  # noqa: E402
from watermark_anything.augmentation.geometric import HorizontalFlip  # noqa: E402
from watermark_anything.augmentation.valuemetric import (JPEG, Brightness, Contrast, GaussianBlur, Hue,  # noqa: E402
                                                         MedianFilter, Saturation)
from watermark_anything.data.metrics import msg_predict_inference  # noqa: E402
from watermark_anything.data.transforms import default_transform  # noqa: E402
from wamset.config import progress  # noqa: E402
from wamset.dbscan_gpu import dbscan_bits  # noqa: E402

OUT = Path("/workspace/wam/artifacts/h1_pretest")
RULE = {"condition": "hflip_contrast1.5", "t": 3, "min_gap": 0.15}
HF, CO, BR, SA, HU, GB, MF, JP = HorizontalFlip(), Contrast(), Brightness(), Saturation(), Hue(), GaussianBlur(), MedianFilter(), JPEG()
CONDITIONS = {  # WAM App. D.4 evaluation values (geometry-preserving) and the Sec. 5.5 / E.2 combinations
    "none": [], "hflip": [(HF, None)], "hflip_contrast1.5": [(HF, None), (CO, 1.5)],
    "hflip_contrast1.5_jpeg80": [(HF, None), (CO, 1.5), (JP, 80)],
    "brightness1.5": [(BR, 1.5)], "brightness2.0": [(BR, 2.0)], "contrast1.5": [(CO, 1.5)], "contrast2.0": [(CO, 2.0)],
    "hue-0.1": [(HU, -0.1)], "hue0.1": [(HU, 0.1)], "saturation1.5": [(SA, 1.5)], "saturation2.0": [(SA, 2.0)],
    "blur3": [(GB, 3)], "blur17": [(GB, 17)], "median3": [(MF, 3)], "median7": [(MF, 7)],
    "jpeg80": [(JP, 80)], "jpeg50": [(JP, 50)],
}


# ------------------------------------------------------------------ extended BCH codes over GF(2^5)
def gf32():
    exp, log = [0] * 62, [0] * 32
    x = 1
    for i in range(31):
        exp[i] = exp[i + 31] = x
        log[x] = i
        x <<= 1
        if x & 32:
            x ^= 0b100101  # primitive polynomial x^5 + x^2 + 1
    return exp, log


def minimal_poly(j, exp, log):
    conj, e = [], j % 31
    while e not in conj:
        conj.append(e)
        e = (e * 2) % 31
    poly = [1]  # coefficients in GF(32), lowest degree first
    for c in conj:
        root = exp[c]
        new = [0] * (len(poly) + 1)
        for i, a in enumerate(poly):
            new[i + 1] ^= a
            if a:
                new[i] ^= exp[(log[a] + log[root]) % 31]
        poly = new
    assert all(a in (0, 1) for a in poly)
    return poly


def gf2_mul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] ^= y
    return out


def bch_codebook(t):
    exp, log = gf32()
    g = [1]
    for j in range(1, 2 * t, 2):
        g = gf2_mul(g, minimal_poly(j, exp, log))
    k = 31 - (len(g) - 1)
    gint = sum(1 << i for i, c in enumerate(g) if c)
    cb = np.zeros(1 << k, dtype=np.int64)
    for i in range(k):
        cb[1 << i:1 << (i + 1)] = cb[:1 << i] ^ (gint << i)
    weight = np.zeros_like(cb)
    v = cb.copy()
    while v.any():
        weight += v & 1
        v >>= 1
    cb = cb | ((weight & 1) << 31)  # overall parity bit -> extended code, length 32
    w = weight + (weight & 1)
    dmin = int(w[1:].min())
    bits = ((cb[:, None] >> np.arange(32)) & 1).astype(np.uint8)
    return k, dmin, bits


# ------------------------------------------------------------------ evaluation
def evaluate(pred, gts, kind, ids, msgs, books):
    """Per message outcome for every decoder. pred (33,H,W); gts (k,H,W) bool; msgs (k,32) 0/1 float on device."""
    k = gts.shape[0]
    logits = pred[1:]
    res = {}
    centers, labels = dbscan_bits(pred, 1000, 1.0)
    lab = torch.as_tensor(labels, device=pred.device)
    dets = [(lab == c, torch.as_tensor(v, device=pred.device).float()) for c, v in centers.items()]
    detected = (torch.sigmoid(pred[0].detach().float().cpu()) > .5).numpy()
    comp, n = cc_label(detected)
    sizes = np.bincount(comp.ravel(), minlength=n + 1)
    comp = torch.as_tensor(comp, device=pred.device)
    ccs = [comp == c for c in range(1, n + 1) if sizes[c] >= 1000]
    unit_sets = {"DBSCAN": [m for m, _ in dets], "M1": ccs, "ORACLE": [gts[j] for j in range(k)]}
    raw_errs = {}
    for uname, units in unit_sets.items():
        decoded = []  # (assigned gt, outcome)
        for u_i, region in enumerate(units):
            if uname == "ORACLE":
                j = u_i
            else:
                ov = (gts & region[None]).flatten(1).sum(1)
                j = int(ov.argmax()) if int(ov.max()) > 0 else -1
            mean_logit = logits[:, region].mean(1)
            if kind == "raw":
                if uname == "DBSCAN":
                    bits = dets[u_i][1]
                else:
                    bits = msg_predict_inference(logits[None], region[None, None].float())[0].float()
                if j >= 0:
                    e = int((bits != msgs[j]).sum())
                    decoded.append((j, {"exact": e == 0, "errors": e}))
                else:
                    decoded.append((j, None))
            else:
                book = books[kind]
                hard = (dets[u_i][1] if uname == "DBSCAN" else (mean_logit > 0).float()) * 2 - 1
                hid = int((book @ hard).argmax())
                sid = int((book @ mean_logit).argmax())
                decoded.append((j, None if j < 0 else {"hard_ok": hid == ids[j], "soft_ok": sid == ids[j]}))
        out = []
        for j in range(k):
            mine = [d for jj, d in decoded if jj == j]
            if kind == "raw":
                errs = [d["errors"] for d in mine]
                out.append({"found": bool(mine), "exact": any(d["exact"] for d in mine), "best_errors": min(errs) if errs else None})
            else:
                out.append({"found": bool(mine), "hard_ok": any(d["hard_ok"] for d in mine), "soft_ok": any(d["soft_ok"] for d in mine),
                            "wrong_soft_units": sum(1 for d in mine if not d["soft_ok"])})
        res[uname] = {"messages": out, "spurious_units": sum(1 for jj, _ in decoded if jj < 0)}
    return res


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    codes = {}
    for name, t in (("c16", 3), ("c21", 2)):
        k_info, dmin, bits = bch_codebook(t)
        codes[name] = {"k": k_info, "dmin": dmin, "bits": bits}
        print(f"[H1 pretest] {name}: extended BCH(32,{k_info}), d_min = {dmin}, codewords = {len(bits)}", flush=True)
    assert codes["c16"]["k"] == 16 and codes["c16"]["dmin"] == 8 and codes["c21"]["k"] == 21 and codes["c21"]["dmin"] == 6
    books = {n: torch.as_tensor(c["bits"], device=device).float() * 2 - 1 for n, c in codes.items()}
    wam = load_model_from_checkpoint(str(base.REPO / "checkpoints" / "params.json"), str(base.CKPT)).to(device).eval()
    files = sorted(base.COCO.glob("*.jpg"))[:base.N_IMAGES]
    resize = transforms.Resize((base.SIZE, base.SIZE))
    imgs = torch.stack([default_transform(resize(Image.open(f).convert("RGB"))) for f in files])
    masks = base.checkerboard_masks(device)
    gen = torch.Generator().manual_seed(base.SEED)
    sets = {"raw": (None, torch.randint(0, 2, (base.N_IMAGES, 5, 32), generator=gen).float())}
    for name, c in codes.items():
        ids = torch.randint(0, 1 << c["k"], (base.N_IMAGES, 5), generator=gen)
        sets[name] = (ids, torch.as_tensor(c["bits"])[ids].float())
    records = defaultdict(list)
    jobs = [(kind, k, s) for kind in sets for k in range(1, 6) for s in range(0, base.N_IMAGES, base.BATCH)]
    for kind, k, s in progress(jobs, desc="H1 pretest (set x k x batch)", unit="batch"):
        ids_all, msgs_all = sets[kind]
        x = imgs[s:s + base.BATCH].to(device)
        msgs = msgs_all[s:s + base.BATCH, :k].to(device)
        with torch.no_grad():
            multi = x.clone()
            for i in range(k):
                w = wam.embed(x, msgs[:, i])["imgs_w"]
                m = masks[i][None, None]
                multi = w * m + multi * (1 - m)
            for cname, ops in CONDITIONS.items():
                gt = masks[:k][None].expand(len(x), -1, -1, -1).clone()
                img, gt = base.augment(multi.clone(), gt, ops)
                preds = wam.detect(img)["preds"].float()
                for b in range(len(x)):
                    ids = None if ids_all is None else [int(v) for v in ids_all[s + b, :k]]
                    r = evaluate(preds[b], gt[b] > .5, kind, ids, msgs[b], books)
                    records[(kind, cname)].append({"image": s + b, "k": k, **r})
    summary = {"codes": {n: {"k": c["k"], "dmin": c["dmin"]} for n, c in codes.items()}, "rule": RULE, "conditions": {}}
    for cname in CONDITIONS:
        row = {}
        raw = records[("raw", cname)]
        for uname in ("DBSCAN", "M1", "ORACLE"):
            msgs = [m for r in raw for m in r[uname]["messages"]]
            errs = [m["best_errors"] for m in msgs if m["best_errors"] is not None]
            row[f"raw_{uname}_exact"] = float(np.mean([m["exact"] for m in msgs]))
            row[f"raw_{uname}_found"] = float(np.mean([m["found"] for m in msgs]))
            if uname == "ORACLE":
                hist = np.bincount(np.array(errs), minlength=33)[:33]
                row["raw_ORACLE_error_hist"] = hist.tolist()
                row["raw_ORACLE_mean_errors"] = float(np.mean(errs))
                for t in (1, 2, 3, 4):
                    row[f"UB_t{t}"] = float(np.mean([e <= t for e in errs]))
        for name in codes:
            rr = records[(name, cname)]
            for uname in ("DBSCAN", "M1", "ORACLE"):
                msgs = [m for r in rr for m in r[uname]["messages"]]
                row[f"{name}_{uname}_hard"] = float(np.mean([m["hard_ok"] for m in msgs]))
                row[f"{name}_{uname}_soft"] = float(np.mean([m["soft_ok"] for m in msgs]))
                row[f"{name}_{uname}_wrong_units"] = int(sum(m["wrong_soft_units"] for m in msgs))
                row[f"{name}_{uname}_spurious"] = int(sum(r[uname]["spurious_units"] for r in rr))
        summary["conditions"][cname] = row
    c = summary["conditions"][RULE["condition"]]
    gap = c[f"UB_t{RULE['t']}"] - c["raw_DBSCAN_exact"]
    summary["verdict"] = {"UB_t3": c["UB_t3"], "WAM_raw_exact": c["raw_DBSCAN_exact"], "gap": gap,
                          "PROCEED": bool(gap >= RULE["min_gap"]),
                          "realised_c16_DBSCAN_soft_minus_WAM": c["c16_DBSCAN_soft"] - c["raw_DBSCAN_exact"],
                          "realised_c21_DBSCAN_soft_minus_WAM": c["c21_DBSCAN_soft"] - c["raw_DBSCAN_exact"]}
    summary["seconds"] = time.perf_counter() - t0
    summary["bootstrap"] = paired_bootstrap(records)
    import gzip
    with gzip.open(OUT / "h1_pretest_records.jsonl.gz", "wt", encoding="utf-8") as f:
        for (kind, cname), rr in records.items():
            for r in rr:
                f.write(json.dumps({"set": kind, "condition": cname, "image": r["image"], "k": r["k"],
                                    **{u: r[u]["messages"] for u in ("DBSCAN", "M1", "ORACLE")}}) + "\n")
    (OUT / "h1_pretest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n[H1 pretest] message recovery pooled over k = 1..5 (200 images each)", flush=True)
    print(f"{'condition':26s} {'oracle_err':>10s} {'UB_t2':>6s} {'UB_t3':>6s} | {'WAM raw':>7s} | "
          f"{'c21 DB-s':>8s} {'c21 M1-s':>8s} {'c21 OR-s':>8s} | {'c16 DB-h':>8s} {'c16 DB-s':>8s} {'c16 M1-s':>8s} {'c16 OR-s':>8s} | wrong c16 DB-s", flush=True)
    for cname, r in summary["conditions"].items():
        print(f"{cname:26s} {r['raw_ORACLE_mean_errors']:10.2f} {r['UB_t2']:6.3f} {r['UB_t3']:6.3f} | {r['raw_DBSCAN_exact']:7.3f} | "
              f"{r['c21_DBSCAN_soft']:8.3f} {r['c21_M1_soft']:8.3f} {r['c21_ORACLE_soft']:8.3f} | "
              f"{r['c16_DBSCAN_hard']:8.3f} {r['c16_DBSCAN_soft']:8.3f} {r['c16_M1_soft']:8.3f} {r['c16_ORACLE_soft']:8.3f} | {r['c16_DBSCAN_wrong_units']}", flush=True)
    print(f"\n[H1 pretest] VERDICT ({RULE['condition']}): {json.dumps(summary['verdict'])}  ({summary['seconds']:.0f}s)", flush=True)
    print(f"\n[H1 pretest] paired image-cluster bootstrap ({BOOT['repeats']} resamples of {base.N_IMAGES} images; "
          f"95% and Bonferroni {100 * (1 - 0.05 / BOOT['family']):.2f}% percentile intervals)", flush=True)
    for cname, comps in summary["bootstrap"].items():
        print(f"  {cname:26s} " + "  ".join(
            f"{name}: {v['diff']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] B[{v['ci_bonf'][0]:+.3f},{v['ci_bonf'][1]:+.3f}]{'*' if v['ci_bonf'][0] > 0 else ''}"
            for name, v in comps.items()), flush=True)


BOOT = {"repeats": 10000, "seed": 20261005, "family": 36,
        "comparisons": {"c16s-WAM": (("c16", "DBSCAN", "soft_ok"), ("raw", "DBSCAN", "exact")),
                        "c21s-WAM": (("c21", "DBSCAN", "soft_ok"), ("raw", "DBSCAN", "exact")),
                        "c16s-c16h": (("c16", "DBSCAN", "soft_ok"), ("c16", "DBSCAN", "hard_ok"))}}


def paired_bootstrap(records):
    """Per image: recovery rate over its 15 messages (k = 1..5); paired differences resampled over images.
    Family for Bonferroni: 18 conditions x 2 codes against WAM (the soft-vs-hard comparison is reported with the
    same level). Pairing: same image, same k and the same checkerboard areas; only the embedded messages differ."""
    rng = np.random.default_rng(BOOT["seed"])
    idx = rng.integers(0, base.N_IMAGES, (BOOT["repeats"], base.N_IMAGES))
    a_lo, a_hi = 0.05 / BOOT["family"] / 2, 1 - 0.05 / BOOT["family"] / 2
    out = {}
    for cname in CONDITIONS:
        rates = {}
        for (kind, cn), rr in records.items():
            if cn != cname:
                continue
            for uname in ("DBSCAN",):
                for field in ("exact", "hard_ok", "soft_ok"):
                    per = np.zeros(base.N_IMAGES)
                    cnt = np.zeros(base.N_IMAGES)
                    ok = False
                    for r in rr:
                        for m in r[uname]["messages"]:
                            if field in m:
                                ok = True
                                per[r["image"]] += m[field]
                                cnt[r["image"]] += 1
                    if ok:
                        rates[(kind, uname, field)] = per / cnt
        comps = {}
        for name, (x, y) in BOOT["comparisons"].items():
            d = rates[x] - rates[y]
            boot = d[idx].mean(1)
            comps[name] = {"diff": float(d.mean()), "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                           "ci_bonf": [float(np.quantile(boot, a_lo)), float(np.quantile(boot, a_hi))],
                           "images_better": int((d > 0).sum()), "images_worse": int((d < 0).sum())}
        out[cname] = comps
    return out


if __name__ == "__main__":
    main()
