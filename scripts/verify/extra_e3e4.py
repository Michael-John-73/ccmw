"""Extra analyses E3 and E4 (configs/protocol_v5_addendum_E.json, locked). Reserve images only (val2017 4500-4999).

E3: message sets RAW, BCH16, REP16 and PAD16 (bits 0-15 = 16-bit ID, bits 16-31 = one fixed key) under the WAM
multi-message conditions (k = 1..5, 26 transforms, wam_coco). PAD16 is decoded with the shared evaluator over its
65,536-word codebook: since bits 16-31 are constant, the nearest codeword to the centroid has the centroid's bits
0-15 (PAD16 hard) and the soft maximum-likelihood codeword has the signs of the mean logits at bits 0-15 (PAD16 soft).
E4: per-stage GPU time on the first 100 reserve images (none, hflip_contrast1.5; k = 5; batch size 1)."""
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v4_test as v4  # noqa: E402
from v4_test import base, v1, v3  # noqa: E402
import extra_e1e2 as e12  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402
from notebooks.inference_utils import load_model_from_checkpoint  # noqa: E402
from watermark_anything.data.transforms import default_transform  # noqa: E402
from wamset.config import progress, stable_seed  # noqa: E402
from wamset.dbscan_gpu import dbscan_bits  # noqa: E402

OUT = Path("/workspace/wam/artifacts/extra")
SETS = ["RAW", "BCH16", "REP16", "PAD16"]


def log(msg):
    print(f"[E3/E4] {msg}", flush=True)


def verify():
    al = e12.verify()
    plan = json.loads(e12.ADD.read_text(encoding="utf-8"))
    lock = json.loads(e12.BASE_LOCK.read_text(encoding="utf-8"))
    files = sorted(base.COCO.glob("*.jpg"))[4500:5000]
    import hashlib
    blob = "".join(f"{f.name}:{v3.sha(f)}\n" for f in files).encode()
    if hashlib.sha256(blob).hexdigest() != al["reserve_image_list_sha256"] or al["reserve_image_list_sha256"] != lock["image_lists_sha256"]["reserve"]:
        raise SystemExit("[E3/E4] STOP: reserve image list hash mismatch")
    if v3.sha(base.REPO / "checkpoints" / "wam_coco.pth") != al["checkpoint_wam_coco_sha256"]:
        raise SystemExit("[E3/E4] STOP: wam_coco checkpoint hash mismatch")
    log(f"verified reserve images {len(files)} and wam_coco")
    return al, plan, files


def pad_book(key):
    ids = np.arange(1 << 16)
    info = ((ids[:, None] >> np.arange(16)) & 1).astype(np.uint8)
    return np.concatenate([info, np.broadcast_to(key, (len(ids), 16))], 1)


def run_e3(wam, imgs, books_gpu, sets, masks, seed, fh):
    per_img = defaultdict(lambda: defaultdict(v4.new_acc))
    jobs = [(kind, s) for kind in SETS for s in range(0, len(imgs), v1.BATCH)]
    for kind, s in progress(jobs, desc="E3 (set x batch)", unit="batch"):
        ids_all, msgs_all = sets[kind]
        x = imgs[s:s + v1.BATCH].cuda()
        msgs = msgs_all[s:s + v1.BATCH].cuda()
        with torch.no_grad():
            wm = [wam.embed(x, msgs[:, i])["imgs_w"] for i in range(5)]
            for k in range(1, 6):
                multi = x.clone()
                for i in range(k):
                    mk = masks[i][None, None]
                    multi = wm[i] * mk + multi * (1 - mk)
                for cname, ops in v1.CONDITIONS.items():
                    gt0 = masks[:k][None].expand(len(x), -1, -1, -1).clone()
                    img, gt = v1.apply(multi.clone(), gt0, ops, stable_seed(seed, "v5-reserve-e3", k, s, cname))
                    preds = wam.detect(img)["preds"].float()
                    for b in range(len(x)):
                        ids = None if ids_all is None else [int(v) for v in ids_all[s + b, :k]]
                        res = v1.evaluate(preds[b], gt[b], kind, ids, msgs[b, :k], books_gpu.get(kind))
                        fh.write(json.dumps({"set": kind, "condition": cname, "image": 4500 + s + b, "k": k, **res}) + "\n")
                        v4.add_h1(per_img[(kind, cname)][s + b], res, kind)
    return per_img


def summarize_e3(per_img):
    rate = v4.rate
    out = {"conditions": {}}
    for c in v1.CONDITIONS:
        tot = {}
        for kind in SETS:
            t = v4.new_acc()
            for a in per_img[(kind, c)].values():
                for key, val in a.items():
                    t[key] += val
            tot[kind] = t
        row = {"RAW_exact": rate(tot["RAW"]), "BCH16_soft": rate(tot["BCH16"]), "BCH16_hard": rate(tot["BCH16"], "hard"),
               "REP16_soft": rate(tot["REP16"]), "PAD16_soft": rate(tot["PAD16"]), "PAD16_hard": rate(tot["PAD16"], "hard"),
               "eligible": int(tot["RAW"]["elig"]), "excluded": int(tot["RAW"]["excluded"])}

        def paired(a, fa, b, fb):
            A, B = per_img[(a, c)], per_img[(b, c)]
            keys = [i for i in sorted(A) if A[i]["elig"] and B[i]["elig"]]
            return e12.boot([rate(A[i], fa) - rate(B[i], fb) for i in keys])
        row["BCH16s_minus_PAD16s"] = paired("BCH16", "ok", "PAD16", "ok")
        row["BCH16s_minus_PAD16h"] = paired("BCH16", "ok", "PAD16", "hard")
        row["PAD16s_minus_RAW"] = paired("PAD16", "ok", "RAW", "ok")
        row["BCH16s_minus_RAW"] = paired("BCH16", "ok", "RAW", "ok")
        out["conditions"][c] = row
    out["holm_BCH16s_minus_PAD16s"] = e12.holm({c: r["BCH16s_minus_PAD16s"]["p_one_sided"] for c, r in out["conditions"].items()})
    return out


def run_e4(wam, imgs, bch_pm, pad_key, masks):
    sync = torch.cuda.synchronize
    rng = np.random.default_rng(20261105)
    stages = defaultdict(lambda: defaultdict(list))
    pad_bits = torch.as_tensor(pad_key, device="cuda:0").float()
    for idx in progress(range(100), desc="E4 timing (images)", unit="img"):
        x = imgs[idx:idx + 1].cuda()
        ids = rng.integers(0, 1 << 16, 5)
        bits = (bch_pm[torch.as_tensor(ids, device="cuda:0")] > 0).float()
        with torch.no_grad():
            wm = []
            for i in range(5):
                sync(); t0 = time.perf_counter()
                w = wam.embed(x, bits[i:i + 1])["imgs_w"]
                sync(); dt = time.perf_counter() - t0
                wm.append(w)
                if idx >= 10:
                    stages["any"]["embed_one_message"].append(dt * 1000)
            multi = x.clone()
            for i in range(5):
                mk = masks[i][None, None]
                multi = wm[i] * mk + multi * (1 - mk)
            for cname in ("none", "hflip_contrast1.5"):
                gt0 = masks[:5][None].clone()
                img, _ = v1.apply(multi.clone(), gt0, v1.CONDITIONS[cname], 20261105 + idx)
                sync(); t0 = time.perf_counter()
                pred = wam.detect(img)["preds"].float()[0]
                sync(); t1 = time.perf_counter()
                centers, labels = dbscan_bits(pred, 1000, 1.0)
                lab = torch.as_tensor(labels, device="cuda:0")
                sync(); t2 = time.perf_counter()
                regions = [(lab == cid, torch.as_tensor(cen, device="cuda:0").float()) for cid, cen in centers.items()]
                means = [pred[1:][:, r].mean(1) for r, _ in regions]
                soft = [int((bch_pm @ l).argmax()) for l in means]
                sync(); t3 = time.perf_counter()
                hard = [int((bch_pm @ (c * 2 - 1)).argmax()) for _, c in regions]
                sync(); t4 = time.perf_counter()
                pad = [int(((l[:16] > 0).long() << torch.arange(16, device="cuda:0")).sum()) for l in means]
                sync(); t5 = time.perf_counter()
                if idx >= 10:
                    st = stages[cname]
                    st["detect"].append((t1 - t0) * 1000)
                    st["dbscan_gpu"].append((t2 - t1) * 1000)
                    st["bch16_soft_decode_all_regions"].append((t3 - t2) * 1000)
                    st["bch16_hard_decode_all_regions"].append((t4 - t3) * 1000)
                    st["pad16_decode_all_regions"].append((t5 - t4) * 1000)
                    st["regions"].append(len(regions))
                del soft, hard, pad
    out = {"gpu": torch.cuda.get_device_name(0), "images_timed": 90, "warmup_images": 10, "batch_size": 1, "stages_ms": {}}
    for cname, st in stages.items():
        out["stages_ms"][cname] = {k: ({"median": float(np.median(v)), "p95": float(np.percentile(v, 95))} if k != "regions"
                                       else {"mean": float(np.mean(v))}) for k, v in st.items()}
    return out


def main():
    t0 = time.perf_counter()
    al, plan, files = verify()
    OUT.mkdir(parents=True, exist_ok=True)
    resize = transforms.Resize((base.SIZE, base.SIZE))
    imgs = torch.stack([default_transform(resize(Image.open(f).convert("RGB"))) for f in files])
    masks = base.checkerboard_masks(torch.device("cuda:0"))
    books = v1.codebooks()
    key = np.random.default_rng(plan["E3"]["seeds"]["pad_key"]).integers(0, 2, 16).astype(np.uint8)
    books["PAD16"] = pad_book(key)
    books_gpu = {k: torch.as_tensor(books[k], device="cuda:0").float() * 2 - 1 for k in ("BCH16", "REP16", "PAD16")}
    rng = np.random.default_rng(plan["E3"]["seeds"]["messages"])
    n = len(files)
    sets = {"RAW": (None, torch.as_tensor(rng.integers(0, 2, (n, 5, 32)), dtype=torch.float32))}
    for kind in ("BCH16", "REP16", "PAD16"):
        ids = torch.as_tensor(rng.integers(0, 1 << 16, (n, 5)))
        sets[kind] = (ids, torch.as_tensor(books[kind])[ids].float())
    seed = plan["E3"]["seeds"]["messages"][0]
    wam = load_model_from_checkpoint(str(base.REPO / "checkpoints" / "params.json"), str(base.REPO / "checkpoints" / "wam_coco.pth")).to("cuda:0").eval()
    log(f"PAD16 key {''.join(map(str, key))}; loaded {n} reserve images ({time.perf_counter() - t0:.0f}s)")
    with gzip.open(OUT / "extra_E3_records.jsonl.gz", "wt", encoding="utf-8") as fh:
        per_img = run_e3(wam, imgs, books_gpu, sets, masks, seed, fh)
    s3 = {"addendum_sha256": al["addendum_sha256"], "script_sha256": v3.sha(Path(__file__)), "pad_key": key.tolist(), "images": n,
          "E3": summarize_e3(per_img)}
    (OUT / "extra_E3_summary.json").write_text(json.dumps(s3, indent=2), encoding="utf-8")
    log(f"E3 done ({time.perf_counter() - t0:.0f}s)")
    print(f"{'condition':26s} {'WAM':>6s} {'BCH16s':>6s} {'BCH16h':>6s} {'REP16s':>6s} {'PAD16s':>6s} {'PAD16h':>6s} | {'BCH16s-PAD16s':>26s} holm p | {'PAD16s-WAM':>8s}", flush=True)
    for c, r in s3["E3"]["conditions"].items():
        x = r["BCH16s_minus_PAD16s"]
        print(f"{c:26s} {r['RAW_exact']:6.3f} {r['BCH16_soft']:6.3f} {r['BCH16_hard']:6.3f} {r['REP16_soft']:6.3f} {r['PAD16_soft']:6.3f} {r['PAD16_hard']:6.3f} | "
              f"{x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] {s3['E3']['holm_BCH16s_minus_PAD16s'][c]:.4f} | {r['PAD16s_minus_RAW']['diff']:+.3f}", flush=True)
    s4 = {"addendum_sha256": al["addendum_sha256"], "script_sha256": v3.sha(Path(__file__)), "E4": run_e4(wam, imgs, books_gpu["BCH16"], key, masks)}
    (OUT / "extra_E4_summary.json").write_text(json.dumps(s4, indent=2), encoding="utf-8")
    log(f"E4 timing on {s4['E4']['gpu']} (median / p95 ms):")
    for c, st in s4["E4"]["stages_ms"].items():
        print(f"  {c}: " + "; ".join(f"{k} {v['median']:.2f}/{v['p95']:.2f}" if "median" in v else f"{k} {v['mean']:.2f}" for k, v in st.items()), flush=True)
    log(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
