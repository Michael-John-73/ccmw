"""Review stage R4 (configs/protocol_v5_addendum_R4.json, locked): embedding quality (PSNR, SSIM) of RAW, BCH16 and
PAD16 messages with WAM (wam_coco) on the 500 reserve images."""
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extra_e3e4 as e34  # noqa: E402
from extra_e3e4 import v1, v3, base  # noqa: E402
from h1_pretest import bch_codebook  # noqa: E402
from PIL import Image  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402
from torchvision import transforms  # noqa: E402
from notebooks.inference_utils import load_model_from_checkpoint  # noqa: E402
from watermark_anything.data.transforms import default_transform, unnormalize_img  # noqa: E402
from wamset.config import progress  # noqa: E402

STUDY = HERE.parents[1]
PLAN = STUDY / "configs" / "protocol_v5_addendum_R4.json"
PLAN_LOCK = STUDY / "configs" / "protocol_v5_addendum_R4_lock.json"
BASE_LOCK = STUDY / "configs" / "protocol_v5_lock.json"
OUT = Path("/workspace/wam/artifacts/review")
SETS = ["RAW", "BCH16", "PAD16"]
LEVEL, SEED = 0.9875, 20261006


def log(msg):
    print(f"[R4] {msg}", flush=True)


def verify():
    pl, bl = json.loads(PLAN_LOCK.read_text(encoding="utf-8")), json.loads(BASE_LOCK.read_text(encoding="utf-8"))
    bad = []
    if v3.sha(PLAN) != pl["addendum_sha256"]:
        bad.append("R4 plan")
    if v3.sha(STUDY / "configs" / "protocol_v5.json") != bl["protocol_sha256"]:
        bad.append("base protocol")
    if v3.sha(base.REPO / "checkpoints" / "wam_coco.pth") != bl["checkpoints_sha256"]["wam_coco"]:
        bad.append("wam_coco")
    files = sorted(base.COCO.glob("*.jpg"))[4500:5000]
    if hashlib.sha256("".join(f"{f.name}:{v3.sha(f)}\n" for f in files).encode()).hexdigest() != bl["image_lists_sha256"]["reserve"]:
        bad.append("reserve image list")
    if bad:
        raise SystemExit(f"[R4] STOP: hash mismatch {bad}")
    log(f"verified R4 plan {pl['addendum_sha256'][:16]}..., base protocol, wam_coco, reserve images {len(files)}")
    return pl, json.loads(PLAN.read_text(encoding="utf-8")), files


def boot(d):
    d = np.asarray(d, float)
    bt = d[np.random.default_rng(SEED).integers(0, len(d), (10000, len(d)))].mean(1)
    a = (1 - LEVEL) / 2
    return {"diff": float(d.mean()), "ci": [float(np.quantile(bt, a)), float(np.quantile(bt, 1 - a))], "n": int(len(d))}


def quality(orig, wm):
    """orig, wm: (3, H, W) normalised tensors -> (psnr, ssim) on [0, 1] float images."""
    a = unnormalize_img(orig[None]).clamp(0, 1)[0]
    b = unnormalize_img(wm[None]).clamp(0, 1)[0]
    mse = float(((a - b) ** 2).mean())
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    ssim = structural_similarity(a.permute(1, 2, 0).cpu().numpy(), b.permute(1, 2, 0).cpu().numpy(), channel_axis=2, data_range=1.0)
    return psnr, float(ssim)


def main():
    t0 = time.perf_counter()
    pl, plan, files = verify()
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, bch = bch_codebook(3)
    key = np.random.default_rng(20261104).integers(0, 2, 16).astype(np.uint8)
    pad = e34.pad_book(key)
    rng = np.random.default_rng(plan["messages"]["seed"])
    n = len(files)
    msgs = {"RAW": rng.integers(0, 2, (n, 5, 32)).astype(np.float32),
            "BCH16": bch[rng.integers(0, 1 << 16, (n, 5))].astype(np.float32),
            "PAD16": pad[rng.integers(0, 1 << 16, (n, 5))].astype(np.float32)}
    resize = transforms.Resize((base.SIZE, base.SIZE))
    imgs = torch.stack([default_transform(resize(Image.open(f).convert("RGB"))) for f in files])
    masks = base.checkerboard_masks(torch.device("cuda:0"))
    wam = load_model_from_checkpoint(str(base.REPO / "checkpoints" / "params.json"), str(base.REPO / "checkpoints" / "wam_coco.pth")).to("cuda:0").eval()
    rec = []
    with gzip.open(OUT / "review_R4_records.jsonl.gz", "wt", encoding="utf-8") as fh:
        for s in progress(range(0, n, v1.BATCH), desc="R4 quality (batch of 25 images)", unit="batch"):
            x = imgs[s:s + v1.BATCH].cuda()
            for kind in SETS:
                m = torch.as_tensor(msgs[kind][s:s + v1.BATCH], device="cuda:0")
                with torch.no_grad():
                    wm = [wam.embed(x, m[:, i])["imgs_w"] for i in range(5)]
                    comp = x.clone()
                    for i in range(5):
                        mk = masks[i][None, None]
                        comp = wm[i] * mk + comp * (1 - mk)
                for b in range(len(x)):
                    full = [quality(x[b], wm[i][b]) for i in range(5)]
                    cp = quality(x[b], comp[b])
                    r = {"image": 4500 + s + b, "set": kind, "full_psnr": float(np.mean([q[0] for q in full])), "full_ssim": float(np.mean([q[1] for q in full])),
                         "comp_psnr": cp[0], "comp_ssim": cp[1]}
                    rec.append(r)
                    fh.write(json.dumps(r) + "\n")
    by = {k: {r["image"]: r for r in rec if r["set"] == k} for k in SETS}
    imgs_ix = sorted(by["RAW"])
    out = {"addendum_R4_sha256": pl["addendum_sha256"], "script_sha256": v3.sha(Path(__file__)), "images": n, "means": {}, "differences": {}}
    for k in SETS:
        out["means"][k] = {f: float(np.mean([by[k][i][f] for i in imgs_ix])) for f in ("full_psnr", "full_ssim", "comp_psnr", "comp_ssim")}
    for k in ("BCH16", "PAD16"):
        for f in ("full_psnr", "comp_psnr", "full_ssim", "comp_ssim"):
            out["differences"][f"{k}-RAW:{f}"] = boot([by[k][i][f] - by["RAW"][i][f] for i in imgs_ix])
    p = out["differences"]["BCH16-RAW:full_psnr"]
    out["decision_supplementary"] = {"BCH16-RAW full PSNR": p, "within_0.1dB": bool(-0.1 < p["ci"][0] and p["ci"][1] < 0.1)}
    out["seconds"] = time.perf_counter() - t0
    (OUT / "review_R4_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in out["means"].items():
        log(f"{k:6s} full PSNR {v['full_psnr']:.2f} dB, SSIM {v['full_ssim']:.4f} | composite (k=5) PSNR {v['comp_psnr']:.2f} dB, SSIM {v['comp_ssim']:.4f}")
    for k, v in out["differences"].items():
        log(f"  {k:24s} {v['diff']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
    log(f"supplementary decision (BCH16 - RAW full PSNR within +-0.1 dB): {out['decision_supplementary']['within_0.1dB']}; done in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
