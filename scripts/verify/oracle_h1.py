"""H1 feasibility (oracle) under the WAM paper's multi-watermark conditions (Sec. 5.5, App. D.6, E.2).

Conditions taken from the WAM paper / source unchanged: COCO validation images resized to 256x256; k = 1..5 random
32-bit messages, each embedded by feeding the SAME image to WAM's embedder and pasting the watermarked area onto the
original; areas are equal 10% squares in a checkerboard (non-overlapping); detection tau = 0.5; DBSCAN eps = 1,
min_samples = 1000; augmentations from WAM's own modules: none / hflip + contrast 1.5 / hflip + contrast 1.5 + JPEG 80
(Sec. 5.5, App. E.2), plus JPEG 80 and JPEG 50 alone (App. D.4 evaluation values). Model: wam_mit.pth via WAM's
load_model_from_checkpoint (params.json: scaling_w 2.0, scaling_i 1.0, JND).

Decoders compared on the same WAM outputs:
  (a) WAM      : DBSCAN on per-pixel hard bits (exactly equivalent GPU implementation, verified against
                 WAM's sklearn routine), centroid = decoded message (WAM Sec. 3).
  (b) ORACLE   : WAM's msg_predict_inference (Eq. 2, README default) restricted to each ground-truth area.
                 Upper bound of what the extractor signal allows when the areas are known.
  (c) M1       : connected components (4-connectivity) of WAM's detected mask (tau 0.5), components >= 1000 px,
                 each decoded with the same msg_predict_inference.
Each decoded unit is assigned to the ground-truth message with the largest pixel overlap (WAM Sec. 5.5 rule).
A ground-truth message is recovered when an assigned unit decodes it with 0 bit errors.

Decision rule (fixed before running): H1 is FEASIBLE if, under hflip + contrast 1.5 + JPEG 80 pooled over k = 1..5,
exact recovery(ORACLE) - exact recovery(WAM) >= 0.20 and the image-cluster bootstrap 95% CI lower bound of that
difference is > 0. Otherwise H1 is NOT FEASIBLE (the extractor signal itself is lost; post-processing cannot fix it).
Images: the first 200 files of COCO val2017 in name order; they are reserved for development and excluded from any
later test set."""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

STUDY = Path(__file__).resolve().parents[2]
REPO = Path("/workspace/wam/JISA_selected_sources/watermark-anything")
COCO = Path("/workspace/wam/datasets/coco2017/val2017")
OUT = Path("/workspace/wam/artifacts/oracle_h1")
CKPT = REPO / "checkpoints" / "wam_mit.pth"
N_IMAGES, SIZE, SIDE, BATCH, SEED = 200, 256, 81, 25, 20261004
CELLS = [(0, 0), (2, 2), (0, 2), (2, 0), (1, 1)]  # checkerboard cells of a 3x3 grid, first k used
RULE = {"condition": "hflip_contrast1.5_jpeg80", "min_gap": 0.20, "ci_lower_gt": 0.0, "bootstrap": 2000}

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(STUDY / "src"))
os.chdir(REPO)  # params.json refers to configs/*.yaml relative to the repository

from PIL import Image  # noqa: E402
from scipy.ndimage import label as cc_label  # noqa: E402
from torchvision import transforms  # noqa: E402
from notebooks.inference_utils import load_model_from_checkpoint  # noqa: E402
from watermark_anything.augmentation.geometric import HorizontalFlip  # noqa: E402
from watermark_anything.augmentation.valuemetric import JPEG, Contrast  # noqa: E402
from watermark_anything.data.metrics import msg_predict_inference  # noqa: E402
from watermark_anything.data.transforms import default_transform  # noqa: E402
from wamset.config import progress  # noqa: E402
from wamset.dbscan_gpu import dbscan_bits  # noqa: E402

HFLIP, CONTRAST, JPEG_OP = HorizontalFlip(), Contrast(), JPEG()
CONDITIONS = {
    "none": [],
    "hflip_contrast1.5": [(HFLIP, None), (CONTRAST, 1.5)],
    "hflip_contrast1.5_jpeg80": [(HFLIP, None), (CONTRAST, 1.5), (JPEG_OP, 80)],
    "jpeg80": [(JPEG_OP, 80)],
    "jpeg50": [(JPEG_OP, 50)],
}


def checkerboard_masks(device):
    cell = SIZE / 3
    masks = torch.zeros((len(CELLS), SIZE, SIZE), device=device)
    for i, (r, c) in enumerate(CELLS):
        y, x = round(r * cell + (cell - SIDE) / 2), round(c * cell + (cell - SIDE) / 2)
        masks[i, y:y + SIDE, x:x + SIDE] = 1
    assert masks.sum(0).max() == 1, "areas overlap"
    return masks


def augment(image, masks, ops):
    for op, value in ops:
        image, masks = op(image, masks, value) if value is not None else op(image, masks)
    return image, masks


def assign(unit, gts):
    overlap = (gts & unit[None]).flatten(1).sum(1)
    j = int(overlap.argmax())
    return (j if int(overlap[j]) > 0 else -1), overlap


def errors(bits, msg):
    return int((bits.to(torch.bool).cpu() != msg.to(torch.bool).cpu()).sum())


def evaluate_image(pred, gts, msgs):
    """pred (33, H, W) logits; gts (k, H, W) bool; msgs (k, 32). Returns per-method per-message best errors etc."""
    k = gts.shape[0]
    out = {}
    bit_logits = pred[1:][None]
    # (a) WAM DBSCAN
    centers, labels = dbscan_bits(pred, 1000, 1.0)
    lab = torch.as_tensor(labels, device=pred.device)
    units = []
    for cid, centroid in centers.items():
        j, ov = assign(lab == cid, gts)
        units.append((j, None if j < 0 else errors(torch.as_tensor(centroid), msgs[j]), ov))
    out["WAM"] = units
    # (b) ORACLE
    units = []
    for j in range(k):
        bits = msg_predict_inference(bit_logits, gts[j][None, None].float())[0]
        units.append((j, errors(bits, msgs[j]), None))
    out["ORACLE"] = units
    # (c) M1: connected components of the detected mask, decoded with WAM's Eq. 2 routine
    detected = (torch.sigmoid(pred[0].detach().float().cpu()) > .5).numpy()
    comp, n = cc_label(detected)
    comp = torch.as_tensor(comp, device=pred.device)
    units = []
    for c in range(1, n + 1):
        region = comp == c
        if int(region.sum()) < 1000:
            continue
        bits = msg_predict_inference(bit_logits, region[None, None].float())[0]
        j, ov = assign(region, gts)
        units.append((j, None if j < 0 else errors(bits, msgs[j]), ov))
    out["M1"] = units
    union = gts.any(0)
    inter, uni = int((torch.as_tensor(detected, device=pred.device) & union).sum()), int((torch.as_tensor(detected, device=pred.device) | union).sum())
    return out, inter / uni if uni else 0.0


def summarize(rows):
    """rows: list of dicts (image, k, method -> units). Message-level and WAM-style metrics."""
    res = {}
    for method in ("WAM", "ORACLE", "M1"):
        exact, best_err, found_acc, n_units, spurious, missed = [], [], [], [], 0, 0
        for r in rows:
            units = r[method]
            n_units.append(len(units))
            for j, e, _ in units:
                if j < 0:
                    spurious += 1
                else:
                    found_acc.append(1 - e / 32)
            for j in range(r["k"]):
                errs = [e for jj, e, _ in units if jj == j]
                b = min(errs) if errs else None
                missed += b is None
                exact.append(int(b == 0))
                best_err.append(32 if b is None else b)
        res[method] = {"exact_recovery": float(np.mean(exact)), "mean_best_bit_errors": float(np.mean(best_err)),
                       "bit_acc_found_units(WAM metric)": float(np.mean(found_acc)) if found_acc else None,
                       "units_per_image": float(np.mean(n_units)), "missed_messages": missed, "messages": len(exact),
                       "spurious_units": spurious}
    return res


def per_image_exact(rows, method):
    acc = defaultdict(list)
    for r in rows:
        for j in range(r["k"]):
            errs = [e for jj, e, _ in r[method] if jj == j]
            acc[r["image"]].append(int(bool(errs) and min(errs) == 0))
    return acc


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    wam = load_model_from_checkpoint(str(REPO / "checkpoints" / "params.json"), str(CKPT)).to(device).eval()
    files = sorted(COCO.glob("*.jpg"))[:N_IMAGES]
    resize = transforms.Resize((SIZE, SIZE))
    imgs = torch.stack([default_transform(resize(Image.open(f).convert("RGB"))) for f in files])
    masks = checkerboard_masks(device)
    gen = torch.Generator().manual_seed(SEED)
    msgs_all = torch.randint(0, 2, (N_IMAGES, len(CELLS), 32), generator=gen).float()
    rows = defaultdict(list)
    miou = defaultdict(list)
    jobs = [(k, s) for k in range(1, len(CELLS) + 1) for s in range(0, N_IMAGES, BATCH)]
    for k, s in progress(jobs, desc="oracle H1 (k x batch)", unit="batch"):
        x = imgs[s:s + BATCH].to(device)
        msgs = msgs_all[s:s + BATCH, :k].to(device)
        with torch.no_grad():
            multi = x.clone()
            for i in range(k):
                w = wam.embed(x, msgs[:, i])["imgs_w"]
                m = masks[i][None, None]
                multi = w * m + multi * (1 - m)
            for cname, ops in CONDITIONS.items():
                gt = masks[:k][None].expand(len(x), -1, -1, -1).clone()
                img, gt = augment(multi.clone(), gt, ops)
                preds = wam.detect(img)["preds"].float()
                for b in range(len(x)):
                    units, iou = evaluate_image(preds[b], gt[b] > .5, msgs[b])
                    rows[cname].append({"image": s + b, "k": k, **units})
                    miou[cname].append(iou)
    summary = {"conditions": {}, "rule": RULE, "images": [f.name for f in files], "checkpoint": CKPT.name,
               "seconds": None}
    for cname, rr in rows.items():
        summary["conditions"][cname] = {"pooled": summarize(rr), "mean_IoU_detected_vs_union": float(np.mean(miou[cname])),
                                        "by_k": {k: summarize([r for r in rr if r["k"] == k]) for k in range(1, len(CELLS) + 1)}}
    rr = rows[RULE["condition"]]
    a, b = per_image_exact(rr, "WAM"), per_image_exact(rr, "ORACLE")
    keys = sorted(a)
    diff = np.array([np.mean(b[i]) - np.mean(a[i]) for i in keys])
    rng = np.random.default_rng(SEED)
    boot = diff[rng.integers(0, len(diff), (RULE["bootstrap"], len(diff)))].mean(1)
    lo, hi = np.quantile(boot, [.025, .975])
    gap = summary["conditions"][RULE["condition"]]["pooled"]["ORACLE"]["exact_recovery"] - summary["conditions"][RULE["condition"]]["pooled"]["WAM"]["exact_recovery"]
    verdict = {"gap_oracle_minus_wam": gap, "ci95": [float(lo), float(hi)],
               "FEASIBLE": bool(gap >= RULE["min_gap"] and lo > RULE["ci_lower_gt"])}
    summary["verdict"] = verdict
    summary["seconds"] = time.perf_counter() - t0
    (OUT / "oracle_h1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n[oracle H1] exact message recovery (pooled k=1..5) | mean best bit errors | WAM-style bit acc on found units | units/img | missed | spurious | IoU", flush=True)
    for cname, v in summary["conditions"].items():
        p = v["pooled"]
        print(f"  {cname:26s} " + "  ".join(
            f"{m}: {p[m]['exact_recovery']:.3f} / {p[m]['mean_best_bit_errors']:5.2f} / "
            f"{(p[m]['bit_acc_found_units(WAM metric)'] or 0):.3f} / {p[m]['units_per_image']:.2f} / {p[m]['missed_messages']} / {p[m]['spurious_units']}"
            for m in ("WAM", "ORACLE", "M1")) + f"  IoU {v['mean_IoU_detected_vs_union']:.3f}", flush=True)
    print("\n[oracle H1] exact recovery by k (WAM / ORACLE / M1):", flush=True)
    for cname, v in summary["conditions"].items():
        print(f"  {cname:26s} " + "  ".join(
            f"k{k}: {v['by_k'][k]['WAM']['exact_recovery']:.2f}/{v['by_k'][k]['ORACLE']['exact_recovery']:.2f}/{v['by_k'][k]['M1']['exact_recovery']:.2f}"
            for k in range(1, len(CELLS) + 1)), flush=True)
    print(f"\n[oracle H1] VERDICT ({RULE['condition']}): {json.dumps(verdict)}  ({summary['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
