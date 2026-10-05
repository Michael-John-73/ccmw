"""Reproducibility check of Algorithm 1 (codebook-constrained soft decoding) and Algorithm 2 (calibrated registry attribution).

The decoding/attribution steps are re-implemented here directly from the pseudocode in figures/algorithms.md, without the
study's decoding functions (v1b_h2h3.units_of / h2_eval, v3_cal.threshold). WAM itself (embedder, extractor) and WAM's DBSCAN
(exact GPU equivalent wamset.dbscan_gpu.dbscan_bits, verified against the upstream implementation) are fixed inputs.

Checks (no new statistic; records and reserve images only):
  A1  Algorithm 1 on the 500 reserve images, E_pub embeddings of review F16 (targets seed 20261108, transforms seed 20261110,
      none and hflip_contrast1.5): decoded codeword and score of every region vs the F16 region records.
  A2a Algorithm 2 calibration on the V3 calibration records (C_max scene null scores): tau vs v3_thresholds.json.
  A2b Algorithm 2 acceptance on the V4 test records: false attributions and correct attribution rate vs v4_summary.json.
  A2c Algorithm 2 acceptance per region on the F16 region records: correct attributions vs review_F16_summary.json.
Output: reports/analysis/algo_repro_summary.json"""
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
sys.path.insert(0, str(HERE))
import v1_h1 as v1  # noqa: E402  (WAM paths, distortions, checkerboard layout)
from v1_h1 import base  # noqa: E402
from h1_pretest import bch_codebook  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402
from notebooks.inference_utils import load_model_from_checkpoint  # noqa: E402
from watermark_anything.data.transforms import default_transform  # noqa: E402
from wamset.config import stable_seed  # noqa: E402
from wamset.dbscan_gpu import dbscan_bits  # noqa: E402

ART = Path("/workspace/wam/artifacts")
AN = STUDY / "reports" / "analysis"
NEG = float("-inf")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ------------------------------------------------------------------ Algorithm 1 (decoding part), from the pseudocode
def alg1_decode(out, C_pm):
    """out: (33, H, W) WAM extractor output; C_pm: (65536, 32) codebook in +-1 form.
    Returns labels (H, W) and a list of (label, codeword index, score)."""
    _, labels = dbscan_bits(out, 1000, 1.0)            # steps 3-6: p > 0.5 pixels, bits = logit > 0, WAM DBSCAN (eps 1, min 1000)
    lab = torch.as_tensor(labels, device=out.device)
    Lam = out[1:]
    res = []
    for r in sorted(int(v) for v in np.unique(labels) if v >= 0):
        l_r = Lam[:, lab == r].mean(1)                 # step 7: region mean bit logits
        sc = C_pm @ l_r                                # step 8: correlation with all 65,536 codewords
        c = int(torch.argmax(sc))
        res.append((r, c, float(sc[c])))
    return labels, res


# ------------------------------------------------------------------ Algorithm 2, from the pseudocode
def alg2_calibrate(z, alpha):
    """z: calibration scene null scores (-inf if none). Returns tau and m."""
    n = len(z)
    m = math.floor(alpha * (n + 1)) - 1                # null scores allowed at or above an accepted score
    fin = sorted((v for v in z if math.isfinite(v)), reverse=True)
    return (fin[m] if 0 <= m < len(fin) else NEG), m


def alg2_accept(decoded_id, score, registry, tau):
    return bool(registry[decoded_id]) and score > tau


def main():
    proto = json.loads((STUDY / "configs" / "protocol_v5.json").read_text(encoding="utf-8"))
    lock = json.loads((STUDY / "configs" / "protocol_v5_lock.json").read_text(encoding="utf-8"))
    th = json.loads((AN / "v3_thresholds.json").read_text(encoding="utf-8"))
    v4 = json.loads((AN / "v4_summary.json").read_text(encoding="utf-8"))
    f16 = json.loads((AN / "review_F16_summary.json").read_text(encoding="utf-8"))
    files = sorted(base.COCO.glob("*.jpg"))[4500:5000]
    if hashlib.sha256("".join(f"{f.name}:{sha(f)}\n" for f in files).encode()).hexdigest() != lock["image_lists_sha256"]["reserve"]:
        raise SystemExit("[REPRO] STOP: reserve image list hash mismatch")
    if sha(base.REPO / "checkpoints" / "wam_coco.pth") != lock["checkpoints_sha256"]["wam_coco"]:
        raise SystemExit("[REPRO] STOP: wam_coco hash mismatch")
    _, _, bch = bch_codebook(3)
    reg_ids = np.array(th["registry"]["BCH16_ids"])
    registry = np.zeros(1 << 16, bool)
    registry[reg_ids] = True
    out = {"script_sha256": sha(Path(__file__)), "inputs": {}, "checks": {}}

    # ---------------------------------------------------------------- A2a calibration
    cal_path = ART / "v3" / "v3_cal_records.jsonl.gz"
    cal = load(cal_path)
    z = [max(r["m"]["C_max"]["null"].values(), default=NEG) for r in cal]
    alpha = th["alpha_cal"]
    tau, m = alg2_calibrate(z, alpha)
    out["inputs"]["v3_cal_records_sha256"] = sha(cal_path)
    out["checks"]["A2a_calibration"] = {"n_cal": len(z), "alpha": alpha, "m": m, "finite_nulls": int(sum(math.isfinite(v) for v in z)),
                                        "tau_reference_impl": tau, "tau_locked_v3": th["methods"]["C"]["tau"],
                                        "pass": tau == th["methods"]["C"]["tau"]}

    # ---------------------------------------------------------------- A2b acceptance on V4 test records
    v4_path = ART / "v4" / "v4_h2_records.jsonl.gz"
    test = load(v4_path)
    zt = np.array([max(r["m"]["C_max"]["null"].values(), default=NEG) for r in test])
    succ = np.array([s for r in test for s in r["m"]["C_max"]["succ"]])
    fa, R = int((zt > tau).sum()), float(np.mean(succ > tau))
    mC = v4["H2"]["methods"]["C"]
    out["inputs"]["v4_h2_records_sha256"] = sha(v4_path)
    out["checks"]["A2b_test_acceptance"] = {"scenes": len(test), "false_attributions": fa, "false_attributions_v4": mC["false_attributions"],
                                            "R": R, "R_v4": mC["R"], "pass": fa == mC["false_attributions"] and abs(R - mC["R"]) < 1e-12}

    # ---------------------------------------------------------------- A1 + A2c on reserve images (F16 E_pub)
    f16_path = ART / "review" / "review_F16_records.jsonl.gz"
    rec = {(r["image"], r["cond"]): r for r in load(f16_path) if r["set"] == "E_pub"}
    out["inputs"]["review_F16_records_sha256"] = sha(f16_path)
    rng = np.random.default_rng(20261108)
    targets = np.stack([rng.choice(reg_ids, 5, replace=False) for _ in range(len(files))])
    resize = transforms.Resize((base.SIZE, base.SIZE))
    imgs = torch.stack([default_transform(resize(Image.open(f).convert("RGB"))) for f in files])
    masks = base.checkerboard_masks(torch.device("cuda:0"))
    wam = load_model_from_checkpoint(str(base.REPO / "checkpoints" / "params.json"), str(base.REPO / "checkpoints" / "wam_coco.pth")).to("cuda:0").eval()
    C_pm = torch.as_tensor(bch, device="cuda:0").float() * 2 - 1
    a1 = {"regions": 0, "same_codeword": 0, "max_abs_score_diff": 0.0, "region_count_mismatch_images": 0}
    a2c = {}
    for s in range(0, len(files), v1.BATCH):
        x = imgs[s:s + v1.BATCH].cuda()
        B = len(x)
        msgs = torch.as_tensor(bch[targets[s:s + B]], device="cuda:0").float()
        with torch.no_grad():
            wm = [wam.embed(x, msgs[:, j])["imgs_w"] for j in range(5)]      # step 1
            y = x.clone()
            for j in range(5):                                              # step 2
                mk = masks[j][None, None]
                y = wm[j] * mk + y * (1 - mk)
        for cond in ("none", "hflip_contrast1.5"):
            yd, gt = v1.apply(y.clone(), masks[None].expand(B, -1, -1, -1).clone(), v1.CONDITIONS[cond], stable_seed(20261110, "f16", s, cond))
            with torch.no_grad():
                pred = wam.detect(yd)["preds"].float()                       # step 3
            acc = a2c.setdefault(cond, {"slots": 0, "correct_accepted": 0})
            for b in range(B):
                i = 4500 + s + b
                labels, res = alg1_decode(pred[b], C_pm)
                g = gt[b].cpu().numpy()
                mine = []
                for r, c, sc in res:
                    ov = [int((g[j] & (labels == r)).sum()) for j in range(5)]
                    mine.append((int(np.argmax(ov)) if max(ov) > 0 else -1, c, sc))
                ref = rec[(i, cond)]["units"]
                if len(mine) != len(ref):
                    a1["region_count_mismatch_images"] += 1
                for (j1, c1, s1), (j2, c2, s2) in zip(sorted(mine), sorted((u[0], u[1], u[2]) for u in ref)):
                    a1["regions"] += 1
                    a1["same_codeword"] += int(j1 == j2 and c1 == c2)
                    a1["max_abs_score_diff"] = max(a1["max_abs_score_diff"], abs(s1 - s2))
                for j in range(5):
                    if not rec[(i, cond)]["eligible"][j]:
                        continue
                    acc["slots"] += 1
                    acc["correct_accepted"] += any(jj == j and c == int(targets[i - 4500][j]) and alg2_accept(c, sc, registry, tau) for jj, c, sc in mine)
    a1["pass"] = a1["region_count_mismatch_images"] == 0 and a1["same_codeword"] == a1["regions"] and a1["max_abs_score_diff"] < 1e-3
    out["checks"]["A1_soft_decoding_reserve"] = a1
    for cond, acc in a2c.items():
        ref = f16["conditions"][cond]
        acc.update({"correct_accepted_F16": ref["F_target_PUB"]["successes"], "slots_F16": ref["eligible_slots"],
                    "pass": acc["correct_accepted"] == ref["F_target_PUB"]["successes"] and acc["slots"] == ref["eligible_slots"]})
    out["checks"]["A2c_region_acceptance_reserve"] = a2c
    out["all_pass"] = all(v["pass"] for k, v in out["checks"].items() if "pass" in v) and all(v["pass"] for v in a2c.values())
    (AN / "algo_repro_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["checks"], indent=1))
    print("[REPRO] all_pass", out["all_pass"])


if __name__ == "__main__":
    main()
