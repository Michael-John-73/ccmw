"""Recompute the development (V1b) H3 distance-wise merge differences from the V1b H3 records, so that the development
values quoted in WAM.md section 4.4 (RAW d<=4 - BCH16 nearest, and d=4 - BCH16 nearest) have a result file.
Records only (no new measurement): /workspace/wam/artifacts/v1b/v1b_h3_records.jsonl.gz (development images 0-499, wam_coco).
Bootstrap as in V1b: per-image paired differences, 10,000 repeats, seed 20261009, 98.75%."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

REC = Path("/workspace/wam/artifacts/v1b/v1b_h3_records.jsonl.gz")
OUT = Path(__file__).resolve().parents[2] / "reports" / "analysis" / "review_dev_h3_summary.json"
BOOT = {"repeats": 10000, "seed": 20261009, "level": 0.9875}  # v1b_h2h3.BOOT


def boot(d):
    d = np.asarray(d, float)
    bt = d[np.random.default_rng(BOOT["seed"]).integers(0, len(d), (BOOT["repeats"], len(d)))].mean(1)
    a = (1 - BOOT["level"]) / 2
    return {"diff": float(d.mean()), "ci": [float(np.quantile(bt, a)), float(np.quantile(bt, 1 - a))], "n": int(len(d))}


def main():
    with gzip.open(REC, "rt", encoding="utf-8") as f:
        rec = [json.loads(line) for line in f]
    m = {(r["image"], r["type"], r["cond"]): float(r["merged"]) for r in rec}
    imgs = sorted({r["image"] for r in rec})
    conds = sorted({r["cond"] for r in rec})
    out = {"source_records": str(REC), "source_sha256": hashlib.sha256(REC.read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "bootstrap": BOOT,
           "images": len(imgs), "conditions": conds, "rates": {}, "differences": {}}
    for c in conds:
        out["rates"][c] = {t: float(np.mean([m[(i, t, c)] for i in imgs])) for t in
                           ["RAW_d1", "RAW_d2", "RAW_d3", "RAW_d4", "RAW_d6", "RAW_d8", "RAW_rand", "BCH_near", "BCH_rand"]}
        for d in (1, 2, 3, 4, 6, 8):
            out["differences"][f"RAW_d{d}-BCH_near:{c}"] = boot([m[(i, f"RAW_d{d}", c)] - m[(i, "BCH_near", c)] for i in imgs])
    out["RAW_dle4_mean"] = float(np.mean([m[(i, f"RAW_d{d}", c)] for i in imgs for d in (1, 2, 3, 4) for c in conds]))
    out["BCH_near_mean"] = float(np.mean([m[(i, "BCH_near", c)] for i in imgs for c in conds]))
    out["differences"]["RAW_dle4_mean-BCH_near (both transforms)"] = boot(
        [np.mean([m[(i, f"RAW_d{d}", c)] for d in (1, 2, 3, 4) for c in conds]) - np.mean([m[(i, "BCH_near", c)] for c in conds]) for i in imgs])
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in out["differences"].items():
        print(f"{k:45s} {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
    print(f"RAW d<=4 mean {out['RAW_dle4_mean']:.3f}, BCH near mean {out['BCH_near_mean']:.3f}")


if __name__ == "__main__":
    main()
