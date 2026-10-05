"""Descriptive curve data for Figure 3 (H2): scene-level false attribution rate vs correct attribution rate R as the
acceptance threshold varies, for methods A (WAM-t), B (registry-only soft, margin) and C (full-codebook soft + registry),
from the V4 test records (3,000 mixed scenes). The locked thresholds (V3) are marked separately; the curves do not change
any decision. A second curve set removes the 'unregistered near-miss' null source (threat model without near-miss, as E2).
Output: reports/analysis/fig_h2_curves.json"""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[2]
REC = Path("/workspace/wam/artifacts/v4/v4_h2_records.jsonl.gz")
M = {"A": "A", "B": "B_margin", "C": "C_max"}
NEG = float("-inf")


def curve(nulls, succ):
    cand = np.unique(np.concatenate([nulls[np.isfinite(nulls)], succ[np.isfinite(succ)]]))
    taus = np.concatenate([[NEG], cand])  # accept score > tau
    fa = np.array([np.mean(nulls > t) for t in taus])
    r = np.array([np.mean(succ > t) for t in taus])
    keep = np.r_[True, (np.diff(fa) != 0) | (np.diff(r) != 0)]
    return taus[keep], fa[keep], r[keep]


def main():
    with gzip.open(REC, "rt", encoding="utf-8") as f:
        rec = [json.loads(line) for line in f]
    th = json.loads((STUDY / "reports" / "analysis" / "v3_thresholds.json").read_text(encoding="utf-8"))
    out = {"source_records": str(REC), "source_sha256": hashlib.sha256(REC.read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "scenes": len(rec), "curves": {}, "locked": {}}
    for name, v in M.items():
        succ = np.array([s for r in rec for s in r["m"][v]["succ"]], float)
        for tm, drop in (("with_near_miss", ()), ("without_near_miss", ("unreg_near",))):
            nulls = np.array([max([s for src, s in r["m"][v]["null"].items() if src not in drop], default=NEG) for r in rec], float)
            t, fa, rr = curve(nulls, succ)
            out["curves"][f"{name}:{tm}"] = {"tau": [float(x) for x in t], "FA": fa.tolist(), "R": rr.tolist()}
        tau = th["methods"][name]["tau"]
        nulls = np.array([max(r["m"][v]["null"].values(), default=NEG) for r in rec], float)
        out["locked"][name] = {"tau": tau, "FA": float(np.mean(nulls > tau)), "R": float(np.mean(succ > tau))}
    (STUDY / "reports" / "analysis" / "fig_h2_curves.json").write_text(json.dumps(out), encoding="utf-8")
    print({k: v for k, v in out["locked"].items()})


if __name__ == "__main__":
    main()
