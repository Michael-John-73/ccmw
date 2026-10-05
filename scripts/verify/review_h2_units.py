"""H2 estimands (post hoc, existing records only, no new measurement).

The locked protocol defines R as the share of registered eligible messages that are correctly attributed (pooled over messages)
and the H2 decision statistic as the mean over images of the per-image difference R_i(C) - R_i(A) (images with at least one
registered eligible message). This script reports both estimands for every method so that each reported rate, difference and
interval refers to the same quantity:
  pooled     R = sum of accepted correct messages / sum of messages; difference with an image-level (cluster) bootstrap interval
  per-image  mean of R_i over images; difference reproduces the locked statistic exactly (same resampling seed)
for the locked thresholds (V3) and for the E2 sensitivity thresholds (near-miss scores excluded from the null scores).
Input : /workspace/wam/artifacts/v4/v4_h2_records.jsonl.gz (sha256 checked against algo_repro_summary.json)
Output: reports/analysis/review_h2_units_summary.json
Usage : python scripts/verify/review_h2_units.py [records_path]
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[2]
REC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace/wam/artifacts/v4/v4_h2_records.jsonl.gz")
OUT = STUDY / "reports" / "analysis" / "review_h2_units_summary.json"
J = lambda p: json.loads((STUDY / p).read_text(encoding="utf-8"))  # noqa: E731
VARIANT = {"A": "A", "B": "B_margin", "C": "C_max"}
LEVEL, REPEATS = 0.9875, 10000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert sha(REC) == J("reports/analysis/algo_repro_summary.json")["inputs"]["v4_h2_records_sha256"], "records hash mismatch"
    proto, th, v4 = J("configs/protocol_v5.json"), J("reports/analysis/v3_thresholds.json"), J("reports/analysis/v4_summary.json")
    e2 = J("reports/analysis/extra_E2_summary.json")["E2"]
    seed = proto["decisions"]["bootstrap"]["seed"]
    with gzip.open(REC, "rt", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    settings = {"locked": {m: th["methods"][m]["tau"] for m in VARIANT},
                "without_near_miss_scores": {m: e2["without_near_miss"][m]["tau"] for m in VARIANT}}
    out = {"script_sha256": sha(__file__), "v4_h2_records_sha256": sha(REC), "level": LEVEL, "repeats": REPEATS, "seed": seed,
           "note": "post hoc: computed after the locked analyses from existing records; the locked H2 verdict is unchanged"}
    alpha = 1 - LEVEL
    for label, taus in settings.items():
        keys = [r["image"] for r in records if r["m"]["C_max"]["succ"]]
        byimg = {r["image"]: r for r in records}
        hit, cnt = {}, np.array([len(byimg[i]["m"]["C_max"]["succ"]) for i in keys], float)
        for m, v in VARIANT.items():
            assert all(len(byimg[i]["m"][v]["succ"]) == cnt[j] for j, i in enumerate(keys))
            hit[m] = np.array([np.sum(np.array(byimg[i]["m"][v]["succ"]) > taus[m]) for i in keys], float)
        n = len(keys)
        idx = np.random.default_rng(seed).integers(0, n, (REPEATS, n))
        res = {"images": n, "messages": int(cnt.sum()), "methods": {}}
        for m in VARIANT:
            res["methods"][m] = {"tau": taus[m], "R_pooled": float(hit[m].sum() / cnt.sum()), "R_per_image_mean": float(np.mean(hit[m] / cnt))}
        for b in ("A", "B"):
            d_img = hit["C"] / cnt - hit[b] / cnt
            bt_img = d_img[idx].mean(1)
            pooled_bt = hit["C"][idx].sum(1) / cnt[idx].sum(1) - hit[b][idx].sum(1) / cnt[idx].sum(1)
            res[f"C_minus_{b}"] = {
                "per_image_mean": {"diff": float(d_img.mean()), "ci": [float(x) for x in np.quantile(bt_img, [alpha / 2, 1 - alpha / 2])]},
                "pooled": {"diff": res["methods"]["C"]["R_pooled"] - res["methods"][b]["R_pooled"],
                           "ci": [float(x) for x in np.quantile(pooled_bt, [alpha / 2, 1 - alpha / 2])]}}
        out[label] = res
    lk, nn = out["locked"], out["without_near_miss_scores"]
    for m in VARIANT:
        assert abs(lk["methods"][m]["R_pooled"] - v4["H2"]["methods"][m]["R"]) < 1e-12
        assert abs(nn["methods"][m]["R_pooled"] - e2["without_near_miss"][m]["R"]) < 1e-12
    for b in ("A", "B"):
        locked = v4["H2"][f"R_C_minus_{b}"]
        got = lk[f"C_minus_{b}"]["per_image_mean"]
        assert abs(got["diff"] - locked["diff"]) < 1e-12 and np.allclose(got["ci"], locked["ci"], atol=1e-12), (b, got, locked)
        assert abs(nn[f"C_minus_{b}"]["per_image_mean"]["diff"] - e2["without_near_miss"][f"R_C_minus_{b}"]["diff"]) < 1e-12
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for label in ("locked", "without_near_miss_scores"):
        r = out[label]
        print(label, {m: (round(v["R_pooled"], 4), round(v["R_per_image_mean"], 4)) for m, v in r["methods"].items()})
        for b in ("A", "B"):
            print(f"  C-{b}", {k: (round(v["diff"], 4), [round(x, 4) for x in v["ci"]]) for k, v in r[f"C_minus_{b}"].items()})


if __name__ == "__main__":
    main()
