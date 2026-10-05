"""Review stage R3 (AI_리뷰대응_프롬프트.md): re-analysis of existing records only, no new measurement.

(1) H4 decomposition: test V4 RAW records (wam_coco): failures of eligible messages split into
    (a) no DBSCAN unit assigned, (b) best assigned unit with 1-3 bit errors, (c) 4 or more bit errors.
(2) H3 non-trivial range: test V4 H3 records, per-image paired merge differences RAW d - BCH16 nearest for
    d = 1, 2, 3, 4, 6, 8 and both transforms (98.75% bootstrap).
(3) H2 threshold stability: resample the 1,000 cal scenes with replacement 2,000 times, recompute the threshold with
    the locked rule (accept s > second-largest finite cal null), apply to the test records.
(4) Registry size N (approximation, not a measurement): per-scene probability p_N that a wrong decoding hits a
    registered ID, p_N = 1 - (1 - p_1000)^(N/1000) (uniform wrong codewords); simulated cal of 1,000 scenes with
    null scores drawn from the development (V1b) wrong-decoding scores; threshold by the locked rule; R(N) on the
    test success scores; expected false attribution stays <= alpha by construction of the rule."""
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ART = Path("/workspace/wam/artifacts")
OUT = Path("/workspace/wam/artifacts/review")
STUDY = Path(__file__).resolve().parents[2]
SEED, REPS, LEVEL = 20261006, 10000, 0.9875
NEG = float("-inf")
M = {"A": "A", "B": "B_margin", "C": "C_max"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def boot(d):
    d = np.asarray(d, float)
    bt = d[np.random.default_rng(SEED).integers(0, len(d), (REPS, len(d)))].mean(1)
    a = (1 - LEVEL) / 2
    return {"diff": float(d.mean()), "ci": [float(np.quantile(bt, a)), float(np.quantile(bt, 1 - a))], "n": int(len(d))}


def load(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def h4_decomposition():
    acc = defaultdict(lambda: np.zeros(6))  # eligible, exact, missed, err1_3, err4p, found
    imgs = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
    with gzip.open(ART / "v4" / "v4_h1_records_wam_coco.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            if '"set": "RAW"' not in line:
                continue
            r = json.loads(line)
            for m in r["messages"]:
                if not m["eligible"]:
                    continue
                a = acc[r["condition"]]
                a[0] += 1
                if not m["found"]:
                    a[2] += 1
                    cat = 1
                else:
                    a[5] += 1
                    e = round((1 - max(m["bit_acc"])) * 32)
                    if e == 0:
                        a[1] += 1
                        cat = 0
                    elif e <= 3:
                        a[3] += 1
                        cat = 2
                    else:
                        a[4] += 1
                        cat = 3
                imgs[r["condition"]][r["image"]][cat] += 1
    out = {}
    for c, a in acc.items():
        n = a[0]
        out[c] = {"eligible": int(n), "exact": a[1] / n, "missed_no_unit": a[2] / n, "found_1_3_bit_errors": a[3] / n,
                  "found_4plus_bit_errors": a[4] / n,
                  "share_of_failures": {"missed": a[2] / max(n - a[1], 1), "1_3_bits": a[3] / max(n - a[1], 1), "4plus_bits": a[4] / max(n - a[1], 1)}}
    return out


def h3_nontrivial():
    rec = load(ART / "v4" / "v4_h3_records.jsonl.gz")
    m = {(r["image"], r["type"], r["cond"]): float(r["merged"]) for r in rec}
    imgs = sorted({r["image"] for r in rec})
    conds = sorted({r["cond"] for r in rec})
    out = {}
    for c in conds:
        for d in (1, 2, 3, 4, 6, 8):
            out[f"RAW_d{d}-BCH_near:{c}"] = boot([m[(i, f"RAW_d{d}", c)] - m[(i, "BCH_near", c)] for i in imgs])
    out["RAW_d3_d4_mean-BCH_near (both transforms)"] = boot(
        [np.mean([m[(i, f"RAW_d{d}", c)] for d in (3, 4) for c in conds]) - np.mean([m[(i, "BCH_near", c)] for c in conds]) for i in imgs])
    return out


def tau_rule(nulls, n_cal=1000, alpha=0.002):
    allowed = math.floor(alpha * (n_cal + 1) - 1 + 1e-9)
    fin = np.sort(np.asarray(nulls)[np.isfinite(nulls)])[::-1]
    return fin[allowed] if allowed < len(fin) else NEG


def h2_stability(cal, test):
    rng = np.random.default_rng(SEED)
    out = {}
    for name, v in M.items():
        cn = np.array([max(r["m"][v]["null"].values(), default=NEG) for r in cal])
        tn = np.sort(np.array([max(r["m"][v]["null"].values(), default=NEG) for r in test]))
        ts = np.sort(np.array([x for r in test for x in r["m"][v]["succ"]]))
        taus, fa, rr = [], [], []
        for _ in range(2000):
            tau = tau_rule(cn[rng.integers(0, len(cn), len(cn))])
            taus.append(tau)
            fa.append(1 - np.searchsorted(tn, tau, side="right") / len(tn))
            rr.append(1 - np.searchsorted(ts, tau, side="right") / len(ts))
        q = lambda a: {"median": float(np.median(a)), "p05": float(np.quantile(a, .05)), "p95": float(np.quantile(a, .95))}  # noqa: E731
        ft = [t for t in taus if math.isfinite(t)]
        out[name] = {"locked_tau": float(tau_rule(cn)), "finite_cal_nulls": int(np.isfinite(cn).sum()),
                     "tau": q(ft) if ft else None, "share_tau_minus_inf": 1 - len(ft) / len(taus),
                     "test_FA_rate": q(fa), "test_R": q(rr), "share_test_FA_above_1pct": float(np.mean(np.array(fa) > 0.01))}
    return out


def registry_size(cal, test, dev):
    v = "C_max"
    fin_ct = sum(math.isfinite(max(r["m"][v]["null"].values(), default=NEG)) for r in cal + test)
    p1000 = fin_ct / (len(cal) + len(test))
    pool = np.array([s for r in dev for s in r["m"].get(v, {}).get("null", {}).values() if math.isfinite(s)])
    ts = np.sort(np.array([x for r in test for x in r["m"][v]["succ"]]))
    rng = np.random.default_rng(SEED)
    out = {"p_1000_observed (cal+test scenes with a registered wrong decoding)": p1000, "dev_wrong_decoding_scores": int(len(pool)),
           "assumptions": "wrong decodings are uniform over the 65,536 codewords and independent of N; their scores follow the development distribution; "
                          "unregistered messages stay outside the registry (for N = 65,536 there are no unregistered IDs); the locked conformal rule is re-applied on a cal set of 1,000 scenes",
           "N": {}}
    for N in (1000, 4000, 10000, 32768, 65536):
        pN = 1 - (1 - p1000) ** (N / 1000)
        rs, taus = [], []
        for _ in range(2000):
            k = rng.binomial(1000, pN)
            nulls = np.concatenate([rng.choice(pool, k), np.full(1000 - k, NEG)])
            tau = tau_rule(nulls)
            taus.append(tau)
            rs.append(1 - np.searchsorted(ts, tau, side="right") / len(ts))
        out["N"][N] = {"p_N": pN, "expected_cal_wrong_decodings": 1000 * pN,
                       "tau_median": float(np.median([t for t in taus if math.isfinite(t)])) if any(math.isfinite(t) for t in taus) else None,
                       "R_median": float(np.median(rs)), "R_p05": float(np.quantile(rs, .05)), "R_p95": float(np.quantile(rs, .95))}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {k: ART / p for k, p in {"v4_h1": "v4/v4_h1_records_wam_coco.jsonl.gz", "v4_h2": "v4/v4_h2_records.jsonl.gz", "v4_h3": "v4/v4_h3_records.jsonl.gz",
                                     "v3_cal": "v3/v3_cal_records.jsonl.gz", "v1b_h2": "v1b/v1b_h2_records.jsonl.gz"}.items()}
    res = {"script_sha256": sha(__file__), "input_sha256": {k: sha(p) for k, p in files.items()}}
    res["H4_decomposition"] = h4_decomposition()
    print("[R3] H4 decomposition done", flush=True)
    res["H3_nontrivial"] = h3_nontrivial()
    print("[R3] H3 done", flush=True)
    cal, test = load(files["v3_cal"]), load(files["v4_h2"])
    res["H2_threshold_stability"] = h2_stability(cal, test)
    print("[R3] H2 stability done", flush=True)
    dev = load(files["v1b_h2"])
    res["H2_registry_size"] = registry_size(cal, test, dev)
    (OUT / "review_R3_summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    d = res["H4_decomposition"]
    for c in ("none", "hflip_contrast1.5", "jpeg80", "rotate10"):
        x = d[c]
        print(f"[R3] H4 {c:18s} exact {x['exact']:.3f} | missed {x['missed_no_unit']:.3f} | 1-3 bits {x['found_1_3_bit_errors']:.3f} | 4+ bits {x['found_4plus_bit_errors']:.3f}", flush=True)
    for k, v in res["H3_nontrivial"].items():
        print(f"[R3] H3 {k:42s} {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]", flush=True)
    for k, v in res["H2_threshold_stability"].items():
        t = v["tau"]
        print(f"[R3] H2 {k}: locked tau {v['locked_tau']:.2f}, finite cal nulls {v['finite_cal_nulls']}, tau median {t['median'] if t else None} "
              f"[{t['p05'] if t else None}, {t['p95'] if t else None}], -inf share {v['share_tau_minus_inf']:.3f}; test FA median {v['test_FA_rate']['median']:.4f} "
              f"[{v['test_FA_rate']['p05']:.4f}, {v['test_FA_rate']['p95']:.4f}], share > 1% {v['share_test_FA_above_1pct']:.3f}; R median {v['test_R']['median']:.3f} "
              f"[{v['test_R']['p05']:.3f}, {v['test_R']['p95']:.3f}]", flush=True)
    rs = res["H2_registry_size"]
    print(f"[R3] registry: p_1000 {rs['p_1000_observed (cal+test scenes with a registered wrong decoding)']:.4f}, dev wrong-decoding scores {rs['dev_wrong_decoding_scores']}", flush=True)
    for N, v in rs["N"].items():
        print(f"[R3]   N {N:6d}: p_N {v['p_N']:.4f}, cal wrong decodings {v['expected_cal_wrong_decodings']:.1f}, tau median {v['tau_median']}, R median {v['R_median']:.3f} [{v['R_p05']:.3f}, {v['R_p95']:.3f}]", flush=True)


if __name__ == "__main__":
    main()
