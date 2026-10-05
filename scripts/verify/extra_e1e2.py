"""Extra analyses E1 and E2 (configs/protocol_v5_addendum_E.json, locked). Existing records only; no new measurement.

E1: BCH16 soft vs hard decoding on the V4 test records (wam_coco).
E2: H2 threat-model sensitivity: null scores without the 'unreg_near' source, recalibrated on the V3 cal records with
the locked rule and applied to the V4 test records. As a check, the locked (with near-miss) V4 numbers are recomputed
from the same records and must match v4_summary.json."""
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

STUDY = Path(__file__).resolve().parents[2]
ART = Path("/workspace/wam/artifacts")
OUT = ART / "extra"
ADD = STUDY / "configs" / "protocol_v5_addendum_E.json"
ADD_LOCK = STUDY / "configs" / "protocol_v5_addendum_E_lock.json"
BASE_LOCK = STUDY / "configs" / "protocol_v5_lock.json"
LEVEL, SEED, REPS = 0.9875, 20261006, 10000
NEG = float("-inf")
METHODS = {"A": "A", "B": "B_margin", "C": "C_max"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    al, bl = json.loads(ADD_LOCK.read_text(encoding="utf-8")), json.loads(BASE_LOCK.read_text(encoding="utf-8"))
    bad = []
    if sha(ADD) != al["addendum_sha256"]:
        bad.append("addendum")
    if sha(STUDY / "configs" / "protocol_v5.json") != bl["protocol_sha256"] or al["base_protocol_sha256"] != bl["protocol_sha256"]:
        bad.append("base protocol")
    if bad:
        raise SystemExit(f"[E] STOP: hash mismatch {bad}")
    print(f"[E] verified addendum {al['addendum_sha256'][:16]}..., base protocol {bl['protocol_sha256'][:16]}...", flush=True)
    return al


def boot(d):
    d = np.asarray(d, float)
    bt = d[np.random.default_rng(SEED).integers(0, len(d), (REPS, len(d)))].mean(1)
    a = (1 - LEVEL) / 2
    return {"diff": float(d.mean()), "ci": [float(np.quantile(bt, a)), float(np.quantile(bt, 1 - a))], "n": int(len(d)),
            "p_one_sided": float((np.sum(bt <= 0) + 1) / (REPS + 1))}


def holm(p):
    names, out, run = sorted(p, key=p.get), {}, 0.0
    for i, n in enumerate(names):
        run = max(run, min(1.0, (len(names) - i) * p[n]))
        out[n] = run
    return out


def cp_upper(x, n):
    return 1.0 if x >= n else float(beta_dist.ppf(LEVEL, x + 1, n - x))


def e1():
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    with gzip.open(ART / "v4" / "v4_h1_records_wam_coco.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            if '"set": "BCH16"' not in line:
                continue
            r = json.loads(line)
            for m in r["messages"]:
                if m["eligible"]:
                    a = acc[r["condition"]][r["image"]]
                    a[0] += 1
                    a[1] += m["soft"]
                    a[2] += m["hard"]
    out = {}
    for c, imgs in acc.items():
        tot = np.array(list(imgs.values())).sum(0)
        out[c] = {"soft": tot[1] / tot[0], "hard": tot[2] / tot[0],
                  "soft_minus_hard": boot([(a[1] - a[2]) / a[0] for a in imgs.values() if a[0]])}
    adj = holm({c: v["soft_minus_hard"]["p_one_sided"] for c, v in out.items()})
    for c in out:
        out[c]["holm_p"] = adj[c]
    return out


def null_score(rec, v, exclude):
    return max((s for src, s in rec["m"][v]["null"].items() if src not in exclude), default=NEG)


def tau_rule(nulls, n_cal=1000, alpha=0.002):
    allowed = math.floor(alpha * (n_cal + 1) - 1 + 1e-9)
    finite = sorted((x for x in nulls if math.isfinite(x)), reverse=True)
    return finite[allowed] if allowed < len(finite) else NEG


def e2_block(cal, test, exclude):
    res, rates = {}, {}
    for name, v in METHODS.items():
        tau = tau_rule([null_score(r, v, exclude) for r in cal])
        nulls = np.array([null_score(r, v, exclude) for r in test])
        fa = int((nulls > tau).sum())
        succ = np.array([x for r in test for x in r["m"][v]["succ"]])
        res[name] = {"tau": tau, "t_equivalent": (int(math.ceil(-tau) - 1) if name == "A" and math.isfinite(tau) else None),
                     "false_attributions": fa, "FA_rate": fa / len(test), "FA_upper_98.75": cp_upper(fa, len(test)),
                     "R": float(np.mean(succ > tau)), "registered_eligible_messages": int(len(succ))}
        rates[name] = {r["image"]: float(np.mean(np.array(r["m"][v]["succ"]) > tau)) for r in test if r["m"][v]["succ"]}
    for ref in ("A", "B"):
        keys = sorted(set(rates["C"]) & set(rates[ref]))
        res[f"R_C_minus_{ref}"] = boot([rates["C"][i] - rates[ref][i] for i in keys])
    return res


def e2():
    load = lambda p: [json.loads(l) for l in gzip.open(p, "rt", encoding="utf-8")]  # noqa: E731
    cal, test = load(ART / "v3" / "v3_cal_records.jsonl.gz"), load(ART / "v4" / "v4_h2_records.jsonl.gz")
    assert len(cal) == 1000 and len(test) == 3000
    locked = e2_block(cal, test, set())
    v4 = json.loads((ART / "v4" / "v4_summary.json").read_text(encoding="utf-8"))["H2"]
    check = all(locked[n]["false_attributions"] == v4["methods"][n]["false_attributions"] and abs(locked[n]["R"] - v4["methods"][n]["R"]) < 1e-12
                for n in METHODS) and abs(locked["R_C_minus_A"]["diff"] - v4["R_C_minus_A"]["diff"]) < 1e-12
    if not check:
        raise SystemExit("[E2] STOP: recomputed locked H2 numbers do not match v4_summary.json")
    no_near = e2_block(cal, test, {"unreg_near"})
    cal_near = sum(1 for r in cal for v in METHODS.values() if "unreg_near" in r["m"][v]["null"])
    return {"check_locked_reproduced": True, "locked_with_near_miss": locked, "without_near_miss": no_near,
            "approximation": "near-miss messages remain embedded; only their units' scores are excluded from the null scores",
            "cal_scene_method_pairs_with_near_null": cal_near}


def main():
    al = verify()
    OUT.mkdir(parents=True, exist_ok=True)
    r1 = e1()
    s1 = {"addendum_sha256": al["addendum_sha256"], "script_sha256": sha(__file__), "E1": r1}
    (OUT / "extra_E1_summary.json").write_text(json.dumps(s1, indent=2), encoding="utf-8")
    h = r1["hflip_contrast1.5"]
    print(f"[E1] hflip_contrast1.5: soft {h['soft']:.3f}, hard {h['hard']:.3f}, soft - hard {h['soft_minus_hard']['diff']:+.4f} "
          f"[{h['soft_minus_hard']['ci'][0]:+.4f}, {h['soft_minus_hard']['ci'][1]:+.4f}] (images {h['soft_minus_hard']['n']})", flush=True)
    for c, v in r1.items():
        x = v["soft_minus_hard"]
        print(f"     {c:26s} soft {v['soft']:.3f} hard {v['hard']:.3f} diff {x['diff']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] holm p {v['holm_p']:.4f}", flush=True)
    r2 = e2()
    s2 = {"addendum_sha256": al["addendum_sha256"], "script_sha256": sha(__file__), "E2": r2}
    (OUT / "extra_E2_summary.json").write_text(json.dumps(s2, indent=2), encoding="utf-8")
    print("[E2] check: locked V4 H2 numbers reproduced from records", flush=True)
    for label in ("locked_with_near_miss", "without_near_miss"):
        b = r2[label]
        print(f"[E2] {label}:", flush=True)
        for n in METHODS:
            m = b[n]
            t = f" (t = {m['t_equivalent']})" if n == "A" else ""
            print(f"     {n}: tau {m['tau']}{t}; FA {m['false_attributions']}/3000 upper {m['FA_upper_98.75']:.4f}; R {m['R']:.3f}", flush=True)
        for ref in ("A", "B"):
            c = b[f"R_C_minus_{ref}"]
            print(f"     R(C) - R({ref}) {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] (images {c['n']})", flush=True)


if __name__ == "__main__":
    main()
