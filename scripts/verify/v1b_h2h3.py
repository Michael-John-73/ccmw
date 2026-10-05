"""V1b (development images only): H2 and H3 measurements that must be fixed before the protocol lock (V2).

Same WAM conditions as V1 (scripts/verify/v1_h1.py): COCO val2017 images 0-499, 256x256, same-image multi-message
embedding in equal 10% checkerboard squares, WAM.detect, tau 0.5, DBSCAN eps 1 / min_samples 1000, all 26 WAM
transforms with WAM's own classes, decision checkpoint wam_coco.pth.

H2 (registry attribution, 1,000 registered IDs). Every image has five message slots; each slot is registered
(p 0.5) or unregistered (p 0.5; half uniformly random, half "near": RAW at Hamming 2 from a registered message,
BCH16 at distance 8 = the nearest possible codeword). Scenes per image: k = 0 (no watermark) and k = 1..5, each under
all 26 transforms. Methods (scores, higher = more confident):
  A  WAM-t       RAW, DBSCAN centroid, nearest registered ID by Hamming; score = -distance (accept if d <= t)
  B  RAW-reg-s   RAW, soft correlation of the region's mean logits with the 1,000 registered messages only;
                 score = best correlation (B_max) or best - second (B_margin)
  C  BCH16-s     BCH16, soft ML decoding over the whole 2^16 codebook, accepted only if the codeword is registered;
                 score = best correlation (C_max) or best - second (C_margin)
A false attribution in a scene = an accepted registered ID that was not embedded in the scene. Reported per method:
partial AUC of correct attribution R over scene-level false-attribution rate 0-1%, R at 1% and 0.2%, the WAM-t
table for t = 0..8, false attribution by component (no watermark, signal-lost transforms, unregistered random/near,
wrong decoding of a registered message, spurious units), image-paired differences and the test sample size they imply.
Dev rates are per scene; scenes of one image are not independent (the test decision unit is fixed in V2).

H3 (message distance and DBSCAN merging). k = 2 messages in cells (0,0) and (2,2); pairs: RAW at Hamming distance
1, 2, 3, 4, 6, 8 and random; BCH16 nearest codewords (distance 8) and random codewords; transforms none and
hflip + contrast 1.5. Merge = one DBSCAN unit covers >= 50% of the visible area of both messages. Reported: merge
rate, both-recovered rate, image-paired differences against the random pair (98.75%), and sample sizes for
detecting the difference and for equivalence margins 0.02 / 0.05."""
import gzip
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v1_h1 as v1  # noqa: E402  (imports oracle_h1: sys.path / cwd for the WAM repository)
from v1_h1 import base  # noqa: E402
from h1_pretest import bch_codebook  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402
from notebooks.inference_utils import load_model_from_checkpoint  # noqa: E402
from watermark_anything.data.transforms import default_transform  # noqa: E402
from wamset.config import progress, stable_seed  # noqa: E402
from wamset.dbscan_gpu import dbscan_bits  # noqa: E402

OUT = Path("/workspace/wam/artifacts/v1b")
CKPT = "wam_coco"
SEED, REG_N, BATCH, MIN_VISIBLE = 20261008, 1000, 25, 1000
SIGNAL_LOST = {"hflip_contrast1.5_jpeg80", "combination", "jpeg50"}
H3_CONDS = ["none", "hflip_contrast1.5"]
RAW_D = [1, 2, 3, 4, 6, 8]
H3_TYPES = [f"RAW_d{d}" for d in RAW_D] + ["RAW_rand", "BCH_near", "BCH_rand"]
VARIANTS = ["A", "B_max", "B_margin", "C_max", "C_margin"]
NEG = float("-inf")
Z2, Z1, ZP = norm.ppf(1 - 0.0125 / 2), norm.ppf(1 - 0.0125), norm.ppf(0.90)
BOOT = {"repeats": 10000, "seed": 20261009, "level": 0.9875}
POW = np.arange(32, dtype=np.uint64)


def log(msg):
    print(f"[V1b] {msg}", flush=True)


def int_bits(x):
    return ((np.uint64(x) >> POW) & np.uint64(1)).astype(np.uint8)


# ------------------------------------------------------------------ messages and registries
def setup(n):
    rng = np.random.default_rng(SEED)
    _, _, bch = bch_codebook(3)
    reg_ids = rng.choice(len(bch), REG_N, replace=False)
    reg_mask = np.zeros(len(bch), bool)
    reg_mask[reg_ids] = True
    raw_reg, seen = [], set()
    while len(raw_reg) < REG_N:
        x = int(rng.integers(0, 1 << 32, dtype=np.uint64))
        if x not in seen:
            seen.add(x)
            raw_reg.append(x)
    raw_index = {x: i for i, x in enumerate(raw_reg)}

    def bch_near(base_id):
        dist = (bch != bch[base_id]).sum(1)
        cand = np.nonzero((dist == 8) & ~reg_mask)[0]
        return int(rng.choice(cand))

    def raw_flip(x, d):
        pos = rng.choice(32, d, replace=False)
        for p in pos:
            x ^= 1 << int(p)
        return x

    h2 = []  # per image: list of 5 slots {registered, utype, raw (int), raw_id, bch_id}
    for _ in range(n):
        slots, used_raw, used_bch = [], set(), set()
        for _ in range(5):
            registered = bool(rng.random() < .5)
            utype = "reg" if registered else ("near" if rng.random() < .5 else "rand")
            while True:
                if registered:
                    r = int(rng.integers(REG_N))
                    raw, raw_id, bch_id = raw_reg[r], r, int(reg_ids[int(rng.integers(REG_N))])
                elif utype == "rand":
                    raw = int(rng.integers(0, 1 << 32, dtype=np.uint64))
                    bch_id = int(rng.integers(len(bch)))
                    raw_id = -1
                    if raw in raw_index or reg_mask[bch_id]:
                        continue
                else:
                    raw = raw_flip(raw_reg[int(rng.integers(REG_N))], 2)
                    bch_id = bch_near(int(reg_ids[int(rng.integers(REG_N))]))
                    raw_id = -1
                    if raw in raw_index:
                        continue
                if raw not in used_raw and bch_id not in used_bch:
                    break
            used_raw.add(raw)
            used_bch.add(bch_id)
            slots.append({"registered": registered, "utype": utype, "raw": raw, "raw_id": raw_id, "bch_id": bch_id})
        h2.append(slots)
    h3 = []  # per image: {type: (bits m1, bits m2, ids or None)}
    for _ in range(n):
        pairs = {}
        for d in RAW_D:
            m1 = int(rng.integers(0, 1 << 32, dtype=np.uint64))
            pairs[f"RAW_d{d}"] = (int_bits(m1), int_bits(raw_flip(m1, d)), None)
        a, b = (int(rng.integers(0, 1 << 32, dtype=np.uint64)) for _ in range(2))
        pairs["RAW_rand"] = (int_bits(a), int_bits(b), None)
        c1 = int(rng.integers(len(bch)))
        dist = (bch != bch[c1]).sum(1)
        c2 = int(rng.choice(np.nonzero(dist == 8)[0]))
        pairs["BCH_near"] = (bch[c1], bch[c2], (c1, c2))
        c1, c2 = (int(v) for v in rng.choice(len(bch), 2, replace=False))
        pairs["BCH_rand"] = (bch[c1], bch[c2], (c1, c2))
        h3.append(pairs)
    raw_reg_bits = np.stack([int_bits(x) for x in raw_reg])
    return {"bch": bch, "reg_ids": reg_ids, "reg_mask": reg_mask, "raw_reg": raw_reg, "raw_reg_bits": raw_reg_bits,
            "h2": h2, "h3": h3}


# ------------------------------------------------------------------ per-scene evaluation
def units_of(pred, gts):
    centers, labels = dbscan_bits(pred, 1000, 1.0)
    lab = torch.as_tensor(labels, device=pred.device)
    logits = pred[1:]
    out = []
    for cid, cen in centers.items():
        region = lab == cid
        if gts.shape[0]:
            ov = (gts & region[None]).flatten(1).sum(1)
            j = int(ov.argmax()) if int(ov.max()) > 0 else -1
        else:
            ov, j = None, -1
        out.append((j, ov, torch.as_tensor(cen, device=pred.device).float(), logits[:, region].mean(1)))
    return out


def h2_eval(units, emb, family, G):
    """emb: [(registered, id, utype, eligible)]; ids are registry indices (RAW) or codeword indices (BCH)."""
    emb_reg = {e[1] for e in emb if e[0]}
    variants = ["A", "B_max", "B_margin"] if family == "RAW" else ["C_max", "C_margin"]
    nulls = {v: {} for v in variants}
    succ = {v: [NEG] * len(emb) for v in variants}
    for j, _, cen, l in units:
        src = "spurious" if j < 0 else ("reg_wrong" if emb[j][0] else "unreg_" + emb[j][2])
        if family == "RAW":
            dist = (G["raw_bits"] != cen[None]).sum(1)
            ra = int(dist.argmin())
            top = torch.topk(G["raw_pm"] @ l, 2)
            rb, s1, s2 = int(top.indices[0]), float(top.values[0]), float(top.values[1])
            decs = {"A": (ra, -float(dist[ra])), "B_max": (rb, s1), "B_margin": (rb, s1 - s2)}
            registered = True
        else:
            top = torch.topk(G["bch_pm"] @ l, 2)
            c, s1, s2 = int(top.indices[0]), float(top.values[0]), float(top.values[1])
            decs = {"C_max": (c, s1), "C_margin": (c, s1 - s2)}
            registered = bool(G["reg_mask"][c])
        for v, (dec, sc) in decs.items():
            if registered and dec not in emb_reg:
                nulls[v][src] = max(nulls[v].get(src, NEG), sc)
            if j >= 0 and emb[j][0] and dec == emb[j][1]:
                succ[v][j] = max(succ[v][j], sc)
    return {v: {"null": nulls[v], "succ": [succ[v][j] for j in range(len(emb)) if emb[j][0] and emb[j][3]]} for v in variants}


def h3_eval(units, gts, pair, G):
    vis = gts.flatten(1).sum(1)
    merged = any(ov is not None and int(ov[0]) >= .5 * int(vis[0]) and int(ov[1]) >= .5 * int(vis[1]) for _, ov, _, _ in units)
    rec = [False, False]
    m = [torch.as_tensor(pair[0], device=gts.device).float(), torch.as_tensor(pair[1], device=gts.device).float()]
    for j, _, cen, l in units:
        if j < 0:
            continue
        if pair[2] is None:
            rec[j] |= bool(torch.equal(cen, m[j]))
        else:
            rec[j] |= int((G["bch_pm"] @ l).argmax()) == pair[2][j]
    return {"merged": merged, "rec": rec, "units": len(units)}


# ------------------------------------------------------------------ analysis
def bootstrap(d):
    rng = np.random.default_rng(BOOT["seed"])
    boot = d[rng.integers(0, len(d), (BOOT["repeats"], len(d)))].mean(1)
    a = (1 - BOOT["level"]) / 2
    return [float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))]


def n_detect(sd, delta):
    return None if delta <= 0 or sd == 0 else math.ceil(((Z2 + ZP) * sd / delta) ** 2)


def h2_analysis(records):
    by_v = {v: [r for r in records if v in r["m"]] for v in VARIANTS}
    out, thr_used = {}, {}
    grid = np.linspace(1e-4, 0.01, 100)
    for v, rs in by_v.items():
        nulls = np.array([max(r["m"][v]["null"].values(), default=NEG) for r in rs])
        succ = np.array([s for r in rs for s in r["m"][v]["succ"]])
        srt = np.sort(nulls)[::-1]

        def thr_at(f):
            m = int(math.floor(f * len(srt) + 1e-9))
            return srt[m] if m < len(srt) else NEG
        res = {"scenes": len(rs), "registered_messages": int(len(succ)),
               "pAUC_0_1pct": float(np.mean([np.mean(succ > thr_at(f)) for f in grid])),
               "R_no_threshold": float(np.mean(succ > NEG))}
        for f in (0.01, 0.002):
            thr = thr_at(f)
            res[f"thr_{f}"] = float(thr)
            res[f"R_at_FA_{f}"] = float(np.mean(succ > thr))
            res[f"FA_at_{f}"] = float(np.mean(nulls > thr))
        thr = thr_at(0.01)
        thr_used[v] = thr
        groups = {"k0_no_watermark": [r for r in rs if r["k"] == 0],
                  "signal_lost": [r for r in rs if r["k"] > 0 and r["cond"] in SIGNAL_LOST],
                  "other": [r for r in rs if r["k"] > 0 and r["cond"] not in SIGNAL_LOST]}
        res["FA_by_group_at_1pct"] = {g: float(np.mean([max(r["m"][v]["null"].values(), default=NEG) > thr for r in gr])) for g, gr in groups.items()}
        res["FA_by_source_at_1pct"] = {s: float(np.mean([r["m"][v]["null"].get(s, NEG) > thr for r in rs]))
                                       for s in ("spurious", "reg_wrong", "unreg_rand", "unreg_near")}
        res["FA_no_threshold"] = float(np.mean(nulls > NEG))
        if v == "A":
            res["WAM_t_table"] = {t: {"FA": float(np.mean(nulls >= -t)), "R": float(np.mean(succ >= -t))} for t in range(9)}
            ok = [t for t in range(9) if res["WAM_t_table"][t]["FA"] <= 0.01]
            res["t_max_FA_le_1pct"] = max(ok) if ok else None
        out[v] = res
    best_b = max(("B_max", "B_margin"), key=lambda v: out[v]["pAUC_0_1pct"])
    best_c = max(("C_max", "C_margin"), key=lambda v: out[v]["pAUC_0_1pct"])

    def per_image(v):
        acc = defaultdict(list)
        for r in by_v[v]:
            acc[r["image"]].extend(r["m"][v]["succ"])
        return {i: float(np.mean(np.array(s) > thr_used[v])) for i, s in acc.items() if s}
    comps = {}
    pc = per_image(best_c)
    for ref in ("A", best_b):
        pr = per_image(ref)
        keys = sorted(set(pc) & set(pr))
        d = np.array([pc[i] - pr[i] for i in keys])
        sd = float(d.std(ddof=1))
        comps[f"{best_c}-{ref}"] = {"diff": float(d.mean()), "ci": bootstrap(d), "sd_per_image": sd,
                                    "n_detect_observed": n_detect(sd, float(d.mean())), "n_detect_0.05": n_detect(sd, 0.05)}
    return {"variants": out, "selected": {"B": best_b, "C": best_c}, "comparisons_at_dev_1pct": comps}


def h3_analysis(records):
    out = {}
    for cond in H3_CONDS:
        rows = {}
        per = {t: {r["image"]: r for r in records if r["type"] == t and r["cond"] == cond} for t in H3_TYPES}
        for t in H3_TYPES:
            rr = list(per[t].values())
            rows[t] = {"merge_rate": float(np.mean([r["merged"] for r in rr])),
                       "both_recovered": float(np.mean([all(r["rec"]) for r in rr])),
                       "message_recovered": float(np.mean([x for r in rr for x in r["rec"]])),
                       "units_per_image": float(np.mean([r["units"] for r in rr]))}
        comps = {}
        for t, ref in [(f"RAW_d{d}", "RAW_rand") for d in RAW_D] + [("BCH_near", "BCH_rand")]:
            keys = sorted(set(per[t]) & set(per[ref]))
            for metric, f in (("merge", lambda r: float(r["merged"])), ("both_recovered", lambda r: float(all(r["rec"])))):
                d = np.array([f(per[t][i]) - f(per[ref][i]) for i in keys])
                sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
                c = {"diff": float(d.mean()), "ci": bootstrap(d), "sd_per_image": sd,
                     "n_detect_observed": n_detect(sd, abs(float(d.mean())))}
                for m in (0.02, 0.05):
                    c[f"n_equivalence_{m}"] = (math.ceil(((Z1 + ZP) * sd / m) ** 2) if sd > 0
                                               else math.ceil(math.log(0.0125) / math.log(1 - m)))
                comps[f"{t}-{ref}:{metric}"] = c
        out[cond] = {"rates": rows, "comparisons": comps}
    return out


# ------------------------------------------------------------------ main
def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    path, digest = v1.checkpoint(CKPT)
    wam = load_model_from_checkpoint(str(base.REPO / "checkpoints" / "params.json"), str(path)).to("cuda:0").eval()
    log(f"{CKPT}: sha256 {digest[:16]}..., loaded")
    files = sorted(base.COCO.glob("*.jpg"))[v1.DEV[0]:v1.DEV[1]]
    resize = transforms.Resize((base.SIZE, base.SIZE))
    imgs = torch.stack([default_transform(resize(Image.open(f).convert("RGB"))) for f in files])
    masks = base.checkerboard_masks(torch.device("cuda:0"))
    n = len(files)
    S = setup(n)
    log(f"registries: BCH16 {REG_N} of {len(S['bch'])} codewords, RAW {REG_N} random 32-bit messages; setup {time.perf_counter() - t0:.0f}s")
    G = {"raw_bits": torch.as_tensor(S["raw_reg_bits"], device="cuda:0").float(),
         "raw_pm": torch.as_tensor(S["raw_reg_bits"], device="cuda:0").float() * 2 - 1,
         "bch_pm": torch.as_tensor(S["bch"], device="cuda:0").float() * 2 - 1,
         "reg_mask": S["reg_mask"]}
    h2_rec, h3_rec = [], []
    for s in progress(range(0, n, BATCH), desc="V1b (batch of 25 images)", unit="batch"):
        x = imgs[s:s + BATCH].cuda()
        nb = len(x)
        with torch.no_grad():
            for cname, ops in v1.CONDITIONS.items():  # k = 0: original images, no watermark
                gt0 = torch.zeros((nb, 1, base.SIZE, base.SIZE), device=x.device)
                img, _ = v1.apply(x.clone(), gt0, ops, stable_seed(SEED, "v1b", 0, s, cname))
                preds = wam.detect(img)["preds"].float()
                for b in range(nb):
                    units = units_of(preds[b], torch.zeros((0, base.SIZE, base.SIZE), dtype=torch.bool, device=x.device))
                    m = {**h2_eval(units, [], "RAW", G), **h2_eval(units, [], "BCH", G)}
                    h2_rec.append({"image": s + b, "k": 0, "cond": cname, "set": "NONE", "m": m})
            for fam in ("RAW", "BCH"):
                bits = np.stack([[int_bits(sl["raw"]) if fam == "RAW" else S["bch"][sl["bch_id"]] for sl in S["h2"][s + b]] for b in range(nb)])
                msgs = torch.as_tensor(bits, device=x.device).float()
                wm = [wam.embed(x, msgs[:, i])["imgs_w"] for i in range(5)]
                for k in range(1, 6):
                    multi = x.clone()
                    for i in range(k):
                        mk = masks[i][None, None]
                        multi = wm[i] * mk + multi * (1 - mk)
                    for cname, ops in v1.CONDITIONS.items():
                        gt0 = masks[:k][None].expand(nb, -1, -1, -1).clone()
                        img, gt = v1.apply(multi.clone(), gt0, ops, stable_seed(SEED, "v1b", k, s, cname))
                        preds = wam.detect(img)["preds"].float()
                        for b in range(nb):
                            vis = gt[b].flatten(1).sum(1)
                            emb = [(sl["registered"], sl["raw_id"] if fam == "RAW" else sl["bch_id"], sl["utype"], int(vis[j]) >= MIN_VISIBLE)
                                   for j, sl in enumerate(S["h2"][s + b][:k])]
                            units = units_of(preds[b], gt[b])
                            h2_rec.append({"image": s + b, "k": k, "cond": cname, "set": fam, "m": h2_eval(units, emb, fam, G)})
            for t in H3_TYPES:
                pair_bits = torch.as_tensor(np.stack([[S["h3"][s + b][t][0], S["h3"][s + b][t][1]] for b in range(nb)]), device=x.device).float()
                wm = [wam.embed(x, pair_bits[:, i])["imgs_w"] for i in range(2)]
                multi = x.clone()
                for i in range(2):
                    mk = masks[i][None, None]
                    multi = wm[i] * mk + multi * (1 - mk)
                for cname in H3_CONDS:
                    gt0 = masks[:2][None].expand(nb, -1, -1, -1).clone()
                    img, gt = v1.apply(multi.clone(), gt0, v1.CONDITIONS[cname], stable_seed(SEED, "v1b-h3", s, cname))
                    preds = wam.detect(img)["preds"].float()
                    for b in range(nb):
                        units = units_of(preds[b], gt[b])
                        h3_rec.append({"image": s + b, "type": t, "cond": cname, **h3_eval(units, gt[b], S["h3"][s + b][t], G)})
    log(f"measured: {len(h2_rec)} H2 scenes, {len(h3_rec)} H3 scenes ({time.perf_counter() - t0:.0f}s)")
    with gzip.open(OUT / "v1b_h2_records.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in h2_rec:
            f.write(json.dumps(r) + "\n")
    with gzip.open(OUT / "v1b_h3_records.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in h3_rec:
            f.write(json.dumps(r) + "\n")
    summary = {"checkpoint": {CKPT: digest}, "images": [f.name for f in files], "seed": SEED,
               "registry": {"BCH16_ids": [int(v) for v in S["reg_ids"]], "RAW_ints": [int(v) for v in S["raw_reg"]]},
               "H2": h2_analysis(h2_rec), "H3": h3_analysis(h3_rec)}
    summary["seconds"] = time.perf_counter() - t0
    (OUT / "v1b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    h2 = summary["H2"]
    log("H2 (dev, per scene): pAUC of correct attribution over false attribution 0-1%, R at FA 1% / 0.2%, FA without threshold")
    print(f"{'method':9s} {'pAUC':>6s} {'R@1%':>6s} {'R@0.2%':>6s} {'R(no thr)':>9s} {'FA(no thr)':>10s} | FA@1% by group (k0 / lost / other) | by source (spur / regwrong / unreg_rand / unreg_near)", flush=True)
    for v, r in h2["variants"].items():
        g, so = r["FA_by_group_at_1pct"], r["FA_by_source_at_1pct"]
        print(f"{v:9s} {r['pAUC_0_1pct']:6.3f} {r['R_at_FA_0.01']:6.3f} {r['R_at_FA_0.002']:6.3f} {r['R_no_threshold']:9.3f} {r['FA_no_threshold']:10.4f} | "
              f"{g['k0_no_watermark']:.4f} / {g['signal_lost']:.4f} / {g['other']:.4f} | "
              f"{so['spurious']:.4f} / {so['reg_wrong']:.4f} / {so['unreg_rand']:.4f} / {so['unreg_near']:.4f}", flush=True)
    log("WAM-t table (A): t: FA / R")
    print("  " + "  ".join(f"t={t}: {v['FA']:.4f}/{v['R']:.3f}" for t, v in h2["variants"]["A"]["WAM_t_table"].items()), flush=True)
    log(f"selected scores: {h2['selected']}; WAM-t largest t with dev FA <= 1%: {h2['variants']['A']['t_max_FA_le_1pct']}")
    for k, c in h2["comparisons_at_dev_1pct"].items():
        log(f"H2 {k} at dev 1% thresholds: {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}], per-image sd {c['sd_per_image']:.3f}, "
            f"n (observed effect) {c['n_detect_observed']}, n (effect 0.05) {c['n_detect_0.05']}")
    for cond, h in summary["H3"].items():
        log(f"H3 {cond}: merge rate / both recovered / units per image")
        print("  " + "  ".join(f"{t}: {r['merge_rate']:.3f}/{r['both_recovered']:.3f}/{r['units_per_image']:.2f}" for t, r in h["rates"].items()), flush=True)
        for k, c in h["comparisons"].items():
            print(f"    {k:34s} {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] sd {c['sd_per_image']:.3f} "
                  f"n_detect {c['n_detect_observed']} n_eq0.02 {c['n_equivalence_0.02']} n_eq0.05 {c['n_equivalence_0.05']}", flush=True)
    log(f"done in {summary['seconds']:.0f}s")


if __name__ == "__main__":
    main()
