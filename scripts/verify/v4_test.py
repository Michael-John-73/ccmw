"""V4: final test measurements and decisions for H1-H4 (configs/protocol_v5.json, locked in V2; H2 thresholds from V3).

Before running, the lock hashes are verified (protocol, code files, WAM checkpoints, test image list) and the V3
threshold file must carry the locked protocol hash; any mismatch stops the run. Test images: val2017 indices
1500-4499 (3,000). Seeds derive from the test seed 20261102.

Order: (1) H1/H4 on wam_coco: message sets RAW, BCH16, BCH21, RND16, REP16 (five messages per image and set),
k = 1..5, all 26 transforms; (2) H2: one scene per image (V3 procedure with the test seed), methods A, B, C with the
V3 thresholds; (3) H3: nine pair types x two transforms; (4) decisions written and printed; (5) H1/H4 secondary on
wam_mit. Per-image records are streamed to jsonl.gz before any summary is computed.
Transforms: images in index order, batches of 25; WAM's random transforms drawn once per batch after
torch.manual_seed(stable_seed(20261102, tag, k, batch_start, transform)), tag = v5-test-h1 / v5-test-h2 / v5-test-h3."""
import gzip
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta as beta_dist

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
sys.path.insert(0, str(HERE))
import v1_h1 as v1  # noqa: E402
import v1b_h2h3 as v1b  # noqa: E402
import v3_cal as v3  # noqa: E402
from v1_h1 import base  # noqa: E402
from h1_pretest import bch_codebook  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402
from notebooks.inference_utils import load_model_from_checkpoint  # noqa: E402
from watermark_anything.data.transforms import default_transform  # noqa: E402
from wamset.config import progress, stable_seed  # noqa: E402

PROTOCOL = STUDY / "configs" / "protocol_v5.json"
LOCK = STUDY / "configs" / "protocol_v5_lock.json"
THRESH = Path("/workspace/wam/artifacts/v3/v3_thresholds.json")
OUT = Path("/workspace/wam/artifacts/v4")
SETS = ["RAW", "BCH16", "BCH21", "RND16", "REP16"]
H2_METHODS = {"A": "A", "B": "B_margin", "C": "C_max"}
ALPHA = 0.0125
LEVEL = 1 - ALPHA
NEG = float("-inf")


def log(msg):
    print(f"[V4] {msg}", flush=True)


def verify():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    problems = []
    if v3.sha(PROTOCOL) != lock["protocol_sha256"]:
        problems.append("protocol sha256")
    for rel, digest in proto["code"].items():
        if v3.sha(STUDY / rel) != digest:
            problems.append(f"code {rel}")
    for name in ("wam_coco", "wam_mit"):
        if v3.sha(base.REPO / "checkpoints" / f"{name}.pth") != lock["checkpoints_sha256"][name]:
            problems.append(f"checkpoint {name}")
    files = sorted(base.COCO.glob("*.jpg"))
    a, b = proto["images"]["splits"]["test"]["index"]
    test = files[a:b + 1]
    blob = "".join(f"{f.name}:{v3.sha(f)}\n" for f in test).encode()
    if hashlib.sha256(blob).hexdigest() != lock["image_lists_sha256"]["test"]:
        problems.append("test image list")
    th = json.loads(THRESH.read_text(encoding="utf-8"))
    if th["protocol_sha256"] != lock["protocol_sha256"]:
        problems.append("V3 thresholds protocol hash")
    if problems:
        raise SystemExit(f"[V4] STOP: hash mismatch: {problems}")
    log(f"verified: protocol {lock['protocol_sha256'][:16]}..., code {len(proto['code'])}, checkpoints 2, test images {len(test)}, "
        f"V3 thresholds {v3.sha(THRESH)[:16]}...")
    return proto, lock, test, th


# ------------------------------------------------------------------ statistics
def boot_ci(d, seed):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    bt = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    lo, hi = np.quantile(bt, [ALPHA / 2, 1 - ALPHA / 2])
    p_one = float((np.sum(bt <= 0) + 1) / (len(bt) + 1))
    return {"diff": float(d.mean()), "ci": [float(lo), float(hi)], "n": int(len(d)), "p_one_sided": p_one}


def holm(pvals):
    names = sorted(pvals, key=pvals.get)
    out, m, running = {}, len(names), 0.0
    for i, n in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvals[n]))
        out[n] = running
    return out


def cp_upper(x, n):
    return 1.0 if x >= n else float(beta_dist.ppf(LEVEL, x + 1, n - x))


# ------------------------------------------------------------------ H1 / H4
def new_acc():
    return defaultdict(float)


def add_h1(acc, res, kind):
    for m in res["messages"]:
        acc["ba_sum"] += sum(m["bit_acc"])
        acc["ba_n"] += len(m["bit_acc"])
        if not m["eligible"]:
            acc["excluded"] += 1
            continue
        acc["elig"] += 1
        if kind == "RAW":
            acc["ok"] += m["exact"]
        else:
            acc["ok"] += m["soft"]
            acc["hard"] += m["hard"]
        if m["found"]:
            acc["tp"] += 1
            acc["dup"] += len(m["bit_acc"]) - 1
            acc["sq"] += max(m["bit_acc"])
        else:
            acc["fn"] += 1
    acc["fp"] += res["spurious"]


def run_h1(name, imgs, books_gpu, sets, masks, seed, fh):
    path = base.REPO / "checkpoints" / f"{name}.pth"
    wam = load_model_from_checkpoint(str(base.REPO / "checkpoints" / "params.json"), str(path)).to("cuda:0").eval()
    per_img = defaultdict(lambda: defaultdict(new_acc))  # (set, cond) -> image -> acc
    per_k = defaultdict(new_acc)  # (set, cond, k) -> acc
    n = len(imgs)
    jobs = [(kind, s) for kind in SETS for s in range(0, n, v1.BATCH)]
    for kind, s in progress(jobs, desc=f"V4 H1/H4 {name} (set x batch)", unit="batch"):
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
                    img, gt = v1.apply(multi.clone(), gt0, ops, stable_seed(seed, "v5-test-h1", k, s, cname))
                    preds = wam.detect(img)["preds"].float()
                    for b in range(len(x)):
                        ids = None if ids_all is None else [int(v) for v in ids_all[s + b, :k]]
                        res = v1.evaluate(preds[b], gt[b], kind, ids, msgs[b, :k], books_gpu.get(kind))
                        fh.write(json.dumps({"ckpt": name, "set": kind, "condition": cname, "image": s + b, "k": k, **res}) + "\n")
                        add_h1(per_img[(kind, cname)][s + b], res, kind)
                        add_h1(per_k[(kind, cname, k)], res, kind)
    del wam
    torch.cuda.empty_cache()
    return per_img, per_k


def rate(acc, field="ok"):
    return acc[field] / acc["elig"] if acc["elig"] else float("nan")


def h1_summary(per_img, per_k, boot_seed):
    out = {"conditions": {}}
    for cname in v1.CONDITIONS:
        row = {}
        for kind in SETS:
            tot = new_acc()
            for a in per_img[(kind, cname)].values():
                for key, v in a.items():
                    tot[key] += v
            fp = tot["fp"] + tot["dup"]
            rq = tot["tp"] / (tot["tp"] + .5 * fp + .5 * tot["fn"]) if tot["tp"] + fp + tot["fn"] else float("nan")
            sq = tot["sq"] / tot["tp"] if tot["tp"] else float("nan")
            row[kind] = {"exact_recovery" if kind == "RAW" else "soft": rate(tot), "eligible": int(tot["elig"]), "excluded": int(tot["excluded"]),
                         "wam_bitacc_found_units": tot["ba_sum"] / tot["ba_n"] if tot["ba_n"] else None,
                         "RQ": rq, "SQ": sq, "PQ": rq * sq, "spurious_units": int(tot["fp"]),
                         "by_k": {k: rate(per_k[(kind, cname, k)]) for k in range(1, 6)}}
            if kind != "RAW":
                row[kind]["hard"] = rate(tot, "hard")

        def paired(a_kind, b_kind):
            A, B = per_img[(a_kind, cname)], per_img[(b_kind, cname)]
            keys = [i for i in sorted(A) if A[i]["elig"] and B[i]["elig"]]
            return boot_ci([rate(A[i]) - rate(B[i]) for i in keys], boot_seed)
        row["BCH16s_minus_WAM"] = paired("BCH16", "RAW")
        row["BCH16s_minus_RND16s"] = paired("BCH16", "RND16")
        row["BCH16s_minus_REP16s"] = paired("BCH16", "REP16")
        row["BCH21s_minus_WAM"] = paired("BCH21", "RAW")
        A = per_img[("RAW", cname)]
        d = [A[i]["ba_sum"] / A[i]["ba_n"] - rate(A[i]) for i in sorted(A) if A[i]["ba_n"] and A[i]["elig"]]
        row["H4_wam_metric_minus_exact"] = {**boot_ci(d, boot_seed), "images_without_units": sum(1 for i in A if not A[i]["ba_n"])}
        out["conditions"][cname] = row
    p = {c: r["BCH16s_minus_WAM"]["p_one_sided"] for c, r in out["conditions"].items()}
    out["holm_BCH16s_minus_WAM"] = holm(p)
    return out


# ------------------------------------------------------------------ H2
def run_h2(wam, test_files, imgs, proto, th, masks, seed, fh):
    _, _, bch = bch_codebook(3)
    v3.proto_reg_n = proto["h2"]["registry_size"]
    reg_ids, raw_reg = v3.build_registry(proto["execution"]["operational_registry_seed"], bch)
    if [int(v) for v in reg_ids] != th["registry"]["BCH16_ids"] or [int(v) for v in raw_reg] != th["registry"]["RAW_ints"]:
        raise SystemExit("[V4] STOP: operational registry differs from the V3 registry")
    conditions = list(proto["transforms"]["list"])
    scenes, reg_mask = v3.build_scenes(len(test_files), seed, bch, reg_ids, raw_reg, conditions)
    raw_bits = np.stack([v1b.int_bits(x) for x in raw_reg])
    G = {"raw_bits": torch.as_tensor(raw_bits, device="cuda:0").float(), "raw_pm": torch.as_tensor(raw_bits, device="cuda:0").float() * 2 - 1,
         "bch_pm": torch.as_tensor(bch, device="cuda:0").float() * 2 - 1, "reg_mask": reg_mask}
    records = {}
    for s in progress(range(0, len(test_files), v1.BATCH), desc="V4 H2 (batch of 25 images)", unit="batch"):
        idx = list(range(s, min(s + v1.BATCH, len(test_files))))
        x = imgs[idx[0]:idx[-1] + 1].cuda()
        with torch.no_grad():
            composite = {}
            for fam in ("RAW", "BCH"):
                bits = np.stack([[v1b.int_bits(sl["raw"]) if fam == "RAW" else bch[sl["bch_id"]] for sl in scenes[i]["slots"]] for i in idx])
                msgs = torch.as_tensor(bits, device="cuda:0").float()
                wm = [wam.embed(x, msgs[:, j])["imgs_w"] for j in range(5)]
                comp = []
                for b, i in enumerate(idx):
                    img = x[b].clone()
                    for j in range(scenes[i]["k"]):
                        mk = masks[j][None]
                        img = wm[j][b] * mk + img * (1 - mk)
                    comp.append(img)
                composite[fam] = torch.stack(comp)
            groups = defaultdict(list)
            for b, i in enumerate(idx):
                groups[(scenes[i]["k"], scenes[i]["cond"])].append(b)
            for (k, cond), members in groups.items():
                tseed = stable_seed(seed, "v5-test-h2", k, s, cond)
                for fam in (("RAW", "BCH") if k else ("NONE",)):
                    src = x[members] if fam == "NONE" else composite[fam][members]
                    gt0 = (masks[:k] if k else torch.zeros((1, base.SIZE, base.SIZE), device="cuda:0"))[None].expand(len(members), -1, -1, -1).clone()
                    img, gt = v1.apply(src.clone(), gt0, v1.CONDITIONS[cond], tseed)
                    preds = wam.detect(img)["preds"].float()
                    for m, b in enumerate(members):
                        i = idx[b]
                        gts = gt[m] if k else torch.zeros((0, base.SIZE, base.SIZE), dtype=torch.bool, device="cuda:0")
                        units = v1b.units_of(preds[m], gts)
                        if fam == "NONE":
                            res = {**v1b.h2_eval(units, [], "RAW", G), **v1b.h2_eval(units, [], "BCH", G)}
                        else:
                            vis = gts.flatten(1).sum(1)
                            emb = [(sl["registered"], sl["raw_id"] if fam == "RAW" else sl["bch_id"], sl["utype"], int(vis[j]) >= v1.MIN_VISIBLE)
                                   for j, sl in enumerate(scenes[i]["slots"][:k])]
                            res = v1b.h2_eval(units, emb, fam, G)
                        rec = records.setdefault(i, {"image": i, "file": test_files[i].name, "k": k, "cond": cond, "m": {}})
                        rec["m"].update(res)
    records = [records[i] for i in sorted(records)]
    for r in records:
        fh.write(json.dumps(r) + "\n")
    return records


def h2_summary(records, th, boot_seed):
    n = len(records)
    out = {"scenes": n, "methods": {}, "k_counts": np.bincount([r["k"] for r in records], minlength=6).tolist()}
    taus = {name: th["methods"][name]["tau"] for name in H2_METHODS}
    acc_rate = {}
    for name, v in H2_METHODS.items():
        tau = taus[name]
        nulls = np.array([max(r["m"][v]["null"].values(), default=NEG) for r in records])
        fa = int((nulls > tau).sum())
        succ = np.array([x for r in records for x in r["m"][v]["succ"]])
        groups = {"k0_no_watermark": [r["k"] == 0 for r in records],
                  "signal_lost": [r["k"] > 0 and r["cond"] in v1b.SIGNAL_LOST for r in records],
                  "other": [r["k"] > 0 and r["cond"] not in v1b.SIGNAL_LOST for r in records]}
        out["methods"][name] = {"variant": v, "tau": tau, "false_attributions": fa, "FA_rate": fa / n, "FA_upper_98.75": cp_upper(fa, n),
                                "R": float(np.mean(succ > tau)) if len(succ) else None, "registered_eligible_messages": int(len(succ)),
                                "FA_by_group": {g: {"n": int(sum(mk)), "rate": float(np.mean(nulls[np.array(mk)] > tau)) if any(mk) else None} for g, mk in groups.items()},
                                "FA_by_source": {src: int(sum(r["m"][v]["null"].get(src, NEG) > tau for r in records))
                                                 for src in ("spurious", "reg_wrong", "unreg_rand", "unreg_near")}}
        acc_rate[name] = {r["image"]: np.mean(np.array(r["m"][v]["succ"]) > tau) for r in records if r["m"][v]["succ"]}
    for ref in ("A", "B"):
        keys = sorted(set(acc_rate["C"]) & set(acc_rate[ref]))
        out[f"R_C_minus_{ref}"] = {**boot_ci([acc_rate["C"][i] - acc_rate[ref][i] for i in keys], boot_seed),
                                   "images_without_registered_eligible": n - len(keys)}
    return out


# ------------------------------------------------------------------ H3
def h3_pairs(n, seed, bch):
    rng = np.random.default_rng([seed, 3])
    pairs = []
    for _ in range(n):
        p = {}
        for d in v1b.RAW_D:
            m1 = int(rng.integers(0, 1 << 32, dtype=np.uint64))
            m2 = m1
            for pos in rng.choice(32, d, replace=False):
                m2 ^= 1 << int(pos)
            p[f"RAW_d{d}"] = (v1b.int_bits(m1), v1b.int_bits(m2), None)
        a, b = (int(rng.integers(0, 1 << 32, dtype=np.uint64)) for _ in range(2))
        p["RAW_rand"] = (v1b.int_bits(a), v1b.int_bits(b), None)
        c1 = int(rng.integers(len(bch)))
        c2 = int(rng.choice(np.nonzero((bch != bch[c1]).sum(1) == 8)[0]))
        p["BCH_near"] = (bch[c1], bch[c2], (c1, c2))
        c1, c2 = (int(v) for v in rng.choice(len(bch), 2, replace=False))
        p["BCH_rand"] = (bch[c1], bch[c2], (c1, c2))
        pairs.append(p)
    return pairs


def run_h3(wam, imgs, masks, seed, fh):
    _, _, bch = bch_codebook(3)
    pairs = h3_pairs(len(imgs), seed, bch)
    G = {"bch_pm": torch.as_tensor(bch, device="cuda:0").float() * 2 - 1}
    records = []
    for s in progress(range(0, len(imgs), v1.BATCH), desc="V4 H3 (batch of 25 images)", unit="batch"):
        x = imgs[s:s + v1.BATCH].cuda()
        nb = len(x)
        with torch.no_grad():
            for t in v1b.H3_TYPES:
                pb = torch.as_tensor(np.stack([[pairs[s + b][t][0], pairs[s + b][t][1]] for b in range(nb)]), device="cuda:0").float()
                wm = [wam.embed(x, pb[:, i])["imgs_w"] for i in range(2)]
                multi = x.clone()
                for i in range(2):
                    mk = masks[i][None, None]
                    multi = wm[i] * mk + multi * (1 - mk)
                for cname in v1b.H3_CONDS:
                    gt0 = masks[:2][None].expand(nb, -1, -1, -1).clone()
                    img, gt = v1.apply(multi.clone(), gt0, v1.CONDITIONS[cname], stable_seed(seed, "v5-test-h3", 2, s, cname))
                    preds = wam.detect(img)["preds"].float()
                    for b in range(nb):
                        units = v1b.units_of(preds[b], gt[b])
                        r = {"image": s + b, "type": t, "cond": cname, **v1b.h3_eval(units, gt[b], pairs[s + b][t], G)}
                        records.append(r)
                        fh.write(json.dumps(r) + "\n")
    return records


def h3_summary(records, boot_seed):
    m = {(r["image"], r["type"], r["cond"]): float(r["merged"]) for r in records}
    rec = {(r["image"], r["type"], r["cond"]): float(all(r["rec"])) for r in records}
    imgs = sorted({r["image"] for r in records})
    conds = v1b.H3_CONDS
    out = {"rates": {c: {t: {"merge": float(np.mean([m[(i, t, c)] for i in imgs])), "both_recovered": float(np.mean([rec[(i, t, c)] for i in imgs]))}
                         for t in v1b.H3_TYPES} for c in conds}}
    d1 = [np.mean([m[(i, "RAW_d1", c)] for c in conds]) - np.mean([m[(i, "RAW_rand", c)] for c in conds]) for i in imgs]
    d2 = [np.mean([m[(i, f"RAW_d{d}", c)] for d in (1, 2, 3, 4) for c in conds]) - np.mean([m[(i, "BCH_near", c)] for c in conds]) for i in imgs]
    out["part1_RAWd1_minus_RAWrand"] = boot_ci(d1, boot_seed)
    out["part2_RAWdle4_minus_BCHnear"] = boot_ci(d2, boot_seed)
    out["secondary"] = {}
    for c in conds:
        for d in (1, 2, 3, 4, 6, 8):
            out["secondary"][f"RAW_d{d}-BCH_near:{c}"] = boot_ci([m[(i, f"RAW_d{d}", c)] - m[(i, "BCH_near", c)] for i in imgs], boot_seed)
        out["secondary"][f"BCH_near-BCH_rand:{c} (equivalence +-0.02, reported)"] = boot_ci([m[(i, "BCH_near", c)] - m[(i, "BCH_rand", c)] for i in imgs], boot_seed)
    return out


# ------------------------------------------------------------------ main
def main():
    t0 = time.perf_counter()
    proto, lock, test_files, th = verify()
    seed = proto["execution"]["split_seeds"]["test"]
    boot_seed = proto["decisions"]["bootstrap"]["seed"]
    assert set(proto["transforms"]["list"]) == set(v1.CONDITIONS)
    OUT.mkdir(parents=True, exist_ok=True)
    resize = transforms.Resize((base.SIZE, base.SIZE))
    imgs = torch.stack([default_transform(resize(Image.open(f).convert("RGB"))) for f in progress(test_files, desc="load test images", unit="img")])
    masks = base.checkerboard_masks(torch.device("cuda:0"))
    books = v1.codebooks()
    books_gpu = {k: torch.as_tensor(b, device="cuda:0").float() * 2 - 1 for k, b in books.items()}
    rng = np.random.default_rng([seed, 1])
    n = len(test_files)
    sets = {"RAW": (None, torch.as_tensor(rng.integers(0, 2, (n, 5, 32)), dtype=torch.float32))}
    for kind in SETS[1:]:
        ids = torch.as_tensor(rng.integers(0, len(books[kind]), (n, 5)))
        sets[kind] = (ids, torch.as_tensor(books[kind])[ids].float())
    summary = {"protocol_sha256": lock["protocol_sha256"], "v3_thresholds_sha256": v3.sha(THRESH), "script_sha256": v3.sha(Path(__file__)),
               "test_images": n, "seed": seed}
    log(f"loaded {n} test images ({time.perf_counter() - t0:.0f}s)")

    with gzip.open(OUT / "v4_h1_records_wam_coco.jsonl.gz", "wt", encoding="utf-8") as fh:
        per_img, per_k = run_h1("wam_coco", imgs, books_gpu, sets, masks, seed, fh)
    summary["H1_H4_wam_coco"] = h1_summary(per_img, per_k, boot_seed)
    del per_img, per_k
    log(f"H1/H4 wam_coco measured ({time.perf_counter() - t0:.0f}s)")

    wam = load_model_from_checkpoint(str(base.REPO / "checkpoints" / "params.json"), str(base.REPO / "checkpoints" / "wam_coco.pth")).to("cuda:0").eval()
    with gzip.open(OUT / "v4_h2_records.jsonl.gz", "wt", encoding="utf-8") as fh:
        h2_rec = run_h2(wam, test_files, imgs, proto, th, masks, seed, fh)
    summary["H2"] = h2_summary(h2_rec, th, boot_seed)
    with gzip.open(OUT / "v4_h3_records.jsonl.gz", "wt", encoding="utf-8") as fh:
        h3_rec = run_h3(wam, imgs, masks, seed, fh)
    summary["H3"] = h3_summary(h3_rec, boot_seed)
    del wam
    torch.cuda.empty_cache()

    hc = summary["H1_H4_wam_coco"]["conditions"]["hflip_contrast1.5"]
    h2 = summary["H2"]
    verdict = {
        "H1": {"BCH16s_minus_WAM": hc["BCH16s_minus_WAM"], "SUPPORTED": hc["BCH16s_minus_WAM"]["ci"][0] > 0,
               "secondary_BCH16s_minus_RND16s": hc["BCH16s_minus_RND16s"],
               "secondary_equivalence_0.02": -0.02 < hc["BCH16s_minus_RND16s"]["ci"][0] and hc["BCH16s_minus_RND16s"]["ci"][1] < 0.02},
        "H2": {"FA_upper_C": h2["methods"]["C"]["FA_upper_98.75"], "FA_upper_A": h2["methods"]["A"]["FA_upper_98.75"],
               "R_C_minus_A": h2["R_C_minus_A"],
               "SUPPORTED": h2["methods"]["C"]["FA_upper_98.75"] <= 0.01 and h2["methods"]["A"]["FA_upper_98.75"] <= 0.01 and h2["R_C_minus_A"]["ci"][0] > 0},
        "H3": {"part1": summary["H3"]["part1_RAWd1_minus_RAWrand"], "part2": summary["H3"]["part2_RAWdle4_minus_BCHnear"],
               "SUPPORTED": summary["H3"]["part1_RAWd1_minus_RAWrand"]["ci"][0] > 0 and summary["H3"]["part2_RAWdle4_minus_BCHnear"]["ci"][0] > 0},
        "H4": {"wam_metric_minus_exact": hc["H4_wam_metric_minus_exact"], "SUPPORTED": hc["H4_wam_metric_minus_exact"]["ci"][0] > 0}}
    summary["verdict"] = verdict
    summary["seconds_decisions"] = time.perf_counter() - t0
    (OUT / "v4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("DECISIONS (test, wam_coco, 98.75%):")
    log(f"  H1 BCH16s - WAM (hflip_contrast1.5): {hc['BCH16s_minus_WAM']['diff']:+.3f} [{hc['BCH16s_minus_WAM']['ci'][0]:+.3f}, {hc['BCH16s_minus_WAM']['ci'][1]:+.3f}] "
        f"-> {'SUPPORTED' if verdict['H1']['SUPPORTED'] else 'NOT_SUPPORTED'}; secondary BCH16s - RND16s {hc['BCH16s_minus_RND16s']['diff']:+.3f} "
        f"[{hc['BCH16s_minus_RND16s']['ci'][0]:+.3f}, {hc['BCH16s_minus_RND16s']['ci'][1]:+.3f}]")
    for name in ("A", "B", "C"):
        mm = h2["methods"][name]
        log(f"  H2 {name}: FA {mm['false_attributions']}/{h2['scenes']} = {mm['FA_rate']:.4f}, upper {mm['FA_upper_98.75']:.4f}; R {mm['R']:.3f}; by group "
            + ", ".join(f"{g} {v['rate']:.4f} (n {v['n']})" for g, v in mm["FA_by_group"].items()))
    log(f"  H2 R(C) - R(A): {h2['R_C_minus_A']['diff']:+.3f} [{h2['R_C_minus_A']['ci'][0]:+.3f}, {h2['R_C_minus_A']['ci'][1]:+.3f}] -> "
        f"{'SUPPORTED' if verdict['H2']['SUPPORTED'] else 'NOT_SUPPORTED'}")
    for part in ("part1", "part2"):
        c = verdict["H3"][part]
        log(f"  H3 {part}: {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]")
    log(f"  H3 -> {'SUPPORTED' if verdict['H3']['SUPPORTED'] else 'NOT_SUPPORTED'}")
    c = verdict["H4"]["wam_metric_minus_exact"]
    log(f"  H4 WAM bit accuracy (found units) - exact recovery: {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] -> "
        f"{'SUPPORTED' if verdict['H4']['SUPPORTED'] else 'NOT_SUPPORTED'}")
    log(f"decisions written ({summary['seconds_decisions']:.0f}s); starting the wam_mit secondary run")

    with gzip.open(OUT / "v4_h1_records_wam_mit.jsonl.gz", "wt", encoding="utf-8") as fh:
        per_img, per_k = run_h1("wam_mit", imgs, books_gpu, sets, masks, seed, fh)
    summary["H1_H4_wam_mit_secondary"] = h1_summary(per_img, per_k, boot_seed)
    summary["seconds_total"] = time.perf_counter() - t0
    (OUT / "v4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    hm = summary["H1_H4_wam_mit_secondary"]["conditions"]["hflip_contrast1.5"]["BCH16s_minus_WAM"]
    log(f"wam_mit secondary: BCH16s - WAM (hflip_contrast1.5) {hm['diff']:+.3f} [{hm['ci'][0]:+.3f}, {hm['ci'][1]:+.3f}]")
    log(f"done in {summary['seconds_total']:.0f}s")


if __name__ == "__main__":
    main()
