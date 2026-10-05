"""V3: H2 calibration on the 1,000 cal images (configs/protocol_v5.json, locked in V2). Nothing else is measured.

Before running, every hash recorded in configs/protocol_v5_lock.json is checked (protocol file, code files, WAM
checkpoint, cal image list); any mismatch stops the run. Only cal images (val2017 indices 500-1499) are opened.

Per protocol: operational registry from seed 20261100 (V1b procedure: 1,000 BCH16 IDs, 1,000 distinct random RAW
messages); one H2 scene per cal image from the cal seed 20261101 (k uniform 0..5, transform uniform over the 26
transforms, each message registered 0.5 / unregistered 0.5 with half random and half near). Images are processed in
index order in batches of 25; within a batch, images sharing (k, transform) are transformed together after
torch.manual_seed(stable_seed(20261101, "v5-cal", k, batch_start, transform)). Methods A (WAM-t, -Hamming),
B (RAW registry soft, best - second) and C (BCH16 full-codebook soft, max correlation) are scored on wam_coco.
Threshold per method: accept a score s iff (1 + #{cal null >= s}) / 1001 <= 0.002, i.e. s > second-largest cal null."""
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

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
sys.path.insert(0, str(HERE))
import v1_h1 as v1  # noqa: E402
import v1b_h2h3 as v1b  # noqa: E402
from v1_h1 import base  # noqa: E402
from h1_pretest import bch_codebook  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402
from notebooks.inference_utils import load_model_from_checkpoint  # noqa: E402
from watermark_anything.data.transforms import default_transform  # noqa: E402
from wamset.config import progress, stable_seed  # noqa: E402

PROTOCOL = STUDY / "configs" / "protocol_v5.json"
LOCK = STUDY / "configs" / "protocol_v5_lock.json"
OUT = Path("/workspace/wam/artifacts/v3")
METHODS = {"A": "A", "B": "B_margin", "C": "C_max"}


def log(msg):
    print(f"[V3] {msg}", flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    problems = []
    if sha(PROTOCOL) != lock["protocol_sha256"]:
        problems.append("protocol sha256")
    for rel, digest in proto["code"].items():
        if sha(STUDY / rel) != digest:
            problems.append(f"code {rel}")
    files = sorted(base.COCO.glob("*.jpg"))
    a, b = proto["images"]["splits"]["cal"]["index"]
    cal = files[a:b + 1]
    blob = "".join(f"{f.name}:{sha(f)}\n" for f in cal).encode()
    if hashlib.sha256(blob).hexdigest() != lock["image_lists_sha256"]["cal"]:
        problems.append("cal image list")
    if problems:
        raise SystemExit(f"[V3] STOP: hash mismatch: {problems}")
    log(f"verified: protocol {lock['protocol_sha256'][:16]}..., {len(proto['code'])} code files, cal images {len(cal)}")
    return proto, lock, cal


def build_registry(seed, bch):
    rng = np.random.default_rng(seed)
    reg_ids = rng.choice(len(bch), proto_reg_n, replace=False)
    raw_reg, seen = [], set()
    while len(raw_reg) < proto_reg_n:
        x = int(rng.integers(0, 1 << 32, dtype=np.uint64))
        if x not in seen:
            seen.add(x)
            raw_reg.append(x)
    return reg_ids, raw_reg


def build_scenes(n, seed, bch, reg_ids, raw_reg, conditions):
    """One scene per image: k, transform, five message slots (same slot procedure as V1b)."""
    rng = np.random.default_rng(seed)
    reg_mask = np.zeros(len(bch), bool)
    reg_mask[reg_ids] = True
    raw_index = {x: i for i, x in enumerate(raw_reg)}
    scenes = []
    for _ in range(n):
        k = int(rng.integers(6))
        cond = conditions[int(rng.integers(len(conditions)))]
        slots, used_raw, used_bch = [], set(), set()
        for _ in range(5):
            registered = bool(rng.random() < .5)
            utype = "reg" if registered else ("near" if rng.random() < .5 else "rand")
            while True:
                if registered:
                    r = int(rng.integers(len(raw_reg)))
                    raw, raw_id, bch_id = raw_reg[r], r, int(reg_ids[int(rng.integers(len(reg_ids)))])
                elif utype == "rand":
                    raw, raw_id = int(rng.integers(0, 1 << 32, dtype=np.uint64)), -1
                    bch_id = int(rng.integers(len(bch)))
                    if raw in raw_index or reg_mask[bch_id]:
                        continue
                else:
                    x = raw_reg[int(rng.integers(len(raw_reg)))]
                    for p in rng.choice(32, 2, replace=False):
                        x ^= 1 << int(p)
                    raw, raw_id = x, -1
                    base_id = int(reg_ids[int(rng.integers(len(reg_ids)))])
                    cand = np.nonzero(((bch != bch[base_id]).sum(1) == 8) & ~reg_mask)[0]
                    bch_id = int(rng.choice(cand))
                    if raw in raw_index:
                        continue
                if raw not in used_raw and bch_id not in used_bch:
                    break
            used_raw.add(raw)
            used_bch.add(bch_id)
            slots.append({"registered": registered, "utype": utype, "raw": raw, "raw_id": raw_id, "bch_id": bch_id})
        scenes.append({"k": k, "cond": cond, "slots": slots})
    return scenes, reg_mask


def threshold(nulls, n_cal, alpha):
    """Largest tau such that accepting s > tau satisfies (1 + #{null >= s}) / (n + 1) <= alpha for every s > tau."""
    allowed = math.floor(alpha * (n_cal + 1) - 1 + 1e-9)  # max #{null >= s}
    finite = sorted((x for x in nulls if math.isfinite(x)), reverse=True)
    return finite[allowed] if allowed < len(finite) else float("-inf"), allowed


def main():
    global proto_reg_n
    t0 = time.perf_counter()
    proto, lock, cal_files = verify()
    proto_reg_n = proto["h2"]["registry_size"]
    seeds = proto["execution"]["split_seeds"]
    conditions = list(proto["transforms"]["list"])
    assert set(conditions) == set(v1.CONDITIONS)
    OUT.mkdir(parents=True, exist_ok=True)
    path, digest = v1.checkpoint("wam_coco")
    if digest != lock["checkpoints_sha256"]["wam_coco"]:
        raise SystemExit("[V3] STOP: wam_coco checkpoint hash mismatch")
    wam = load_model_from_checkpoint(str(base.REPO / "checkpoints" / "params.json"), str(path)).to("cuda:0").eval()
    _, _, bch = bch_codebook(3)
    reg_ids, raw_reg = build_registry(proto["execution"]["operational_registry_seed"], bch)
    scenes, reg_mask = build_scenes(len(cal_files), seeds["cal"], bch, reg_ids, raw_reg, conditions)
    raw_bits = np.stack([v1b.int_bits(x) for x in raw_reg])
    G = {"raw_bits": torch.as_tensor(raw_bits, device="cuda:0").float(), "raw_pm": torch.as_tensor(raw_bits, device="cuda:0").float() * 2 - 1,
         "bch_pm": torch.as_tensor(bch, device="cuda:0").float() * 2 - 1, "reg_mask": reg_mask}
    log(f"operational registry built; scenes: k {np.bincount([s['k'] for s in scenes], minlength=6).tolist()}, "
        f"signal-lost transforms {sum(s['cond'] in v1b.SIGNAL_LOST for s in scenes)} ({time.perf_counter() - t0:.0f}s)")
    resize = transforms.Resize((base.SIZE, base.SIZE))
    masks = base.checkerboard_masks(torch.device("cuda:0"))
    a = proto["images"]["splits"]["cal"]["index"][0]
    records = {}
    for s in progress(range(0, len(cal_files), v1.BATCH), desc="V3 cal (batch of 25 images)", unit="batch"):
        idx = list(range(s, min(s + v1.BATCH, len(cal_files))))
        x = torch.stack([default_transform(resize(Image.open(cal_files[i]).convert("RGB"))) for i in idx]).cuda()
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
                seed = stable_seed(seeds["cal"], "v5-cal", k, s, cond)
                fams = ("RAW", "BCH") if k else ("NONE",)
                for fam in fams:
                    src = x[members] if fam == "NONE" else composite[fam][members]
                    gt0 = (masks[:k] if k else torch.zeros((1, base.SIZE, base.SIZE), device="cuda:0"))[None].expand(len(members), -1, -1, -1).clone()
                    img, gt = v1.apply(src.clone(), gt0, v1.CONDITIONS[cond], seed)
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
                        rec = records.setdefault(a + i, {"image": a + i, "file": cal_files[i].name, "k": k, "cond": cond, "m": {}})
                        rec["m"].update(res)
    records = [records[key] for key in sorted(records)]
    n = len(records)
    assert n == len(cal_files) and all(set(METHODS.values()) <= set(r["m"]) for r in records)
    with gzip.open(OUT / "v3_cal_records.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    alpha = 0.002
    out = {"protocol_sha256": lock["protocol_sha256"], "checkpoint_sha256": digest, "script_sha256": sha(Path(__file__)),
           "n_cal": n, "alpha_cal": alpha, "rule": proto["h2"]["calibration_rule"], "methods": {},
           "scene_counts": {"k": np.bincount([r["k"] for r in records], minlength=6).tolist(),
                            "signal_lost": sum(r["cond"] in v1b.SIGNAL_LOST for r in records)},
           "registry": {"BCH16_ids": [int(v) for v in reg_ids], "RAW_ints": [int(v) for v in raw_reg]}}
    for name, v in METHODS.items():
        nulls = [max(r["m"][v]["null"].values(), default=float("-inf")) for r in records]
        tau, allowed = threshold(nulls, n, alpha)
        succ = np.array([x for r in records for x in r["m"][v]["succ"]])
        entry = {"variant": v, "tau": tau, "accept": "score > tau", "max_cal_nulls_at_or_above_accepted_score": allowed,
                 "finite_cal_nulls": int(sum(math.isfinite(x) for x in nulls)),
                 "cal_false_attribution_at_tau": float(np.mean(np.array(nulls) > tau)),
                 "cal_R_at_tau (calibration data, not a result)": float(np.mean(succ > tau)) if len(succ) else None,
                 "cal_registered_messages": int(len(succ))}
        if name == "A":
            entry["t_equivalent"] = None if not math.isfinite(tau) else int(math.ceil(-tau) - 1)
        out["methods"][name] = entry
    out["seconds"] = time.perf_counter() - t0
    (OUT / "v3_thresholds.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("thresholds (accept score > tau):")
    for name, e in out["methods"].items():
        extra = f", t = {e['t_equivalent']}" if name == "A" else ""
        log(f"  {name} ({e['variant']}): tau {e['tau']}{extra}; finite cal nulls {e['finite_cal_nulls']}; cal FA at tau {e['cal_false_attribution_at_tau']:.4f}; "
            f"cal R {e['cal_R_at_tau (calibration data, not a result)']:.3f} over {e['cal_registered_messages']} registered messages")
    log(f"scene counts {out['scene_counts']}; done in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
