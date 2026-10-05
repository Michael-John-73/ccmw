"""Review F16 (configs/protocol_v5_addendum_F16.json, locked): public (PUB) vs keyed (KEY) assignment of IDs to BCH(32,16)
codewords under a forging attacker who knows the public codebook and the target's ID but not the key. wam_coco, 500 reserve
images, WAM Sec. 5.5 checkerboard with k = 5, distortions none and hflip_contrast1.5, method C as locked (tau_C from V3)."""
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta as beta_dist

HERE = Path(__file__).resolve().parent
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

STUDY = HERE.parents[1]
PLAN = STUDY / "configs" / "protocol_v5_addendum_F16.json"
PLAN_LOCK = STUDY / "configs" / "protocol_v5_addendum_F16_lock.json"
BASE_LOCK = STUDY / "configs" / "protocol_v5_lock.json"
THRESH = STUDY / "reports" / "analysis" / "v3_thresholds.json"
OUT = Path("/workspace/wam/artifacts/review")
CONDS = ["none", "hflip_contrast1.5"]
DECISION_COND = "hflip_contrast1.5"
TARGET_SEED, KEY0_SEED, TRANSFORM_SEED, BOOT_SEED = 20261108, 20261109, 20261110, 20261006
KEY_SEEDS = range(20262000, 20263000)
LEVEL, MARGIN, NCODE = 0.9875, 0.02, 1 << 16


def log(msg):
    print(f"[F16] {msg}", flush=True)


def verify():
    pl, bl = json.loads(PLAN_LOCK.read_text(encoding="utf-8")), json.loads(BASE_LOCK.read_text(encoding="utf-8"))
    bad = []
    if v3.sha(PLAN) != pl["addendum_sha256"]:
        bad.append("F16 plan")
    if v3.sha(STUDY / "configs" / "protocol_v5.json") != bl["protocol_sha256"]:
        bad.append("base protocol")
    if v3.sha(base.REPO / "checkpoints" / "wam_coco.pth") != bl["checkpoints_sha256"]["wam_coco"]:
        bad.append("wam_coco")
    files = sorted(base.COCO.glob("*.jpg"))[4500:5000]
    if hashlib.sha256("".join(f"{f.name}:{v3.sha(f)}\n" for f in files).encode()).hexdigest() != bl["image_lists_sha256"]["reserve"]:
        bad.append("reserve image list")
    if bad:
        raise SystemExit(f"[F16] STOP: hash mismatch {bad}")
    log(f"verified F16 plan {pl['addendum_sha256'][:16]}..., base protocol, wam_coco, reserve images {len(files)}")
    return pl, files


def cp_upper(x, n):
    return float(beta_dist.ppf(LEVEL, x + 1, n - x)) if x < n else 1.0


def cp_lower(x, n):
    return float(beta_dist.ppf(1 - LEVEL, x, n - x + 1)) if x > 0 else 0.0


def boot(d):
    d = np.asarray(d, float)
    bt = d[np.random.default_rng(BOOT_SEED).integers(0, len(d), (10000, len(d)))].mean(1)
    a = (1 - LEVEL) / 2
    return {"diff": float(d.mean()), "ci": [float(np.quantile(bt, a)), float(np.quantile(bt, 1 - a))], "n": int(len(d))}


def main():
    t0 = time.perf_counter()
    pl, files = verify()
    proto = json.loads((STUDY / "configs" / "protocol_v5.json").read_text(encoding="utf-8"))
    th = json.loads(THRESH.read_text(encoding="utf-8"))
    _, _, bch = bch_codebook(3)
    v3.proto_reg_n = proto["h2"]["registry_size"]
    reg_ids, _ = v3.build_registry(proto["execution"]["operational_registry_seed"], bch)
    if [int(v) for v in reg_ids] != th["registry"]["BCH16_ids"]:
        raise SystemExit("[F16] STOP: operational registry differs from the V3 registry")
    tau = float(th["methods"]["C"]["tau"])
    N = len(reg_ids)
    reg = np.zeros(NCODE, bool)
    reg[reg_ids] = True
    n = len(files)
    rng = np.random.default_rng(TARGET_SEED)
    targets = np.stack([rng.choice(reg_ids, 5, replace=False) for _ in range(n)])
    perm0 = np.random.default_rng(KEY0_SEED).permutation(NCODE)  # pi_K0: ID -> codeword index
    inv0 = np.argsort(perm0)
    expect = {"E_pub": targets, "E_key": perm0[targets]}
    log(f"registry N = {N}, tau_C = {tau:.4f}, targets {targets.shape}, primary key seed {KEY0_SEED}")

    resize = transforms.Resize((base.SIZE, base.SIZE))
    imgs = torch.stack([default_transform(resize(Image.open(f).convert("RGB"))) for f in files])
    masks = base.checkerboard_masks(torch.device("cuda:0"))
    wam = load_model_from_checkpoint(str(base.REPO / "checkpoints" / "params.json"), str(base.REPO / "checkpoints" / "wam_coco.pth")).to("cuda:0").eval()
    bch_pm = torch.as_tensor(bch, device="cuda:0").float() * 2 - 1
    OUT.mkdir(parents=True, exist_ok=True)
    rec = []
    with gzip.open(OUT / "review_F16_records.jsonl.gz", "wt", encoding="utf-8") as fh:
        for s in progress(range(0, n, v1.BATCH), desc="F16 forgery (batch of 25 images)", unit="batch"):
            x = imgs[s:s + v1.BATCH].cuda()
            B = len(x)
            for es, cw in expect.items():
                msgs = torch.as_tensor(bch[cw[s:s + B]], device="cuda:0").float()
                with torch.no_grad():
                    wm = [wam.embed(x, msgs[:, j])["imgs_w"] for j in range(5)]
                    comp = x.clone()
                    for j in range(5):
                        mk = masks[j][None, None]
                        comp = wm[j] * mk + comp * (1 - mk)
                for cond in CONDS:
                    gt0 = masks[None].expand(B, -1, -1, -1).clone()
                    img, gt = v1.apply(comp.clone(), gt0, v1.CONDITIONS[cond], stable_seed(TRANSFORM_SEED, "f16", s, cond))
                    with torch.no_grad():
                        preds = wam.detect(img)["preds"].float()
                    for b in range(B):
                        units = v1b.units_of(preds[b], gt[b])
                        vis = gt[b].flatten(1).sum(1)
                        us = []
                        for j, _, _, l in units:
                            top = torch.max(bch_pm @ l, 0)
                            us.append([int(j), int(top.indices), float(top.values)])
                        r = {"image": 4500 + s + b, "set": es, "cond": cond, "targets": [int(t) for t in targets[s + b]],
                             "eligible": [bool(int(vis[j]) >= v1.MIN_VISIBLE) for j in range(5)], "units": us}
                        rec.append(r)
                        fh.write(json.dumps(r) + "\n")

    out = {"addendum_F16_sha256": pl["addendum_sha256"], "script_sha256": v3.sha(Path(__file__)), "images": n, "registry_N": N,
           "tau_C": tau, "keys_simulated": len(KEY_SEEDS), "conditions": {}}
    key_masks = []
    for ks in KEY_SEEDS:
        p = np.random.default_rng(ks).permutation(NCODE)
        m = np.zeros(NCODE, bool)
        m[p[reg_ids]] = True  # codewords whose (keyed) ID is registered
        key_masks.append(m)
    key_masks = np.stack(key_masks)
    for cond in CONDS:
        R = {es: {r["image"]: r for r in rec if r["set"] == es and r["cond"] == cond} for es in expect}
        ims = sorted(R["E_pub"])
        per_img = {k: [] for k in ("legit_pub", "legit_key")}
        cnt = dict(slots=0, legit_pub=0, legit_key=0, ftarget_key=0, funtarget_key0=0, funtarget_pub=0)
        slot_units = []  # per eligible E_pub slot: list of (c, s > tau)
        for i in ims:
            rp, rk = R["E_pub"][i], R["E_key"][i]
            lp, lk = [], []
            for j in range(5):
                if not (rp["eligible"][j] and rk["eligible"][j]):
                    continue
                T = rp["targets"][j]
                up = [(c, sc) for jj, c, sc in rp["units"] if jj == j]
                uk = [(c, sc) for jj, c, sc in rk["units"] if jj == j]
                a = any(c == T and sc > tau for c, sc in up)
                k = any(c == int(perm0[T]) and sc > tau for c, sc in uk)
                cnt["slots"] += 1
                cnt["legit_pub"] += a
                cnt["legit_key"] += k
                cnt["ftarget_key"] += any(int(inv0[c]) == T and sc > tau for c, sc in up)
                cnt["funtarget_key0"] += any(reg[int(inv0[c])] and sc > tau for c, sc in up)
                cnt["funtarget_pub"] += any(reg[c] and sc > tau for c, sc in up)
                slot_units.append([(c, sc > tau) for c, sc in up])
                lp.append(a)
                lk.append(k)
            if lp:
                per_img["legit_pub"].append(np.mean(lp))
                per_img["legit_key"].append(np.mean(lk))
        ns = cnt["slots"]
        hits = np.zeros((len(KEY_SEEDS), ns), bool)
        for t, us in enumerate(slot_units):
            for c, ok in us:
                if ok:
                    hits[:, t] |= key_masks[:, c]
        fu = hits.mean(1)
        d2 = boot(np.array(per_img["legit_key"]) - np.array(per_img["legit_pub"]))
        res = {
            "eligible_slots": ns,
            "R_legit_PUB": cnt["legit_pub"] / ns, "R_legit_KEY": cnt["legit_key"] / ns, "R_legit_KEY_minus_PUB": d2,
            "F_target_PUB": {"successes": cnt["legit_pub"], "rate": cnt["legit_pub"] / ns, "CP_lower_98.75": cp_lower(cnt["legit_pub"], ns)},
            "F_target_KEY": {"successes": cnt["ftarget_key"], "rate": cnt["ftarget_key"] / ns, "CP_upper_98.75": cp_upper(cnt["ftarget_key"], ns)},
            "F_untarget_PUB": {"successes": cnt["funtarget_pub"], "rate": cnt["funtarget_pub"] / ns},
            "F_untarget_KEY_K0": {"successes": cnt["funtarget_key0"], "rate": cnt["funtarget_key0"] / ns, "CP_upper_98.75": cp_upper(cnt["funtarget_key0"], ns)},
            "F_untarget_KEY_over_keys": {"median": float(np.median(fu)), "p05": float(np.quantile(fu, .05)), "p95": float(np.quantile(fu, .95)),
                                         "mean": float(fu.mean()), "share_keys_above_1pct": float(np.mean(fu > 0.01))},
            "F_untarget_expected": N / NCODE * cnt["legit_pub"] / ns,
        }
        res["D1_keyed_prevents_targeted"] = bool(res["F_target_KEY"]["CP_upper_98.75"] < res["F_target_PUB"]["CP_lower_98.75"])
        res["D2_legit_equivalent"] = bool(-MARGIN < d2["ci"][0] and d2["ci"][1] < MARGIN)
        out["conditions"][cond] = res
    out["decision_condition"] = DECISION_COND
    out["decision_supplementary"] = {k: out["conditions"][DECISION_COND][k] for k in ("D1_keyed_prevents_targeted", "D2_legit_equivalent")}
    out["seconds"] = time.perf_counter() - t0
    (OUT / "review_F16_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for cond, r in out["conditions"].items():
        log(f"{cond}: slots {r['eligible_slots']}, R_legit PUB {r['R_legit_PUB']:.4f} KEY {r['R_legit_KEY']:.4f}, "
            f"KEY-PUB {r['R_legit_KEY_minus_PUB']['diff']:+.4f} [{r['R_legit_KEY_minus_PUB']['ci'][0]:+.4f}, {r['R_legit_KEY_minus_PUB']['ci'][1]:+.4f}]")
        log(f"  targeted forgery PUB {r['F_target_PUB']['successes']} (lower {r['F_target_PUB']['CP_lower_98.75']:.4f}), "
            f"KEY {r['F_target_KEY']['successes']} (upper {r['F_target_KEY']['CP_upper_98.75']:.5f}); D1 {r['D1_keyed_prevents_targeted']}, D2 {r['D2_legit_equivalent']}")
        u = r["F_untarget_KEY_over_keys"]
        log(f"  untargeted framing KEY K0 {r['F_untarget_KEY_K0']['successes']} ({r['F_untarget_KEY_K0']['rate']:.4f}); over keys median {u['median']:.4f} "
            f"[{u['p05']:.4f}, {u['p95']:.4f}], keys > 1%: {u['share_keys_above_1pct']:.3f}; expected {r['F_untarget_expected']:.4f}")
    log(f"done in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
