"""E3b (configs/protocol_v5_addendum_E3b.json, locked): E3 repeated on the 3,000 test images.

Only PAD16 is measured anew, with the same transform seeds as V4 H1 (stable_seed(20261102, 'v5-test-h1', k, s, cond)),
so it shares image, k, areas and transform realisation with the V4 RAW / BCH16 / REP16 records, which are reused."""
import gzip
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extra_e3e4 as e34  # noqa: E402
from extra_e3e4 import v4, v1, v3, base, e12  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402
from notebooks.inference_utils import load_model_from_checkpoint  # noqa: E402
from watermark_anything.data.transforms import default_transform  # noqa: E402
from wamset.config import progress, stable_seed  # noqa: E402

STUDY = HERE.parents[1]
PLAN = STUDY / "configs" / "protocol_v5_addendum_E3b.json"
PLAN_LOCK = STUDY / "configs" / "protocol_v5_addendum_E3b_lock.json"
V4REC = Path("/workspace/wam/artifacts/v4/v4_h1_records_wam_coco.jsonl.gz")
OUT = Path("/workspace/wam/artifacts/extra")


def log(msg):
    print(f"[E3b] {msg}", flush=True)


def verify():
    pl = json.loads(PLAN_LOCK.read_text(encoding="utf-8"))
    lock = json.loads(e12.BASE_LOCK.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    bad = []
    if v3.sha(PLAN) != pl["addendum_sha256"]:
        bad.append("E3b plan")
    if v3.sha(STUDY / "configs" / "protocol_v5.json") != lock["protocol_sha256"]:
        bad.append("base protocol")
    if v3.sha(e12.ADD) != plan["amends"]["sha256"]:
        bad.append("addendum E")
    if v3.sha(base.REPO / "checkpoints" / "wam_coco.pth") != lock["checkpoints_sha256"]["wam_coco"]:
        bad.append("wam_coco")
    files = sorted(base.COCO.glob("*.jpg"))[1500:4500]
    blob = "".join(f"{f.name}:{v3.sha(f)}\n" for f in files).encode()
    if hashlib.sha256(blob).hexdigest() != lock["image_lists_sha256"]["test"]:
        bad.append("test image list")
    if bad:
        raise SystemExit(f"[E3b] STOP: hash mismatch {bad}")
    log(f"verified E3b plan {pl['addendum_sha256'][:16]}..., base protocol, addendum E, wam_coco, test images {len(files)}")
    return pl, plan, files


def main():
    t0 = time.perf_counter()
    pl, plan, files = verify()
    OUT.mkdir(parents=True, exist_ok=True)
    key = np.random.default_rng(20261104).integers(0, 2, 16).astype(np.uint8)
    assert "".join(map(str, key)) == plan["new_measurement"]["message_set"].split("key ")[-1].rstrip(")")
    book = e34.pad_book(key)
    book_gpu = torch.as_tensor(book, device="cuda:0").float() * 2 - 1
    n = len(files)
    ids = torch.as_tensor(np.random.default_rng(plan["new_measurement"]["id_seed"]).integers(0, 1 << 16, (n, 5)))
    msgs_all = torch.as_tensor(book)[ids].float()
    resize = transforms.Resize((base.SIZE, base.SIZE))
    imgs = torch.stack([default_transform(resize(Image.open(f).convert("RGB"))) for f in progress(files, desc="load test images", unit="img")])
    masks = base.checkerboard_masks(torch.device("cuda:0"))
    wam = load_model_from_checkpoint(str(base.REPO / "checkpoints" / "params.json"), str(base.REPO / "checkpoints" / "wam_coco.pth")).to("cuda:0").eval()
    seed = 20261102
    per_img = defaultdict(lambda: defaultdict(v4.new_acc))
    log(f"loaded {n} test images ({time.perf_counter() - t0:.0f}s); measuring PAD16")
    with gzip.open(OUT / "extra_E3b_records.jsonl.gz", "wt", encoding="utf-8") as fh:
        for s in progress(range(0, n, v1.BATCH), desc="E3b PAD16 on test (batch)", unit="batch"):
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
                            res = v1.evaluate(preds[b], gt[b], "PAD16", [int(v) for v in ids[s + b, :k]], msgs[b, :k], book_gpu)
                            fh.write(json.dumps({"set": "PAD16", "condition": cname, "image": s + b, "k": k, **res}) + "\n")
                            v4.add_h1(per_img[("PAD16", cname)][s + b], res, "PAD16")
    log(f"PAD16 measured ({time.perf_counter() - t0:.0f}s); reading V4 records")
    with gzip.open(V4REC, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["set"] in ("RAW", "BCH16", "REP16"):
                v4.add_h1(per_img[(r["set"], r["condition"])][r["image"]], r, r["set"])
    summ = e34.summarize_e3(per_img)
    v4sum = json.loads((Path("/workspace/wam/artifacts/v4/v4_summary.json")).read_text(encoding="utf-8"))["H1_H4_wam_coco"]["conditions"]
    check = all(abs(summ["conditions"][c]["BCH16_soft"] - v4sum[c]["BCH16"]["soft"]) < 1e-12 and abs(summ["conditions"][c]["RAW_exact"] - v4sum[c]["RAW"]["exact_recovery"]) < 1e-12
                for c in v1.CONDITIONS)
    if not check:
        raise SystemExit("[E3b] STOP: reused V4 records do not reproduce v4_summary.json")
    reserve = json.loads((OUT / "extra_E3_summary.json").read_text(encoding="utf-8"))["E3"]["conditions"]
    out = {"addendum_E3b_sha256": pl["addendum_sha256"], "script_sha256": v3.sha(Path(__file__)), "v4_records_sha256": v3.sha(V4REC),
           "pad_key": key.tolist(), "images": n, "check_v4_reproduced": True, "E3b": summ,
           "reserve_E3_primary": reserve["hflip_contrast1.5"]["BCH16s_minus_PAD16s"], "seconds": time.perf_counter() - t0}
    (OUT / "extra_E3b_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{'condition':26s} {'WAM':>6s} {'BCH16s':>6s} {'BCH16h':>6s} {'REP16s':>6s} {'PAD16s':>6s} {'PAD16h':>6s} | {'BCH16s-PAD16s (test)':>26s} holm p | {'PAD16s-WAM':>10s} | reserve", flush=True)
    for c, r in summ["conditions"].items():
        x = r["BCH16s_minus_PAD16s"]
        print(f"{c:26s} {r['RAW_exact']:6.3f} {r['BCH16_soft']:6.3f} {r['BCH16_hard']:6.3f} {r['REP16_soft']:6.3f} {r['PAD16_soft']:6.3f} {r['PAD16_hard']:6.3f} | "
              f"{x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] {summ['holm_BCH16s_minus_PAD16s'][c]:.4f} | {r['PAD16s_minus_RAW']['diff']:+.3f} | "
              f"{reserve[c]['BCH16s_minus_PAD16s']['diff']:+.3f}", flush=True)
    p = summ["conditions"]["hflip_contrast1.5"]["BCH16s_minus_PAD16s"]
    log(f"PRIMARY (test, hflip_contrast1.5) BCH16s - PAD16s {p['diff']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}] -> lower bound {'> 0' if p['ci'][0] > 0 else '<= 0'}")
    log(f"done in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
