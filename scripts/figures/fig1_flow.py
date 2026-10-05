"""Figure 1 (illustration only, no statistic): the processing flow on one reserve COCO image (4,500-4,999), wam_coco.

Stages: input -> 16-bit IDs to BCH(32,16) codewords -> WAM embedding per 10% square (WAM Sec. 5.5 layout, k = 5) -> residual
-> distortion (horizontal flip + contrast 1.5) -> WAM detection -> DBSCAN regions (eps 1, min_samples 1000) -> region mean bit
logits -> soft maximum-likelihood decoding over the 65,536 codewords -> registry check and calibrated threshold (tau_C, V3)
-> attribution. The raw WAM readout (DBSCAN centroid as the 32-bit message) is shown for the same image.

Selection rule: scan the reserve images in name order and take the first image where
(0) the COCO image license allows reuse in a CC BY article (license id 4 CC BY 2.0, 7 no known copyright restrictions,
    8 United States Government Work; annotations/captions_val2017.json),
(0b) none of the five COCO captions mentions a person (word list PERSON_WORDS; a proxy, checked visually), because
    Scientific Reports requires consent to publish images that could identify a person, and
(1) every one of the 5 BCH16 regions is decoded to its own registered ID with score > tau_C and no other registered ID is accepted,
(2) every raw message has a region, and (3) at least one raw centroid has 1-3 bit errors (so the raw readout fails).
Condition (0) was added after the first run picked a CC BY-NC-SA image (000000522007.jpg) and condition (0b) after the second
run picked an image showing people (000000523957.jpg); conditions (1)-(3) are unchanged.
IDs: the F16 targets of the image (5 registered IDs, seed 20261108). Raw messages: seed 20261111.
Outputs (study root): figures/fig1_flow.png/.pdf, figures/fig1/*.png (panels), figures/fig1_data.json, figures/fig1_stages.md."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
sys.path.insert(0, str(STUDY / "scripts" / "verify"))
import v1_h1 as v1  # noqa: E402  (changes cwd to the WAM repository)
import v1b_h2h3 as v1b  # noqa: E402
import v3_cal as v3  # noqa: E402
from v1_h1 import base  # noqa: E402
from h1_pretest import bch_codebook  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402
from notebooks.inference_utils import load_model_from_checkpoint  # noqa: E402
from watermark_anything.data.transforms import default_transform, unnormalize_img  # noqa: E402
from wamset.config import stable_seed  # noqa: E402
from wamset.dbscan_gpu import dbscan_bits  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import ConnectionPatch, Rectangle  # noqa: E402

OUTD = STUDY / "figures"
PAN = OUTD / "fig1"
COND = "hflip_contrast1.5"
TARGET_SEED, RAW_SEED, TRANSFORM_SEED = 20261108, 20261111, 20261112
OPEN_LICENSES = {4, 7, 8}
PERSON_WORDS = {"person", "persons", "people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "child", "children",
                "kid", "kids", "baby", "player", "players", "guy", "guys", "lady", "ladies", "someone", "he", "she", "his", "her", "they",
                "crowd", "skier", "surfer", "skateboarder", "rider", "biker", "cyclist", "adult", "adults", "teenager", "couple", "family",
                "chef", "cook", "worker", "workers", "officer", "police", "soldier", "student", "students", "tourist", "tourists", "pedestrian",
                "pedestrians", "spectators", "fans", "team", "batter", "catcher", "pitcher", "umpire", "tennis", "snowboarder", "hand", "hands"}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    bl = json.loads((STUDY / "configs" / "protocol_v5_lock.json").read_text(encoding="utf-8"))
    files = sorted(base.COCO.glob("*.jpg"))[4500:5000]
    bad = []
    if hashlib.sha256("".join(f"{f.name}:{v3.sha(f)}\n" for f in files).encode()).hexdigest() != bl["image_lists_sha256"]["reserve"]:
        bad.append("reserve image list")
    if v3.sha(base.REPO / "checkpoints" / "wam_coco.pth") != bl["checkpoints_sha256"]["wam_coco"]:
        bad.append("wam_coco")
    if bad:
        raise SystemExit(f"[FIG1] STOP: hash mismatch {bad}")
    return files, bl


def to_np(t):
    return unnormalize_img(t[None]).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()


def main():
    files, bl = verify()
    proto = json.loads((STUDY / "configs" / "protocol_v5.json").read_text(encoding="utf-8"))
    th = json.loads((STUDY / "reports" / "analysis" / "v3_thresholds.json").read_text(encoding="utf-8"))
    _, _, bch = bch_codebook(3)
    v3.proto_reg_n = proto["h2"]["registry_size"]
    reg_ids, _ = v3.build_registry(proto["execution"]["operational_registry_seed"], bch)
    assert [int(v) for v in reg_ids] == th["registry"]["BCH16_ids"]
    reg = np.zeros(1 << 16, bool)
    reg[reg_ids] = True
    tau = float(th["methods"]["C"]["tau"])
    n = len(files)
    rng = np.random.default_rng(TARGET_SEED)
    targets = np.stack([rng.choice(reg_ids, 5, replace=False) for _ in range(n)])
    raw = np.random.default_rng(RAW_SEED).integers(0, 2, (n, 5, 32)).astype(np.uint8)
    ann = json.loads((base.COCO.parent / "annotations" / "captions_val2017.json").read_text(encoding="utf-8"))
    lic = {im["file_name"]: im["license"] for im in ann["images"]}
    lic_name = {l["id"]: (l["name"], l["url"]) for l in ann["licenses"]}
    flickr = {im["file_name"]: im.get("flickr_url") for im in ann["images"]}
    caps = {}
    id2f = {im["id"]: im["file_name"] for im in ann["images"]}
    for c in ann["annotations"]:
        caps.setdefault(id2f[c["image_id"]], []).append(c["caption"])
    import re as _re
    has_person = {f: any(w in PERSON_WORDS for c in cs for w in _re.findall(r"[a-z]+", c.lower())) for f, cs in caps.items()}
    resize = transforms.Resize((base.SIZE, base.SIZE))
    masks = base.checkerboard_masks(torch.device("cuda:0"))
    wam = load_model_from_checkpoint(str(base.REPO / "checkpoints" / "params.json"), str(base.REPO / "checkpoints" / "wam_coco.pth")).to("cuda:0").eval()
    bch_pm = torch.as_tensor(bch, device="cuda:0").float() * 2 - 1

    def run(x, bits, s):
        msgs = torch.as_tensor(bits, device="cuda:0").float()
        with torch.no_grad():
            wm = [wam.embed(x, msgs[:, j])["imgs_w"] for j in range(5)]
            comp = x.clone()
            for j in range(5):
                mk = masks[j][None, None]
                comp = wm[j] * mk + comp * (1 - mk)
            img, gt = v1.apply(comp.clone(), masks[None].expand(len(x), -1, -1, -1).clone(), v1.CONDITIONS[COND],
                               stable_seed(TRANSFORM_SEED, "fig1", s, COND))
            preds = wam.detect(img)["preds"].float()
        return comp, img, gt, preds

    chosen = None
    for s in range(0, n, v1.BATCH):
        B = min(v1.BATCH, n - s)
        x = torch.stack([default_transform(resize(Image.open(f).convert("RGB"))) for f in files[s:s + B]]).cuda()
        cb, ib, gb, pb = run(x, bch[targets[s:s + B]], s)
        _, _, gr, pr = run(x, raw[s:s + B], s)
        for b in range(B):
            i = s + b
            if lic[files[i].name] not in OPEN_LICENSES or has_person[files[i].name]:
                continue
            dec = []
            for j, _, _, l in v1b.units_of(pb[b], gb[b]):
                top = torch.topk(bch_pm @ l, 2)
                dec.append((j, int(top.indices[0]), float(top.values[0]), float(top.values[1]), l.cpu().numpy()))
            ok_all = all(any(j == jj and c == int(targets[i][jj]) and sc > tau for j, c, sc, _, _ in dec) for jj in range(5))
            no_false = not any(reg[c] and sc > tau and (j < 0 or c != int(targets[i][j])) for j, c, sc, _, _ in dec)
            err = {}
            for j, _, cen, _ in v1b.units_of(pr[b], gr[b]):
                if j >= 0:
                    err[j] = min(err.get(j, 99), int((cen.cpu().numpy().astype(np.uint8) != raw[i][j]).sum()))
            if ok_all and no_false and len(err) == 5 and any(1 <= e <= 3 for e in err.values()):
                chosen = dict(i=i, x=x[b], comp=cb[b], img=ib[b], gt=gb[b], preds=pb[b], dec=dec, err=err)
                break
        if chosen:
            break
    if chosen is None:
        raise SystemExit("[FIG1] no reserve image satisfies the selection rule")
    i = chosen["i"]
    PAN.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- panel data
    x0, comp, img = to_np(chosen["x"]), to_np(chosen["comp"]), to_np(chosen["img"])
    rabs = np.abs(comp - x0)
    rscale = float(np.quantile(rabs, 0.995))
    resid = np.clip(rabs / rscale, 0, 1)
    preds = chosen["preds"]
    prob = torch.sigmoid(preds[0].cpu()).numpy()
    _, labels = dbscan_bits(preds, 1000, 1.0)
    gt = chosen["gt"].cpu().numpy()
    lab_of = {}
    for lab in np.unique(labels):
        if lab >= 0:
            ov = [int((gt[j] & (labels == lab)).sum()) for j in range(5)]
            lab_of[int(lab)] = int(np.argmax(ov)) if max(ov) > 0 else -1
    best = {}
    for d in chosen["dec"]:
        if d[0] >= 0 and (d[0] not in best or d[2] > best[d[0]][2]):
            best[d[0]] = d
    dec = [best[j] for j in range(5)]
    L = np.stack([d[4] for d in dec])  # (5, 32) region mean logits, ordered by message slot
    ids = [int(t) for t in targets[i]]

    # ---------------------------------------------------------------- figure
    plt.rcParams.update({"font.size": 6.5, "font.family": "DejaVu Sans", "axes.titlesize": 6.8, "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(7.2, 4.0), dpi=300)
    gs = fig.add_gridspec(2, 5, left=0.065, right=0.968, top=0.86, bottom=0.08, wspace=0.70, hspace=0.62)
    pos = {"a": (0, 0), "b": (0, 1), "c": (0, 2), "d": (0, 3), "e": (0, 4), "f": (1, 4), "g": (1, 3), "h": (1, 2), "i": (1, 1), "j": (1, 0)}
    ax = {k: fig.add_subplot(gs[r, c]) for k, (r, c) in pos.items()}

    def outline(a, m, color, lw=0.8):
        ys, xs = np.nonzero(m)
        if len(ys):
            a.add_patch(Rectangle((xs.min() - .5, ys.min() - .5), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1, fill=False, ec=color, lw=lw))

    def ylabels(a, texts):
        a.set_yticks(range(5))
        a.set_yticklabels(texts, fontsize=5.3)
        for j, t in enumerate(a.get_yticklabels()):
            t.set_color(COLORS[j])

    a = ax["a"]; a.imshow(x0); a.set_title("(a) Input image\n(COCO, 256×256)")
    a = ax["b"]; a.imshow(bch[targets[i]], cmap="Greys", aspect="auto", interpolation="nearest")
    ylabels(a, [str(t) for t in ids]); a.set_xticks([0, 15, 31]); a.set_xticklabels(["1", "16", "32"], fontsize=5.3)
    a.set_title("(b) Encode: 16-bit ID\n→ BCH(32,16) codeword")
    a = ax["c"]; a.imshow(comp)
    for j in range(5):
        outline(a, masks[j].cpu().numpy() > .5, COLORS[j])
    a.set_title("(c) WAM embedding\n(5 regions, 10% each)")
    a = ax["d"]; a.imshow(resid); a.set_title("(d) Residual |c − a|\n(contrast-stretched)")
    a = ax["e"]; a.imshow(img); a.set_title("(e) Distortion: h-flip\n+ contrast 1.5")
    a = ax["f"]; a.imshow(prob, cmap="magma", vmin=0, vmax=1); a.set_title("(f) WAM detection:\np(watermark)")
    a = ax["g"]
    rgb = np.ones((*labels.shape, 3)) * 0.92
    for lab, j in lab_of.items():
        rgb[labels == lab] = matplotlib.colors.to_rgb(COLORS[j]) if j >= 0 else (0.4, 0.4, 0.4)
    a.imshow(rgb); a.set_title("(g) DBSCAN regions\n(ε = 1, min 1,000 px)")
    a = ax["h"]
    lim = float(np.abs(L).max())
    a.imshow(L, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
    ylabels(a, [f"R{j + 1}" for j in range(5)]); a.set_xticks([0, 15, 31]); a.set_xticklabels(["1", "16", "32"], fontsize=5.3)
    a.set_title("(h) Region mean bit\nlogits (soft input)")
    a = ax["i"]
    s1 = [d[2] for d in dec]
    s2 = [d[3] for d in dec]
    a.barh(range(5), s1, color=COLORS, height=0.6, label="best codeword")
    a.scatter(s2, range(5), marker="|", color="k", s=40, zorder=3, label="2nd best")
    a.axvline(tau, color="k", ls="--", lw=0.7)
    a.text(tau + max(s1) * 0.04, -0.55, f"τC = {tau:.1f}", ha="left", va="center", fontsize=5.0)
    ylabels(a, [f"R{j + 1}" for j in range(5)])
    a.set_xlim(0, max(s1) * 1.12); a.set_xlabel("correlation score c·l", fontsize=5.3); a.tick_params(axis="x", labelsize=5.3)
    a.set_ylim(4.6, -1.0)
    a.set_title("(i) Soft ML decoding\n(65,536 codewords)")
    a = ax["j"]; a.imshow(img)
    for j in range(5):
        outline(a, gt[j], COLORS[j])
        ys, xs = np.nonzero(gt[j])
        if len(ys):
            d = dec[j]
            okb = bool(reg[d[1]] and d[2] > tau and d[1] == ids[j])
            e = chosen["err"][j]
            a.text(xs.mean(), ys.mean(), f"{'✓' if okb else '✗'} ID {d[1]}\nraw: {e}-bit err",
                   ha="center", va="center", fontsize=4.3, color="white", bbox=dict(boxstyle="round,pad=0.12", fc=COLORS[j], ec="none", alpha=0.85))
    a.set_title("(j) Attribution (registered\n& score > τC) vs raw WAM")
    for k in "acdefgj":
        ax[k].set_xticks([])
        ax[k].set_yticks([])
    order = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    for p, q in zip(order[:-1], order[1:]):
        if p == "e":
            xyA, xyB = (0.5, -0.03), (0.5, 1.36)
        elif p in "abcd":
            xyA, xyB = (1.03, 0.5), (-0.47 if q == "b" else -0.04, 0.5)
        else:
            xyA, xyB = (-0.30 if p in "hi" else -0.04, 0.5), (1.03, 0.5)
        fig.add_artist(ConnectionPatch(xyA=xyA, coordsA=ax[p].transAxes, xyB=xyB, coordsB=ax[q].transAxes,
                                       arrowstyle="-|>", mutation_scale=7, lw=0.8, color="0.3"))
    fig.text(0.5, 0.975, "Embedding (a–e)  →  decoding and attribution (f–j);  WAM model and DBSCAN unchanged", ha="center", va="top", fontsize=7, weight="bold")
    fig.savefig(OUTD / "fig1_flow.png", dpi=300)
    fig.savefig(OUTD / "fig1_flow.pdf")
    plt.close(fig)
    for name, arr in (("a_input", x0), ("c_watermarked", comp), ("d_residual_stretched", resid), ("e_distorted", img)):
        plt.imsave(PAN / f"{name}.png", arr)
    plt.imsave(PAN / "f_detection_prob.png", prob, cmap="magma", vmin=0, vmax=1)
    plt.imsave(PAN / "g_dbscan_regions.png", rgb)

    # ---------------------------------------------------------------- data + stage table
    pix = {j: int(sum((labels == lab).sum() for lab, jj in lab_of.items() if jj == j)) for j in range(5)}
    regions = [{"slot": j, "region": f"R{j + 1}", "true_id": ids[j], "decoded_id": d[1], "score": d[2], "second_score": d[3],
                "registered": bool(reg[d[1]]), "accepted": bool(reg[d[1]] and d[2] > tau), "correct": d[1] == ids[j],
                "raw_centroid_bit_errors": int(chosen["err"][j]), "region_pixels": pix[j]} for j, d in enumerate(dec)]
    data = {"figure": "fig1_flow", "image_index": i, "image_file": files[i].name, "images_scanned": i + 1,
            "image_license": {"coco_license_id": lic[files[i].name], "name": lic_name[lic[files[i].name]][0], "url": lic_name[lic[files[i].name]][1],
                              "flickr_url": flickr[files[i].name]}, "captions": caps[files[i].name], "residual_display_scale_q995": rscale,
            "selection_rule": __doc__.split("Selection rule")[1].split("Outputs")[0].strip(),
            "condition": COND, "tau_C": tau, "detection_threshold": 0.5, "dbscan": {"eps": 1, "min_samples": 1000},
            "registry_N": int(len(reg_ids)), "codebook_size": 1 << 16, "seeds": {"targets": TARGET_SEED, "raw": RAW_SEED, "transform": TRANSFORM_SEED},
            "regions": regions, "detected_pixels": int((prob > .5).sum()), "dbscan_regions": int(len(lab_of)),
            "psnr_composite_db": float(10 * np.log10(1 / np.mean((comp - x0) ** 2))),
            "reserve_image_list_sha256": bl["image_lists_sha256"]["reserve"], "checkpoint_wam_coco_sha256": bl["checkpoints_sha256"]["wam_coco"],
            "script_sha256": sha(Path(__file__))}
    (OUTD / "fig1_data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    nreg = len(regions)
    acc = sum(r["accepted"] and r["correct"] for r in regions)
    rawok = sum(r["raw_centroid_bit_errors"] == 0 for r in regions)
    md = f"""# Figure 1 단계 설명표

`scripts/figures/fig1_flow.py`가 `figures/fig1_data.json`과 함께 생성한다. 예시 이미지는 예비 이미지 `{files[i].name}`이다(예비 목록 이름 순 {i + 1}번째, 선택 규칙을 처음 만족한 이미지). 왜곡은 좌우반전 + 대비 1.5, 체크포인트는 `wam_coco`. 이미지 라이선스: {lic_name[lic[files[i].name]][0]} ({lic_name[lic[files[i].name]][1]}), 원본 {flickr[files[i].name]}.

| 단계 | 패널 | 입력 | 처리 (매개변수) | 출력 | 이 예시의 값 |
|---|---|---|---|---|---|
| 1 입력 | (a) | COCO 이미지 | 256×256으로 크기 조정 (WAM §5.5) | 원본 x | `{files[i].name}` |
| 2 부호화 | (b) | 등록 ID 5개 (16비트) | 확장 BCH(32,16) 부호어로 변환 (최소 거리 8, 부호책 65,536개) | 32비트 부호어 5개 | ID {", ".join(str(t) for t in ids)} |
| 3 삽입 | (c) | x, 부호어 | 부호어마다 WAM 삽입기로 전체 삽입한 뒤 10% 정사각형 5개에 붙임 (WAM §5.5, k = 5) | 워터마크 이미지 | 원본 대비 PSNR {data['psnr_composite_db']:.2f} dB |
| 4 차이 | (d) | x, 워터마크 이미지 | 절대 차이를 상위 0.5% 값으로 나눠 표시(대비 확대) | 잔차 영상 | 표시 배율 기준값 {rscale:.4f} |
| 5 왜곡 | (e) | 워터마크 이미지 | 좌우반전 + 대비 1.5 (WAM 부록 D.4 모듈). 정답 영역도 같이 변환 | 왜곡 이미지 | – |
| 6 검출 | (f) | 왜곡 이미지 | WAM 추출기: 픽셀별 워터마크 확률과 32비트 logit | 확률 > 0.5인 픽셀 | {data['detected_pixels']:,} 픽셀 |
| 7 영역 | (g) | 검출 픽셀의 비트(logit > 0) | DBSCAN (Hamming ε = 1, min_samples 1,000), WAM과 같음 | 영역 | {data['dbscan_regions']}개 |
| 8 soft 입력 | (h) | 영역, 32비트 logit | 영역 안 픽셀 logit의 평균 l ∈ R^32 | 영역별 평균 logit | – |
| 9 복호 | (i) | l | 부호책 65,536개 전체와 상관 c·l(±1 표기)을 계산해 최댓값의 부호어 선택 (알고리즘 1) | 복호 ID, 점수 s | 점수 {', '.join(f'{r["score"]:.1f}' for r in regions)} |
| 10 귀속 | (j) | 복호 ID, s | 등록 목록(N = {len(reg_ids):,})에 있고 s > τ_C = {tau:.2f}이면 수락 (알고리즘 2. τ_C는 보정 1,000장에서 conformal 규칙으로 결정) | 수락 ID | 정확 수락 {acc}/{nreg}, 오귀속 0 |
| 비교 | (j) | 같은 이미지, 무작위 32비트 메시지 | WAM 원시 판독: DBSCAN 중심점(다수결)을 메시지로 읽음 | 32비트 문자열 | 정확 {rawok}/{nreg}, 중심점 비트 오류 {', '.join(str(r['raw_centroid_bit_errors']) for r in regions)} |

이 그림은 처리 과정을 보이는 예시이며 통계 결과가 아니다. 집계 결과는 시험 3,000장에서 얻었다.
"""
    (OUTD / "fig1_stages.md").write_text(md, encoding="utf-8")
    print(f"[FIG1] chosen reserve image {files[i].name} (#{i + 1}); BCH16 accepted {acc}/{nreg}; raw exact {rawok}/{nreg}; "
          f"raw errors {[r['raw_centroid_bit_errors'] for r in regions]}; scores {[round(r['score'], 1) for r in regions]}")


if __name__ == "__main__":
    main()
