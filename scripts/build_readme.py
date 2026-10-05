"""Build README.md from the result files of this repository.

Inputs: reports/analysis/*.json, configs/protocol_v5*.json, figures/fig1_data.json, figures/table1_hypotheses.md, figures/table2_forgery.md.
Usage:  python scripts/build_readme.py          # write README.md
        python scripts/build_readme.py --check  # exit 1 if README.md differs from what the result files give
"""
import json
import re
import sys
from pathlib import Path

from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
J = lambda p: json.loads((ROOT / p).read_text(encoding="utf-8"))  # noqa: E731
DC = "hflip_contrast1.5"
NAMES = {
    "none": "None", "hflip": "H-flip", "brightness1.5": "Brightness 1.5", "brightness2.0": "Brightness 2.0",
    "contrast1.5": "Contrast 1.5", "contrast2.0": "Contrast 2.0", "hue-0.1": "Hue −0.1", "hue0.1": "Hue +0.1",
    "saturation1.5": "Saturation 1.5", "saturation2.0": "Saturation 2.0", "blur3": "Gaussian blur 3", "blur17": "Gaussian blur 17",
    "median3": "Median filter 3", "median7": "Median filter 7", "jpeg80": "JPEG 80", "jpeg50": "JPEG 50", "crop0.33": "Crop 0.33",
    "crop0.5": "Crop 0.5", "resize0.5": "Resize 0.5", "rotate-10": "Rotate −10°", "rotate10": "Rotate +10°",
    "perspective0.1": "Perspective 0.1", "perspective0.5": "Perspective 0.5", "combination": "JPEG 80 + brightness 1.5 + crop 0.5",
    DC: "**H-flip + contrast 1.5 (decision)**", "hflip_contrast1.5_jpeg80": "H-flip + contrast 1.5 + JPEG 80",
}


def s(x, k=3):
    v = f"{x:+.{k}f}"
    return (f"{0:.{k}f}" if float(v) == 0 else v).replace("-", "−")


def ci(c, k=3):
    return f"{s(c['diff'], k)} [{s(c['ci'][0], k)}, {s(c['ci'][1], k)}]"


def f(x, k=3):
    return f"{x:.{k}f}"


def pct(x, k=2):
    return f"{x * 100:.{k}f}%"


def n_(x):
    return f"{x:,}"


def holm(p):
    return "< 0.003" if p < 0.003 else f"{p:.3f}"


def body(md):
    return "\n".join(l for l in md.splitlines() if not l.startswith("# ")).strip()


def values():
    v4, th, lock, proto = J("reports/analysis/v4_summary.json"), J("reports/analysis/v3_thresholds.json"), J("configs/protocol_v5_lock.json"), J("configs/protocol_v5.json")
    e1, e2 = J("reports/analysis/extra_E1_summary.json")["E1"], J("reports/analysis/extra_E2_summary.json")["E2"]
    e3b_all, e4 = J("reports/analysis/extra_E3b_summary.json"), J("reports/analysis/extra_E4_summary.json")["E4"]
    r3, r4, f16 = J("reports/analysis/review_R3_summary.json"), J("reports/analysis/review_R4_summary.json"), J("reports/analysis/review_F16_summary.json")
    rp, f1, r4plan = J("reports/analysis/algo_repro_summary.json"), J("figures/fig1_data.json"), J("configs/protocol_v5_addendum_R4.json")
    hu = J("reports/analysis/review_h2_units_summary.json")
    H, HM, HOLM = v4["H1_H4_wam_coco"]["conditions"], v4["H1_H4_wam_mit_secondary"]["conditions"], v4["H1_H4_wam_coco"]["holm_BCH16s_minus_WAM"]
    hc, e3c, h2 = H[DC], e3b_all["E3b"]["conditions"][DC], v4["H2"]
    mA, mB, mC = (h2["methods"][k] for k in "ABC")
    nn, stab, nt, dec = e2["without_near_miss"], r3["H2_threshold_stability"], r3["H3_nontrivial"], r3["H4_decomposition"]
    rt, vd, sf = v4["H3"]["rates"], v4["verdict"], dec[DC]["share_of_failures"]
    st = e4["stages_ms"][DC]
    a1, a2a, a2b, a2c = (rp["checks"][k] for k in ("A1_soft_decoding_reserve", "A2a_calibration", "A2b_test_acceptance", "A2c_region_acceptance_reserve"))
    assert all(vd[h]["SUPPORTED"] for h in ("H1", "H2", "H3", "H4"))
    V = {
        "lockhash": lock["protocol_sha256"], "locktime": lock["locked_at_utc"], "commit": lock["wam_commit"],
        "coco_sha": lock["checkpoints_sha256"]["wam_coco"][:16], "mit_sha": lock["checkpoints_sha256"]["wam_mit"][:16],
        "ntest": n_(v4["test_images"] if isinstance(v4["test_images"], int) else len(v4["test_images"])),
        "seedboot": str(proto["decisions"]["bootstrap"]["seed"]), "seedcal": str(proto["execution"]["split_seeds"]["cal"]),
        "seedtest": str(proto["execution"]["split_seeds"]["test"]), "seedreg": str(proto["execution"]["operational_registry_seed"]),
        "ncal": n_(th["n_cal"]), "alpha": str(th["alpha_cal"]), "N": n_(len(th["registry"]["BCH16_ids"])),
        "tauC": f(mC["tau"], 2), "tauB": f(mB["tau"], 2), "at": str(th["methods"]["A"]["t_equivalent"]),
        # H1
        "raw": f(hc["RAW"]["exact_recovery"]), "bch": f(hc["BCH16"]["soft"]), "h1ci": ci(hc["BCH16s_minus_WAM"]),
        "nsig": str(sum(p < 0.0125 for p in HOLM.values())), "ncond": str(len(H)), "mit": ci(HM[DC]["BCH16s_minus_WAM"]),
        "rnd": ci(hc["BCH16s_minus_RND16s"]), "padwam": ci(e3c["PAD16s_minus_RAW"]), "pad": ci(e3c["BCH16s_minus_PAD16s"]),
        "rep": ci(hc["BCH16s_minus_REP16s"]), "sh": ci(e1[DC]["soft_minus_hard"]), "bch21": f(hc["BCH21"]["soft"]),
        "padres": ci(e3b_all["reserve_E3_primary"]),
        # H2
        "cfa": str(mC["false_attributions"]), "cub": pct(mC["FA_upper_98.75"]), "cr": f(mC["R"]),
        "afa": str(mA["false_attributions"]), "aub": pct(mA["FA_upper_98.75"]), "ar": f(mA["R"]),
        "bfa": str(mB["false_attributions"]), "bub": pct(mB["FA_upper_98.75"]), "br": f(mB["R"]),
        "ca": ci(h2["R_C_minus_A"]), "cb": ci(h2["R_C_minus_B"]),
        "nmsg": n_(hu["locked"]["messages"]), "nimg": n_(hu["locked"]["images"]), "rcimg": f(hu["locked"]["methods"]["C"]["R_per_image_mean"]),
        "raimg": f(hu["locked"]["methods"]["A"]["R_per_image_mean"]), "capool": ci(hu["locked"]["C_minus_A"]["pooled"]),
        "cbpool": ci(hu["locked"]["C_minus_B"]["pooled"]), "nncapool": ci(hu["without_near_miss_scores"]["C_minus_A"]["pooled"]),
        "keyub": pct(f16["conditions"][DC]["F_target_KEY"]["CP_upper_98.75"]), "keyubimg": pct(float(beta.ppf(0.9875, 1, f16["images"]))), "nnat": str(nn["A"]["t_equivalent"]), "nnar": f(nn["A"]["R"]),
        "nnca": ci(nn["R_C_minus_A"]), "tlo": f"{stab['C']['tau']['p05']:.1f}", "thi": f"{stab['C']['tau']['p95']:.1f}",
        "famed": pct(stab["C"]["test_FA_rate"]["median"]), "fa95": pct(stab["C"]["test_FA_rate"]["p95"]),
        # H3
        "h3p1": ci(vd["H3"]["part1"]), "h3p2": ci(vd["H3"]["part2"]), "d34": ci(nt["RAW_d3_d4_mean-BCH_near (both transforms)"]),
        # H4
        "wm": f(hc["RAW"]["wam_bitacc_found_units"]), "h4ci": ci(hc["H4_wam_metric_minus_exact"]),
        "miss": f"{sf['missed'] * 100:.1f}%", "b13": f"{sf['1_3_bits'] * 100:.1f}%", "b4": f"{sf['4plus_bits'] * 100:.1f}%",
        "pq": f(hc["RAW"]["PQ"]), "pqb": f(hc["BCH16"]["PQ"]),
        # quality, cost
        "psnr": ci(r4["differences"]["BCH16-RAW:full_psnr"], 4), "psnrraw": f(r4["means"]["RAW"]["full_psnr"], 2),
        "ssim": f(r4["means"]["RAW"]["full_ssim"], 4), "wampsnr": re.search(r"PSNR ([\d.]+) dB", r4plan["reference"]).group(1),
        "nqual": str(r4["images"]), "emb": f"{e4['stages_ms']['any']['embed_one_message']['median']:.2f}",
        "det": f"{st['detect']['median']:.2f}", "db": f"{st['dbscan_gpu']['median']:.2f}", "dec": f"{st['bch16_soft_decode_all_regions']['median']:.2f}",
        "ntime": str(e4["images_timed"]),
        # forgery, reproducibility, Figure 1
        "nf16": str(f16["images"]), "nkeys": n_(f16["keys_simulated"]),
        "a1reg": n_(a1["regions"]), "a1same": n_(a1["same_codeword"]), "a2fa": str(a2b["false_attributions"]), "a2r": f(a2b["R"], 5),
        "a2tau": f(a2a["tau_reference_impl"], 5), "a2cnone": f"{n_(a2c['none']['correct_accepted'])}/{n_(a2c['none']['slots'])}",
        "a2chc": f"{n_(a2c[DC]['correct_accepted'])}/{n_(a2c[DC]['slots'])}",
        "f1img": f1["image_file"], "f1lic": f1["image_license"]["name"], "f1licurl": f1["image_license"]["url"],
        "f1rawok": str(sum(r["raw_centroid_bit_errors"] == 0 for r in f1["regions"])),
        "table1": body((ROOT / "figures/table1_hypotheses.md").read_text(encoding="utf-8")),
        "table2": body((ROOT / "figures/table2_forgery.md").read_text(encoding="utf-8")),
    }
    assert lock["test_images_measured_before_lock"] is False
    g = f16["conditions"][DC]
    assert g["F_untarget_KEY_over_keys"]["median"] > 0.01 and g["F_target_KEY"]["successes"] == 0  # statements in the forgery paragraph
    nboot = proto["decisions"]["bootstrap"]["repeats"]
    V["nboot"], V["nboot1"] = n_(nboot), n_(nboot + 1)
    V["nr3"] = n_(int(re.search(r"for _ in range\((\d+)\)", (ROOT / "scripts/verify/review_r3.py").read_text(encoding="utf-8")).group(1)))
    acc = sum(r["accepted"] and r["correct"] for r in f1["regions"])
    V["f1acc"] = "All five" if acc == len(f1["regions"]) == 5 else f"{acc} of the {len(f1['regions'])}"
    n = f1["images_scanned"]
    V["f1nth"] = f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
    hi = [c for c in H if H[c]["RAW"]["wam_bitacc_found_units"] >= 0.9]
    V["nba90"], V["ba90lo"], V["ba90hi"] = str(len(hi)), f(min(H[c]["RAW"]["exact_recovery"] for c in hi)), f(max(H[c]["RAW"]["exact_recovery"] for c in hi))
    rows = ["| Distortion | WAM raw | BCH16 hard | BCH16 soft | Soft − raw [98.75% CI] | Holm p |", "|---|---|---|---|---|---|"]
    for c in H:
        rows.append(f"| {NAMES[c]} | {f(H[c]['RAW']['exact_recovery'])} | {f(H[c]['BCH16']['hard'])} | {f(H[c]['BCH16']['soft'])} | "
                    f"{ci(H[c]['BCH16s_minus_WAM'])} | {holm(HOLM[c])} |")
    V["h1table"] = "\n".join(rows)
    dist = [("RAW_d1", "1"), ("RAW_d2", "2"), ("RAW_d3", "3"), ("RAW_d4", "4"), ("RAW_d6", "6"), ("RAW_d8", "8"), ("RAW_rand", "random"),
            ("BCH_near", "8 (nearest codewords)"), ("BCH_rand", "random codewords")]
    rows = ["| Pair | Hamming distance | Merge rate, no distortion | Merge rate, h-flip + contrast 1.5 |", "|---|---|---|---|"]
    for k, d in dist:
        rows.append(f"| {'RAW' if k.startswith('RAW') else 'BCH16'} | {d} | {f(rt['none'][k]['merge'])} | {f(rt[DC][k]['merge'])} |")
    V["h3table"] = "\n".join(rows)
    return V


TEMPLATE = r"""# CCMW — Codebook-Constrained soft decoding and calibrated attribution for localized Multi-message Watermarks

We provide the code, the locked execution protocol, the result files, figures and tables for the paper

> **Codebook-constrained soft decoding improves exact message recovery and calibrated user attribution in localized multi-message image watermarks**

We do **not** propose a new watermarking model, and we do not retrain one. We keep [Watermark Anything (WAM)](https://github.com/facebookresearch/watermark-anything) (Sander et al., ICLR 2025) and its multi-message scene construction unchanged and change only two steps: (i) before embedding, messages are restricted to the codewords of the extended BCH(32,16) code; (ii) after WAM's DBSCAN clustering, every region is decoded by soft maximum-likelihood search over all 65,536 codewords, and an identity is accepted only if it is registered and its score exceeds a conformally calibrated threshold. We fixed the hypotheses, decision rules and protocol before testing, recorded them by SHA-256, and tested four hypotheses (H1–H4) on @@ntest@@ COCO val2017 test images. All four were supported.

[![Processing flow](figures/fig1_flow.png)](figures/fig1_flow.pdf)

_Figure 1 — We follow one reserve image (COCO val2017 `@@f1img@@`, @@f1lic@@, CC BY 2.0) through embedding (a–e) and decoding with calibrated attribution (f–j). The WAM model and its DBSCAN clustering are unchanged. @@f1acc@@ registered IDs were accepted. The raw WAM readout of the same image carrying random 32-bit messages recovered @@f1rawok@@ of the five messages exactly. The figure illustrates the flow; it is not a statistic._

* * *

## 1. Method

For a detected region R_r, WAM's extractor returns per-pixel bit logits L_i ∈ ℝ³². We average them, l_r = (1/|R_r|) Σ_{i∈R_r} L_i, and decode

ĉ_r = argmax_{c ∈ 𝒞} ⟨c, l_r⟩, score s_r = ⟨ĉ_r, l_r⟩, with codewords written as c ∈ {−1, +1}³².

Because all codewords have the same norm, this is maximum-likelihood decoding for l_r = a·c* + Gaussian noise for any gain a > 0. The 65,536 × 32 correlation is one matrix product on the GPU.

| Decoder | Input | Rule |
|---|---|---|
| WAM raw (baseline) | DBSCAN centroid (majority bits of the region) | read directly as the 32-bit message |
| BCH16 hard | DBSCAN centroid | nearest codeword in Hamming distance |
| **BCH16 soft (ours)** | mean bit logits l_r | argmax correlation over all 65,536 codewords |

Comparison codebooks (all decoded with the same soft decoder): RND16 (65,536 random 32-bit words), REP16 (16 information bits repeated twice), PAD16 (16-bit ID followed by a fixed 16-bit suffix, which adds no distance between IDs and no error correction), BCH21 (extended BCH(32,21), minimum distance 6).

| Attribution rule | Accept a region if | Threshold (calibrated on @@ncal@@ cal images, α = @@alpha@@) |
|---|---|---|
| A — WAM-t (baseline) | the raw message is within Hamming distance t of a registered message | t = @@at@@ |
| B — registry soft decoding | soft decoding within the registry only; margin best − second best > τ_B | τ_B = @@tauB@@ |
| **C — full-codebook soft decoding + registry check (ours)** | the best of all 65,536 codewords is a registered ID and s_r > τ_C | τ_C = @@tauC@@ |

All three thresholds use the same conformal rank rule. For each calibration scene t the null score z_t is the largest score of a region decoded to a registered ID that was not embedded, and τ = z_(m+1) with m = ⌊α(n+1)⌋ − 1. Under exchangeability, P(z_new > τ) ≤ (m+1)/(n+1) ≤ α; the guarantee is marginal (over calibration sets and new scenes), and ties at −∞ only make it conservative. The calibration level α is stricter than the 1% criterion of H2, which is checked on the test scenes with a one-sided Clopper–Pearson upper confidence bound. This holds for scenes distributed like the calibration scenes, not for inputs crafted by an attacker. The registry holds @@N@@ BCH16 IDs and @@N@@ random 32-bit messages (seed @@seedreg@@).

## 2. Pre-specified hypotheses and results

The protocol `configs/protocol_v5.json` (SHA-256 `@@lockhash@@`) was locked at @@locktime@@ (`configs/protocol_v5_lock.json`). The lock also covers the WAM checkpoints, the per-split image lists and the measurement code; the test images were not measured before the lock. The locks are SHA-256 hashes that we recorded ourselves: they show that the locked files were not changed afterwards, but they do not independently certify when the files were created, because the plans were not deposited with an external registry before testing. The four hypothesis tests are the primary analyses; E1–E4, E3b, R4 and F16 are supplementary analyses with plans locked before measurement; R3 and the H2 estimand analysis are post hoc analyses of existing records.

@@table1@@

## 3. Setup

| Item | What we used |
|---|---|
| Model | WAM at commit `@@commit@@` (unchanged); `wam_coco.pth` (the checkpoint evaluated in the WAM paper; all decisions, sha256 `@@coco_sha@@…`), `wam_mit.pth` (secondary check, sha256 `@@mit_sha@@…`); released embedding strength (scaling_w 2.0, scaling_i 1.0, JND attenuation) |
| Images | COCO val2017 (5,000 images), sorted by file name: development 0–499, calibration 500–1,499, test 1,500–4,499, reserve 4,500–4,999; resized to 256 × 256 |
| Scenes | WAM Sec. 5.5 layout: k = 1–5 messages, each in a square of 10% of the image on a 3 × 3 checkerboard; the same image is embedded once per message and each watermarked square is pasted onto the original |
| Distortions | 26 conditions: all WAM App. D.4 settings, JPEG 80 + brightness 1.5 + crop 0.5, and the two compound distortions of WAM Sec. 5.5 (h-flip + contrast 1.5; the same followed by JPEG 80), applied with WAM's augmentation classes |
| Detection and regions | σ(p_i) > 0.5; WAM's DBSCAN on bits (Hamming ε = 1, at least 1,000 pixels), GPU re-implementation verified identical to WAM's routine (`src/wamset/dbscan_gpu.py`) |
| Statistics | 98.75% image-paired bootstrap intervals (@@nboot@@ resamples, seed @@seedboot@@; Bonferroni over four hypotheses), one-sided 98.75% Clopper–Pearson bounds for false attribution, Holm correction for secondary comparisons (reported only) |
| H2 scenes | one scene per image (seeds: cal @@seedcal@@, test @@seedtest@@): k uniform in 0–5, distortion uniform over the 26 conditions, each message registered with probability 0.5, unregistered messages random or near-miss |
| Compute | NVIDIA RTX 4090 (24 GB), Python 3.10.12, torch 2.3.1+cu121, torchvision 0.18.1, FP32 |

## 4. Installation

```bash
python3.10 -m venv venv && . venv/bin/activate
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

We use WAM from its own repository and do not redistribute it or its checkpoints. The measurement scripts are hash-locked, so we did not edit their absolute paths: `scripts/verify/oracle_h1.py` sets `REPO = /workspace/wam/JISA_selected_sources/watermark-anything` and `COCO = /workspace/wam/datasets/coco2017/val2017`, and every script writes its records to `/workspace/wam/artifacts/<stage>/`. Create these paths (or symbolic links to them):

```bash
mkdir -p /workspace/wam/JISA_selected_sources /workspace/wam/datasets/coco2017 /workspace/wam/artifacts
git clone https://github.com/facebookresearch/watermark-anything /workspace/wam/JISA_selected_sources/watermark-anything
git -C /workspace/wam/JISA_selected_sources/watermark-anything checkout @@commit@@
wget -P /workspace/wam/JISA_selected_sources/watermark-anything/checkpoints \
     https://dl.fbaipublicfiles.com/watermark_anything/wam_coco.pth https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth
ln -s /path/to/coco2017/val2017 /workspace/wam/datasets/coco2017/val2017
ln -s /path/to/coco2017/annotations /workspace/wam/datasets/coco2017/annotations   # Figure 1 only (licences, captions)
python scripts/check_locks.py --wam /workspace/wam/JISA_selected_sources/watermark-anything --coco /workspace/wam/datasets/coco2017/val2017
```

`check_locks.py` verifies the protocol, the four addenda, the locked code files, the WAM commit, source files and checkpoints, and the SHA-256 of each of the four image lists. The scripts run from the repository root, for example `python scripts/verify/v4_test.py`. `src/wamset/` contains only the two helper modules that the scripts import (`config.py`, `dbscan_gpu.py`); the package was first written for an earlier design that we abandoned before the v5 protocol.

## 5. Images

We do not redistribute COCO. The splits are defined by file-name order (`configs/protocol_v5.json`, `"images"`); `reports/analysis/v5_image_manifest.json` lists the index range, count and first file of each split, and the lock holds the SHA-256 of each list of `name:sha256` lines. Development images fixed the methods and decision rules, calibration images were used only for the H2 thresholds, test images were measured once after the lock, and reserve images were used only for the supplementary analyses (equal-capacity baseline, timing, image quality, forgery, Figure 1). Messages whose visible area was smaller than 1,000 pixels were excluded from the decisions and counted separately. The Figure 1 image (`@@f1img@@`) is the @@f1nth@@ reserve image in name order and the first that met the selection rule of `scripts/figures/fig1_flow.py` (open licence, no person in its captions, all five codewords accepted, at least one raw readout with 1–3 bit errors); licence: [@@f1lic@@](@@f1licurl@@).

## 6. Repository layout and pipeline

```
configs/              locked protocol v5, its lock, four locked addenda (E, E3b, R4, F16) and their locks
src/wamset/           helper modules used by the scripts (progress bar, seeds, GPU DBSCAN)
scripts/verify/       measurement and analysis scripts (stages below)
scripts/figures/      figures, tables and algorithm boxes
scripts/check_locks.py, scripts/build_readme.py
reports/analysis/     summary results (JSON) of every stage
figures/              Figures 1–5 (PNG, PDF), Tables 1–2 (Markdown, LaTeX), Algorithms 1–2 (LaTeX), Figure 1 panels and data
```

| Stage | Script | Images | GPU | Output (`reports/analysis/`) |
|---|---|---|---|---|
| Oracle feasibility (WAM conditions) | `verify/oracle_h1.py` | dev 0–199 | yes | `oracle_h1_summary.json` |
| H1 pre-test (codes on WAM's 32 bits) | `verify/h1_pretest.py` | dev 0–199 | yes | `h1_pretest_summary.json` |
| V1: H1 on development data | `verify/v1_h1.py` | dev 0–499 | yes | `v1_summary.json` |
| V1b: H2/H3 on development data | `verify/v1b_h2h3.py` | dev 0–499 | yes | `v1b_summary.json` |
| V2: protocol lock | — | — | — | `configs/protocol_v5_lock.json` |
| V3: H2 calibration | `verify/v3_cal.py` | cal | yes | `v3_thresholds.json` |
| V4: test, decisions H1–H4 | `verify/v4_test.py` | test | yes | `v4_summary.json` |
| E1/E2: soft vs hard, threat-model sensitivity | `verify/extra_e1e2.py` | records of V3/V4 | no | `extra_E1_summary.json`, `extra_E2_summary.json` |
| E3/E4: equal-capacity baseline, timing | `verify/extra_e3e4.py` | reserve | yes | `extra_E3_summary.json`, `extra_E4_summary.json` |
| E3b: equal-capacity baseline on test | `verify/extra_e3b.py` | test | yes | `extra_E3b_summary.json` |
| R3: threshold stability, registry size, merging at d = 3–4, failure decomposition | `verify/review_r3.py` | records of V1b/V3/V4 | no | `review_R3_summary.json` |
| R4: image quality (PSNR, SSIM) | `verify/review_r4.py` | reserve | yes | `review_R4_summary.json` |
| F16: forgery, public vs keyed assignment | `verify/review_f16.py` | reserve | yes | `review_F16_summary.json` |
| Development H3 differences by distance | `verify/review_dev_h3.py` | records of V1b | no | `review_dev_h3_summary.json` |
| H2 estimands: pooled and per-image differences (post hoc) | `verify/review_h2_units.py` | records of V4 | no | `review_h2_units_summary.json` |
| Re-implementation of Algorithms 1–2 | `verify/algo_repro.py` | reserve + records | yes | `algo_repro_summary.json` |
| Figure 3 curves | `figures/h2_curves.py` | records of V4 | no | `fig_h2_curves.json` |
| Figure 1 | `figures/fig1_flow.py` | reserve | yes | `figures/fig1_flow.*`, `figures/fig1_data.json`, `figures/fig1/` |
| Figures 2–5, Tables 1–2 | `figures/build_results.py` | result files | no | `figures/fig2_h1.*` … `figures/fig5_h4.*`, `figures/table*.{md,tex}` |
| Algorithms 1–2 | `figures/build_algorithms.py` | result files | no | `figures/algorithms.tex` |
| README | `build_readme.py` | result files | no | `README.md` |

The scripts write their per-image records (`*.jsonl.gz`) and summaries to `/workspace/wam/artifacts/<stage>/`; we copied the summaries into `reports/analysis/`. We do not include the per-image records; they are regenerated by the scripts and are available from the corresponding author on request. V3 and V4 stop if any locked hash differs; E1–E4, E3b, R4 and F16 also check their addendum locks. R3, the development H3 recomputation and the Figure 3 curves reuse existing records without new measurement. We generate every number in this README from the files above with `scripts/build_readme.py`; `python scripts/build_readme.py --check` confirms that README.md matches them.

## 7. Results (@@ntest@@ test images, `wam_coco`, unless stated)

### H1 — Exact message recovery

Under h-flip + contrast 1.5, the decision condition, exact recovery rose from @@raw@@ (WAM raw) to @@bch@@ (BCH16 soft), @@h1ci@@. The gain was significant after Holm correction in @@nsig@@ of @@ncond@@ conditions. With `wam_mit` the difference was @@mit@@. BCH21 soft reached @@bch21@@.

[![H1](figures/fig2_h1.png)](figures/fig2_h1.pdf)

_Figure 2 — (a) Exact recovery of WAM raw decoding and BCH16 soft decoding for the 26 distortions (k = 1–5 pooled). (b) Paired differences in the decision condition, each removing one candidate explanation of the gain._

| Comparison (decision condition) | Difference [98.75% CI] | What it tests |
|---|---|---|
| BCH16 soft − RND16 soft | @@rnd@@ | algebraic structure of BCH beyond a random codebook (equivalent within ±0.02 under this condition) |
| PAD16 soft − WAM raw | @@padwam@@ | a smaller candidate set with a fixed suffix and no error correction (test, E3b) |
| BCH16 soft − PAD16 soft | @@pad@@ | codeword separation over 32 bits (test, E3b; reserve E3: @@padres@@) |
| BCH16 soft − REP16 soft | @@rep@@ | BCH16 vs. a repetition code with 16 neighbours at distance 2 per codeword |
| BCH16 soft − BCH16 hard | @@sh@@ | soft decoding (E1) |

Under this condition the gain therefore comes from well-separated codewords over all 32 bits and from soft decoding, not from the algebraic structure of BCH or a smaller candidate set.

<details><summary>Exact recovery for all 26 distortions</summary>

@@h1table@@

Holm-adjusted p for BCH16 soft − WAM raw (the bootstrap p cannot fall below 1/@@nboot1@@).
</details>

### H2 — Registry attribution in mixed scenes

| Method | Calibrated threshold | False attributions | Upper bound (98.75%) | Correct attribution R |
|---|---|---|---|---|
| A — WAM-t | t = @@at@@ | @@afa@@/@@ntest@@ | @@aub@@ | @@ar@@ |
| B — registry soft decoding | τ_B = @@tauB@@ | @@bfa@@/@@ntest@@ | @@bub@@ | @@br@@ |
| **C — ours** | τ_C = @@tauC@@ | @@cfa@@/@@ntest@@ | @@cub@@ | @@cr@@ |

FA counts scenes in which a registered ID absent from the scene is accepted; a region of one embedded user decoded as another embedded user lowers R but is not counted in FA. B and C differ in message representation, candidate set and acceptance score, so their comparison is between complete pipelines. R is pooled over the @@nmsg@@ registered eligible messages; the pooled differences are R(C) − R(A) = @@capool@@ and R(C) − R(B) = @@cbpool@@ (image-level bootstrap, post hoc). The locked H2 decision statistic is the mean per-image difference over the @@nimg@@ images with at least one registered eligible message: @@ca@@ for C − A (per-image means @@rcimg@@ and @@raimg@@) and @@cb@@ for C − B. In the sensitivity analysis E2, the null scores from regions of near-miss messages were excluded from the existing records before recalibration (no new scenes; the near-miss messages stay embedded); A then calibrates to t = @@nnat@@ (R = @@nnar@@) and R(C) − R(A) = @@nncapool@@ pooled (@@nnca@@ per image). Over @@nr3@@ resamples of the calibration scenes, τ_C ranged from @@tlo@@ to @@thi@@ (5th–95th percentile), with a median test false-attribution rate of @@famed@@ (95th percentile @@fa95@@) (R3).

[![H2](figures/fig3_h2.png)](figures/fig3_h2.pdf)

_Figure 3 — (a) False attribution versus correct attribution as the threshold varies (descriptive); diamonds mark the thresholds calibrated beforehand. (b) Sensitivity analysis E2: pooled correct attribution at the calibrated thresholds with the locked null scores and with the null scores of near-miss messages excluded; differences are pooled with image-level bootstrap intervals._

### H3 — Merging of nearby messages

H3 (1), RAW d = 1 − RAW random: @@h3p1@@; H3 (2), RAW d ≤ 4 − BCH16 nearest pairs: @@h3p2@@. The informative range is d = 3–4: @@d34@@.

@@h3table@@

[![H3](figures/fig4_h3.png)](figures/fig4_h3.pdf)

_Figure 4 — (a) DBSCAN merge rate of two messages in one image versus their Hamming distance, compared with the nearest BCH16 codeword pairs and random pairs. (b) Differences from the nearest BCH16 pairs; distance 1 is expected from ε = 1._

### H4 — Overestimation by the WAM metric

In the decision condition, WAM's multi-message metric (bit accuracy over the clusters found) was @@wm@@ while exact recovery was @@raw@@; difference @@h4ci@@. Of the failed messages, @@miss@@ were missed, @@b13@@ were found with 1–3 bit errors and @@b4@@ with at least 4 bit errors (R3). The @@nba90@@ distortions with a WAM metric of at least 0.90 span exact recovery from @@ba90lo@@ to @@ba90hi@@. A panoptic-quality-style metric was the same for both decoders (@@pq@@ and @@pqb@@).

[![H4](figures/fig5_h4.png)](figures/fig5_h4.pdf)

_Figure 5 — (a) WAM metric versus exact recovery for the 26 distortions; dashed line, equality. (b) Outcome of every embedded message: exact, 1–3 bit errors, ≥ 4 bit errors, or missed._

### Image quality and computational cost (reserve images, supplementary)

On @@nqual@@ reserve images the PSNR of BCH16 embedding differed from raw embedding by @@psnr@@ dB (equivalence margin ±0.1 dB); mean PSNR @@psnrraw@@ dB, SSIM @@ssim@@ (raw messages). The WAM paper reports @@wampsnr@@ dB under its own protocol; the values are not directly comparable. Median time on an RTX 4090 with batch size 1 (@@ntime@@ images, h-flip + contrast 1.5): embedding @@emb@@ ms per message, detection @@det@@ ms, DBSCAN @@db@@ ms, BCH16 soft decoding of all regions @@dec@@ ms.

### Forgery by an attacker who knows the public codebook (@@nf16@@ reserve images, supplementary)

@@table2@@

Calibrated acceptance controls only non-adversarial false attribution. With keyed assignment no targeted impersonation was observed (one-sided upper bound @@keyub@@ per region treating regions as independent, @@keyubimg@@ at the image level), but forged regions were attributed to unrelated registered users at a per-region rate above 1% (median over @@nkeys@@ keys). These are per-region rates under attack, not the scene-level false-attribution rate bounded in H2.

### Reproducibility of Algorithms 1 and 2

We re-implemented both algorithms from the pseudocode (`figures/algorithms.tex`) without the study's decoding and calibration functions (`scripts/verify/algo_repro.py`). The re-implementation reproduced the decoded codeword of all @@a1reg@@ forgery-experiment regions (@@a1same@@ identical), the threshold τ = @@a2tau@@, the test false attributions (@@a2fa@@/@@ntest@@), R = @@a2r@@, and the per-region acceptance counts (@@a2cnone@@ without distortion, @@a2chc@@ under h-flip + contrast 1.5).

## 8. Limitations

We used one model (WAM; decisions with `wam_coco`, `wam_mit` as a check), the 5,000 COCO val2017 images (WAM used the first 10,000) and WAM's 10% checkerboard scenes at 256 × 256 pixels, without irregular masks, other areas or high resolution. The payload drops from 32 to 16 bits (65,536 IDs). Under strong compression combined with other distortions no tested decoder recovered messages. H2 concerns non-adversarial false attribution in mixed scenes with a registry of @@N@@ IDs; the effect of the registry size is an approximation. A public codebook cannot resist forgery, and copying, replay, removal and adaptive attacks were not studied. The equal-capacity baseline, image quality, timing and forgery results are supplementary analyses, not locked decisions.

## Third-party material

WAM code: MIT licence ([facebookresearch/watermark-anything](https://github.com/facebookresearch/watermark-anything)); checkpoints: `wam_mit.pth` MIT, `wam_coco.pth` CC BY-NC (as listed in `configs/protocol_v5.json`); neither is redistributed here. COCO images are used under their Creative Commons licences and are not redistributed, except the Figure 1 image (@@f1lic@@).
"""


def render():
    V = values()
    out = TEMPLATE
    for k, v in V.items():
        out = out.replace(f"@@{k}@@", v)
    left = sorted(set(re.findall(r"@@(\w+)@@", out)))
    if left:
        raise SystemExit(f"unresolved tokens: {left}")
    return out


def main():
    text = render()
    path = ROOT / "README.md"
    if "--check" in sys.argv:
        same = path.is_file() and path.read_text(encoding="utf-8") == text
        print("README.md matches the result files" if same else "README.md differs from the result files")
        sys.exit(0 if same else 1)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {path} ({len(text):,} characters)")


if __name__ == "__main__":
    main()
