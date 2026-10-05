"""Build the hypothesis-driven display items from result files only (no new statistic):
  Table 1  hypotheses, locked decision rules, test results, verdicts (H1-H4)          figures/table1_hypotheses.{md,tex}
  Figure 2 H1: exact message recovery over 26 conditions + cause separation            figures/fig2_h1.{png,pdf}
  Figure 3 H2: false attribution vs correct attribution, threat-model sensitivity      figures/fig3_h2.{png,pdf}
  Figure 4 H3: DBSCAN merge rate vs Hamming distance                                   figures/fig4_h3.{png,pdf}
  Figure 5 H4: WAM metric vs exact recovery, failure decomposition                     figures/fig5_h4.{png,pdf}
  Table 2  forgery experiment (F16), public vs keyed assignment                        figures/table2_forgery.{md,tex}"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import beta as beta_dist  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
J = lambda p: json.loads((ROOT / p).read_text(encoding="utf-8"))  # noqa: E731
v4 = J("reports/analysis/v4_summary.json")
e1 = J("reports/analysis/extra_E1_summary.json")["E1"]
e2 = J("reports/analysis/extra_E2_summary.json")["E2"]
e3b = J("reports/analysis/extra_E3b_summary.json")["E3b"]["conditions"]
r3 = J("reports/analysis/review_R3_summary.json")
f16 = J("reports/analysis/review_F16_summary.json")
cur = J("reports/analysis/fig_h2_curves.json")
hu = J("reports/analysis/review_h2_units_summary.json")
H = v4["H1_H4_wam_coco"]["conditions"]
HOLM = v4["H1_H4_wam_coco"]["holm_BCH16s_minus_WAM"]
VD = v4["verdict"]
h2 = v4["H2"]
mA, mB, mC = (h2["methods"][k] for k in "ABC")
rt = v4["H3"]["rates"]
nt = r3["H3_nontrivial"]
dec = r3["H4_decomposition"]
DC = "hflip_contrast1.5"
MINUS = "\u2212"


def s(x, k=3):
    v = f"{x:+.{k}f}"
    return ("0.000" if float(v) == 0 else v).replace("-", MINUS)


def ci(c, k=3):
    return f"{s(c['diff'], k)} [{s(c['ci'][0], k)}, {s(c['ci'][1], k)}]"


def pct(x, k=2):
    return f"{x * 100:.{k}f}%"


NAMES = {"none": "None", "hflip": "H-flip", "brightness1.5": "Brightness 1.5", "brightness2.0": "Brightness 2.0",
         "contrast1.5": "Contrast 1.5", "contrast2.0": "Contrast 2.0", "hue-0.1": "Hue −0.1", "hue0.1": "Hue +0.1",
         "saturation1.5": "Saturation 1.5", "saturation2.0": "Saturation 2.0", "blur3": "Blur 3", "blur17": "Blur 17",
         "median3": "Median 3", "median7": "Median 7", "jpeg80": "JPEG 80", "jpeg50": "JPEG 50", "crop0.33": "Crop 0.33",
         "crop0.5": "Crop 0.5", "resize0.5": "Resize 0.5", "rotate-10": "Rotate −10°", "rotate10": "Rotate +10°",
         "perspective0.1": "Perspective 0.1", "perspective0.5": "Perspective 0.5", "combination": "JPEG80+Bright.+Crop",
         "hflip_contrast1.5": "H-flip + contrast 1.5", "hflip_contrast1.5_jpeg80": "H-flip + contr. + JPEG80"}
C_RAW, C_BCH, C_ALT = "#7f7f7f", "#1f77b4", "#d62728"
plt.rcParams.update({"font.size": 6.5, "font.family": "DejaVu Sans", "axes.titlesize": 7, "axes.titleweight": "bold",
                     "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.8,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def errbar(ax, y, c, color, **kw):
    ax.errorbar(c["diff"], y, xerr=[[c["diff"] - c["ci"][0]], [c["ci"][1] - c["diff"]]], fmt="o", color=color, ms=3.2, capsize=1.5, lw=0.8, **kw)


# ============================================================== Figure 2 (H1)
def fig2():
    conds = sorted(H, key=lambda c: H[c]["RAW"]["exact_recovery"])
    fig = plt.figure(figsize=(7.2, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], left=0.165, right=0.985, top=0.92, bottom=0.11, wspace=0.62)
    a = fig.add_subplot(gs[0])
    for y, c in enumerate(conds):
        r, b = H[c]["RAW"]["exact_recovery"], H[c]["BCH16"]["soft"]
        a.plot([r, b], [y, y], color="0.75", lw=1.0, zorder=1)
        a.scatter(r, y, color=C_RAW, s=10, zorder=2, label="WAM raw (32-bit)" if y == 0 else None)
        sig = HOLM[c] < 0.0125
        a.scatter(b, y, s=12, zorder=3, facecolors=C_BCH if sig else "white", edgecolors=C_BCH,
                  label="BCH16 soft (Holm-significant)" if y == 0 else None)
    a.set_yticks(range(len(conds)))
    a.set_yticklabels([NAMES[c] for c in conds])
    for t, c in zip(a.get_yticklabels(), conds):
        if c == DC:
            t.set_fontweight("bold")
    yd = conds.index(DC)
    a.axhspan(yd - 0.5, yd + 0.5, color="#ffe9a8", zorder=0)
    a.set_xlim(-0.02, 1.02)
    a.set_xlabel("Exact message recovery (test, 3,000 images)")
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], marker="o", ls="", color=C_RAW, ms=3.5, label="WAM raw (32-bit)"),
                      Line2D([], [], marker="o", ls="", color=C_BCH, ms=3.5, label="BCH16 soft, Holm-significant"),
                      Line2D([], [], marker="o", ls="", mfc="white", mec=C_BCH, ms=3.5, label="BCH16 soft, not significant")],
             loc="lower right", frameon=False)
    a.set_title("(a) Recovery under the 26 WAM distortions", loc="left")
    b = fig.add_subplot(gs[1])
    comps = [("BCH16 soft\n− WAM raw", H[DC]["BCH16s_minus_WAM"], C_BCH),
             ("BCH16 soft − random\ncodebook (RND16)", H[DC]["BCH16s_minus_RND16s"], C_ALT),
             ("BCH16 soft − ID + fixed\nsuffix (PAD16)", e3b[DC]["BCH16s_minus_PAD16s"], C_BCH),
             ("PAD16 − WAM raw", e3b[DC]["PAD16s_minus_RAW"], C_RAW),
             ("BCH16 soft −\nrepetition (REP16)", H[DC]["BCH16s_minus_REP16s"], C_BCH),
             ("BCH16 soft −\nBCH16 hard", e1[DC]["soft_minus_hard"], C_BCH)]
    for y, (lab, c, col) in enumerate(comps):
        errbar(b, y, c, col)
        b.text(c["ci"][1] + 0.015, y, ci(c), va="center", fontsize=5.4)
    b.axvline(0, color="k", lw=0.6)
    b.set_yticks(range(len(comps)))
    b.set_yticklabels([x[0] for x in comps])
    b.invert_yaxis()
    b.set_xlim(-0.05, 1.0)
    b.set_xlabel("Difference in exact recovery (98.75% CI)")
    b.set_title("(b) Source of the gain\n(h-flip + contrast 1.5)", loc="left")
    save(fig, "fig2_h1")


# ============================================================== Figure 3 (H2)
def fig3():
    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], left=0.08, right=0.98, top=0.88, bottom=0.17, wspace=0.35)
    a = fig.add_subplot(gs[0])
    style = {"A": ("A: WAM raw + Hamming tolerance", C_RAW), "B": ("B: registry-only soft decoding", "#ff7f0e"), "C": ("C: full-codebook soft + registry", C_BCH)}
    for k, (lab, col) in style.items():
        c = cur["curves"][f"{k}:with_near_miss"]
        fa, rr = np.array(c["FA"]), np.array(c["R"])
        o = np.lexsort((rr, fa))  # ascending false attribution; R achievable at that rate
        a.step(fa[o] * 100, rr[o], where="post", color=col, lw=1.0, label=lab)
        L = cur["locked"][k]
        a.scatter(L["FA"] * 100, L["R"], color=col, edgecolors="k", s=22, zorder=4, marker="D")
    a.axvline(1.0, color="k", ls="--", lw=0.6)
    a.text(1.02, 0.03, "1%: H2 criterion for\nthe upper bound", fontsize=5.5)
    a.set_xlim(0, 2.0)
    a.set_ylim(0, 1)
    a.set_xlabel("Scene-level false attribution rate (%)")
    a.set_ylabel("Correct attribution rate R (pooled)")
    a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.53), frameon=False)
    a.set_title("(a) Test scenes (3,000); ◆ locked thresholds from calibration", loc="left")
    b = fig.add_subplot(gs[1])
    tm = [("Locked null scores", e2["locked_with_near_miss"], hu["locked"]),
          ("Near-miss scores\nexcluded (E2)", e2["without_near_miss"], hu["without_near_miss_scores"])]
    w = 0.35
    for i, (lab, d, u) in enumerate(tm):
        ra, rc = u["methods"]["A"]["R_pooled"], u["methods"]["C"]["R_pooled"]
        assert abs(ra - d["A"]["R"]) < 1e-12 and abs(rc - d["C"]["R"]) < 1e-12
        b.bar(i - w / 2, ra, w, color=C_RAW, label="A (WAM-t)" if i == 0 else None)
        b.bar(i + w / 2, rc, w, color=C_BCH, label="C (proposed)" if i == 0 else None)
        b.text(i - w / 2, ra + 0.02, f"t = {d['A']['t_equivalent']}", ha="center", fontsize=5.5)
        b.text(i, max(ra, rc) + 0.09, f"R(C) − R(A), pooled\n{ci(u['C_minus_A']['pooled'])}", ha="center", fontsize=5.5)
    b.set_xticks(range(2))
    b.set_xticklabels([x[0] for x in tm])
    b.set_ylim(0, 1.32)
    b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    b.set_ylabel("Correct attribution rate R (pooled)")
    b.legend(loc="upper center", ncol=2, frameon=False)
    b.set_title("(b) Sensitivity to near-miss scores", loc="left")
    save(fig, "fig3_h2")


# ============================================================== Figure 4 (H3)
def fig4():
    ds = [1, 2, 3, 4, 6, 8]
    fig = plt.figure(figsize=(7.2, 2.7))
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.98, top=0.88, bottom=0.17, wspace=0.3)
    a = fig.add_subplot(gs[0])
    for c, col, lab in (("none", "#2ca02c", "no distortion"), (DC, "#9467bd", "h-flip + contrast 1.5")):
        a.plot(ds, [rt[c][f"RAW_d{d}"]["merge"] for d in ds], "o-", color=col, ms=3, lw=1, label=f"RAW pairs, {lab}")
        a.axhline(rt[c]["BCH_near"]["merge"], color=col, ls="--", lw=0.7)
        a.scatter([9.5], [rt[c]["RAW_rand"]["merge"]], color=col, marker="x", s=14)
    a.text(7.2, 0.10, "dashed: BCH16 nearest\ncodeword pairs (d = 8)", fontsize=5.4, ha="center")
    a.set_xticks(ds + [9.5])
    a.set_xticklabels([str(d) for d in ds] + ["rand."])
    a.set_xlabel("Hamming distance between the two messages")
    a.set_ylabel("DBSCAN merge rate")
    a.set_ylim(-0.02, 1.0)
    a.legend(loc="upper right", frameon=False)
    a.set_title("(a) Two messages in one image (test, 3,000)", loc="left")
    b = fig.add_subplot(gs[1])
    for off, c, col in ((-0.13, "none", "#2ca02c"), (0.13, DC, "#9467bd")):
        for d in ds:
            q = nt[f"RAW_d{d}-BCH_near:{c}"]
            b.errorbar(d + off, q["diff"], yerr=[[q["diff"] - q["ci"][0]], [q["ci"][1] - q["diff"]]], fmt="o", color=col, ms=3, capsize=1.5, lw=0.8)
    b.axhline(0, color="k", lw=0.6)
    b.axvspan(2.5, 4.5, color="#ffe9a8", zorder=0)
    q = nt["RAW_d3_d4_mean-BCH_near (both transforms)"]
    b.text(3.5, 0.55, f"d = 3–4 (non-trivial range)\n{ci(q)}", ha="center", fontsize=5.5)
    b.set_xticks(ds)
    b.set_xlabel("Hamming distance d of the RAW pair")
    b.set_ylabel("Merge rate difference\n(RAW d − BCH16 nearest)")
    b.set_ylim(-0.05, 1.0)
    b.set_title("(b) Difference to distance-8 codewords (98.75% CI)", loc="left")
    save(fig, "fig4_h3")


# ============================================================== Figure 5 (H4)
def fig5():
    conds = sorted(H, key=lambda c: dec[c]["exact"])
    fig = plt.figure(figsize=(7.2, 4.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], left=0.07, right=0.98, top=0.93, bottom=0.2, wspace=0.62)
    a = fig.add_subplot(gs[0])
    x = [H[c]["RAW"]["wam_bitacc_found_units"] for c in H]
    y = [H[c]["RAW"]["exact_recovery"] for c in H]
    a.plot([0, 1], [0, 1], color="k", lw=0.6, ls="--")
    a.scatter(x, y, s=10, color=C_RAW)
    hx, hy = H[DC]["RAW"]["wam_bitacc_found_units"], H[DC]["RAW"]["exact_recovery"]
    a.scatter([hx], [hy], s=22, color=C_ALT, zorder=3)
    a.annotate(f"H-flip + contrast 1.5\n{hx:.3f} vs {hy:.3f}\n{ci(H[DC]['H4_wam_metric_minus_exact'])}", (hx, hy), xytext=(0.43, 0.22),
               fontsize=5.5, arrowprops=dict(arrowstyle="-", lw=0.5))
    a.set_xlim(0.4, 1.0)
    a.set_ylim(-0.02, 1.0)
    a.set_xlabel("WAM metric: bit accuracy of found clusters")
    a.set_ylabel("Exact message recovery")
    a.set_title("(a) 26 conditions (WAM raw, test)", loc="left")
    b = fig.add_subplot(gs[1])
    parts = [("exact", "Exact", "#2ca02c"), ("found_1_3_bit_errors", "Found, 1–3 bit errors", "#ff7f0e"),
             ("found_4plus_bit_errors", "Found, ≥4 bit errors", "#d62728"), ("missed_no_unit", "Missed (no region)", "#7f7f7f")]
    left = np.zeros(len(conds))
    for key, lab, col in parts:
        v = np.array([dec[c][key] for c in conds])
        b.barh(range(len(conds)), v, left=left, color=col, height=0.75, label=lab)
        left += v
    b.set_yticks(range(len(conds)))
    b.set_yticklabels([NAMES[c] for c in conds])
    for t, c in zip(b.get_yticklabels(), conds):
        if c == DC:
            t.set_fontweight("bold")
    b.set_xlim(0, 1)
    b.set_xlabel("Share of messages (WAM raw)")
    b.legend(loc="upper center", bbox_to_anchor=(0.4, -0.08), ncol=2, frameon=False, columnspacing=1.2, handlelength=1.0)
    b.set_title("(b) Outcome of each embedded message", loc="left")
    save(fig, "fig5_h4")


# ============================================================== Tables
def tables():
    sf = dec[DC]["share_of_failures"]
    rows = [
        ("H1", "Codebook-constrained soft decoding recovers more messages exactly than WAM raw decoding",
         "Lower bound of BCH16 soft − WAM raw > 0 (h-flip + contrast 1.5, k = 1–5 pooled)",
         f"{H[DC]['RAW']['exact_recovery']:.3f} → {H[DC]['BCH16']['soft']:.3f}; {ci(H[DC]['BCH16s_minus_WAM'])}", VD["H1"]["SUPPORTED"], "Fig. 2"),
        ("H2", "Calibrated full-codebook attribution (C) attributes more messages correctly than Hamming-tolerance attribution (A) at ≤ 1% false attribution",
         "One-sided 98.75% upper bounds on scene-level false attribution ≤ 1% for A and C; lower end of the 98.75% interval of Δ_img (C − A) > 0 (mixed scenes)",
         f"C {mC['false_attributions']}/3,000 (≤ {pct(mC['FA_upper_98.75'])}), R {mC['R']:.3f}; A {mA['false_attributions']}/3,000 (≤ {pct(mA['FA_upper_98.75'])}), R_msg {mA['R']:.3f}; Δ_img {ci(h2['R_C_minus_A'])} (R_msg difference {ci(hu['locked']['C_minus_A']['pooled'])})",
         VD["H2"]["SUPPORTED"], "Fig. 3"),
        ("H3", "DBSCAN merges messages that are close in Hamming distance; codewords at distance 8 are rarely merged",
         "(1) RAW d = 1 − RAW random > 0; (2) RAW d ≤ 4 − BCH16 nearest > 0 (lower bounds)",
         f"(1) {ci(VD['H3']['part1'])}; (2) {ci(VD['H3']['part2'])}; d = 3–4: {ci(nt['RAW_d3_d4_mean-BCH_near (both transforms)'])}",
         VD["H3"]["SUPPORTED"], "Fig. 4"),
        ("H4", "WAM's multi-message metric overestimates exact message recovery",
         "Lower bound of WAM metric − exact recovery > 0 (h-flip + contrast 1.5)",
         f"{H[DC]['RAW']['wam_bitacc_found_units']:.3f} vs {H[DC]['RAW']['exact_recovery']:.3f}; {ci(H[DC]['H4_wam_metric_minus_exact'])}; "
         f"failures: {sf['missed'] * 100:.1f}% missed, {sf['1_3_bits'] * 100:.1f}% 1–3-bit, {sf['4plus_bits'] * 100:.1f}% ≥4-bit",
         VD["H4"]["SUPPORTED"], "Fig. 5"),
    ]
    note = ("Test set: 3,000 COCO val2017 images; the primary evaluation was run once after the protocol was locked (SHA-256 1b2a2d0f…); checkpoint wam_coco. "
            "Intervals: 98.75% image-paired bootstrap (10,000 resamples; Bonferroni over four hypotheses). False-attribution bounds: one-sided "
            "Clopper–Pearson 98.75%. R is pooled over messages (R_msg); the H2 decision statistic Δ_img is the mean per-image difference in correct attribution. H3 (1) and H4 are expected from the definitions of DBSCAN (ε = 1) and of the WAM metric; their information is in the size and causes. "
            "The distance 3–4 difference (H3), the failure decomposition (H4) and the pooled H2 difference are post hoc analyses of the test records.")
    md = "# Table 1. Hypotheses, locked decision rules and test results\n\n| | Hypothesis | Locked decision rule | Test result | Verdict | Shown in |\n|---|---|---|---|---|---|\n"
    md += "".join(f"| {h} | {t} | {r} | {res} | {'Supported' if v else 'Not supported'} | {f} |\n" for h, t, r, res, v, f in rows) + f"\n{note}\n"
    (OUT / "table1_hypotheses.md").write_text(md, encoding="utf-8")
    tex = ("% Generated by scripts/figures/build_results.py. Requires booktabs, tabularx.\n\\begin{table}[t]\n\\caption{Hypotheses, locked decision rules and test results.}\\label{tab:hyp}\n"
           "\\footnotesize\n\\begin{tabularx}{\\textwidth}{@{}l>{\\raggedright}X>{\\raggedright}X>{\\raggedright}Xll@{}}\n\\toprule\n & Hypothesis & Locked decision rule & Test result & Verdict & Shown in\\\\\n\\midrule\n")
    esc = lambda t: t.replace("%", "\\%").replace("≤", "$\\le$").replace("≥", "$\\ge$").replace("→", "$\\rightarrow$").replace(MINUS, "$-$").replace("–", "--")  # noqa: E731
    tex += "".join(f"{h} & {esc(t)} & {esc(r)} & {esc(res)} & {'Supported' if v else 'Not supported'} & {f}\\\\\n" for h, t, r, res, v, f in rows)
    tex += f"\\bottomrule\n\\end{{tabularx}}\n\\par\\smallskip {esc(note)}\n\\end{{table}}\n"
    (OUT / "table1_hypotheses.tex").write_text(tex, encoding="utf-8")

    t2 = []
    for c, g in f16["conditions"].items():
        u = g["F_untarget_KEY_over_keys"]
        t2.append((NAMES[c], f"{g['eligible_slots']:,}", f"{g['R_legit_PUB']:.3f} / {g['R_legit_KEY']:.3f}", ci(g["R_legit_KEY_minus_PUB"]),
                   f"{g['F_target_PUB']['successes']:,}/{g['eligible_slots']:,}", f"{g['F_target_KEY']['successes']}/{g['eligible_slots']:,} (≤ {pct(g['F_target_KEY']['CP_upper_98.75'])})",
                   (f"0/{f16['images']} (≤ {pct(float(beta_dist.ppf(0.9875, 1, f16['images'])))})" if g['F_target_KEY']['successes'] == 0 else "n/a"),
                   f"{pct(g['F_untarget_KEY_K0']['rate'])} / {pct(u['median'])} [{pct(u['p05'])}, {pct(u['p95'])}]", pct(g["F_untarget_expected"])))
    hdr = ["Distortion", "Target regions", "Legitimate attribution PUB / KEY", "KEY − PUB (98.75% CI)", "Targeted impersonation, PUB",
           "Targeted impersonation, KEY: regions (upper bound, independent regions)", "Targeted impersonation, KEY: images with ≥ 1 successful region (upper bound)", "Framing another registered user, KEY: K0 / median of 1,000 keys [5–95%]", "Expected N/65,536 × R_legit"]
    note2 = (f"Supplementary analysis on the {f16['images']} reserve images (plan locked before measurement). Attacker: same WAM embedder, public codebook and decoding rule, "
             "public ID of the target; no key and no detector queries. PUB: ID = codeword index; KEY: secret permutation of IDs to codewords. "
             "Rates are per forged region (five regions per image) unless stated otherwise; region-level upper bounds treat regions as independent, the image-level column does not. "
             f"Method C with the locked threshold and the registry of {f16['registry_N']:,} IDs. Copy/replay, removal and adaptive attacks are not covered.")
    md2 = "# Table 2. Forgery by an attacker who knows the public codebook\n\n| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n"
    md2 += "".join("| " + " | ".join(r) + " |\n" for r in t2) + f"\n{note2}\n"
    (OUT / "table2_forgery.md").write_text(md2, encoding="utf-8")
    tex2 = ("% Generated by scripts/figures/build_results.py. Requires booktabs, tabularx.\n\\begin{table}[t]\n\\caption{Forgery by an attacker who knows the public codebook.}\\label{tab:forgery}\n"
            "\\footnotesize\n\\begin{tabularx}{\\textwidth}{@{}l" + ">{\\raggedright}X" * (len(hdr) - 1) + "@{}}\n\\toprule\n" + " & ".join(esc(h) for h in hdr) + "\\\\\n\\midrule\n")
    tex2 += "".join(" & ".join(esc(x) for x in r) + "\\\\\n" for r in t2)
    tex2 += f"\\bottomrule\n\\end{{tabularx}}\n\\par\\smallskip {esc(note2)}\n\\end{{table}}\n"
    (OUT / "table2_forgery.tex").write_text(tex2, encoding="utf-8")
    return rows, t2


if __name__ == "__main__":
    fig2()
    fig3()
    fig4()
    fig5()
    tables()
    print("written fig2_h1, fig3_h2, fig4_h3, fig5_h4, table1_hypotheses, table2_forgery")
