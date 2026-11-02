#!/usr/bin/env python3
"""
09_abstract_figures.py: figures for the ISMRM abstract.

Reads the corrected statistics written by 05b_dfc_group_statistics.py and
produces, into report/abstract/figures/:

    fig2_dynamic_fc.png          two panels: effect sizes for the four
                                 whole-brain measures and the network-block
                                 allegiance difference
    fig3_significant_regions.png glass brains of the 38 literature-defined
                                 anorexia-relevant regions (17_apriori_region_test),
                                 one row per measure

Run:
    conda activate fmri
    python 09_abstract_figures.py            # 21 AN v 22 HC (default)
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from paths import ROOT
OUT = ROOT / "report/abstract/figures"
ATLAS = {
    "cortical200": dict(stats=ROOT / "output/results/dfc_statistics_cortical200",
                        n=200, label="cortical-200"),
    "combined216": dict(stats=ROOT / "output/results/dfc_statistics_combined216",
                        n=216, label="combined-216"),
}
CFG = ATLAS["combined216"]          # replaced in main()
COHORT = "21v22"                    # replaced in main()
APRIORI = ROOT / "output/results/apriori_regions_combined216"

_spec = importlib.util.spec_from_file_location(
    "dfc", Path(__file__).with_name("05b_dfc_group_statistics.py"))
_dfc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dfc)
OUT.mkdir(parents=True, exist_ok=True)

AN_COLOR, HC_COLOR = "#C0392B", "#2C6FBB"
ALPHA = 0.05
MEASURES = [("RC", "rec"), ("IC", "int"),
            ("Flexibility", "flex"), ("Promiscuity", "prom")]
# Figure 3 regions: one colour family per direction (blue AN > HC, orange
# AN < HC); light = p < 0.05 uncorrected, dark = surviving FDR
UNC_UP, FDR_UP = "#6BAED6", "#08519C"
UNC_DOWN, FDR_DOWN = "#FDAE6B", "#D94801"

# Sized for the printed page: figures are drawn near their final width so that
# text stays legible (about 8-9 pt once placed in the abstract). All text black.
SUBCORT_COLOR = "#8B3A00"   # tick labels of subcortical groups
plt.rcParams.update({"font.size": 16, "axes.titlesize": 17, "axes.labelsize": 17,
                     "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "axes.edgecolor": "black", "axes.linewidth": 1.3,
                     "xtick.color": "black", "ytick.color": "black",
                     "text.color": "black", "axes.labelcolor": "black",
                     "savefig.dpi": 600, "figure.dpi": 150})


def d_ci(d, n1, n2, z=1.96):
    """Large-sample confidence interval for Cohen's d.

    The standard error is sqrt((n1+n2)/(n1 n2) + d^2 / (2(n1+n2))), the usual
    approximation for an independent-groups d.
    """
    se = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    return d - z * se, d + z * se


def figure2(z, gsum):
    NETWORKS = [str(v) for v in z["group_names"]]
    n1 = z["rec_subj_an"].shape[0]
    n2 = z["rec_subj_hc"].shape[0]
    g = gsum.set_index("measure")

    fig = plt.figure(figsize=(12, 13.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.62, 1.45], hspace=0.42)

    # --- A: effect sizes for all four whole-brain measures ---
    ax = fig.add_subplot(gs[0, 0])
    rows = [("RC", "Mean_Recruitment"), ("IC", "Mean_Integration"),
            ("Flexibility", "Flexibility"), ("Promiscuity", "Promiscuity")]
    ys = np.arange(len(rows))[::-1]
    ax.axvspan(-0.5, 0.5, color="#E6E6E6", zorder=0)
    ax.text(0, len(rows) - 0.32, "|d| < 0.5", fontsize=14, ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="#E6E6E6", ec="none"))
    for y, (lab, key) in zip(ys, rows):
        d = float(g.loc[key, "cohens_d"]); p = float(g.loc[key, "pvalue"])
        lo, hi = d_ci(d, n1, n2)
        sig = p < 0.05
        col = AN_COLOR if sig else "#333333"
        ax.plot([lo, hi], [y, y], color=col, lw=3, solid_capstyle="round")
        ax.scatter([d], [y], s=140, color=col, zorder=4, edgecolor="black", linewidth=1.2)
        ax.text(max(hi + 0.12, 0.12), y, f"$d$ = {d:+.2f}\n$p$ = {p:.3f}", va="center",
                fontsize=14, fontweight="bold" if sig else "normal", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    ax.axvline(0, color="black", lw=1.2, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=16)
    ax.set_ylim(-0.6, len(rows) - 0.05)
    ax.set_xlabel("Cohen's $d$ (AN − HC) with 95% CI")
    ax.set_xlim(-1.7, 2.5)
    ax.set_title("A  Whole-brain measures", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # --- B: network-block allegiance difference, cortex vs subcortex marked ---
    ax = fig.add_subplot(gs[1, :])
    can, chc = z["coarse_mean_an"], z["coarse_mean_hc"]
    diff = can - chc
    ng = len(NETWORKS)
    n_ctx = sum(not n.startswith("~") for n in NETWORKS)
    pc = np.full((ng, ng), np.nan)
    iu, ju = np.triu_indices(ng, 0)
    pc[iu, ju] = z["p_coarse"]; pc[ju, iu] = z["p_coarse"]
    v = np.abs(diff).max()
    im = ax.imshow(diff, cmap="bwr", vmin=-v, vmax=v)
    labels = [n[1:] if n.startswith("~") else n for n in NETWORKS]
    ax.set_xticks(range(ng)); ax.set_yticks(range(ng))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=15)
    ax.set_yticklabels(labels, fontsize=15)
    for t in ax.get_xticklabels()[n_ctx:] + ax.get_yticklabels()[n_ctx:]:
        t.set_color(SUBCORT_COLOR); t.set_fontweight("bold")
    # divider between cortical and subcortical groups
    for k in (ax.axhline, ax.axvline):
        k(n_ctx - 0.5, color="black", lw=3)
    # brackets naming the two parts, outside the matrix
    for a, b, txt, col in ((-0.5, n_ctx - 0.5, "Cortical", "black"),
                           (n_ctx - 0.5, ng - 0.5, "Subcortical", SUBCORT_COLOR)):
        ax.annotate("", xy=(a + 0.1, -0.9), xytext=(b - 0.1, -0.9), annotation_clip=False,
                    arrowprops=dict(arrowstyle="|-|", lw=1.6, color=col))
        ax.text((a + b) / 2, -1.25, txt, ha="center", va="bottom", fontsize=16,
                fontweight="bold", color=col)
        ax.annotate("", xy=(ng - 0.5 + 0.9, a + 0.1), xytext=(ng - 0.5 + 0.9, b - 0.1),
                    annotation_clip=False, arrowprops=dict(arrowstyle="|-|", lw=1.6, color=col))
        ax.text(ng - 0.5 + 1.25, (a + b) / 2, txt, ha="left", va="center", fontsize=16,
                fontweight="bold", color=col, rotation=270)
    for i_ in range(ng):
        for j_ in range(ng):
            if pc[i_, j_] < 0.05:
                ax.add_patch(Rectangle((j_ - .5, i_ - .5), 1, 1, fill=False,
                                       edgecolor="black", lw=2.6))
    ax.text(0.5, -0.2, "Networks with subcortical groups appear twice: once for their cortical regions (black)\n"
            "and once for the subcortical regions assigned to them (brown, e.g. caudate to FPN).",
            transform=ax.transAxes, ha="center", va="top", fontsize=14)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.13)
    cb.set_label("Δ allegiance (AN − HC)", fontsize=16)
    cb.ax.tick_params(labelsize=14)
    ax.set_title("B  Network-block allegiance difference\n"
                 "(outlined: $p$ < 0.05, uncorrected)",
                 loc="left", fontweight="bold", pad=48)

    dest = OUT / "fig2_dynamic_fc.png"
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)
    print(f"  {dest}")


def figure3():
    """Glass brains of the 38 a priori regions, one row per measure."""
    from nilearn import plotting
    coords = np.load(ROOT / "data/analysis/coords216.npy")
    reg = pd.read_csv(APRIORI / "region_tests.csv")
    reg = reg[(reg.cohort == COHORT) & (reg.fd_adjust == "none")]
    name = {"rec": "recruitment", "int": "integration",
            "flex": "flexibility", "prom": "promiscuity"}
    # a measure with no region at p<0.05 is named in the title, not drawn
    shown = [(l, k) for l, k in MEASURES if (reg[reg.measure == name[k]].p < ALPHA).any()]
    empty = [l for l, k in MEASURES if (l, k) not in shown]
    height = 3.3 * len(shown) + 3.3
    fig, axes = plt.subplots(len(shown), 1, figsize=(10, height),
                             gridspec_kw=dict(hspace=0.5), squeeze=False)
    fig.subplots_adjust(top=1 - 3.3 / height)
    # orientation labels per view (nilearn: l = left lateral, z = axial,
    # r = right lateral, y = coronal; neurological convention)
    ORIENT = {"l": [("A", 0.08, 0.12), ("P", 0.92, 0.12)],
              "r": [("P", 0.08, 0.12), ("A", 0.92, 0.12)],
              "z": [("A", 0.5, 0.94), ("P", 0.5, 0.05), ("L", 0.1, 0.5), ("R", 0.9, 0.5)],
              "y": [("L", 0.1, 0.1), ("R", 0.9, 0.1)]}
    for ax, (label, key) in zip(axes[:, 0], shown):
        m = reg[reg.measure == name[key]]
        xyz = coords[m["index"].to_numpy() - 1]
        sig = (m.p < ALPHA).to_numpy()
        surv = (m.p_fdr < ALPHA).to_numpy()
        d = m.d.to_numpy()
        disp = plotting.plot_glass_brain(None, display_mode="lzry", axes=ax, annotate=False)
        sub = (m["index"].to_numpy() > 200)          # Tian S1 subcortical regions
        for shape, part in (("o", ~sub), ("^", sub)):
            hit = sig & part
            if hit.any():
                col = [(FDR_UP if v > 0 else FDR_DOWN) if s_ else (UNC_UP if v > 0 else UNC_DOWN)
                       for v, s_ in zip(d[hit], surv[hit])]
                size = 45 + 95 * np.abs(d[hit]) / np.abs(d).max()
                disp.add_markers(xyz[hit], marker_color=col, marker=shape, edgecolors="black",
                                 linewidths=0.8, marker_size=np.where(surv[hit], size * 1.8, size))
        for view, labs in ORIENT.items():
            vax = disp.axes[view].ax
            for txt, x, y in labs:
                vax.text(x, y, txt, transform=vax.transAxes, ha="center", va="center",
                         fontsize=15, fontweight="bold", color="black", zorder=10,
                         bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
        n_fdr = int(surv.sum())
        ax.set_title(f"{label}: {int(sig.sum())} of {len(m)} regions at p < 0.05"
                     + (f", {n_fdr} after FDR correction" if n_fdr else ""),
                     fontsize=18, fontweight="bold", y=1.08)
    n_an, n_hc = (19, 22) if COHORT == "19v22" else (21, 22)
    fig.suptitle(f"38 anorexia-related regions, {n_an} AN vs {n_hc} HC",
                 fontsize=19, fontweight="bold", y=0.997)
    # legend drawn with the actual symbols and colours
    from matplotlib.lines import Line2D
    def mk(c, shape="o", face=None):
        return Line2D([], [], ls="", marker=shape, markersize=15, markerfacecolor=face or c,
                      markeredgecolor="black",
                      markeredgewidth=2.2 if face == "none" else 0.8, color=c)
    head = lambda txt: Line2D([], [], ls="", marker="", label=txt)
    rows = [
        ([head("AN > HC:"), mk(UNC_UP), mk(FDR_UP)],
         ["AN > HC:", "p < 0.05, uncorrected", "FDR-corrected"]),
        ([head("AN < HC:"), mk(UNC_DOWN), mk(FDR_DOWN)],
         ["AN < HC:", "p < 0.05, uncorrected", "FDR-corrected"]),
        ([mk("black", "o", "none"), mk("black", "^", "none"), head("size ∝ |Cohen's d|")],
         ["cortical", "subcortical", "size ∝ |Cohen's d|"]),
    ]
    for k, (h, l) in enumerate(rows):
        leg = fig.legend(h, l, loc="upper center", ncol=len(h), frameon=False, fontsize=16,
                         bbox_to_anchor=(0.5, 1 - (0.8 + 0.62 * k) / height),
                         handletextpad=0.4, columnspacing=1.6)
        for t in leg.get_texts()[:1] if k < 2 else []:
            t.set_fontweight("bold")
    dest = OUT / "fig3_significant_regions.png"
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)
    print(f"  {dest}")


def main():
    global CFG, COHORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", choices=list(ATLAS), default="combined216")
    ap.add_argument("--cohort", choices=list(_dfc.COHORTS), default="21v22")
    args = ap.parse_args()
    CFG, COHORT = ATLAS[args.atlas], args.cohort
    STATS = _dfc.cohort_out(dict(out=CFG["stats"]), COHORT)
    z = np.load(STATS / "dfc_group_stats.npz", allow_pickle=True)
    gsum = pd.read_csv(STATS / "dfc_global_summary.csv")
    print("Building abstract figures from", STATS)
    figure2(z, gsum)
    figure3()


if __name__ == "__main__":
    sys.exit(main())
