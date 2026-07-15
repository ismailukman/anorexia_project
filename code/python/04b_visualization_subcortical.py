#!/usr/bin/env python3
"""
04b_visualization_subcortical.py — Figures and export for subcortical MLCD results.
Tian S2 atlas, 5+5 subjects.
"""
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

start_time = datetime.now()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT     = Path("/Users/ismaila/Documents/C-Codes/AnorexiaProject")
STAT_DIR = ROOT / "output/results/statistical_results_subcortical_5subj"
FIG_DIR  = ROOT / "output/figures/stage4_viz_subcortical_5subj"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_REGIONS = 32
N_SUBJ    = 5
PALETTE   = ["#FF6F61", "#1E90FF"]

LABEL_FILE = ROOT / "data/atlas/tian_s2/Tian_Subcortex_S2_3T_label.txt"
tian_labels = LABEL_FILE.read_text().strip().splitlines()

STRUCT_MAP   = {"HIP": 1, "AMY": 2, "THA": 3, "NAc": 4, "GP": 5, "PUT": 6, "CAU": 7}
STRUCT_NAMES = {1:"Hippocampus", 2:"Amygdala", 3:"Thalamus",
                4:"Nuc.Accumbens", 5:"Globus Pallidus", 6:"Putamen", 7:"Caudate"}
STRUCT_COLORS= {1:"#E41A1C", 2:"#377EB8", 3:"#4DAF4A", 4:"#984EA3",
                5:"#FF7F00", 6:"#A65628", 7:"#F781BF"}

def label_to_community(lbl):
    for k, v in STRUCT_MAP.items():
        if k.lower() in lbl.lower(): return v
    return 0

static_communities = np.array([label_to_community(l) for l in tian_labels], dtype=int)

alg_an   = np.load(STAT_DIR / "allegiance_group_an.npy")
alg_hc   = np.load(STAT_DIR / "allegiance_group_hc.npy")
alg_diff = np.load(STAT_DIR / "allegiance_diff_AN_HC.npy")
p_alg    = np.load(STAT_DIR / "pvalue_allegiance.npy")
rec_an   = np.load(STAT_DIR / "recruitment_subj_an.npy")
rec_hc   = np.load(STAT_DIR / "recruitment_subj_hc.npy")
int_an   = np.load(STAT_DIR / "integration_subj_an.npy")
int_hc   = np.load(STAT_DIR / "integration_subj_hc.npy")
flex_an  = np.load(STAT_DIR / "flexibility_subj_an.npy")
flex_hc  = np.load(STAT_DIR / "flexibility_subj_hc.npy")
prom_an  = np.load(STAT_DIR / "promiscuity_subj_an.npy")
prom_hc  = np.load(STAT_DIR / "promiscuity_subj_hc.npy")
df_nodal = pd.read_csv(STAT_DIR / "nodal_statistics_subcortical.csv")
summary  = pd.read_csv(STAT_DIR / "summary_global_tests.csv")

def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return f"ns"

def struct_boundary_lines(ax, orientation="both"):
    cur = 0
    for sid in sorted(STRUCT_MAP.values()):
        n = (static_communities == sid).sum()
        b = cur + n - 0.5
        if orientation in ("both","h"): ax.axhline(b, color="white", lw=0.8)
        if orientation in ("both","v"): ax.axvline(b, color="white", lw=0.8)
        cur += n

log.info("[Fig 1] Allegiance heatmaps (AN | HC | diff | p<0.05) …")
fig, axes = plt.subplots(1, 4, figsize=(22, 6))
titles = ["AN Allegiance", "HC Allegiance", "Diff (AN − HC)", "Significant (p<0.05)"]
data   = [alg_an, alg_hc, alg_diff, (p_alg < 0.05).astype(float)]
cmaps  = ["jet", "jet", "RdBu_r", "Reds"]
vlims  = [(0,1), (0,1), (-alg_diff.max(), alg_diff.max()), (0,1)]

for ax, d, t, cm, vl in zip(axes, data, titles, cmaps, vlims):
    im = ax.imshow(d, cmap=cm, vmin=vl[0], vmax=vl[1], aspect="auto")
    ax.set_title(t, fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    struct_boundary_lines(ax)
    plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

cur = 0
for sid in sorted(STRUCT_MAP.values()):
    n = (static_communities == sid).sum()
    axes[0].text(-1.5, cur + n/2 - 0.5, STRUCT_NAMES[sid][:3],
                 ha="right", va="center", fontsize=7, color=STRUCT_COLORS[sid], fontweight="bold")
    cur += n

plt.suptitle("Subcortical Allegiance — Tian Scale II (5 AN + 5 HC)", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "allegiance_subcortical_4panel.png", dpi=300, bbox_inches="tight")
plt.close(fig)
log.info("  Saved allegiance_subcortical_4panel.png")

log.info("[Fig 2] Recruitment & Integration per structure …")
structs = sorted(STRUCT_MAP.values())
struct_labels = [STRUCT_NAMES[s] for s in structs]

rec_an_struct = [rec_an[:, static_communities == s].mean(1) for s in structs]
rec_hc_struct = [rec_hc[:, static_communities == s].mean(1) for s in structs]
int_an_struct = [int_an[:, static_communities == s].mean(1) for s in structs]
int_hc_struct = [int_hc[:, static_communities == s].mean(1) for s in structs]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
x = np.arange(len(structs))
w = 0.35

for ax, an_vals, hc_vals, ylabel in zip(
    axes,
    [rec_an_struct, int_an_struct],
    [rec_hc_struct, int_hc_struct],
    ["Recruitment", "Integration"],
):
    an_m = np.array([v.mean() for v in an_vals])
    hc_m = np.array([v.mean() for v in hc_vals])
    an_e = np.array([v.std(ddof=1) for v in an_vals])
    hc_e = np.array([v.std(ddof=1) for v in hc_vals])

    ax.bar(x - w/2, an_m, w, yerr=an_e, color=PALETTE[0], alpha=0.85, label="AN", capsize=4)
    ax.bar(x + w/2, hc_m, w, yerr=hc_e, color=PALETTE[1], alpha=0.85, label="HC", capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(struct_labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f"{ylabel} by Subcortical Structure", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(an_m.max(), hc_m.max()) * 1.25)

plt.suptitle("Subcortical Recruitment & Integration — Tian S2 (5+5)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "barplot_rec_int_by_structure.png", dpi=300, bbox_inches="tight")
plt.close(fig)
log.info("  Saved barplot_rec_int_by_structure.png")

log.info("[Fig 3] Flexibility & Promiscuity violin plots …")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
rng = np.random.default_rng(42)

for ax, an_vals, hc_vals, label in zip(
    axes,
    [flex_an, prom_an],
    [flex_hc, prom_hc],
    ["Flexibility", "Promiscuity"],
):
    parts = ax.violinplot([an_vals, hc_vals], positions=[0, 1],
                          showmeans=True, showmedians=False)
    for pc, col in zip(parts["bodies"], PALETTE):
        pc.set_facecolor(col); pc.set_alpha(0.7)
    for part in ("cmeans","cbars","cmins","cmaxes"):
        if part in parts: parts[part].set_color("black")

    for pos, vals, col in zip([0, 1], [an_vals, hc_vals], PALETTE):
        jitter = rng.uniform(-0.07, 0.07, size=len(vals))
        ax.scatter(pos + jitter, vals, color=col, s=50, zorder=3,
                   alpha=0.9, edgecolors="k", linewidths=0.5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["AN (n=5)", "HC (n=5)"], fontsize=12)
    ax.set_ylabel(label, fontsize=13)
    ax.set_title(f"{label}: AN vs HC", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.5, 1.5)

plt.suptitle("Subcortical Dynamic Measures — Tian S2 (5+5)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "flexibility_promiscuity_subcortical.png", dpi=300, bbox_inches="tight")
plt.close(fig)
log.info("  Saved flexibility_promiscuity_subcortical.png")

log.info("[Fig 4] Nodal recruitment & integration heatmap …")
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax, an_vals, hc_vals, title in zip(
    axes,
    [rec_an, int_an],
    [rec_hc, int_hc],
    ["Recruitment (AN top, HC bottom)", "Integration (AN top, HC bottom)"],
):
    combined = np.vstack([an_vals, hc_vals])
    divider  = N_SUBJ - 0.5
    im = ax.imshow(combined, aspect="auto", cmap="viridis",
                   vmin=combined.min(), vmax=combined.max())
    ax.axhline(divider, color="white", lw=2)
    ax.set_yticks(list(range(N_SUBJ*2)))
    ax.set_yticklabels([f"AN{i+1}" for i in range(N_SUBJ)] +
                       [f"HC{i+1}" for i in range(N_SUBJ)], fontsize=9)
    ax.set_xticks(range(N_REGIONS))
    ax.set_xticklabels(tian_labels, rotation=90, fontsize=6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Value")

    # Structure boundary lines
    cur = 0
    for sid in sorted(STRUCT_MAP.values()):
        n = (static_communities == sid).sum()
        ax.axvline(cur + n - 0.5, color="white", lw=0.8)
        cur += n

plt.suptitle("Per-Subject Nodal Measures — Subcortical 32 ROIs (5+5)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "nodal_heatmap_rec_int_subcortical.png", dpi=300, bbox_inches="tight")
plt.close(fig)
log.info("  Saved nodal_heatmap_rec_int_subcortical.png")

log.info("[Fig 5] Allegiance difference bubble plot …")
sig_mask = p_alg < 0.05
xx, yy = np.meshgrid(range(N_REGIONS), range(N_REGIONS))
sig_pts = sig_mask & (np.abs(alg_diff) > 0.01)

fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(alg_diff, cmap="RdBu_r",
               vmin=-np.abs(alg_diff).max(), vmax=np.abs(alg_diff).max(),
               aspect="auto")
if sig_pts.any():
    sx, sy = xx[sig_pts], yy[sig_pts]
    sizes = np.abs(alg_diff[sig_pts]) * 600
    ax.scatter(sx, sy, s=sizes, facecolors="none", edgecolors="black", lw=0.8, alpha=0.8)

struct_boundary_lines(ax)
plt.colorbar(im, ax=ax, shrink=0.75, label="Allegiance diff (AN − HC)")
ax.set_xticks(range(N_REGIONS))
ax.set_xticklabels(tian_labels, rotation=90, fontsize=6)
ax.set_yticks(range(N_REGIONS))
ax.set_yticklabels(tian_labels, fontsize=6)
ax.set_title("Subcortical Allegiance Difference (AN − HC)\nCircles = p<0.05",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "allegiance_diff_subcortical.png", dpi=300, bbox_inches="tight")
plt.close(fig)
log.info("  Saved allegiance_diff_subcortical.png")

log.info("[Fig 6] Global summary bar chart …")
measures = summary["Measure"].tolist()
an_means = summary["AN_mean"].tolist()
hc_means = summary["HC_mean"].tolist()

x = np.arange(len(measures))
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - 0.2, an_means, 0.4, color=PALETTE[0], alpha=0.85, label="AN (n=5)")
ax.bar(x + 0.2, hc_means, 0.4, color=PALETTE[1], alpha=0.85, label="HC (n=5)")
ax.set_xticks(x)
ax.set_xticklabels(measures, fontsize=12)
ax.set_ylabel("Mean value", fontsize=13)
ax.set_title("Subcortical Global Measures: AN vs HC (Tian S2, 5+5)", fontsize=14, fontweight="bold")
ax.legend(fontsize=12)
plt.tight_layout()
fig.savefig(FIG_DIR / "global_summary_subcortical.png", dpi=300, bbox_inches="tight")
plt.close(fig)
log.info("  Saved global_summary_subcortical.png")

log.info("Saving comprehensive Excel workbook …")
excel_path = STAT_DIR / "subcortical_results_comprehensive.xlsx"
with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    df_nodal.to_excel(writer, sheet_name="Nodal_Statistics", index=False)
    summary.to_excel(writer, sheet_name="Global_Tests", index=False)
    pd.DataFrame(alg_an, index=tian_labels, columns=tian_labels).to_excel(writer, sheet_name="Allegiance_AN")
    pd.DataFrame(alg_hc, index=tian_labels, columns=tian_labels).to_excel(writer, sheet_name="Allegiance_HC")
    pd.DataFrame(alg_diff, index=tian_labels, columns=tian_labels).to_excel(writer, sheet_name="Allegiance_Diff")
log.info(f"  Saved {excel_path.name}")

log.info(f"Complete — duration: {datetime.now() - start_time}")
