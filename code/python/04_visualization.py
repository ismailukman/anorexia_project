#!/usr/bin/env python3
"""
04_visualization_results_v2.py — Figures and result export for Anorexia vs Healthy Control.
Requires 03_statistical_analysis_v2.py outputs. Reduced subset: 5 subjects per group.
Atlas: Schaefer-2018, 200 parcels, Yeo-7 networks.
"""

import os
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (saves to file only)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import h5py

from nilearn import datasets, plotting

start_time = datetime.now()
warnings.filterwarnings("ignore", category=FutureWarning)
sns.set(style="white")

PROJECT_ROOT = Path("/Users/ismaila/Documents/C-Codes/AnorexiaProject")
STAT_DIR     = PROJECT_ROOT / "output" / "results" / "statistical_results_v2"
MLCD_DIR     = PROJECT_ROOT / "data" / "analysis" / "mlcd_subjs" / "subjs_mlcd"
FIG_DIR      = PROJECT_ROOT / "output" / "figures" / "stage4_viz_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR      = STAT_DIR

N_SUBJ_PER_GROUP = 5
N_SUBJ_TOTAL_MAT = 22

NETWORKS      = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NETWORK_SHORT = ["VN", "SMN", "DAN", "VAN", "LN", "FPN", "DMN"]

YEO7_COLORS = [
    "#A251AC", "#789AC1", "#409832", "#E165FE",
    "#F6FDC9", "#EFB944", "#D9717D",
]
PALETTE_2G = ["#FF6F61", "#1E90FF"]   # Anorexia, HC

GROUP_LABELS = ["Anorexia", "Healthy Control"]

NETWORK_BOUNDARIES = [
    (-0.5,  13.5), (13.5,  29.5), (29.5,  42.5), (42.5,  53.5),
    (53.5,  59.5), (59.5,  72.5), (72.5,  99.5),
    (99.5, 114.5), (114.5, 133.5), (133.5, 146.5), (146.5, 157.5),
    (157.5, 163.5), (163.5, 180.5), (180.5, 199.5),
]

ALPHA = 0.05

atlas       = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=1)
labels_200  = np.array(atlas.labels[1:]).astype("U")

static_communities = np.zeros((200,), dtype=int)
for i, network in enumerate(NETWORKS):
    idx = np.array([network in s for s in labels_200], dtype=bool)
    static_communities[idx] = i + 1

try:
    node_coords = np.array(atlas.region_coords)
except (AttributeError, KeyError):
    node_coords = plotting.find_parcellation_cut_coords(atlas.maps)

_node_color_map = {
    1: "purple", 2: "blue", 3: "green", 4: "violet",
    5: "moccasin", 6: "orange", 7: "red",
}
node_colors = [_node_color_map.get(static_communities[i], "grey") for i in range(200)]


print(f"Loading pre-computed statistical results (v2 — {N_SUBJ_PER_GROUP} subj/group) …")

def _load(name):
    p = STAT_DIR / name
    return np.load(p, allow_pickle=True)


alg_an = _load("allegiance_group_an.npy")
alg_hc = _load("allegiance_group_hc.npy")

alg_coarse_an = _load("allegiance_coarse_an.npy")
alg_coarse_hc = _load("allegiance_coarse_hc.npy")

rec_an = _load("recruitment_group_an.npy")
rec_hc = _load("recruitment_group_hc.npy")
int_an = _load("integration_group_an.npy")
int_hc = _load("integration_group_hc.npy")

pval_alg_fine   = _load("pvalue_allegiance_fine.npy")
pval_alg_coarse = _load("pvalue_allegiance_coarse.npy")
pval_rec        = _load("pvalue_recruitment_nodal.npy")
pval_int        = _load("pvalue_integration_nodal.npy")
pval_rec_coarse = _load("pvalue_recruitment_coarse.npy")
pval_int_coarse = _load("pvalue_integration_coarse.npy")

rec_an_subjs = _load("recruitment_subj_an.npy")
rec_hc_subjs = _load("recruitment_subj_hc.npy")
int_an_subjs = _load("integration_subj_an.npy")
int_hc_subjs = _load("integration_subj_hc.npy")

df_nodal = pd.read_csv(STAT_DIR / "nodal_statistics_200.csv")

print("  All results loaded successfully.")


def draw_network_color_bars(ax):
    """Add Yeo-7 network colour patches on top & right of a 200×200 matshow."""
    colors_14 = YEO7_COLORS * 2  # LH + RH
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    h     = (ymax - ymin) / 30.0
    space = h / 5.0
    w     = (ymax - ymin) / 30.0

    marker_y = ymax + space
    for (start, end), c in zip(NETWORK_BOUNDARIES, colors_14):
        ax.add_patch(patches.Rectangle(
            (start, marker_y), end - start, h,
            facecolor=c, clip_on=False, linewidth=1.5, edgecolor="k",
        ))
    marker_x = ymax
    for (start, end), c in zip(NETWORK_BOUNDARIES, colors_14):
        ax.add_patch(patches.Rectangle(
            (marker_x + space, start + 0.5), w, end - start,
            facecolor=c, clip_on=False, linewidth=1.5, edgecolor="k",
        ))


def _common_matrix_style(ax):
    """Shared axis style for 200×200 matrices."""
    tick_pos = [13.5, 29.5, 42.5, 53.5, 59.5, 72.5, 99.5,
                114.5, 133.5, 146.5, 157.5, 163.5, 180.5]
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.axvline(x=99.5, color="white", linewidth=3)
    ax.axhline(y=99.5, color="white", linewidth=3)
    ax.grid(color="white", linestyle="-", linewidth=0.7)
    ax.tick_params(axis="both", which="both", bottom=False, top=False,
                   left=False, right=False, labeltop=False, labelleft=False)


print("\n[Fig 1] Fine-level allegiance matrices (AN, HC, p-value, thresholded)")

cmap_alg = "jet"

fig, axes = plt.subplots(2, 2, figsize=(28, 24))

panels = [
    (axes[0, 0], alg_an, "Anorexia — Allegiance",      cmap_alg, 0, 1),
    (axes[0, 1], alg_hc, "Healthy Control — Allegiance", cmap_alg, 0, 1),
    (axes[1, 0], pval_alg_fine, "P-value (AN vs HC)",    "hot",    0, 1),
]

for ax, mat, title, cmap, vmin, vmax in panels:
    im = ax.matshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title, fontsize=22, pad=40)
    _common_matrix_style(ax)
    cb = fig.colorbar(im, ax=ax, shrink=0.65)
    cb.ax.tick_params(labelsize=14)
    draw_network_color_bars(ax)

# Panel 4: thresholded (significant only)
thresh_mat = np.where(pval_alg_fine < ALPHA, np.abs(alg_an - alg_hc), np.nan)
ax4 = axes[1, 1]
im4 = ax4.matshow(thresh_mat, cmap="hot", vmin=0, vmax=0.3)
ax4.set_title(f"Significant Difference (p<{ALPHA})", fontsize=22, pad=40)
_common_matrix_style(ax4)
cb4 = fig.colorbar(im4, ax=ax4, shrink=0.65)
cb4.ax.tick_params(labelsize=14)
draw_network_color_bars(ax4)

plt.tight_layout(pad=3)
fig.savefig(FIG_DIR / "allegiance_fine_200x200.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("  Saved allegiance_fine_200x200.png")


print("\n[Fig 2] Coarse allegiance (Functional Cartography)")

cmap_coarse = "jet"
sig_cells = np.argwhere(pval_alg_coarse < ALPHA)
hatch_props = dict(color="black", alpha=1.0, linewidth=1.5, hatch="///")
diff_coarse = np.abs(alg_coarse_an - alg_coarse_hc)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

im1 = axes[0].imshow(alg_coarse_an, vmin=0, vmax=1, cmap=cmap_coarse)
axes[0].set_title("Anorexia", fontsize=18)
axes[0].set_yticks(range(7))
axes[0].set_yticklabels(NETWORK_SHORT, fontsize=12)
axes[0].set_xticks(range(7))
axes[0].set_xticklabels(NETWORK_SHORT, fontsize=10, rotation=45)
axes[0].tick_params(left=False, right=False, bottom=False, top=False,
                    labelleft=True, labelbottom=True, labeltop=False)
fig.colorbar(im1, ax=axes[0], shrink=0.6)

im2 = axes[1].imshow(alg_coarse_hc, vmin=0, vmax=1, cmap=cmap_coarse)
axes[1].set_title("Healthy Control", fontsize=18)
axes[1].set_yticks(range(7))
axes[1].set_yticklabels(NETWORK_SHORT, fontsize=12)
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(NETWORK_SHORT, fontsize=10, rotation=45)
axes[1].tick_params(left=False, right=False, bottom=False, top=False,
                    labelleft=False, labelbottom=True, labeltop=False)
fig.colorbar(im2, ax=axes[1], shrink=0.6)

im3 = axes[2].imshow(diff_coarse, vmax=0.5, cmap="binary")
axes[2].set_title("Absolute Difference (AN − HC)", fontsize=18)
axes[2].set_yticks(range(7))
axes[2].set_yticklabels(NETWORK_SHORT, fontsize=12)
axes[2].set_xticks(range(7))
axes[2].set_xticklabels(NETWORK_SHORT, fontsize=10, rotation=45)
axes[2].tick_params(left=False, right=False, bottom=False, top=False,
                    labelleft=False, labelbottom=True, labeltop=False)
for cell in sig_cells:
    axes[2].add_patch(Rectangle(
        (cell[1] - 0.5, cell[0] - 0.5), 1, 1, fill=False, **hatch_props
    ))
fig.colorbar(im3, ax=axes[2], shrink=0.6)

for row in range(7):
    for col in range(7):
        for k, ax_k in enumerate(axes[:2]):
            val = [alg_coarse_an, alg_coarse_hc][k][row, col]
            ax_k.text(col, row, f"{val:.2f}", ha="center", va="center",
                      fontsize=8, color="white" if val > 0.5 else "black")
        val_d = diff_coarse[row, col]
        axes[2].text(col, row, f"{val_d:.3f}", ha="center", va="center",
                     fontsize=8, color="white" if val_d > 0.25 else "black")

plt.tight_layout(pad=3)
fig.savefig(FIG_DIR / "allegiance_coarse_functional_cartography.png",
            bbox_inches="tight", dpi=300)
plt.close(fig)
print("  Saved allegiance_coarse_functional_cartography.png")


print("\n[Fig 3] Circos / chord diagrams")

try:
    from mne.viz import plot_connectivity_circle

    palette_circos = YEO7_COLORS
    node_names     = NETWORK_SHORT
    cmap_circos    = "hot_r"

    for grp_idx, (con, grp_name) in enumerate(
        zip([alg_coarse_an, alg_coarse_hc], GROUP_LABELS)
    ):
        fig_c = plt.figure(num=None, figsize=(8, 8), facecolor="white")
        plot_connectivity_circle(
            con, node_names,
            title=grp_name,
            facecolor="white", textcolor="black",
            colormap=cmap_circos, vmin=0, vmax=0.6,
            colorbar=True, colorbar_size=0.5, colorbar_pos=(-0.6, 0.5),
            node_width=None, node_colors=palette_circos,
            linewidth=7, fontsize_names=10,
            fig=fig_c,
        )
        fname = f"circos_{grp_name.replace(' ', '_').lower()}.png"
        fig_c.savefig(FIG_DIR / fname, facecolor="white",
                      bbox_inches="tight", dpi=300)
        plt.close(fig_c)
        print(f"  Saved {fname}")

    # Difference circos
    diff_con = alg_coarse_an - alg_coarse_hc
    fig_d = plt.figure(num=None, figsize=(8, 8), facecolor="white")
    plot_connectivity_circle(
        diff_con, node_names,
        title="Difference (AN − HC)",
        facecolor="white", textcolor="black",
        colormap="bwr", vmin=-0.3, vmax=0.3,
        colorbar=True, colorbar_size=0.5, colorbar_pos=(-0.6, 0.5),
        node_width=None, node_colors=palette_circos,
        linewidth=7, fontsize_names=10,
        fig=fig_d,
    )
    fig_d.savefig(FIG_DIR / "circos_difference_AN_HC.png", facecolor="white",
                  bbox_inches="tight", dpi=300)
    plt.close(fig_d)
    print("  Saved circos_difference_AN_HC.png")

except ImportError:
    print("  [SKIP] mne not installed — circos diagrams skipped.")


print("\n[Fig 4] Directionality scatter plots: HC vs AN")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

sns.regplot(x=rec_hc, y=rec_an, ci=95,
            scatter_kws={"color": "black", "s": 60, "alpha": 0.6},
            line_kws={"color": "red", "label": "Regression line"},
            ax=ax1)
ax1.set_xlabel("Healthy Control", fontsize=18)
ax1.set_ylabel("Anorexia", fontsize=18)
ax1.set_title("Recruitment", fontsize=22)
lims_r = [min(ax1.get_xlim()[0], ax1.get_ylim()[0]),
          max(ax1.get_xlim()[1], ax1.get_ylim()[1])]
ax1.plot(lims_r, lims_r, "k--", alpha=0.8, zorder=0, label="Identity line", linewidth=2)
ax1.set_xlim(lims_r); ax1.set_ylim(lims_r)
ax1.set_aspect("equal", adjustable="box")
ax1.legend(fontsize=14, loc="best")
ax1.tick_params(labelsize=14)

sns.regplot(x=int_hc, y=int_an, ci=95,
            scatter_kws={"color": "black", "s": 60, "alpha": 0.6},
            line_kws={"color": "red", "label": "Regression line"},
            ax=ax2)
ax2.set_xlabel("Healthy Control", fontsize=18)
ax2.set_ylabel("Anorexia", fontsize=18)
ax2.set_title("Integration", fontsize=22)
lims_i = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]),
          max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
ax2.plot(lims_i, lims_i, "k--", alpha=0.8, zorder=0, label="Identity line", linewidth=2)
ax2.set_xlim(lims_i); ax2.set_ylim(lims_i)
ax2.set_aspect("equal", adjustable="box")
ax2.legend(fontsize=14, loc="best")
ax2.tick_params(labelsize=14)

for ax in [ax1, ax2]:
    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_linewidth(2)

plt.tight_layout(pad=3)
fig.savefig(FIG_DIR / "directionality_scatter_rec_int.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("  Saved directionality_scatter_rec_int.png")


print("\n[Fig 5] Subject-level directionality scatter")

mean_rec_an = np.mean(rec_an_subjs, axis=1)
mean_rec_hc = np.mean(rec_hc_subjs, axis=1)
mean_int_an = np.mean(int_an_subjs, axis=1)
mean_int_hc = np.mean(int_hc_subjs, axis=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

ax1.scatter(mean_rec_hc, mean_rec_an, s=100, c="black", alpha=0.7)
ax1.set_xlabel("HC Mean Recruitment", fontsize=16)
ax1.set_ylabel("AN Mean Recruitment", fontsize=16)
ax1.set_title("Subject-Level Recruitment", fontsize=18)
lims = [min(mean_rec_hc.min(), mean_rec_an.min()) - 0.02,
        max(mean_rec_hc.max(), mean_rec_an.max()) + 0.02]
ax1.plot(lims, lims, "k--", alpha=0.6, label="Identity")
ax1.set_xlim(lims); ax1.set_ylim(lims)
ax1.set_aspect("equal")
ax1.legend(fontsize=14)
ax1.tick_params(labelsize=12)

ax2.scatter(mean_int_hc, mean_int_an, s=100, c="black", alpha=0.7)
ax2.set_xlabel("HC Mean Integration", fontsize=16)
ax2.set_ylabel("AN Mean Integration", fontsize=16)
ax2.set_title("Subject-Level Integration", fontsize=18)
lims2 = [min(mean_int_hc.min(), mean_int_an.min()) - 0.02,
         max(mean_int_hc.max(), mean_int_an.max()) + 0.02]
ax2.plot(lims2, lims2, "k--", alpha=0.6, label="Identity")
ax2.set_xlim(lims2); ax2.set_ylim(lims2)
ax2.set_aspect("equal")
ax2.legend(fontsize=14)
ax2.tick_params(labelsize=12)

plt.tight_layout(pad=2)
fig.savefig(FIG_DIR / "directionality_subject_level.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("  Saved directionality_subject_level.png")


print("\n[Fig 6] Coarse barplots: recruitment & integration by network")

rec_coarse_an = np.diag(alg_coarse_an)
rec_coarse_hc = np.diag(alg_coarse_hc)
int_coarse_an = (alg_coarse_an.sum(1) - np.diag(alg_coarse_an)) / (alg_coarse_an.shape[1] - 1)
int_coarse_hc = (alg_coarse_hc.sum(1) - np.diag(alg_coarse_hc)) / (alg_coarse_hc.shape[1] - 1)

data = np.concatenate([rec_coarse_an, rec_coarse_hc, int_coarse_an, int_coarse_hc])
df_cat = pd.DataFrame({"Values": data})
df_cat["Metric"]  = np.repeat(["Recruitment", "Integration"], 14)
df_cat["Network"] = np.tile(NETWORK_SHORT, 4)
df_cat["Group"]   = np.tile(np.repeat(GROUP_LABELS, 7), 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(x="Network", y="Values", hue="Group", ax=ax1,
            data=df_cat.loc[df_cat["Metric"] == "Recruitment"],
            palette=PALETTE_2G, alpha=0.85)
ax1.set_title("Recruitment", fontsize=16)
ax1.set_xlabel(None); ax1.set_ylabel(None)
ax1.legend(fontsize=11, loc="upper center")
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

for net_idx, net in enumerate(NETWORK_SHORT):
    if pval_rec_coarse[net_idx] < ALPHA:
        y_max = max(rec_coarse_an[net_idx], rec_coarse_hc[net_idx])
        ax1.text(net_idx, y_max + 0.01, "*", ha="center", fontsize=16, color="red")

sns.barplot(x="Network", y="Values", hue="Group", ax=ax2,
            data=df_cat.loc[df_cat["Metric"] == "Integration"],
            palette=PALETTE_2G, alpha=0.85)
ax2.set_title("Integration", fontsize=16)
ax2.set_xlabel(None); ax2.set_ylabel(None)
ax2.get_legend().remove()
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

for net_idx, net in enumerate(NETWORK_SHORT):
    if pval_int_coarse[net_idx] < ALPHA:
        y_max = max(int_coarse_an[net_idx], int_coarse_hc[net_idx])
        ax2.text(net_idx, y_max + 0.005, "*", ha="center", fontsize=16, color="red")

plt.tight_layout()
fig.savefig(FIG_DIR / "catplot_rec_int_by_network.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("  Saved catplot_rec_int_by_network.png")


print("\n[Fig 7] Brain glass-brain maps (significant regions)")

diff_rec = np.abs(rec_an - rec_hc)
diff_int = np.abs(int_an - int_hc)

sig_rec_mask = pval_rec < ALPHA
sig_rec_idx  = np.where(sig_rec_mask)[0]

if len(sig_rec_idx) > 0:
    coords_rec  = [tuple(node_coords[i]) for i in sig_rec_idx]
    colors_rec  = [node_colors[i] for i in sig_rec_idx]
    size_rec    = [max(int(diff_rec[i] * 200), 5) for i in sig_rec_idx]

    try:
        view_rec = plotting.view_markers(coords_rec, colors_rec, marker_size=np.array(size_rec))
        view_rec.save_as_html(str(FIG_DIR / "brain_glass_recruitment_sig.html"))
        print(f"  Saved brain_glass_recruitment_sig.html ({len(sig_rec_idx)} regions)")
    except Exception as e:
        print(f"  [WARN] Brain glass recruitment failed: {e}")

    # Static glass-brain plot
    fig_brain, (bax1, bax2) = plt.subplots(2, 1, figsize=(10, 6))
    plotting.plot_markers(rec_an, node_coords, node_cmap="jet", title="Recruitment — Anorexia",
                          colorbar=True, axes=bax1)
    plotting.plot_markers(rec_hc, node_coords, node_cmap="jet", title="Recruitment — HC",
                          colorbar=True, axes=bax2)
    fig_brain.savefig(FIG_DIR / "brain_recruitment_AN_HC.png", bbox_inches="tight", dpi=300)
    plt.close(fig_brain)
    print("  Saved brain_recruitment_AN_HC.png")
else:
    print("  No significant recruitment regions at p<0.05")

sig_int_mask = pval_int < ALPHA
sig_int_idx  = np.where(sig_int_mask)[0]

if len(sig_int_idx) > 0:
    coords_int  = [tuple(node_coords[i]) for i in sig_int_idx]
    colors_int  = [node_colors[i] for i in sig_int_idx]
    size_int    = [max(int(diff_int[i] * 300), 5) for i in sig_int_idx]

    try:
        view_int = plotting.view_markers(coords_int, colors_int, marker_size=np.array(size_int))
        view_int.save_as_html(str(FIG_DIR / "brain_glass_integration_sig.html"))
        print(f"  Saved brain_glass_integration_sig.html ({len(sig_int_idx)} regions)")
    except Exception as e:
        print(f"  [WARN] Brain glass integration failed: {e}")

    fig_brain2, (bax1, bax2) = plt.subplots(2, 1, figsize=(10, 6))
    plotting.plot_markers(int_an, node_coords, node_cmap="jet", title="Integration — Anorexia",
                          colorbar=True, axes=bax1)
    plotting.plot_markers(int_hc, node_coords, node_cmap="jet", title="Integration — HC",
                          colorbar=True, axes=bax2)
    fig_brain2.savefig(FIG_DIR / "brain_integration_AN_HC.png", bbox_inches="tight", dpi=300)
    plt.close(fig_brain2)
    print("  Saved brain_integration_AN_HC.png")
else:
    print("  No significant integration regions at p<0.05")

for measure_name, vals_an, vals_hc in [
    ("recruitment", rec_an, rec_hc),
    ("integration", int_an, int_hc),
]:
    fig_diff, ax_diff = plt.subplots(1, 1, figsize=(10, 4))
    plotting.plot_markers(
        vals_an - vals_hc, node_coords, node_cmap="bwr",
        title=f"{measure_name.capitalize()} Difference (AN − HC)",
        colorbar=True, axes=ax_diff,
    )
    fig_diff.savefig(FIG_DIR / f"brain_{measure_name}_diff_AN_HC.png",
                     bbox_inches="tight", dpi=300)
    plt.close(fig_diff)
    print(f"  Saved brain_{measure_name}_diff_AN_HC.png")


print("\n[Fig 8] Distribution plots (recruitment & integration)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sns.histplot(rec_an, kde=True, ax=ax1, alpha=0.5, color=PALETTE_2G[0], label="Anorexia")
sns.histplot(rec_hc, kde=True, ax=ax1, alpha=0.5, color=PALETTE_2G[1], label="Healthy Control")
ax1.set_title("Distribution of Recruitment Values", fontsize=16)
ax1.set_xlabel("Recruitment Coefficient", fontsize=14)
ax1.set_ylabel("Frequency", fontsize=14)
ax1.legend(fontsize=12)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

sns.histplot(int_an, kde=True, ax=ax2, alpha=0.5, color=PALETTE_2G[0], label="Anorexia")
sns.histplot(int_hc, kde=True, ax=ax2, alpha=0.5, color=PALETTE_2G[1], label="Healthy Control")
ax2.set_title("Distribution of Integration Values", fontsize=16)
ax2.set_xlabel("Integration Coefficient", fontsize=14)
ax2.set_ylabel("Frequency", fontsize=14)
ax2.legend(fontsize=12)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

plt.tight_layout(pad=2)
fig.savefig(FIG_DIR / "distribution_rec_int.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("  Saved distribution_rec_int.png")


print("\n[Fig 9] Subject-level violin plots (recruitment & integration)")

# Build long-form DataFrame for seaborn
rows = []
for s in range(N_SUBJ_PER_GROUP):
    rows.append({"Group": "Anorexia", "Measure": "Recruitment", "Value": np.mean(rec_an_subjs[s])})
    rows.append({"Group": "Anorexia", "Measure": "Integration", "Value": np.mean(int_an_subjs[s])})
for s in range(N_SUBJ_PER_GROUP):
    rows.append({"Group": "Healthy Control", "Measure": "Recruitment", "Value": np.mean(rec_hc_subjs[s])})
    rows.append({"Group": "Healthy Control", "Measure": "Integration", "Value": np.mean(int_hc_subjs[s])})

df_violin = pd.DataFrame(rows)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


sns.violinplot(x="Group", y="Value", data=df_violin[df_violin["Measure"] == "Recruitment"],
               palette=PALETTE_2G, inner="box", ax=ax1, alpha=0.7)
sns.stripplot(x="Group", y="Value", data=df_violin[df_violin["Measure"] == "Recruitment"],
              color="black", size=5, alpha=0.5, ax=ax1, jitter=True)
ax1.set_title("Mean Recruitment per Subject", fontsize=16)
ax1.set_xlabel(None); ax1.set_ylabel("Mean Recruitment", fontsize=14)

sns.violinplot(x="Group", y="Value", data=df_violin[df_violin["Measure"] == "Integration"],
               palette=PALETTE_2G, inner="box", ax=ax2, alpha=0.7)
sns.stripplot(x="Group", y="Value", data=df_violin[df_violin["Measure"] == "Integration"],
              color="black", size=5, alpha=0.5, ax=ax2, jitter=True)
ax2.set_title("Mean Integration per Subject", fontsize=16)
ax2.set_xlabel(None); ax2.set_ylabel("Mean Integration", fontsize=14)

plt.tight_layout()
fig.savefig(FIG_DIR / "violin_rec_int_subject_level.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("  Saved violin_rec_int_subject_level.png")


print("\n[Fig 10] P-value matrix with network colour bars")

cmap_pval = "jet"
fig_pval = plt.figure(figsize=(15, 11))
ax_pval = fig_pval.add_subplot(111)
im_pval = ax_pval.matshow(pval_alg_fine, vmin=0, vmax=1, cmap=cmap_pval)
ax_pval.set_title("P-value (Anorexia vs Healthy Control)", fontsize=20, pad=40)
_common_matrix_style(ax_pval)
cb_pval = fig_pval.colorbar(im_pval, shrink=0.75)
cb_pval.ax.tick_params(labelsize=14)
draw_network_color_bars(ax_pval)

plt.tight_layout()
fig_pval.savefig(FIG_DIR / "pvalue_matrix_200x200.png", bbox_inches="tight", dpi=300)
plt.close(fig_pval)
print("  Saved pvalue_matrix_200x200.png")


print("\n[Fig 11] Cohen's d brain maps")

if "Cohens_d_Rec" in df_nodal.columns:
    d_rec = df_nodal["Cohens_d_Rec"].values
    d_int = df_nodal["Cohens_d_Int"].values

    fig_cd, (cdax1, cdax2) = plt.subplots(2, 1, figsize=(10, 6))
    plotting.plot_markers(d_rec, node_coords, node_cmap="bwr",
                          title="Cohen's d — Recruitment (AN − HC)", colorbar=True, axes=cdax1)
    plotting.plot_markers(d_int, node_coords, node_cmap="bwr",
                          title="Cohen's d — Integration (AN − HC)", colorbar=True, axes=cdax2)
    fig_cd.savefig(FIG_DIR / "cohens_d_brain_maps.png", bbox_inches="tight", dpi=300)
    plt.close(fig_cd)
    print("  Saved cohens_d_brain_maps.png")


print("\n[Fig 12] Per-subject community label changes across time windows")

_VAR_BASE = "N_all_g"


def _load_community_labels(tag):
    """Load first N_SUBJ_PER_GROUP subjects' community label matrices (200 × W)."""
    mat_path = MLCD_DIR / f"mlcd_{tag}_wins.mat"
    candidates = [f"{_VAR_BASE}_{tag}", f"{_VAR_BASE}{tag}", f"{_VAR_BASE}__{tag}"]
    with h5py.File(mat_path, "r") as f:
        keys = set(f.keys())
        mat = None
        for name in candidates:
            if name in keys:
                mat = np.squeeze(np.asarray(f[name][()]))
                break
        if mat is None:
            raise KeyError(f"None of {candidates} found in {mat_path}. Keys: {list(f.keys())}")
    # Ensure shape (200, total_windows)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D, got {mat.shape}")
    if mat.shape[0] != 200 and mat.shape[1] == 200:
        mat = mat.T
    wins_per_subj = mat.shape[1] // N_SUBJ_TOTAL_MAT
    # Select first N_SUBJ_PER_GROUP subjects only
    subjects = [
        mat[:, i * wins_per_subj : (i + 1) * wins_per_subj]
        for i in range(N_SUBJ_PER_GROUP)
    ]
    return subjects, wins_per_subj


try:
    comm_an, W = _load_community_labels("anorexia")
    comm_hc, _ = _load_community_labels("control")

    n_comm = int(np.nanmax([np.nanmax(comm_an[0]), np.nanmax(comm_hc[0])]))

    def community_proportions(subj_list, n_windows, n_communities):
        """Return (n_windows, n_communities) array of mean region proportions."""
        props = np.zeros((n_windows, n_communities))
        for subj_mat in subj_list:
            for w in range(n_windows):
                col = subj_mat[:, w]
                valid = col[~np.isnan(col)].astype(int)
                for c in range(1, n_communities + 1):
                    props[w, c - 1] += np.sum(valid == c)
        props /= (200.0 * len(subj_list))        # normalise to proportions
        return props

    props_an = community_proportions(comm_an, W, n_comm)
    props_hc = community_proportions(comm_hc, W, n_comm)

    _base_colors = YEO7_COLORS
    comm_colors = [_base_colors[i % len(_base_colors)] for i in range(n_comm)]
    comm_labels = [f"C{i+1}" for i in range(n_comm)]
    x = np.arange(W)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    ax1.stackplot(x, props_an.T, labels=comm_labels,
                  colors=comm_colors, alpha=0.85)
    ax1.set_ylabel("Proportion of Regions", fontsize=14)
    ax1.set_title(f"Anorexia — Community Membership Across Time Windows (n={N_SUBJ_PER_GROUP})", fontsize=16)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper right", fontsize=10, ncol=min(n_comm, 7))

    ax2.stackplot(x, props_hc.T, labels=comm_labels,
                  colors=comm_colors, alpha=0.85)
    ax2.set_xlabel("Time Window", fontsize=14)
    ax2.set_ylabel("Proportion of Regions", fontsize=14)
    ax2.set_title(f"Healthy Control — Community Membership Across Time Windows (n={N_SUBJ_PER_GROUP})", fontsize=16)
    ax2.set_ylim(0, 1)

    plt.tight_layout(pad=2)
    fig.savefig(FIG_DIR / "community_label_changes_across_time.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Saved community_label_changes_across_time.png")

    def compute_flexibility(subj_list, n_windows):
        """Return (n_subj, 200) flexibility array."""
        flex = np.zeros((len(subj_list), 200))
        for s, mat in enumerate(subj_list):
            for r in range(200):
                vals = mat[r, :]
                valid = vals[~np.isnan(vals)]
                if len(valid) > 1:
                    changes = np.sum(valid[1:] != valid[:-1])
                    flex[s, r] = changes / (len(valid) - 1)
        return flex

    flex_an = compute_flexibility(comm_an, W)
    flex_hc = compute_flexibility(comm_hc, W)

    fig2, (hax1, hax2) = plt.subplots(1, 2, figsize=(18, 5), sharey=True)

    im1 = hax1.imshow(flex_an, aspect="auto", cmap="hot", vmin=0, vmax=1)
    hax1.set_title(f"Anorexia — Node Flexibility (n={N_SUBJ_PER_GROUP})", fontsize=14)
    hax1.set_xlabel("Brain Region (Schaefer-200)", fontsize=12)
    hax1.set_ylabel("Subject", fontsize=12)
    hax1.set_yticks(range(N_SUBJ_PER_GROUP))
    hax1.set_yticklabels([f"S{i+1}" for i in range(N_SUBJ_PER_GROUP)])

    im2 = hax2.imshow(flex_hc, aspect="auto", cmap="hot", vmin=0, vmax=1)
    hax2.set_title(f"Healthy Control — Node Flexibility (n={N_SUBJ_PER_GROUP})", fontsize=14)
    hax2.set_xlabel("Brain Region (Schaefer-200)", fontsize=12)

    fig2.colorbar(im2, ax=[hax1, hax2], shrink=0.6, label="Flexibility")
    plt.tight_layout(pad=2)
    fig2.savefig(FIG_DIR / "node_flexibility_heatmap.png",
                 bbox_inches="tight", dpi=300)
    plt.close(fig2)
    print("  Saved node_flexibility_heatmap.png")

except Exception as e:
    print(f"  [WARN] Community label change plot failed: {e}")


print("\n[CSV 1] Per-network significant regions table")

sig_rows = []
for i in range(200):
    sig_rec_flag = pval_rec[i] < ALPHA
    sig_int_flag = pval_int[i] < ALPHA
    if sig_rec_flag or sig_int_flag:
        net_id = static_communities[i]
        net_name = NETWORKS[net_id - 1] if net_id > 0 else "Unknown"
        sig_rows.append({
            "ROI_idx": i + 1,
            "ROI_label": labels_200[i],
            "Network": net_name,
            "Rec_AN": rec_an[i],
            "Rec_HC": rec_hc[i],
            "Rec_diff": rec_an[i] - rec_hc[i],
            "Rec_pval": pval_rec[i],
            "Rec_sig": sig_rec_flag,
            "Int_AN": int_an[i],
            "Int_HC": int_hc[i],
            "Int_diff": int_an[i] - int_hc[i],
            "Int_pval": pval_int[i],
            "Int_sig": sig_int_flag,
        })

df_sig = pd.DataFrame(sig_rows)
df_sig.to_csv(CSV_DIR / "significant_regions_rec_int.csv", index=False)
df_sig.to_excel(CSV_DIR / "significant_regions_rec_int.xlsx", index=False)
print(f"  Saved significant_regions_rec_int.csv/xlsx ({len(df_sig)} significant ROIs)")


print("\n[CSV 2] Coarse allegiance difference table")

rows_coarse = []
for r in range(7):
    for c in range(7):
        rows_coarse.append({
            "Row_Network": NETWORK_SHORT[r],
            "Col_Network": NETWORK_SHORT[c],
            "AN": alg_coarse_an[r, c],
            "HC": alg_coarse_hc[r, c],
            "Diff_AN_HC": alg_coarse_an[r, c] - alg_coarse_hc[r, c],
            "pvalue": pval_alg_coarse[r, c],
            "Significant": pval_alg_coarse[r, c] < ALPHA,
        })

df_coarse_full = pd.DataFrame(rows_coarse)
df_coarse_full.to_csv(CSV_DIR / "coarse_allegiance_difference_full.csv", index=False)
print(f"  Saved coarse_allegiance_difference_full.csv ({len(df_coarse_full)} cells)")


print("\n[Excel] Comprehensive results workbook")

excel_path = CSV_DIR / "anorexia_results_comprehensive_v2.xlsx"
with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    df_nodal.to_excel(writer, sheet_name="Nodal_Statistics_200", index=False)
    df_coarse_full.to_excel(writer, sheet_name="Coarse_Allegiance_7x7", index=False)
    if len(df_sig) > 0:
        df_sig.to_excel(writer, sheet_name="Significant_Regions", index=False)

    # Coarse RC/IC summary
    df_coarse_ri = pd.DataFrame({
        "Network": NETWORK_SHORT,
        "RC_AN": rec_coarse_an,
        "RC_HC": rec_coarse_hc,
        "RC_pvalue": pval_rec_coarse,
        "IC_AN": int_coarse_an,
        "IC_HC": int_coarse_hc,
        "IC_pvalue": pval_int_coarse,
    })
    df_coarse_ri.to_excel(writer, sheet_name="Coarse_RC_IC", index=False)

    # Global summary
    summary_path = STAT_DIR / "summary_global_tests.csv"
    if summary_path.exists():
        pd.read_csv(summary_path).to_excel(writer, sheet_name="Global_Tests", index=False)

print(f"  Saved {excel_path}")


end_time = datetime.now()
print(f"\nVisualisation & results export complete  (v2 — {N_SUBJ_PER_GROUP} subj/group)")
print(f"Figure directory: {FIG_DIR}")
print(f"CSV/Excel directory: {CSV_DIR}")
print(f"Duration: {end_time - start_time}")
print(f"\nFigures created:")
for f in sorted(FIG_DIR.glob("*")):
    print(f"  {f.name}")
print(f"\nCSV/Excel files:")
for f in sorted(CSV_DIR.glob("*.csv")) + sorted(CSV_DIR.glob("*.xlsx")):
    print(f"  {f.name}")
