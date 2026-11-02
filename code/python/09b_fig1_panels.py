#!/usr/bin/env python3
"""
09b_fig1_panels.py: data panels for Figure 1 (pipeline) of the ISMRM abstract.

Each panel is drawn from this study's data:

    p1_atlas.png     the 216-region atlas (Schaefer-200 + Tian S1), coloured
                     by functional network
    p2_ts.png        BOLD time series of four regions from one control, with
                     one 38-TR window and the next window one TR later
    p2_mat.png       that window's 216 x 216 correlation matrix, ordered by
                     functional group
    p3_raster.png    community labels of every region (rows) in every window
                     (columns) for the same participant, after label matching
    p4_alleg.png     control-group mean allegiance, ordered by functional
                     group, with the 13 groups outlined
    p5_null.png      permutation null of the recruitment difference with the
                     observed value (21 AN v 22 HC)

Output: report/abstract/figures/fig1_panels/, used to build Figure 1.

Run:
    conda activate fmri
    python 09b_fig1_panels.py
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap, to_rgb
from nilearn import image, plotting

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import atlas_groups                                    # noqa: E402
from mlcd_measures import allegiance, match_community_affiliation   # noqa: E402

spec = importlib.util.spec_from_file_location("dfc", HERE / "05b_dfc_group_statistics.py")
dfc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dfc)

from paths import ROOT
OUT = ROOT / "report/abstract/figures/fig1_panels"
SCHAEFER = ROOT / "data/atlas/schaefer_2018/schaefer_2018/Schaefer2018_200Parcels_7Networks_order_subcortMasked.nii.gz"
TIAN = ROOT / "data/atlas/tian_s1/Tian_Subcortex_S1_3T_2009cAsym.nii.gz"
TS = ROOT / "data/analysis/timeseries_subjs_masked"
STATS = ROOT / "output/results/dfc_statistics_combined216"

HC_SUBJ = 3                 # example participant (control 03)
WIN, STEP = 38, 1
WIN_START = 60              # example window (TR index)
N_PERM = 20000
SEED = 42

INK = "#1E2733"
ACCENT = "#1F3B73"          # single dark accent
RED = "#B03A2E"             # observed value / AN
GREY = "#B9C0C9"
# muted network tones (VN, SMN, DAN, VAN, LN, FPN, DMN)
NET = ["#8796B5", "#9DB3C9", "#86A396", "#A897B8", "#C8BB92", "#C79E7A", "#B48686"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "savefig.dpi": 400, "savefig.transparent": True})


def groups():
    gv, names, labels = atlas_groups.combined216_groups()
    order = np.argsort(gv, kind="stable")
    return gv, names, order


def net_of_group(names):
    """Yeo network index (0-6) of each functional group (cortical or ~subcortical)."""
    return [atlas_groups.YEO_NAMES.index(n.lstrip("~")) for n in names]


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  ", OUT / name)


def p1_atlas(gv, names):
    sch = nib.load(str(SCHAEFER))
    tian = image.resample_to_img(nib.load(str(TIAN)), sch, interpolation="nearest",
                                 force_resample=True, copy_header=True)
    s, t = np.asarray(sch.dataobj).astype(int), np.asarray(tian.dataobj).astype(int)
    lab = np.where(s > 0, s, np.where(t > 0, 200 + t, 0))
    netg = net_of_group(names)
    # value per voxel = network (1-7), subcortex offset by 7 so it can be darker
    reg_net = np.array([netg[g - 1] for g in gv])
    vol = np.zeros_like(lab)
    m = lab > 0
    vol[m] = reg_net[lab[m] - 1] + 1 + np.where(lab[m] > 200, 7, 0)
    cols = NET + [tuple(0.72 * c for c in to_rgb(h)) for h in NET]
    img = nib.Nifti1Image(vol.astype(np.int16), sch.affine)
    fig = plt.figure(figsize=(3.6, 1.6))
    plotting.plot_roi(img, display_mode="xz", cut_coords=(8, 4), cmap=ListedColormap(cols),
                      vmin=0.5, vmax=14.5, draw_cross=False, annotate=False, colorbar=False, figure=fig,
                      black_bg=False, alpha=0.95)
    save(fig, "p1_atlas.png")


def p2(gv, order):
    ctx = np.load(TS / f"ts_hc_patients_subj{HC_SUBJ:02d}.npy").astype(float)
    sub = np.load(TS / f"ts_sub_hc_patients_subj{HC_SUBJ:02d}.npy").astype(float)
    X = np.column_stack([ctx, sub])
    X = (X - X.mean(0)) / X.std(0)
    picks = [0, 1, 2, 215]                                    # regions 1, 2, 3, 216
    t0, t1 = 0, 160
    fig, ax = plt.subplots(figsize=(2.05, 1.35))
    for k, r in enumerate(picks):
        ax.plot(np.arange(t0, t1), X[t0:t1, r] * 0.32 - 1.25 * k, color=INK, lw=0.6)
    ax.axvspan(WIN_START, WIN_START + WIN, color=ACCENT, alpha=0.12, lw=0)
    ax.axvline(WIN_START, color=ACCENT, lw=0.8)
    ax.axvline(WIN_START + WIN, color=ACCENT, lw=0.8)
    ax.axvline(WIN_START + STEP, color=ACCENT, lw=0.6, ls=(0, (3, 2)))
    ax.axvline(WIN_START + WIN + STEP, color=ACCENT, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks([-1.25 * k for k in range(4)], ["1", "2", "3", "216"],
                  fontsize=7)
    ax.set_xticks([0, 50, 100, 150], ["0", "40", "80", "120 s"], fontsize=7)
    ax.set_ylabel("region", fontsize=7, labelpad=1)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "p2_ts.png")

    W = X[WIN_START:WIN_START + WIN]
    C = np.corrcoef(W.T)[np.ix_(order, order)]
    np.fill_diagonal(C, np.nan)
    fig, ax = plt.subplots(figsize=(1.0, 1.0))
    ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    edges = np.flatnonzero(np.diff(gv[order])) + 0.5
    for e in edges:
        ax.axhline(e, color="white", lw=0.3)
        ax.axvline(e, color="white", lw=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(INK); sp.set_linewidth(0.6)
    save(fig, "p2_mat.png")


def load_hc_subject():
    dfc.select_cohort("21v22")
    cfg = dfc.ATLAS_CFG["combined216"]
    subs, _, _ = dfc.load_group(cfg, "control", 216)
    return match_community_affiliation(subs[HC_SUBJ - 1]), subs


def p3_raster(order, C):
    # relabel communities by size so the largest get the darkest tones
    C = np.asarray(C, dtype=float)[order]
    ok = ~np.isnan(C)
    labs, counts = np.unique(C[ok], return_counts=True)
    rank = {l: r for r, l in enumerate(labs[np.argsort(-counts)])}
    R = np.full(C.shape, np.nan)
    R[ok] = [rank[v] for v in C[ok]]
    n = int(np.nanmax(R)) + 1
    base = [ACCENT, "#4F6FA0", "#8EA6C8", "#C99A6B", "#C3D0E2", "#E3CDB0", "#7B8592", "#A9B1BB"]
    cols = [base[i] if i < len(base) else "#DDE1E6" for i in range(n)]
    fig, ax = plt.subplots(figsize=(2.15, 1.15))
    ax.imshow(R, aspect="auto", cmap=ListedColormap(cols), interpolation="nearest")
    ax.set_xlabel("window", fontsize=7, labelpad=1)
    ax.set_ylabel("region", fontsize=7, labelpad=1)
    ax.set_xticks([0, 331, 662], ["1", "332", "663"], fontsize=6.5)
    ax.set_yticks([0, 215], ["1", "216"], fontsize=6.5)
    for sp in ax.spines.values():
        sp.set_color(INK); sp.set_linewidth(0.6)
    save(fig, "p3_raster.png")


def p4_alleg(gv, order, subs):
    A = np.nanmean([allegiance(s) for s in subs], axis=0)[np.ix_(order, order)]
    np.fill_diagonal(A, np.nan)
    fig, ax = plt.subplots(figsize=(1.25, 1.25))
    ax.imshow(A, cmap="Greys", vmin=0, vmax=np.nanpercentile(A, 99), interpolation="nearest")
    g = gv[order]
    for k in np.unique(g):
        idx = np.flatnonzero(g == k)
        a, b = idx[0] - 0.5, idx[-1] + 0.5
        ax.add_patch(plt.Rectangle((a, a), b - a, b - a, fill=False, ec=RED, lw=0.6))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(INK); sp.set_linewidth(0.6)
    save(fig, "p4_alleg.png")


def p5_null():
    z = np.load(STATS / "dfc_group_stats.npz", allow_pickle=True)
    an, hc = z["rec_subj_an"].mean(1), z["rec_subj_hc"].mean(1)
    x = np.concatenate([an, hc])
    n1 = an.size

    def d(v):
        a, b = v[:n1], v[n1:]
        sp = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (x.size - 2))
        return (a.mean() - b.mean()) / sp
    obs = d(x)
    rng = np.random.default_rng(SEED)
    null = np.array([d(rng.permutation(x)) for _ in range(N_PERM)])
    fig, ax = plt.subplots(figsize=(1.75, 1.15))
    bins = np.linspace(-1.4, 1.4, 41)
    h, e = np.histogram(null, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    tail = np.abs(c) >= abs(obs)
    ax.bar(c[~tail], h[~tail], width=e[1] - e[0], color=GREY, edgecolor="none")
    ax.bar(c[tail], h[tail], width=e[1] - e[0], color=RED, alpha=0.55, edgecolor="none")
    ax.axvline(obs, color=RED, lw=1.2)
    ax.text(obs - 0.05, h.max() * 0.98, f"observed\nd = {obs:.2f}", color=RED, ha="right", va="top",
            fontsize=7, fontweight="bold")
    ax.set_xlabel("Cohen's d, shuffled labels", fontsize=7, labelpad=1)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    save(fig, "p5_null.png")
    print(f"   observed d = {obs:.3f}; two-sided permutation p = {(np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1):.4f}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gv, names, order = groups()
    p1_atlas(gv, names)
    p2(gv, order)
    C, subs = load_hc_subject()
    p3_raster(order, C)
    p4_alleg(gv, order, subs)
    p5_null()


if __name__ == "__main__":
    main()
