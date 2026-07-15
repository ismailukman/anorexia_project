#!/usr/bin/env python3
"""
01d_static_fc.py
Static FC for the combined 216-region atlas (Schaefer-200 + Tian S1).
Computes per-subject Pearson correlation matrices and saves QC figures.
"""
from pathlib import Path
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

ROOT       = Path("/Users/ismaila/Documents/C-Codes/AnorexiaProject")
DATA       = ROOT / "data/analysis/combined_subjs"
LABELS_TXT = ROOT / "data/atlas/combined_216/combined_216_labels.txt"
OUT        = ROOT / "output/figures/stage1_fc/static_fc"
OUT.mkdir(parents=True, exist_ok=True)

N_CORTICAL = 200
N_SUBCORT  = 16
N_REGIONS  = 216
K_EXPAND   = 3
N_EXPANDED = N_CORTICAL + N_SUBCORT * K_EXPAND   # 248

all_labels = []
for line in LABELS_TXT.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    parts = line.split()
    if len(parts) >= 2:
        all_labels.append(parts[1])

YEO7_COLORS = ["#A251AC", "#789AC1", "#409832", "#E165FE",
               "#F6FDC9", "#EFB944", "#D9717D"]
YEO_MAP = {"Vis": 1, "SomMot": 2, "DorsAttn": 3, "SalVentAttn": 4,
           "Limbic": 5, "Cont": 6, "Default": 7}
YEO_NAMES = {1: "VN", 2: "SMN", 3: "DAN", 4: "VAN",
             5: "LN", 6: "FPN", 7: "DMN"}
YEO_FULL  = {1: "Visual", 2: "Somatomotor", 3: "Dorsal Attention",
             4: "Ventral Attention", 5: "Limbic",
             6: "Frontoparietal", 7: "Default Mode"}
SUBCORT_YEO = {"HIP": 7, "AMY": 5, "pTHA": 2, "aTHA": 5,
               "NAc": 5, "GP": 2, "PUT": 2, "CAU": 6}

def yeo_code(i):
    lbl = all_labels[i]
    if i < N_CORTICAL:
        for k, c in YEO_MAP.items():
            if k in lbl:
                return c
        return 0
    return SUBCORT_YEO.get(lbl.split("-")[0], 0)

CORTICAL_BOUNDS = [
    (-0.5, 13.5), (13.5, 29.5), (29.5, 42.5), (42.5, 53.5),
    (53.5, 59.5), (59.5, 72.5), (72.5, 99.5),
    (99.5, 114.5), (114.5, 133.5), (133.5, 146.5), (146.5, 157.5),
    (157.5, 163.5), (163.5, 180.5), (180.5, 199.5),
]
CORTICAL_COLORS_14 = YEO7_COLORS * 2   # VN…DMN × LH then RH

_SUBCORT_ORIG = [
    "HIP-rh", "AMY-rh", "pTHA-rh", "aTHA-rh",
    "NAc-rh", "GP-rh",  "PUT-rh",  "CAU-rh",
    "HIP-lh", "AMY-lh", "pTHA-lh", "aTHA-lh",
    "NAc-lh", "GP-lh",  "PUT-lh",  "CAU-lh",
]
_yeo_counts_sc = {}
for _l in _SUBCORT_ORIG:
    _y = SUBCORT_YEO[_l.split("-")[0]]
    _yeo_counts_sc[_y] = _yeo_counts_sc.get(_y, 0) + 1

_SORTED_YEOS = [2, 5, 6, 7]   # SMN, LN, FPN, DMN

def _sort_key(k):
    lbl = _SUBCORT_ORIG[k]
    y = SUBCORT_YEO[lbl.split("-")[0]]
    return (y, 0 if lbl.endswith("rh") else 1)

SUBCORT_SORTED_IDX    = sorted(range(16), key=_sort_key)
SUBCORT_LABELS_SORTED = [_SUBCORT_ORIG[k] for k in SUBCORT_SORTED_IDX]
PERM_216              = list(range(200)) + [200 + k for k in SUBCORT_SORTED_IDX]

# Expanded subcortical group boundaries (K_EXPAND cells per ROI)
_pos_exp = 199.5
SUBCORT_NET_GROUPS_EXP = []
for _y in _SORTED_YEOS:
    _end = _pos_exp + _yeo_counts_sc[_y] * K_EXPAND
    SUBCORT_NET_GROUPS_EXP.append(
        (f"~{YEO_NAMES[_y]}", _pos_exp, _end, YEO7_COLORS[_y - 1]))
    _pos_exp = _end
SUBCORT_INNER_TICKS_EXP = [g[2] for g in SUBCORT_NET_GROUPS_EXP[:-1]]


_cum = 0
SUBCORT_ZOOM_TICKS = []
for _y in _SORTED_YEOS[:-1]:
    _cum += _yeo_counts_sc[_y]
    SUBCORT_ZOOM_TICKS.append(_cum - 0.5)


def expand_matrix(mat_perm):
    """Replicate subcortical rows/cols K_EXPAND times: 216×216 → 248×248."""
    cort   = mat_perm[:N_CORTICAL, :]
    subc   = mat_perm[N_CORTICAL:, :]
    subc_r = np.repeat(subc, K_EXPAND, axis=0)
    mat_r  = np.vstack([cort, subc_r])
    cort_c   = mat_r[:, :N_CORTICAL]
    subc_c   = mat_r[:, N_CORTICAL:]
    subc_c_e = np.repeat(subc_c, K_EXPAND, axis=1)
    return np.hstack([cort_c, subc_c_e])


def _add_strips(ax):
    """Color strips along the top and left edges of the N_EXPANDED matrix.

    Cortical (14 blocks): solid black edge, no text.
    Subcortical (4 network groups): light gray edge, no text.
    """
    xmin, xmax, ymin, ymax = ax.axis()
    # ymin=N-0.5 (bottom), ymax=-0.5 (top) for matshow; h is negative
    h     = (ymax - ymin) / 28.0
    space = h / 5.0
    i_top  = ymax + space
    i_left = ymax + space

    # Cortical: solid black border
    for (start, end), color in zip(CORTICAL_BOUNDS, CORTICAL_COLORS_14):
        blk = end - start
        ax.add_patch(mpatches.Rectangle(
            (start, i_top), blk, h,
            facecolor=color, clip_on=False, linewidth=1.2, edgecolor="k"))
        ax.add_patch(mpatches.Rectangle(
            (i_left, start + 0.5), h, blk,
            facecolor=color, clip_on=False, linewidth=1.2, edgecolor="k"))

    # Subcortical: 4 functional group blocks, light gray border
    for _, gs, ge, nc in SUBCORT_NET_GROUPS_EXP:
        blk = ge - gs
        ax.add_patch(mpatches.Rectangle(
            (gs, i_top), blk, h,
            facecolor=nc, clip_on=False, linewidth=0.7, edgecolor="#aaaaaa"))
        ax.add_patch(mpatches.Rectangle(
            (i_left, gs + 0.5), h, blk,
            facecolor=nc, clip_on=False, linewidth=0.7, edgecolor="#aaaaaa"))


def _setup_matrix_ax(ax, mat_exp, vmin=-1, vmax=1, cmap="RdBu_r"):
    """Display N_EXPANDED×N_EXPANDED matrix; suppress all tick labels."""
    im = ax.matshow(mat_exp, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.axvline(x=99.5,  color="white", lw=1.5, alpha=0.7)
    ax.axhline(y=99.5,  color="white", lw=1.5, alpha=0.7)
    ax.axvline(x=199.5, color="white", lw=2.5)
    ax.axhline(y=199.5, color="white", lw=2.5)
    for t in SUBCORT_INNER_TICKS_EXP:
        ax.axvline(x=t, color="white", lw=1.2)
        ax.axhline(y=t, color="white", lw=1.2)
    ax.tick_params(which="both",
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labeltop=False,
                   labelleft=False,   labelright=False)
    return im


def load_timeseries(grp, subj_id):
    pat = str(DATA / f"subj_timeseries_combined_{grp}_patients_subj{subj_id:02d}.mat")
    files = glob.glob(pat)
    if not files:
        return None
    d = sio.loadmat(files[0])
    key = [k for k in d if not k.startswith("_")][0]
    return np.array(d[key])


def compute_static_fc(ts):
    """Pearson correlation over full timeseries → 216×216."""
    fc = np.corrcoef(ts.T)
    np.fill_diagonal(fc, 0)
    return fc


def zmean(r_stack, axis=0):
    """Average correlation matrices in Fisher z-space, return r."""
    z = np.arctanh(np.clip(r_stack, -0.9999, 0.9999))
    return np.tanh(z.mean(axis=axis))


def load_dfc_mean(grp, subj_id):
    """Time-averaged DFC: average 663 windows in Fisher z-space."""
    pat = str(DATA / f"subj_fc_combined_{grp}_patients_subj{subj_id:02d}.mat")
    files = glob.glob(pat)
    if not files:
        return None
    d = sio.loadmat(files[0])
    key = [k for k in d if not k.startswith("_")][0]
    fc_wins = np.array(d[key])
    mean_dfc = zmean(fc_wins, axis=0)
    np.fill_diagonal(mean_dfc, 0)
    return mean_dfc


def plot_static_fc(sfc, title, out_path):
    perm    = np.array(PERM_216)
    sfc_p   = sfc[np.ix_(perm, perm)]
    sfc_exp = expand_matrix(sfc_p)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                             gridspec_kw={"wspace": 0.38})
    fig.subplots_adjust(top=0.76, bottom=0.05, left=0.06, right=0.96)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)

    ax = axes[0]
    im = _setup_matrix_ax(ax, sfc_exp)
    ax.set_title("Static FC  (216 ROIs, subcortical ×3 display)",
                 fontsize=10, fontweight="bold", y=1.12)
    plt.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
    _add_strips(ax)

    ax2 = axes[1]
    sc_idx        = list(range(N_CORTICAL, N_REGIONS))
    sc_perm_local = [PERM_216.index(i) for i in sc_idx]
    sc_block      = sfc_p[np.ix_(sc_perm_local, sc_perm_local)]
    im2 = ax2.matshow(sc_block, vmin=-0.5, vmax=1, cmap="RdBu_r")
    ax2.set_title("Subcortical block zoom  (16×16)",
                  fontsize=10, fontweight="bold", y=1.12)
    plt.colorbar(im2, ax=ax2, shrink=0.72, pad=0.02)
    for t in SUBCORT_ZOOM_TICKS:
        ax2.axvline(x=t, color="white", lw=1.5)
        ax2.axhline(y=t, color="white", lw=1.5)
    ax2.tick_params(which="both",
                    bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False,
                    labelleft=False,   labelright=False)

    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def plot_static_vs_dfc(sfc, dfc_mean, title, out_path):
    perm  = np.array(PERM_216)
    sfc_p = sfc[np.ix_(perm, perm)]
    dfc_p = dfc_mean[np.ix_(perm, perm)]

    # Difference in Fisher z-space (linear scale, no compression near ±1)
    z_sfc  = np.arctanh(np.clip(sfc_p,  -0.9999, 0.9999))
    z_dfc  = np.arctanh(np.clip(dfc_p,  -0.9999, 0.9999))
    z_diff = z_sfc - z_dfc

    sfc_exp    = expand_matrix(sfc_p)
    dfc_exp    = expand_matrix(dfc_p)
    zdiff_exp  = expand_matrix(z_diff)

    fig, axes = plt.subplots(1, 3, figsize=(26, 8),
                             gridspec_kw={"wspace": 0.35})
    fig.subplots_adjust(top=0.76, bottom=0.05, left=0.04, right=0.97)
    fig.suptitle(f"{title} — Static FC vs Dynamic FC (time-averaged)",
                 fontsize=12, fontweight="bold", y=0.97)

    panel_data = [
        (sfc_exp,   "Static FC",            "RdBu_r",  -1,   1,   "Pearson r"),
        (dfc_exp,   "DFC Mean",             "RdBu_r",  -1,   1,   "Pearson r"),
        (zdiff_exp, "Static − DFC (z)",     "coolwarm", -0.5, 0.5, "Δz (Fisher)"),
    ]
    for ax, (mat, t, cm, vn, vx, cblbl) in zip(axes, panel_data):
        im = _setup_matrix_ax(ax, mat, vmin=vn, vmax=vx, cmap=cm)
        ax.set_title(t, fontsize=10, fontweight="bold", y=1.12)
        cb = plt.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
        cb.set_label(cblbl, fontsize=8)
        _add_strips(ax)

    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def make_legend(out_path):
    """Save a standalone ROI legend listing all 216 region names."""

    NET_ORDER = ["VN", "SMN", "DAN", "VAN", "LN", "FPN", "DMN"]
    SC_GRP_ORDER  = ["~SMN", "~LN", "~FPN", "~DMN"]
    SC_GRP_YEOS   = [2, 5, 6, 7]
    SC_GRP_COLORS = [YEO7_COLORS[y - 1] for y in SC_GRP_YEOS]
    yeo_to_grp    = {2: "~SMN", 5: "~LN", 6: "~FPN", 7: "~DMN"}

    cortical_by_net = {n: [] for n in NET_ORDER}
    for i in range(N_CORTICAL):
        c = yeo_code(i)
        if c > 0:
            short = all_labels[i].replace("7Networks_", "")
            cortical_by_net[YEO_NAMES[c]].append(f"{i+1:3d}  {short}")

    sc_by_grp = {g: [] for g in SC_GRP_ORDER}
    for k, lbl in enumerate(SUBCORT_LABELS_SORTED):
        g = yeo_to_grp[SUBCORT_YEO[lbl.split("-")[0]]]
        orig_idx = SUBCORT_SORTED_IDX[k]
        sc_by_grp[g].append(f"{201 + orig_idx:3d}  {lbl}")

    fig = plt.figure(figsize=(18, 26), facecolor="white")
    gs_root = gridspec.GridSpec(3, 1, figure=fig,
                                hspace=0.06, left=0.02, right=0.98,
                                top=0.97, bottom=0.01,
                                height_ratios=[1, 1.4, 9])

    ax1 = fig.add_subplot(gs_root[0])
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis("off")
    ax1.text(0.0, 0.98, "Cortical Networks — Yeo-7",
             va="top", fontsize=12, fontweight="bold")
    for i, code in enumerate(range(1, 8)):
        x = i / 7.0 + 0.02
        col = YEO7_COLORS[code - 1]
        ax1.add_patch(mpatches.Rectangle(
            (x, 0.48), 0.07, 0.34,
            facecolor=col, edgecolor="k", linewidth=1.0, clip_on=False))
        ax1.text(x + 0.035, 0.32,
                 f"{YEO_NAMES[code]}\n{YEO_FULL[code]}\n({len(cortical_by_net[YEO_NAMES[code]])} ROIs)",
                 ha="center", va="top", fontsize=8)

    ax2 = fig.add_subplot(gs_root[1])
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
    ax2.text(0.0, 0.98, "Subcortical Functional Groups — Tian Scale I (functional overlap)",
             va="top", fontsize=12, fontweight="bold")

    col_w = 0.25
    for ci, (grp, col) in enumerate(zip(SC_GRP_ORDER, SC_GRP_COLORS)):
        cx = ci * col_w + 0.01
        ax2.add_patch(mpatches.Rectangle(
            (cx, 0.72), 0.06, 0.18,
            facecolor=col, edgecolor="#aaaaaa", linewidth=0.8, clip_on=False))
        ax2.text(cx + 0.08, 0.81, grp,
                 va="center", fontsize=10, fontweight="bold")
        for ri, roi_str in enumerate(sc_by_grp[grp]):
            ax2.text(cx + 0.01, 0.62 - ri * 0.13, roi_str,
                     va="top", fontsize=8, family="monospace")

    ax3 = fig.add_subplot(gs_root[2])
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.axis("off")
    ax3.text(0.0, 0.995, "Cortical ROI Names — Schaefer-200 (sorted by Yeo-7 network, L→R hemisphere)",
             va="top", fontsize=12, fontweight="bold")

    max_n  = max(len(v) for v in cortical_by_net.values())
    row_h  = 0.95 / (max_n + 2)
    col_w3 = 1.0 / 7
    for ci, (net, col) in enumerate(zip(NET_ORDER, YEO7_COLORS)):
        cx  = ci * col_w3 + 0.005
        rois = cortical_by_net[net]
        ax3.add_patch(mpatches.Rectangle(
            (cx, 0.97 - row_h), col_w3 * 0.95, row_h * 0.85,
            facecolor=col, edgecolor="k", linewidth=1.0, clip_on=False))
        ax3.text(cx + col_w3 * 0.475, 0.97 - row_h / 2,
                 f"{net}  ({len(rois)})",
                 ha="center", va="center", fontsize=8.5, fontweight="bold")
        for ri, roi_str in enumerate(rois):
            ax3.text(cx, 0.97 - (ri + 2) * row_h, roi_str,
                     va="top", fontsize=5.8, family="monospace")

    plt.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


print("Computing static FC …")
sfc_all = {"an": [], "hc": []}
dfc_all = {"an": [], "hc": []}

for grp in ["an", "hc"]:
    tag = grp.upper()
    for sid in range(1, 6):
        ts = load_timeseries(grp, sid)
        if ts is None:
            print(f"  [{tag} subj{sid:02d}] timeseries not found"); continue
        sfc = compute_static_fc(ts)
        dfc = load_dfc_mean(grp, sid)
        sfc_all[grp].append(sfc)
        if dfc is not None:
            dfc_all[grp].append(dfc)

        plot_static_fc(sfc,
                       f"Static FC — {tag} Subject {sid:02d}",
                       OUT / f"static_fc_{grp}_subj{sid:02d}.png")
        if dfc is not None:
            plot_static_vs_dfc(sfc, dfc,
                               f"{tag} Subject {sid:02d}",
                               OUT / f"static_vs_dfc_{grp}_subj{sid:02d}.png")

for grp, tag in [("an", "AN"), ("hc", "HC")]:
    if sfc_all[grp]:
        sfc_m = zmean(np.array(sfc_all[grp]), axis=0)
        plot_static_fc(sfc_m,
                       f"Group-Mean Static FC — {tag}  (N={len(sfc_all[grp])})",
                       OUT / f"static_fc_{grp}_groupmean.png")
    if dfc_all[grp] and sfc_all[grp]:
        dfc_m = zmean(np.array(dfc_all[grp]), axis=0)
        sfc_m = zmean(np.array(sfc_all[grp]), axis=0)
        plot_static_vs_dfc(sfc_m, dfc_m,
                           f"{tag} Group Mean  (N={len(sfc_all[grp])})",
                           OUT / f"static_vs_dfc_{grp}_groupmean.png")

print("Saving ROI legend …")
make_legend(OUT / "static_fc_roi_legend.png")

print("Done — figures saved to", OUT)
