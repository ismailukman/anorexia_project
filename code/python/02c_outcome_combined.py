#!/usr/bin/env python3
"""
02c_outcome_measures_combined.py — Combined outcome measures (Schaefer-200 + Tian S1, 216 ROIs).
Subcortical ROIs are mapped to Yeo-7 networks following Tian et al. 2020.
Run after: combined_mlcd_5subj_d05.m
"""
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from teneto import communitymeasures

start_time = datetime.now()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

log.info("STAGE 3c — Combined Outcome Measures (216 ROIs: Schaefer-200 + Tian S1)")

ROOT        = Path("/Users/ismaila/Documents/C-Codes/AnorexiaProject")
MLCD_DIR    = ROOT / "data/analysis/mlcd_subjs_combined_5subj_d05/subjs_mlcd"
OUT_METRICS = ROOT / "output/results/subject_metrics_combined_5subj_d05"
OUT_XLSX    = ROOT / "output/results/xlsx_exports_combined_5subj_d05"
OUT_FIG     = ROOT / "output/figures/stage3_combined_5subj_d05"
for d in [OUT_METRICS, OUT_XLSX, OUT_FIG]:
    d.mkdir(parents=True, exist_ok=True)

N_CORTICAL = 200
N_SUBCORT  = 16
N_REGIONS  = N_CORTICAL + N_SUBCORT   # 216
N_SUBJ     = 5
K_EXPAND   = 3                        # each subcortical ROI → 3 display cells
N_EXPANDED = N_CORTICAL + N_SUBCORT * K_EXPAND  # 248

LABEL_FILE = ROOT / "data/atlas/combined_216/combined_216_labels.txt"
all_labels = []
for line in LABEL_FILE.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    parts = line.split()
    if len(parts) >= 2:
        all_labels.append(parts[1])

assert len(all_labels) == N_REGIONS, f"Expected {N_REGIONS} labels, got {len(all_labels)}"
log.info(f"Loaded {len(all_labels)} combined labels")

SUBCORT_YEO = {
    "HIP":  7,   # DMN
    "AMY":  5,   # LN
    "pTHA": 2,   # SMN  (posterior thalamus = sensorimotor relay)
    "aTHA": 5,   # LN   (anterior thalamus = limbic/cingulate relay)
    "NAc":  5,   # LN
    "GP":   2,   # SMN
    "PUT":  2,   # SMN
    "CAU":  6,   # FPN  (frontoparietal)
}

_SUBCORT_ORIG = [
    "HIP-rh", "AMY-rh", "pTHA-rh", "aTHA-rh",
    "NAc-rh", "GP-rh",  "PUT-rh",  "CAU-rh",
    "HIP-lh", "AMY-lh", "pTHA-lh", "aTHA-lh",
    "NAc-lh", "GP-lh",  "PUT-lh",  "CAU-lh",
]


def _subcort_yeo(lbl):
    return SUBCORT_YEO[lbl.split("-")[0]]


def _sort_key(k):
    lbl = _SUBCORT_ORIG[k]
    return (_subcort_yeo(lbl), 0 if lbl.endswith("rh") else 1, k)


SUBCORT_SORTED_IDX = sorted(range(16), key=_sort_key)
SUBCORT_LABELS_SORTED = [_SUBCORT_ORIG[k] for k in SUBCORT_SORTED_IDX]
PERM_216 = list(range(200)) + [200 + k for k in SUBCORT_SORTED_IDX]

_yeo_counts = Counter(_subcort_yeo(lbl) for lbl in SUBCORT_LABELS_SORTED)
_SORTED_YEOS = []
_seen: set = set()
for lbl in SUBCORT_LABELS_SORTED:
    y = _subcort_yeo(lbl)
    if y not in _seen:
        _SORTED_YEOS.append(y)
        _seen.add(y)

YEO_MAP = {"Vis": 1, "SomMot": 2, "DorsAttn": 3, "SalVentAttn": 4,
           "Limbic": 5, "Cont": 6, "Default": 7}
YEO_NAMES = {1: "VN", 2: "SMN", 3: "DAN",
             4: "VAN", 5: "LN", 6: "FPN", 7: "DMN"}


def _label_to_community(lbl, idx_0based):
    if idx_0based < N_CORTICAL:
        parts = lbl.split("_")
        for net_key, comm in YEO_MAP.items():
            if net_key in parts:
                return comm
        return 0
    else:
        return SUBCORT_YEO.get(lbl.split("-")[0], 0)


static_communities = np.array(
    [_label_to_community(l, i) for i, l in enumerate(all_labels)], dtype=int
)
log.info(f"Static communities (Yeo-7): "
         f"{ {int(u): int(c) for u, c in zip(*np.unique(static_communities, return_counts=True))} }")

roi_index = pd.Index(all_labels, name="ROI")

YEO7_COLORS = ["#A251AC", "#789AC1", "#409832", "#E165FE",
               "#F6FDC9", "#EFB944", "#D9717D"]

CORTICAL_BOUNDS = [
    (-0.5,  13.5), (13.5,  29.5), (29.5,  42.5), (42.5,  53.5),
    (53.5,  59.5), (59.5,  72.5), (72.5,  99.5),
    (99.5, 114.5), (114.5, 133.5), (133.5, 146.5), (146.5, 157.5),
    (157.5, 163.5), (163.5, 180.5), (180.5, 199.5),
]
CORTICAL_COLORS_14 = YEO7_COLORS * 2
YEO_SHORT_14 = ["VN", "SMN", "DAN", "VAN", "LN", "FPN", "DMN"] * 2

SUBCORT_STRUCT_SHORT = {
    "HIP": "HIP", "AMY": "AMY", "pTHA": "pTH",
    "aTHA": "aTH", "NAc": "NAc", "GP": "GP",
    "PUT": "PUT", "CAU": "CAU",
}
SUBCORT_ROI_COLORS = [YEO7_COLORS[_subcort_yeo(l) - 1] for l in SUBCORT_LABELS_SORTED]
SUBCORT_ROI_SHORT  = [SUBCORT_STRUCT_SHORT[l.split("-")[0]] for l in SUBCORT_LABELS_SORTED]

_sc_pos = 199.5
SUBCORT_NET_GROUPS = []   # (overlap_label, start, end, color)
for _yeo in _SORTED_YEOS:
    _end = _sc_pos + _yeo_counts[_yeo] * K_EXPAND
    SUBCORT_NET_GROUPS.append(
        (f"~{YEO_NAMES[_yeo]}", _sc_pos, _end, YEO7_COLORS[_yeo - 1])
    )
    _sc_pos = _end

SUBCORT_INNER_TICKS_EXP = [g[2] for g in SUBCORT_NET_GROUPS[:-1]]


def _expand_allegiance(A_perm):
    """Replicate each subcortical row/col K_EXPAND times → 248×248 display matrix."""
    idx = list(range(N_CORTICAL))
    for j in range(N_SUBCORT):
        idx.extend([N_CORTICAL + j] * K_EXPAND)
    return A_perm[np.ix_(np.array(idx), np.array(idx))]


def _save_allegiance_figure(A, title_str, save_path):
    """Plot the combined allegiance matrix (216 ROIs → 248×248 display)."""
    perm   = np.array(PERM_216)
    A_perm = A[np.ix_(perm, perm)]
    A_plot = _expand_allegiance(A_perm)   # 248×248

    fig = plt.figure(figsize=(15, 11))
    plt.matshow(A_plot, fignum=fig.number, vmin=0, vmax=1, cmap="jet")
    plt.title(title_str, fontsize=26, y=1.05)
    ax = fig.axes[0]

    cortical_inner = [13.5, 29.5, 42.5, 53.5, 59.5, 72.5, 99.5,
                      114.5, 133.5, 146.5, 157.5, 163.5, 180.5]
    all_ticks = cortical_inner + SUBCORT_INNER_TICKS_EXP
    plt.xticks(all_ticks)
    plt.yticks(all_ticks)
    plt.grid(color="white", linestyle="-", linewidth=0.5)
    plt.tick_params(axis="both", which="both",
                    bottom=False, top=False, left=False, right=False,
                    labeltop=False, labelleft=False)

    ax.axvline(x=99.5,  color="white", lw=3)
    ax.axhline(y=99.5,  color="white", lw=3)
    ax.axvline(x=199.5, color="white", lw=4)
    ax.axhline(y=199.5, color="white", lw=4)
    for tick in SUBCORT_INNER_TICKS_EXP:
        ax.axvline(x=tick, color="white", lw=2)
        ax.axhline(y=tick, color="white", lw=2)

    cb = plt.colorbar(shrink=0.75)
    cb.ax.tick_params(labelsize=18)

    xmin, xmax, ymin, ymax = plt.axis()
    h     = (ymax - ymin) / 30.0
    space = h / 5.0
    i_top  = ymax + space
    i_left = ymax + space

    for (start, end), color, lbl in zip(CORTICAL_BOUNDS, CORTICAL_COLORS_14, YEO_SHORT_14):
        blk = end - start
        mid = (start + end) / 2.0
        ax.add_patch(patches.Rectangle(
            (start, i_top), blk, h,
            facecolor=color, clip_on=False, linewidth=1.5, edgecolor="k"
        ))
        ax.add_patch(patches.Rectangle(
            (i_left, start + 0.5), h, blk,
            facecolor=color, clip_on=False, linewidth=1.5, edgecolor="k"
        ))
        if blk >= 5:
            ax.text(mid, i_top + h / 2, lbl,
                    ha="center", va="center", fontsize=7,
                    clip_on=False, fontweight="bold")
            ax.text(i_left + h / 2, mid, lbl,
                    ha="center", va="center", fontsize=7,
                    clip_on=False, fontweight="bold", rotation=90)

    for i, (color, lbl) in enumerate(zip(SUBCORT_ROI_COLORS, SUBCORT_ROI_SHORT)):
        start = 199.5 + i * K_EXPAND
        mid   = start + K_EXPAND / 2.0
        ax.add_patch(patches.Rectangle(
            (start, i_top), K_EXPAND, h,
            facecolor=color, clip_on=False, linewidth=1.0, edgecolor="k"
        ))
        ax.add_patch(patches.Rectangle(
            (i_left, start + 0.5), h, K_EXPAND,
            facecolor=color, clip_on=False, linewidth=1.0, edgecolor="k"
        ))
        ax.text(mid, i_top + h / 2, lbl,
                ha="center", va="center", fontsize=5,
                clip_on=False, fontweight="bold")
        ax.text(i_left + h / 2, mid, lbl,
                ha="center", va="center", fontsize=5,
                clip_on=False, fontweight="bold", rotation=90)

    outer_h    = h * 0.85
    outer_top  = i_top  + h + space / 2
    outer_left = i_left + h + space / 2
    for net_name, grp_start, grp_end, net_color in SUBCORT_NET_GROUPS:
        blk = grp_end - grp_start
        mid = (grp_start + grp_end) / 2.0
        ax.add_patch(patches.Rectangle(
            (grp_start, outer_top), blk, outer_h,
            facecolor=net_color, clip_on=False, linewidth=1.2, edgecolor="k"
        ))
        ax.add_patch(patches.Rectangle(
            (outer_left, grp_start + 0.5), outer_h, blk,
            facecolor=net_color, clip_on=False, linewidth=1.2, edgecolor="k"
        ))
        ax.text(mid, outer_top + outer_h / 2, net_name,
                ha="center", va="center", fontsize=6,
                clip_on=False, fontweight="bold")
        ax.text(outer_left + outer_h / 2, mid, net_name,
                ha="center", va="center", fontsize=6,
                clip_on=False, fontweight="bold", rotation=90)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_group(tag):
    mat_path = MLCD_DIR / f"mlcd_combined_{tag}_wins_d05.mat"
    var_key  = f"N_all_g_{tag}"
    with h5py.File(mat_path, "r") as f:
        keys = list(f.keys())
        log.info(f"  [{tag}] h5py keys: {keys}")
        if var_key not in keys:
            var_key = next(k for k in keys if k.startswith("N_all_g"))
        mat = np.squeeze(np.array(f[var_key][()]))
    if mat.shape[0] != N_REGIONS and mat.shape[1] == N_REGIONS:
        mat = mat.T
    wins_per_subj = mat.shape[1] // N_SUBJ
    subjects = [mat[:, i * wins_per_subj:(i + 1) * wins_per_subj] for i in range(N_SUBJ)]
    log.info(f"  [{tag}] shape={mat.shape} | wins/subj={wins_per_subj}")
    return subjects, wins_per_subj


log.info("Loading MLCD outputs …")
subjs_an, W = load_group("anorexia")
subjs_hc, _ = load_group("control")
all_subjects = subjs_an + subjs_hc
N_TOTAL = len(all_subjects)
log.info(f"Total subjects: {N_TOTAL} ({N_SUBJ} AN + {N_SUBJ} HC)")

log.info(f"Computing outcome measures for {N_TOTAL} subjects …")

allegiance_list  = []
recruitment_list = []
integration_list = []
flexibility_list = []
promiscuity_list = []

for s, C in enumerate(all_subjects):
    grp = "AN" if s < N_SUBJ else "HC"
    A = communitymeasures.allegiance(C)
    R = communitymeasures.recruitment(C, static_communities)
    I = communitymeasures.integration(C, static_communities)
    F = communitymeasures.flexibility(C)
    P = communitymeasures.promiscuity(C)
    allegiance_list.append(A)
    recruitment_list.append(R)
    integration_list.append(I)
    flexibility_list.append(F)
    promiscuity_list.append(P)
    log.info(f"  Subject {s+1:02d}/{N_TOTAL} [{grp}] — "
             f"A{A.shape} | R mean={R.mean():.3f} | I mean={I.mean():.3f} | "
             f"F mean={F.mean():.3f} | P mean={P.mean():.3f}")

log.info("Saving per-subject Excel …")
for s in range(N_TOTAL):
    grp         = "AN" if s < N_SUBJ else "HC"
    subj_in_grp = (s % N_SUBJ) + 1
    out_path    = OUT_METRICS / f"combined_{grp}_subj{subj_in_grp:02d}_metrics.xlsx"
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as w:
        pd.DataFrame({"Recruitment": recruitment_list[s]},
                     index=roi_index).to_excel(w, sheet_name="Recruitment")
        pd.DataFrame({"Integration": integration_list[s]},
                     index=roi_index).to_excel(w, sheet_name="Integration")
        pd.DataFrame({"Flexibility": flexibility_list[s]},
                     index=roi_index).to_excel(w, sheet_name="Flexibility")
        pd.DataFrame({"Promiscuity": promiscuity_list[s]},
                     index=roi_index).to_excel(w, sheet_name="Promiscuity")
        pd.DataFrame(allegiance_list[s],
                     index=roi_index, columns=roi_index).to_excel(w, sheet_name="Allegiance")
    log.info(f"  Saved {out_path.name}")

for tag, idxs in [("anorexia", range(N_SUBJ)), ("control", range(N_SUBJ, N_TOTAL))]:
    rec_mean  = np.mean([recruitment_list[s]  for s in idxs], axis=0)
    int_mean  = np.mean([integration_list[s]  for s in idxs], axis=0)
    flex_mean = np.mean([flexibility_list[s]  for s in idxs], axis=0)
    prom_mean = np.mean([promiscuity_list[s]  for s in idxs], axis=0)
    atlas_col = ["Cortical"] * N_CORTICAL + ["Subcortical"] * N_SUBCORT
    yeo_col   = [YEO_NAMES.get(int(static_communities[i]), "?") for i in range(N_REGIONS)]
    df = pd.DataFrame({
        "ROI":              all_labels,
        "Atlas":            atlas_col,
        "Yeo7_Network":     yeo_col,
        "Recruitment_mean": rec_mean,
        "Integration_mean": int_mean,
        "Flexibility_mean": flex_mean,
        "Promiscuity_mean": prom_mean,
    })
    df.to_excel(OUT_XLSX / f"combined_{tag}_group_means.xlsx", index=False)
    log.info(f"  Saved combined_{tag}_group_means.xlsx")

log.info("Saving allegiance PNG figures …")
for s in range(N_TOTAL):
    grp         = "AN" if s < N_SUBJ else "HC"
    subj_in_grp = (s % N_SUBJ) + 1
    title = f"{grp} | Subject #{subj_in_grp}"
    fname = f"allegiance_combined_{grp}_subj{subj_in_grp:02d}.png"
    _save_allegiance_figure(allegiance_list[s], title, OUT_FIG / fname)
    log.info(f"  {fname}")

log.info("Saving group-mean allegiance figures …")
for tag, idxs, grp_label in [
    ("anorexia", range(N_SUBJ),          "Anorexia"),
    ("control",  range(N_SUBJ, N_TOTAL), "Control"),
]:
    A_mean = np.mean([allegiance_list[s] for s in idxs], axis=0)
    title  = f"{grp_label} | Group Mean (N={N_SUBJ})  [Schaefer-200 + Tian S1]"
    fname  = f"allegiance_combined_{tag}_groupmean.png"
    _save_allegiance_figure(A_mean, title, OUT_FIG / fname)
    log.info(f"  {fname}")

log.info(f"STAGE 3c COMPLETE — duration: {datetime.now() - start_time}")
