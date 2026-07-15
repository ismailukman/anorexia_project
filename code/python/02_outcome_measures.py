#!/usr/bin/env python3
"""
02_outcome_measures.py — Outcome Measures for Multi-layer Community Detection.
Runs after MATLAB MLCD processing (cons_mlcd_win_parallel_an_v2.m).
"""
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from nilearn import datasets
from teneto import communitymeasures

start_time = datetime.now()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

log.info("STAGE 3 — Outcome Measures (allegiance, recruitment, integration)")

# --- Paths ---
PROJECT_ROOT = Path("/Users/ismaila/Documents/C-Codes/AnorexiaProject")
BASE_DIR = PROJECT_ROOT / "data" / "analysis" / "mlcd_subjs"
MLCD_DIR = BASE_DIR / "subjs_mlcd"
SAVE_FIG_DIR = PROJECT_ROOT / "output" / "figures" / "stage2_mlcd"

FNAME_TEMPLATE = "mlcd_{tag}_wins.mat"
GROUP_TAGS = ["anorexia", "control"]

N_EXPECTED = 200
N_SUBJ_PER_GROUP = 22
N_SUBJ_TOTAL = N_SUBJ_PER_GROUP * len(GROUP_TAGS)

VAR_BASE = "N_all_g"   # "N_all_g" or "S_g"

XLSX_OUT_DIR = PROJECT_ROOT / "output" / "results" / "xlsx_exports"
XLSX_OUT_DIR.mkdir(parents=True, exist_ok=True)

YEO7_COLORS = [
    "#A251AC", "#789AC1", "#409832", "#E165FE",
    "#F6FDC9", "#EFB944", "#D9717D",
]

NETWORK_BOUNDARIES = [
    (-0.5,  13.5), (13.5,  29.5), (29.5,  42.5), (42.5,  53.5),
    (53.5,  59.5), (59.5,  72.5), (72.5,  99.5),   # LH
    (99.5, 114.5), (114.5, 133.5), (133.5, 146.5), (146.5, 157.5),
    (157.5, 163.5), (163.5, 180.5), (180.5, 199.5), # RH
]

def _key_candidates(var_base: str, tag: str):
    """Return possible dataset names inside the v7.3 .mat (HDF5) file."""
    return [f"{var_base}_{tag}", f"{var_base}{tag}", f"{var_base}__{tag}"]


def _metrics_candidates(tag: str):
    """Candidate names for per-group metrics."""
    return {
        "Q_g": [f"Q_g_{tag}", f"Q_g{tag}", f"Q_{tag}", "Q_g"],
        "comm_cons_all_g": [
            f"comm_cons_all_g_{tag}", f"comm_cons_all_g{tag}",
            f"comm_cons_{tag}", "comm_cons_all_g",
        ],
        "S_g": [f"S_g_{tag}", f"S_g{tag}", f"S_{tag}", "S_g"],
    }


def read_var_any(f: h5py.File, candidates):
    """Read first matching dataset among candidates."""
    keys = set(f.keys())
    for name in candidates:
        if name in keys:
            arr = f[name][()]
            return np.squeeze(np.asarray(arr)), name
        slash = f"/{name}"
        if slash in keys:
            arr = f[slash][()]
            return np.squeeze(np.asarray(arr)), slash
    raise KeyError(f"None of {candidates} found. Keys: {list(f.keys())}")


def fix_orientation(mat: np.ndarray, n_expected: int) -> np.ndarray:
    """Ensure matrix is shaped (N, T). Transpose if (T, N)."""
    mat = np.asarray(mat)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {mat.shape}")
    r, c = mat.shape
    if r == n_expected:
        return mat
    if c == n_expected:
        return mat.T
    raise ValueError(f"Neither dim matches N={n_expected}. Got {mat.shape}")


def safe_mode_int(arr_1d):
    """Compute mode of an integer array."""
    vals, counts = np.unique(arr_1d.astype(int), return_counts=True)
    return int(vals[np.argmax(counts)]) if len(vals) else None


def split_into_subjects(primary_mat: np.ndarray, n_subj: int, n_expected: int):
    """Split (N, total_cols) -> list of n_subj arrays, each (N, wins_per_subj)."""
    primary_mat = fix_orientation(primary_mat, n_expected)
    N, total_cols = primary_mat.shape
    if N != n_expected:
        raise ValueError(f"Expected N={n_expected}, got N={N}")
    if total_cols % n_subj != 0:
        raise ValueError(f"total_cols={total_cols} not divisible by n_subj={n_subj}")
    wins_per_subj = total_cols // n_subj
    subjects = [
        primary_mat[:, i * wins_per_subj : (i + 1) * wins_per_subj]
        for i in range(n_subj)
    ]
    return subjects, wins_per_subj


def load_group_file(tag: str):
    """Load VAR_BASE and supplementary metrics from one group .mat file."""
    mat_path = MLCD_DIR / FNAME_TEMPLATE.format(tag=tag)
    if not mat_path.exists():
        raise FileNotFoundError(f"File not found: {mat_path}")

    out = {"tag": tag, "mat_path": mat_path}

    with h5py.File(mat_path, "r") as f:
        primary, found_primary = read_var_any(f, _key_candidates(VAR_BASE, tag))
        primary = fix_orientation(primary, N_EXPECTED)
        out[VAR_BASE] = primary
        out["_found_primary_name"] = found_primary

        metric_names = _metrics_candidates(tag)
        for std_name, cands in metric_names.items():
            try:
                arr, found = read_var_any(f, cands)
                if std_name == "S_g" and arr.ndim == 2:
                    arr = fix_orientation(arr, N_EXPECTED)
                out[std_name] = np.squeeze(np.asarray(arr))
                out[f"_found_{std_name}_name"] = found
            except KeyError:
                pass

    subjects, wins_per_subj = split_into_subjects(
        out[VAR_BASE], N_SUBJ_PER_GROUP, N_EXPECTED
    )
    out["subjects"] = subjects
    out["wins_per_subj"] = wins_per_subj
    return out


groups = {}
all_subjects = []
communities = []

log.info("Loading MLCD group files …")
for tag in GROUP_TAGS:
    g = load_group_file(tag)
    groups[tag] = g

    log.info(
        f"  Loaded {g['_found_primary_name']} from {g['mat_path'].name}: "
        f"{g[VAR_BASE].shape}"
    )
    log.info(f"  wins_per_subj={g['wins_per_subj']} | subjects={len(g['subjects'])} each {g['subjects'][0].shape}")

    if "Q_g" in g:
        q = np.asarray(g["Q_g"]).squeeze()
        if q.ndim == 0:
            log.info(f"  Q_g: {float(q):.4f}")
        else:
            log.info(f"  Q_g: mean={float(np.mean(q)):.4f} (shape {q.shape})")
    else:
        log.info("  Q_g: (not available)")

    if "comm_cons_all_g" in g:
        cc = np.asarray(g["comm_cons_all_g"]).squeeze()
        log.info(f"  comm_cons_all_g: shape {cc.shape} | mode={safe_mode_int(cc)}")

    xlsx_file = XLSX_OUT_DIR / f"{VAR_BASE}_{tag}.xlsx"
    pd.DataFrame(g[VAR_BASE]).to_excel(xlsx_file, index=True, header=True)
    log.info(f"  Saved [{VAR_BASE}_{tag}] → {xlsx_file.name}")

    all_subjects.extend(g["subjects"])

communities = all_subjects

assert len(all_subjects) == N_SUBJ_TOTAL, (
    f"Expected {N_SUBJ_TOTAL} total subjects, got {len(all_subjects)}"
)
log.info(f"Total subjects loaded: {len(all_subjects)} ({N_SUBJ_PER_GROUP} AN + {N_SUBJ_PER_GROUP} HC)")

for tag in GROUP_TAGS:
    log.info("-" * 40)
    log.info(f"GROUP SUMMARY: {tag}")

    if "comm_cons_all_g" in groups[tag]:
        comm_arr = np.asarray(groups[tag]["comm_cons_all_g"]).squeeze()
        if "comm_cons_all_g_mode" not in groups[tag]:
            vals, counts = np.unique(comm_arr, return_counts=True)
            groups[tag]["comm_cons_all_g_mode"] = int(vals[np.argmax(counts)])
        mode_comm = groups[tag]["comm_cons_all_g_mode"]
        log.info(f"  comm_cons_all_g mode: {mode_comm}")
    else:
        log.info("  comm_cons_all_g: (not available)")

    if "Q_g" in groups[tag]:
        q_arr = np.asarray(groups[tag]["Q_g"]).squeeze()
        if q_arr.ndim == 0:
            log.info(f"  Q_g: {float(q_arr):.4f}")
        else:
            q_mean = float(np.mean(q_arr))
            groups[tag]["Q_g_mean"] = q_mean
            log.info(f"  Q_g mean: {q_mean:.4f} (n={q_arr.size})")
    else:
        log.info("  Q_g: (not available)")


assert isinstance(communities, list), "`communities` must be a list"
assert all(
    isinstance(C, np.ndarray) and C.shape[0] == 200 for C in communities
), "Each communities[s] must be an array with 200 rows (nodes)."

n_set = len(communities)
log.info(f"Computing outcome measures for {n_set} subjects …")

# Atlas labels (skip 'Background')
atlas = datasets.fetch_atlas_schaefer_2018(
    n_rois=200, yeo_networks=7, resolution_mm=1,
    data_dir=str(PROJECT_ROOT / "data" / "atlas" / "schaefer_2018"),
)
labels = np.array(atlas.labels[1:]).astype("U")

networks = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
static_communities = np.zeros((200,), dtype=int)
for i, network in enumerate(networks):
    idx = np.array([network in s for s in labels], dtype=bool)
    static_communities[idx] = i + 1  # 1-7

pivot = np.where(static_communities[:-1] != static_communities[1:])[0]
pivot = np.concatenate([pivot, [199]])


def create_coarse_allegiance(alleg: np.ndarray) -> np.ndarray:
    """Reduce 200x200 allegiance to 7x7 network-level allegiance."""
    allegiance_coarse_lr = np.zeros((14, 14))
    p1, q1 = 0, 0
    for _i, p2 in enumerate(pivot):
        for _j, q2 in enumerate(pivot):
            allegiance_coarse_lr[_i, _j] = np.nanmean(alleg[p1 : p2 + 1, q1 : q2 + 1])
            q1 = q2 + 1
        p1 = p2 + 1
        q1 = 0
    allegiance_coarse = np.mean(
        allegiance_coarse_lr.reshape(2, 7, 2, 7).transpose(0, 2, 1, 3).reshape(-1, 7, 7),
        axis=0,
    )
    return allegiance_coarse


allegiance_list = []
integration_list = []
recruitment_list = []
allegiance_coarse_list = []
flexibility_list = []
promiscuity_list = []

for s in range(n_set):
    C = communities[s]

    A = communitymeasures.allegiance(C)
    I = communitymeasures.integration(C, static_communities)
    R = communitymeasures.recruitment(C, static_communities)
    F = communitymeasures.flexibility(C)
    P = communitymeasures.promiscuity(C)

    allegiance_list.append(A)
    integration_list.append(I)
    recruitment_list.append(R)
    allegiance_coarse_list.append(create_coarse_allegiance(A))
    flexibility_list.append(F)
    promiscuity_list.append(P)

    grp_lbl = "AN" if s < 22 else "HC"
    log.info(
        f"  Subject {s + 1:02d}/44 [{grp_lbl}] — "
        f"allegiance {A.shape} | R {R.shape} | I {I.shape} | "
        f"F mean={F.mean():.3f} | P mean={P.mean():.3f}"
    )


OUT_DIR_FULL = PROJECT_ROOT / "output" / "results" / "subject_metrics"
OUT_DIR_FULL.mkdir(parents=True, exist_ok=True)

if isinstance(labels, np.ndarray) and len(labels) == 200:
    roi_index = pd.Index(labels, name="ROI")
else:
    roi_index = pd.Index([f"ROI_{i:03d}" for i in range(1, 201)], name="ROI")

assert len(recruitment_list) == n_set == 44
assert len(integration_list) == 44 and len(allegiance_list) == 44
assert len(flexibility_list) == 44 and len(promiscuity_list) == 44
for s in range(44):
    assert recruitment_list[s].shape == (200,)
    assert integration_list[s].shape == (200,)
    assert allegiance_list[s].shape == (200, 200)
    assert flexibility_list[s].shape == (200,)
    assert promiscuity_list[s].shape == (200,)

for s in range(44):
    R = np.asarray(recruitment_list[s]).reshape(-1)
    I = np.asarray(integration_list[s]).reshape(-1)
    A = np.asarray(allegiance_list[s])
    F = np.asarray(flexibility_list[s]).reshape(-1)
    P = np.asarray(promiscuity_list[s]).reshape(-1)

    df_R = pd.DataFrame({"Recruitment": R}, index=roi_index)
    df_I = pd.DataFrame({"Integration": I}, index=roi_index)
    df_A = pd.DataFrame(A, index=roi_index, columns=roi_index)
    df_F = pd.DataFrame({"Flexibility": F}, index=roi_index)
    df_P = pd.DataFrame({"Promiscuity": P}, index=roi_index)

    out_path = OUT_DIR_FULL / f"subject_{s + 1:02d}_metrics.xlsx"
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        df_R.to_excel(writer, sheet_name="Recruitment")
        df_I.to_excel(writer, sheet_name="Integration")
        df_A.to_excel(writer, sheet_name="Allegiance")
        df_F.to_excel(writer, sheet_name="Flexibility")
        df_P.to_excel(writer, sheet_name="Promiscuity")
    log.info(f"  Saved subject {s + 1:02d}/44 metrics → {out_path.name}")


def _set_to_group(s_idx: int):
    """Map 0-based subject index to group label (22 AN + 22 HC)."""
    set_num = s_idx + 1
    if 1 <= set_num <= 22:
        grp_label, grp_start = "Anorexia", 1
    elif 23 <= set_num <= 44:
        grp_label, grp_start = "Control", 23
    else:
        grp_label, grp_start = "(Unlabeled)", set_num
    subj_in_group = set_num - grp_start + 1
    return grp_label, set_num, subj_in_group


def draw_network_patches_200(ax, orientation="both"):
    """Draw Yeo-7 network colour bars on top and/or right side of 200-region matrix."""
    colors_14 = YEO7_COLORS * 2  # 7 LH + 7 RH

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    h = (ymax - ymin) / 30.0
    space = h / 5.0
    w = (ymax - ymin) / 30.0

    if orientation in ("both", "top"):
        i_marker = ymax + space
        for (start, end), color in zip(NETWORK_BOUNDARIES, colors_14):
            ax.add_patch(
                patches.Rectangle(
                    (start, i_marker), end - start, h,
                    facecolor=color, clip_on=False, linewidth=1.5, edgecolor="k",
                )
            )

    if orientation in ("both", "right"):
        i_marker2 = ymax
        for (start, end), color in zip(NETWORK_BOUNDARIES, colors_14):
            ax.add_patch(
                patches.Rectangle(
                    (i_marker2 + space, start + 0.5), w, end - start,
                    facecolor=color, clip_on=False, linewidth=1.5, edgecolor="k",
                )
            )


s_idx = 0
i, j = 22, 18

grp_label, set_num, subj_in_group = _set_to_group(s_idx)
A_sel = allegiance_list[s_idx]

print(
    f"[{grp_label}] Allegiance[set {set_num} | subj {subj_in_group}]"
    f"[{i},{j}] = {A_sel[i, j]:.4f}"
)

cmap = "jet"
f = plt.figure(figsize=(15, 11))
plt.matshow(A_sel, fignum=f.number, vmin=0, vmax=1, cmap=cmap)
plt.title(f"Anorexia | Subject #1", fontsize=26, y=1.05)
plt.xticks(
    [13.5, 29.5, 42.5, 53.5, 59.5, 72.5, 99.5,
     114.5, 133.5, 146.5, 157.5, 163.5, 180.5]
)
plt.yticks(
    [13.5, 29.5, 42.5, 53.5, 59.5, 72.5, 99.5,
     114.5, 133.5, 146.5, 157.5, 163.5, 180.5]
)
cb = plt.colorbar(shrink=0.75)
cb.ax.tick_params(labelsize=18)
plt.axvline(x=100 - 0.5, color="white", linewidth=3)
plt.axhline(y=100 - 0.5, color="white", linewidth=3)
plt.grid(color="white", linestyle="-", linewidth=0.7)
plt.tick_params(
    axis="both", which="both", bottom=False, top=False,
    left=False, right=False, labeltop=False, labelleft=False,
)

colors_14 = YEO7_COLORS * 2
xmin, xmax, ymin, ymax = plt.axis()
h = (ymax - ymin) / 30
space = h / 5
i_marker = ymax + space

for (start, end), color in zip(NETWORK_BOUNDARIES, colors_14):
    plt.gca().add_patch(
        patches.Rectangle(
            (start, i_marker), end - start, h,
            facecolor=color, clip_on=False, linewidth=1.5, edgecolor="k",
        )
    )

w = (ymax - ymin) / 30
i_marker2 = ymax
for (start, end), color in zip(NETWORK_BOUNDARIES, colors_14):
    plt.gca().add_patch(
        patches.Rectangle(
            (i_marker2 + space, start + 0.5), w, end - start,
            facecolor=color, clip_on=False, linewidth=1.5, edgecolor="k",
        )
    )

plt.show()


s_idx1 = 0
s_idx2 = 22
i, j = 18, 20

A_sel1 = allegiance_list[s_idx1]
A_sel2 = allegiance_list[s_idx2]

grp_label1, set_num1, subj_in_group1 = _set_to_group(s_idx1)
grp_label2, set_num2, subj_in_group2 = _set_to_group(s_idx2)

print(
    f"Subj {set_num1} {grp_label1} vs Subj {set_num2} {grp_label2}; "
    f"Node [{i},{j}] = {A_sel1[i, j]:.4f} vs {A_sel2[i, j]:.4f}"
)

cmap = "jet"
fig, axes = plt.subplots(1, 2, figsize=(26, 11))

for ax, A_sel, (grp_label, set_num_v, subj_in_group_v) in zip(
    axes,
    [A_sel1, A_sel2],
    [
        (grp_label1, set_num1, subj_in_group1),
        (grp_label2, set_num2, subj_in_group2),
    ],
):
    im = ax.imshow(A_sel, vmin=0, vmax=1, cmap=cmap)
    A_clean = np.nan_to_num(A_sel, nan=0)
    ax.set_title(
        f"{grp_label} | Subject {set_num_v} "
        f"mean:{np.mean(A_clean):.4f} +/- {np.std(A_clean):.4f}",
        fontsize=28, pad=35,
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.ax.tick_params(labelsize=14)
    ax.axvline(x=100 - 0.5, color="white", linewidth=3)
    ax.axhline(y=100 - 0.5, color="white", linewidth=3)
    ax.grid(color="white", linestyle="-", linewidth=0.7)
    ax.tick_params(
        axis="both", which="both", bottom=False, top=False,
        left=False, right=False, labeltop=False, labelleft=False,
    )

    xmin_ax, xmax_ax = ax.get_xlim()
    ymin_ax, ymax_ax = ax.get_ylim()
    h = (ymax_ax - ymin_ax) / 30.0
    space = h / 5.0

    i_marker = ymax_ax + space
    for (start, end), color in zip(NETWORK_BOUNDARIES, colors_14):
        ax.add_patch(
            patches.Rectangle(
                (start, i_marker), end - start, h,
                facecolor=color, clip_on=False, linewidth=1.5, edgecolor="k",
            )
        )

    w = (ymax_ax - ymin_ax) / 30.0
    i_marker2 = ymax_ax
    for (start, end), color in zip(NETWORK_BOUNDARIES, colors_14):
        ax.add_patch(
            patches.Rectangle(
                (i_marker2 + space, start + 0.5), w, end - start,
                facecolor=color, clip_on=False, linewidth=1.5, edgecolor="k",
            )
        )

plt.tight_layout()
plt.show()


out_dir_alleg = PROJECT_ROOT / "output" / "figures" / "stage2_mlcd" / "allegiance_200_regions"
out_dir_alleg.mkdir(parents=True, exist_ok=True)

i, j = 22, 18

log.info("Saving batch allegiance PNG figures (200-region, all 44 subjects) …")

for s_idx_b in range(44):
    grp_label, set_num_b, subj_in_group_b = _set_to_group(s_idx_b)
    A_sel = allegiance_list[s_idx_b]

    cmap = "jet"
    f_fig = plt.figure(figsize=(15, 11))
    plt.matshow(A_sel, fignum=f_fig.number, vmin=0, vmax=1, cmap=cmap)
    plt.title(f"{grp_label} | Subject #{subj_in_group_b}", fontsize=26, y=1.05)
    plt.xticks(
        [13.5, 29.5, 42.5, 53.5, 59.5, 72.5, 99.5,
         114.5, 133.5, 146.5, 157.5, 163.5, 180.5]
    )
    plt.yticks(
        [13.5, 29.5, 42.5, 53.5, 59.5, 72.5, 99.5,
         114.5, 133.5, 146.5, 157.5, 163.5, 180.5]
    )
    cb = plt.colorbar(shrink=0.75)
    cb.ax.tick_params(labelsize=18)
    plt.axvline(x=100 - 0.5, color="white", linewidth=3)
    plt.axhline(y=100 - 0.5, color="white", linewidth=3)
    plt.grid(color="white", linestyle="-", linewidth=0.7)
    plt.tick_params(
        axis="both", which="both", bottom=False, top=False,
        left=False, right=False, labeltop=False, labelleft=False,
    )

    xmin, xmax, ymin, ymax = plt.axis()
    h = (ymax - ymin) / 30
    space = h / 5
    i_marker = ymax + space

    for (start, end), color in zip(NETWORK_BOUNDARIES, colors_14):
        plt.gca().add_patch(
            patches.Rectangle(
                (start, i_marker), end - start, h,
                facecolor=color, clip_on=False, linewidth=1.5, edgecolor="k",
            )
        )

    w = (ymax - ymin) / 30
    i_marker2 = ymax
    for (start, end), color in zip(NETWORK_BOUNDARIES, colors_14):
        plt.gca().add_patch(
            patches.Rectangle(
                (i_marker2 + space, start + 0.5), w, end - start,
                facecolor=color, clip_on=False, linewidth=1.5, edgecolor="k",
            )
        )

    fname = f"{grp_label.replace(' ', '')}_set{set_num_b:02d}_subj{subj_in_group_b:02d}.png"
    plt.savefig(out_dir_alleg / fname, dpi=300, bbox_inches="tight")
    plt.close(f_fig)
    log.info(f"  Saved allegiance PNG {s_idx_b + 1}/44 [{grp_label}]: {fname}")


end_time = datetime.now()
log.info(f"STAGE 3 COMPLETE — duration: {end_time - start_time}")


# # Provided values
# lems_cfes_pre   = np.array([5, 8, 19, 43, 0, 6, 5], dtype=float)   # 7
# lems_cfes_post  = np.array([8, 11, 17, 45, 0, 14, 6], dtype=float) # 7
# lems_pass_pre   = np.array([0, 39, 4, 9, 0], dtype=float)          # 5
# lems_pass_post  = np.array([0, 37, 6, 11, 3], dtype=float)         # 5