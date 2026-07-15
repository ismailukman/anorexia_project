#!/usr/bin/env python3
"""
01_data_preparation.py — sliding-window Pearson FC for MLCD (Anorexia / Control).
Outputs per-subject FC tensors (.mat) and QC figures using Schaefer-200 Yeo-7.
"""

import logging
import os
import gc
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

from nilearn import datasets, plotting
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from scipy.io import savemat


@dataclass
class Config:
    """Pipeline configuration."""
    data_path: Path
    groups: List[str]
    tr: float = 0.8                   # seconds
    window_length_sec: float = 30.0   # seconds
    step_size_tr: int = 1             # in TRs  (1 or 19)
    fmri_prefix: str = "errts"        # file prefix
    expect_regions: int = 200         # atlas parcels
    runs_per_subject: int = 1
    verbose: bool = True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path("/Users/ismaila/Documents/C-Codes/AnorexiaProject")

cfg = Config(
    data_path=PROJECT_ROOT / "data",
    groups=["an_patients", "hc_patients"],
    tr=0.8,
    window_length_sec=30.0,
    step_size_tr=1,             # 1 TR step  (change to 19 for 50 % overlap)
    fmri_prefix="errts",
    expect_regions=200,
    runs_per_subject=1,
    verbose=True,
)

SAVE_FIG_DIR  = PROJECT_ROOT / "output" / "figures" / "stage1_fc"
ANALYSIS_DIR  = PROJECT_ROOT / "data" / "analysis"
TS_CACHE_DIR  = ANALYSIS_DIR / "timeseries_subjs"   # cached (T×200) per subject

log.info("STAGE 1 — FC Data Preparation (Schaefer-200)")
log.info(f"Data path : {cfg.data_path}")
log.info(f"Groups    : {cfg.groups}")
log.info(f"TR={cfg.tr}s | Window={cfg.window_length_sec}s | Step={cfg.step_size_tr} TR")

start_time = datetime.now()

ATLAS_DIR = PROJECT_ROOT / "data" / "atlas" / "schaefer_2018"
ATLAS_DIR.mkdir(parents=True, exist_ok=True)

atlas = datasets.fetch_atlas_schaefer_2018(
    n_rois=cfg.expect_regions, yeo_networks=7, resolution_mm=1,
    data_dir=str(ATLAS_DIR),
)
atlas_filename = atlas.maps
labels = np.asarray(atlas.labels).astype("U")

# Yeo-7 inspired palette (7 colours)
YEO7_COLORS = [
    "#A251AC", "#789AC1", "#409832", "#E165FE",
    "#F6FDC9", "#EFB944", "#D9717D",
]
cmap_yeo = ListedColormap(YEO7_COLORS)

disp = plotting.plot_roi(atlas_filename, cmap=cmap_yeo)
log.info(f"Atlas loaded: {atlas_filename}")

def build_masker(atlas_file: str, tr: float) -> NiftiLabelsMasker:
    """Build a NiftiLabelsMasker with z-score standardisation (fallback to bool)."""
    _cache = str(PROJECT_ROOT / "nilearn_cache")
    try:
        masker = NiftiLabelsMasker(
            labels_img=atlas_file,
            standardize="zscore_sample",
            standardize_confounds=True,
            memory=_cache,
            verbose=0,
            t_r=tr,
        )
    except TypeError:
        masker = NiftiLabelsMasker(
            labels_img=atlas_file,
            standardize=True,
            standardize_confounds=True,
            memory=_cache,
            verbose=0,
            t_r=tr,
        )
    return masker


masker = build_masker(atlas_filename, cfg.tr)
connectome = ConnectivityMeasure(kind="correlation", standardize="zscore_sample")

log.info(f"Masker ready (TR={cfg.tr}s, zscore_sample) | FC: Pearson correlation")

def list_group_niis(group_dir: str, prefix: str) -> List[str]:
    """Return sorted list of NIfTI files matching *prefix* in *group_dir*."""
    patt1 = os.path.join(group_dir, f"{prefix}*.nii")
    patt2 = os.path.join(group_dir, f"{prefix}*.nii.gz")
    return sorted(glob.glob(patt1) + glob.glob(patt2))


def pair_runs_consecutively(
    run_paths: List[str], runs_per_subject: int = 2
) -> List[Tuple[str, ...]]:
    """Pair consecutive runs into subject-level tuples."""
    if len(run_paths) % runs_per_subject != 0:
        raise ValueError(
            f"Found {len(run_paths)} runs, not divisible by "
            f"runs_per_subject={runs_per_subject}."
        )
    pairs = []
    for i in range(0, len(run_paths), runs_per_subject):
        pairs.append(tuple(run_paths[i : i + runs_per_subject]))
    return pairs


group_runs: Dict[str, List[str]] = {}
group_pairs: Dict[str, List[Tuple[str, ...]]] = {}

log.info("Scanning groups for raw NIfTI files …")
for grp in cfg.groups:
    grp_dir = os.path.join(cfg.data_path, grp)
    if not os.path.isdir(grp_dir):
        raise FileNotFoundError(f"Group directory not found: {grp_dir}")

    runs = list_group_niis(grp_dir, cfg.fmri_prefix)
    group_runs[grp] = runs
    pairs = pair_runs_consecutively(runs, cfg.runs_per_subject)
    group_pairs[grp] = pairs

    log.info(f"  [{grp}] {len(runs)} runs → {len(pairs)} subjects")

def extract_run_timeseries(nii_path: str) -> np.ndarray:
    """Extract atlas-based time series from a single NIfTI file."""
    ts = masker.fit_transform(nii_path)  # (T, R)
    if ts.ndim != 2:
        raise ValueError(f"Timeseries not 2D for {nii_path}: {ts.shape}")
    if ts.shape[1] != cfg.expect_regions:
        raise ValueError(
            f"Expected {cfg.expect_regions} regions, got {ts.shape[1]} in {nii_path}"
        )
    return ts


def concat_subject_runs(run_ts_list: List[np.ndarray]) -> np.ndarray:
    """Concatenate multiple runs along the time axis."""
    region_counts = {ts.shape[1] for ts in run_ts_list}
    if len(region_counts) != 1:
        raise ValueError(f"Region mismatch across runs: {region_counts}")
    return np.concatenate(run_ts_list, axis=0)


def sliding_window_indices(
    n_timepoints: int, window_trs: int, step_trs: int
) -> List[Tuple[int, int]]:
    """Return (start, end) index pairs for each sliding window."""
    idx = []
    s = 0
    while s + window_trs <= n_timepoints:
        idx.append((s, s + window_trs))
        s += step_trs
    return idx


def timeseries_to_window_list(
    ts: np.ndarray, window_trs: int, step_trs: int
) -> List[np.ndarray]:
    """Slice a time-series array into overlapping windows."""
    idx = sliding_window_indices(ts.shape[0], window_trs, step_trs)
    return [ts[s:e, :] for (s, e) in idx]


def compute_fc_per_window(windows: List[np.ndarray]) -> np.ndarray:
    """Compute Pearson-correlation FC per window (no Fisher Z-transform)."""
    if len(windows) == 0:
        return np.zeros((0, 0, 0), dtype=np.float32)
    fc_list = connectome.fit_transform(windows)  # (n_windows, R, R)
    return np.asarray(fc_list, dtype=np.float32)


window_trs = int(round(cfg.window_length_sec / cfg.tr))
step_trs = int(cfg.step_size_tr)

if window_trs < 2:
    raise ValueError("window_trs must be >= 2.")
if step_trs < 1:
    raise ValueError("step_trs must be >= 1.")

expected_trs_per_run = int(round(9.3 * 60 / cfg.tr))  # ~9.3 min per run
log.info(f"Window={window_trs} TRs ({cfg.window_length_sec}s @ TR={cfg.tr}s) | step={step_trs} TRs | expected TRs/run≈{expected_trs_per_run}")

TS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_grp(grp: str) -> str:
    s = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in grp)
    return s if s and s[0].isalpha() else f"grp_{s}"


def load_or_extract_subject_ts(grp: str, si: int, pair: Tuple[str, ...]) -> np.ndarray:
    """Return (T, R) timeseries for one subject.

    First call: runs NiftiLabelsMasker on the raw NIfTI and saves a .npy cache.
    Subsequent calls: loads the .npy directly, skipping the NIfTI entirely.
    Cache location: data/analysis/timeseries_subjs/ts_<group>_subj<NN>.npy
    """
    cache_file = TS_CACHE_DIR / f"ts_{_safe_grp(grp)}_subj{si:02d}.npy"

    if cache_file.exists():
        ts = np.load(str(cache_file))
        log.info(f"  [{grp}] Subject {si:02d} — loaded from cache {cache_file.name} {ts.shape}")
        return ts

    run_ts_list = []
    for rp in pair:
        log.info(f"  [{grp}] Subject {si:02d} — extracting NIfTI: {os.path.basename(rp)}")
        ts_run = extract_run_timeseries(rp)
        if abs(ts_run.shape[0] - expected_trs_per_run) > 10 and cfg.verbose:
            log.warning(
                f"  [{grp}] subj {si}: {os.path.basename(rp)} "
                f"timepoints={ts_run.shape[0]} (expected ~{expected_trs_per_run})"
            )
        run_ts_list.append(ts_run)

    ts_concat = concat_subject_runs(run_ts_list).astype(np.float32)
    np.save(str(cache_file), ts_concat)
    log.info(f"  [{grp}] Subject {si:02d} — TS saved to cache {cache_file.name} {ts_concat.shape}")
    return ts_concat


test_grp = cfg.groups[0]
if len(group_pairs[test_grp]) == 0:
    raise RuntimeError(f"No subject pairings in group: {test_grp}")

test_pair = group_pairs[test_grp][0]

log.info(f"Test run on group '{test_grp}', subject 1 …")
ts_concat = load_or_extract_subject_ts(test_grp, 1, test_pair)
log.info(f"  TS shape: {ts_concat.shape}")

windows = timeseries_to_window_list(ts_concat, window_trs, step_trs)
log.info(f"  Windows: {len(windows)}")

fc_w = compute_fc_per_window(windows)
log.info(f"  FC shape: {fc_w.shape}")

assert fc_w.ndim == 3 and fc_w.shape[1] == cfg.expect_regions and fc_w.shape[2] == cfg.expect_regions
log.info("Test subject OK")


group_fc_results: Dict[str, np.ndarray] = {}
log.info("Processing all groups — full pipeline")

for grp in cfg.groups:
    pairs = group_pairs[grp]
    if len(pairs) == 0:
        log.info(f"[{grp}] No subjects found; skipping.")
        continue

    subj_fc_list: List[np.ndarray] = []
    n_windows_ref = None

    log.info(f"[{grp}] {len(pairs)} subjects")
    for si, pair in enumerate(pairs, start=1):
        ts_concat = load_or_extract_subject_ts(grp, si, pair)
        windows = timeseries_to_window_list(ts_concat, window_trs, step_trs)
        if len(windows) == 0:
            raise ValueError(
                f"No windows for subject {si} in group {grp} "
                f"(T={ts_concat.shape[0]}, window_trs={window_trs})."
            )

        fc_w = compute_fc_per_window(windows)

        if n_windows_ref is None:
            n_windows_ref = fc_w.shape[0]
        elif fc_w.shape[0] != n_windows_ref:
            raise ValueError(
                f"[{grp}] Inconsistent window counts across subjects: "
                f"first had {n_windows_ref}, now {fc_w.shape[0]}"
            )

        subj_fc_list.append(fc_w)
        log.info(
            f"  [{grp}] Subject {si}/{len(pairs)} — "
            f"TS {ts_concat.shape} | {fc_w.shape[0]} windows | FC {fc_w.shape}"
        )

    grp_tensor = np.stack(subj_fc_list, axis=0).astype(np.float32)
    group_fc_results[grp] = grp_tensor
    log.info(
        f"[{grp}] FC tensor complete: {grp_tensor.shape} "
        f"(subjects × windows × {cfg.expect_regions} × {cfg.expect_regions})"
    )

log.info("All groups processed — FC extraction done")

def describe_group(group_key: str, result_map: Dict[str, np.ndarray]):
    arr = result_map[group_key]
    n_subj, n_win, r1, r2 = arr.shape
    print(f"Group={group_key}: subjects={n_subj}, windows={n_win}, regions={r1}x{r2}")
    return n_subj, n_win, r1, r2


for g in cfg.groups:
    describe_group(g, group_fc_results)

g0 = cfg.groups[0]
FC_example = group_fc_results[g0][0, 0]  # subject 0, window 0 -> (200, 200)
print("Example FC matrix stats:", np.min(FC_example), np.max(FC_example), np.mean(FC_example))

# --- Single subject / window FC matrix ---
for g, arr in group_fc_results.items():
    print(f"   {g:>12}: {arr.shape}")

g_idx = 0   # group index (0-based)
subj  = 1   # subject number (1-based)
win   = 1   # window number  (1-based)

if len(group_fc_results) > 0:
    if 0 <= g_idx < len(cfg.groups):
        g0 = cfg.groups[g_idx]
        if g0 in group_fc_results and group_fc_results[g0].size > 0:
            arr = group_fc_results[g0]
            n_subj, n_win, r1, r2 = arr.shape

            if 1 <= subj <= n_subj and 1 <= win <= n_win:
                subj_idx = subj - 1
                win_idx  = win  - 1
                FC_example = arr[subj_idx, win_idx]
                print(
                    f"\n   Example from [{g0}] subj {subj}, win {win} -> FC stats: "
                    f"min={np.min(FC_example):.4f}, max={np.max(FC_example):.4f}"
                )

                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(FC_example, cmap="coolwarm", vmin=-1, vmax=1)
                ax.set_title(
                    f"{g0} | subj {subj}, win {win} | "
                    f"max={np.max(FC_example):.4f}, min={np.min(FC_example):.4f}"
                )
                ax.set_xlabel("Brain Regions")
                ax.set_ylabel("Brain Regions")
                fig.colorbar(im, ax=ax, label="Connectivity Strength")
                plt.tight_layout()
                plt.show()

# --- Side-by-side FC comparison (2 groups) ---
for g, arr in group_fc_results.items():
    print(f"   {g:>12}: {arr.shape}")

g_idx_1, subj_1, win_1 = 0, 1, 2   # First visualisation
g_idx_2, subj_2, win_2 = 1, 1, 2   # Second visualisation

if len(group_fc_results) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_i, (gi, si, wi) in enumerate(
        [(g_idx_1, subj_1, win_1), (g_idx_2, subj_2, win_2)]
    ):
        if not (0 <= gi < len(cfg.groups)):
            print(f"   Invalid g_idx={gi}.")
            continue
        g0 = cfg.groups[gi]
        if g0 not in group_fc_results or group_fc_results[g0].size == 0:
            print(f"   No FC results for group {g0}.")
            continue
        arr = group_fc_results[g0]
        n_subj, n_win, r1, r2 = arr.shape
        if not (1 <= si <= n_subj) or not (1 <= wi <= n_win):
            print(f"   Invalid subj={si} or win={wi} for [{g0}].")
            continue

        FC_example = arr[si - 1, wi - 1]
        print(
            f"\n   Example {ax_i + 1} from [{g0}] subj {si}, win {wi} -> FC stats: "
            f"min={np.min(FC_example):.4f}, max={np.max(FC_example):.4f}"
        )

        ax = axes[ax_i]
        im = ax.imshow(FC_example, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(
            f"{g0} | subj {si}, win {wi} | "
            f"max={np.max(FC_example):.4f}, min={np.min(FC_example):.4f}"
        )
        ax.set_xlabel("Brain Regions")
        ax.set_ylabel("Brain Regions")
        fig.colorbar(im, ax=ax, label="Connectivity Strength")

    plt.tight_layout()
    plt.show()

# --- Grid visualisation (stride-sampled windows, single subject) ---
for g, arr in group_fc_results.items():
    print(f"   {g:>12}: {arr.shape}")

g_idx = 0
subj  = 1
n_cols = 5
n_rows = 13  # 5 x 13 = 65 panels

if len(group_fc_results) > 0:
    if 0 <= g_idx < len(cfg.groups):
        g0 = cfg.groups[g_idx]
        if g0 in group_fc_results and group_fc_results[g0].size > 0:
            arr = group_fc_results[g0]
            n_subj_arr, n_win_arr, r1, r2 = arr.shape

            if 1 <= subj <= n_subj_arr:
                subj_idx = subj - 1
                max_panels = n_cols * n_rows
                stride = max(1, int(np.floor((n_win_arr - 1) / (max_panels - 1))))
                selected_windows = np.arange(0, n_win_arr, stride)[:max_panels]
                if len(selected_windows) < max_panels:
                    selected_windows = np.round(
                        np.linspace(0, n_win_arr - 1, max_panels)
                    ).astype(int)

                k = len(selected_windows)
                if k > 0:
                    fig, axes_grid = plt.subplots(
                        n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.0)
                    )
                    axes_flat = axes_grid.ravel()
                    mins, maxs, means = [], [], []
                    im = None

                    for i in range(n_cols * n_rows):
                        ax = axes_flat[i]
                        if i < k:
                            w_idx = selected_windows[i]
                            FC_ex = arr[subj_idx, w_idx]
                            mins.append(np.min(FC_ex))
                            maxs.append(np.max(FC_ex))
                            means.append(np.mean(FC_ex))
                            im = ax.imshow(FC_ex, cmap="coolwarm", vmin=-1, vmax=1)
                            ax.set_title(f"win {w_idx + 1}", fontsize=9)
                            ax.set_xlabel("Regions", fontsize=7)
                            ax.set_ylabel("Regions", fontsize=7)
                            ax.tick_params(labelsize=6)
                        else:
                            ax.axis("off")

                    if im is not None:
                        cax = fig.add_axes([0.2, 0.975, 0.6, 0.008])
                        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
                        cbar.set_label("Connectivity Strength", fontsize=10)
                        cbar.ax.tick_params(labelsize=8)
                        fig.subplots_adjust(top=0.955, wspace=0.15, hspace=0.35)

                    fig.suptitle(
                        f"{g0} | Subject {subj} | {k} Windows (stride={stride}) "
                        f"| min={np.min(mins):.3f}, max={np.max(maxs):.3f}",
                        fontsize=12, y=0.995, fontweight="bold",
                    )
                    plt.show()
                    print(f"\nDisplayed windows: {selected_windows + 1}")

# --- Batch save FC grid figures (all subjects) ---
out_dir_fc = SAVE_FIG_DIR
out_dir_fc.mkdir(parents=True, exist_ok=True)

for g, arr in group_fc_results.items():
    print(f"   {g:>12}: {arr.shape}")

n_cols = 5
n_rows = 13

if len(group_fc_results) > 0:
    for g_idx_b in range(len(cfg.groups)):
        g0 = cfg.groups[g_idx_b]
        if g0 not in group_fc_results or group_fc_results[g0].size == 0:
            print(f"   No FC results available for group: {g0}")
            continue

        arr = group_fc_results[g0]
        n_subj_arr, n_win_arr, r1, r2 = arr.shape
        log.info(f"Saving FC grid figures — [{g0}] {n_subj_arr} subjects, {n_win_arr} windows each")

        for subj_b in range(1, n_subj_arr + 1):
            subj_idx = subj_b - 1
            max_panels = n_cols * n_rows

            if n_win_arr < max_panels:
                selected_windows = np.arange(n_win_arr)
            else:
                stride = max(1, int(np.floor((n_win_arr - 1) / (max_panels - 1))))
                selected_windows = np.arange(0, n_win_arr, stride)[:max_panels]
                if len(selected_windows) < max_panels:
                    selected_windows = np.round(
                        np.linspace(0, n_win_arr - 1, max_panels)
                    ).astype(int)

            k = len(selected_windows)
            if k <= 0:
                print(f"   No windows available in [{g0}] for subject {subj_b}.")
                continue

            fig, axes_grid = plt.subplots(
                n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.0)
            )
            axes_flat = axes_grid.ravel()
            mins, maxs, means = [], [], []
            im = None

            for i in range(n_cols * n_rows):
                ax = axes_flat[i]
                if i < k:
                    w_idx = selected_windows[i]
                    FC_ex = arr[subj_idx, w_idx]
                    mins.append(np.min(FC_ex))
                    maxs.append(np.max(FC_ex))
                    means.append(np.mean(FC_ex))
                    im = ax.imshow(FC_ex, cmap="coolwarm", vmin=-1, vmax=1)
                    ax.set_title(f"win {w_idx + 1}", fontsize=9)
                    ax.set_xlabel("Regions", fontsize=7)
                    ax.set_ylabel("Regions", fontsize=7)
                    ax.tick_params(labelsize=6)
                else:
                    ax.axis("off")

            if im is not None:
                cax = fig.add_axes([0.2, 0.975, 0.6, 0.008])
                cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
                cbar.set_label("Connectivity Strength", fontsize=10)
                cbar.ax.tick_params(labelsize=8)
                fig.subplots_adjust(top=0.955, wspace=0.15, hspace=0.35)

            if k > 0:
                fig.suptitle(
                    f"{g0} | Subject {subj_b} | {k} Windows"
                    f"| min={np.min(mins):.3f}, max={np.max(maxs):.3f}",
                    fontsize=12, y=0.995, fontweight="bold",
                )

            fname = f"{g0.replace(' ', '_')}_subj{subj_b:02d}.png"
            plt.savefig(out_dir_fc / fname, dpi=300, bbox_inches="tight")
            plt.close(fig)
            log.info(f"  Saved FC figure [{g0}] subject {subj_b}/{n_subj_arr}: {fname}")

    log.info("FC grid figures saved for all groups")

out_dir_mat = ANALYSIS_DIR / "corr_subjs"
out_dir_mat.mkdir(parents=True, exist_ok=True)

if len(group_fc_results) == 0:
    log.warning("No FC results available to save")
else:
    log.info("Saving per-subject FC tensors → data/analysis/corr_subjs/")
    for g, arr in group_fc_results.items():
        safe_g = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in g)
        if not safe_g or not safe_g[0].isalpha():
            safe_g = f"grp_{safe_g}"

        S, W, R1, R2 = arr.shape
        assert R1 == R2 == 200, f"Unexpected region size for {g}: {(R1, R2)}"
        log.info(f"[{g}] S={S} subjects | W={W} windows | R={R1} regions")

        for s in range(S):
            subj_idx = s + 1
            subj_data = arr[s].astype(np.float32, copy=False)

            payload = {
                f"corr_{safe_g}_subj{subj_idx:02d}": subj_data,
                "__meta__": {
                    "group": np.array([g], dtype=object),
                    "subject_index_1based": np.array([subj_idx], dtype=np.int32),
                    "n_windows": np.array([W], dtype=np.int32),
                    "n_regions": np.array([R1], dtype=np.int32),
                    "note": np.array(
                        ["corr_* has shape (windows, regions, regions)"], dtype=object
                    ),
                    "windowing": np.array(
                        [f"{cfg.step_size_tr}TR step"], dtype=object
                    ),
                },
            }

            out_file = str(
                out_dir_mat
                / f"corr_{safe_g}_subj{subj_idx:02d}_{cfg.step_size_tr}tr_windows.mat"
            )
            savemat(out_file, payload, do_compression=True)
            log.info(f"  [{g}] Saved subject {subj_idx:02d}/{S} → {Path(out_file).name}")

            del payload, subj_data
            gc.collect()

    log.info("All .mat files saved")

end_time = datetime.now()
log.info(f"STAGE 1 COMPLETE — duration: {end_time - start_time}")
