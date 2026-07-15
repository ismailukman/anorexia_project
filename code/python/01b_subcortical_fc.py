#!/usr/bin/env python3
"""
01b_subcortical_preparation.py — Tian Scale II subcortical FC extraction (run after 01).
Outputs per-subject timeseries and windowed FC .mat files for Amygdala/Hippocampus ROIs.
"""

import gc
import glob
import logging
import os
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat, loadmat

from nilearn.input_data import NiftiLabelsMasker

start_time = datetime.now()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

log.info("STAGE 1b — Subcortical Extraction (Tian Scale II)")

PROJECT_ROOT = Path("/Users/ismaila/Documents/C-Codes/AnorexiaProject")
DATA_DIR     = PROJECT_ROOT / "data"
ATLAS_DIR    = DATA_DIR / "atlas" / "tian_s2"
OUT_MAT_DIR  = DATA_DIR / "analysis" / "subcortical_subjs"
OUT_RES_DIR  = PROJECT_ROOT / "output" / "results" / "subcortical"
OUT_FIG_DIR  = PROJECT_ROOT / "output" / "figures" / "stage1_fc" / "subcortical"

for d in [ATLAS_DIR, OUT_MAT_DIR, OUT_RES_DIR, OUT_FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TR               = 0.8        # seconds
WINDOW_SEC       = 30.0       # seconds
STEP_TR          = 1          # TR step
WINDOW_TRS       = int(round(WINDOW_SEC / TR))   # 38 TRs
FMRI_PREFIX      = "errts"
GROUPS           = ["an_patients", "hc_patients"]
GROUP_TAGS       = ["an_patients", "hc_patients"]
RUNS_PER_SUBJECT = 1
VERBOSE          = True

log.info(f"TR={TR}s | Window={WINDOW_SEC}s ({WINDOW_TRS} TRs) | Step={STEP_TR} TR")
log.info(f"Groups: {GROUPS}")


ATLAS_NII   = ATLAS_DIR / "Tian_Subcortex_S2_3T_2009cAsym.nii.gz"
ATLAS_LABEL = ATLAS_DIR / "Tian_Subcortex_S2_3T_label.txt"

_RAW = "https://raw.githubusercontent.com/yetianmed/subcortex/master"
_URL_NII   = f"{_RAW}/Group-Parcellation/3T/Subcortex-Only/Tian_Subcortex_S2_3T_2009cAsym.nii.gz"
_URL_LABEL = f"{_RAW}/Group-Parcellation/3T/Subcortex-Only/Tian_Subcortex_S2_3T_label.txt"


def _download(url: str, dest: Path) -> None:
    log.info(f"Downloading {dest.name} …")
    urllib.request.urlretrieve(url, str(dest))
    log.info(f"Saved → {dest}")


if not ATLAS_NII.exists():
    _download(_URL_NII, ATLAS_NII)
else:
    log.info(f"Atlas NIfTI already present: {ATLAS_NII.name}")

if not ATLAS_LABEL.exists():
    _download(_URL_LABEL, ATLAS_LABEL)
else:
    log.info(f"Atlas labels already present: {ATLAS_LABEL.name}")


def parse_tian_labels(label_file: Path) -> List[str]:
    """
    Parse the Tian label text file.
    Format is either '<index> <label>' (tab/space separated)
    or plain '<label>' (one per line, 1-indexed).
    Returns list of label strings ordered by integer index.
    """
    labels = []
    with open(label_file, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(parts[0])
    return labels


tian_labels = parse_tian_labels(ATLAS_LABEL)
n_subcortical = len(tian_labels)

log.info(f"Tian Scale II: {n_subcortical} parcels loaded")

label_out = OUT_RES_DIR / "subcortical_labels.txt"
with open(label_out, "w") as fh:
    for i, lbl in enumerate(tian_labels, 1):
        fh.write(f"{i:2d}\t{lbl}\n")
log.info(f"Label list saved → {label_out}")

# Tian S2 uses "HIP" (not "HIPP") and "AMY"
_AMY_KEYS  = ("amy", "amygdala")
_HIPP_KEYS = ("hip", "hippocampus")

amy_idx  = [i for i, l in enumerate(tian_labels) if any(k in l.lower() for k in _AMY_KEYS)]
hipp_idx = [i for i, l in enumerate(tian_labels) if any(k in l.lower() for k in _HIPP_KEYS)]

log.info(f"Amygdala   ({len(amy_idx)}): {[tian_labels[i] for i in amy_idx]}")
log.info(f"Hippocampus ({len(hipp_idx)}): {[tian_labels[i] for i in hipp_idx]}")

if not amy_idx:
    log.warning("No amygdala labels matched — check label file content.")
if not hipp_idx:
    log.warning("No hippocampus labels matched — check label file content.")


_CACHE = str(PROJECT_ROOT / "nilearn_cache")

try:
    masker = NiftiLabelsMasker(
        labels_img=str(ATLAS_NII),
        standardize="zscore_sample",
        standardize_confounds=True,
        memory=_CACHE,
        verbose=0,
        t_r=TR,
        resampling_target="data",   # resample atlas to fMRI voxel grid
    )
except TypeError:
    masker = NiftiLabelsMasker(
        labels_img=str(ATLAS_NII),
        standardize=True,
        standardize_confounds=True,
        memory=_CACHE,
        verbose=0,
        t_r=TR,
        resampling_target="data",
    )

log.info("NiftiLabelsMasker (Tian Scale II) ready.")


def list_group_niis(group_dir: str, prefix: str) -> List[str]:
    p1 = os.path.join(group_dir, f"{prefix}*.nii")
    p2 = os.path.join(group_dir, f"{prefix}*.nii.gz")
    return sorted(glob.glob(p1) + glob.glob(p2))


def pair_consecutively(run_paths: List[str], n: int) -> List[Tuple[str, ...]]:
    if len(run_paths) % n != 0:
        raise ValueError(f"{len(run_paths)} runs not divisible by {n}")
    return [tuple(run_paths[i:i+n]) for i in range(0, len(run_paths), n)]


def extract_timeseries(nii_path: str) -> np.ndarray:
    """Return (T, n_subcortical) timeseries array."""
    ts = masker.fit_transform(nii_path)
    if ts.ndim != 2:
        raise ValueError(f"Expected 2D timeseries, got {ts.shape}")
    return ts


def sliding_windows(ts: np.ndarray, w_trs: int, step_trs: int) -> List[np.ndarray]:
    out, s = [], 0
    while s + w_trs <= ts.shape[0]:
        out.append(ts[s:s + w_trs, :])
        s += step_trs
    return out


def compute_fc_windows(windows: List[np.ndarray]) -> np.ndarray:
    """Pearson correlation per window → (W, n, n)."""
    if not windows:
        return np.zeros((0, 0, 0), dtype=np.float32)
    W, n = len(windows), windows[0].shape[1]
    fc = np.zeros((W, n, n), dtype=np.float32)
    for i, win in enumerate(windows):
        c = np.corrcoef(win.T)            # (n, n)
        np.fill_diagonal(c, 0.0)
        fc[i] = c.astype(np.float32)
    return fc


group_pairs: Dict[str, List[Tuple[str, ...]]] = {}

log.info("Scanning groups …")
for grp in GROUPS:
    grp_dir = str(DATA_DIR / grp)
    if not os.path.isdir(grp_dir):
        raise FileNotFoundError(f"Group directory not found: {grp_dir}")
    runs = list_group_niis(grp_dir, FMRI_PREFIX)
    pairs = pair_consecutively(runs, RUNS_PER_SUBJECT)
    group_pairs[grp] = pairs
    log.info(f"  [{grp}] {len(runs)} runs → {len(pairs)} subjects")


log.info("SUBCORTICAL EXTRACTION — Tian Scale II")

for grp, tag in zip(GROUPS, GROUP_TAGS):
    pairs = group_pairs[grp]
    if not pairs:
        log.info(f"[{grp}] No subjects found; skipping.")
        continue

    log.info(f"[{grp}] — {len(pairs)} subjects")

    for si, pair in enumerate(pairs, start=1):
        ts_list = []
        for rp in pair:
            ts = extract_timeseries(rp)
            ts_list.append(ts)
        ts_concat = np.concatenate(ts_list, axis=0).astype(np.float32)  # (T, n_sub)

        n_regions_found = ts_concat.shape[1]
        if n_regions_found != n_subcortical:
            log.warning(
                f"  [{grp}] subj {si}: expected {n_subcortical} regions, "
                f"got {n_regions_found}. Proceeding."
            )

        windows = sliding_windows(ts_concat, WINDOW_TRS, STEP_TR)
        if not windows:
            log.warning(f"  [{grp}] No windows for subj {si}; skipping.")
            continue
        fc_wins = compute_fc_windows(windows)   # (W, n, n)

        log.info(
            f"  [{grp}] Subject {si:02d}/{len(pairs)} — "
            f"TS {ts_concat.shape} | {len(windows)} windows | FC {fc_wins.shape}"
        )

        ts_out = OUT_MAT_DIR / f"subj_timeseries_{tag}_subj{si:02d}.mat"
        ts_payload = {
            f"ts_{tag}_subj{si:02d}": ts_concat,
            "__meta__": {
                "group":              np.array([grp], dtype=object),
                "subject_1based":     np.array([si], dtype=np.int32),
                "atlas":              np.array(["Tian_Scale2_3T_32rois"], dtype=object),
                "n_timepoints":       np.array([ts_concat.shape[0]], dtype=np.int32),
                "n_regions":          np.array([n_regions_found], dtype=np.int32),
                "labels":             np.array(tian_labels, dtype=object),
                "amygdala_idx_0based":np.array(amy_idx, dtype=np.int32),
                "hippocampus_idx_0based": np.array(hipp_idx, dtype=np.int32),
            },
        }
        savemat(str(ts_out), ts_payload, do_compression=True)

        fc_out = OUT_MAT_DIR / f"subj_fc_windows_{tag}_subj{si:02d}.mat"
        fc_payload = {
            f"fc_{tag}_subj{si:02d}": fc_wins,
            "__meta__": {
                "group":              np.array([grp], dtype=object),
                "subject_1based":     np.array([si], dtype=np.int32),
                "atlas":              np.array(["Tian_Scale2_3T_32rois"], dtype=object),
                "shape_note":         np.array(["(windows, regions, regions)"], dtype=object),
                "n_windows":          np.array([fc_wins.shape[0]], dtype=np.int32),
                "n_regions":          np.array([n_regions_found], dtype=np.int32),
                "window_trs":         np.array([WINDOW_TRS], dtype=np.int32),
                "step_trs":           np.array([STEP_TR], dtype=np.int32),
                "tr_sec":             np.array([TR], dtype=np.float32),
                "labels":             np.array(tian_labels, dtype=object),
                "amygdala_idx_0based":np.array(amy_idx, dtype=np.int32),
                "hippocampus_idx_0based": np.array(hipp_idx, dtype=np.int32),
            },
        }
        savemat(str(fc_out), fc_payload, do_compression=True)
        log.info(f"    Saved: {ts_out.name} | {fc_out.name}")

        del ts_payload, fc_payload, ts_list
        gc.collect()


log.info("Generating QC figures …")

N_COLS, N_ROWS = 5, 6   # 30 panels per grid

for grp, tag in zip(GROUPS, GROUP_TAGS):
    pairs = group_pairs[grp]
    for si in range(1, len(pairs) + 1):
        fc_path = OUT_MAT_DIR / f"subj_fc_windows_{tag}_subj{si:02d}.mat"
        if not fc_path.exists():
            continue

        key = f"fc_{tag}_subj{si:02d}"
        mat = loadmat(str(fc_path))
        fc_wins = mat[key]   # (W, n, n) as saved

        W = fc_wins.shape[0]
        max_panels = N_COLS * N_ROWS
        idx = np.round(np.linspace(0, W - 1, min(max_panels, W))).astype(int)
        k = len(idx)

        fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 3 + 1, N_ROWS * 3))
        axes_flat = axes.ravel()

        for i in range(N_COLS * N_ROWS):
            ax = axes_flat[i]
            if i < k:
                ax.imshow(fc_wins[idx[i]], cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
                ax.set_title(f"win {idx[i]+1}", fontsize=8)
                ax.axis("off")
            else:
                ax.axis("off")

        fig.subplots_adjust(left=0.02, right=0.88, top=0.94, bottom=0.02,
                            hspace=0.30, wspace=0.15)
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(cmap="coolwarm", norm=Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        cbar_ax = fig.add_axes([0.90, 0.10, 0.02, 0.75])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Pearson r", fontsize=11)
        cbar.ax.tick_params(labelsize=9)

        fig.suptitle(
            f"{grp} | Subject {si:02d} | Tian Scale II FC ({W} windows)",
            fontsize=10, fontweight="bold",
        )

        fname = f"{tag}_subj{si:02d}_subcortical_fc.png"
        fig.savefig(OUT_FIG_DIR / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  QC figure saved: {fname}")


end_time = datetime.now()

log.info("STAGE 1b COMPLETE")
log.info(f"Atlas            : Tian Scale II — {n_subcortical} parcels")
log.info(f"Amygdala parcels : {[tian_labels[i] for i in amy_idx]}")
log.info(f"Hippocampus      : {[tian_labels[i] for i in hipp_idx]}")
log.info(f"Timeseries .mat  → {OUT_MAT_DIR}")
log.info(f"QC figures       → {OUT_FIG_DIR}")
log.info(f"Label list       → {OUT_RES_DIR / 'subcortical_labels.txt'}")
log.info(f"Duration: {end_time - start_time}")
