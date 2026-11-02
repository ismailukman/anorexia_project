#!/usr/bin/env python3
"""
01c_combined_fc.py: combined cortical + subcortical FC extraction.
Schaefer-200 (Yeo-7) + Tian Scale I (S1, 16 bilateral ROIs) = 216 regions.
Cortical regions use the subcortex-masked Schaefer image, so no voxel is shared
with Tian S1. Series are cached in data/analysis/timeseries_subjs_masked/.
"""
import gc
import glob
import logging
import os
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
from nilearn.input_data import NiftiLabelsMasker
from nilearn import datasets
from scipy.io import savemat

start_time = datetime.now()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

log.info("STAGE 1c - Combined Cortical+Subcortical FC (216 regions)")

from paths import ROOT as PROJECT_ROOT
DATA_DIR       = PROJECT_ROOT / "data"
ATLAS_DIR_S200 = DATA_DIR / "atlas" / "schaefer_2018"
ATLAS_DIR_S1   = DATA_DIR / "atlas" / "tian_s1"
ATLAS_DIR_COMB = DATA_DIR / "atlas" / "combined_216"
# Cortical series for the combined atlas, extracted with the subcortex-masked
# Schaefer image. Kept apart from timeseries_subjs/ (original image), which
# the cortical-only 200-region pipeline uses.
TS_CACHE_DIR   = DATA_DIR / "analysis" / "timeseries_subjs_masked"
OUT_MAT_DIR    = DATA_DIR / "analysis" / "combined_subjs"
OUT_RES_DIR    = PROJECT_ROOT / "output" / "results" / "combined_216"
OUT_FIG_DIR    = PROJECT_ROOT / "output" / "figures" / "stage1_fc" / "combined_216"

for d in [ATLAS_DIR_S1, ATLAS_DIR_COMB, TS_CACHE_DIR, OUT_MAT_DIR, OUT_RES_DIR, OUT_FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TR          = 0.8
WINDOW_SEC  = 30.0
STEP_TR     = 1
WINDOW_TRS  = int(round(WINDOW_SEC / TR))   # 38
FMRI_PREFIX = "errts"
GROUPS      = ["an_patients", "hc_patients"]
N_CORTICAL  = 200
N_SUBCORT   = 16
N_COMBINED  = N_CORTICAL + N_SUBCORT        # 216

log.info(f"TR={TR}s | Window={WINDOW_SEC}s ({WINDOW_TRS} TRs) | Step={STEP_TR} TR")
log.info(f"Cortical={N_CORTICAL} | Subcortical={N_SUBCORT} | Combined={N_COMBINED}")


ATLAS_S1_NII   = ATLAS_DIR_S1 / "Tian_Subcortex_S1_3T_2009cAsym.nii.gz"
ATLAS_S1_LABEL = ATLAS_DIR_S1 / "Tian_Subcortex_S1_3T_label.txt"
_RAW = "https://raw.githubusercontent.com/yetianmed/subcortex/master/Group-Parcellation/3T/Subcortex-Only"


def _download(url: str, dest: Path) -> None:
    log.info(f"Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, str(dest))
    log.info(f"  Saved -> {dest}")


if not ATLAS_S1_NII.exists():
    _download(f"{_RAW}/Tian_Subcortex_S1_3T_2009cAsym.nii.gz", ATLAS_S1_NII)
else:
    log.info(f"Tian S1 NIfTI present: {ATLAS_S1_NII.name}")

if not ATLAS_S1_LABEL.exists():
    _download(f"{_RAW}/Tian_Subcortex_S1_3T_label.txt", ATLAS_S1_LABEL)
else:
    log.info(f"Tian S1 labels present: {ATLAS_S1_LABEL.name}")



def parse_labels(label_file: Path) -> List[str]:
    """Parse label file - handles both '<idx> <label>' and plain '<label>' formats."""
    labels = []
    with open(label_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(parts[0])
    return labels


# Cortical labels from Schaefer .txt (col 1: index, col 2: label)
cortical_labels: List[str] = []
schaefer_txt = ATLAS_DIR_S200 / "schaefer_2018" / "Schaefer2018_200Parcels_7Networks_order.txt"
with open(schaefer_txt) as fh:
    for line in fh:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            cortical_labels.append(parts[1])

subcortical_labels = parse_labels(ATLAS_S1_LABEL)
assert len(cortical_labels) == N_CORTICAL, \
    f"Expected {N_CORTICAL} cortical labels, got {len(cortical_labels)}"
assert len(subcortical_labels) == N_SUBCORT, \
    f"Expected {N_SUBCORT} subcortical labels, got {len(subcortical_labels)}"

combined_labels = cortical_labels + subcortical_labels
log.info(f"Cortical labels: {len(cortical_labels)} | Subcortical: {len(subcortical_labels)}")


comb_label_file = ATLAS_DIR_COMB / "combined_216_labels.txt"
with open(comb_label_file, "w") as fh:
    fh.write("# Combined Schaefer-200 + Tian Scale I (S1) - 216 regions\n")
    fh.write(f"# {'Index':<6} {'Label':<48} Atlas\n")
    for i, lbl in enumerate(cortical_labels, 1):
        fh.write(f"  {i:<6} {lbl:<48} Schaefer200_Yeo7\n")
    for j, lbl in enumerate(subcortical_labels, 1):
        fh.write(f"  {N_CORTICAL+j:<6} {lbl:<48} Tian_S1\n")
log.info(f"Combined label file -> {comb_label_file}")

log.info("Checking voxel overlap between Schaefer-200 and Tian S1 ...")


atlas_s200 = datasets.fetch_atlas_schaefer_2018(
    n_rois=N_CORTICAL, yeo_networks=7, resolution_mm=1,
    data_dir=str(ATLAS_DIR_S200),
)
# The subcortex-masked Schaefer image removes the 2,014 voxels that the
# original image shares with Tian S1, so no voxel enters both a cortical and a
# subcortical region. The overlap check below enforces this.
schaefer_nii_path = str(ATLAS_DIR_S200 / "schaefer_2018" /
                        "Schaefer2018_200Parcels_7Networks_order_subcortMasked.nii.gz")
log.info(f"Schaefer atlas: {Path(schaefer_nii_path).name}")

schaefer_img  = nib.load(str(schaefer_nii_path))
tian_s1_img   = nib.load(str(ATLAS_S1_NII))

tian_resampled = resample_to_img(
    tian_s1_img, schaefer_img, interpolation="nearest"
)

schaefer_mask = (schaefer_img.get_fdata() > 0)
tian_mask     = (tian_resampled.get_fdata() > 0)
overlap_vox   = int(np.sum(schaefer_mask & tian_mask))

overlap_report = (
    f"Schaefer-200 voxels  : {int(schaefer_mask.sum()):>8,}\n"
    f"Tian S1 voxels       : {int(tian_mask.sum()):>8,}\n"
    f"Overlap voxels       : {overlap_vox:>8,}\n"
    f"Status               : {'NO OVERLAP' if overlap_vox == 0 else 'OVERLAP'}\n"
)
log.info("\n" + overlap_report)

overlap_file = OUT_RES_DIR / "overlap_check.txt"
with open(overlap_file, "w") as fh:
    fh.write("Schaefer-200 vs Tian Scale I - Voxel Overlap Check\n")
    fh.write("=" * 50 + "\n")
    fh.write(overlap_report)
log.info(f"Overlap report -> {overlap_file}")
if overlap_vox > 0:
    raise RuntimeError(f"{overlap_vox} voxels are shared by Schaefer-200 and Tian S1")


# Region centroids (MNI mm, combined order) for plotting. Centre of mass of
# every voxel in the parcel, so a parcel always plots in its own hemisphere.
def _centroids(img) -> np.ndarray:
    d = np.asarray(img.dataobj).astype(int)
    return np.array([nib.affines.apply_affine(img.affine, np.argwhere(d == v)).mean(0)
                     for v in range(1, int(d.max()) + 1)])

coords = np.vstack([_centroids(schaefer_img), _centroids(tian_s1_img)])
assert coords.shape == (N_COMBINED, 3), coords.shape
assert all((coords[i, 0] < 0) == ("_LH_" in lbl or lbl.endswith("-lh"))
           for i, lbl in enumerate(combined_labels)), "centroid hemisphere mismatch"
np.save(DATA_DIR / "analysis" / "coords216.npy", coords)
log.info(f"Region centroids -> {DATA_DIR / 'analysis' / 'coords216.npy'}")

del schaefer_mask, tian_mask, tian_resampled, schaefer_img, tian_s1_img
gc.collect()


_CACHE = str(PROJECT_ROOT / "nilearn_cache")


def _build_masker(nii_path: str) -> NiftiLabelsMasker:
    try:
        return NiftiLabelsMasker(
            labels_img=nii_path,
            standardize="zscore_sample",
            standardize_confounds=True,
            memory=_CACHE,
            verbose=0,
            t_r=TR,
            resampling_target="data",
        )
    except TypeError:
        return NiftiLabelsMasker(
            labels_img=nii_path,
            standardize=True,
            standardize_confounds=True,
            memory=_CACHE,
            verbose=0,
            t_r=TR,
            resampling_target="data",
        )


masker_cortical    = _build_masker(str(schaefer_nii_path))
masker_subcortical = _build_masker(str(ATLAS_S1_NII))
log.info("Maskers ready: Schaefer-200 (cortical) + Tian S1 (subcortical)")



def list_group_niis(group_dir: str, prefix: str) -> List[str]:
    p1 = os.path.join(group_dir, f"{prefix}*.nii")
    p2 = os.path.join(group_dir, f"{prefix}*.nii.gz")
    return sorted(glob.glob(p1) + glob.glob(p2))


def extract_ts(grp: str, si: int, nii_path: str):
    """Return (T, 200) cortical and (T, 16) subcortical series.

    The NIfTI is read once for both maskers. Cortical series are cached in
    TS_CACHE_DIR; a cache hit skips reading the NIfTI only if the subcortical
    series is cached too.
    """
    safe_g = "".join(c if (c.isalnum() or c == "_") else "_" for c in grp)
    c_ctx = TS_CACHE_DIR / f"ts_{safe_g}_subj{si:02d}.npy"
    c_sub = TS_CACHE_DIR / f"ts_sub_{safe_g}_subj{si:02d}.npy"
    if c_ctx.exists() and c_sub.exists():
        log.info(f"    TS loaded from cache: {c_ctx.name}, {c_sub.name}")
        return np.load(str(c_ctx)), np.load(str(c_sub))
    img = nib.load(nii_path)
    img = nib.Nifti1Image(np.asarray(img.dataobj, dtype=np.float32), img.affine, img.header)
    ts_ctx = masker_cortical.fit_transform(img).astype(np.float32)
    ts_sub = masker_subcortical.fit_transform(img).astype(np.float32)
    del img
    np.save(str(c_ctx), ts_ctx)
    np.save(str(c_sub), ts_sub)
    log.info(f"    TS extracted and cached: cortical {ts_ctx.shape}, subcortical {ts_sub.shape}")
    return ts_ctx, ts_sub


def sliding_windows(ts: np.ndarray, w_trs: int, step_trs: int) -> List[np.ndarray]:
    out, s = [], 0
    while s + w_trs <= ts.shape[0]:
        out.append(ts[s:s + w_trs, :])
        s += step_trs
    return out


def compute_fc_windows(windows: List[np.ndarray]) -> np.ndarray:
    """Pearson correlation per window -> (W, N, N)."""
    if not windows:
        return np.zeros((0, 0, 0), dtype=np.float32)
    W, N = len(windows), windows[0].shape[1]
    fc = np.zeros((W, N, N), dtype=np.float32)
    for i, win in enumerate(windows):
        c = np.corrcoef(win.T).astype(np.float32)
        np.fill_diagonal(c, 0.0)
        fc[i] = c
    return fc



def _save_qc_figure(
    fc_mat: np.ndarray, grp: str, si: int,
    win_idx: int, n_wins: int, fig_dir: Path,
) -> None:
    """Save a 216×216 FC matrix with cortical/subcortical block annotations."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(fc_mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Pearson r")

    # Block boundary
    for pos in [N_CORTICAL - 0.5]:
        ax.axhline(pos, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.axvline(pos, color="black", linewidth=1.2, linestyle="--", alpha=0.7)

    mid_ctx = N_CORTICAL / 2
    mid_sub = N_CORTICAL + N_SUBCORT / 2
    ax.text(mid_ctx, -5, "Cortical (200)",   ha="center", va="bottom", fontsize=8, color="navy")
    ax.text(mid_sub, -5, "Subcortical (16)", ha="center", va="bottom", fontsize=8, color="darkred")
    ax.text(-5, mid_ctx, "Cortical (200)",   ha="right",  va="center", fontsize=8, color="navy",    rotation=90)
    ax.text(-5, mid_sub, "Subcortical (16)", ha="right",  va="center", fontsize=8, color="darkred", rotation=90)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{grp} | Subj {si:02d} | Window {win_idx+1}/{n_wins}\n"
        f"Combined FC 216×216  (Schaefer-200 + Tian S1)",
        fontsize=9, fontweight="bold",
    )
    fname = f"{grp}_subj{si:02d}_combined_fc.png"
    fig.savefig(fig_dir / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"    QC figure -> {fname}")


group_pairs: Dict[str, List[str]] = {}
log.info("Scanning fMRI files ...")
for grp in GROUPS:
    grp_dir = str(DATA_DIR / grp)
    if not os.path.isdir(grp_dir):
        raise FileNotFoundError(f"Group directory not found: {grp_dir}")
    runs = list_group_niis(grp_dir, FMRI_PREFIX)
    group_pairs[grp] = runs
    log.info(f"  [{grp}] {len(runs)} NIfTI files (1 per subject)")

log.info("COMBINED EXTRACTION - 200 cortical + 16 subcortical = 216")

for grp in GROUPS:
    nii_files = group_pairs[grp]
    if not nii_files:
        log.info(f"[{grp}] No files found; skipping.")
        continue

    log.info(f"\n[{grp}] {len(nii_files)} subjects")

    for si, nii_path in enumerate(nii_files, start=1):
        log.info(f"  Subject {si:02d}/{len(nii_files)} - {Path(nii_path).name}")

        ts_ctx, ts_sub = extract_ts(grp, si, nii_path)   # (T, 200), (T, 16)

        T_ctx, T_sub = ts_ctx.shape[0], ts_sub.shape[0]
        if T_ctx != T_sub:
            log.warning(f"    Timepoint mismatch: cortical={T_ctx}, subcortical={T_sub}. Trimming to min.")
            T = min(T_ctx, T_sub)
            ts_ctx, ts_sub = ts_ctx[:T], ts_sub[:T]
        else:
            T = T_ctx

        ts_combined = np.concatenate([ts_ctx, ts_sub], axis=1).astype(np.float32)
        log.info(f"    Combined TS: {ts_combined.shape}  (T={T}, N=200+16={ts_combined.shape[1]})")

        assert ts_combined.shape[1] == N_COMBINED, \
            f"Expected {N_COMBINED} regions, got {ts_combined.shape[1]}"

        windows = sliding_windows(ts_combined, WINDOW_TRS, STEP_TR)
        if not windows:
            log.warning(f"    No windows for subj {si}; skipping.")
            continue
        fc_wins = compute_fc_windows(windows)
        log.info(f"    FC: {fc_wins.shape}  ({len(windows)} windows × {N_COMBINED}×{N_COMBINED})")

        tag = grp
        ts_out = OUT_MAT_DIR / f"subj_timeseries_combined_{tag}_subj{si:02d}.mat"
        savemat(str(ts_out), {
            f"ts_combined_{tag}_subj{si:02d}": ts_combined,
            "meta": {
                "group":              np.array([grp], dtype=object),
                "subject_1based":     np.array([si], dtype=np.int32),
                "atlas":              np.array(["Schaefer200_Yeo7+TianS1"], dtype=object),
                "n_timepoints":       np.array([T], dtype=np.int32),
                "n_regions":          np.array([N_COMBINED], dtype=np.int32),
                "n_cortical":         np.array([N_CORTICAL], dtype=np.int32),
                "n_subcortical":      np.array([N_SUBCORT], dtype=np.int32),
                "labels":             np.array(combined_labels, dtype=object),
                "cortical_idx_1based":    np.arange(1, N_CORTICAL+1, dtype=np.int32),
                "subcortical_idx_1based": np.arange(N_CORTICAL+1, N_COMBINED+1, dtype=np.int32),
            },
        }, do_compression=True)

        # --- Save windowed FC .mat ---
        fc_out = OUT_MAT_DIR / f"subj_fc_combined_{tag}_subj{si:02d}.mat"
        savemat(str(fc_out), {
            f"fc_combined_{tag}_subj{si:02d}": fc_wins,
            "meta": {
                "group":              np.array([grp], dtype=object),
                "subject_1based":     np.array([si], dtype=np.int32),
                "atlas":              np.array(["Schaefer200_Yeo7+TianS1"], dtype=object),
                "shape_note":         np.array(["(windows, regions, regions)"], dtype=object),
                "n_windows":          np.array([fc_wins.shape[0]], dtype=np.int32),
                "n_regions":          np.array([N_COMBINED], dtype=np.int32),
                "n_cortical":         np.array([N_CORTICAL], dtype=np.int32),
                "n_subcortical":      np.array([N_SUBCORT], dtype=np.int32),
                "window_trs":         np.array([WINDOW_TRS], dtype=np.int32),
                "step_trs":           np.array([STEP_TR], dtype=np.int32),
                "tr_sec":             np.array([TR], dtype=np.float32),
                "labels":             np.array(combined_labels, dtype=object),
                "cortical_idx_1based":    np.arange(1, N_CORTICAL+1, dtype=np.int32),
                "subcortical_idx_1based": np.arange(N_CORTICAL+1, N_COMBINED+1, dtype=np.int32),
            },
        }, do_compression=True)

        log.info(f"    Saved: {ts_out.name}")
        log.info(f"    Saved: {fc_out.name}")

        mid_win = fc_wins.shape[0] // 2
        _save_qc_figure(fc_wins[mid_win], grp, si, mid_win, fc_wins.shape[0], OUT_FIG_DIR)

        del ts_ctx, ts_sub, ts_combined, windows, fc_wins
        gc.collect()


end_time = datetime.now()
log.info("STAGE 1c COMPLETE")
log.info(f"Atlas            : Schaefer-200 (Yeo-7) + Tian S1 = {N_COMBINED} regions")
log.info(f"Cortical ROIs    : 1–{N_CORTICAL}  (Schaefer-200)")
log.info(f"Subcortical ROIs : {N_CORTICAL+1}–{N_COMBINED} (Tian S1 - " +
         ", ".join(subcortical_labels) + ")")
log.info(f"Output .mat      : {OUT_MAT_DIR}")
log.info(f"QC figures       : {OUT_FIG_DIR}")
log.info(f"Labels           : {comb_label_file}")
log.info(f"Overlap report   : {overlap_file}")
log.info(f"Duration         : {end_time - start_time}")
