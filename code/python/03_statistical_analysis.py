#!/usr/bin/env python3
"""
03_statistical_analysis_v2.py — Permutation tests (Monte-Carlo) for Anorexia vs Control.
Reduced subset: 5 subjects per group. Subject is the unit of observation.
Atlas: Schaefer-2018, 200 parcels, Yeo-7 networks.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from tqdm import tqdm
import statsmodels.stats.multitest as smm

from nilearn import datasets
from teneto import communitymeasures

start_time = datetime.now()
np.random.seed(42)

PROJECT_ROOT = Path("/Users/ismaila/Documents/C-Codes/AnorexiaProject")
BASE_DIR     = PROJECT_ROOT / "data" / "analysis" / "mlcd_subjs"
MLCD_DIR     = BASE_DIR / "subjs_mlcd"
STAT_OUT     = PROJECT_ROOT / "output" / "results" / "statistical_results_v2"
STAT_OUT.mkdir(parents=True, exist_ok=True)


FNAME_TEMPLATE = "mlcd_{tag}_wins.mat"
GROUP_TAGS     = ["anorexia", "control"]
GROUP_LABELS   = ["Anorexia", "Healthy Control"]

N_EXPECTED         = 200
N_SUBJ_TOTAL       = 22            # total subjects available per group in .mat
N_SUBJ_PER_GROUP   = 5             # <<< reduced subset
VAR_BASE           = "N_all_g"     # primary community-label variable

# Permutation settings
N_PERM          = 5000              # fewer perms (smaller sample → faster)
N_JOBS          = -1                # -1 = all CPU cores
ALPHA           = 0.05              # significance level
FDR_METHOD      = "fdr_bh"         # Benjamini-Hochberg FDR

atlas  = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=1)
labels = np.array(atlas.labels[1:]).astype("U")   # skip 'Background' at index 0

NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NETWORK_SHORT = ["VN", "SMN", "DAN", "VAN", "LN", "FPN", "DMN"]

static_communities = np.zeros((200,), dtype=int)
for i, network in enumerate(NETWORKS):
    idx = np.array([network in s for s in labels], dtype=bool)
    static_communities[idx] = i + 1

pivot = np.where(static_communities[:-1] != static_communities[1:])[0]
pivot = np.concatenate([pivot, [199]])

NETWORK_BOUNDARIES = [
    (-0.5,  13.5), (13.5,  29.5), (29.5,  42.5), (42.5,  53.5),
    (53.5,  59.5), (59.5,  72.5), (72.5,  99.5),
    (99.5, 114.5), (114.5, 133.5), (133.5, 146.5), (146.5, 157.5),
    (157.5, 163.5), (163.5, 180.5), (180.5, 199.5),
]

YEO7_COLORS = [
    "#A251AC", "#789AC1", "#409832", "#E165FE",
    "#F6FDC9", "#EFB944", "#D9717D",
]



def _key_candidates(var_base, tag):
    return [f"{var_base}_{tag}", f"{var_base}{tag}", f"{var_base}__{tag}"]


def read_var_any(f, candidates):
    keys = set(f.keys())
    for name in candidates:
        if name in keys:
            return np.squeeze(np.asarray(f[name][()])), name
    raise KeyError(f"None of {candidates} found. Keys: {list(f.keys())}")


def fix_orientation(mat, n_expected):
    mat = np.asarray(mat)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D, got {mat.shape}")
    r, c = mat.shape
    if r == n_expected:
        return mat
    if c == n_expected:
        return mat.T
    raise ValueError(f"Neither dim matches N={n_expected}. Got {mat.shape}")


def split_into_subjects(primary_mat, n_subj_total, n_expected):
    """Split the full matrix into per-subject chunks (all 22 subjects)."""
    primary_mat = fix_orientation(primary_mat, n_expected)
    N, total_cols = primary_mat.shape
    wins_per_subj = total_cols // n_subj_total
    return [
        primary_mat[:, i * wins_per_subj : (i + 1) * wins_per_subj]
        for i in range(n_subj_total)
    ], wins_per_subj


def load_group(tag):
    """Load all 22 subjects then return only the first N_SUBJ_PER_GROUP."""
    mat_path = MLCD_DIR / FNAME_TEMPLATE.format(tag=tag)
    with h5py.File(mat_path, "r") as f:
        primary, _ = read_var_any(f, _key_candidates(VAR_BASE, tag))
        primary = fix_orientation(primary, N_EXPECTED)
    all_subjects, wins = split_into_subjects(primary, N_SUBJ_TOTAL, N_EXPECTED)
    # Select first N_SUBJ_PER_GROUP subjects
    selected = all_subjects[:N_SUBJ_PER_GROUP]
    return selected, wins


print(f"Loading MLCD community labels ({N_SUBJ_PER_GROUP} subjects per group) …")

subjs_an, wins_an = load_group("anorexia")
subjs_hc, wins_hc = load_group("control")

print(f"  Anorexia : {len(subjs_an)} subjects, {wins_an} wins/subj, shape {subjs_an[0].shape}")
print(f"  Control  : {len(subjs_hc)} subjects, {wins_hc} wins/subj, shape {subjs_hc[0].shape}")

communities_an = subjs_an
communities_hc = subjs_hc
all_communities = communities_an + communities_hc
N_TOTAL = len(all_communities)


print(f"\nComputing per-subject measures ({N_TOTAL} subjects) …")

def create_coarse_allegiance(alleg):
    """Reduce 200×200 allegiance → 7×7 network-level."""
    coarse_lr = np.zeros((14, 14))
    p1, q1 = 0, 0
    for _i, p2 in enumerate(pivot):
        for _j, q2 in enumerate(pivot):
            coarse_lr[_i, _j] = np.nanmean(alleg[p1:p2+1, q1:q2+1])
            q1 = q2 + 1
        p1 = p2 + 1
        q1 = 0
    return np.mean(
        coarse_lr.reshape(2, 7, 2, 7).transpose(0, 2, 1, 3).reshape(-1, 7, 7),
        axis=0,
    )


allegiance_per_subj     = []
integration_per_subj    = []
recruitment_per_subj    = []
flexibility_per_subj    = []
promiscuity_per_subj    = []

for s, C in enumerate(all_communities):
    A = communitymeasures.allegiance(C)
    I = communitymeasures.integration(C, static_communities)
    R = communitymeasures.recruitment(C, static_communities)
    F = communitymeasures.flexibility(C)
    P = communitymeasures.promiscuity(C)
    allegiance_per_subj.append(A)
    integration_per_subj.append(I)
    recruitment_per_subj.append(R)
    flexibility_per_subj.append(F)
    promiscuity_per_subj.append(P)
    print(f"  Computed subject {s + 1}/{N_TOTAL}")

N = N_SUBJ_PER_GROUP
alleg_an_subjs = np.array(allegiance_per_subj[:N])
alleg_hc_subjs = np.array(allegiance_per_subj[N:])
rec_an_subjs   = np.array(recruitment_per_subj[:N])
rec_hc_subjs   = np.array(recruitment_per_subj[N:])
int_an_subjs   = np.array(integration_per_subj[:N])
int_hc_subjs   = np.array(integration_per_subj[N:])

alleg_coarse_per_subj = [create_coarse_allegiance(a) for a in allegiance_per_subj]
alleg_coarse_an_subjs = np.array(alleg_coarse_per_subj[:N])
alleg_coarse_hc_subjs = np.array(alleg_coarse_per_subj[N:])

allegiance_mean_an        = np.mean(alleg_an_subjs, axis=0)
allegiance_mean_hc        = np.mean(alleg_hc_subjs, axis=0)
allegiance_coarse_mean_an = np.mean(alleg_coarse_an_subjs, axis=0)
allegiance_coarse_mean_hc = np.mean(alleg_coarse_hc_subjs, axis=0)
recruitment_mean_an       = np.mean(rec_an_subjs, axis=0)
recruitment_mean_hc       = np.mean(rec_hc_subjs, axis=0)
integration_mean_an       = np.mean(int_an_subjs, axis=0)
integration_mean_hc       = np.mean(int_hc_subjs, axis=0)

rec_coarse_an = np.diag(allegiance_coarse_mean_an)
rec_coarse_hc = np.diag(allegiance_coarse_mean_hc)
int_coarse_an = (allegiance_coarse_mean_an.sum(1)
                 - np.diag(allegiance_coarse_mean_an)) / 6
int_coarse_hc = (allegiance_coarse_mean_hc.sum(1)
                 - np.diag(allegiance_coarse_mean_hc)) / 6

print(f"\n  Group means: allegiance AN {allegiance_mean_an.shape}, HC {allegiance_mean_hc.shape}")
print(f"  Coarse allegiance: AN {allegiance_coarse_mean_an.shape}, HC {allegiance_coarse_mean_hc.shape}")
print(f"  Recruitment: AN {recruitment_mean_an.shape}, HC {recruitment_mean_hc.shape}")
print(f"  Integration: AN {integration_mean_an.shape}, HC {integration_mean_hc.shape}")



def perm_test_scalar(xs, ys, nmc=5000):
    """Two-sample permutation test on scalar means."""
    n, k = len(xs), 0
    diff = np.abs(np.mean(xs) - np.mean(ys))
    zs = np.concatenate([xs, ys])
    for _ in range(nmc):
        np.random.shuffle(zs)
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc


print(f"\nPermutation test: fine allegiance (200×200) — {N_PERM} permutations …")

all_alleg = np.array(allegiance_per_subj)   # (10, 200, 200)
obs_diff_alleg = np.abs(allegiance_mean_an - allegiance_mean_hc)


def _single_perm_alleg_subj(all_alleg, obs_diff, n_an=N_SUBJ_PER_GROUP):
    """One permutation: shuffle subject labels, compare group mean allegiance."""
    perm_idx = np.random.permutation(all_alleg.shape[0])
    g1_mean = np.mean(all_alleg[perm_idx[:n_an]], axis=0)
    g2_mean = np.mean(all_alleg[perm_idx[n_an:]], axis=0)
    return (np.abs(g1_mean - g2_mean) >= obs_diff).astype(int)


results_alg_fine = Parallel(n_jobs=N_JOBS)(
    delayed(_single_perm_alleg_subj)(all_alleg, obs_diff_alleg)
    for _ in tqdm(range(N_PERM), desc="Fine allegiance perm (subject-level)")
)
pvalue_alg_fine = np.sum(results_alg_fine, axis=0) / N_PERM

pd.DataFrame(pvalue_alg_fine).to_csv(STAT_OUT / "pvalue_allegiance_fine_200x200.csv", index=False)
print(f"  Saved pvalue_allegiance_fine_200x200.csv  (sig cells at 0.05: {np.sum(pvalue_alg_fine < ALPHA)})")


print(f"\nPermutation test: coarse allegiance (7×7) — {N_PERM} permutations …")

all_alleg_coarse = np.array(alleg_coarse_per_subj)   # (10, 7, 7)
obs_diff_alleg_coarse = np.abs(allegiance_coarse_mean_an - allegiance_coarse_mean_hc)

k_alleg_coarse = np.zeros((7, 7))
for _ in tqdm(range(N_PERM), desc="Coarse allegiance perm (subject-level)"):
    perm_idx = np.random.permutation(N_TOTAL)
    g1_mean = np.mean(all_alleg_coarse[perm_idx[:N]], axis=0)
    g2_mean = np.mean(all_alleg_coarse[perm_idx[N:]], axis=0)
    k_alleg_coarse += (np.abs(g1_mean - g2_mean) >= obs_diff_alleg_coarse).astype(int)

pvalue_alg_coarse = k_alleg_coarse / N_PERM

df_coarse = pd.DataFrame(pvalue_alg_coarse, index=NETWORK_SHORT, columns=NETWORK_SHORT)
df_coarse.to_csv(STAT_OUT / "pvalue_allegiance_coarse_7x7.csv")
df_coarse.to_excel(STAT_OUT / "pvalue_allegiance_coarse_7x7.xlsx")
print(f"  Saved pvalue_allegiance_coarse_7x7")
print(f"  Significant cells (p<0.05):\n{df_coarse[df_coarse < ALPHA].stack().dropna()}")


print(f"\nPermutation test: nodal recruitment (200) — {N_PERM} permutations …")

all_rec = np.array(recruitment_per_subj)   # (10, 200)
obs_diff_rec = np.abs(recruitment_mean_an - recruitment_mean_hc)

k_rec = np.zeros(200)
for _ in tqdm(range(N_PERM), desc="Recruitment perm (subject-level)"):
    perm_idx = np.random.permutation(N_TOTAL)
    g1_mean = np.mean(all_rec[perm_idx[:N]], axis=0)
    g2_mean = np.mean(all_rec[perm_idx[N:]], axis=0)
    k_rec += (np.abs(g1_mean - g2_mean) >= obs_diff_rec).astype(int)

pvalue_rec = k_rec / N_PERM

reject_rec, pvals_corrected_rec, _, _ = smm.multipletests(pvalue_rec, ALPHA, method=FDR_METHOD)

print(f"  Significant nodes (uncorrected p<{ALPHA}): {np.sum(pvalue_rec < ALPHA)}")
print(f"  Significant nodes (FDR-corrected):          {np.sum(reject_rec)}")


print(f"\nPermutation test: nodal integration (200) — {N_PERM} permutations …")

all_int = np.array(integration_per_subj)   # (10, 200)
obs_diff_int = np.abs(integration_mean_an - integration_mean_hc)

k_int = np.zeros(200)
for _ in tqdm(range(N_PERM), desc="Integration perm (subject-level)"):
    perm_idx = np.random.permutation(N_TOTAL)
    g1_mean = np.mean(all_int[perm_idx[:N]], axis=0)
    g2_mean = np.mean(all_int[perm_idx[N:]], axis=0)
    k_int += (np.abs(g1_mean - g2_mean) >= obs_diff_int).astype(int)

pvalue_int = k_int / N_PERM

reject_int, pvals_corrected_int, _, _ = smm.multipletests(pvalue_int, ALPHA, method=FDR_METHOD)

print(f"  Significant nodes (uncorrected p<{ALPHA}): {np.sum(pvalue_int < ALPHA)}")
print(f"  Significant nodes (FDR-corrected):          {np.sum(reject_int)}")


print(f"\nPermutation test: coarse recruitment (7 networks) — {N_PERM} permutations …")

all_coarse_rec = np.array([np.diag(c) for c in alleg_coarse_per_subj])  # (10, 7)
obs_diff_coarse_rec = np.abs(rec_coarse_an - rec_coarse_hc)

k_coarse_rec = np.zeros(7)
for _ in tqdm(range(N_PERM), desc="Coarse recruitment perm (subject-level)"):
    perm_idx = np.random.permutation(N_TOTAL)
    g1_mean = np.mean(all_coarse_rec[perm_idx[:N]], axis=0)
    g2_mean = np.mean(all_coarse_rec[perm_idx[N:]], axis=0)
    k_coarse_rec += (np.abs(g1_mean - g2_mean) >= obs_diff_coarse_rec).astype(int)

pvalue_rec_coarse = k_coarse_rec / N_PERM

df_rec_coarse = pd.DataFrame({
    "Network": NETWORK_SHORT,
    "Rec_AN": rec_coarse_an,
    "Rec_HC": rec_coarse_hc,
    "Diff": obs_diff_coarse_rec,
    "pvalue": pvalue_rec_coarse,
    "Significant": pvalue_rec_coarse < ALPHA,
})
df_rec_coarse.to_csv(STAT_OUT / "pvalue_recruitment_coarse_7.csv", index=False)
print(df_rec_coarse.to_string(index=False))


print(f"\nPermutation test: coarse integration (7 networks) — {N_PERM} permutations …")

all_coarse_int = np.array([
    (c.sum(1) - np.diag(c)) / (c.shape[1] - 1)
    for c in alleg_coarse_per_subj
])  # (10, 7)
obs_diff_coarse_int = np.abs(int_coarse_an - int_coarse_hc)

k_coarse_int = np.zeros(7)
for _ in tqdm(range(N_PERM), desc="Coarse integration perm (subject-level)"):
    perm_idx = np.random.permutation(N_TOTAL)
    g1_mean = np.mean(all_coarse_int[perm_idx[:N]], axis=0)
    g2_mean = np.mean(all_coarse_int[perm_idx[N:]], axis=0)
    k_coarse_int += (np.abs(g1_mean - g2_mean) >= obs_diff_coarse_int).astype(int)

pvalue_int_coarse = k_coarse_int / N_PERM

df_int_coarse = pd.DataFrame({
    "Network": NETWORK_SHORT,
    "Int_AN": int_coarse_an,
    "Int_HC": int_coarse_hc,
    "Diff": obs_diff_coarse_int,
    "pvalue": pvalue_int_coarse,
    "Significant": pvalue_int_coarse < ALPHA,
})
df_int_coarse.to_csv(STAT_OUT / "pvalue_integration_coarse_7.csv", index=False)
print(df_int_coarse.to_string(index=False))


print("\nComputing effect sizes (Cohen's d) per node …")


def cohens_d(x, y):
    """Compute Cohen's d for two independent samples (pooled std)."""
    nx, ny = len(x), len(y)
    mx, my = np.mean(x, axis=0), np.mean(y, axis=0)
    sx, sy = np.std(x, axis=0, ddof=1), np.std(y, axis=0, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    pooled_std[pooled_std == 0] = np.finfo(float).eps
    return (mx - my) / pooled_std


d_rec = cohens_d(rec_an_subjs, rec_hc_subjs)   # (200,)
d_int = cohens_d(int_an_subjs, int_hc_subjs)   # (200,)


print("\nPermutation tests: flexibility and promiscuity …")

flex_an = np.array([np.mean(flexibility_per_subj[s])  for s in range(N)])
flex_hc = np.array([np.mean(flexibility_per_subj[s])  for s in range(N, N_TOTAL)])
prom_an = np.array([np.mean(promiscuity_per_subj[s])  for s in range(N)])
prom_hc = np.array([np.mean(promiscuity_per_subj[s])  for s in range(N, N_TOTAL)])

p_flex = perm_test_scalar(flex_an, flex_hc, N_PERM)
p_prom = perm_test_scalar(prom_an, prom_hc, N_PERM)

print(f"  Flexibility — AN mean: {np.mean(flex_an):.4f}, HC mean: {np.mean(flex_hc):.4f}, p = {p_flex:.4f}")
print(f"  Promiscuity — AN mean: {np.mean(prom_an):.4f}, HC mean: {np.mean(prom_hc):.4f}, p = {p_prom:.4f}")


print("\nSaving comprehensive results …")

df_nodal = pd.DataFrame({
    "ROI": labels,
    "Network": [
        NETWORKS[static_communities[i] - 1] if static_communities[i] > 0 else "Unknown"
        for i in range(200)
    ],
    "Rec_AN_mean": recruitment_mean_an,
    "Rec_HC_mean": recruitment_mean_hc,
    "Rec_diff": recruitment_mean_an - recruitment_mean_hc,
    "Rec_AN_std": np.std(rec_an_subjs, axis=0, ddof=1),
    "Rec_HC_std": np.std(rec_hc_subjs, axis=0, ddof=1),
    "Rec_pvalue": pvalue_rec,
    "Rec_pvalue_FDR": pvals_corrected_rec,
    "Rec_reject_FDR": reject_rec,
    "Int_AN_mean": integration_mean_an,
    "Int_HC_mean": integration_mean_hc,
    "Int_diff": integration_mean_an - integration_mean_hc,
    "Int_AN_std": np.std(int_an_subjs, axis=0, ddof=1),
    "Int_HC_std": np.std(int_hc_subjs, axis=0, ddof=1),
    "Int_pvalue": pvalue_int,
    "Int_pvalue_FDR": pvals_corrected_int,
    "Int_reject_FDR": reject_int,
    "Cohens_d_Rec": d_rec,
    "Cohens_d_Int": d_int,
})

df_nodal.to_csv(STAT_OUT / "nodal_statistics_200.csv", index=False)
df_nodal.to_excel(STAT_OUT / "nodal_statistics_200.xlsx", index=False)
print(f"  Saved nodal_statistics_200.csv / .xlsx ({len(df_nodal)} rows)")

np.save(STAT_OUT / "pvalue_allegiance_fine.npy", pvalue_alg_fine)
np.save(STAT_OUT / "pvalue_allegiance_coarse.npy", pvalue_alg_coarse)
np.save(STAT_OUT / "pvalue_recruitment_nodal.npy", pvalue_rec)
np.save(STAT_OUT / "pvalue_integration_nodal.npy", pvalue_int)
np.save(STAT_OUT / "pvalue_recruitment_coarse.npy", pvalue_rec_coarse)
np.save(STAT_OUT / "pvalue_integration_coarse.npy", pvalue_int_coarse)

np.save(STAT_OUT / "allegiance_group_an.npy", allegiance_mean_an)
np.save(STAT_OUT / "allegiance_group_hc.npy", allegiance_mean_hc)
np.save(STAT_OUT / "allegiance_coarse_an.npy", allegiance_coarse_mean_an)
np.save(STAT_OUT / "allegiance_coarse_hc.npy", allegiance_coarse_mean_hc)
np.save(STAT_OUT / "recruitment_group_an.npy", recruitment_mean_an)
np.save(STAT_OUT / "recruitment_group_hc.npy", recruitment_mean_hc)
np.save(STAT_OUT / "integration_group_an.npy", integration_mean_an)
np.save(STAT_OUT / "integration_group_hc.npy", integration_mean_hc)

np.save(STAT_OUT / "recruitment_subj_an.npy", rec_an_subjs)
np.save(STAT_OUT / "recruitment_subj_hc.npy", rec_hc_subjs)
np.save(STAT_OUT / "integration_subj_an.npy", int_an_subjs)
np.save(STAT_OUT / "integration_subj_hc.npy", int_hc_subjs)

summary = pd.DataFrame({
    "Measure": ["Flexibility", "Promiscuity",
                "Mean_Recruitment", "Mean_Integration"],
    "AN_mean": [np.mean(flex_an), np.mean(prom_an),
                np.mean(rec_an_subjs), np.mean(int_an_subjs)],
    "HC_mean": [np.mean(flex_hc), np.mean(prom_hc),
                np.mean(rec_hc_subjs), np.mean(int_hc_subjs)],
    "pvalue": [p_flex, p_prom,
               perm_test_scalar(np.mean(rec_an_subjs, axis=1), np.mean(rec_hc_subjs, axis=1), N_PERM),
               perm_test_scalar(np.mean(int_an_subjs, axis=1), np.mean(int_hc_subjs, axis=1), N_PERM)],
})
summary.to_csv(STAT_OUT / "summary_global_tests.csv", index=False)
print(f"\n{summary.to_string(index=False)}")


end_time = datetime.now()
print(f"\nStatistical analysis complete  (v2 — {N_SUBJ_PER_GROUP} subjects/group)")
print(f"Output directory: {STAT_OUT}")
print(f"Duration: {end_time - start_time}")
print(f"Files created:")
for f in sorted(STAT_OUT.glob("*")):
    print(f"  {f.name}")
