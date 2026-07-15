#!/usr/bin/env python3
"""
03b_statistical_analysis_subcortical.py — Permutation tests for subcortical measures.
Tian S2 atlas, 32 ROIs, 5+5 subjects.
"""
import logging
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from teneto import communitymeasures
from tqdm import tqdm

start_time = datetime.now()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT     = Path("/Users/ismaila/Documents/C-Codes/AnorexiaProject")
MLCD_DIR = ROOT / "data/analysis/mlcd_subjs_subcortical_5subj/subjs_mlcd"
STAT_OUT = ROOT / "output/results/statistical_results_subcortical_5subj"
STAT_OUT.mkdir(parents=True, exist_ok=True)

N_REGIONS  = 32
N_SUBJ     = 5
N_PERM     = 5000
GROUP_TAGS = ["anorexia", "control"]

LABEL_FILE = ROOT / "data/atlas/tian_s2/Tian_Subcortex_S2_3T_label.txt"
tian_labels = LABEL_FILE.read_text().strip().splitlines()

STRUCT_MAP = {"HIP": 1, "AMY": 2, "THA": 3, "NAc": 4, "GP": 5, "PUT": 6, "CAU": 7}
STRUCT_NAMES = {1:"Hippocampus", 2:"Amygdala", 3:"Thalamus",
                4:"Nuc.Accumbens", 5:"Globus Pallidus", 6:"Putamen", 7:"Caudate"}

def _label_to_community(lbl):
    for key, comm in STRUCT_MAP.items():
        if key.lower() in lbl.lower():
            return comm
    return 0

static_communities = np.array([_label_to_community(l) for l in tian_labels], dtype=int)


def load_group(tag):
    mat_path = MLCD_DIR / f"mlcd_subcortical_{tag}_wins.mat"
    var_key  = f"N_all_g_{tag}"
    with h5py.File(mat_path, "r") as f:
        keys = list(f.keys())
        if var_key not in keys:
            var_key = next(k for k in keys if k.startswith("N_all_g"))
        mat = np.squeeze(np.array(f[var_key][()]))
    if mat.shape[0] != N_REGIONS and mat.shape[1] == N_REGIONS:
        mat = mat.T
    wins_per_subj = mat.shape[1] // N_SUBJ
    return [mat[:, i*wins_per_subj:(i+1)*wins_per_subj] for i in range(N_SUBJ)]

log.info("Loading MLCD outputs …")
subjs_an = load_group("anorexia")
subjs_hc = load_group("control")
all_subjs = subjs_an + subjs_hc

log.info("Computing per-subject measures …")
rec_list  = []; int_list  = []; alg_list  = []
flex_list = []; prom_list = []

for s, C in enumerate(all_subjs):
    grp = "AN" if s < N_SUBJ else "HC"
    A = communitymeasures.allegiance(C)
    R = communitymeasures.recruitment(C, static_communities)
    I = communitymeasures.integration(C, static_communities)
    F = communitymeasures.flexibility(C)
    P = communitymeasures.promiscuity(C)
    rec_list.append(R); int_list.append(I); alg_list.append(A)
    flex_list.append(F); prom_list.append(P)
    log.info(f"  [{grp}] subj {(s%N_SUBJ)+1}/{N_SUBJ} — R={R.mean():.3f} I={I.mean():.3f} F={F.mean():.3f} P={P.mean():.3f}")

rec_an  = np.array(rec_list[:N_SUBJ])
rec_hc  = np.array(rec_list[N_SUBJ:])
int_an  = np.array(int_list[:N_SUBJ])
int_hc  = np.array(int_list[N_SUBJ:])
alg_an  = np.mean(alg_list[:N_SUBJ], axis=0)
alg_hc  = np.mean(alg_list[N_SUBJ:], axis=0)
flex_an = np.array([f.mean() for f in flex_list[:N_SUBJ]])
flex_hc = np.array([f.mean() for f in flex_list[N_SUBJ:]])
prom_an = np.array([p.mean() for p in prom_list[:N_SUBJ]])
prom_hc = np.array([p.mean() for p in prom_list[N_SUBJ:]])

np.save(STAT_OUT / "allegiance_group_an.npy", alg_an)
np.save(STAT_OUT / "allegiance_group_hc.npy", alg_hc)
np.save(STAT_OUT / "recruitment_subj_an.npy", rec_an)
np.save(STAT_OUT / "recruitment_subj_hc.npy", rec_hc)
np.save(STAT_OUT / "integration_subj_an.npy", int_an)
np.save(STAT_OUT / "integration_subj_hc.npy", int_hc)
np.save(STAT_OUT / "flexibility_subj_an.npy", flex_an)
np.save(STAT_OUT / "flexibility_subj_hc.npy", flex_hc)
np.save(STAT_OUT / "promiscuity_subj_an.npy", prom_an)
np.save(STAT_OUT / "promiscuity_subj_hc.npy", prom_hc)

rng = np.random.default_rng(42)

def perm_test_nodal(arr_an, arr_hc, n_perm):
    """Per-node permutation test. arr shape: (n_subj, n_nodes)."""
    obs = arr_an.mean(0) - arr_hc.mean(0)
    all_s = np.concatenate([arr_an, arr_hc], axis=0)
    n_an  = len(arr_an)
    null  = np.zeros((n_perm, arr_an.shape[1]))
    for i in range(n_perm):
        idx = rng.permutation(len(all_s))
        null[i] = all_s[idx[:n_an]].mean(0) - all_s[idx[n_an:]].mean(0)
    p = (np.abs(null) >= np.abs(obs)).mean(0)
    return p

def perm_test_scalar(a, b, n_perm):
    obs  = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    n_a  = len(a)
    null = np.array([abs(pool[rng.permutation(len(pool))[:n_a]].mean() -
                         pool[rng.permutation(len(pool))[n_a:]].mean())
                     for _ in range(n_perm)])
    return (null >= obs).mean()

log.info(f"Running nodal permutation tests (N_PERM={N_PERM}) …")
log.info("  Recruitment …")
p_rec = perm_test_nodal(rec_an, rec_hc, N_PERM)
log.info("  Integration …")
p_int = perm_test_nodal(int_an, int_hc, N_PERM)

rej_rec, p_rec_fdr, _, _ = multipletests(p_rec, alpha=0.05, method="fdr_bh")
rej_int, p_int_fdr, _, _ = multipletests(p_int, alpha=0.05, method="fdr_bh")

log.info(f"  FDR-significant: Recruitment={rej_rec.sum()}/32 | Integration={rej_int.sum()}/32")

def cohens_d(a, b):
    s = np.sqrt(((len(a)-1)*a.std(0,ddof=1)**2 + (len(b)-1)*b.std(0,ddof=1)**2)
                / (len(a)+len(b)-2))
    return (a.mean(0) - b.mean(0)) / (s + 1e-8)

d_rec = cohens_d(rec_an, rec_hc)
d_int = cohens_d(int_an, int_hc)

log.info("Running global scalar permutation tests …")
p_flex = perm_test_scalar(flex_an, flex_hc, N_PERM)
p_prom = perm_test_scalar(prom_an, prom_hc, N_PERM)
p_rec_g = perm_test_scalar(rec_an.mean(1), rec_hc.mean(1), N_PERM)
p_int_g = perm_test_scalar(int_an.mean(1), int_hc.mean(1), N_PERM)

log.info(f"  Flexibility : AN={flex_an.mean():.4f} HC={flex_hc.mean():.4f} p={p_flex:.4f}")
log.info(f"  Promiscuity : AN={prom_an.mean():.4f} HC={prom_hc.mean():.4f} p={p_prom:.4f}")
log.info(f"  Recruitment : AN={rec_an.mean():.4f}  HC={rec_hc.mean():.4f}  p={p_rec_g:.4f}")
log.info(f"  Integration : AN={int_an.mean():.4f}  HC={int_hc.mean():.4f}  p={p_int_g:.4f}")

log.info("Running allegiance permutation tests (nodal) …")
alg_an_arr = np.array(alg_list[:N_SUBJ])   # (5, 32, 32)
alg_hc_arr = np.array(alg_list[N_SUBJ:])

obs_alg = alg_an_arr.mean(0) - alg_hc_arr.mean(0)
all_alg = np.concatenate([alg_an_arr, alg_hc_arr], axis=0)
null_alg = np.zeros((N_PERM, N_REGIONS, N_REGIONS))
for i in range(N_PERM):
    idx = rng.permutation(len(all_alg))
    null_alg[i] = all_alg[idx[:N_SUBJ]].mean(0) - all_alg[idx[N_SUBJ:]].mean(0)
p_alg = (np.abs(null_alg) >= np.abs(obs_alg)).mean(0)
np.save(STAT_OUT / "pvalue_allegiance.npy", p_alg)
np.save(STAT_OUT / "allegiance_diff_AN_HC.npy", obs_alg)

df_nodal = pd.DataFrame({
    "ROI"           : tian_labels,
    "Structure"     : [STRUCT_NAMES.get(static_communities[i], "?") for i in range(N_REGIONS)],
    "Rec_AN_mean"   : rec_an.mean(0),
    "Rec_HC_mean"   : rec_hc.mean(0),
    "Rec_diff"      : rec_an.mean(0) - rec_hc.mean(0),
    "Rec_pvalue"    : p_rec,
    "Rec_pvalue_FDR": p_rec_fdr,
    "Rec_reject_FDR": rej_rec,
    "Int_AN_mean"   : int_an.mean(0),
    "Int_HC_mean"   : int_hc.mean(0),
    "Int_diff"      : int_an.mean(0) - int_hc.mean(0),
    "Int_pvalue"    : p_int,
    "Int_pvalue_FDR": p_int_fdr,
    "Int_reject_FDR": rej_int,
    "Cohens_d_Rec"  : d_rec,
    "Cohens_d_Int"  : d_int,
})
df_nodal.to_csv(STAT_OUT / "nodal_statistics_subcortical.csv", index=False)
df_nodal.to_excel(STAT_OUT / "nodal_statistics_subcortical.xlsx", index=False)
log.info("  Saved nodal_statistics_subcortical.csv/xlsx")

summary = pd.DataFrame({
    "Measure" : ["Flexibility", "Promiscuity", "Mean_Recruitment", "Mean_Integration"],
    "AN_mean" : [flex_an.mean(), prom_an.mean(), rec_an.mean(), int_an.mean()],
    "HC_mean" : [flex_hc.mean(), prom_hc.mean(), rec_hc.mean(), int_hc.mean()],
    "pvalue"  : [p_flex, p_prom, p_rec_g, p_int_g],
    "significant": [p_flex<0.05, p_prom<0.05, p_rec_g<0.05, p_int_g<0.05],
})
summary.to_csv(STAT_OUT / "summary_global_tests.csv", index=False)
log.info("  Saved summary_global_tests.csv")

sig = df_nodal[df_nodal["Rec_reject_FDR"] | df_nodal["Int_reject_FDR"]]
sig.to_csv(STAT_OUT / "significant_regions.csv", index=False)
log.info(f"  Significant regions (FDR): {len(sig)}/32 -> significant_regions.csv")

log.info(f"Complete — duration: {datetime.now() - start_time}")
