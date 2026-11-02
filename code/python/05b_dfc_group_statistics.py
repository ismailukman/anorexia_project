#!/usr/bin/env python3
"""
05b_dfc_group_statistics.py: group-difference statistics on the dynamic FC
(MLCD) outcome measures, with the participant as the unit of observation.

Per-participant measures are derived from the MLCD consensus community labels
(regions x windows): allegiance (N x N), recruitment, integration, flexibility
and promiscuity (per region). Promiscuity is normalised by the community count
of the whole run (mlcd_measures.promiscuity_runN) rather than teneto's
per-participant denominator, which differs between groups. The teneto value is
still written to dfc_global_summary.csv as "Promiscuity_teneto_superseded".
Group differences (AN - HC) are tested with subject-label permutation
(N_PERM permutations), Benjamini-Hochberg FDR and Cohen's d, per region/edge
and per functional group.

Atlases (--atlas):
    cortical200 : Schaefer-200, 7 Yeo groups
    combined216 : Schaefer-200 + Tian S1, 13 functional groups (default)
Cohorts (--cohort): 21v22 (default; one AN participant excluded at QC).

Run:
    python code/python/05b_dfc_group_statistics.py --atlas combined216
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import h5py
from scipy import stats
import statsmodels.stats.multitest as smm
from teneto import communitymeasures

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlcd_measures as mm
from mlcd_measures import (run_community_count, promiscuity_runN,
                           promiscuity_teneto, match_community_affiliation)

# ---------------------------------------------------------------------------
from paths import ROOT

ATLAS_CFG = {
    "cortical200": {
        "mlcd_dir": ROOT / "data/analysis/mlcd_subjs/subjs_mlcd",
        "fname": "mlcd_{tag}_wins.mat",
        "var_base": "N_all_g",
        "n_regions": 200,
        "labels_txt": None,   # built from Schaefer labels via nilearn
        "out": ROOT / "output/results/dfc_statistics_cortical200",
    },
    "combined216": {
        "mlcd_dir": ROOT / "data/analysis/mlcd_subjs_combined_full44_d05/subjs_mlcd",
        "fname": "mlcd_combined_{tag}_wins_full44_d05.mat",
        "var_base": "N_all_g",
        "n_regions": 216,
        "labels_txt": ROOT / "data/atlas/combined_216/combined_216_labels.txt",
        "out": ROOT / "output/results/dfc_statistics_combined216",
    },
}

GROUP_TAGS = ["anorexia", "control"]
N_PER_GROUP = 22
# 700 volumes, 38-TR window, 1-TR step. Fixed rather than inferred from the
# file: the anorexia group file of combined216 also holds AN_K020 and AN_K064
# appended as subjects 23 and 24, and dividing its 15,912 windows by 22 would
# cut every subject at the wrong place.
WINS_PER_SUBJ = 663

# Quality-control exclusions (1-based subject indices within each group).
# anorexia subj08 = errts.an_k032: 17/200 Schaefer parcels have zero signal in
# the preprocessed errts (partial coverage / dropout). Excluded from all group
# statistics so the same cohort is used for the static and dynamic branches.
# The 19v22 sensitivity cohort also removes anorexia subjects 1 and 2.
# Select with --cohort; 19v22 writes to <out>_19v22.
COHORTS = {"21v22": {"anorexia": [8], "control": []},
           "19v22": {"anorexia": [1, 2, 8], "control": []}}
EXCLUDE = COHORTS["21v22"]


def select_cohort(cohort):
    """Set the exclusions used by load_group for this cohort."""
    global EXCLUDE
    EXCLUDE = COHORTS[cohort]


def cohort_out(cfg, cohort):
    """Output directory of 05b for an atlas and cohort."""
    return cfg["out"] if cohort == "21v22" else cfg["out"].with_name(cfg["out"].name + "_19v22")

N_PERM   = int(os.environ.get("N_PERM", 20000))
ALPHA    = 0.05
FDR      = "fdr_bh"
SEED     = 42


# ---------------------------------------------------------------------------
def build_group_vector(atlas, n_regions):
    """Return (group_vec [1..G], group_names, group_full)."""
    if atlas == "cortical200":
        from nilearn import datasets
        at = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7,
                                                resolution_mm=1)
        labels = np.array(at.labels).astype("U")
        labels = labels[1:] if labels[0].lower().find("background") >= 0 or len(labels) == 201 else labels
        keys = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
        names = ["VN", "SMN", "DAN", "VAN", "LN", "FPN", "DMN"]
        gv = np.zeros(n_regions, dtype=int)
        for i, lbl in enumerate(labels):
            for gi, k in enumerate(keys):
                if k in lbl:
                    gv[i] = gi + 1
                    break
        roi_labels = list(labels)
        return gv, names, names, roi_labels
    else:
        from atlas_groups import combined216_groups
        gv, names, labels = combined216_groups()
        assert len(labels) == n_regions
        return gv, names, names, labels


def read_var_any(f, base, tag):
    for name in (f"{base}_{tag}", f"{base}{tag}"):
        if name in f.keys():
            return np.squeeze(np.asarray(f[name][()]))
    raise KeyError(f"{base}_{tag} not found. keys={list(f.keys())}")


def load_group(cfg, tag, n_regions):
    """Return list of per-subject community-label matrices (n_regions, T)."""
    path = cfg["mlcd_dir"] / cfg["fname"].format(tag=tag)
    with h5py.File(path, "r") as f:
        mat = read_var_any(f, cfg["var_base"], tag)
    mat = np.asarray(mat)
    # orient to (n_regions, total_windows)
    if mat.shape[0] != n_regions and mat.shape[1] == n_regions:
        mat = mat.T
    assert mat.shape[0] == n_regions, f"bad shape {mat.shape}"
    wins = WINS_PER_SUBJ
    if mat.shape[1] % wins:
        raise ValueError(f"{path.name}: {mat.shape[1]} layers is not a whole "
                         f"number of {wins}-window subjects")
    n_in_file = mat.shape[1] // wins
    if n_in_file < N_PER_GROUP:
        raise ValueError(f"{path.name}: {n_in_file} subjects, expected {N_PER_GROUP}")
    if n_in_file > N_PER_GROUP:
        print(f"  {path.name} holds {n_in_file} subjects; "
              f"using subjects 1-{N_PER_GROUP}")
        mat = mat[:, :N_PER_GROUP * wins]
    subs = [mat[:, i * wins:(i + 1) * wins] for i in range(N_PER_GROUP)]
    drop = set(i - 1 for i in EXCLUDE.get(tag, []))          # 1-based -> 0-based
    subs = [s for i, s in enumerate(subs) if i not in drop]
    return subs, wins, mat


def coarse_from_alleg(alleg, group_vec, n_groups):
    """Average an NxN allegiance matrix into a GxG functional-group matrix."""
    out = np.zeros((n_groups, n_groups))
    for a in range(1, n_groups + 1):
        ia = np.where(group_vec == a)[0]
        for b in range(a, n_groups + 1):
            ib = np.where(group_vec == b)[0]
            if a == b:
                sub = alleg[np.ix_(ia, ia)]
                m = np.nanmean(sub[np.triu_indices(len(ia), 1)]) if len(ia) > 1 else np.nan
            else:
                m = np.nanmean(alleg[np.ix_(ia, ib)])
            out[a - 1, b - 1] = out[b - 1, a - 1] = m
    return out


def tstat(x, lab, n1, n2):
    a, b = x[lab == 0], x[lab == 1]
    va, vb = a.var(0, ddof=1), b.var(0, ddof=1)
    sp = np.sqrt(((n1 - 1) * va + (n2 - 1) * vb) / (n1 + n2 - 2))
    sp = np.where(sp == 0, np.finfo(float).eps, sp)
    return (a.mean(0) - b.mean(0)) / (sp * np.sqrt(1.0 / n1 + 1.0 / n2))


def cohens_d(x, lab, n1, n2):
    a, b = x[lab == 0], x[lab == 1]
    sp = np.sqrt(((n1 - 1) * a.var(0, ddof=1) + (n2 - 1) * b.var(0, ddof=1))
                 / (n1 + n2 - 2))
    sp = np.where(sp == 0, np.finfo(float).eps, sp)
    return (a.mean(0) - b.mean(0)) / sp


def perm_pvalues(x, lab, rng, n1, n2, n_perm):
    """Two-sided add-one permutation p per column of x (subjects x features)."""
    t_obs = tstat(x, lab, n1, n2)
    count = np.zeros(x.shape[1])
    at = np.abs(t_obs)
    for _ in range(n_perm):
        lp = lab[rng.permutation(n1 + n2)]
        count += (np.abs(tstat(x, lp, n1, n2)) >= at)
    return t_obs, (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", choices=list(ATLAS_CFG), default="combined216")
    ap.add_argument("--cohort", choices=list(COHORTS), default="21v22")
    ap.add_argument("--groups", choices=["primary", "alt"], default="primary",
                    help="subcortical network assignment (alt = runner-up networks)")
    args = ap.parse_args()
    if args.groups == "alt":
        import atlas_groups
        atlas_groups.MAPPING = "network_alt"
    atlas = args.atlas
    cfg = ATLAS_CFG[atlas]
    N = cfg["n_regions"]
    select_cohort(args.cohort)
    out = cohort_out(cfg, args.cohort)
    if args.groups == "alt":
        out = out.with_name(out.name + "_altgroups")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    start = datetime.now()

    group_vec, gnames, _, roi_labels = build_group_vector(atlas, N)
    n_groups = len(gnames)
    print(f"=== DFC group statistics | atlas={atlas} | N={N} | "
          f"{n_groups} functional groups | perms={N_PERM} ===")

    subs_an, wins_an, mat_an = load_group(cfg, "anorexia", N)
    subs_hc, wins_hc, mat_hc = load_group(cfg, "control", N)

    print(f"AN: {len(subs_an)} subj, {wins_an} win/subj | "
          f"HC: {len(subs_hc)} subj, {wins_hc} win/subj")
    all_comms_raw = subs_an + subs_hc
    n1, n2 = len(subs_an), len(subs_hc)

    # Community labels are assigned independently in each layer, so the same
    # integer names different communities in different layers. Flexibility and
    # promiscuity read those integers and are meaningless until the labels are
    # matched across layers. Allegiance, recruitment and integration only ask
    # whether two nodes share a label inside one layer, so they are unchanged
    # by the alignment; that invariance is asserted below.
    # The MATLAB pipeline computes this alignment as
    # multi_module_consensus_sorted but saves the unaligned
    # multi_module_consensus, so it has to be applied here.
    print("aligning community labels across layers ...")
    all_comms = [match_community_affiliation(C) for C in all_comms_raw]

    # Promiscuity denominator, fixed for the whole run rather than per subject.
    # After alignment a community that disappears and reappears is a new
    # identity, so this count is much larger than the per-layer label range.
    run_ncomm = run_community_count(all_comms)
    print(f"distinct community identities after alignment = {run_ncomm} "
          f"(promiscuity denominator = {run_ncomm - 1}); "
          f"raw per-layer label count was "
          f"{run_community_count([mat_an, mat_hc])}")
    if EXCLUDE.get("anorexia") or EXCLUDE.get("control"):
        print(f"QC-excluded -> {EXCLUDE}  (final n = {n1} AN vs {n2} HC)")
    lab = np.array([0] * n1 + [1] * n2)

    # per-subject outcome measures
    A_list, R_list, I_list, F_list, P_list, Cc_list = [], [], [], [], [], []
    Pt_list, Ft_list = [], []               # superseded, unaligned labels
    for k, C in enumerate(all_comms):
        # one allegiance per subject, reused for recruitment and integration.
        # mm.allegiance is numerically identical to teneto's but ~250x faster;
        # teneto's recruitment/integration would each recompute it internally.
        A = mm.allegiance(C)
        A_list.append(A)
        R_list.append(mm.recruitment(alleg=A, static_communities=group_vec))
        I_list.append(mm.integration(alleg=A, static_communities=group_vec))
        F_list.append(mm.flexibility(C))
        P_list.append(promiscuity_runN(C, run_ncomm))
        # superseded: raw labels, teneto's per-subject denominator, NaN
        # treated as a community. Kept so the change stays auditable.
        Ft_list.append(mm.flexibility(all_comms_raw[k], nan_aware=False))
        Pt_list.append(promiscuity_teneto(all_comms_raw[k]))
        if k == 0:
            # allegiance must not depend on how the labels are named
            A_raw = mm.allegiance(all_comms_raw[k])
            if not np.allclose(np.nan_to_num(A), np.nan_to_num(A_raw), atol=1e-12):
                raise AssertionError(
                    "allegiance changed after label alignment; it should be "
                    "invariant to relabelling, so something is wrong")
            print("  checked: allegiance is unchanged by the label alignment")
        Cc_list.append(coarse_from_alleg(A, group_vec, n_groups))
        print(f"  measures {k + 1}/{len(all_comms)}")

    A_arr = np.stack(A_list)               # (44, N, N)
    R_arr = np.stack(R_list)               # (44, N)
    I_arr = np.stack(I_list)               # (44, N)
    Cc_arr = np.stack(Cc_list)             # (44, G, G)
    F_arr = np.stack(F_list)               # (44, N) nodal flexibility
    P_arr = np.stack(P_list)               # (44, N) nodal promiscuity (run-wide norm)
    flex = np.array([np.mean(f) for f in F_list])   # (44,)
    prom = np.array([np.mean(p) for p in P_list])   # (44,)
    prom_teneto = np.array([np.mean(p) for p in Pt_list])   # (44,) superseded
    flex_raw = np.array([np.mean(f) for f in Ft_list])      # (44,) superseded

    iu, ju = np.triu_indices(N, 1)
    A_edges = A_arr[:, iu, ju]             # (44, n_edges)
    ca, cb = np.triu_indices(n_groups, 0)
    Cc_edges = Cc_arr[:, ca, cb]           # (44, n_cblocks)

    # ---- permutation tests ----
    print("\nFine allegiance permutation ...")
    t_alleg, p_alleg = perm_pvalues(A_edges, lab, rng, n1, n2, N_PERM)
    rej_alleg, pfdr_alleg, _, _ = smm.multipletests(p_alleg, ALPHA, method=FDR)
    d_alleg = cohens_d(A_edges, lab, n1, n2)

    print("Coarse allegiance permutation ...")
    t_coarse, p_coarse = perm_pvalues(Cc_edges, lab, rng, n1, n2, N_PERM)
    rej_coarse, pfdr_coarse, _, _ = smm.multipletests(p_coarse, ALPHA, method=FDR)
    d_coarse = cohens_d(Cc_edges, lab, n1, n2)

    print("Nodal recruitment permutation ...")
    t_rec, p_rec = perm_pvalues(R_arr, lab, rng, n1, n2, N_PERM)
    rej_rec, pfdr_rec, _, _ = smm.multipletests(p_rec, ALPHA, method=FDR)
    d_rec = cohens_d(R_arr, lab, n1, n2)

    print("Nodal integration permutation ...")
    t_int, p_int = perm_pvalues(I_arr, lab, rng, n1, n2, N_PERM)
    rej_int, pfdr_int, _, _ = smm.multipletests(p_int, ALPHA, method=FDR)
    d_int = cohens_d(I_arr, lab, n1, n2)

    print("Nodal flexibility permutation ...")
    t_flexn, p_flexn = perm_pvalues(F_arr, lab, rng, n1, n2, N_PERM)
    rej_flexn, pfdr_flexn, _, _ = smm.multipletests(p_flexn, ALPHA, method=FDR)
    d_flexn = cohens_d(F_arr, lab, n1, n2)

    print("Nodal promiscuity permutation ...")
    t_promn, p_promn = perm_pvalues(P_arr, lab, rng, n1, n2, N_PERM)
    rej_promn, pfdr_promn, _, _ = smm.multipletests(p_promn, ALPHA, method=FDR)
    d_promn = cohens_d(P_arr, lab, n1, n2)

    print("Global flexibility & promiscuity permutation ...")
    _, p_flex = perm_pvalues(flex[:, None], lab, rng, n1, n2, N_PERM)
    _, p_prom = perm_pvalues(prom[:, None], lab, rng, n1, n2, N_PERM)
    d_flex = cohens_d(flex[:, None], lab, n1, n2)[0]
    d_prom = cohens_d(prom[:, None], lab, n1, n2)[0]

    # superseded values on unaligned labels, kept so the change is auditable
    _, p_prom_ten = perm_pvalues(prom_teneto[:, None], lab, rng, n1, n2, N_PERM)
    d_prom_ten = cohens_d(prom_teneto[:, None], lab, n1, n2)[0]
    _, p_flex_raw = perm_pvalues(flex_raw[:, None], lab, rng, n1, n2, N_PERM)
    d_flex_raw = cohens_d(flex_raw[:, None], lab, n1, n2)[0]

    # ---- multiple-comparison correction across the four nodal measures ----
    # `*_pfdr` above is Benjamini-Hochberg within one measure family (N tests).
    # `*_pfdr_all` corrects across all four nodal measures together
    # (4 x N tests), which is the stricter reading of the same question.
    p_nodal_all = np.concatenate([p_rec, p_int, p_flexn, p_promn])
    rej_all, pfdr_all, _, _ = smm.multipletests(p_nodal_all, ALPHA, method=FDR)
    pfdr_rec_all, pfdr_int_all, pfdr_flex_all, pfdr_prom_all = np.split(pfdr_all, 4)
    rej_rec_all, rej_int_all, rej_flex_all, rej_prom_all = np.split(rej_all, 4)

    # ---- save ----
    np.savez(out / "dfc_group_stats.npz",
             atlas=atlas, group_vec=group_vec, group_names=np.array(gnames),
             iu=iu, ju=ju,
             alleg_mean_an=A_arr[:n1].mean(0), alleg_mean_hc=A_arr[n1:].mean(0),
             coarse_mean_an=Cc_arr[:n1].mean(0), coarse_mean_hc=Cc_arr[n1:].mean(0),
             rec_mean_an=R_arr[:n1].mean(0), rec_mean_hc=R_arr[n1:].mean(0),
             int_mean_an=I_arr[:n1].mean(0), int_mean_hc=I_arr[n1:].mean(0),
             rec_subj_an=R_arr[:n1], rec_subj_hc=R_arr[n1:],
             int_subj_an=I_arr[:n1], int_subj_hc=I_arr[n1:],
             t_alleg=t_alleg, p_alleg=p_alleg, pfdr_alleg=pfdr_alleg, d_alleg=d_alleg,
             t_coarse=t_coarse, p_coarse=p_coarse, pfdr_coarse=pfdr_coarse, d_coarse=d_coarse,
             t_rec=t_rec, p_rec=p_rec, pfdr_rec=pfdr_rec, d_rec=d_rec,
             t_int=t_int, p_int=p_int, pfdr_int=pfdr_int, d_int=d_int,
             flex_an=flex[:n1], flex_hc=flex[n1:], prom_an=prom[:n1], prom_hc=prom[n1:],
             flex_subj_an=F_arr[:n1], flex_subj_hc=F_arr[n1:],
             prom_subj_an=P_arr[:n1], prom_subj_hc=P_arr[n1:],
             t_flex=t_flexn, p_flex=p_flexn, pfdr_flex=pfdr_flexn, d_flex_nodal=d_flexn,
             t_prom=t_promn, p_prom=p_promn, pfdr_prom=pfdr_promn, d_prom_nodal=d_promn,
             run_ncomm=run_ncomm,
             pfdr_rec_all=pfdr_rec_all, pfdr_int_all=pfdr_int_all,
             pfdr_flex_all=pfdr_flex_all, pfdr_prom_all=pfdr_prom_all,
             prom_teneto_an=prom_teneto[:n1], prom_teneto_hc=prom_teneto[n1:],
             flex_raw_an=flex_raw[:n1], flex_raw_hc=flex_raw[n1:])

    # nodal table
    nodal = pd.DataFrame({
        "roi": np.arange(1, N + 1),
        "label": roi_labels,
        "group": [gnames[g - 1] for g in group_vec],
        "rec_AN": R_arr[:n1].mean(0), "rec_HC": R_arr[n1:].mean(0),
        "rec_diff": R_arr[:n1].mean(0) - R_arr[n1:].mean(0),
        "rec_t": t_rec, "rec_d": d_rec, "rec_p": p_rec,
        "rec_pfdr": pfdr_rec, "rec_sig_fdr": rej_rec,
        "int_AN": I_arr[:n1].mean(0), "int_HC": I_arr[n1:].mean(0),
        "int_diff": I_arr[:n1].mean(0) - I_arr[n1:].mean(0),
        "int_t": t_int, "int_d": d_int, "int_p": p_int,
        "int_pfdr": pfdr_int, "int_sig_fdr": rej_int,
        "flex_AN": F_arr[:n1].mean(0), "flex_HC": F_arr[n1:].mean(0),
        "flex_diff": F_arr[:n1].mean(0) - F_arr[n1:].mean(0),
        "flex_t": t_flexn, "flex_d": d_flexn, "flex_p": p_flexn,
        "flex_pfdr": pfdr_flexn, "flex_sig_fdr": rej_flexn,
        "prom_AN": P_arr[:n1].mean(0), "prom_HC": P_arr[n1:].mean(0),
        "prom_diff": P_arr[:n1].mean(0) - P_arr[n1:].mean(0),
        "prom_t": t_promn, "prom_d": d_promn, "prom_p": p_promn,
        "prom_pfdr": pfdr_promn, "prom_sig_fdr": rej_promn,
        "rec_pfdr_all": pfdr_rec_all,   "rec_sig_fdr_all": rej_rec_all,
        "int_pfdr_all": pfdr_int_all,   "int_sig_fdr_all": rej_int_all,
        "flex_pfdr_all": pfdr_flex_all, "flex_sig_fdr_all": rej_flex_all,
        "prom_pfdr_all": pfdr_prom_all, "prom_sig_fdr_all": rej_prom_all,
    })
    nodal.to_csv(out / "dfc_nodal_statistics.csv", index=False)
    nodal.to_excel(out / "dfc_nodal_statistics.xlsx", index=False)

    # coarse block table
    cblk = []
    diff_c = Cc_arr[:n1].mean(0) - Cc_arr[n1:].mean(0)
    for k, (a, b) in enumerate(zip(ca, cb)):
        cblk.append({"block": f"{gnames[a]}-{gnames[b]}",
                     "type": "within" if a == b else "between",
                     "alleg_AN": Cc_arr[:n1].mean(0)[a, b],
                     "alleg_HC": Cc_arr[n1:].mean(0)[a, b],
                     "diff_AN_HC": diff_c[a, b],
                     "t": t_coarse[k], "cohens_d": d_coarse[k],
                     "p": p_coarse[k], "p_fdr": pfdr_coarse[k],
                     "sig_fdr": rej_coarse[k]})
    pd.DataFrame(cblk).sort_values("p").to_csv(
        out / "dfc_coarse_block_tests.csv", index=False)

    # global summary
    gsum = pd.DataFrame({
        "measure": ["Flexibility", "Promiscuity",
                    "Mean_Recruitment", "Mean_Integration",
                    "Promiscuity_unaligned_superseded",
                    "Flexibility_unaligned_superseded"],
        "AN_mean": [flex[:n1].mean(), prom[:n1].mean(),
                    R_arr[:n1].mean(), I_arr[:n1].mean(),
                    prom_teneto[:n1].mean(), flex_raw[:n1].mean()],
        "HC_mean": [flex[n1:].mean(), prom[n1:].mean(),
                    R_arr[n1:].mean(), I_arr[n1:].mean(),
                    prom_teneto[n1:].mean(), flex_raw[n1:].mean()],
        "cohens_d": [d_flex, d_prom,
                     cohens_d(R_arr.mean(1)[:, None], lab, n1, n2)[0],
                     cohens_d(I_arr.mean(1)[:, None], lab, n1, n2)[0],
                     d_prom_ten, d_flex_raw],
        "pvalue": [p_flex[0], p_prom[0],
                   perm_pvalues(R_arr.mean(1)[:, None], lab, rng, n1, n2, N_PERM)[1][0],
                   perm_pvalues(I_arr.mean(1)[:, None], lab, rng, n1, n2, N_PERM)[1][0],
                   p_prom_ten[0], p_flex_raw[0]],
    })
    gsum.to_csv(out / "dfc_global_summary.csv", index=False)

    # how many of the N regions differ between groups, per measure
    counts = []
    for mname, pv, pf, pfa in [
        ("Recruitment", p_rec,   pfdr_rec,   pfdr_rec_all),
        ("Integration", p_int,   pfdr_int,   pfdr_int_all),
        ("Flexibility", p_flexn, pfdr_flexn, pfdr_flex_all),
        ("Promiscuity", p_promn, pfdr_promn, pfdr_prom_all),
    ]:
        counts.append(dict(
            measure=mname, n_regions=N,
            n_uncorrected_p05=int((pv < ALPHA).sum()),
            n_fdr_within_measure=int((pf < ALPHA).sum()),
            n_fdr_across_measures=int((pfa < ALPHA).sum()),
            min_p=float(pv.min()), min_pfdr=float(pf.min()),
            min_pfdr_all=float(pfa.min()),
        ))
    pd.DataFrame(counts).to_csv(out / "dfc_nodal_significance_counts.csv", index=False)
    print("\nSignificant regions out of", N)
    print(pd.DataFrame(counts)[["measure", "n_uncorrected_p05",
                                "n_fdr_within_measure",
                                "n_fdr_across_measures"]].to_string(index=False))

    print("\n================ DFC GROUP-DIFFERENCE SUMMARY ================")
    print(f"atlas={atlas}  edges={A_edges.shape[1]}  perms={N_PERM}")
    print(f"Fine allegiance sig (uncorr / FDR): "
          f"{(p_alleg < ALPHA).sum()} / {rej_alleg.sum()}")
    print(f"Coarse allegiance sig (uncorr / FDR): "
          f"{(p_coarse < ALPHA).sum()} / {rej_coarse.sum()}")
    print(f"Nodal recruitment sig (uncorr / FDR): "
          f"{(p_rec < ALPHA).sum()} / {rej_rec.sum()}")
    print(f"Nodal integration sig (uncorr / FDR): "
          f"{(p_int < ALPHA).sum()} / {rej_int.sum()}")
    print(gsum.to_string(index=False))
    print(f"\nDuration: {datetime.now() - start}  ->  {out}")


if __name__ == "__main__":
    main()
