#!/usr/bin/env python3
"""
14_individual_vs_group.py: individual-pair tests and the group test built from
all pairs, for the dynamic outcome measures.

Two analyses, answering two different questions.

PART A: two named participants
------------------------------
Each participant has 663 sliding windows, and every outcome measure is an
average over those windows, so a measure has a sampling distribution within a
participant and two participants can be compared directly.

Windows advance by 1 TR with a 38-TR width, so neighbouring windows share 37 of
their 38 time points and are not independent. The windows are therefore cut
into BLOCKS of 39 consecutive windows, giving 17 near-independent blocks per
participant, and the measures are computed inside each block. The test pools
the two participants' 34 blocks, relabels them 17 against 17, and recompares;
single-window permutation would instead treat 663 overlapping windows as
independent and declare almost everything significant.

A significant region means these two scans differ by more than the fluctuation
within them. It is a statement about two people, not about the groups: ordinary
between-person variation produces the same result, which is why a same-group
pair is run alongside as the reference.

PART B: all pairs at once
-------------------------
Comparing every cross-group pair against every within-group pair turns the same
raw material into a group test. With a subject-by-subject distance matrix this
is PERMANOVA (Anderson 2001, Austral Ecol 26:32-46): the pseudo-F statistic
compares between-group to within-group distance, and subject labels are
permuted, so exchangeability holds at the level of the participant.

PERMANOVA can be driven by the two groups having different spread rather than
different location, so a dispersion check (PERMDISP) is reported next to it.

Outputs, into output/results/individual_vs_group_<atlas>/:
    pairwise_<A>_vs_<B>.csv        per-region result for each named pair
    pairwise_summary.csv           regions differing, per pair and measure
    allpairs_distances.csv         every pair's distance, cross and within
    mdmr_results.csv               PERMANOVA and PERMDISP per input

Run:
    conda activate fmri
    python 14_individual_vs_group.py --atlas combined216
"""
import argparse
import importlib.util
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlcd_measures as mm
from mlcd_measures import match_community_affiliation, promiscuity_runN, run_community_count

from paths import ROOT
ALPHA = 0.05
FDR = "fdr_bh"
N_BLOCKS = 17          # 17 x 39 = 663 windows
N_PERM_PAIR = 20000
N_PERM_MDMR = 20000
SEED = 42
N_PER_GROUP = 22

_spec = importlib.util.spec_from_file_location(
    "dfc05b", Path(__file__).resolve().parent / "05b_dfc_group_statistics.py")
_dfc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dfc)


def block_measures(C, group_vec, run_ncomm, n_blocks=N_BLOCKS):
    """Measures computed inside each block of consecutive windows.

    Returns a dict of arrays shaped (n_blocks, n_regions).
    """
    T = C.shape[1]
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    out = {k: [] for k in ("rec", "int", "flex", "prom")}
    for b in range(n_blocks):
        Cb = C[:, edges[b]:edges[b + 1]]
        A = mm.allegiance(Cb)
        out["rec"].append(mm.recruitment(alleg=A, static_communities=group_vec))
        out["int"].append(mm.integration(alleg=A, static_communities=group_vec))
        out["flex"].append(mm.flexibility(Cb))
        out["prom"].append(promiscuity_runN(Cb, run_ncomm))
    return {k: np.vstack(v) for k, v in out.items()}


def pair_test(blocks_a, blocks_b, rng, n_perm=N_PERM_PAIR):
    """Block-permutation test between two participants, per region."""
    na = blocks_a.shape[0]
    pooled = np.vstack([blocks_a, blocks_b])
    obs = np.abs(np.nanmean(blocks_a, 0) - np.nanmean(blocks_b, 0))
    count = np.zeros(obs.size)
    for _ in range(n_perm):
        idx = rng.permutation(pooled.shape[0])
        d = np.abs(np.nanmean(pooled[idx[:na]], 0) - np.nanmean(pooled[idx[na:]], 0))
        count += d >= obs
    return obs, (count + 1) / (n_perm + 1)


def permanova(D, labels, rng, n_perm=N_PERM_MDMR):
    """Pseudo-F for a grouping on a distance matrix, with a permutation p."""
    D2 = D ** 2
    N = D.shape[0]
    groups = np.unique(labels)
    a = groups.size

    def ss(lab):
        sst = D2[np.triu_indices(N, 1)].sum() / N
        ssw = 0.0
        for g in groups:
            i = np.where(lab == g)[0]
            if i.size > 1:
                ssw += D2[np.ix_(i, i)][np.triu_indices(i.size, 1)].sum() / i.size
        return sst - ssw, ssw

    ssa, ssw = ss(labels)
    F_obs = (ssa / (a - 1)) / (ssw / (N - a))
    count = 0
    for _ in range(n_perm):
        f_a, f_w = ss(labels[rng.permutation(N)])
        count += ((f_a / (a - 1)) / (f_w / (N - a))) >= F_obs
    return F_obs, (count + 1) / (n_perm + 1)


def permdisp(D, labels, rng, n_perm=N_PERM_MDMR):
    """Do the groups differ in spread? Distance to the group medoid."""
    disp = np.zeros(D.shape[0])
    for g in np.unique(labels):
        i = np.where(labels == g)[0]
        medoid = i[np.argmin(D[np.ix_(i, i)].sum(1))]
        disp[i] = D[i, medoid]
    a = disp[labels == labels[0]]
    b = disp[labels != labels[0]]
    obs = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b]); n1 = a.size
    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(pooled.size)
        count += abs(pooled[idx[:n1]].mean() - pooled[idx[n1:]].mean()) >= obs
    return obs, (count + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", choices=list(_dfc.ATLAS_CFG), default="combined216")
    ap.add_argument("--cohort", choices=list(_dfc.COHORTS), default="21v22")
    ap.add_argument("--n-perm-pair", type=int, default=N_PERM_PAIR)
    args = ap.parse_args()
    _dfc.select_cohort(args.cohort)
    stats_dir = _dfc.cohort_out(_dfc.ATLAS_CFG[args.atlas], args.cohort)

    t0 = datetime.now()
    cfg = _dfc.ATLAS_CFG[args.atlas]
    N = cfg["n_regions"]
    group_vec, gnames, _, roi_labels = _dfc.build_group_vector(args.atlas, N)
    out_dir = ROOT / f"output/results/individual_vs_group_{args.atlas}"
    if args.cohort != "21v22":
        out_dir = out_dir.with_name(out_dir.name + f"_{args.cohort}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    subs_an, _, mat_an = _dfc.load_group(cfg, "anorexia", N)
    subs_hc, _, mat_hc = _dfc.load_group(cfg, "control", N)
    an_ids = [i for i in range(1, N_PER_GROUP + 1) if i not in _dfc.EXCLUDE["anorexia"]]
    hc_ids = [i for i in range(1, N_PER_GROUP + 1) if i not in _dfc.EXCLUDE["control"]]
    print(f"=== atlas={args.atlas} | {len(subs_an)} AN vs {len(subs_hc)} HC | "
          f"{N} regions | {N_BLOCKS} blocks per participant ===")

    print("aligning community labels ...")
    subs_an = [match_community_affiliation(C) for C in subs_an]
    subs_hc = [match_community_affiliation(C) for C in subs_hc]
    run_ncomm = run_community_count(subs_an + subs_hc)

    print(f"computing per-block measures ({N_BLOCKS} blocks x "
          f"{len(subs_an) + len(subs_hc)} participants) ...")
    ids = [("AN", s) for s in an_ids] + [("HC", s) for s in hc_ids]
    allC = subs_an + subs_hc
    blocks = [block_measures(C, group_vec, run_ncomm) for C in allC]

    # ---------------- PART A: named pairs ----------------
    whole = np.array([b["rec"].mean() for b in blocks])
    n_an = len(subs_an)
    med_an = int(np.argmin(np.abs(whole[:n_an] - np.median(whole[:n_an]))))
    med_hc = n_an + int(np.argmin(np.abs(whole[n_an:] - np.median(whole[n_an:]))))
    other_hc = n_an + int(np.argsort(np.abs(
        whole[n_an:] - np.median(whole[n_an:])))[1])
    pairs = [(med_an, med_hc, "cross-group"),
             (med_hc, other_hc, "within-group reference")]

    rows = []
    for ia, ib, kind in pairs:
        ga, sa = ids[ia]; gb, sb = ids[ib]
        tag = f"{ga}{sa:02d}_vs_{gb}{sb:02d}"
        print(f"\n  pair {tag}  ({kind})")
        per_region = {"roi": np.arange(1, N + 1), "label": roi_labels,
                      "functional_group": [gnames[g - 1] for g in group_vec]}
        for key, lab in (("rec", "Recruitment"), ("int", "Integration"),
                         ("flex", "Flexibility"), ("prom", "Promiscuity")):
            obs, p = pair_test(blocks[ia][key], blocks[ib][key], rng,
                               args.n_perm_pair)
            pf = smm.multipletests(p, ALPHA, method=FDR)[1]
            per_region[f"{key}_abs_diff"] = obs
            per_region[f"{key}_p"] = p
            per_region[f"{key}_pfdr"] = pf
            rows.append(dict(pair=tag, kind=kind, measure=lab, n_regions=N,
                             n_nominal=int((p < ALPHA).sum()),
                             n_fdr=int((pf < ALPHA).sum())))
            print(f"    {lab:12s} {int((p < ALPHA).sum()):3d}/{N} at p<0.05, "
                  f"{int((pf < ALPHA).sum()):3d} after FDR")
        pd.DataFrame(per_region).to_csv(out_dir / f"pairwise_{tag}.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "pairwise_summary.csv", index=False)

    # ---------------- PART B: all pairs, group test ----------------
    print("\nbuilding subject-by-subject distances ...")
    labels = np.array([0] * n_an + [1] * len(subs_hc))
    feats = {}
    A_all = np.stack([mm.allegiance(C) for C in allC])
    iu, ju = np.triu_indices(N, 1)
    feats["allegiance edges"] = A_all[:, iu, ju]
    z = np.load(stats_dir / "dfc_group_stats.npz", allow_pickle=True)
    for key, lab in (("rec", "Recruitment"), ("int", "Integration"),
                     ("flex", "Flexibility"), ("prom", "Promiscuity")):
        feats[lab] = np.vstack([z[f"{key}_subj_an"], z[f"{key}_subj_hc"]])

    mdmr_rows, dist_rows = [], []
    for name, X in feats.items():
        X = np.nan_to_num(X)
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        F, p = permanova(D, labels, np.random.default_rng(SEED))
        stat, pd_ = permdisp(D, labels, np.random.default_rng(SEED))
        cross = [D[i, j] for i, j in combinations(range(len(ids)), 2)
                 if labels[i] != labels[j]]
        within = [D[i, j] for i, j in combinations(range(len(ids)), 2)
                  if labels[i] == labels[j]]
        mdmr_rows.append(dict(input=name, pseudo_F=F, permanova_p=p,
                              n_cross_pairs=len(cross), n_within_pairs=len(within),
                              mean_cross=np.mean(cross), mean_within=np.mean(within),
                              permdisp_stat=stat, permdisp_p=pd_))
        print(f"  {name:18s} pseudo-F={F:.3f} p={p:.4f} | "
              f"cross {np.mean(cross):.4f} vs within {np.mean(within):.4f} | "
              f"PERMDISP p={pd_:.4f}")
        if name == "allegiance edges":
            for i, j in combinations(range(len(ids)), 2):
                dist_rows.append(dict(
                    a=f"{ids[i][0]}{ids[i][1]:02d}", b=f"{ids[j][0]}{ids[j][1]:02d}",
                    kind="cross-group" if labels[i] != labels[j] else "within-group",
                    within_which=("-" if labels[i] != labels[j] else ids[i][0]),
                    distance=D[i, j]))
    pd.DataFrame(mdmr_rows).to_csv(out_dir / "mdmr_results.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(out_dir / "allpairs_distances.csv", index=False)

    print(f"\nwritten -> {out_dir}")
    print(f"Duration: {datetime.now() - t0}")


if __name__ == "__main__":
    sys.exit(main())
