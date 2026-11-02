#!/usr/bin/env python3
"""
17_apriori_region_test.py: hypothesis tests restricted to anorexia-relevant regions.

The 38 regions and their seven functional systems are defined from published
anorexia nervosa work (Gaudio et al. 2016; Kaye et al. 2009) and anatomy. A
cortical parcel is included only if at least 40% of its voxels lie in the
Harvard-Oxford structure named by the literature; every selected parcel's
overlap is written to apriori_regions.csv, and the script stops if one falls
below 40%.

Community detection stays on all 216 regions; only the testing is restricted.
The per-participant regional values are those computed by 05b
(dfc_group_stats.npz): recruitment and integration from the 13-group vector,
flexibility and promiscuity from labels aligned across layers.

The participant is the unit of observation. Each participant contributes one
value per region and measure, and group labels are permuted across
participants (05b.perm_pvalues: pooled-variance t, two-sided, add-one p,
20,000 permutations). Windows are never permuted.

H1, primary. Regional recruitment and integration differ between AN and HC
(AN expected lower in recruitment; integration non-directional).
    - each of the 38 regions tested separately; BH FDR over the 38 regions
      within a measure (primary family), also over both measures (76 tests)
    - summary tests: the mean over the 38 regions and over all 216, and
      PERMANOVA with PERMDISP on the 38- and 216-region profiles (functions
      from 14_individual_vs_group.py)
    - secondary summary: the mean over each system's regions, BH over 7
H2, secondary. Flexibility and promiscuity, to separate active
reconfiguration (flexibility raised) from a weaker home network (flexibility
unchanged) as the account of any recruitment difference.
    - whole-brain mean, with Cohen's d and its 95% confidence interval
    - each of the 38 regions, BH over the 38 regions

Every test is run for both candidate cohorts, 21 AN v 22 HC (one AN
participant excluded on quality control) and 19 AN v 22 HC (two further AN
participants removed), each without and with mean framewise displacement (FD) as a
covariate. With FD, univariate tests use Freedman-Lane permutation (Freedman &
Lane 1983; Winkler et al. 2014): the FD-only model is fitted, its residuals are
permuted and added back to its fitted values, and the group t of the full
model (group + FD) is recomputed. d is then the FD-adjusted group coefficient
over the residual SD of the full model. PERMANOVA and PERMDISP, which have no
covariate term, use profiles with FD regressed out.

A motion check on whole-brain recruitment is written to motion_check.csv: the
FD-recruitment correlation within each group, the share of the group gap that
FD accounts for in a joint model, and the group test after dropping the
highest-moving AN participants until mean FD matches HC (within 0.005 mm).

Run:
    conda activate fmri
    python code/python/17_apriori_region_test.py
"""

import importlib.util
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smm
from scipy.spatial.distance import pdist, squareform

from paths import ROOT
STATS_DIR = ROOT / "output/results/dfc_statistics_combined216"
LABELS_TXT = ROOT / "data/atlas/combined_216/combined_216_labels.txt"
SCHAEFER = (ROOT / "data/atlas/schaefer_2018/schaefer_2018/"
            "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz")
MOTION = ROOT / "ml_experiment/data/motion/motion_summary.csv"
OUT = ROOT / "output/results/apriori_regions_combined216"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_dfc = _load("dfc", "05b_dfc_group_statistics.py")
_ivg = _load("ivg", "14_individual_vs_group.py")

N_PERM = int(os.environ.get("N_PERM", 20000))
SEED = 42
ALPHA = 0.05
N_EXTRA = 2   # AN rows 0 and 1, also removed in the 19v22 cohort

# 1-based indices into combined_216_labels.txt, with the Harvard-Oxford
# structure(s) each cortical system must overlap by at least 40%.
SYSTEMS = {
    "Interoception: insula":            ([47, 48, 118, 152, 155], ["Insular Cortex"]),
    "Salience: dorsal ACC":             ([73, 156, 179], ["Cingulate Gyrus, anterior division"]),
    "Reward valuation: OFC, vmPFC":     ([55, 83, 85, 160, 84, 191],
                                         ["Frontal Orbital Cortex", "Subcallosal Cortex",
                                          "Frontal Medial Cortex", "Paracingulate Gyrus"]),
    "Cognitive control: dlPFC":         ([69, 70, 173, 175, 176], ["Middle Frontal Gyrus"]),
    "Self-referential: precuneus, PCC": ([96, 97, 98, 99, 198, 199, 200],
                                         ["Precuneous Cortex", "Cingulate Gyrus, posterior division"]),
    "Striatum: NAc, caudate, putamen":  ([205, 213, 208, 216, 207, 215], None),
    "Thalamus and amygdala":            ([203, 204, 211, 212, 202, 210], None),
}
H1 = {"recruitment": "rec", "integration": "int"}
H2 = {"flexibility": "flex", "promiscuity": "prom"}
COHORTS = [("21v22", 0), ("19v22", N_EXTRA)]
ADJUST = ["none", "FD"]


def anatomy_table(labels):
    """Harvard-Oxford overlap of every selected cortical parcel."""
    from nilearn import datasets, image
    ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm",
                                             symmetric_split=True)
    sch = nib.load(str(SCHAEFER))
    H = np.asarray(image.resample_to_img(ho.maps, sch, interpolation="nearest",
                                         force_resample=True,
                                         copy_header=True).dataobj).astype(int)
    S = np.asarray(sch.dataobj).astype(int)
    names = list(ho.labels)
    rows = []
    for system, (regions, structures) in SYSTEMS.items():
        for r in regions:
            row = {"system": system, "index": r, "label": labels[r - 1]}
            if structures is None:
                row.update(target_overlap_pct=np.nan,
                           largest_HO_structure="Tian S1 subcortex")
            else:
                h = H[S == r]
                frac = sum((h == i).sum() for i, nm in enumerate(names)
                           if any(nm.endswith(s) for s in structures)) / h.size
                u, c = np.unique(h[h > 0], return_counts=True)
                row.update(target_overlap_pct=round(100 * frac, 1),
                           largest_HO_structure=names[u[c.argmax()]])
                if frac < 0.40:
                    raise SystemExit(f"region {r} {labels[r - 1]} overlaps its target "
                                     f"by {100 * frac:.1f}%, below the 40% rule")
            rows.append(row)
    return pd.DataFrame(rows)


def freedman_lane(x, n1, n2, fd, rng, n_perm=N_PERM):
    """Group test with FD as covariate, per column of x (subjects x features).

    Returns adjusted d (AN minus HC, in residual SD units) and two-sided
    add-one p.
    """
    n = n1 + n2
    g = np.r_[np.ones(n1), np.zeros(n2)]                 # 1 = AN
    Xf = np.column_stack([np.ones(n), g, fd])
    Xr = np.column_stack([np.ones(n), fd])
    Pf = np.linalg.pinv(Xf)
    c = np.linalg.inv(Xf.T @ Xf)[1, 1]

    def group_t(y):
        b = Pf @ y
        sse = ((y - Xf @ b) ** 2).sum(0)
        return b[1] / np.sqrt(sse / (n - 3) * c), b[1] / np.sqrt(sse / (n - 3))

    t_obs, d = group_t(x)
    fitted = Xr @ (np.linalg.pinv(Xr) @ x)
    resid = x - fitted
    count = np.zeros(x.shape[1])
    for _ in range(n_perm):
        t_p, _ = group_t(fitted + resid[rng.permutation(n)])
        count += np.abs(t_p) >= np.abs(t_obs)
    return d, (count + 1) / (n_perm + 1)


def residualise(x, fd):
    """Remove the linear effect of mean FD from every column of x."""
    X = np.column_stack([np.ones_like(fd), fd])
    beta, *_ = np.linalg.lstsq(X, x, rcond=None)
    return x - X @ beta + x.mean(0)


def cohort_fd():
    """Mean FD in the row order of dfc_group_stats.npz (21 AN, 22 HC)."""
    m = pd.read_csv(MOTION)
    m = m[~m.subject.isin(["an_k032", "an_k020", "an_k064"])]
    return (m.loc[m.group == "AN", "mean_fd"].to_numpy(),
            m.loc[m.group == "HC", "mean_fd"].to_numpy())


def group_test(x, n1, n2, fd=None):
    """Per-column permutation test. Returns AN mean, HC mean, d, CI, p.

    Without fd: subject-label permutation of 05b. With fd: Freedman-Lane.
    Means are always the raw group means.
    """
    lab = np.r_[np.zeros(n1, int), np.ones(n2, int)]
    rng = np.random.default_rng(SEED)
    if fd is None:
        _, p = _dfc.perm_pvalues(x, lab, rng, n1, n2, N_PERM)
        d = _dfc.cohens_d(x, lab, n1, n2)
    else:
        d, p = freedman_lane(x, n1, n2, fd, rng)
    se = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    return x[:n1].mean(0), x[n1:].mean(0), d, d - 1.96 * se, d + 1.96 * se, p


def bh(p):
    return smm.multipletests(p, method="fdr_bh")[1]


def check_against_05b(z):
    """Reproduce 05b's whole-brain recruitment test before trusting anything."""
    g = pd.read_csv(STATS_DIR / "dfc_global_summary.csv").set_index("measure")
    x = np.r_[z["rec_subj_an"].mean(1), z["rec_subj_hc"].mean(1)][:, None]
    _, _, d, _, _, p = group_test(x, z["rec_subj_an"].shape[0], z["rec_subj_hc"].shape[0])
    ref_d, ref_p = g.loc["Mean_Recruitment", "cohens_d"], g.loc["Mean_Recruitment", "pvalue"]
    assert abs(d[0] - ref_d) < 1e-9, (d[0], ref_d)
    assert abs(p[0] - ref_p) < 0.01, (p[0], ref_p)
    print(f"check: whole-brain recruitment d={d[0]:.3f} p={p[0]:.4f} "
          f"reproduces 05b (d={ref_d:.3f} p={ref_p:.4f})")


def motion_check(z, fd_an, fd_hc):
    """How much of the whole-brain recruitment difference can FD account for?"""
    from scipy import stats
    rows = []
    for cohort, drop in COHORTS:
        ra, rh = z["rec_subj_an"][drop:].mean(1), z["rec_subj_hc"].mean(1)
        fa = fd_an[drop:]
        row = dict(cohort=cohort)
        _, _, d_fd, _, _, p_fd = group_test(np.r_[fa, fd_hc][:, None], fa.size, fd_hc.size)
        row.update(FD_AN=fa.mean(), FD_HC=fd_hc.mean(), FD_d=d_fd[0], FD_p=p_fd[0])
        for g, f, r in [("AN", fa, ra), ("HC", fd_hc, rh)]:
            row[f"r_FD_rec_{g}"], row[f"p_FD_rec_{g}"] = stats.pearsonr(f, r)
        y, fd = np.r_[ra, rh], np.r_[fa, fd_hc]
        X = np.column_stack([np.ones(y.size), np.r_[np.ones(ra.size), np.zeros(rh.size)], fd])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        gap, fd_gap = ra.mean() - rh.mean(), fa.mean() - fd_hc.mean()
        row.update(rec_gap=gap, group_coef_with_FD=b[1],
                   share_of_gap_from_FD=b[2] * fd_gap / gap,
                   r_group_FD=stats.pearsonr(X[:, 1], fd)[0])
        keep = np.argsort(fa)
        while abs(fa[keep].mean() - fd_hc.mean()) > 0.005:
            keep = keep[:-1]                     # drop the highest-moving AN
        _, _, d, lo, hi, p = group_test(np.r_[ra[keep], rh][:, None], keep.size, rh.size)
        row.update(matched_n_AN=keep.size, matched_FD_AN=fa[keep].mean(),
                   matched_d=d[0], matched_d_ci_low=lo[0], matched_d_ci_high=hi[0],
                   matched_p=p[0])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = [l.split()[1] for l in LABELS_TXT.read_text().splitlines()
              if l.strip() and not l.startswith("#")]
    anat = anatomy_table(labels)
    anat.to_csv(OUT / "apriori_regions.csv", index=False)
    print(f"{len(anat)} a priori regions; every cortical parcel meets the 40% rule")

    z = np.load(STATS_DIR / "dfc_group_stats.npz", allow_pickle=True)
    check_against_05b(z)
    fd_an, fd_hc = cohort_fd()
    assert len(fd_an) == z["rec_subj_an"].shape[0] and len(fd_hc) == z["rec_subj_hc"].shape[0]

    mc = motion_check(z, fd_an, fd_hc)
    mc.to_csv(OUT / "motion_check.csv", index=False)
    print(mc.round(4).T.to_string())

    idx = np.array([r - 1 for regions, _ in SYSTEMS.values() for r in regions])
    sys_of = np.array([s for s, (regions, _) in SYSTEMS.items() for _ in regions])
    region_rows, system_rows, summary_rows = [], [], []

    for cohort, drop in COHORTS:
        for adj in ADJUST:
            fd = np.r_[fd_an[drop:], fd_hc] if adj == "FD" else None

            def data(key):
                an, hc = z[f"{key}_subj_an"][drop:], z[f"{key}_subj_hc"]
                return np.vstack([an, hc]).astype(float), an.shape[0], hc.shape[0]

            tag = dict(cohort=cohort, fd_adjust=adj)
            for hyp, measures in [("H1", H1), ("H2", H2)]:
                for mname, key in measures.items():
                    x, n1, n2 = data(key)
                    lab = np.r_[np.zeros(n1, int), np.ones(n2, int)]

                    # each a priori region separately
                    a, h, d, lo, hi, p = group_test(x[:, idx], n1, n2, fd)
                    q = bh(p)
                    for j, r in enumerate(idx):
                        region_rows.append(dict(**tag, hypothesis=hyp, measure=mname,
                                                system=sys_of[j], index=r + 1,
                                                label=labels[r], AN=a[j], HC=h[j],
                                                d=d[j], d_ci_low=lo[j], d_ci_high=hi[j],
                                                p=p[j], p_fdr=q[j]))

                    # whole-brain mean (all 216) and a priori set mean
                    for scope, cols in [("whole brain (216)", slice(None)),
                                        ("a priori set (38)", idx)]:
                        a, h, d, lo, hi, p = group_test(x[:, cols].mean(1)[:, None], n1, n2, fd)
                        summary_rows.append(dict(**tag, hypothesis=hyp, measure=mname,
                                                 test=f"mean, {scope}", AN=a[0], HC=h[0],
                                                 d=d[0], d_ci_low=lo[0], d_ci_high=hi[0],
                                                 p=p[0]))

                    if hyp == "H1":
                        # multivariate profile over the 38 regions
                        for scope, cols in [("216", slice(None)), ("38", idx)]:
                            xp = x[:, cols] if fd is None else residualise(x[:, cols], fd)
                            D = squareform(pdist(xp, "euclidean"))
                            F, p_f = _ivg.permanova(D, lab, np.random.default_rng(SEED), N_PERM)
                            disp, p_d = _ivg.permdisp(D, lab, np.random.default_rng(SEED), N_PERM)
                            summary_rows.append(dict(**tag, hypothesis=hyp, measure=mname,
                                                     test=f"PERMANOVA, {scope}-region profile",
                                                     pseudo_F=F, p=p_f))
                            summary_rows.append(dict(**tag, hypothesis=hyp, measure=mname,
                                                     test=f"PERMDISP, {scope}-region profile",
                                                     dispersion_diff=disp, p=p_d))

                        # per-system mean: secondary, network-level summary
                        xs = np.column_stack([x[:, [r - 1 for r in regions]].mean(1)
                                              for regions, _ in SYSTEMS.values()])
                        a, h, d, lo, hi, p = group_test(xs, n1, n2, fd)
                        q = bh(p)
                        for j, s in enumerate(SYSTEMS):
                            system_rows.append(dict(**tag, hypothesis=hyp, measure=mname,
                                                    system=s, n_regions=len(SYSTEMS[s][0]),
                                                    AN=a[j], HC=h[j], d=d[j],
                                                    d_ci_low=lo[j], d_ci_high=hi[j],
                                                    p=p[j], p_fdr=q[j]))

            # H1 family over both measures (76 tests)
            m = [i for i, r in enumerate(region_rows)
                 if r["cohort"] == cohort and r["fd_adjust"] == adj and r["hypothesis"] == "H1"]
            for i, qq in zip(m, bh(np.array([region_rows[i]["p"] for i in m]))):
                region_rows[i]["p_fdr_H1_both_measures"] = qq

    reg = pd.DataFrame(region_rows)
    sysdf = pd.DataFrame(system_rows)
    summ = pd.DataFrame(summary_rows)
    reg.to_csv(OUT / "region_tests.csv", index=False)
    sysdf.to_csv(OUT / "system_tests.csv", index=False)
    summ.to_csv(OUT / "summary_tests.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    for cohort, _ in COHORTS:
        for adj in ADJUST:
            print(f"\n=================== {cohort} | FD adjustment: {adj} ===================")
            s = summ[(summ.cohort == cohort) & (summ.fd_adjust == adj)]
            print(s.drop(columns=["cohort", "fd_adjust"]).round(4).to_string(index=False))
            r = reg[(reg.cohort == cohort) & (reg.fd_adjust == adj)]
            for (hyp, mname), g in r.groupby(["hypothesis", "measure"], sort=False):
                top = g.sort_values("p").head(5)
                print(f"\n  {hyp} {mname}: {(g.p < ALPHA).sum()}/38 nominal, "
                      f"{(g.p_fdr < ALPHA).sum()}/38 after FDR; smallest p:")
                print(top[["label", "system", "d", "p", "p_fdr"]].round(4)
                      .to_string(index=False))
            sy = sysdf[(sysdf.cohort == cohort) & (sysdf.fd_adjust == adj)]
            print("\n  system means (secondary summary):")
            print(sy[["measure", "system", "d", "p", "p_fdr"]].round(4).to_string(index=False))
    print(f"\nResults -> {OUT}")


if __name__ == "__main__":
    main()
