#!/usr/bin/env python3
"""
18_censoring_sensitivity.py: how much do zero-filled (censored) volumes affect the results?

AFNI's 3dTproject (-cenmode ZERO) sets censored volumes to zero in the residual
time series, and the sliding-window correlations use the full 700-volume series,
so windows that span a censored volume include those zeros. This script compares,
for every participant, the measures computed from all 663 windows with the same
measures computed only from windows that contain no censored volume.

The community labels are those of the main run (communities were detected with
all windows), so this is an approximate check; the full check would recompute
the window correlations without the censored volumes and rerun MLCD.

Outputs (output/results/censoring_sensitivity/):
    censored_volumes.json     censored volume indices per participant (from the errts files)
    per_participant.csv       group, censored volumes, windows kept, RC and pTHA-rh IC (all / clean)
    group_summary.csv         Cohen's d and permutation p for each measure and window set

Run:
    conda activate fmri
    python 18_censoring_sensitivity.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import atlas_groups                                                   # noqa: E402
from mlcd_measures import allegiance, match_community_affiliation     # noqa: E402

spec = importlib.util.spec_from_file_location("dfc", HERE / "05b_dfc_group_statistics.py")
dfc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dfc)

from paths import ROOT
OUT = ROOT / "output/results/censoring_sensitivity"
WIN = 38
N_PERM = 20000
SEED = 42
QC_EXCLUDED = "an_k032"


def censored_indices():
    """Indices of all-zero volumes in each participant's AFNI residual file."""
    path = OUT / "censored_volumes.json"
    if path.exists():
        return json.loads(path.read_text())
    out = {}
    for grp in ("an", "hc"):
        for f in sorted((ROOT / "data" / f"{grp}_patients").glob("errts*.nii.gz")):
            d = np.asarray(nib.load(str(f)).dataobj)
            brain = np.abs(d).sum(3) > 0
            out[f.name.split(".")[1]] = np.flatnonzero(np.abs(d[brain]).sum(0) == 0).tolist()
    path.write_text(json.dumps(out))
    return out


def measures(C, cens, clean_only, gv, pth):
    C = match_community_affiliation(C)
    if clean_only and cens:
        bad = np.zeros(C.shape[1], bool)
        for t in cens:                                   # windows that contain volume t
            bad[max(0, t - WIN + 1):min(C.shape[1], t + 1)] = True
        C = C[:, ~bad]
    A = allegiance(C)
    rc = np.array([np.nanmean(A[i, gv == gv[i]]) for i in range(len(gv))])
    ic = np.array([np.nanmean(A[i, gv != gv[i]]) for i in range(len(gv))])
    return np.nanmean(rc), ic[pth], C.shape[1]


def d_and_p(a, b):
    x = np.concatenate([a, b]); n1 = a.size

    def d(v):
        u, w = v[:n1], v[n1:]
        sp = np.sqrt(((u.size - 1) * u.var(ddof=1) + (w.size - 1) * w.var(ddof=1)) / (v.size - 2))
        return (u.mean() - w.mean()) / sp
    obs = d(x)
    rng = np.random.default_rng(SEED)
    null = np.array([d(rng.permutation(x)) for _ in range(N_PERM)])
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cens = censored_indices()
    an_ids = [k for k in sorted(cens) if k.startswith("an_") and k != QC_EXCLUDED]
    hc_ids = sorted(k for k in cens if k.startswith("hc_"))
    gv, _, labels = atlas_groups.combined216_groups()
    pth = labels.index("pTHA-rh")
    dfc.select_cohort("21v22")
    cfg = dfc.ATLAS_CFG["combined216"]
    subs_an, _, _ = dfc.load_group(cfg, "anorexia", 216)
    subs_hc, _, _ = dfc.load_group(cfg, "control", 216)
    assert len(subs_an) == len(an_ids) == 21 and len(subs_hc) == len(hc_ids) == 22

    rows = []
    for grp, ids, subs in (("AN", an_ids, subs_an), ("HC", hc_ids, subs_hc)):
        for pid, C in zip(ids, subs):
            rc_a, ic_a, n_a = measures(C, cens[pid], False, gv, pth)
            rc_c, ic_c, n_c = measures(C, cens[pid], True, gv, pth)
            rows.append(dict(participant=pid, group=grp, censored_volumes=len(cens[pid]),
                             windows_all=n_a, windows_clean=n_c,
                             RC_all=rc_a, RC_clean=rc_c, pTHA_IC_all=ic_a, pTHA_IC_clean=ic_c))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_participant.csv", index=False)

    summ = []
    for meas in ("RC", "pTHA_IC"):
        for ws in ("all", "clean"):
            col = f"{meas}_{ws}"
            d, p = d_and_p(df.loc[df.group == "AN", col].to_numpy(), df.loc[df.group == "HC", col].to_numpy())
            summ.append(dict(measure=meas, windows=ws, cohens_d=d, permutation_p=p))
    pd.DataFrame(summ).to_csv(OUT / "group_summary.csv", index=False)
    print(pd.DataFrame(summ).round(4).to_string(index=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
