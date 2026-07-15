#!/usr/bin/env python3
"""
02b_outcome_measures_subcortical.py
Subcortical outcome measures (Tian Scale II, 32 ROIs).
Reads MLCD outputs and computes allegiance, recruitment, integration, flexibility, promiscuity.
Run after: subcortical_mlcd_5subj.m
"""
import logging
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from teneto import communitymeasures

start_time = datetime.now()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

log.info("STAGE 3b — Subcortical Outcome Measures (Tian S2, 32 ROIs)")

ROOT        = Path("/Users/ismaila/Documents/C-Codes/AnorexiaProject")
MLCD_DIR    = ROOT / "data/analysis/mlcd_subjs_subcortical_5subj/subjs_mlcd"
OUT_METRICS = ROOT / "output/results/subject_metrics_subcortical_5subj"
OUT_XLSX    = ROOT / "output/results/xlsx_exports_subcortical_5subj"
OUT_FIG     = ROOT / "output/figures/stage3_subcortical_5subj"
for d in [OUT_METRICS, OUT_XLSX, OUT_FIG]:
    d.mkdir(parents=True, exist_ok=True)

N_REGIONS  = 32
N_SUBJ     = 5
GROUP_TAGS = ["anorexia", "control"]

LABEL_FILE = ROOT / "data/atlas/tian_s2/Tian_Subcortex_S2_3T_label.txt"
tian_labels = LABEL_FILE.read_text().strip().splitlines()

STRUCT_MAP = {"HIP": 1, "AMY": 2, "THA": 3, "NAc": 4, "GP": 5, "PUT": 6, "CAU": 7}
STRUCT_NAMES = {1:"Hippocampus", 2:"Amygdala", 3:"Thalamus",
                4:"Nuc.Accumbens", 5:"Globus Pallidus", 6:"Putamen", 7:"Caudate"}

def _label_to_community(lbl: str) -> int:
    for key, comm in STRUCT_MAP.items():
        if key.lower() in lbl.lower():
            return comm
    return 0

static_communities = np.array([_label_to_community(l) for l in tian_labels], dtype=int)
log.info(f"Tian labels loaded: {len(tian_labels)} ROIs")
log.info(f"Static communities: {dict(zip(*np.unique(static_communities, return_counts=True)))}")

roi_index = pd.Index(tian_labels, name="ROI")


def load_group(tag: str):
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
    subjects = [mat[:, i*wins_per_subj:(i+1)*wins_per_subj] for i in range(N_SUBJ)]
    log.info(f"  [{tag}] shape={mat.shape} | wins/subj={wins_per_subj} | subjects={len(subjects)}")
    return subjects, wins_per_subj

log.info("Loading MLCD outputs …")
subjs_an, W = load_group("anorexia")
subjs_hc, _ = load_group("control")
all_subjects = subjs_an + subjs_hc
N_TOTAL = len(all_subjects)
log.info(f"Total subjects: {N_TOTAL} ({N_SUBJ} AN + {N_SUBJ} HC)")
log.info(f"Computing outcome measures for {N_TOTAL} subjects …")

allegiance_list   = []
recruitment_list  = []
integration_list  = []
flexibility_list  = []
promiscuity_list  = []

for s, C in enumerate(all_subjects):
    grp = "AN" if s < N_SUBJ else "HC"
    A = communitymeasures.allegiance(C)
    R = communitymeasures.recruitment(C, static_communities)
    I = communitymeasures.integration(C, static_communities)
    F = communitymeasures.flexibility(C)
    P = communitymeasures.promiscuity(C)
    allegiance_list.append(A)
    recruitment_list.append(R)
    integration_list.append(I)
    flexibility_list.append(F)
    promiscuity_list.append(P)
    log.info(f"  Subject {s+1:02d}/{N_TOTAL} [{grp}] — "
             f"A{A.shape} | R mean={R.mean():.3f} | I mean={I.mean():.3f} | "
             f"F mean={F.mean():.3f} | P mean={P.mean():.3f}")

log.info("Saving per-subject Excel …")
for s in range(N_TOTAL):
    grp = "AN" if s < N_SUBJ else "HC"
    subj_in_grp = (s % N_SUBJ) + 1
    out_path = OUT_METRICS / f"subcortical_{grp}_subj{subj_in_grp:02d}_metrics.xlsx"
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as w:
        pd.DataFrame({"Recruitment": recruitment_list[s]},  index=roi_index).to_excel(w, sheet_name="Recruitment")
        pd.DataFrame({"Integration": integration_list[s]},  index=roi_index).to_excel(w, sheet_name="Integration")
        pd.DataFrame({"Flexibility": flexibility_list[s]},  index=roi_index).to_excel(w, sheet_name="Flexibility")
        pd.DataFrame({"Promiscuity": promiscuity_list[s]},  index=roi_index).to_excel(w, sheet_name="Promiscuity")
        pd.DataFrame(allegiance_list[s], index=roi_index, columns=roi_index).to_excel(w, sheet_name="Allegiance")
    log.info(f"  Saved {out_path.name}")

for tag, idxs in [("anorexia", range(N_SUBJ)), ("control", range(N_SUBJ, N_TOTAL))]:
    rec_mean = np.mean([recruitment_list[s] for s in idxs], axis=0)
    int_mean = np.mean([integration_list[s] for s in idxs], axis=0)
    flex_mean= np.mean([flexibility_list[s] for s in idxs], axis=0)
    prom_mean= np.mean([promiscuity_list[s] for s in idxs], axis=0)
    df = pd.DataFrame({
        "ROI": tian_labels,
        "Structure": [STRUCT_NAMES.get(static_communities[i], "?") for i in range(N_REGIONS)],
        "Recruitment_mean": rec_mean,
        "Integration_mean": int_mean,
        "Flexibility_mean": flex_mean,
        "Promiscuity_mean": prom_mean,
    })
    df.to_excel(OUT_XLSX / f"subcortical_{tag}_group_means.xlsx", index=False)
    log.info(f"  Saved subcortical_{tag}_group_means.xlsx")

log.info("Saving allegiance PNG figures …")
struct_ids  = np.unique(static_communities[static_communities > 0])
struct_cols = plt.cm.tab10(np.linspace(0, 0.7, len(struct_ids)))

for s in range(N_TOTAL):
    grp = "AN" if s < N_SUBJ else "HC"
    subj_in_grp = (s % N_SUBJ) + 1
    A = allegiance_list[s]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(A, cmap="jet", vmin=0, vmax=1, aspect="auto")
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("Allegiance", fontsize=11)
    cb.ax.tick_params(labelsize=9)

    boundaries = []
    for sid in struct_ids:
        idx = np.where(static_communities == sid)[0]
        if len(idx): boundaries.append(idx[-1] + 0.5)
    for b in boundaries:
        ax.axhline(b, color="white", lw=1.0)
        ax.axvline(b, color="white", lw=1.0)

    bar_h = N_REGIONS / 7
    cur = 0
    for sid, col in zip(struct_ids, struct_cols):
        n = np.sum(static_communities == sid)
        ax.add_patch(plt.Rectangle(
            (cur - 0.5, N_REGIONS - 0.5), n, bar_h,
            facecolor=col, clip_on=False, linewidth=0.8, edgecolor="k"))
        ax.text(cur + n/2 - 0.5, N_REGIONS - 0.5 + bar_h / 2,
                STRUCT_NAMES[sid][:3], ha="center", va="center",
                fontsize=9, fontweight="bold", clip_on=False)
        cur += n

    title_str = (f"{grp} | Subject {subj_in_grp} - "
                 f"Subcortical Allegiance (Tian S2)")
    ax.set_title(title_str, fontsize=12, fontweight="bold", pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fname = f"allegiance_subcortical_{grp}_subj{subj_in_grp:02d}.png"
    fig.savefig(OUT_FIG / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  {fname}")

log.info(f"STAGE 3b COMPLETE — duration: {datetime.now() - start_time}")
