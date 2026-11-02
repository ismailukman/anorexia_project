#!/usr/bin/env python3
"""
01e_subcortical_network_assignment.py: assign each Tian S1 region to a Yeo-7
network.

No single published source labels every Tian S1 region, so each structure is
assigned from the best available large-sample evidence, by one stated rule:

    striatum (CAU, PUT, NAc)   majority network of the region's voxels in the
                               Choi et al. (2012) winner-take-all map of the
                               striatum onto the Yeo et al. (2011) 7 networks,
                               1,000 participants (J Neurophysiol 108:2242)
    thalamus (aTHA, pTHA)      majority network in the Buckner-lab 1,000-subject
                               Yeo-7 winner-take-all map of the thalamus
                               (ThalamusParcellation release, as used by Hong
                               et al. 2023, J Psychiatry Neurosci 48:E50)
    hippocampus                default mode: hippocampal formation belongs to
                               the medial temporal subsystem of the default
                               network (Andrews-Hanna et al. 2010, Neuron
                               65:550); winner-take-all assigns the whole
                               hippocampus to default mode (Seitzman et al.
                               2020, NeuroImage 206:116290)
    amygdala                   limbic, inferred from its strong coupling with
                               the orbitofrontal and temporal-pole cortex that
                               forms the Yeo limbic network (Girn et al. 2024,
                               Netw Neurosci 8:860); default mode is the
                               alternative (Seitzman et al. 2020)
    globus pallidus            somatomotor (Seitzman et al. 2020: putamen and
                               pallidum join the somatomotor network); no Yeo-7
                               map covers the pallidum

Left and right voxels are pooled, so each structure takes one label in both
hemispheres. `network_alt` holds the runner-up (second network in the map, or
the literature alternative) for a sensitivity analysis. As a consistency check,
a winner-take-all assignment from this study's own controls is reported beside
it: mean Fisher-z correlation of each region with each network's parcels,
excluding parcels within EXCLUDE_MM, with bootstrap stability.

Output: data/atlas/combined_216/subcortical_network_assignment.csv, read by
atlas_groups.py.

Run after 01c_combined_fc.py:
    conda activate fmri
    python code/python/01e_subcortical_network_assignment.py
"""
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from paths import ROOT
MAPS = ROOT / "data/atlas/subcortical_network_maps"
CHOI = MAPS / "Choi2012_7Networks_MNI152_FreeSurferConformed1mm_LooseMask.nii.gz"
THAL = MAPS / "1000subjects_TightThalamus_clusters007_ref.nii.gz"
TIAN = ROOT / "data/atlas/tian_s1/Tian_Subcortex_S1_3T_2009cAsym.nii.gz"
TS = ROOT / "data/analysis/timeseries_subjs_masked"
LABELS = ROOT / "data/atlas/combined_216/combined_216_labels.txt"
COORDS = ROOT / "data/analysis/coords216.npy"
OUT = ROOT / "data/atlas/combined_216/subcortical_network_assignment.csv"

YEO = [("Vis", "VN"), ("SomMot", "SMN"), ("DorsAttn", "DAN"), ("SalVentAttn", "VAN"),
       ("Limbic", "LN"), ("Cont", "FPN"), ("Default", "DMN")]
NETS = [abbr for _, abbr in YEO]
MAP_SOURCE = {"CAU": "Choi2012", "PUT": "Choi2012", "NAc": "Choi2012",
              "aTHA": "Buckner-lab thalamus", "pTHA": "Buckner-lab thalamus"}
LITERATURE = {"HIP": ("DMN", "DMN", "Andrews-Hanna 2010; Seitzman 2020"),
              "AMY": ("LN", "DMN", "Girn 2024 (LN); Seitzman 2020 (DMN)"),
              "GP": ("SMN", "SMN", "Seitzman 2020")}
EXCLUDE_MM = 20.0
N_BOOT = 5000
SEED = 42
AN_EXCLUDE = {1, 2, 8}


def map_shares(tian, t_data, value, net_img):
    """Share of a Tian region's voxels falling in each network of a map."""
    ijk = np.argwhere(t_data == value)
    xyz = nib.affines.apply_affine(tian.affine, ijk)
    inv = np.linalg.inv(net_img.affine)
    vox = np.rint(nib.affines.apply_affine(inv, xyz)).astype(int)
    d = np.asarray(net_img.dataobj).squeeze().astype(int)
    inside = np.all((vox >= 0) & (vox < np.array(d.shape[:3])), axis=1)
    lab = d[tuple(vox[inside].T)]
    lab = lab[lab > 0]
    counts = np.array([(lab == k).sum() for k in range(1, 8)])
    return counts, lab.size / ijk.shape[0]


def subject_z(group, si):
    ctx = np.load(TS / f"ts_{group}_subj{si:02d}.npy").astype(float)
    sub = np.load(TS / f"ts_sub_{group}_subj{si:02d}.npy").astype(float)
    ctx = (ctx - ctx.mean(0)) / ctx.std(0)
    sub = (sub - sub.mean(0)) / sub.std(0)
    return np.arctanh(np.clip(sub.T @ ctx / ctx.shape[0], -0.999999, 0.999999))


def own_data_check(labels):
    """Winner-take-all from this study's controls, with bootstrap stability."""
    net_of = np.array([next(k for k, (key, _) in enumerate(YEO) if key in lbl)
                       for lbl in labels[:200]])
    xyz = np.load(COORDS)
    keep = np.linalg.norm(xyz[200:, None] - xyz[None, :200], axis=2) > EXCLUDE_MM
    hc = np.stack([subject_z("hc_patients", s) for s in range(1, 23)])

    def means(z):
        return np.array([[z[i, (net_of == k) & keep[i]].mean() for k in range(7)]
                         for i in range(16)])
    win = means(hc.mean(0)).argmax(1)
    rng = np.random.default_rng(SEED)
    agree = np.zeros(16)
    for _ in range(N_BOOT):
        agree += means(hc[rng.integers(0, 22, 22)].mean(0)).argmax(1) == win
    return [NETS[w] for w in win], agree / N_BOOT


def main():
    labels = [l.split()[1] for l in LABELS.read_text().splitlines()
              if l.strip() and not l.startswith("#")]
    tian = nib.load(str(TIAN))
    t_data = np.asarray(tian.dataobj).astype(int)
    maps = {"Choi2012": nib.load(str(CHOI)), "Buckner-lab thalamus": nib.load(str(THAL))}

    # pooled left + right shares per structure
    pooled = {}
    for struct, src in MAP_SOURCE.items():
        counts, cover = np.zeros(7), []
        for v in range(1, 17):
            if labels[199 + v].split("-")[0] == struct:
                c, cov = map_shares(tian, t_data, v, maps[src])
                counts += c
                cover.append(cov)
        pooled[struct] = (counts / counts.sum(), float(np.mean(cover)))

    own_net, own_agree = own_data_check(labels)
    rows = []
    for i in range(16):
        lbl = labels[200 + i]
        struct = lbl.split("-")[0]
        if struct in MAP_SOURCE:
            share, cover = pooled[struct]
            order = np.argsort(share)[::-1]
            net, alt = NETS[order[0]], NETS[order[1]]
            basis = (f"{MAP_SOURCE[struct]} map: "
                     + ", ".join(f"{NETS[k]} {100 * share[k]:.0f}%" for k in order[:3])
                     + f" (map covers {100 * cover:.0f}% of voxels)")
        else:
            net, alt, src = LITERATURE[struct]
            basis = f"literature: {src}"
        rows.append(dict(index=201 + i, region=lbl, network=net, network_alt=alt,
                         basis=basis, own_data_network=own_net[i],
                         own_data_bootstrap_agreement=round(own_agree[i], 3)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 90)
    print(df.drop(columns="index").to_string(index=False))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
