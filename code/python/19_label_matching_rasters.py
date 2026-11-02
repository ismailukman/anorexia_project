#!/usr/bin/env python3
"""
19_label_matching_rasters.py: verification figure for community-label matching.

For example participants, shows the per-window consensus labels as produced by
GenLouvain (labels are arbitrary in each window) and the same labels after
matching across consecutive windows (mlcd_measures.match_community_affiliation,
port of cocoanCORE match_community_affiliation.m). Regions are ordered by the 13
functional groups. A final page reports how much promiscuity depends on the
numbering of new labels (current numbering vs globally unique labels).

Output: report/abstract/followup/Label_matching_verification.pdf

Run:
    conda activate fmri
    python 19_label_matching_rasters.py
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import atlas_groups                                                                         # noqa: E402
from mlcd_measures import (match_community_affiliation, promiscuity_runN,                   # noqa: E402
                           run_community_count)

spec = importlib.util.spec_from_file_location("dfc", HERE / "05b_dfc_group_statistics.py")
dfc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dfc)

from paths import ROOT
OUT = ROOT / "report/abstract/followup"
EXAMPLES = {"AN": [0, 10], "HC": [0, 10]}        # indices within each group (21 AN, 22 HC)
ZOOM = 60


def ranked(C):
    """Relabel by size so colours are comparable within a panel (NaN kept)."""
    C = np.asarray(C, float)
    ok = ~np.isnan(C)
    labs, counts = np.unique(C[ok], return_counts=True)
    rank = {l: r for r, l in enumerate(labs[np.argsort(-counts)])}
    R = np.full(C.shape, np.nan)
    R[ok] = [rank[v] for v in C[ok]]
    return R


def colormap(n):
    base = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors + plt.get_cmap("tab20c").colors
    return ListedColormap([base[i % len(base)] for i in range(n)])


def match_global(Ci):
    """Same greedy matching, but labels for new communities are never reused."""
    Ci = np.asarray(Ci, float); out = Ci.copy(); top = np.nanmax(Ci[:, 0])
    for t in range(1, Ci.shape[1]):
        before, after = out[:, t - 1], Ci[:, t]
        ub = np.unique(before[~np.isnan(before)]); ua = np.unique(after[~np.isnan(after)])
        ov = np.array([[np.sum((before == b) & (after == a)) for a in ua] for b in ub], float)
        mapped = np.full(ua.size, np.nan); flat = ov.ravel(order="F")
        order = np.argsort(-flat, kind="stable")
        rows, cols = np.unravel_index(order, ov.shape, order="F")
        ru = np.zeros(ub.size, bool); cu = np.zeros(ua.size, bool)
        for r, c, ok in zip(rows, cols, flat[order] > 0):
            if not ok or ru[r] or cu[c]:
                continue
            mapped[c] = ub[r]; ru[r] = cu[c] = True
        for k in np.where(np.isnan(mapped))[0]:
            top += 1; mapped[k] = top
        col = out[:, t].copy()
        for a, nw in zip(ua, mapped):
            col[after == a] = nw
        out[:, t] = col
        top = max(top, np.nanmax(col))
    return out


def cohens_d(a, b):
    sp = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    return (a.mean() - b.mean()) / sp


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gv, names, _ = atlas_groups.combined216_groups()
    order = np.argsort(gv, kind="stable")
    edges = np.flatnonzero(np.diff(gv[order])) + 0.5
    dfc.select_cohort("21v22")
    cfg = dfc.ATLAS_CFG["combined216"]
    subs = {"AN": dfc.load_group(cfg, "anorexia", 216)[0], "HC": dfc.load_group(cfg, "control", 216)[0]}

    with PdfPages(OUT / "Label_matching_verification.pdf") as pdf:
        for grp, idxs in EXAMPLES.items():
            for i in idxs:
                raw = np.asarray(subs[grp][i], float)[order]
                mat = match_community_affiliation(subs[grp][i])[order]
                fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), gridspec_kw=dict(width_ratios=[3, 1]))
                for r, (lab, C) in enumerate((("Before matching (labels as output per window)", raw),
                                              ("After matching across consecutive windows", mat))):
                    R = ranked(C)
                    cm = colormap(int(np.nanmax(R)) + 1)
                    for c, (sl, ttl) in enumerate(((slice(None), lab), (slice(0, ZOOM), f"first {ZOOM} windows"))):
                        ax = axes[r, c]
                        ax.imshow(R[:, sl], aspect="auto", cmap=cm, interpolation="nearest")
                        for e in edges:
                            ax.axhline(e, color="white", lw=0.6)
                        ax.set_title(ttl, fontsize=10, loc="left")
                        ax.set_xlabel("window", fontsize=9); ax.set_ylabel("region (by functional group)", fontsize=9)
                        ax.tick_params(labelsize=8)
                fig.suptitle(f"{grp} participant {i + 1}: community labels before and after matching "
                             f"({raw.shape[1]} windows, 216 regions; white lines separate the 13 groups)",
                             fontsize=11, fontweight="bold")
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig); plt.close(fig)

        # summary page: dependence of promiscuity on new-label numbering
        all_subs = subs["AN"] + subs["HC"]
        n1 = len(subs["AN"])
        res = {}
        for name, fn in (("current numbering", match_community_affiliation), ("globally unique labels", match_global)):
            Ms = [fn(C) for C in all_subs]
            K = run_community_count(Ms)
            p = np.array([np.nanmean(promiscuity_runN(M, K)) for M in Ms])
            res[name] = (K, cohens_d(p[:n1], p[n1:]))
        fig = plt.figure(figsize=(11, 4)); fig.patch.set_facecolor("white")
        lines = ["Check: does the numbering of new communities affect the results?", "",
                 "Matching pairs communities in consecutive windows by largest overlap; a community with no overlap",
                 "receives a new label. Allegiance, recruitment and integration use only co-membership within a window",
                 "and flexibility compares only adjacent windows, so they do not depend on label numbers.",
                 "Promiscuity counts distinct labels per region, so it was recomputed with labels that are never reused:", ""]
        for name, (K, d) in res.items():
            lines.append(f"    {name:24s}  K = {K:3d} communities   whole-brain promiscuity AN vs HC  d = {d:+.3f}")
        fig.text(0.04, 0.92, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
        pdf.savefig(fig); plt.close(fig)
    print("->", OUT / "Label_matching_verification.pdf", {k: (v[0], round(v[1], 3)) for k, v in res.items()})


if __name__ == "__main__":
    main()
