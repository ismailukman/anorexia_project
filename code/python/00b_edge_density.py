#!/usr/bin/env python3
"""Compare 5% vs 30% edge density thresholds on combined 216-region FC (Schaefer-200 + Tian S1)."""
from pathlib import Path
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

ROOT    = Path("/Users/ismaila/Documents/C-Codes/AnorexiaProject")
DATA    = ROOT / "data/analysis/combined_subjs"
LABELS  = ROOT / "data/atlas/combined_216/combined_216_labels.txt"
OUT     = ROOT / "output/figures/stage1_fc/edge_density"
OUT.mkdir(parents=True, exist_ok=True)

N_CORTICAL = 200
N_SUBCORT  = 16
N_REGIONS  = 216
DENSITIES  = [0.05, 0.30]
DENSITY_LABELS = ["5%", "30%"]

all_labels = []
for line in LABELS.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    parts = line.split()
    if len(parts) >= 2:
        all_labels.append(parts[1])

subcort_labels = all_labels[N_CORTICAL:]   # 16 subcortical

YEO7_COLORS = ["#A251AC", "#789AC1", "#409832", "#E165FE",
               "#F6FDC9", "#EFB944", "#D9717D"]
SUBCORT_YEO = {
    "HIP": 7, "AMY": 5, "pTHA": 2, "aTHA": 5,
    "NAc": 5, "GP": 2, "PUT": 2, "CAU": 6,
}

def region_color(i):
    if i < N_CORTICAL:
        # Yeo-7 from label name
        lbl = all_labels[i]
        for net, code in [("Vis", 1), ("SomMot", 2), ("DorsAttn", 3),
                          ("SalVentAttn", 4), ("Limbic", 5),
                          ("Cont", 6), ("Default", 7)]:
            if net in lbl:
                return YEO7_COLORS[code - 1]
        return "#888888"
    else:
        struct = all_labels[i].split("-")[0]
        return YEO7_COLORS[SUBCORT_YEO.get(struct, 1) - 1]

ROI_COLORS = [region_color(i) for i in range(N_REGIONS)]


def threshold_matrix(fc_2d, density):
    """Apply top-k% edge density threshold to a symmetric FC matrix."""
    n = fc_2d.shape[0]
    # Upper triangle only (excluding diagonal)
    triu_idx = np.triu_indices(n, k=1)
    vals = fc_2d[triu_idx]
    k = int(np.ceil(density * len(vals)))
    thresh = np.sort(vals)[-k]
    binary = (fc_2d >= thresh).astype(float)
    np.fill_diagonal(binary, 0)
    return binary, thresh


def degree_per_roi(binary):
    return binary.sum(axis=1)


# ── Load one representative subject (AN subj01, HC subj01) ────────────────────
def zmean(r_stack, axis=0):
    """Average correlation matrices in Fisher z-space, return r."""
    z = np.arctanh(np.clip(r_stack, -0.9999, 0.9999))
    return np.tanh(z.mean(axis=axis))


def load_static_fc(grp, subj_id):
    """Load FC windows and average in Fisher z-space."""
    pat = str(DATA / f"subj_fc_combined_{grp}_patients_subj{subj_id:02d}.mat")
    files = glob.glob(pat)
    if not files:
        return None
    d = sio.loadmat(files[0])
    key = [k for k in d if not k.startswith("_")][0]
    fc_wins = np.array(d[key])          # (W, 216, 216)
    return zmean(fc_wins, axis=0)       # z-space average → back to r


YEO7_NAMES  = ["VN", "SMN", "DAN", "VAN", "LN", "FPN", "DMN"]
YEO7_KEYS   = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
SUBCORT_NET_NAMES = ["~SMN", "~LN", "~FPN", "~DMN"]
SUBCORT_NET_STRUCTS = {
    "~SMN": ["PUT", "GP", "pTHA"],
    "~LN":  ["AMY", "aTHA", "NAc"],
    "~FPN": ["CAU"],
    "~DMN": ["HIP"],
}

def cortical_net_degrees(deg_cort):
    """Mean degree per Yeo-7 network across cortical ROIs."""
    net_means = []
    for key in YEO7_KEYS:
        idx = [i for i, lbl in enumerate(all_labels[:N_CORTICAL]) if key in lbl]
        net_means.append(np.mean(deg_cort[idx]) if idx else 0.0)
    return net_means

def subcort_net_degrees(deg_subc):
    """Mean degree per functional-overlap group across subcortical ROIs."""
    net_means = []
    for grp_name in SUBCORT_NET_NAMES:
        structs = SUBCORT_NET_STRUCTS[grp_name]
        idx = [i for i, lbl in enumerate(subcort_labels)
               if any(s in lbl for s in structs)]
        net_means.append(np.mean(deg_subc[idx]) if idx else 0.0)
    return net_means


def plot_density_comparison(fc_mean, title_prefix, out_path):
    fig = plt.figure(figsize=(22, 15))
    outer = gridspec.GridSpec(3, 3, figure=fig,
                              hspace=0.48, wspace=0.35,
                              left=0.06, right=0.97,
                              top=0.92, bottom=0.06)

    cmap_mat = plt.cm.RdBu_r.copy()
    cmap_mat.set_bad("lightgray")

    # Pre-compute both thresholds
    thresh_data = {}
    for density, dlbl in zip(DENSITIES, DENSITY_LABELS):
        binary, thresh = threshold_matrix(fc_mean, density)
        deg = degree_per_roi(binary)
        thresh_data[dlbl] = dict(binary=binary, thresh=thresh, deg=deg)

        for col, (density, dlbl) in enumerate(zip(DENSITIES, DENSITY_LABELS)):
        td = thresh_data[dlbl]
        binary, thresh, deg = td["binary"], td["thresh"], td["deg"]
        n_edges = int(binary.sum()) // 2

        ax_mat = fig.add_subplot(outer[0, col])
        mat_masked = np.ma.masked_where(binary == 0, fc_mean)
        im = ax_mat.imshow(mat_masked, cmap=cmap_mat, vmin=-1, vmax=1,
                           interpolation="nearest", aspect="auto")
        ax_mat.axvline(x=N_CORTICAL - 0.5, color="cyan", lw=1.5, linestyle="--")
        ax_mat.axhline(y=N_CORTICAL - 0.5, color="cyan", lw=1.5, linestyle="--")
        # Quadrant labels
        ax_mat.text(N_CORTICAL / 2, N_CORTICAL / 2, "Cortical\n(200×200)",
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold", alpha=0.7)
        ax_mat.text(N_CORTICAL + N_SUBCORT / 2, N_CORTICAL + N_SUBCORT / 2,
                    "Sub\ncort", ha="center", va="center", fontsize=6,
                    color="white", fontweight="bold", alpha=0.7)
        ax_mat.set_title(f"Top {dlbl}  ({n_edges} edges,  thresh={thresh:.3f})",
                         fontsize=10, fontweight="bold")
        ax_mat.set_xlabel("ROI index  (0–199 cortical | 200–215 subcortical)",
                          fontsize=7)
        ax_mat.set_ylabel("ROI index", fontsize=7)
        ax_mat.tick_params(labelsize=6)
        cb = plt.colorbar(im, ax=ax_mat, shrink=0.85, pad=0.02)
        cb.set_label("Pearson r", fontsize=7)
        cb.ax.tick_params(labelsize=6)

        ax_blk = fig.add_subplot(outer[0, 2])
    block_labels = ["Cortical\n(200×200)", "Cross\n(200×16)", "Subcortical\n(16×16)"]
    colors_blk   = ["#789AC1", "#888888", "#D9717D"]
    possible_blk = [N_CORTICAL * (N_CORTICAL - 1) // 2,
                    N_CORTICAL * N_SUBCORT,
                    N_SUBCORT  * (N_SUBCORT  - 1) // 2]
    x = np.arange(len(block_labels))
    w = 0.35
    for ci, (density, dlbl) in enumerate(zip(DENSITIES, DENSITY_LABELS)):
        td = thresh_data[dlbl]
        binary = td["binary"]
        surv_cort = int(binary[:N_CORTICAL, :N_CORTICAL].sum()) // 2
        surv_cross = int(binary[:N_CORTICAL, N_CORTICAL:].sum())
        surv_subc  = int(binary[N_CORTICAL:, N_CORTICAL:].sum()) // 2
        surv_vals  = [surv_cort, surv_cross, surv_subc]
        bars_ = ax_blk.bar(x + ci * w, surv_vals, width=w,
                           label=f"Top {dlbl}",
                           color=["#2196F3", "#FF5722"][ci], alpha=0.85)
        for b, sv, pv in zip(bars_, surv_vals, possible_blk):
            ax_blk.text(b.get_x() + b.get_width() / 2,
                        b.get_height() + max(possible_blk) * 0.01,
                        f"{sv}\n({100*sv/pv:.0f}%)",
                        ha="center", va="bottom", fontsize=6)
    ax_blk.set_xticks(x + w / 2)
    ax_blk.set_xticklabels(block_labels, fontsize=8)
    ax_blk.set_ylabel("Surviving edges", fontsize=8)
    ax_blk.set_title("Surviving edges by block\n(5% vs 30%)", fontsize=10,
                     fontweight="bold")
    ax_blk.legend(fontsize=8)
    ax_blk.tick_params(labelsize=7)

        for col, (density, dlbl) in enumerate(zip(DENSITIES, DENSITY_LABELS)):
        deg = thresh_data[dlbl]["deg"]
        ax_deg = fig.add_subplot(outer[1, col])
        ax_deg.bar(range(N_REGIONS), deg, color=ROI_COLORS,
                   width=1.0, edgecolor="none")
        ax_deg.axvline(x=N_CORTICAL - 0.5, color="black", lw=1.5, linestyle="--")
        # Region labels inside axes using transAxes coordinates
        ax_deg.text(0.46, 0.97, "Cortical →", ha="right", va="top",
                    fontsize=7, color="#333333", transform=ax_deg.transAxes)
        ax_deg.text(0.48, 0.97, "← Subcortical", ha="left", va="top",
                    fontsize=7, color="#333333", transform=ax_deg.transAxes)
        ax_deg.set_xlim(-0.5, N_REGIONS - 0.5)
        ax_deg.set_xlabel("ROI index  (0–199 cortical | 200–215 subcortical)",
                          fontsize=7)
        ax_deg.set_ylabel("Degree (surviving edges)", fontsize=8)
        ax_deg.set_title(f"Degree per ROI — top {dlbl}", fontsize=10,
                         fontweight="bold")
        ax_deg.tick_params(labelsize=7)
        # Annotate zero-degree subcortical ROIs
        subcort_deg = deg[N_CORTICAL:]
        zero_rois = [subcort_labels[i] for i, d_ in enumerate(subcort_deg) if d_ == 0]
        for i, (lbl, d_) in enumerate(zip(subcort_labels, subcort_deg)):
            if d_ == 0:
                ax_deg.annotate(lbl, xy=(N_CORTICAL + i, 0.5),
                                fontsize=6, color="red", ha="center", rotation=90)
        if zero_rois:
            ax_deg.text(0.98, 0.97,
                        "Zero-degree subcortical:\n" + ", ".join(zero_rois),
                        transform=ax_deg.transAxes, fontsize=6, color="red",
                        va="top", ha="right",
                        bbox=dict(facecolor="white", edgecolor="red",
                                  alpha=0.8, boxstyle="round,pad=0.3"))

        ax_cnet = fig.add_subplot(outer[1, 2])
    x = np.arange(len(YEO7_NAMES))
    w = 0.35
    for ci, (density, dlbl) in enumerate(zip(DENSITIES, DENSITY_LABELS)):
        deg = thresh_data[dlbl]["deg"]
        net_deg = cortical_net_degrees(deg[:N_CORTICAL])
        ax_cnet.bar(x + ci * w, net_deg, width=w, label=f"Top {dlbl}",
                    color=["#2196F3", "#FF5722"][ci], alpha=0.85)
    ax_cnet.set_xticks(x + w / 2)
    ax_cnet.set_xticklabels(YEO7_NAMES, fontsize=8)
    ax_cnet.set_ylabel("Mean degree", fontsize=8)
    ax_cnet.set_title("Cortical mean degree\nby Yeo-7 network", fontsize=10,
                      fontweight="bold")
    ax_cnet.legend(fontsize=8)
    ax_cnet.tick_params(labelsize=7)

        ax_sc = fig.add_subplot(outer[2, 0])
    x = np.arange(N_SUBCORT)
    w = 0.35
    for ci, (density, dlbl) in enumerate(zip(DENSITIES, DENSITY_LABELS)):
        deg = thresh_data[dlbl]["deg"][N_CORTICAL:]
        ax_sc.bar(x + ci * w, deg, width=w, label=f"Top {dlbl}",
                  color=["#2196F3", "#FF5722"][ci], alpha=0.85)
    ax_sc.set_xticks(x + w / 2)
    ax_sc.set_xticklabels(subcort_labels, rotation=45, ha="right", fontsize=7)
    ax_sc.set_ylabel("Degree (surviving edges)", fontsize=8)
    ax_sc.set_title("Subcortical ROI degree\n(5% vs 30%)", fontsize=10,
                    fontweight="bold")
    ax_sc.legend(fontsize=8)
    ax_sc.tick_params(labelsize=7)

    ax_snet = fig.add_subplot(outer[2, 1])
    x = np.arange(len(SUBCORT_NET_NAMES))
    for ci, (density, dlbl) in enumerate(zip(DENSITIES, DENSITY_LABELS)):
        deg = thresh_data[dlbl]["deg"][N_CORTICAL:]
        net_deg = subcort_net_degrees(deg)
        ax_snet.bar(x + ci * w, net_deg, width=w, label=f"Top {dlbl}",
                    color=["#2196F3", "#FF5722"][ci], alpha=0.85)
    ax_snet.set_xticks(x + w / 2)
    ax_snet.set_xticklabels(SUBCORT_NET_NAMES, fontsize=9)
    ax_snet.set_ylabel("Mean degree", fontsize=8)
    ax_snet.set_title("Subcortical mean degree\nby functional group", fontsize=10,
                      fontweight="bold")
    ax_snet.legend(fontsize=8)
    ax_snet.tick_params(labelsize=7)

        ax_subc = fig.add_subplot(outer[2, 2])
    possible_sc = N_SUBCORT * (N_SUBCORT - 1) // 2
    cats, surv_list, lost_list = [], [], []
    for density, dlbl in zip(DENSITIES, DENSITY_LABELS):
        binary = thresh_data[dlbl]["binary"]
        sc_block = binary[N_CORTICAL:, N_CORTICAL:]
        surviving = int(sc_block.sum()) // 2
        cats.append(dlbl); surv_list.append(surviving)
        lost_list.append(possible_sc - surviving)
    bars1 = ax_subc.bar(cats, surv_list, color="#4CAF50", label="Surviving")
    ax_subc.bar(cats, lost_list, bottom=surv_list, color="#F44336",
                alpha=0.6, label="Thresholded out")
    ax_subc.axhline(y=possible_sc, color="black", lw=1, linestyle=":",
                    label=f"All possible ({possible_sc})")
    ax_subc.set_ylabel("Subcortical edges", fontsize=8)
    ax_subc.set_title("Subcortical block:\nsurviving vs thresholded", fontsize=10,
                      fontweight="bold")
    ax_subc.legend(fontsize=7)
    for b, n in zip(bars1, surv_list):
        ax_subc.text(b.get_x() + b.get_width() / 2, b.get_height() / 2,
                     str(n), ha="center", va="center", fontsize=9,
                     fontweight="bold", color="white")
    ax_subc.tick_params(labelsize=7)

    fig.suptitle(f"{title_prefix} — Edge Density Comparison (5% vs 30%)",
                 fontsize=13, fontweight="bold")
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


print("Edge density comparison …")
for grp, tag in [("an", "AN"), ("hc", "HC")]:
    fc = load_static_fc(grp, 1)
    if fc is None:
        print(f"  [{tag}] FC file not found — skipping")
        continue
    np.fill_diagonal(fc, 0)
    plot_density_comparison(fc, f"{tag} Subject 01",
                            OUT / f"edge_density_comparison_{grp}_subj01.png")

print("Group-mean edge density comparison …")
for grp, tag, subj_ids in [("an", "AN", range(1, 6)),
                            ("hc", "HC", range(1, 6))]:
    mats = []
    for sid in subj_ids:
        fc = load_static_fc(grp, sid)
        if fc is not None:
            np.fill_diagonal(fc, 0)
            mats.append(fc)
    if mats:
        fc_mean = zmean(np.array(mats), axis=0)
        plot_density_comparison(fc_mean, f"{tag} Group Mean (N={len(mats)})",
                                OUT / f"edge_density_comparison_{grp}_groupmean.png")

print("Done — figures saved to", OUT)
