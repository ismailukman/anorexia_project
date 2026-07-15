#!/usr/bin/env python3
"""Two-panel atlas overview: Schaefer-200 (Yeo-7) and Tian Scale I (16 ROIs)."""
from io import BytesIO
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from nilearn import plotting, datasets

ROOT  = Path("/Users/ismaila/Documents/C-Codes/AnorexiaProject")
ATLAS = ROOT / "data/atlas"
OUT   = ROOT / "output/figures/stage1_fc/atlas_overview_s1_only.png"

SCHAEFER_NII = ATLAS / "schaefer_2018/schaefer_2018/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
TIAN_S1_NII  = ATLAS / "tian_s1/Tian_Subcortex_S1_3T_2009cAsym.nii.gz"

YEO7 = [
    ("VN",  "#781286", "Visual Network"),
    ("SMN", "#4682B4", "Somatomotor Network"),
    ("DAN", "#00760E", "Dorsal Attention Network"),
    ("VAN", "#C43AFA", "Ventral Attention Network"),
    ("LN",  "#DCF8A4", "Limbic Network"),
    ("FPN", "#E69422", "Frontoparietal Network"),
    ("DMN", "#CD3E4E", "Default Mode Network"),
]
YEO_SIZES_LH = [14, 16, 13, 11, 6, 13, 27]
YEO_SIZES_RH = [15, 19, 13, 11, 6, 17, 19]

S1_STRUCTURES = {
    "Hippocampus":       [(0, "#E31A1C", "Hippocampus (L,R)")],
    "Amygdala":          [(1, "#33A02C", "Amygdala (L,R)")],
    "Thalamus": [
        (2, "#1F78B4", "Post. Thalamus (L,R)"),
        (3, "#A6CEE3", "Ant. Thalamus (L,R)"),
    ],
    "Nucleus Accumbens": [(4, "#FF7F00", "Nucleus Accumbens (L,R)")],
    "Globus Pallidus":   [(5, "#6A3D9A", "Globus Pallidus (L,R)")],
    "Putamen":           [(6, "#FB9A99", "Putamen (L,R)")],
    "Caudate":           [(7, "#B15928", "Caudate (L,R)")],
}

S1_COLORS_LIST = ["black"] * 9
for entries in S1_STRUCTURES.values():
    for idx, color, _ in entries:
        S1_COLORS_LIST[idx + 1] = color
s1_cmap = ListedColormap(S1_COLORS_LIST)


def make_schaefer_network_img():
    img  = nib.load(str(SCHAEFER_NII))
    data = img.get_fdata().astype(int)
    out  = np.zeros_like(data, dtype=np.float32)
    p = 1
    for net_i, sz in enumerate(YEO_SIZES_LH):
        for _ in range(sz):
            out[data == p] = net_i + 1
            p += 1
    for net_i, sz in enumerate(YEO_SIZES_RH):
        for _ in range(sz):
            out[data == p] = net_i + 1
            p += 1
    return nib.Nifti1Image(out, img.affine, img.header)


def make_s1_img():
    img  = nib.load(str(TIAN_S1_NII))
    data = img.get_fdata().astype(int)
    out  = np.zeros_like(data, dtype=np.float32)
    for parcel in range(1, 17):
        out[data == parcel] = (parcel - 1) % 8 + 1
    return nib.Nifti1Image(out, img.affine, img.header)


def capture(display, dpi=200):
    buf = BytesIO()
    display.savefig(buf, dpi=dpi)
    buf.seek(0)
    from PIL import Image
    return np.array(Image.open(buf).convert("RGBA"))


def make_legend_yeo(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.04, 0.97, "Yeo-7 Network", fontsize=8, fontweight="bold",
            va="top", transform=ax.transAxes)
    y = 0.87
    for abbr, color, name in YEO7:
        patch = mpatches.FancyBboxPatch(
            (0.04, y - 0.05), 0.12, 0.055,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=color, transform=ax.transAxes, clip_on=False
        )
        ax.add_patch(patch)
        ax.text(0.20, y - 0.01, f"{abbr}   {name}",
                fontsize=7, va="top", transform=ax.transAxes)
        y -= 0.115


def make_legend_s1(ax):
    y = 0.98; dy_hdr = 0.09; dy_entry = 0.075
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    for struct_name, entries in S1_STRUCTURES.items():
        ax.text(0.02, y, struct_name, fontsize=8, fontweight="bold",
                va="top", ha="left", transform=ax.transAxes)
        y -= dy_hdr
        for _, color, label in entries:
            patch = mpatches.FancyBboxPatch(
                (0.02, y - 0.045), 0.09, 0.05,
                boxstyle="square,pad=0", linewidth=0,
                facecolor=color, transform=ax.transAxes, clip_on=False
            )
            ax.add_patch(patch)
            ax.text(0.14, y - 0.01, label, fontsize=7.5, va="top",
                    ha="left", transform=ax.transAxes)
            y -= dy_entry
        y -= 0.01


print("Building Schaefer Yeo-7 image …")
schaefer_net_img = make_schaefer_network_img()
yeo_cmap = ListedColormap(["black"] + [e[1] for e in YEO7])

print("Building Tian S1 image …")
tian_s1_img = make_s1_img()

print("Loading MNI template …")
mni_bg = datasets.load_mni152_template(resolution=1)

common_kw = dict(annotate=True, draw_cross=False, colorbar=False,
                 black_bg=False, dim=0)

print("Rendering Panel A …")
disp_A = plotting.plot_roi(
    schaefer_net_img, bg_img=mni_bg,
    display_mode="ortho", cut_coords=(40, 30, 40),
    cmap=yeo_cmap, vmin=0.5, vmax=7.5,
    alpha=0.75, title="", **common_kw
)

print("Rendering Panel B …")
disp_B = plotting.plot_roi(
    tian_s1_img, bg_img=mni_bg,
    display_mode="y", cut_coords=[-38, -24, -16, -6, 6, 16],
    cmap=s1_cmap, vmin=0.5, vmax=8.5,
    alpha=0.85, title="", **common_kw
)

print("Capturing …")
arr_A = capture(disp_A)
arr_B = capture(disp_B)

print("Composing …")
fig = plt.figure(figsize=(24, 14), facecolor="white")
outer = gridspec.GridSpec(2, 2, figure=fig,
                          width_ratios=[5, 1.4],
                          hspace=0.30, wspace=0.01,
                          left=0.005, right=0.995,
                          top=0.975, bottom=0.02)

panel_labels = [
    "A   Schaefer-200 Cortical Atlas  ·  Yeo-7 Networks",
    "B   Tian Scale I Subcortical Atlas  ·  16 Bilateral ROIs",
]

for row, (arr, plbl, legend_fn) in enumerate([
    (arr_A, panel_labels[0], make_legend_yeo),
    (arr_B, panel_labels[1], make_legend_s1),
]):
    ax_brain = fig.add_subplot(outer[row, 0])
    ax_brain.imshow(arr)
    ax_brain.axis("off")
    pos = ax_brain.get_position()
    fig.text(pos.x0, pos.y1 + 0.004, plbl,
             fontsize=11, fontweight="bold", va="bottom", ha="left")

    ax_leg = fig.add_subplot(outer[row, 1])
    legend_fn(ax_leg)

fig.text(0.5, 0.005,
         "All atlases in MNI152 2009cAsym space  ·  "
         "Subcortical slices: y = −38, −24, −16, −6, +6, +16 mm  ·  "
         "Cortical views: y=30, x=40, z=40",
         ha="center", va="bottom", fontsize=7.5, color="gray", style="italic")

plt.savefig(str(OUT), dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved → {OUT}")
