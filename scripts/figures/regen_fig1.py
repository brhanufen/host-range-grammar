#!/usr/bin/env python3
"""
regen_fig1.py — Figure 1 (Dataset curation and evaluation framework).

Figure 1 is a 4-panel composite:
  A  Data-acquisition / QC schematic (drawn here in matplotlib).
  B  Evaluation-scheme schematic (accession- vs species-grouped).
  C  Class + genome-length distributions.
  D  Summary count table.

Panels B, C, D are pre-rendered PNGs under corrected_figures/panels/.
This script (re)draws Panel A cleanly with generous margins so no box is
clipped, then assembles the composite at the canonical 3000-px width.

Panel A is a fixed schematic: the values shown (3,511 raw genomes,
3,031 curated = 2,816 ARB + 215 ISFV, 59,980 windows) are the dataset
constants reported throughout the manuscript.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager as fm
import matplotlib.image as mpimg
from PIL import Image
import os

PANELS = os.path.join(os.path.dirname(__file__), "..", "..", "figures")  # adjust if needed
PANELS = os.environ.get("FIG1_PANELS_DIR", "corrected_figures/panels")

AB = fm.FontProperties(fname="/System/Library/Fonts/Supplemental/Arial Bold.ttf")
BLUE_F, BLUE_B = "#BDD7EE", "#2E75B6"
GRAY_F, GRAY_B = "#F2F2F2", "#404040"
GRN_F,  GRN_B  = "#C6EFCE", "#2E7D32"
TXT, ARR = "#333333", "#404040"


def build_panel_A(outbase):
    W, Hf = 100.0, 58.1
    fig = plt.figure(figsize=(12, 12 * Hf / W), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, Hf); ax.axis("off")

    def box(cx, cy, w, h, fc, ec, text, fs=21, lw=2.2):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle="round,pad=0.3,rounding_size=1.2",
                     fc=fc, ec=ec, lw=lw, mutation_aspect=1.0))
        ax.text(cx, cy, text, ha="center", va="center",
                fontproperties=AB, fontsize=fs, color=TXT, zorder=5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=22, lw=2.2, color=ARR,
                     shrinkA=0, shrinkB=0, zorder=3))

    box(24, 50.5, 30, 9,   BLUE_F, BLUE_B, "NCBI Virus\nGenBank")
    box(62, 50.5, 34, 9,   BLUE_F, BLUE_B, "Literature\n(Blitvich et al. 2015)")
    box(43, 35.5, 56, 8.4, GRAY_F, GRAY_B, "Raw: 3,511 genomes (3,270 ARB + 241 ISFV)")
    box(30, 22.5, 46, 8.4, GRAY_F, GRAY_B, "QC: Length 9\u201312 kb, N<1%, Deduplicate")
    box(76, 22.5, 34, 9,   GRN_F,  GRN_B,  "Curated: 3,031\n(2,816 ARB + 215 ISFV)")
    box(58, 8.5,  56, 8.0, GRN_F,  GRN_B,  "Windows: 1 kb, stride 500 bp  \u00b7  59,980 windows")

    arrow(24, 46.0, 38, 39.9)
    arrow(62, 46.0, 50, 39.9)
    arrow(43, 31.3, 43, 26.9)
    arrow(53, 22.5, 59, 22.5)
    arrow(76, 18.0, 64, 12.7)

    fig.savefig(outbase + ".png", dpi=300, facecolor="white")
    fig.savefig(outbase + ".pdf", facecolor="white")
    plt.close(fig)


def assemble(panels_dir, out_png="Figure1.png", out_pdf="Figure1.pdf"):
    aA = Image.open(f"{panels_dir}/Fig1_A.png").size
    asp = {"A": aA[1] / aA[0], "B": 0.803, "C": 0.584, "D": 0.304}
    imgs = {k: mpimg.imread(f"{panels_dir}/Fig1_{k}.png") for k in "ABCD"}
    Wc, LM, LET_DX = 10.0, 1.05, 0.60
    wA, wB, wC, wD = 6.0, 3.5, 4.7, 6.8
    hA, hB, hC, hD = wA * asp["A"], wB * asp["B"], wC * asp["C"], wD * asp["D"]
    top_m = bot_m = 0.30; g1 = g2 = 0.55
    Hc = top_m + hA + g1 + max(hB, hC) + g2 + hD + bot_m
    fig = plt.figure(figsize=(Wc, Hc), dpi=300)

    def place(img, left, bottom, w, h):
        ax = fig.add_axes([left / Wc, bottom / Hc, w / Wc, h / Hc]); ax.imshow(img); ax.axis("off")

    def letter(ch, xleft, ytop):
        fig.text((xleft - LET_DX) / Wc, ytop / Hc, ch, fontproperties=AB,
                 fontsize=26, va="top", ha="left")

    yA_top = Hc - top_m; bA = yA_top - hA
    place(imgs["A"], LM, bA, wA, hA); letter("A", LM, yA_top)
    yBC_top = bA - g1; lB = LM; bB = yBC_top - hB; lC = lB + wB + 0.55; bC = yBC_top - hC
    place(imgs["B"], lB, bB, wB, hB); letter("B", lB, yBC_top)
    place(imgs["C"], lC, bC, wC, hC); letter("C", lC, yBC_top)
    yD_top = min(bB, bC) - g2; bD = yD_top - hD
    place(imgs["D"], LM, bD, wD, hD); letter("D", LM, yD_top)
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    panels = os.environ.get("FIG1_PANELS_DIR", "corrected_figures/panels")
    build_panel_A(f"{panels}/Fig1_A")
    assemble(panels)
    print("Figure1 rebuilt.")
