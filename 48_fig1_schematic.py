"""
48 — New Figure 1 (revision, Referee 1): schematic of the routes, the integration operators and the
bandwidth-matched pseudo-fusion.

The referee asked for a schematic showing what the three routes are, where the integration operators
enter, and how the pseudo-fusion is built, so that the control can be understood before any numbers
are read. Panel (a) is the real cross-order ensemble, panel (b) the two baselines, panel (c) the
bandwidth-matched pseudo-fusion. No data are plotted.

python 48_fig1_schematic.py  ->  figures/fig1_schematic.png (+ 300 dpi submission copies)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figexport import export_submission
import _style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"
_style.apply()
C_A, C_V, C_P, C_E, C_S, C_B = "#4C72B0", "#DD8452", "#55A868", "#2f6fb3", "#c2603a", "#9aa0a6"


def box(ax, x, y, w, h, text, fc, fs=9, tc="white", lw=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=lw, edgecolor="0.35", facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3)


def arrow(ax, x1, y1, x2, y2, style="-|>", color="0.35", lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9,
                                 linewidth=lw, color=color, linestyle=ls, zorder=1,
                                 shrinkA=1, shrinkB=1))


fig, ax = plt.subplots(3, 1, figsize=(7.2, 7.8))
for a in ax:
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off")

# ---------------- (a) cross-order ensemble ----------------
a = ax[0]
a.set_title("(a) Cross-order ensemble  E_cross", fontsize=9.5, loc="left")
box(a, 0.02, 0.42, 0.17, 0.16, "MEG\n(192 grad,\n6 lags)", "0.35")
ys = [0.74, 0.44, 0.14]
for y, lab, c in zip(ys, ["decode $a$", "decode $v$", "decode $p$"], [C_A, C_V, C_P]):
    box(a, 0.28, y, 0.20, 0.15, lab, c)
    arrow(a, 0.19, 0.50, 0.28, y + 0.075)
for y, op, c in zip(ys, ["$\\mathbf{T}_2$", "$\\mathbf{T}_1$", "identity"], [C_A, C_V, C_P]):
    box(a, 0.54, y, 0.14, 0.15, op, c)
    arrow(a, 0.48, y + 0.075, 0.54, y + 0.075)
box(a, 0.78, 0.42, 0.18, 0.16, "weighted\nfusion", C_E)
for y in ys:
    arrow(a, 0.68, y + 0.075, 0.78, 0.50)
a.text(0.61, 0.045, "position space", ha="center", fontsize=8, color="0.4")

# ---------------- (b) baselines ----------------
a = ax[1]
a.set_title("(b) Same-order baselines", fontsize=9.5, loc="left")
box(a, 0.02, 0.42, 0.17, 0.16, "MEG", "0.35")
for i, y in enumerate([0.76, 0.60, 0.44]):
    box(a, 0.30, y, 0.26, 0.12, "decode $p$ (resample)" if i == 1 else "decode $p$", C_P, fs=8)
    arrow(a, 0.19, 0.50, 0.30, y + 0.06)
a.text(0.43, 0.34, "$\\vdots$  10 members", ha="center", fontsize=8, color="0.4")
box(a, 0.68, 0.52, 0.28, 0.14, "average  →  E_bag", C_B, fs=8)
arrow(a, 0.56, 0.60, 0.68, 0.59)
box(a, 0.30, 0.10, 0.26, 0.12, "decode $p$ on random\nfeature subspaces", C_P, fs=8)
arrow(a, 0.19, 0.46, 0.30, 0.16)
box(a, 0.68, 0.10, 0.28, 0.12, "decorrelation-matched", C_B, fs=8)
arrow(a, 0.56, 0.16, 0.68, 0.16)

# ---------------- (c) bandwidth-matched pseudo-fusion ----------------
a = ax[2]
a.set_title("(c) Bandwidth-matched pseudo-fusion  E_spec", fontsize=9.5, loc="left")
box(a, 0.02, 0.44, 0.19, 0.14, "decode $p$\n(one decoder)", C_P)
for y, lab in zip([0.76, 0.46, 0.16], ["$L_a$", "identity", "$L_v$"]):
    box(a, 0.36, y, 0.16, 0.14, lab, C_S if lab != "identity" else C_P)
    arrow(a, 0.21, 0.51, 0.36, y + 0.07)
box(a, 0.64, 0.44, 0.19, 0.14, "weighted\nfusion", C_S)
for y in [0.76, 0.46, 0.16]:
    arrow(a, 0.52, y + 0.07, 0.64, 0.51)
a.text(0.44, 0.015, "$|L_o(f)|$ matched to route $o$'s bandwidth,\nestimated on training folds only",
       ha="center", fontsize=8, color="0.4")
a.text(0.74, 0.30, "zero cross-order\nneural content", ha="center", fontsize=8, color=C_S)

fig.tight_layout()
fig.savefig(f"{PROJ}/figures/fig1_schematic.png", bbox_inches="tight")
export_submission(fig, list(ax), "fig1_schematic")
print("SAVED figures/fig1_schematic.png")
