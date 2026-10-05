"""
42 — Supplementary Figure S1 (revision R4): descriptive analyses moved out of the main text.

The R4 review moved the lag dissociation (old Fig. 1d) and the six-panel seed-stability figure
(old Fig. 3) to the supplement. To keep the supplement inside its four-display-item budget they are
merged with the route topography into one figure with lettered blocks:

  (a) raw squared-weight lag profiles per decoded order (script 02) — the descriptive pattern;
  (b) the same profiles under permutation importance (script 26) — the dissociation does not survive;
  (c) multi-seed stability of the attention contrasts on both backbones (scripts 10/13/31);
  (d) group-mean decoder-importance topographies per route (script 23).

Blocks (c) and (d) are composited from the rendered panels of scripts 31 and 23.

python 42_figS1_supplement.py  ->  figures/figS1_supplement.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figexport import export_submission
import _style
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"
_style.apply()
C = {"a": "#4C72B0", "v": "#DD8452", "p": "#55A868"}
LAB = {"a": "acceleration", "v": "velocity", "p": "position"}

ri = np.load(f"{PROJ}/results/route_importance.npz", allow_pickle=True)
lags = ri["lags_ms"]
Lg = ri["lag"].mean(0); Lg = Lg / Lg.sum(1, keepdims=True)
hs = list(np.load(f"{PROJ}/results/haufe_surrogate.npz", allow_pickle=True)["recs"])
def _d(x): return x.item() if hasattr(x, "item") else x
perm = {o: np.mean([_d(r["permlag"])[o] for r in hs], axis=0) for o in "avp"}

fig = plt.figure(figsize=(7.2, 9.8))
gs = GridSpec(3, 2, height_ratios=[1.0, 1.45, 1.75], hspace=0.55, wspace=0.34)

ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
for i, o in enumerate("avp"):
    ax_a.plot(lags, Lg[i], "o-", color=C[o], label=LAB[o], lw=1.5)
    ax_b.plot(lags, perm[o] / perm[o].sum(), "o-", color=C[o], label=LAB[o], lw=1.5)
ax_a.set_title("(A) Raw squared decoder weights", fontsize=9.5, loc="left")
ax_b.set_title("(B) Permutation importance (same decoders)", fontsize=9.5, loc="left")
for a in (ax_a, ax_b):
    a.set_xlabel("MEG lag (ms)"); a.set_ylabel("normalised importance"); a.legend(frameon=False, fontsize=9.5, loc="upper center",
             bbox_to_anchor=(0.5, -0.22), ncol=3, handlelength=1.3, columnspacing=1.0)

for row, png, title in [(1, "fig3_seed_stability_combined.png",
                         "(C) Multi-seed stability of the added-block contrasts"),
                        (2, "fig6_topography.png",
                         "(D) Group-mean decoder-importance topographies per route")]:
    ax = fig.add_subplot(gs[row, :])
    ax.imshow(mpimg.imread(f"{PROJ}/figures/_intermediate/{png}"))
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.axis("off")

fig.savefig(f"{PROJ}/figures/figS1_supplement.png", bbox_inches="tight")
export_submission(fig, fig.axes, "figS1_supplement")
print("SAVED figures/figS1_supplement.png")
