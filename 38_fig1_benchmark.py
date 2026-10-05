"""
38 — Manuscript Figure 1 (revision R4): base reconstruction benchmark.

Replaces figures/fig1_ensemble.png (four panels incl. the lag dissociation) with a three-panel
benchmark. Changes forced by the R4 review:
  * the lag-importance panel moves to the supplement (it does not survive the Haufe / permutation /
    shared-timing controls, and read as a biological dissociation in a main figure);
  * participant-level points (n = 9, the independent unit) are overlaid on every panel;
  * the fold-draw-sensitive trained-LSTM backbone is hatched and labelled directional-only.

python 38_fig1_benchmark.py  ->  figures/fig2_benchmark.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figexport import export_submission
import _style
from _ci import err as ci_err
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"
_style.apply()
C = {"a": "#4C72B0", "v": "#DD8452", "p": "#55A868", "cross": "#2f6fb3",
     "bag": "#9aa0a6", "mtl": "#C44E52"}

def recs(stem):
    try: return list(np.load(f"{PROJ}/results/{stem}.npz", allow_pickle=True)["recs"])
    except Exception: return None
def col(rs, k): return np.array([r[k] for r in rs])
def subj_means(rs, k):
    """collapse the two sessions of each participant -> the independent unit (n = 9)"""
    s = np.array([r["subj"] for r in rs]); v = col(rs, k)
    return np.array([v[s == u].mean() for u in np.unique(s)])

fig, ax = plt.subplots(3, 1, figsize=(7.2, 8.4))

# ---------- (a) routes, fusion and the shared bottleneck (ridge) ----------
r0 = recs("ensemble_distance")
order = ["a_route", "v_route", "p_direct", "E_wt", "E_bag", "mtl_r1"]
labs = ["a→pos", "v→pos", "p direct", "E_cross", "E_bag", "MTL\n(rank 1)"]
cols = [C["a"], C["v"], C["p"], C["cross"], C["bag"], C["mtl"]]
for i, (k, c) in enumerate(zip(order, cols)):
    sm = subj_means(r0, k)
    m, e = ci_err(sm)
    ax[0].bar(i, m, 0.68, color=c, alpha=0.85)
    ax[0].errorbar(i, m, yerr=e, fmt="none", ecolor="0.25", capsize=3, lw=1.1, zorder=4)
    ax[0].scatter(np.full(len(sm), i) + np.linspace(-0.16, 0.16, len(sm)), sm,
                  s=11, color="black", alpha=0.55, zorder=3, linewidth=0)
ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels(labs, fontsize=9.5)
ax[0].set_ylabel("position-residual per-trial $r$")
ax[0].set_title("(a) Routes, fusion and shared bottleneck (ridge)", fontsize=9.5, loc="left")
ax[0].axhline(0, color="k", lw=0.6)
d = subj_means(r0, "E_wt") - subj_means(r0, "E_bag")

# ---------- (b) fusion vs bagging across backbones ----------
bbs = [("ridge", "ensemble_distance"), ("frozen\nSSL", "ensemble_ssl"),
       ("random\nproj.", "ensemble_rand"), ("trained\nLSTM", "ensemble_lstm")]
present = [(nm, st) for nm, st in bbs if recs(st) is not None]
x = np.arange(len(present)); w = 0.38
for i, (nm, st) in enumerate(present):
    rs = recs(st); soft = ("LSTM" in nm)
    for off, k, c, lab in [(-w/2, "E_wt", C["cross"], "E_cross"), (w/2, "E_bag", C["bag"], "E_bag")]:
        sm = subj_means(rs, k)
        m, e = ci_err(sm)
        ax[1].bar(i + off, m, w, color=c, alpha=0.45 if soft else 0.85,
                  hatch="//" if soft else None, edgecolor="white",
                  label=lab if i == 0 else None)
        ax[1].errorbar(i + off, m, yerr=e, fmt="none", ecolor="0.25", capsize=2.5, lw=1.0, zorder=4)
        ax[1].scatter(np.full(len(sm), i + off) + np.linspace(-0.1, 0.1, len(sm)), sm,
                      s=8, color="black", alpha=0.35 if soft else 0.55, zorder=3, linewidth=0)
ax[1].set_xticks(x); ax[1].set_xticklabels([nm for nm, _ in present], fontsize=9.5)
ax[1].set_ylabel("position-residual per-trial $r$")
ax[1].set_title("(b) Fusion exceeds same-order bagging", fontsize=9.5, loc="left")
ax[1].legend(frameon=False, fontsize=9.5); ax[1].axhline(0, color="k", lw=0.6)

# ---------- (c) single-task vs shared-bottleneck multi-task ----------
pm = [(nm, st) for nm, st in present if "mtl_r1" in recs(st)[0]]
xm = np.arange(len(pm))
for i, (nm, st) in enumerate(pm):
    rs = recs(st)
    for off, k, c, lab in [(-w/2, "p_direct", C["p"], "single-task (p direct)"),
                           (w/2, "mtl_r1", C["mtl"], "multi-task (rank-1 shared)")]:
        sm = subj_means(rs, k)
        m, e = ci_err(sm)
        ax[2].bar(i + off, m, w, color=c, alpha=0.85, label=lab if i == 0 else None)
        ax[2].errorbar(i + off, m, yerr=e, fmt="none", ecolor="0.25", capsize=2.5, lw=1.0, zorder=4)
        ax[2].scatter(np.full(len(sm), i + off) + np.linspace(-0.1, 0.1, len(sm)), sm,
                      s=8, color="black", alpha=0.55, zorder=3, linewidth=0)
ax[2].set_xticks(xm); ax[2].set_xticklabels([nm for nm, _ in pm], fontsize=9.5)
ax[2].set_ylabel("position-residual per-trial $r$")
ax[2].set_title("(c) Forced low-rank sharing hurts", fontsize=9.5, loc="left")
ax[2].legend(frameon=False, fontsize=9.5); ax[2].axhline(0, color="k", lw=0.6)

fig.tight_layout()
fig.savefig(f"{PROJ}/figures/fig2_benchmark.png", bbox_inches="tight")
export_submission(fig, list(ax), "fig2_benchmark")
print("SAVED figures/fig2_benchmark.png")
