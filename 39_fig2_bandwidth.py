"""
39 — Manuscript Figure 2 (revision R4): why the offline fusion gain occurs.

Promoted to the paper's central figure by the R4 review. Three panels:
  (a) paired E_bag -> E_spec -> E_cross in the offline zero-phase frame, participant means
      overlaid, with an inset showing the participant-level bandwidth-clean residual against the
      pre-specified SESOI (+/-0.01 r) and its one-sided upper 95% bound (+0.003 r);
  (b) the same decomposition across three backbones (ridge / frozen SSL / random projection),
      split into bandwidth-explained and residual, with the optimally-smoothed single decoder
      (p_smooth) as a reference line;
  (c) estimator-strength control: the same contrast under a stronger learned stack and a
      frequency-resolved Wiener/GLS combiner.

python 39_fig2_bandwidth.py  ->  figures/fig3_bandwidth.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figexport import export_submission
import _style
from _ci import err as ci_err, ci as ci_bounds
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"
_style.apply()
C_BAG, C_CROSS, C_SPEC = "#9aa0a6", "#2f6fb3", "#c2603a"

def recs(stem): return list(np.load(f"{PROJ}/results/{stem}.npz", allow_pickle=True)["recs"])
def col(rs, k): return np.array([r[k] for r in rs])
def subj(rs, k):
    s = np.array([r["subj"] for r in rs]); v = col(rs, k)
    return np.array([v[s == u].mean() for u in np.unique(s)])

sp = recs("spectral_control")
fig, ax = plt.subplots(3, 1, figsize=(7.2, 8.8))

# ---------- (a) paired decomposition ----------
eb, es, ec = subj(sp, "E_bag"), subj(sp, "E_spec"), subj(sp, "E_cross")
xs = [0, 1, 2]
for i in range(len(eb)):
    ax[0].plot(xs, [eb[i], es[i], ec[i]], color="0.8", lw=0.7, zorder=1)
for x, v, c in [(0, eb, C_BAG), (1, es, C_SPEC), (2, ec, C_CROSS)]:
    ax[0].scatter([x]*len(v), v, s=18, color=c, zorder=3, edgecolor="white", linewidth=0.4)
    mv, ev = ci_err(v)
    ax[0].errorbar([x], [mv], yerr=ev, fmt="none", ecolor="black", capsize=4, lw=1.3, zorder=4)
    ax[0].scatter([x], [mv], s=95, color=c, zorder=5, edgecolor="black", linewidth=1.1, marker="D")
ax[0].set_xticks(xs); ax[0].set_xlim(-0.4, 2.75)
ax[0].set_xticklabels(["E_bag\n(same-order)", "E_spec\n(bandwidth-only)", "E_cross\n(cross-order)"], fontsize=9.5)
ax[0].set_ylabel("position-residual per-trial $r$")
ax[0].set_title("(a) Offline: the margin is bandwidth", fontsize=9.5, loc="left")
ax[0].plot([1, 2], [es.mean(), ec.mean()], color="black", lw=0.8, ls=":")
lo, hi = min(eb.min(), es.min(), ec.min()), max(eb.max(), es.max(), ec.max())
ax[0].set_ylim(lo - 0.015, hi + 0.26)

ins = ax[0].inset_axes([0.30, 0.72, 0.44, 0.22])
res = ec - es
ins.axvspan(-0.01, 0.01, color="0.88", zorder=0)
ins.axvline(0, color="0.5", lw=0.7)
ins.scatter(res, np.random.RandomState(0).uniform(-0.35, 0.35, len(res)), s=9,
            color=C_CROSS, alpha=0.8, linewidth=0)
ins.scatter([res.mean()], [0], s=55, color="black", marker="D", zorder=4)
ins.plot([-0.0105, 0.0032], [0, 0], color="black", lw=1.2)
ins.set_yticks([]); ins.set_xlim(-0.022, 0.022); ins.set_ylim(-0.9, 1.25)
ins.set_xticks([-0.01, 0, 0.01]); ins.tick_params(labelsize=9.5)
ins.set_xlabel("E_cross − E_spec  ($\\Delta r$)", fontsize=9.5, labelpad=2)
ins.set_facecolor("white")
for s in ("top", "right"): ins.spines[s].set_visible(False)

# ---------- (b) backbone decomposition ----------
bbs = [("ridge", "backbone_spectral_raw"), ("frozen SSL", "backbone_spectral_ssl"),
       ("random proj.", "backbone_spectral_rand")]
x = np.arange(len(bbs)); w = 0.34
for i, (nm, st) in enumerate(bbs):
    rs = recs(st)
    tot = (subj(rs, "E_cross") - subj(rs, "E_bag")).mean()
    bw = (subj(rs, "E_spec") - subj(rs, "E_bag")).mean()
    resid = (subj(rs, "E_cross") - subj(rs, "E_spec")).mean()
    tot_v = subj(rs, "E_cross") - subj(rs, "E_bag")
    bw_v = subj(rs, "E_spec") - subj(rs, "E_bag")
    mt, et = ci_err(tot_v); mb, eb_ = ci_err(bw_v)
    ax[1].bar(i - w/2, mt, w, color=C_CROSS, alpha=0.85, label="total gain\n(E_cross − E_bag)" if i == 0 else None)
    ax[1].bar(i + w/2, mb, w, color=C_SPEC, alpha=0.85, label="bandwidth-explained\n(E_spec − E_bag)" if i == 0 else None)
    ax[1].errorbar([i - w/2, i + w/2], [mt, mb], yerr=np.hstack([et, eb_]), fmt="none",
                   ecolor="0.25", capsize=2.5, lw=1.0, zorder=4)
    ax[1].scatter([i + w/2 + 0.02], [resid], s=0)  # keep scale
    smooth = (subj(rs, "p_smooth") - subj(rs, "E_bag")).mean()
    ax[1].plot([i - 0.42, i + 0.42], [smooth, smooth], color="black", lw=1.1, ls="--",
               label="single decoder,\noptimally smoothed" if i == 0 else None)
ax[1].set_xticks(x); ax[1].set_xticklabels([nm for nm, _ in bbs], fontsize=9.5)
ax[1].set_ylabel("gain over same-order bagging ($\\Delta r$)")
ax[1].set_title("(b) Across three stable backbones", fontsize=9.5, loc="left")
ax[1].axhline(0, color="k", lw=0.6); ax[1].legend(frameon=False, fontsize=9.5, loc="upper center", ncol=3,
             bbox_to_anchor=(0.5, -0.22), handlelength=1.4, columnspacing=1.2)

# ---------- (c) estimator-strength control ----------
es_rs = recs("estimator_strength")
combs = [("fixed weights\n(paper)", "perf_cross", "perf_spec"),
         ("learned stack\n(nested OOF)", "stack_cross", "stack_spec"),
         ("band-wise\nWiener/GLS", "wien_cross", "wien_spec")]
for i, (nm, kc, ks) in enumerate(combs):
    d = subj(es_rs, kc) - subj(es_rs, ks)
    md, ed = ci_err(d)
    ax[2].bar(i, md, 0.6, color=C_CROSS if md > 0 else C_BAG, alpha=0.85)
    ax[2].errorbar(i, md, yerr=ed, fmt="none", ecolor="0.25", capsize=3, lw=1.1, zorder=4)
    ax[2].scatter(np.full(len(d), i) + np.linspace(-0.14, 0.14, len(d)), d, s=11,
                  color="black", alpha=0.55, zorder=3, linewidth=0)
    pos = int((d > 0).sum())
ax[2].axhspan(-0.01, 0.01, color="0.9", zorder=0)
ax[2].set_xticks(range(len(combs))); ax[2].set_xticklabels([c[0] for c in combs], fontsize=9.5)
ax[2].set_ylabel("cross-order −\nbandwidth-matched ($\\Delta r$)")
ax[2].set_title("(c) A stronger estimator finds no usable residual", fontsize=9.5, loc="left")
ax[2].axhline(0, color="k", lw=0.6)

fig.tight_layout(h_pad=2.4)
fig.savefig(f"{PROJ}/figures/fig3_bandwidth.png", bbox_inches="tight")
export_submission(fig, list(ax), "fig3_bandwidth")
print("SAVED figures/fig3_bandwidth.png")
