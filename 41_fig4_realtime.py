"""
41 — Manuscript Figure 4 (revision R4): real-time behaviour and deployment bounds.

The R4 review's P0 item: the published panel showed the lag-only tau=0 margin (+0.079, non-causal
operators) under the label "causal", while the end-to-end causal pipeline gives +0.044. This figure
separates them and adds the velocity-anchored causal controls, so the causal claim in the figure is
the one the analysis supports: the advantage is available online, its mechanism is unresolved.

  (a) fully causal pipeline at tau = 0: E_bagV / E_specV / E_cross (script 33) with the p-anchored
      reference (script 25);
  (b) latency curves — lag-only budget (script 16) with the fully causal tau = 0 point marked;
  (c) calibrated, native-unit metrics: both models fail;
  (d) cross-session collapse (script 18).

python 41_fig4_realtime.py  ->  figures/fig5_realtime.png
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
C_BAG, C_CROSS, C_SPEC = "#9aa0a6", "#2f6fb3", "#c2603a"

def npz(stem): return np.load(f"{PROJ}/results/{stem}.npz", allow_pickle=True)
def sub(rs, k):
    s = np.array([r["subj"] for r in rs]); v = np.array([r[k] for r in rs])
    return np.array([v[s == u].mean() for u in np.unique(s)])

va = list(npz("causal_velocity_anchor")["recs_tau0"])
cz = list(npz("causal_latency")["recs"])
cp = list(npz("causal_pipeline")["recs_tau0"])
sp = list(npz("spectral_control")["recs"])
dep = npz("deployment"); cs = list(dep["cs"]); ws = list(dep["ws"])

fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.8))
ax = ax.ravel()

# ---------- (a) fully causal decomposition at tau = 0 ----------
bagV, specV, cross = sub(va, "E_bagV"), sub(va, "E_specV"), sub(va, "E_cross")
bagP = sub(va, "E_bag")
xs = [0, 1, 2]
for i in range(len(bagV)):
    ax[0].plot(xs, [bagV[i], specV[i], cross[i]], color="0.85", lw=0.7, zorder=1)
for x, v, c in [(0, bagV, C_BAG), (1, specV, C_SPEC), (2, cross, C_CROSS)]:
    ax[0].scatter([x]*len(v), v, s=16, color=c, zorder=3, edgecolor="white", linewidth=0.4)
    mv, ev = ci_err(v)
    ax[0].errorbar([x], [mv], yerr=ev, fmt="none", ecolor="black", capsize=4, lw=1.3, zorder=4)
    ax[0].scatter([x], [mv], s=85, color=c, zorder=5, edgecolor="black", linewidth=1.1, marker="D")
ax[0].axhline(bagP.mean(), color="0.55", ls="--", lw=0.9, label="position-anchored bag")
ax[0].legend(frameon=False, fontsize=9.5, loc="upper center",
             bbox_to_anchor=(0.5, -0.13), handlelength=1.6)
ax[0].set_xticks(xs); ax[0].set_xlim(-0.4, 2.5)
ax[0].set_xticklabels(["E_bagV", "E_specV", "E_cross"], fontsize=10)
ax[0].set_ylabel("position-residual per-trial $r$")
ax[0].set_title("(a) Fully causal $\\tau$ = 0", fontsize=9.5, loc="left")

# ---------- (b) latency curves ----------
taus = [0, 50, 100, 150]
ecv = [sub(cz, f"E_cross@{t}") for t in taus]; ebv = [sub(cz, f"E_bag@{t}") for t in taus]
ecm = [x.mean() for x in ecv]; ebm = [x.mean() for x in ebv]
ece = np.hstack([ci_err(x)[1] for x in ecv]); ebe = np.hstack([ci_err(x)[1] for x in ebv])
ax[1].errorbar(taus, ecm, yerr=ece, fmt="o-", color=C_CROSS, lw=1.5, capsize=2.5,
               label="E_cross (lag-only budget)")
ax[1].errorbar(taus, ebm, yerr=ebe, fmt="s-", color=C_BAG, lw=1.5, capsize=2.5,
               label="E_bag (lag-only budget)")
ax[1].fill_between(taus, ebm, ecm, color=C_CROSS, alpha=0.10)
fc_c, fc_b = sub(cp, "E_cross").mean(), sub(cp, "E_bag").mean()
ax[1].scatter([0], [fc_c], marker="D", s=55, color=C_CROSS, edgecolor="black", zorder=5,
               label="E_cross (end-to-end causal)")
ax[1].scatter([0], [fc_b], marker="D", s=55, color=C_BAG, edgecolor="black", zorder=5,
               label="E_bag (end-to-end causal)")
ax[1].plot([0, 0], [fc_b, fc_c], color="black", lw=1.0, ls=":")
ax[1].set_xticks(taus); ax[1].set_xlabel("output-latency budget $\\tau$ (ms)")
ax[1].set_ylabel("position-residual per-trial $r$")
ax[1].set_title("(b) Lag-only vs causal pipeline", fontsize=9.5, loc="left")
ax[1].legend(frameon=False, fontsize=9, loc="upper center", ncol=2,
             bbox_to_anchor=(0.5, -0.30), handlelength=1.4, columnspacing=1.0)

# ---------- (c) calibrated, native-unit metrics ----------
r2d, r2c = sub(sp, "cal_r2_dir"), sub(sp, "cal_r2_cross")
exd, exc = sub(sp, "end_r_dir"), sub(sp, "end_r_cross")
m2d, e2d = ci_err(r2d); m2c, e2c = ci_err(r2c)
ax[2].bar([0, 1], [m2d, m2c], 0.55, color=[C_BAG, C_CROSS], alpha=0.9)
ax[2].errorbar([0, 1], [m2d, m2c], yerr=np.hstack([e2d, e2c]), fmt="none", ecolor="0.25",
               capsize=3, lw=1.1, zorder=4)
ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["p direct", "E_cross"], fontsize=9.5)
ax[2].axhline(0, color="k", lw=0.8)
ax[2].set_ylabel("calibrated per-trial $R^2$ (native units)")
ax[2].set_title("(c) Native-unit reconstruction fails", fontsize=9.5, loc="left")

# ---------- (d) cross-session collapse ----------
w_ec = np.mean([w["E_cross"] for w in ws]); w_eb = np.mean([w["E_bag"] for w in ws])
c_ec = np.mean([c["E_cross"] for c in cs]); c_eb = np.mean([c["E_bag"] for c in cs])
xb = np.array([0, 1]); wd = 0.36
wcv = np.array([w["E_cross"] for w in ws]); wbv = np.array([w["E_bag"] for w in ws])
ccv = np.array([c["E_cross"] for c in cs]); cbv = np.array([c["E_bag"] for c in cs])
wcp = np.array([(wcv[2*i] + wcv[2*i+1]) / 2 for i in range(9)])
wbp = np.array([(wbv[2*i] + wbv[2*i+1]) / 2 for i in range(9)])
ax[3].bar(xb - wd/2, [w_ec, c_ec], wd, color=C_CROSS, alpha=0.9, label="E_cross")
ax[3].bar(xb + wd/2, [w_eb, c_eb], wd, color=C_BAG, alpha=0.9, label="E_bag")
for xx, vv in [(0 - wd/2, wcp), (1 - wd/2, ccv), (0 + wd/2, wbp), (1 + wd/2, cbv)]:
    mm, ee = ci_err(vv)
    ax[3].errorbar(xx, mm, yerr=ee, fmt="none", ecolor="0.25", capsize=2.5, lw=1.0, zorder=4)
ax[3].scatter([1 - wd/2]*len(cs), [c["E_cross"] for c in cs], s=12, color="black", zorder=3, alpha=0.6)
ax[3].set_xticks(xb); ax[3].set_xticklabels(["within-\nrun", "between-\nrun"], fontsize=9.5)
ax[3].set_ylabel("position-residual per-trial $r$"); ax[3].legend(frameon=False, fontsize=9.5)
ax[3].set_title("(d) No transfer between runs", fontsize=9.5, loc="left")
ax[3].axhline(0, color="0.7", lw=0.6)

fig.tight_layout(h_pad=4.2, w_pad=2.4)
fig.savefig(f"{PROJ}/figures/fig5_realtime.png", bbox_inches="tight")
export_submission(fig, list(ax), "fig5_realtime")
print("SAVED figures/fig5_realtime.png")
