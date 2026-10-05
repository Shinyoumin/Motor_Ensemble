"""
40 — Manuscript Figure 3 (revision R4): architecture controls.

Replaces the old Fig. 2 (architecture benchmark) and the old Fig. 3 (six-panel seed stability,
which moves to the supplement). The R4 review's point: the manuscript's architecture panels showed
the attention gain without its parameter-matched control and showed the consistency term only at
the null weight, so the figure implied selective cross-order communication that the controls do not
support. Panels:
  (a) latent vs parameter-matched latent_ffn vs latent_att (script 28);
  (b) auxiliary supervision against its shuffled- and noise-target negative controls (script 29);
  (c) physics-consistency weight sweep, with the five-seed value at the optimum (script 35).

python 40_fig3_architecture.py  ->  figures/fig4_architecture.png
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
C_MAIN, C_CTRL, C_NEG = "#2f6fb3", "#7aa6d2", "#9aa0a6"

def recs(stem): return list(np.load(f"{PROJ}/results/{stem}.npz", allow_pickle=True)["recs"])
def col(rs, k): return np.array([r[k] for r in rs])
def subj(rs, k):
    s = np.array([r["subj"] for r in rs]); v = col(rs, k)
    return np.array([v[s == u].mean() for u in np.unique(s)])

fig, ax = plt.subplots(3, 1, figsize=(7.2, 8.6))

# ---------- (a) attention: capacity vs communication ----------
at = recs("attention_control")
keys = [("latent", "per-order\nlatent", C_NEG), ("latent_ffn", "+ per-token MLP\n(parameter-matched)", C_CTRL),
        ("latent_att", "+ cross-latent\nattention", C_MAIN)]
for i, (k, lab, c) in enumerate(keys):
    sm = subj(at, k)
    m, e = ci_err(sm)
    ax[0].bar(i, m, 0.62, color=c, alpha=0.9)
    ax[0].errorbar(i, m, yerr=e, fmt="none", ecolor="0.25", capsize=3, lw=1.1, zorder=4)
    ax[0].scatter(np.full(len(sm), i) + np.linspace(-0.15, 0.15, len(sm)), sm, s=11,
                  color="black", alpha=0.55, zorder=3, linewidth=0)
lat, ffn, att = subj(at, "latent"), subj(at, "latent_ffn"), subj(at, "latent_att")
ax[0].set_xticks(range(3)); ax[0].set_xticklabels([k[1] for k in keys], fontsize=9.5)
ax[0].set_ylim(min(lat.min(), ffn.min(), att.min()) - 0.02, max(att.max(), ffn.max()) + 0.05)
ax[0].set_ylabel("position-head per-trial $r$")
ax[0].set_title("(a) Added-block increment: capacity, not communication", fontsize=9.5, loc="left")

# ---------- (b) auxiliary supervision and its negative controls ----------
cs = recs("consistency_sweep")
keys_b = [("base", "no auxiliary", C_NEG), ("aux", "real kinematic\nauxiliary", C_MAIN),
          ("aux_shuf", "shuffled\nauxiliary", C_CTRL), ("aux_noise", "noise\nauxiliary", C_CTRL)]
for i, (k, lab, c) in enumerate(keys_b):
    sm = subj(cs, k)
    m, e = ci_err(sm)
    ax[1].bar(i, m, 0.62, color=c, alpha=0.9)
    ax[1].errorbar(i, m, yerr=e, fmt="none", ecolor="0.25", capsize=3, lw=1.1, zorder=4)
    ax[1].scatter(np.full(len(sm), i) + np.linspace(-0.15, 0.15, len(sm)), sm, s=11,
                  color="black", alpha=0.55, zorder=3, linewidth=0)
b, a_, sh, no = (subj(cs, k) for k in ("base", "aux", "aux_shuf", "aux_noise"))
ax[1].set_xticks(range(4)); ax[1].set_xticklabels([k[1] for k in keys_b], fontsize=9.5)
ax[1].set_ylim(min(b.min(), sh.min(), no.min()) - 0.02, max(a_.max(), b.max()) + 0.05)
ax[1].set_ylabel("position-head per-trial $r$")
ax[1].set_title("(b) The auxiliary gain needs real kinematics", fontsize=9.5, loc="left")

# ---------- (c) consistency-weight sweep ----------
lams = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]
aux = subj(cs, "aux")
dv = [subj(cs, f"cons{l}") - aux for l in lams]
d = [x.mean() for x in dv]
lo = [ci_err(x)[0] - ci_err(x)[1][0, 0] for x in dv]
hi = [ci_err(x)[0] + ci_err(x)[1][1, 0] for x in dv]
ax[2].plot(lams, d, "o-", color=C_MAIN, lw=1.4, ms=5, label="single-seed sweep")
ax[2].fill_between(lams, lo, hi, color=C_MAIN, alpha=0.18, lw=0, label="95% CI (participants)")
sd = recs("consistency_lambda_seeds")
seeds = [f"_s{i}" for i in range(5)]
d01 = np.mean([(subj(sd, f"cons0.1{s}") - subj(sd, f"aux{s}")).mean() for s in seeds])
sd01 = np.std([(subj(sd, f"cons0.1{s}") - subj(sd, f"aux{s}")).mean() for s in seeds], ddof=1)
d03 = np.mean([(subj(sd, f"cons0.3{s}") - subj(sd, f"aux{s}")).mean() for s in seeds])
ax[2].errorbar([0.1], [d01], yerr=[sd01], fmt="D", color="black", ms=7, capsize=3,
               label="five-seed mean ± SD")
ax[2].errorbar([0.3], [d03], yerr=[0], fmt="D", color="0.55", ms=6)
ax[2].set_xscale("log"); ax[2].axhline(0, color="k", lw=0.6)
ax[2].set_xlabel("physics-consistency weight $\\lambda_c$")
ax[2].set_ylabel("cons − aux ($\\Delta r$, position head)")
ax[2].set_title("(c) Beneficial only at a small tuned weight", fontsize=9.5, loc="left")
ax[2].legend(frameon=False, fontsize=9.5, loc="lower left")

fig.tight_layout()
fig.savefig(f"{PROJ}/figures/fig4_architecture.png", bbox_inches="tight")
export_submission(fig, list(ax), "fig4_architecture")
print("SAVED figures/fig4_architecture.png")
