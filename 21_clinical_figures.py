"""
21 — Control-arm figure (manuscript Fig. 4): (a) bandwidth decomposition, (b) real-time latency curve,
(c) cross-session collapse. Reads results/{spectral_control,causal_latency,deployment}.npz.

python 21_clinical_figures.py  ->  figures/fig5_clinical.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 200})
C_BAG, C_CROSS, C_SPEC = "#9aa0a6", "#2f6fb3", "#c2603a"   # grey / blue / warm

sp = list(np.load(f"{PROJ}/results/spectral_control.npz", allow_pickle=True)['recs'])
cz = list(np.load(f"{PROJ}/results/causal_latency.npz", allow_pickle=True)['recs'])
dep = np.load(f"{PROJ}/results/deployment.npz", allow_pickle=True)
cs = list(dep['cs']); ws = list(dep['ws'])

fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))

# ---- (a) bandwidth decomposition: paired E_bag -> E_spec -> E_cross ----
eb = np.array([r['E_bag'] for r in sp]); es = np.array([r['E_spec'] for r in sp]); ec = np.array([r['E_cross'] for r in sp])
xs = [0, 1, 2]
for i in range(len(sp)):
    ax[0].plot(xs, [eb[i], es[i], ec[i]], color="0.8", lw=0.6, zorder=1)
for x, v, c in [(0, eb, C_BAG), (1, es, C_SPEC), (2, ec, C_CROSS)]:
    ax[0].scatter([x]*len(v), v, s=14, color=c, zorder=3, edgecolor="white", linewidth=0.4)
    ax[0].scatter([x], [v.mean()], s=90, color=c, zorder=4, edgecolor="black", linewidth=1.1, marker="D")
ax[0].set_xticks(xs); ax[0].set_xticklabels(["E_bag\n(same-order)", "E_spec\n(bandwidth-only)", "E_cross\n(cross-order)"])
ax[0].set_ylabel("distance reconstruction  (per-trial r)")
ax[0].set_title("(a) The gain is bandwidth, not neural", fontsize=9, loc="left")
ax[0].annotate(f"+0.042", (0.5, max(es.mean(), eb.mean())+0.02), ha="center", fontsize=8, color=C_SPEC)
ax[0].annotate("E_cross − E_spec\n= −0.004 (n.s.)", (2.02, ec.mean()), fontsize=7.5, color=C_CROSS, va="center")
ax[0].plot([1, 2], [es.mean(), ec.mean()], color="black", lw=0.8, ls=":")

# ---- (b) latency curve ----
taus = [0, 50, 100, 150]
ecm = [np.mean([r[f'E_cross@{t}'] for r in cz]) for t in taus]
ecs = [np.std([r[f'E_cross@{t}'] for r in cz])/np.sqrt(len(cz)) for t in taus]
ebm = [np.mean([r[f'E_bag@{t}'] for r in cz]) for t in taus]
ebs = [np.std([r[f'E_bag@{t}'] for r in cz])/np.sqrt(len(cz)) for t in taus]
ax[1].errorbar(taus, ecm, yerr=ecs, color=C_CROSS, marker="o", capsize=2, label="E_cross", lw=1.5)
ax[1].errorbar(taus, ebm, yerr=ebs, color=C_BAG, marker="s", capsize=2, label="E_bag", lw=1.5)
ax[1].fill_between(taus, ebm, ecm, color=C_CROSS, alpha=0.10)
for t, a, b in zip(taus, ecm, ebm):
    ax[1].annotate(f"+{a-b:.3f}", (t, a+0.006), ha="center", fontsize=7, color=C_CROSS)
ax[1].axvline(0, color="0.7", ls="--", lw=0.8)
ax[1].set_xticks(taus); ax[1].set_xlabel("output-latency budget τ (ms)  ·  τ=0 = strictly causal")
ax[1].set_ylabel("per-trial r"); ax[1].legend(frameon=False, fontsize=8, loc="lower right")
ax[1].set_title("(b) Margin survives real-time (causal) use", fontsize=9, loc="left")

# ---- (c) cross-session collapse ----
w_ec = np.mean([w['E_cross'] for w in ws]); w_eb = np.mean([w['E_bag'] for w in ws])
c_ec = np.mean([c['E_cross'] for c in cs]); c_eb = np.mean([c['E_bag'] for c in cs])
xb = np.array([0, 1]); wd = 0.36
ax[2].bar(xb-wd/2, [w_ec, c_ec], wd, color=C_CROSS, label="E_cross")
ax[2].bar(xb+wd/2, [w_eb, c_eb], wd, color=C_BAG, label="E_bag")
# per-subject cross-session E_cross dots
ax[2].scatter([1-wd/2]*len(cs), [c['E_cross'] for c in cs], s=12, color="black", zorder=3, alpha=0.6)
ax[2].set_xticks(xb); ax[2].set_xticklabels(["within-\nsession", "cross-\nsession"])
ax[2].set_ylabel("per-trial r"); ax[2].legend(frameon=False, fontsize=8)
ax[2].set_title("(c) No cross-session transfer (~18%)", fontsize=9, loc="left")
ax[2].annotate("18% retained", (1, c_ec+0.02), ha="center", fontsize=7.5)
ax[2].axhline(0, color="0.7", lw=0.6)

fig.suptitle("Cross-order fusion improves MEG movement reconstruction by spectral smoothing, "
             "not separable codes — and its clinical bounds", fontsize=9.5, y=1.02, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(f"{PROJ}/figures/fig5_clinical.png", bbox_inches="tight")
print("SAVED figures/fig5_clinical.png")
