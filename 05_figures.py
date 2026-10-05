"""
05 — Summary figure for the ensemble study.

a) Distance-reconstruction accuracy of the three routes + ensembles + MTL (ridge backbone).
b) Backbone robustness: E_cross vs E_bag across backbones (ridge / ssl / rand / lstm).
c) Single-task (p_direct) vs multi-task shared bottleneck (MTL) across backbones — STL wins.
d) Route importance: per-lag profile (velocity back-loaded), from script 02.

Run:  python 05_figures.py
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"
def load(stem):
    try: return np.load(f"{PROJ}/results/{stem}.npz", allow_pickle=True)
    except Exception: return None
def recs(stem):
    d = load(stem); return list(d['recs']) if d is not None else None
def col(rs, k): return np.array([r[k] for r in rs])

C = {'a': '#4C72B0', 'v': '#DD8452', 'p': '#55A868', 'cross': '#8172B3', 'bag': '#937860', 'mtl': '#C44E52'}

fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))

# ---- a) routes + ensembles + MTL (ridge) ----
r0 = recs('ensemble_distance')
order = ['a_route', 'v_route', 'p_direct', 'E_mean', 'E_wt', 'E_bag', 'mtl_r1']
labs = ['a→dist', 'v→dist', 'p direct', 'E_cross\n(mean)', 'E_cross\n(wt)', 'E_bag', 'MTL\n(rank1)']
cols = [C['a'], C['v'], C['p'], C['cross'], C['cross'], C['bag'], C['mtl']]
mu = [col(r0, k).mean() for k in order]; sd = [col(r0, k).std()/np.sqrt(len(r0)) for k in order]
ax[0].bar(range(len(order)), mu, yerr=sd, color=cols, capsize=3)
ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels(labs, fontsize=8)
ax[0].set_ylabel('distance per-trial r (residual)'); ax[0].set_title('(a) routes & fusion — ridge')
ax[0].axhline(0, color='k', lw=0.6)

# ---- b) backbone robustness: E_cross vs E_bag ----
backbones = [('ridge', 'ensemble_distance'), ('ssl', 'ensemble_ssl'),
             ('rand', 'ensemble_rand'), ('lstm', 'ensemble_lstm')]
present = [(nm, st) for nm, st in backbones if recs(st) is not None]
x = np.arange(len(present)); w = 0.38
ec = [col(recs(st), 'E_wt').mean() for _, st in present]
eb = [col(recs(st), 'E_bag').mean() for _, st in present]
ece = [col(recs(st), 'E_wt').std()/np.sqrt(len(recs(st))) for _, st in present]
ebe = [col(recs(st), 'E_bag').std()/np.sqrt(len(recs(st))) for _, st in present]
ax[1].bar(x-w/2, ec, w, yerr=ece, label='E_cross', color=C['cross'], capsize=3)
ax[1].bar(x+w/2, eb, w, yerr=ebe, label='E_bag', color=C['bag'], capsize=3)
ax[1].set_xticks(x); ax[1].set_xticklabels([nm for nm, _ in present])
ax[1].set_ylabel('distance per-trial r'); ax[1].set_title('(b) fusion > bagging, all backbones')
ax[1].legend(fontsize=8); ax[1].axhline(0, color='k', lw=0.6)

# ---- c) single-task vs multi-task (STL p_direct vs MTL) ----
present_m = [(nm, st) for nm, st in backbones if recs(st) is not None and 'mtl_r1' in recs(st)[0]]
xm = np.arange(len(present_m))
stl = [col(recs(st), 'p_direct').mean() for _, st in present_m]
mtl = [col(recs(st), 'mtl_r1').mean() for _, st in present_m]
ax[2].bar(xm-w/2, stl, w, label='single-task (p direct)', color=C['p'], capsize=3)
ax[2].bar(xm+w/2, mtl, w, label='multi-task (shared bottleneck)', color=C['mtl'], capsize=3)
ax[2].set_xticks(xm); ax[2].set_xticklabels([nm for nm, _ in present_m])
ax[2].set_ylabel('distance per-trial r'); ax[2].set_title('(c) single-task > multi-task')
ax[2].legend(fontsize=8); ax[2].axhline(0, color='k', lw=0.6)

# ---- d) route importance lag profile ----
ri = load('route_importance')
if ri is not None:
    L = ri['lag']; lags = ri['lags_ms']
    Lg = L.mean(0); Lg = Lg / Lg.sum(1, keepdims=True)
    for i, o in enumerate(['a', 'v', 'p']):
        ax[3].plot(lags, Lg[i], 'o-', color=C[o], label={'a': 'accel', 'v': 'veloc', 'p': 'posit'}[o])
    ax[3].set_xlabel('MEG lag (ms)'); ax[3].set_ylabel('normalized importance')
    ax[3].set_title('(d) velocity read out later'); ax[3].legend(fontsize=8)

fig.tight_layout()
fig.savefig(f"{PROJ}/figures/fig1_ensemble.png", dpi=200, bbox_inches='tight')
print("saved figures/fig1_ensemble.png")
