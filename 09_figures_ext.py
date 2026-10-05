"""
09 — Aggregate the model-development arms (scripts 06/07/08) into one figure + a methods table.

Reads whichever of these exist and builds figures/fig2_methods.png (one panel per available arm)
and results/methods_bench.md:
  results/ensemble_attn_mtl.npz     (06) architectural MTL: stl / shared / latent / latent+attn
  results/ensemble_fusion.npz       (07) learned fusion: mean / perf_wt / stack_{ridge,nnls,poly}
  results/ensemble_consistency.npz  (08) consistency decoder: base / aux / cons
Baselines for context come from results/ensemble_distance.npz (script 01: p_direct, E_wt, E_bag).

Run:  python 09_figures_ext.py
"""
import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"; R = f"{PROJ}/results"
def load(name):
    fp = f"{R}/{name}"
    if not os.path.exists(fp): return None
    return list(np.load(fp, allow_pickle=True)['recs'])
def col(recs, k): return np.array([r[k] for r in recs])
def sem(x): return x.std(ddof=1)/np.sqrt(len(x))
def star(recs, a, b):
    try: p = wilcoxon(col(recs, a), col(recs, b)).pvalue
    except Exception: return ''
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'ns'

base = load('ensemble_distance.npz')
attn = load('ensemble_attn_mtl.npz')
fus = load('ensemble_fusion.npz')
cons = load('ensemble_consistency.npz')

panels = []
if attn:
    panels.append(('attn', 'MTL architecture (06)', attn,
                   [('stl_p', 'STL'), ('shared_p', 'shared\nbottleneck'), ('latent_p', 'per-order\nlatent'),
                    ('latent_att_p', 'latent\n+attention')], 'latent_att_p', 'latent_p',
                   'latent+attention vs per-order latent'))
if fus:
    panels.append(('fus', 'Learned fusion (07)', fus,
                   [('E_bag', 'bag'), ('mean', 'mean'), ('perf_wt', 'perf-wt'), ('stack_ridge', 'stack\nridge'),
                    ('stack_nnls', 'stack\nnnls'), ('stack_poly', 'stack\npoly')], 'stack_ridge', 'perf_wt',
                   'stack-ridge vs perf-wt'))
if cons:
    panels.append(('cons', 'Consistency decoder (08)', cons,
                   [('base_p', 'base'), ('aux_p', 'aux'), ('cons_p', 'consistency'),
                    ('aux_fuse', 'aux\n+fuse'), ('cons_fuse', 'cons\n+fuse')], 'cons_p', 'aux_p',
                   'consistency vs aux (at $\\lambda_c$=0.3)'))

npan = len(panels)
if npan == 0:
    print("no result files yet"); raise SystemExit

fig, axes = plt.subplots(1, npan, figsize=(4.6*npan, 4.2)); axes = np.atleast_1d(axes)
COL = {'STL': '#4C72B0', 'shared': '#C44E52', 'latent': '#55A868', 'stack': '#8172B3', 'cons': '#CCB974'}
for ax, (tag, title, recs, items, hi_a, hi_b, annot) in zip(axes, panels):
    labels = [lab for _, lab in items]; means = [col(recs, k).mean() for k, _ in items]
    errs = [sem(col(recs, k)) for k, _ in items]
    colors = []
    for k, _ in items:
        c = '#999999'
        if 'stack' in k or 'latent_att' in k or k == 'cons_p' or 'fuse' in k: c = '#8172B3'
        if k in ('stl_p', 'mean', 'base_p'): c = '#4C72B0'
        if 'shared' in k or k == 'E_bag': c = '#C44E52'
        colors.append(c)
    x = np.arange(len(items))
    ax.bar(x, means, yerr=errs, color=colors, capsize=3, edgecolor='k', linewidth=0.4)
    if base is not None:
        for yv, lab, ls in [(col(base, 'p_direct').mean(), 'p_direct', ':'),
                            (col(base, 'E_wt').mean(), 'E_cross (01)', '--')]:
            ax.axhline(yv, color='gray', ls=ls, lw=1)
            ax.text(len(items)-0.4, yv, lab, fontsize=7, va='bottom', ha='right', color='gray')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_title(title, fontsize=11)
    ax.set_ylabel('distance per-trial r (residual)')
    s = star(recs, hi_a, hi_b)
    ax.text(0.5, 0.99, f"{annot}: {s}", transform=ax.transAxes,
            ha='center', va='top', fontsize=8.5, color='#333')
plt.tight_layout(); plt.savefig(f"{PROJ}/figures/fig2_methods.png", dpi=150, bbox_inches='tight')
print(f"SAVED figures/fig2_methods.png ({npan} panels)")

# ---- methods_bench.md ----
def block(title, recs, order):
    out = [f"### {title}", "| variant | mean r | sem |", "|---|---|---|"]
    for k in order:
        c = col(recs, k); out.append(f"| {k} | {c.mean():+.4f} | {sem(c):.4f} |")
    return "\n".join(out)
lines = ["# Model-development benchmark — cross-order MEG distance decoding (scripts 06-08)\n",
         "Per-trial r in the fair position-residual space (n=18 sessions). All arms share the pipeline",
         "and metric of script 01.\n"]
if base:
    lines.append(block("Baselines (script 01)", base, ['p_direct', 'a_route', 'E_wt', 'E_bag', 'mtl_r1']) + "\n")
if attn:
    lines.append(block("06 MTL architecture — position head", attn,
                       ['stl_p', 'shared_p', 'latent_p', 'latent_att_p']))
    lines.append(block("06 — own output fusion", attn,
                       ['stl_fuse', 'shared_fuse', 'latent_fuse', 'latent_att_fuse']) + "\n")
if fus:
    lines.append(block("07 learned fusion", fus,
                       ['mean', 'perf_wt', 'stack_ridge', 'stack_nnls', 'stack_poly', 'E_bag']) + "\n")
if cons:
    lines.append(block("08 consistency decoder", cons,
                       ['base_p', 'aux_p', 'cons_p', 'aux_fuse', 'cons_fuse']) + "\n")
with open(f"{R}/methods_bench.md", "w") as fh: fh.write("\n".join(lines)+"\n")
print("SAVED results/methods_bench.md")
