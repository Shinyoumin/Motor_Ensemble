"""
31 — Combined multi-seed stability figure (replaces the separate fig3 / fig4).

Journal figure budget: main text is capped at 4 figures + 3 tables, so the two
multi-seed panels (static MLP, scripts 06/10; sequential BiLSTM, scripts 12/13)
are merged into a single 2 x 3 figure:

    row 1 (a-c)  MLP     : latent_att-latent | latent_att-stl | fuse-latent_att
    row 2 (d-f)  BiLSTM  : same three contrasts on the sequential backbone

Numbers are recomputed from the per-seed npz archives (identical statistics to
scripts/11_seed_analysis.py), so the figure cannot drift from the tables.

Run: python 31_seed_figure_combined.py   ->  figures/fig3_seed_stability_combined.png
"""
import glob, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"

ROWS = [
    dict(subdir='seed_stability', tag='static MLP backbone',
         contrasts=[('latent_att_p', 'latent_p', 'latent_att $-$ latent'),
                    ('latent_att_p', 'stl_p', 'latent_att $-$ stl'),
                    ('latent_att_fuse', 'latent_att_p', 'output fusion $-$ latent_att')]),
    dict(subdir='seq_seed_stability', tag='sequential BiLSTM backbone',
         contrasts=[('seq_latent_att_p', 'seq_latent_p', 'latent_att $-$ latent'),
                    ('seq_latent_att_p', 'seq_stl_p', 'latent_att $-$ stl'),
                    ('seq_latent_att_fuse', 'seq_latent_att_p', 'output fusion $-$ latent_att')]),
]
SESS_KEY = lambda r: (int(r['subj']), int(r['sess']))


def load(subdir):
    seeds = {}
    for fp in sorted(glob.glob(f"{PROJ}/results/{subdir}/seed_*.npz")):
        s = int(os.path.basename(fp).split('_')[1].split('.')[0])
        seeds[s] = {SESS_KEY(r): r for r in np.load(fp, allow_pickle=True)['recs']}
    return seeds


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
    panel = iter("abcdef")
    log = []
    for ri, row in enumerate(ROWS):
        seeds = load(row['subdir'])
        seed_ids = sorted(seeds)
        sess = sorted(set.intersection(*[set(d) for d in seeds.values()]))
        n = len(sess)
        for ci, (a, b, lab) in enumerate(row['contrasts']):
            M = np.array([[seeds[s][k][a] - seeds[s][k][b] for k in sess] for s in seed_ids])
            per_seed_mean, per_seed_pos = M.mean(1), (M > 0).sum(1)
            sess_mean = M.mean(0)
            subj_ids = sorted(set(k[0] for k in sess))
            subj_d = np.array([np.mean([sess_mean[j] for j, k in enumerate(sess) if k[0] == s])
                               for s in subj_ids])
            p_sess, p_subj = wilcoxon(sess_mean).pvalue, wilcoxon(subj_d).pvalue
            log.append(f"[{row['subdir']}] {a}-{b}: mean {per_seed_mean.mean():+.4f} "
                       f"(sd {per_seed_mean.std():.4f}, range {per_seed_mean.min():+.4f}.."
                       f"{per_seed_mean.max():+.4f}), {int((sess_mean>0).sum())}/{n} sess "
                       f"p={p_sess:.2e}, {int((subj_d>0).sum())}/{len(subj_ids)} subj "
                       f"p={p_subj:.2e}, min per-seed pos {per_seed_pos.min()}/{n}")

            ax = axes[ri, ci]
            order = np.argsort(sess_mean)
            ax.axhline(0, color='0.6', lw=0.8)
            ax.errorbar(np.arange(n), sess_mean[order], yerr=M.std(0)[order], fmt='o', ms=4,
                        color='#2c7fb8', ecolor='#a6bddb', capsize=2)
            ax.set_title(lab, fontsize=10)
            ax.set_xlabel("session (sorted by mean $\\Delta$)", fontsize=9)
            ax.set_xticks(np.arange(0, n, 3))
            if ci == 0:
                ax.set_ylabel("$\\Delta$ per-trial $r$", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.text(-0.13, 1.06, next(panel), transform=ax.transAxes,
                    fontsize=13, fontweight='bold', va='top')
        axes[ri, 0].annotate(row['tag'], xy=(-0.24, 0.5), xycoords='axes fraction',
                             rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')
    fig.suptitle("per-session mean $\\pm$ across-seed spread, 6 seeds", fontsize=11)
    fig.tight_layout(rect=[0.02, 0, 1, 0.95])
    out = f"{PROJ}/figures/fig3_seed_stability_combined.png"
    fig.savefig(out, dpi=140)
    print("\n".join(log))
    print(f"SAVED {out}")


if __name__ == "__main__":
    main()
