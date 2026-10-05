"""
11 — Aggregate the multi-seed stability runs (script 10) and decide whether the 06 attention
gain is seed-robust.

Reads every results/seed_stability/seed_{s}.npz, then for the key contrasts reports:
  * per-seed mean diff, pos/n sessions, Wilcoxon p   (is each individual seed's 18/18 luck?)
  * across-seed stability: mean +/- sd of the per-seed mean-diff, and the min pos/n over seeds
  * session-level robustness: for each of the 18 sessions, the mean diff over seeds and the
    fraction of seeds in which that session is positive  (which sessions, if any, ever flip)

Headline contrast: latent_att_p vs latent_p (attention helps the position head).
Also: latent_att_p vs stl_p, and latent_att_fuse vs latent_att_p (output fusion still adds).

Writes results/seed_stability.md and figures/fig3_seed_stability.png.

Run: python 11_seed_analysis.py
"""
import sys, glob, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"

# Two presets: the static-MLP arm (script 10, default) and the sequential-BiLSTM arm (script 13).
PRESETS = {
    'mlp': dict(subdir='seed_stability', md='seed_stability.md', fig='fig3_seed_stability.png',
                title='attention-MTL (static MLP backbone, scripts 06/10)',
                contrasts=[('latent_att_p', 'latent_p'), ('latent_att_p', 'stl_p'),
                           ('latent_att_fuse', 'latent_att_p')]),
    'seq': dict(subdir='seq_seed_stability', md='seq_seed_stability.md', fig='fig4_seq_seed_stability.png',
                title='attention-MTL (sequential BiLSTM backbone, scripts 12/13)',
                contrasts=[('seq_latent_att_p', 'seq_latent_p'), ('seq_latent_att_p', 'seq_stl_p'),
                           ('seq_latent_att_fuse', 'seq_latent_att_p')]),
}
PRESET = sys.argv[1] if len(sys.argv) > 1 else 'mlp'
CFG = PRESETS[PRESET]
OUTDIR = f"{PROJ}/results/{CFG['subdir']}"
CONTRASTS = CFG['contrasts']
SESS_KEY = lambda r: (int(r['subj']), int(r['sess']))


def load_all():
    seeds = {}
    for fp in sorted(glob.glob(f"{OUTDIR}/seed_*.npz")):
        s = int(os.path.basename(fp).split('_')[1].split('.')[0])
        recs = list(np.load(fp, allow_pickle=True)['recs'])
        seeds[s] = {SESS_KEY(r): r for r in recs}      # keyed by session for alignment
    return seeds


def main():
    seeds = load_all()
    if not seeds:
        print("no seed_*.npz yet"); return
    seed_ids = sorted(seeds)
    # common, consistently-ordered session list present in every seed
    sess_list = sorted(set.intersection(*[set(d) for d in seeds.values()]))
    n_sess = len(sess_list); n_seed = len(seed_ids)
    print(f"[seed_analysis] {n_seed} seeds {seed_ids}, {n_sess} sessions each")

    lines = [f"# Multi-seed stability — {CFG['title']}\n",
             f"{n_seed} seeds ({seed_ids}), {n_sess} sessions each. Fair position-residual per-trial r; "
             f"per-session folds are seed-independent so the only varied factor is the training seed "
             f"(weight init + minibatch order).\n"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ci, (a, b) in enumerate(CONTRASTS):
        # diff matrix (n_seed, n_sess)
        M = np.array([[seeds[s][k][a] - seeds[s][k][b] for k in sess_list] for s in seed_ids])
        per_seed_mean = M.mean(1)
        per_seed_pos = (M > 0).sum(1)
        per_seed_p = []
        for row_a, row_b in [([seeds[s][k][a] for k in sess_list],
                              [seeds[s][k][b] for k in sess_list]) for s in seed_ids]:
            try: per_seed_p.append(wilcoxon(row_a, row_b).pvalue)
            except Exception: per_seed_p.append(float('nan'))
        sess_mean = M.mean(0)                    # mean diff per session across seeds
        sess_frac_pos = (M > 0).mean(0)          # fraction of seeds positive per session
        # PRIMARY test: average out the seed, then Wilcoxon over the 18 sessions (9 subj x 2).
        # (Pooling all n_seed*n_sess diffs is anti-conservative — the seeds share data+folds and
        #  are not independent — so it is reported below as descriptive only, not as the headline p.
        #  NB: the 18 sessions are 9 subjects x 2, so they are not fully independent either — this is
        #  subject-level pseudo-replication; effective n is closer to 9. See manuscript Limitations.)
        try: sess_p = wilcoxon(sess_mean).pvalue
        except Exception: sess_p = float('nan')
        # SUBJECT-LEVEL test (fully independent units): average the seed-averaged session diffs within
        # subject -> one diff per subject, then Wilcoxon over the 9 subjects. Removes the subject-level
        # pseudo-replication of the 18-session test.
        subj_ids = sorted(set(k[0] for k in sess_list))
        subj_diff = np.array([np.mean([sess_mean[j] for j, k in enumerate(sess_list) if k[0] == s])
                              for s in subj_ids])
        try: subj_p = wilcoxon(subj_diff).pvalue
        except Exception: subj_p = float('nan')

        lines += [f"\n## {a} vs {b}",
                  f"- across-seed mean diff = **{per_seed_mean.mean():+.4f}** "
                  f"(sd over seeds {per_seed_mean.std():.4f}, range "
                  f"{per_seed_mean.min():+.4f}..{per_seed_mean.max():+.4f})",
                  f"- **PRIMARY (seed-averaged, 18 sessions = 9 subj x 2): mean {sess_mean.mean():+.4f}, "
                  f"{int((sess_mean>0).sum())}/{n_sess} sessions positive, Wilcoxon p={sess_p:.2e}**",
                  f"- SUBJECT-LEVEL (9 independent subjects, seed+session averaged): mean "
                  f"{subj_diff.mean():+.4f}, {int((subj_diff>0).sum())}/{len(subj_ids)} subjects "
                  f"positive, Wilcoxon p={subj_p:.2e}",
                  f"- per-seed sessions-positive: min **{per_seed_pos.min()}/{n_sess}**, "
                  f"max {per_seed_pos.max()}/{n_sess}",
                  f"- sessions positive in ALL seeds: **{int((sess_frac_pos==1).sum())}/{n_sess}**; "
                  f"positive in >=half of seeds: {int((sess_frac_pos>=0.5).sum())}/{n_sess}",
                  "",
                  "| seed | mean diff | pos/n | wilcoxon p (per-seed, n=18) |", "|---|---|---|---|"]
        for i, s in enumerate(seed_ids):
            lines.append(f"| {s} | {per_seed_mean[i]:+.4f} | {per_seed_pos[i]}/{n_sess} | {per_seed_p[i]:.2e} |")
        # descriptive only (non-independent): pooled across all seed x session diffs
        pooled = M.reshape(-1)
        try: pooled_p = wilcoxon(pooled).pvalue
        except Exception: pooled_p = float('nan')
        lines.append(f"\n_descriptive only (non-independent — seeds share data/folds):_ pooled "
                     f"{n_seed*n_sess} seed x session diffs: mean {pooled.mean():+.4f}, "
                     f"pos {int((pooled>0).sum())}/{len(pooled)}, Wilcoxon p={pooled_p:.2e}")

        # ---- figure panel: per-session mean +/- seed spread, sorted ----
        ax = axes[ci]
        order = np.argsort(sess_mean)
        x = np.arange(n_sess)
        ax.axhline(0, color='0.6', lw=0.8)
        ax.errorbar(x, sess_mean[order], yerr=M.std(0)[order], fmt='o', ms=4,
                    color='#2c7fb8', ecolor='#a6bddb', capsize=2)
        ax.set_title(f"{a}\n- {b}", fontsize=9)
        ax.set_xlabel("session (sorted by mean diff)")
        ax.set_ylabel("$\\Delta$ per-trial r" if ci == 0 else "")
        ax.text(0.03, 0.93, f"mean {per_seed_mean.mean():+.3f}\nmin {per_seed_pos.min()}/{n_sess} pos",
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.8))
    fig.suptitle(f"Multi-seed stability — {CFG['title']} (per-session mean +/- seed spread)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figpath = f"{PROJ}/figures/{CFG['fig']}"
    fig.savefig(figpath, dpi=130); print(f"SAVED {figpath}")

    mdpath = f"{PROJ}/results/{CFG['md']}"
    with open(mdpath, "w") as fh: fh.write("\n".join(lines) + "\n")
    print(f"SAVED {mdpath}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
