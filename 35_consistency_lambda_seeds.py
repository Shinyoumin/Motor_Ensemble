"""
35 — Multi-seed confirmation of the physics-consistency optimum (lambda_c = 0.1).

PAPER.md §3.5 reports that the consistency loss is strongly weight-dependent and apparently BENEFICIAL
at lambda_c = 0.1 (+0.036, 17/18), but flags it explicitly as "a single-seed, non-monotonic peak ...
an indication requiring multi-seed confirmation rather than a headline". This script supplies that
confirmation, using the project's established multi-seed convention (script 10/11): folds are held
seed-independent (seeded by 1000*subj + sess) so the training seed is the ONLY varied factor, the
primary test averages the difference over seeds and applies a paired Wilcoxon over the 18 sessions,
and the subject-level Wilcoxon over 9 participants is reported alongside.

Configurations per fold and seed: aux (lambda_c = 0), cons at lambda_c = 0.1 (the claimed optimum) and
0.3 (the originally-tested weight, expected null). Everything else — trunk, heads, integration graph,
optimiser, epochs — is script 08/29's, imported unchanged.

Run (.venv_deep):  python 35_consistency_lambda_seeds.py <test|full> [gpu]
"""
import sys, time, importlib.util, numpy as np
from scipy.stats import wilcoxon
import torch

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"
_s = importlib.util.spec_from_file_location("c29", f"{PROJ}/scripts/29_consistency_sweep.py")
c29 = importlib.util.module_from_spec(_s); _s.loader.exec_module(c29)
m08 = c29.m08
ORDERS, NFOLD = c29.ORDERS, c29.NFOLD
per_trial_r, to_pos = c29.per_trial_r, c29.to_pos

SEEDS = [0, 1, 2, 3, 4]
LAMS = [0.1, 0.3]
SIB = c29.SIB


def run_session(subj, sess, common, kin):
    X, t = m08.load_meg(subj, sess, common); D, tw = m08.build_design(X, t)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]).astype(np.float32)
          for o in ORDERS}
    ntr, nt, m = D.shape
    folds = np.array_split(np.random.default_rng(1000*subj + sess).permutation(ntr), NFOLD)  # seed-independent
    acc = {f"{cfg}_s{sd}": [] for sd in SEEDS for cfg in ['aux'] + [f'cons{l}' for l in LAMS]}

    for fi in range(NFOLD):
        te = folds[fi]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != fi])
        Dmean = D[tr].mean(0); flat = (D[tr]-Dmean).reshape(-1, m); mu = flat.mean(0); sd_ = flat.std(0)+1e-9
        Xtr = (D[tr]-Dmean-mu)/sd_; Xte = (D[te]-Dmean-mu)/sd_
        Ym = {o: Ys[o][tr].mean(0) for o in ORDERS}; Yr_tr = {o: Ys[o][tr]-Ym[o] for o in ORDERS}
        Ysd = {o: float(Yr_tr[o].std()+1e-9) for o in ORDERS}
        Ystd = np.stack([Yr_tr[o]/Ysd[o] for o in ORDERS], axis=-1).astype(np.float32)
        Ytrue = Ys['p'][te]-Ym['p']

        def evalp(net):
            pr = m08.predict(net, Xte)
            return per_trial_r(to_pos('p', pr[:, :, 2]*Ysd['p']), Ytrue).mean()

        for seed in SEEDS:
            acc[f"aux_s{seed}"].append(evalp(c29.train_cfg(Xtr, Ystd, Ysd, lam_c=0.0, aux=True, seed=seed)))
            for lc in LAMS:
                acc[f"cons{lc}_s{seed}"].append(
                    evalp(c29.train_cfg(Xtr, Ystd, Ysd, lam_c=lc, aux=True, seed=seed)))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def stats(recs, a, b):
    da = np.array([r[a] for r in recs]); db = np.array([r[b] for r in recs]); d = da-db
    subj = np.array([r['subj'] for r in recs])
    ds = np.array([d[subj == s].mean() for s in np.unique(subj)])
    ps = wilcoxon(da, db).pvalue if np.any(d) else 1.0
    pj = wilcoxon(ds).pvalue if np.any(ds) else 1.0
    return d, d.mean(), int((d > 0).sum()), ps, int((ds > 0).sum()), pj


def report(recs):
    L = ["# Multi-seed confirmation of the consistency optimum — script 35 (n = 18 sessions / 9 subjects)\n",
         f"Seeds {SEEDS} (weight init + minibatch order); folds seed-independent (1000*subj+sess), so the "
         "training seed is the only varied factor. Position-head per-trial r. Compare with the single-seed "
         "sweep in `consistency_sweep.md` (script 29): cons0.1 − aux = +0.0356 (17/18), cons0.3 − aux = "
         "+0.0019 (9/18).\n"]
    for lc in LAMS:
        per_seed = []
        for sd in SEEDS:
            d, mn, pos, ps, jp, pj = stats(recs, f'cons{lc}_s{sd}', f'aux_s{sd}')
            per_seed.append((sd, mn, pos, ps, jp, pj))
        # seed-averaged (primary)
        for r in recs:
            r[f'cons{lc}_avg'] = float(np.mean([r[f'cons{lc}_s{sd}'] for sd in SEEDS]))
            r['aux_avg'] = float(np.mean([r[f'aux_s{sd}'] for sd in SEEDS]))
        d, mn, pos, ps, jp, pj = stats(recs, f'cons{lc}_avg', 'aux_avg')
        means = np.array([x[1] for x in per_seed])
        L += [f"## lambda_c = {lc}: cons − aux", "",
              f"- across-seed mean **{means.mean():+.4f}** (sd {means.std():.4f}, range "
              f"{means.min():+.4f}..{means.max():+.4f})",
              f"- **PRIMARY (seed-averaged, 18 sessions): {mn:+.4f}, {pos}/18 positive, Wilcoxon p = {ps:.2e}**",
              f"- SUBJECT-LEVEL (9 participants): {jp}/9 positive, p = {pj:.2e}",
              f"- per-seed sign range: {min(x[2] for x in per_seed)}/18 .. {max(x[2] for x in per_seed)}/18; "
              f"seeds individually significant (p<0.05): {sum(1 for x in per_seed if x[3] < 0.05)}/{len(SEEDS)}",
              "", "| seed | mean diff | pos/18 | session p | subj pos/9 | subj p |", "|---|---|---|---|---|---|"]
        for sd, mn_, pos_, ps_, jp_, pj_ in per_seed:
            L.append(f"| {sd} | {mn_:+.4f} | {pos_}/18 | {ps_:.2e} | {jp_}/9 | {pj_:.2e} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kin = dict(np.load(f"{SIB}/features/kinematics.npz", allow_pickle=True))
    common = m08.common_channels(); print(f"common grad: {len(common)} dev {c29.DEV}", flush=True)
    if mode == "test":
        t0 = time.time(); r = run_session(1, 1, common, kin)
        print(f"Sub1 ses1 ({time.time()-t0:.1f}s)")
        for lc in LAMS:
            for sd in SEEDS:
                print(f"  lam={lc} seed={sd}: cons {r[f'cons{lc}_s{sd}']:+.4f} aux {r[f'aux_s{sd}']:+.4f} "
                      f"diff {r[f'cons{lc}_s{sd}']-r[f'aux_s{sd}']:+.4f}")
    else:
        recs = []
        for s in range(1, 10):
            for se in (1, 2):
                t0 = time.time(); r = run_session(s, se, common, kin)
                recs.append(dict(subj=s, sess=se, **r))
                d1 = np.mean([r[f'cons0.1_s{sd}']-r[f'aux_s{sd}'] for sd in SEEDS])
                print(f"Sub{s} ses{se} ({time.time()-t0:.1f}s) cons0.1−aux(seed-avg) {d1:+.4f}", flush=True)
        np.savez(f"{PROJ}/results/consistency_lambda_seeds.npz", recs=np.array(recs, dtype=object))
        md = report(recs)
        open(f"{PROJ}/results/consistency_lambda_seeds.md", "w").write(md)
        print(md); print("SAVED results/consistency_lambda_seeds.{npz,md}")
