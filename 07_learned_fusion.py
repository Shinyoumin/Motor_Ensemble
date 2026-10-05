"""
07 — Does a LEARNED fusion operator beat the fixed mean / performance-weight combiners?

Script 01 fused the three routes (a_route, v_route, p_direct -> distance) with two FIXED rules:
simple mean (E_mean) and train-performance weights (E_wt). Here we ask the engineering question:
can a *learned* meta-combiner do better, without test leakage? We keep the exact ridge routes of
script 01 and only change the combine step. Meta-learners are fit by NESTED cross-validation:
inside each outer training fold we generate out-of-fold route predictions, fit the combiner on
those, and apply it to the untouched outer test fold — so no route prediction ever trains its own
combiner weights.

Combiners (all in position residual space; routes z-scored per trial before combining, as in 01):
  mean       : (z_a+z_v+z_p)/3                               (baseline, = E_mean)
  perf_wt    : train-route-r weights                         (baseline, = E_wt)
  stack_ridge: OOF linear ridge on [z_a,z_v,z_p] -> position (learned static weights, may be <0)
  stack_nnls : OOF non-negative LS on [z_a,z_v,z_p]          (learned convex combination)
  stack_poly : OOF ridge on [z's, pairwise products, squares] (learned nonlinear fusion)

Metric + eval identical to script 01 (per-trial r in position residual space, 18 sessions,
paired Wilcoxon). Also reports each combiner vs the same-order bagging control E_bag (recomputed).

Run:  python 07_learned_fusion.py test
      python 07_learned_fusion.py full   -> results/ensemble_fusion.npz, results/fusion.md
"""
import sys, time, glob, mne, numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import nnls
from scipy.integrate import cumulative_trapezoid
from scipy.signal import detrend
mne.set_log_level('ERROR')

ROOT = "/mnt/hdd/MEG_Motor"; SIB = f"{ROOT}/motor_comparison_claude"; PROJ = f"{ROOT}/motor_ensemble_claude"
SF_DEC = 100.0; DX = 1.0/SF_DEC; LP = (0.5, 12.0); WIN = (3.5, 6.3)
LAGS_MS = np.arange(-100, 175, 50); ALPHA = 100.0
NFOLD = 5; NINNER = 4; N_BAG = 10; BAG_FRAC = 0.8; ORDERS = ['a', 'v', 'p']; RNG = np.random.default_rng(0)
_norm = lambda s: s.replace(' ', '')


def common_channels():
    common = None
    for fp in glob.glob(f"{ROOT}/00_preprocessed_data/Sub_*_grad_epoched.fif"):
        ep = mne.read_epochs(fp, preload=False)
        g = set(_norm(c) for c, t in zip(ep.ch_names, ep.get_channel_types()) if t == 'grad')
        common = g if common is None else (common & g)
    return sorted(common)

def load_meg(subj, sess, common):
    ep = mne.read_epochs(f"{ROOT}/00_preprocessed_data/Sub_{subj}_ses_{sess}_grad_epoched.fif", preload=True)
    ep.pick(mne.pick_types(ep.info, meg='grad'))
    ep.filter(LP[0], LP[1], picks='all', verbose='ERROR'); ep.resample(SF_DEC, verbose='ERROR')
    nm2idx = {_norm(c): i for i, c in enumerate(ep.ch_names)}
    return ep.get_data(picks=[nm2idx[c] for c in common]), ep.times

def build_design(X, t):
    lags = np.round(LAGS_MS/1000.0*SF_DEC).astype(int)
    tt = np.where((t >= WIN[0]) & (t <= WIN[1]))[0]
    feats = [X[:, :, np.clip(tt+L, 0, X.shape[2]-1)] for L in lags]
    D = np.stack(feats, axis=2).transpose(0, 3, 1, 2).reshape(X.shape[0], len(tt), -1)
    return D.astype(np.float64), t[tt]

def per_trial_r(pred, true):
    pc = pred - pred.mean(-1, keepdims=True); tc = true - true.mean(-1, keepdims=True)
    return (pc*tc).sum(-1) / (np.sqrt((pc**2).sum(-1)*(tc**2).sum(-1)) + 1e-30)
def ztime(x): return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-12)
def T1(x): return detrend(cumulative_trapezoid(x, dx=DX, axis=-1, initial=0.0), axis=-1, type='linear')
def T2(x): return T1(T1(x))
def to_pos(o, pred): return T2(pred) if o == 'a' else (T1(pred) if o == 'v' else pred)

def ridge_coef(Z, Y, alpha=ALPHA):
    A = Z.T @ Z; A[np.diag_indices_from(A)] += alpha
    return cho_solve(cho_factor(A, lower=True), Z.T @ Y)


def routes_for(Dte, coef):
    """Given a standardized design and per-order coefs, return z-scored routes in position space."""
    return {o: ztime(to_pos(o, Dte @ coef[o])) for o in ORDERS}

def poly_feats(F):
    """F:(N,3) -> [f, pairwise products, squares] (N,9)."""
    a, v, p = F[:, 0], F[:, 1], F[:, 2]
    return np.stack([a, v, p, a*v, a*p, v*p, a*a, v*v, p*p], axis=1)


def run_session(subj, sess, common, kin):
    X, t = load_meg(subj, sess, common); D, tw = build_design(X, t)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]) for o in ORDERS}
    ntr_all, nt, m = D.shape
    folds = np.array_split(RNG.permutation(ntr_all), NFOLD)
    keys = ['mean', 'perf_wt', 'stack_ridge', 'stack_nnls', 'stack_poly', 'E_bag']
    acc = {k: [] for k in keys}

    for f in range(NFOLD):
        te = folds[f]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != f])
        def standardize(idx_tr, idx_ev):
            Dmean = D[idx_tr].mean(0)
            flat = (D[idx_tr]-Dmean).reshape(-1, m); mu = flat.mean(0); sd = flat.std(0)+1e-9
            return ((D[idx_tr]-Dmean-mu)/sd), ((D[idx_ev]-Dmean-mu)/sd)
        # base ridge on full tr -> routes on te
        Ztr3, Zte = standardize(tr, te)
        Ym = {o: Ys[o][tr].mean(0) for o in ORDERS}
        Yr_tr = {o: Ys[o][tr]-Ym[o] for o in ORDERS}; Ytrue = Ys['p'][te]-Ym['p']
        Ystack = np.stack([Yr_tr[o].reshape(-1) for o in ORDERS], axis=1)
        B = ridge_coef(Ztr3.reshape(-1, m), Ystack); coef = {o: B[:, i] for i, o in enumerate(ORDERS)}
        rt_te = routes_for(Zte, coef)
        Fte = np.stack([rt_te[o].reshape(-1) for o in ORDERS], axis=1)          # (nte*nt,3)

        # fixed baselines
        acc['mean'].append(per_trial_r((rt_te['a']+rt_te['v']+rt_te['p'])/3, Ytrue).mean())
        w = np.array([max(0.0, per_trial_r(ztime(to_pos(o, (Ztr3 @ coef[o]))), Yr_tr['p']).mean()) for o in ORDERS])
        w = w/(w.sum()+1e-12)
        acc['perf_wt'].append(per_trial_r(w[0]*rt_te['a']+w[1]*rt_te['v']+w[2]*rt_te['p'], Ytrue).mean())

        # ----- OOF route preds on tr (nested) to train meta-combiners -----
        inner = np.array_split(RNG.permutation(len(tr)), NINNER)
        oof_F = np.zeros((len(tr), nt, 3)); oof_y = Yr_tr['p']
        for g in range(NINNER):
            iva = inner[g]; itr = np.concatenate([inner[k] for k in range(NINNER) if k != g])
            Zi_tr, Zi_va = standardize(tr[itr], tr[iva])
            Ymi = {o: Ys[o][tr[itr]].mean(0) for o in ORDERS}
            Ys_i = np.stack([(Ys[o][tr[itr]]-Ymi[o]).reshape(-1) for o in ORDERS], axis=1)
            Bi = ridge_coef(Zi_tr.reshape(-1, m), Ys_i); coefi = {o: Bi[:, j] for j, o in enumerate(ORDERS)}
            for j, o in enumerate(ORDERS):
                oof_F[iva, :, j] = ztime(to_pos(o, Zi_va @ coefi[o]))
        Foof = oof_F.reshape(-1, 3); yoof = oof_y.reshape(-1)

        # stack_ridge / nnls / poly fit on OOF, applied to te routes
        cr = ridge_coef(Foof, yoof, alpha=1.0)
        acc['stack_ridge'].append(per_trial_r((Fte @ cr).reshape(len(te), nt), Ytrue).mean())
        cn = nnls(Foof, yoof)[0]
        acc['stack_nnls'].append(per_trial_r((Fte @ cn).reshape(len(te), nt), Ytrue).mean())
        Pf = poly_feats(Foof); cp = ridge_coef(Pf, yoof, alpha=1.0)
        acc['stack_poly'].append(per_trial_r((poly_feats(Fte) @ cp).reshape(len(te), nt), Ytrue).mean())

        # same-order bagging control (recompute, as in 01)
        bag = []
        for b in range(N_BAG):
            idx = RNG.choice(len(tr), size=int(BAG_FRAC*len(tr)), replace=False)
            cb = ridge_coef(Ztr3[idx].reshape(-1, m), Yr_tr['p'][idx].reshape(-1))
            bag.append(ztime(Zte @ cb))
        acc['E_bag'].append(per_trial_r(np.mean(bag, 0), Ytrue).mean())
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _summ(recs, a, b):
    from scipy.stats import wilcoxon
    da = np.array([r[a] for r in recs]); db = np.array([r[b] for r in recs]); d = da-db
    try: p = wilcoxon(da, db).pvalue
    except Exception: p = float('nan')
    return da.mean(), db.mean(), d.mean(), int((d > 0).sum()), len(recs), p

CONTRASTS = [('stack_ridge', 'perf_wt'), ('stack_nnls', 'perf_wt'), ('stack_poly', 'perf_wt'),
             ('stack_ridge', 'mean'), ('stack_poly', 'stack_ridge'),
             ('stack_ridge', 'E_bag'), ('perf_wt', 'E_bag')]


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kin = dict(np.load(f"{SIB}/features/kinematics.npz", allow_pickle=True))
    common = common_channels(); print(f"[fusion] common grad {len(common)}", flush=True)
    if mode == "test":
        t0 = time.time(); r = run_session(1, 1, common, kin)
        print(f"[fusion] Sub1 ses1 ({time.time()-t0:.1f}s)")
        for k in ['mean', 'perf_wt', 'stack_ridge', 'stack_nnls', 'stack_poly', 'E_bag']:
            print(f"   {k:12s} {r[k]:+.4f}")
    else:
        recs = []
        for s in range(1, 10):
            for se in (1, 2):
                t0 = time.time(); r = run_session(s, se, common, kin)
                recs.append(dict(subj=s, sess=se, **r))
                print(f"[fusion] Sub{s} ses{se} ({time.time()-t0:.1f}s) "
                      f"mean={r['mean']:+.3f} wt={r['perf_wt']:+.3f} sRidge={r['stack_ridge']:+.3f} "
                      f"sPoly={r['stack_poly']:+.3f}", flush=True)
        np.savez(f"{PROJ}/results/ensemble_fusion.npz", recs=np.array(recs, dtype=object))
        lines = ["# Learned fusion operators (script 07, n=18)\n",
                 "Per-trial r in position residual space. Meta-combiners fit by nested OOF (no leakage).\n",
                 "| combiner | mean r |", "|---|---|"]
        for k in ['mean', 'perf_wt', 'stack_ridge', 'stack_nnls', 'stack_poly', 'E_bag']:
            lines.append(f"| {k} | {np.mean([r[k] for r in recs]):+.4f} |")
        lines += ["\n## Paired contrasts", "| contrast | A | B | diff | pos/n | p |", "|---|---|---|---|---|---|"]
        print("\nSAVED results/ensemble_fusion.npz\n=== group contrasts (fusion) ===")
        for a, b in CONTRASTS:
            ma, mb, md, pos, n, p = _summ(recs, a, b)
            print(f"  {a:12s} vs {b:12s}: {ma:+.4f} {mb:+.4f} d={md:+.4f} {pos}/{n} p={p:.4f}")
            lines.append(f"| {a} vs {b} | {ma:+.4f} | {mb:+.4f} | {md:+.4f} | {pos}/{n} | {p:.4f} |")
        with open(f"{PROJ}/results/fusion.md", "w") as fh: fh.write("\n".join(lines)+"\n")
        print("SAVED results/fusion.md")
