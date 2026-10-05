"""
01 — Reconstruct DISTANCE (position) from MEG via three kinematic-order routes, and test
whether fusing them beats (a) the best single route and (b) a same-order bagging control.

Everything is in the SAME fair single-trial residual space as
../motor_comparison_claude/scripts/31 (mean-removed MEG -> mean-removed target). Position is
the common evaluation target; three routes reach it:
  p_direct : ridge -> position_resid                      (decode distance directly)
  a_route  : ridge -> accel_resid,   then T2 = detrend.cumtrapz twice  -> position space
  v_route  : ridge -> veloc_resid,   then T1 = detrend.cumtrapz once   -> position space
(the multi-output ridge decouples across outputs, so p_direct == position column of the joint fit;
routes differ only in which order is decoded then transformed).

Metric: per-trial Pearson r in position residual space (shape correlation; invariant to the
integration constant and linear drift). NOT naive full-trace accuracy (that is the smoothness/
stereotypy confound the sibling study dismantled); a context naive number is printed separately.

Comparisons (paired across 18 sessions):
  - routes: r(a_route), r(v_route), r(p_direct)
  - E_cross (fuse 3 routes; mean + train-perf-weighted) vs best single, and vs E_bag
  - E_bag  : B position-direct decoders on random train subsamples (same-order variance control)
  - MTL vs STL: rank-1/2 reduced-rank (shared bottleneck) position output vs p_direct

Run:  python 01_ensemble_distance.py test     # sub1 ses1 sanity + timing
      python 01_ensemble_distance.py full      # 9x2 -> results/ensemble_distance.npz
"""
import sys, time, glob, mne, numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.integrate import cumulative_trapezoid
from scipy.signal import detrend
mne.set_log_level('ERROR')

ROOT = "/mnt/hdd/MEG_Motor"
SIB = f"{ROOT}/motor_comparison_claude"
PROJ = f"{ROOT}/motor_ensemble_claude"
SF_DEC = 100.0; DX = 1.0/SF_DEC
LP = (0.5, 12.0); WIN = (3.5, 6.3)
LAGS_MS = np.arange(-100, 175, 50)
ALPHA = 100.0
NFOLD = 5
N_BAG = 10; BAG_FRAC = 0.8
ORDERS = ['a', 'v', 'p']
RNG = np.random.default_rng(0)

_norm = lambda s: s.replace(' ', '')

def common_channels():
    common = None
    for fp in glob.glob(f"{ROOT}/00_preprocessed_data/Sub_*_grad_epoched.fif"):
        ep = mne.read_epochs(fp, preload=False)
        g = set(_norm(c) for c, t in zip(ep.ch_names, ep.get_channel_types()) if t == 'grad')
        common = g if common is None else (common & g)
    return sorted(common)

def load_meg(subj, sess, common, band=LP):
    ep = mne.read_epochs(f"{ROOT}/00_preprocessed_data/Sub_{subj}_ses_{sess}_grad_epoched.fif", preload=True)
    ep.pick(mne.pick_types(ep.info, meg='grad'))
    ep.filter(band[0], band[1], picks='all', verbose='ERROR')
    ep.resample(SF_DEC, verbose='ERROR')
    nm2idx = {_norm(c): i for i, c in enumerate(ep.ch_names)}
    X = ep.get_data(picks=[nm2idx[c] for c in common])
    return X, ep.times

def build_design(X, t):
    lags = np.round(LAGS_MS/1000.0*SF_DEC).astype(int)
    tt = np.where((t >= WIN[0]) & (t <= WIN[1]))[0]
    feats = [X[:, :, np.clip(tt+L, 0, X.shape[2]-1)] for L in lags]
    D = np.stack(feats, axis=2).transpose(0, 3, 1, 2).reshape(X.shape[0], len(tt), -1)
    return D.astype(np.float64), t[tt]

def per_trial_r(pred, true):
    pc = pred - pred.mean(-1, keepdims=True); tc = true - true.mean(-1, keepdims=True)
    return (pc*tc).sum(-1) / (np.sqrt((pc**2).sum(-1)*(tc**2).sum(-1)) + 1e-30)

def ztime(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-12)

def T1(x):
    return detrend(cumulative_trapezoid(x, dx=DX, axis=-1, initial=0.0), axis=-1, type='linear')
def T2(x):
    return T1(T1(x))

def ridge_coef(Z, Y):
    """Z:(n,m) standardized design residual; Y:(n,) or (n,k). Returns coef (m,) or (m,k)."""
    A = Z.T @ Z; A[np.diag_indices_from(A)] += ALPHA
    fac = cho_factor(A, lower=True)
    return cho_solve(fac, Z.T @ Y)

def to_pos(order, pred):
    """Transform a per-trial residual prediction of `order` into position space."""
    if order == 'a': return T2(pred)
    if order == 'v': return T1(pred)
    return pred                                   # 'p' is already position


def run_session(subj, sess, common, kin, X=None, t=None):
    if X is None: X, t = load_meg(subj, sess, common)
    D, tw = build_design(X, t)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]) for o in ORDERS}
    ntr_all, nt, m = D.shape
    # Unified fold convention (revision R1): deterministic per-session seed, identical to the
    # control-arm scripts, so every analysis in the paper sees the same splits.
    frng = np.random.default_rng(1000 * subj + sess)
    brng = np.random.default_rng(7_000 + 100 * subj + sess)
    folds = np.array_split(frng.permutation(ntr_all), NFOLD)

    keys = ['a_route', 'v_route', 'p_direct', 'E_mean', 'E_wt', 'E_bag',
            'mtl_r1', 'mtl_r2', 'naive_pdir']
    acc = {k: [] for k in keys}

    for f in range(NFOLD):
        te = folds[f]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != f])
        Dmean = D[tr].mean(0)
        flat_r = (D[tr] - Dmean).reshape(-1, m); mur = flat_r.mean(0); sdr = flat_r.std(0) + 1e-9
        Ztr = ((flat_r - mur) / sdr)                      # (ntr*nt, m)
        Ztr3 = Ztr.reshape(len(tr), nt, m)
        Zte = ((D[te] - Dmean - mur) / sdr)               # (nte, nt, m)

        Ym = {o: Ys[o][tr].mean(0) for o in ORDERS}
        Yr_tr = {o: (Ys[o][tr] - Ym[o]) for o in ORDERS}
        Yr_te = {o: (Ys[o][te] - Ym[o]) for o in ORDERS}
        Ytrue = Yr_te['p']                                # position residual = eval target

        # single multi-output ridge (decouples -> per-order coef)
        Ystack = np.stack([Yr_tr[o].reshape(-1) for o in ORDERS], axis=1)   # (ntr*nt,3)
        B = ridge_coef(Ztr, Ystack)                       # (m,3)
        coef = {o: B[:, i] for i, o in enumerate(ORDERS)}

        # ---- routes -> position space ----
        route_te = {o: to_pos(o, Zte @ coef[o]) for o in ORDERS}           # each (nte,nt)
        route_tr = {o: to_pos(o, Ztr3 @ coef[o]) for o in ORDERS}
        acc['a_route'].append(per_trial_r(route_te['a'], Ytrue).mean())
        acc['v_route'].append(per_trial_r(route_te['v'], Ytrue).mean())
        acc['p_direct'].append(per_trial_r(route_te['p'], Ytrue).mean())

        # ---- cross-order ensembles (z-scored per trial so scales are comparable) ----
        zte = {o: ztime(route_te[o]) for o in ORDERS}
        E_mean = (zte['a'] + zte['v'] + zte['p']) / 3.0
        acc['E_mean'].append(per_trial_r(E_mean, Ytrue).mean())
        # performance weights from TRAIN route accuracy (no test leakage)
        w = np.array([max(0.0, per_trial_r(route_tr[o], Yr_tr['p']).mean()) for o in ORDERS])
        w = w / (w.sum() + 1e-12)
        E_wt = w[0]*zte['a'] + w[1]*zte['v'] + w[2]*zte['p']
        acc['E_wt'].append(per_trial_r(E_wt, Ytrue).mean())

        # ---- same-order bagging control: B position-direct decoders on train subsamples ----
        bag_preds = []
        for b in range(N_BAG):
            idx = brng.choice(len(tr), size=int(BAG_FRAC*len(tr)), replace=False)
            Zb = Ztr3[idx].reshape(-1, m); Yb = Yr_tr['p'][idx].reshape(-1)
            cb = ridge_coef(Zb, Yb)
            bag_preds.append(ztime(Zte @ cb))
        E_bag = np.mean(bag_preds, axis=0)
        acc['E_bag'].append(per_trial_r(E_bag, Ytrue).mean())

        # ---- MTL (reduced-rank shared bottleneck) position output vs STL ----
        Yhat = Ztr @ B; sc = Yhat.std(0) + 1e-9
        Vt = np.linalg.svd(Yhat/sc, full_matrices=False)[2]
        Zte_flat = Zte.reshape(-1, m)
        for k, key in [(1, 'mtl_r1'), (2, 'mtl_r2')]:
            Vk = Vt[:k].T @ Vt[:k]
            Bk = B @ Vk
            pred_k = (Zte_flat @ Bk[:, 2]).reshape(len(te), nt)   # position column
            acc[key].append(per_trial_r(pred_k, Ytrue).mean())

        # ---- context: naive full-trace r for p_direct (confounded regime) ----
        pred_full = Ym['p'] + route_te['p']
        acc['naive_pdir'].append(per_trial_r(pred_full, Ys['p'][te]).mean())

    return {k: float(np.mean(v)) for k, v in acc.items()}


def _summ(recs, a, b):
    from scipy.stats import wilcoxon
    da = np.array([r[a] for r in recs]); db = np.array([r[b] for r in recs])
    d = da - db; pos = int((d > 0).sum())
    try: p = wilcoxon(da, db).pvalue
    except Exception: p = float('nan')
    return da.mean(), db.mean(), d.mean(), pos, len(recs), p


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kin = dict(np.load(f"{SIB}/features/kinematics.npz", allow_pickle=True))
    common = common_channels(); print(f"common grad: {len(common)}", flush=True)

    if mode == "test":
        t0 = time.time(); r = run_session(1, 1, common, kin); dt = time.time()-t0
        print(f"\nSub1 ses1 ({dt:.1f}s)  — per-trial r in POSITION residual space")
        for k in ['a_route', 'v_route', 'p_direct', 'E_mean', 'E_wt', 'E_bag', 'mtl_r1', 'mtl_r2']:
            print(f"   {k:10s} {r[k]:+.4f}")
        print(f"   (context) naive full-trace r p_direct = {r['naive_pdir']:+.3f}")
    elif mode == "full":
        recs = []
        for s in range(1, 10):
            for se in (1, 2):
                t0 = time.time(); r = run_session(s, se, common, kin)
                recs.append(dict(subj=s, sess=se, **r))
                print(f"Sub{s} ses{se} ({time.time()-t0:.1f}s) "
                      f"a={r['a_route']:+.3f} v={r['v_route']:+.3f} p={r['p_direct']:+.3f} "
                      f"Ecross(wt)={r['E_wt']:+.3f} Ebag={r['E_bag']:+.3f}", flush=True)
        np.savez(f"{PROJ}/results/ensemble_distance.npz", recs=np.array(recs, dtype=object))
        print("SAVED results/ensemble_distance.npz")
        # quick group readout
        print("\n=== group paired contrasts (mean_a, mean_b, mean_diff, pos/n, p) ===")
        for a, b in [('a_route', 'p_direct'), ('a_route', 'v_route'),
                     ('E_wt', 'p_direct'), ('E_wt', 'a_route'), ('E_wt', 'E_bag'),
                     ('E_mean', 'E_bag'), ('mtl_r1', 'p_direct'), ('mtl_r2', 'p_direct')]:
            ma, mb, md, pos, n, p = _summ(recs, a, b)
            print(f"  {a:9s} vs {b:9s}: {ma:+.4f} {mb:+.4f}  d={md:+.4f}  {pos}/{n}  p={p:.4f}")
