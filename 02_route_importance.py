"""
02 — Where does each order's distance-relevant decoder draw its information?

For the three single-order decoders (used as routes to distance in script 01), compute analytic
feature importance in the same fair residual space: squared standardized ridge coefficient summed
per SENSOR (over lags) and per LAG (over sensors). Cross-order Pearson r of the importance profiles
answers "do the orders rely on the same sensors / the same latencies?".

Expectation (from the sibling study's Track C): sensor topography DIFFUSE / shared across orders;
the dissociation carried by TIME-LAG (velocity back-loaded). We report honestly, no cortical-region
claim (sensor space, no source localisation).

Run:  python 02_route_importance.py [test|full]
"""
import sys, time, glob, mne, numpy as np
from scipy.linalg import cho_factor, cho_solve
mne.set_log_level('ERROR')

ROOT = "/mnt/hdd/MEG_Motor"; SIB = f"{ROOT}/motor_comparison_claude"; PROJ = f"{ROOT}/motor_ensemble_claude"
SF_DEC = 100.0; LP = (0.5, 12.0); WIN = (3.5, 6.3)
LAGS_MS = np.arange(-100, 175, 50); ALPHA = 100.0; NFOLD = 5
ORDERS = ['a', 'v', 'p']; RNG = np.random.default_rng(0)
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

NCH = None; NLAG = len(LAGS_MS)

def session_importance(subj, sess, common, kin):
    X, t = load_meg(subj, sess, common); D, tw = build_design(X, t)
    global NCH; NCH = X.shape[1]
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]) for o in ORDERS}
    ntr, nt, m = D.shape
    folds = np.array_split(RNG.permutation(ntr), NFOLD)
    imp = np.zeros((3, m))                                   # accumulate squared standardized coef
    for f in range(NFOLD):
        te = folds[f]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != f])
        Dmean = D[tr].mean(0); flat = (D[tr]-Dmean).reshape(-1, m)
        mur = flat.mean(0); sdr = flat.std(0)+1e-9; Z = (flat-mur)/sdr
        A = Z.T @ Z; A[np.diag_indices_from(A)] += ALPHA; fac = cho_factor(A, lower=True)
        for i, o in enumerate(ORDERS):
            Yr = (Ys[o][tr] - Ys[o][tr].mean(0)).reshape(-1)
            coef = cho_solve(fac, Z.T @ Yr)
            imp[i] += coef**2                                # standardized-input coef^2 = importance
    imp /= NFOLD
    # reshape m = nchan*nlag (channel outer, lag inner)
    impr = imp.reshape(3, NCH, NLAG)
    sensor = impr.sum(2)                                     # (3, nchan)
    lag = impr.sum(1)                                        # (3, nlag)
    return sensor, lag

def _corr(a, b): return float(np.corrcoef(a, b)[0, 1])

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kin = dict(np.load(f"{SIB}/features/kinematics.npz", allow_pickle=True))
    common = common_channels(); print(f"common grad {len(common)}", flush=True)
    sess_list = [(1, 1)] if mode == "test" else [(s, se) for s in range(1, 10) for se in (1, 2)]
    S, L = [], []
    for s, se in sess_list:
        t0 = time.time(); sensor, lag = session_importance(s, se, common, kin)
        S.append(sensor); L.append(lag)
        print(f"Sub{s} ses{se} ({time.time()-t0:.1f}s)", flush=True)
    S = np.array(S); L = np.array(L)                         # (nsess,3,nchan/nlag)
    if mode == "full":
        np.savez(f"{PROJ}/results/route_importance.npz", sensor=S, lag=L, lags_ms=LAGS_MS)
        # cross-order correlations per session, then group mean
        def xcorr(P):                                        # P:(nsess,3,K)
            out = {}
            for (i, j, nm) in [(0, 2, 'a_p'), (0, 1, 'a_v'), (1, 2, 'v_p')]:
                out[nm] = np.array([_corr(P[k, i], P[k, j]) for k in range(P.shape[0])])
            return out
        sc = xcorr(S); lc = xcorr(L)
        lines = ["# Route importance — where each order's distance decoder draws information (n=18)\n",
                 "Cross-order Pearson r of per-order importance profiles (mean±sd over 18 sessions).\n",
                 "| grouping | corr(a,p) | corr(a,v) | corr(v,p) |", "|---|---|---|---|",
                 f"| sensor topography | {sc['a_p'].mean():+.3f}±{sc['a_p'].std():.2f} | "
                 f"{sc['a_v'].mean():+.3f}±{sc['a_v'].std():.2f} | {sc['v_p'].mean():+.3f}±{sc['v_p'].std():.2f} |",
                 f"| time-lag profile | {lc['a_p'].mean():+.3f}±{lc['a_p'].std():.2f} | "
                 f"{lc['a_v'].mean():+.3f}±{lc['a_v'].std():.2f} | {lc['v_p'].mean():+.3f}±{lc['v_p'].std():.2f} |",
                 "",
                 "Per-lag group-mean importance (normalized within order):", ""]
        Lg = L.mean(0); Lg = Lg / Lg.sum(1, keepdims=True)
        lines.append("| lag (ms) | " + " | ".join(ORDERS) + " |")
        lines.append("|---|" + "---|"*3)
        for li, lm in enumerate(LAGS_MS):
            lines.append(f"| {lm:+d} | " + " | ".join(f"{Lg[o,li]:.3f}" for o in range(3)) + " |")
        open(f"{PROJ}/results/route_importance.md", "w").write("\n".join(lines))
        print("\n".join(lines)); print("\nSAVED results/route_importance.{npz,md}")
    else:
        print("sensor shape", S.shape, "lag shape", L.shape)
        print("lag profiles (norm):", (L[0]/L[0].sum(1, keepdims=True)).round(3))
