"""
26 — Re-verify the §3.7 "dissociation is time-lag, not topography" claim against two reviewer threats
(reviewer point 8): (i) ridge COEFFICIENTS are not activation patterns in correlated MEG; (ii) the
lag-profile dissociation may be a TARGET-derivative/spectrum artifact, not neural timing.

Three analyses, all in script 01's fair residual space, ridge, meggpt venv (no re-training of models
beyond the analytic ridge fits):

(A) HAUFE activation patterns. For each order o, convert the ridge filter W_o into the activation
    pattern A_o = Cov(Z) W_o (Haufe et al. 2014; scalar output so the pred-covariance term is 1). Report
    cross-order Pearson r of the SENSOR and LAG patterns from BOTH the raw squared-weight importance
    (script 02) and the Haufe patterns, so we can see whether "shared topography / dissociated lag"
    survives the correction. Split-half (odd/even trial) stability of each pattern is reported.

(B) PERMUTATION importance per lag — model-agnostic, no weight interpretation. For each lag, permute
    that lag's features across TEST trials and measure the drop in each route's per-trial r. The
    per-lag importance profile and its cross-order correlations test the dissociation without weights.

(C) SHARED-SIGNAL SPECTRUM SURROGATE — the decisive control for threat (ii). a, v, p are exact temporal
    derivatives of one trajectory, so decoding them emphasises different temporal-frequency bands and
    could produce a lag dissociation with NO order-specific neural timing. We build a surrogate MEG that
    contains ONE common neural timing: each pseudo-sensor = a random projection of the trial's true
    position delayed by a SINGLE common lag tau0, plus noise. We then decode the REAL a/v/p (= s'', s',
    s) from this surrogate and compute the per-lag importance. If the a/v/p lag profiles still dissociate
    despite the single built-in timing, the real dissociation is (at least partly) a target-spectrum
    artifact; if they collapse to one profile, the real dissociation needs a neural-timing explanation.

Run: /mnt/hdd/MEG_Motor/analysis/.venv_meggpt/bin/python scripts/26_haufe_surrogate.py [test|full]
"""
import sys, time, importlib.util, numpy as np
from scipy.linalg import cho_factor, cho_solve
mne_spec = importlib.util.spec_from_file_location(
    "ens01", "/mnt/hdd/MEG_Motor/motor_ensemble_claude/scripts/01_ensemble_distance.py")
ens01 = importlib.util.module_from_spec(mne_spec); mne_spec.loader.exec_module(ens01)
common_channels = ens01.common_channels; load_meg = ens01.load_meg; build_design = ens01.build_design
per_trial_r = ens01.per_trial_r; to_pos = ens01.to_pos

ROOT = "/mnt/hdd/MEG_Motor"; SIB = f"{ROOT}/motor_comparison_claude"; PROJ = f"{ROOT}/motor_ensemble_claude"
ORDERS = ['a', 'v', 'p']; ALPHA = 100.0; NFOLD = 5
LAGS_MS = ens01.LAGS_MS; NLAG = len(LAGS_MS)
_corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])


def fit_ridge_multi(Z, Ystack):
    A = Z.T @ Z; A[np.diag_indices_from(A)] += ALPHA
    return cho_solve(cho_factor(A, lower=True), Z.T @ Ystack)     # (m,3)


def session(subj, sess, common, kin, do_surrogate=True, rng=None):
    rng = rng or np.random.default_rng(0)
    X, t = load_meg(subj, sess, common)
    D, tw = build_design(X, t)
    NCH = X.shape[1]; ntr, nt, m = D.shape
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]) for o in ORDERS}

    # ---- fit on all trials (standardized design) for pattern analysis ----
    Dmean = D.mean(0); flat = (D - Dmean).reshape(-1, m); mur = flat.mean(0); sdr = flat.std(0) + 1e-9
    Z = (flat - mur) / sdr
    Ym = {o: Ys[o].mean(0) for o in ORDERS}
    Ystack = np.stack([(Ys[o] - Ym[o]).reshape(-1) for o in ORDERS], axis=1)
    W = fit_ridge_multi(Z, Ystack)                                # (m,3) filters
    Czz = (Z.T @ Z) / Z.shape[0]                                  # design covariance
    Hauf = Czz @ W                                               # (m,3) activation patterns

    def profiles(M):                                            # M:(m,3) -> sensor(3,NCH), lag(3,NLAG)
        Mr = (M.T).reshape(3, NCH, NLAG)                        # weights laid out channel-outer, lag-inner
        return (Mr**2).sum(2), (Mr**2).sum(1)
    sen_raw, lag_raw = profiles(W)                              # raw squared-weight importance (script 02)
    sen_hf, lag_hf = profiles(Hauf)                            # Haufe patterns

    def xcorr(P):
        return {nm: _corr(P[i], P[j]) for i, j, nm in [(0,2,'a_p'), (0,1,'a_v'), (1,2,'v_p')]}

    # split-half stability of Haufe lag/sensor patterns
    odd = np.arange(0, ntr, 2); even = np.arange(1, ntr, 2)
    def half_patt(idx):
        Zi = ((D[idx]-Dmean).reshape(-1, m)-mur)/sdr
        Yi = np.stack([(Ys[o][idx]-Ym[o]).reshape(-1) for o in ORDERS], 1)
        Wi = fit_ridge_multi(Zi, Yi); Hi = ((Zi.T@Zi)/Zi.shape[0])@Wi
        return profiles(Hi)
    (so, lo), (se, le) = half_patt(odd), half_patt(even)
    stab_lag = {ORDERS[i]: _corr(lo[i], le[i]) for i in range(3)}
    stab_sen = {ORDERS[i]: _corr(so[i], se[i]) for i in range(3)}

    # ---- (B) permutation importance per lag (CV, model-agnostic) ----
    folds = np.array_split(rng.permutation(ntr), NFOLD)
    permimp = {o: np.zeros(NLAG) for o in ORDERS}
    base_r = {o: [] for o in ORDERS}
    for fi in range(NFOLD):
        te = folds[fi]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != fi])
        dm = D[tr].mean(0); fl = (D[tr]-dm).reshape(-1, m); mu = fl.mean(0); sd = fl.std(0)+1e-9
        Ztr = (fl-mu)/sd
        Yt = np.stack([(Ys[o][tr]-Ys[o][tr].mean(0)).reshape(-1) for o in ORDERS], 1)
        Wf = fit_ridge_multi(Ztr, Yt); cf = {o: Wf[:, i] for i, o in enumerate(ORDERS)}
        Ym_te = {o: Ys[o][tr].mean(0) for o in ORDERS}
        Ytrue = Ys['p'][te]-Ym_te['p']
        Dte = D[te]
        def route_r(Dblock, o):
            Zt = ((Dblock-dm).reshape(len(te), nt, m)-mu)/sd
            return per_trial_r(to_pos(o, Zt @ cf[o]), Ytrue).mean()
        for o in ORDERS:
            base_r[o].append(route_r(Dte, o))
        # permute each lag block across test trials
        Dre = Dte.reshape(len(te), nt, NCH, NLAG)
        for li in range(NLAG):
            Dp = Dre.copy(); Dp[:, :, :, li] = Dp[rng.permutation(len(te)), :, :, li]
            Dp = Dp.reshape(len(te), nt, m)
            for o in ORDERS:
                permimp[o][li] += max(0.0, base_r[o][-1] - route_r(Dp, o))
    permimp = {o: permimp[o]/NFOLD for o in ORDERS}
    permlag_norm = {o: permimp[o]/(permimp[o].sum()+1e-30) for o in ORDERS}
    permxc = {nm: _corr(permlag_norm[a], permlag_norm[b]) for a, b, nm in [('a','p','a_p'),('a','v','a_v'),('v','p','v_p')]}

    # ---- (C) shared-signal spectrum surrogate ----
    surr_lagxc = None; surr_lag = None
    if do_surrogate:
        tau0 = 0                                                # single common neural lag (samples)
        s = Ys['p']                                             # shared movement = true position (ntr,nt_full? use tw-length)
        # build full-time position for lag windowing: reinterp over t (full epoch times)
        sfull = np.stack([np.interp(t, kt, kin['p'][sel][i]) for i in range(sel.sum())])  # (ntr, T)
        Tt = sfull.shape[1]
        proj = rng.standard_normal((NCH,))                     # per-sensor random gains
        Xs = sfull[:, None, :]*proj[None, :, None]             # (ntr,NCH,T) common-timed copy of position
        Xs = Xs + 0.5*Xs.std()*rng.standard_normal(Xs.shape)   # additive sensor noise
        Ds, tws = build_design(Xs, t)
        # decode real a/v/p (= s'',s',s) from surrogate; per-lag permutation importance
        simp = {o: np.zeros(NLAG) for o in ORDERS}
        for fi in range(NFOLD):
            te = folds[fi]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != fi])
            dm = Ds[tr].mean(0); fl = (Ds[tr]-dm).reshape(-1, m); mu = fl.mean(0); sd = fl.std(0)+1e-9
            Ztr = (fl-mu)/sd
            Yt = np.stack([(Ys[o][tr]-Ys[o][tr].mean(0)).reshape(-1) for o in ORDERS], 1)
            Wf = fit_ridge_multi(Ztr, Yt); cf = {o: Wf[:, i] for i, o in enumerate(ORDERS)}
            Ym_te = {o: Ys[o][tr].mean(0) for o in ORDERS}; Ytrue = Ys['p'][te]-Ym_te['p']
            Dre = Ds[te].reshape(len(te), nt, NCH, NLAG)
            def rr(Dblock, o):
                Zt = ((Dblock-dm).reshape(len(te), nt, m)-mu)/sd
                return per_trial_r(to_pos(o, Zt @ cf[o]), Ytrue).mean()
            b = {o: rr(Ds[te], o) for o in ORDERS}
            for li in range(NLAG):
                Dp = Dre.copy(); Dp[:, :, :, li] = Dp[rng.permutation(len(te)), :, :, li]
                Dp = Dp.reshape(len(te), nt, m)
                for o in ORDERS: simp[o][li] += max(0.0, b[o]-rr(Dp, o))
        simp = {o: simp[o]/NFOLD for o in ORDERS}
        surr_lag = {o: simp[o]/(simp[o].sum()+1e-30) for o in ORDERS}
        surr_lagxc = {nm: _corr(surr_lag[a], surr_lag[b]) for a, b, nm in [('a','p','a_p'),('a','v','a_v'),('v','p','v_p')]}

    return dict(sen_raw_xc=xcorr(sen_raw), lag_raw_xc=xcorr(lag_raw),
                sen_hf_xc=xcorr(sen_hf), lag_hf_xc=xcorr(lag_hf),
                stab_lag=stab_lag, stab_sen=stab_sen,
                permlag=permlag_norm, permxc=permxc,
                surr_lag=surr_lag, surr_lagxc=surr_lagxc)


def _mean_xc(recs, key):
    out = {}
    for nm in ['a_p', 'a_v', 'v_p']:
        v = np.array([r[key][nm] for r in recs]); out[nm] = (v.mean(), v.std())
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kin = dict(np.load(f"{SIB}/features/kinematics.npz", allow_pickle=True))
    common = common_channels(); print(f"common grad {len(common)}", flush=True)
    sess_list = [(1, 1)] if mode == "test" else [(s, se) for s in range(1, 10) for se in (1, 2)]
    recs = []
    for s, se in sess_list:
        t0 = time.time(); r = session(s, se, common, kin, rng=np.random.default_rng(100*s+se))
        recs.append(r); print(f"Sub{s} ses{se} ({time.time()-t0:.1f}s)", flush=True)
    if mode == "test":
        r = recs[0]
        print("\nRAW lag xcorr:", {k: round(v, 2) for k, v in r['lag_raw_xc'].items()})
        print("HAUFE lag xcorr:", {k: round(v, 2) for k, v in r['lag_hf_xc'].items()})
        print("RAW sensor xcorr:", {k: round(v, 2) for k, v in r['sen_raw_xc'].items()})
        print("HAUFE sensor xcorr:", {k: round(v, 2) for k, v in r['sen_hf_xc'].items()})
        print("perm-imp lag xcorr:", {k: round(v, 2) for k, v in r['permxc'].items()})
        print("perm lag profiles:", {o: np.round(r['permlag'][o], 2).tolist() for o in ORDERS})
        print("SURROGATE lag xcorr:", {k: round(v, 2) for k, v in r['surr_lagxc'].items()})
        print("SURROGATE lag profiles:", {o: np.round(r['surr_lag'][o], 2).tolist() for o in ORDERS})
    else:
        np.savez(f"{PROJ}/results/haufe_surrogate.npz", recs=np.array(recs, dtype=object))
        def fmt(d): return " | ".join(f"{d[k][0]:+.2f}±{d[k][1]:.2f}" for k in ['a_p','a_v','v_p'])
        L = ["# §3.7 re-verification: Haufe patterns, permutation importance, spectrum surrogate — script 26 (n=18)\n",
             "Cross-order Pearson r of importance profiles (mean±sd over 18 sessions). a_p | a_v | v_p.\n",
             "| profile | grouping | a_p | a_v | v_p |", "|---|---|---|---|---|",
             f"| raw squared weight (script 02) | sensor | {fmt(_mean_xc(recs,'sen_raw_xc'))} |",
             f"| raw squared weight | lag | {fmt(_mean_xc(recs,'lag_raw_xc'))} |",
             f"| **Haufe activation pattern** | sensor | {fmt(_mean_xc(recs,'sen_hf_xc'))} |",
             f"| **Haufe activation pattern** | lag | {fmt(_mean_xc(recs,'lag_hf_xc'))} |",
             f"| **permutation importance** | lag | {fmt(_mean_xc(recs,'permxc'))} |",
             f"| **shared-signal SURROGATE** | lag | {fmt(_mean_xc(recs,'surr_lagxc'))} |",
             ""]
        # group-mean lag profiles (perm + surrogate)
        for tag, key in [("permutation-importance", 'permlag'), ("surrogate", 'surr_lag')]:
            P = {o: np.mean([r[key][o] for r in recs], 0) for o in ORDERS}
            L += [f"\nGroup-mean {tag} lag profile (normalized within order):", "",
                  "| lag (ms) | a | v | p |", "|---|---|---|---|"]
            for li, lm in enumerate(LAGS_MS):
                L.append(f"| {lm:+d} | {P['a'][li]:.3f} | {P['v'][li]:.3f} | {P['p'][li]:.3f} |")
        # stabilities
        sl = {o: np.mean([r['stab_lag'][o] for r in recs]) for o in ORDERS}
        ss = {o: np.mean([r['stab_sen'][o] for r in recs]) for o in ORDERS}
        L += ["", "## Split-half Haufe pattern stability (odd vs even trials, mean r over 18)",
              f"- lag patterns: a {sl['a']:.2f}, v {sl['v']:.2f}, p {sl['p']:.2f}",
              f"- sensor patterns: a {ss['a']:.2f}, v {ss['v']:.2f}, p {ss['p']:.2f}", ""]
        L += ["## Reading", "",
              "- If Haufe & permutation importance keep sensor xcorr high (shared) and lag xcorr with the "
              "a–v/v–p ANTI-correlation the raw weights showed, the §3.7 dissociation is not a weight-"
              "interpretation artifact.",
              "- SURROGATE is the key control: it has ONE built-in neural timing (tau0), so any lag "
              "dissociation there is a pure target-derivative/spectrum effect. Compare its lag xcorr to the "
              "real permutation-importance lag xcorr: if the surrogate REPRODUCES the anti-correlation, the "
              "real lag dissociation is (partly) a target-spectrum artifact and §3.7 must be softened to a "
              "descriptive result; if the surrogate shows shared (positive) lag profiles while the real data "
              "dissociate, the neural-timing reading survives."]
        open(f"{PROJ}/results/haufe_surrogate.md", "w").write("\n".join(L))
        print("\n".join(L)); print("\nSAVED results/haufe_surrogate.{npz,md}")
