"""
17 — Bandwidth-matched spectral control + calibrated clinical metrics + alpha sensitivity.
[DECISIVE CONFOUND TEST + CLINICAL INTERPRETABILITY]

Two independent reviewers converged on one deepest threat to the "complementary cross-order neural
code" reading: the integration operators T1 (velocity->position) and T2 (accel->position) LOW-PASS the
a/v routes, while the 10 E_bag members are unfiltered position decoders. Averaging three differently-
*filtered* estimators removes noise that averaging ten collinear *unfiltered* ones cannot. So part of
E_cross - E_bag could be pure BANDWIDTH diversity (a spectral-matching effect), not order-specific
neural information. Script 15 ruled out estimator *decorrelation* but NOT this; the members there are
unfiltered. This script runs the control the paper's Limitations #6 promises.

SPECTRAL PSEUDO-FUSION (the null with zero cross-order neural content).
  1. From TRAIN route predictions, estimate zero-phase magnitude filters L_v, L_a whose gain matches
     the empirical bandwidth of the v-route and a-route relative to p_direct:
         |L_o(f)|^2 = mean_trials|FFT(route_tr[o])|^2 / mean_trials|FFT(route_tr[p])|^2.
  2. Apply L_v, L_a to the p_direct TEST prediction -> p_lpf_v, p_lpf_a. These carry ONLY p_direct's
     neural content, reshaped to the v/a bandwidth.
  3. Fuse {p_direct, p_lpf_v, p_lpf_a} with the same performance-weighting as E_cross -> E_spec.
E_spec - E_bag is the gain achievable from bandwidth diversity ALONE (no cross-order signal).
E_cross - E_spec is the residual gain that must be genuine complementary neural information.

CALIBRATED CLINICAL METRICS (reviewers: per-trial residual r is a shape correlation blind to gain,
endpoint, and reach extent). We add, in the trial's native kinematic units:
  - per-trial R^2 (variance-explained, not just shape) on the CALIBRATED absolute-position prediction
    (per-route LS gain fit on train, then fused; not z-scored);
  - across-trial ENDPOINT decode: correlation of predicted vs true reach extent, and nRMSE (RMSE/range).
These say whether cross-order fusion improves the clinically actionable quantities, not only residual r.

ALPHA SENSITIVITY (reviewer item 4): p_direct and E_cross recomputed at alpha in {100, 1000, 5000} to
show the fusion win is not just p_direct being under-regularised/under-smoothed at the paper's alpha=100.

Run:  python 17_spectral_control.py test        # sub1 ses1
      python 17_spectral_control.py full         # 9x2 -> results/spectral_control.npz + .md
"""
import sys, time, importlib.util, numpy as np
from scipy.stats import wilcoxon

_spec = importlib.util.spec_from_file_location(
    "ens01", "/mnt/hdd/MEG_Motor/motor_ensemble_claude/scripts/01_ensemble_distance.py")
ens01 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ens01)
load_meg = ens01.load_meg; build_design = ens01.build_design; to_pos = ens01.to_pos
per_trial_r = ens01.per_trial_r; ztime = ens01.ztime; common_channels = ens01.common_channels
ORDERS = ens01.ORDERS; NFOLD = ens01.NFOLD
from scipy.linalg import cho_factor, cho_solve

ROOT = "/mnt/hdd/MEG_Motor"; SIB = f"{ROOT}/motor_comparison_claude"
PROJ = f"{ROOT}/motor_ensemble_claude"
N_BAG = 10; ALPHAS = [100.0, 1000.0, 5000.0]


def ridge_coef_a(A0, ZtY, alpha):
    A = A0.copy(); A[np.diag_indices_from(A)] += alpha
    return cho_solve(cho_factor(A, lower=True), ZtY)


def mag_filter(train_route, train_p):
    """Zero-phase magnitude filter |L(f)| = sqrt(mean|FFT(route)|^2 / mean|FFT(p)|^2), per-trial FFT."""
    Rr = np.fft.rfft(train_route, axis=-1); Rp = np.fft.rfft(train_p, axis=-1)
    num = (np.abs(Rr) ** 2).mean(0); den = (np.abs(Rp) ** 2).mean(0) + 1e-30
    return np.sqrt(num / den)


def apply_mag(x, L):
    """Apply zero-phase magnitude filter L (rfft-length) to signal x along last axis."""
    return np.fft.irfft(np.fft.rfft(x, axis=-1) * L, n=x.shape[-1], axis=-1)


def per_trial_r2(pred, true):
    ss_res = ((true - pred) ** 2).sum(-1)
    ss_tot = ((true - true.mean(-1, keepdims=True)) ** 2).sum(-1) + 1e-30
    return 1.0 - ss_res / ss_tot


def run_session(subj, sess, common, kin):
    X, t = load_meg(subj, sess, common)
    D, tw = build_design(X, t)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]) for o in ORDERS}
    ntr_all, nt, m = D.shape
    folds = np.array_split(np.random.default_rng(1000*subj + sess).permutation(ntr_all), NFOLD)
    srng = np.random.default_rng(7000 + 100*subj + sess)

    acc = {k: [] for k in ['E_cross', 'E_bag', 'E_spec', 'p_direct', 'a_route', 'v_route',
                            'cal_r2_dir', 'cal_r2_cross', 'end_r_dir', 'end_r_cross',
                            'end_nrmse_dir', 'end_nrmse_cross']}
    for a in ALPHAS:
        acc[f'p_direct@a{int(a)}'] = []; acc[f'E_cross@a{int(a)}'] = []

    for fi in range(NFOLD):
        te = folds[fi]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != fi])
        Dmean = D[tr].mean(0)
        flat_r = (D[tr] - Dmean).reshape(-1, m); mur = flat_r.mean(0); sdr = flat_r.std(0) + 1e-9
        Ztr = ((flat_r - mur) / sdr); Ztr3 = Ztr.reshape(len(tr), nt, m)
        Zte = ((D[te] - Dmean - mur) / sdr)
        Ym = {o: Ys[o][tr].mean(0) for o in ORDERS}
        Yr_tr = {o: (Ys[o][tr] - Ym[o]) for o in ORDERS}
        Ytrue = (Ys['p'][te] - Ym['p'])
        A0 = Ztr.T @ Ztr
        Ystack = np.stack([Yr_tr[o].reshape(-1) for o in ORDERS], axis=1)
        ZtY = Ztr.T @ Ystack

        def routes_at(alpha):
            B = ridge_coef_a(A0, ZtY, alpha); cf = {o: B[:, i] for i, o in enumerate(ORDERS)}
            rte = {o: to_pos(o, Zte @ cf[o]) for o in ORDERS}
            rtr = {o: to_pos(o, Ztr3 @ cf[o]) for o in ORDERS}
            return rte, rtr

        def fuse_wt(rte, rtr, keys):
            z = {k: ztime(rte[k]) for k in keys}
            w = np.array([max(0.0, per_trial_r(rtr[k], Yr_tr['p']).mean()) for k in keys])
            w = w / (w.sum() + 1e-12)
            return sum(w[i]*z[k] for i, k in enumerate(keys))

        # --- baseline alpha=100 (paper) ---
        rte, rtr = routes_at(100.0)
        for o, key in [('a', 'a_route'), ('v', 'v_route'), ('p', 'p_direct')]:
            acc[key].append(per_trial_r(rte[o], Ytrue).mean())
        E_cross = fuse_wt(rte, rtr, ['a', 'v', 'p'])
        acc['E_cross'].append(per_trial_r(E_cross, Ytrue).mean())
        bag = []
        for _ in range(N_BAG):
            idx = srng.choice(len(tr), size=int(0.8*len(tr)), replace=False)
            cb = ridge_coef_a(Ztr3[idx].reshape(-1, m).T @ Ztr3[idx].reshape(-1, m),
                              Ztr3[idx].reshape(-1, m).T @ Yr_tr['p'][idx].reshape(-1), 100.0)
            bag.append(ztime(Zte @ cb))
        acc['E_bag'].append(per_trial_r(np.mean(bag, axis=0), Ytrue).mean())

        # --- SPECTRAL pseudo-fusion: p_direct content, v/a bandwidth ---
        Lv = mag_filter(rtr['v'], rtr['p']); La = mag_filter(rtr['a'], rtr['p'])
        p_te = rte['p']; p_tr = rtr['p']
        pseudo_te = {'p': p_te, 'v': apply_mag(p_te, Lv), 'a': apply_mag(p_te, La)}
        pseudo_tr = {'p': p_tr, 'v': apply_mag(p_tr, Lv), 'a': apply_mag(p_tr, La)}
        E_spec = fuse_wt(pseudo_te, pseudo_tr, ['a', 'v', 'p'])
        acc['E_spec'].append(per_trial_r(E_spec, Ytrue).mean())

        # --- CALIBRATED metrics (native units, per-route LS gain, not z-scored) ---
        g = {o: float((rtr[o]*Yr_tr['p']).sum() / ((rtr[o]**2).sum() + 1e-30)) for o in ORDERS}
        wc = np.array([max(0.0, per_trial_r(rtr[o], Yr_tr['p']).mean()) for o in ORDERS])
        wc = wc / (wc.sum() + 1e-12)
        cal_dir = Ym['p'] + g['p']*rte['p']                                  # absolute position
        cal_cross = Ym['p'] + sum(wc[i]*g[o]*rte[o] for i, o in enumerate(ORDERS))
        true_abs = Ys['p'][te]
        acc['cal_r2_dir'].append(per_trial_r2(cal_dir, true_abs).mean())
        acc['cal_r2_cross'].append(per_trial_r2(cal_cross, true_abs).mean())
        # endpoint = mean of last 10 window samples (reach extent), across-trial decode
        ke = 10
        ep_t = true_abs[:, -ke:].mean(1); ep_d = cal_dir[:, -ke:].mean(1); ep_c = cal_cross[:, -ke:].mean(1)
        rng_ep = ep_t.max() - ep_t.min() + 1e-30
        acc['end_r_dir'].append(float(np.corrcoef(ep_d, ep_t)[0, 1]))
        acc['end_r_cross'].append(float(np.corrcoef(ep_c, ep_t)[0, 1]))
        acc['end_nrmse_dir'].append(float(np.sqrt(((ep_d-ep_t)**2).mean())/rng_ep))
        acc['end_nrmse_cross'].append(float(np.sqrt(((ep_c-ep_t)**2).mean())/rng_ep))

        # --- alpha sensitivity ---
        for a in ALPHAS:
            rte_a, rtr_a = routes_at(a)
            acc[f'p_direct@a{int(a)}'].append(per_trial_r(rte_a['p'], Ytrue).mean())
            acc[f'E_cross@a{int(a)}'].append(per_trial_r(fuse_wt(rte_a, rtr_a, ['a','v','p']), Ytrue).mean())

    return {k: float(np.mean(v)) for k, v in acc.items()}


def _paired(recs, a, b):
    da = np.array([r[a] for r in recs]); db = np.array([r[b] for r in recs]); d = da - db
    try: p = wilcoxon(da, db).pvalue
    except Exception: p = float('nan')
    return da.mean(), db.mean(), d.mean(), int((d > 0).sum()), len(recs), p


def _write_report(recs):
    L = ["# Bandwidth-matched spectral control + calibrated metrics — script 17 (n = 18)\n"]
    ec, eb, dcb, pcb, n, pcb_p = _paired(recs, 'E_cross', 'E_bag')
    _, es, dcs, pcs, _, pcs_p = _paired(recs, 'E_cross', 'E_spec')
    _, _, dsb, psb, _, psb_p = _paired(recs, 'E_spec', 'E_bag')
    frac = dsb / dcb if dcb != 0 else float('nan')
    L += ["## Decomposition of the fusion gain (per-trial r, position-residual space)",
          "| contrast | mean A | mean B | diff | sign | p | reads as |",
          "|---|---|---|---|---|---|---|",
          f"| E_cross vs E_bag (total gain) | {ec:.3f} | {eb:.3f} | **{dcb:+.3f}** | {pcb}/{n} | {pcb_p:.4f} | published effect |",
          f"| E_spec vs E_bag (bandwidth-only) | {ec if False else np.mean([r['E_spec'] for r in recs]):.3f} | {eb:.3f} | {dsb:+.3f} | {psb}/{n} | {psb_p:.4f} | gain from smoothing alone |",
          f"| **E_cross vs E_spec (residual neural)** | {ec:.3f} | {es:.3f} | **{dcs:+.3f}** | {pcs}/{n} | **{pcs_p:.4f}** | genuine complementary code |",
          "",
          f"- Bandwidth diversity alone (E_spec, p_direct's neural content reshaped to v/a bandwidth) "
          f"reproduces **{frac*100:.0f}%** of the total gain over bagging ({dsb:+.3f} of {dcb:+.3f}).",
          f"- The residual **E_cross - E_spec = {dcs:+.3f}** ({pcs}/{n}, p = {pcs_p:.4f}): both ensembles "
          f"share identical bandwidth diversity, so any E_cross advantage here would be cross-order neural",
          "  content. There is none. (E_spec is if anything a CONSERVATIVE null: its members share",
          "  p_direct's phase, so it lacks the independent decode-noise and phase/lag diversity E_cross's",
          "  real v/a members have — E_cross had strictly more mechanisms available and still did not win.)",
          "",
          "## Calibrated clinical metrics (native kinematic units, not shape-r) — HONEST READ: both fail",
          "| metric | direct | cross-fusion | diff | sign | p |",
          "|---|---|---|---|---|---|"]
    for name, kd, kc in [("per-trial R^2", 'cal_r2_dir', 'cal_r2_cross'),
                         ("endpoint decode r", 'end_r_dir', 'end_r_cross'),
                         ("endpoint nRMSE (lower=better)", 'end_nrmse_dir', 'end_nrmse_cross')]:
        ma, mb, md, pos, nn, p = _paired(recs, kc, kd)   # cross - direct
        L.append(f"| {name} | {mb:.3f} | {ma:.3f} | {md:+.3f} | {pos}/{nn} | {p:.4f} |")
    L += ["",
          "- These are NOT a clinical benefit. Absolute per-trial R^2 is far below 0 (worse than "
          "predicting the trial mean) and endpoint-decode r ~ 0 for BOTH direct and fusion, so neither "
          "achieves usable absolute reconstruction or reach-extent decoding. Cross-fusion is marginally "
          "less-bad on a failing baseline (nRMSE shrinkage with r~0 = variance shrinkage, not decoding). "
          "Reported to document the weak-signal ceiling honestly, not to argue utility.",
          "",
          "## Alpha sensitivity (context only — the confound-relevant alpha sweep is script 20)",
          "| alpha | p_direct | E_cross | E_cross - p_direct |",
          "|---|---|---|---|"]
    for a in ALPHAS:
        pd = np.mean([r[f'p_direct@a{int(a)}'] for r in recs])
        ex = np.mean([r[f'E_cross@a{int(a)}'] for r in recs])
        L.append(f"| {int(a)} | {pd:.3f} | {ex:.3f} | {ex-pd:+.3f} |")
    L += ["",
          "- This shows only that E_cross beats the single direct decoder at every alpha — a gap the "
          "spectral control attributes to bandwidth, so it is not itself the confound test. The proper "
          "check (E_cross - E_spec across alpha) is in `results/spectral_alpha.md` (script 20).",
          "",
          "## Bottom line",
          (f"**No evidence of cross-order neural complementarity beyond bandwidth diversity.** "
           f"The +{dcb:.3f} fusion-over-bagging gain is fully accounted for by the differing bandwidth "
           f"that T1/T2 integration imposes on the routes: a bandwidth-matched pseudo-fusion with ZERO "
           f"cross-order neural content reproduces {frac*100:.0f}% of it, and the residual "
           f"E_cross - E_spec = {dcs:+.3f} ({pcs}/{n}, p = {pcs_p:.4f}) is null. We therefore withdraw the "
           f"MECHANISTIC attribution ('partly separate cross-order codes drive the fusion win'); the data "
           f"do not license it. This does not prove the codes are absent — a fixed-weight fusion in this "
           f"metric cannot detect a weak cross-order signal a better estimator might — but there is no "
           f"positive evidence for one here. The defensible statement is: integration-based multi-route "
           f"fusion is an effective bandwidth/smoothing operation for weak-signal reconstruction.")]
    txt = "\n".join(L) + "\n"
    with open(f"{PROJ}/results/spectral_control.md", "w") as fh:
        fh.write(txt)
    print("\n" + txt)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kin = dict(np.load(f"{SIB}/features/kinematics.npz", allow_pickle=True))
    common = common_channels(); print(f"common grad: {len(common)}", flush=True)
    if mode == "test":
        t0 = time.time(); r = run_session(1, 1, common, kin)
        print(f"\nSub1 ses1 ({time.time()-t0:.1f}s)")
        print(f"  E_cross={r['E_cross']:.3f} E_bag={r['E_bag']:.3f} E_spec={r['E_spec']:.3f}")
        print(f"  gain total={r['E_cross']-r['E_bag']:+.3f}  spectral={r['E_spec']-r['E_bag']:+.3f}  "
              f"residual={r['E_cross']-r['E_spec']:+.3f}")
        print(f"  cal R2: dir={r['cal_r2_dir']:.3f} cross={r['cal_r2_cross']:.3f}")
        print(f"  endpoint r: dir={r['end_r_dir']:.3f} cross={r['end_r_cross']:.3f} "
              f"| nRMSE dir={r['end_nrmse_dir']:.3f} cross={r['end_nrmse_cross']:.3f}")
        print("  alpha sens:", {int(a): (round(r[f'p_direct@a{int(a)}'],3),
                                          round(r[f'E_cross@a{int(a)}'],3)) for a in ALPHAS})
    elif mode == "full":
        recs = []
        for s in range(1, 10):
            for se in (1, 2):
                t0 = time.time(); r = run_session(s, se, common, kin)
                recs.append(dict(subj=s, sess=se, **r))
                print(f"Sub{s} ses{se} ({time.time()-t0:.1f}s) Ecross={r['E_cross']:.3f} "
                      f"Espec={r['E_spec']:.3f} Ebag={r['E_bag']:.3f} "
                      f"resid={r['E_cross']-r['E_spec']:+.3f}", flush=True)
        np.savez(f"{PROJ}/results/spectral_control.npz", recs=np.array(recs, dtype=object))
        print("SAVED results/spectral_control.npz")
        _write_report(recs)
