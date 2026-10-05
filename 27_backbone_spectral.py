"""
27 — Does the BANDWIDTH explanation of the fusion gain generalise across backbones? (reviewer point 3)

The decisive spectral control (script 17) and its alpha-robustness were run only on the ridge design.
The paper claims the fusion-over-bagging gain appears in ridge / frozen-SSL / random-projection / LSTM
backbones, then attributes ALL of it to bandwidth from the ridge result alone. This script closes that
gap: it re-runs the bandwidth-matched pseudo-fusion (E_spec) AND an optimally-smoothed single decoder
control on the SSL and random-projection backbones (and the raw ridge design, for a matched reference).

Per backbone (raw | ssl | rand), in script 01's fair position-residual space, zero-phase (matching the
published primary analysis):
  E_cross  : perf-weighted fusion of a/v/p routes
  E_bag    : 10 position-direct decoders on 0.8 subsamples (variance control)
  E_spec   : p_direct reshaped to the v/a empirical bandwidth (zero cross-order neural content) then fused
  p_smooth : the SINGLE p_direct decoder low-passed at a nested-CV-selected cutoff (train-picked) —
             tests the reviewer's stronger claim that appropriately smoothing ONE position decoder
             already reproduces the cross-order gain, so cross-order decoding is unnecessary for it.
Contrasts reported: E_cross-E_bag, E_spec-E_bag, E_cross-E_spec, E_cross-p_smooth (session + subject).
LSTM is deliberately excluded: its direct position route ~ 0 (script 04), so p_direct-anchored spectral
and smoothing controls are undefined/unstable there — stated, not silently dropped.

Run (.venv_deep, torch):  python 27_backbone_spectral.py <raw|ssl|rand> [test|full]
"""
import sys, time, importlib.util, numpy as np
from scipy.stats import wilcoxon
b03 = importlib.util.spec_from_file_location("b03", "/mnt/hdd/MEG_Motor/motor_ensemble_claude/scripts/03_backbone_ensemble.py")
m03 = importlib.util.module_from_spec(b03); b03.loader.exec_module(m03)

ROOT = "/mnt/hdd/MEG_Motor"; SIB = f"{ROOT}/motor_comparison_claude"; PROJ = f"{ROOT}/motor_ensemble_claude"
ORDERS = ['a', 'v', 'p']; NFOLD = m03.NFOLD; N_BAG = 10
per_trial_r = m03.per_trial_r; ztime = m03.ztime; to_pos = m03.to_pos; ridge_coef = m03.ridge_coef


def mag_filter(train_route, train_p):
    Rr = np.fft.rfft(train_route, axis=-1); Rp = np.fft.rfft(train_p, axis=-1)
    return np.sqrt((np.abs(Rr)**2).mean(0) / ((np.abs(Rp)**2).mean(0) + 1e-30))

def apply_mag(x, L):
    return np.fft.irfft(np.fft.rfft(x, axis=-1) * L, n=x.shape[-1], axis=-1)

def lp_ideal(x, fc, sf=100.0):
    """zero-phase ideal low-pass at fc Hz (FFT truncation)."""
    R = np.fft.rfft(x, axis=-1); f = np.fft.rfftfreq(x.shape[-1], 1.0/sf)
    R[:, f > fc] = 0.0
    return np.fft.irfft(R, n=x.shape[-1], axis=-1)

CUTS = [1, 1.5, 2, 3, 4, 6, 8, 12]


def fuse_wt(zte, rtr, keys, Yr_tr_p):
    w = np.array([max(0.0, per_trial_r(rtr[k], Yr_tr_p).mean()) for k in keys]); w = w/(w.sum()+1e-12)
    return sum(w[i]*zte[k] for i, k in enumerate(keys))


def run_session(kind, subj, sess, common, kin):
    X, t = m03.load_meg(subj, sess, common); D, tw = m03.build_design(X, t)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]) for o in ORDERS}
    ntr, nt, m = D.shape
    folds = np.array_split(np.random.default_rng(1000*subj+sess).permutation(ntr), NFOLD)
    srng = np.random.default_rng(7000+100*subj+sess)
    acc = {k: [] for k in ['E_cross', 'E_bag', 'E_spec', 'p_smooth', 'p_direct', 'a_route', 'v_route']}

    for fi in range(NFOLD):
        te = folds[fi]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != fi])
        Dmean = D[tr].mean(0); flat = (D[tr]-Dmean).reshape(-1, m); mur = flat.mean(0); sdr = flat.std(0)+1e-9
        Ztr_raw = (flat-mur)/sdr; Zte_raw = ((D[te]-Dmean-mur)/sdr).reshape(-1, m)
        if kind == 'raw':
            Ptr, Pte = Ztr_raw, Zte_raw
        else:
            Ptr = m03.embed(Ztr_raw); Pte = m03.embed(Zte_raw)
            muP = Ptr.mean(0); sdP = Ptr.std(0)+1e-9; Ptr = (Ptr-muP)/sdP; Pte = (Pte-muP)/sdP
        e = Ptr.shape[1]; Ptr3 = Ptr.reshape(len(tr), nt, e); Pte3 = Pte.reshape(len(te), nt, e)
        Ym = {o: Ys[o][tr].mean(0) for o in ORDERS}; Yr_tr = {o: Ys[o][tr]-Ym[o] for o in ORDERS}
        Ytrue = Ys['p'][te]-Ym['p']
        B = ridge_coef(Ptr, np.stack([Yr_tr[o].reshape(-1) for o in ORDERS], 1)); cf = {o: B[:, i] for i, o in enumerate(ORDERS)}
        rte = {o: to_pos(o, Pte3 @ cf[o]) for o in ORDERS}; rtr = {o: to_pos(o, Ptr3 @ cf[o]) for o in ORDERS}
        for o, k in [('a','a_route'),('v','v_route'),('p','p_direct')]:
            acc[k].append(per_trial_r(rte[o], Ytrue).mean())
        zte = {o: ztime(rte[o]) for o in ORDERS}
        acc['E_cross'].append(per_trial_r(fuse_wt(zte, rtr, ['a','v','p'], Yr_tr['p']), Ytrue).mean())
        bag = []
        for _ in range(N_BAG):
            idx = srng.choice(len(tr), size=int(0.8*len(tr)), replace=False)
            cb = ridge_coef(Ptr3[idx].reshape(-1, e), Yr_tr['p'][idx].reshape(-1)); bag.append(ztime(Pte3 @ cb))
        acc['E_bag'].append(per_trial_r(np.mean(bag, 0), Ytrue).mean())
        # spectral pseudo-fusion
        Lv = mag_filter(rtr['v'], rtr['p']); La = mag_filter(rtr['a'], rtr['p'])
        ps_te = {'p': rte['p'], 'v': apply_mag(rte['p'], Lv), 'a': apply_mag(rte['p'], La)}
        ps_tr = {'p': rtr['p'], 'v': apply_mag(rtr['p'], Lv), 'a': apply_mag(rtr['p'], La)}
        zsp = {k: ztime(ps_te[k]) for k in ORDERS}
        acc['E_spec'].append(per_trial_r(fuse_wt(zsp, ps_tr, ['a','v','p'], Yr_tr['p']), Ytrue).mean())
        # optimally-smoothed single p_direct decoder (nested cutoff pick on TRAIN)
        best_fc = max(CUTS, key=lambda fc: per_trial_r(lp_ideal(rtr['p'], fc), Yr_tr['p']).mean())
        acc['p_smooth'].append(per_trial_r(ztime(lp_ideal(rte['p'], best_fc)), Ytrue).mean())
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _paired(recs, a, b):
    da = np.array([r[a] for r in recs]); db = np.array([r[b] for r in recs]); d = da-db
    subj = np.array([r['subj'] for r in recs]); dss = np.array([d[subj == s].mean() for s in np.unique(subj)])
    ps = wilcoxon(da, db).pvalue if np.any(d) else float('nan')
    pj = wilcoxon(dss).pvalue if np.any(dss) else float('nan')
    return da.mean(), db.mean(), d.mean(), int((d > 0).sum()), len(recs), ps, int((dss > 0).sum()), len(dss), pj


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "raw"
    mode = sys.argv[2] if len(sys.argv) > 2 else "test"
    if kind != 'raw':
        m03.ENC = m03.build_encoder(kind)
    kin = dict(np.load(f"{SIB}/features/kinematics.npz", allow_pickle=True))
    common = m03.common_channels(); print(f"[{kind}] common {len(common)} dev {m03.DEV}", flush=True)
    if mode == "test":
        t0 = time.time(); r = run_session(kind, 1, 1, common, kin)
        print(f"[{kind}] Sub1 ses1 ({time.time()-t0:.1f}s)")
        print(f"  a={r['a_route']:.3f} v={r['v_route']:.3f} p={r['p_direct']:.3f}")
        print(f"  E_cross={r['E_cross']:.3f} E_bag={r['E_bag']:.3f} E_spec={r['E_spec']:.3f} p_smooth={r['p_smooth']:.3f}")
        print(f"  tot={r['E_cross']-r['E_bag']:+.4f} spec={r['E_spec']-r['E_bag']:+.4f} "
              f"res={r['E_cross']-r['E_spec']:+.4f}  cross-vs-psmooth={r['E_cross']-r['p_smooth']:+.4f}")
    else:
        recs = []
        for s in range(1, 10):
            for se in (1, 2):
                t0 = time.time(); r = run_session(kind, s, se, common, kin); recs.append(dict(subj=s, sess=se, **r))
                print(f"[{kind}] Sub{s} ses{se} ({time.time()-t0:.1f}s) tot={r['E_cross']-r['E_bag']:+.3f} "
                      f"spec={r['E_spec']-r['E_bag']:+.3f} res={r['E_cross']-r['E_spec']:+.3f}", flush=True)
        np.savez(f"{PROJ}/results/backbone_spectral_{kind}.npz", recs=np.array(recs, dtype=object))
        print(f"SAVED results/backbone_spectral_{kind}.npz")
        for a, b, nm in [('E_cross','E_bag','total'),('E_spec','E_bag','bandwidth-only'),
                         ('E_cross','E_spec','residual neural'),('E_cross','p_smooth','vs optimally-smoothed direct')]:
            ma, mb, md, sp, sn, ps, jp, jn, pj = _paired(recs, a, b)
            print(f"  [{kind}] {nm:28s}: {md:+.4f}  sess {sp}/{sn} p={ps:.4f}  subj {jp}/{jn} p={pj:.4f}")
