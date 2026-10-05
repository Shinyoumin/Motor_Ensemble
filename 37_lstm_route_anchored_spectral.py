"""
37 — Route-anchored bandwidth control for the trained-BiLSTM backbone.

Supplementary S1 (block C) reports the bandwidth decomposition on ridge, frozen-SSL and
random-projection backbones and states that the trained BiLSTM "is excluded from this control because
its direct-position route is ~0, which leaves the p-anchored E_spec and p_smooth undefined; we state
this rather than dropping it silently." This script removes that exclusion by anchoring the control on
the route that actually works for this backbone: acceleration (LSTM a_route ~0.094 vs p_direct ~0).

Reuses script 04 unchanged (data, LSTM, folds, training) and adds, per fold:
  E_bagA  — same-order bagging of acceleration decoders (N_BAG members, integrated by T2 into
            position space): the same-order variance control matched to the carrier route.
  E_specA — acceleration-anchored bandwidth-matched pseudo-fusion {a, L_p|a * a, L_v|a * a} with
            zero-phase magnitude filters estimated on TRAIN predictions (script 17's mag_filter),
            i.e. three deterministic filterings of ONE decoder: zero cross-order neural content.

Decisive contrast: E_cross - E_specA. If null, the bandwidth account of the fusion margin now covers
all four backbones of Table 1 rather than three.

Run (.venv_deep):  python 37_lstm_route_anchored_spectral.py <test|full> [gpu]
"""
import sys, time, importlib.util, numpy as np
from scipy.stats import wilcoxon

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"
def _load(n, p):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m); return m
m04 = _load("lstm04", f"{PROJ}/scripts/04_lstm_ensemble.py")
s17 = _load("spec17", f"{PROJ}/scripts/17_spectral_control.py")
mag_filter, apply_mag = s17.mag_filter, s17.apply_mag
ORDERS, NFOLD, N_BAG, BAG_FRAC = m04.ORDERS, m04.NFOLD, m04.N_BAG, m04.BAG_FRAC
per_trial_r, ztime, to_pos = m04.per_trial_r, m04.ztime, m04.to_pos
WIN, SIB = m04.WIN, m04.SIB


def run_session(subj, sess, common, kin):
    X, t = m04.load_meg(subj, sess, common)
    tt = np.where((t >= WIN[0]) & (t <= WIN[1]))[0]; tw = t[tt]
    Xw = X[:, :, tt].transpose(0, 2, 1)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]) for o in ORDERS}
    ntr_all, nt, nch = Xw.shape
    folds = np.array_split(np.random.default_rng(1000*subj + sess).permutation(ntr_all), NFOLD)
    brng = np.random.default_rng(7000 + 100*subj + sess)
    acc = {k: [] for k in ['a_route', 'v_route', 'p_direct', 'E_cross', 'E_bag', 'E_bagA', 'E_specA']}

    for f in range(NFOLD):
        te = folds[f]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != f])
        Xmean = Xw[tr].mean(0)
        mu = (Xw[tr]-Xmean).reshape(-1, nch).mean(0); sd = (Xw[tr]-Xmean).reshape(-1, nch).std(0)+1e-9
        Xtr = (Xw[tr]-Xmean-mu)/sd; Xte = (Xw[te]-Xmean-mu)/sd
        Ym = {o: Ys[o][tr].mean(0) for o in ORDERS}
        Yr_tr = {o: Ys[o][tr]-Ym[o] for o in ORDERS}
        Ytrue = Ys['p'][te]-Ym['p']

        rte, rtr = {}, {}
        for o in ORDERS:
            net = m04.train_lstm(Xtr, Yr_tr[o], seed=0)
            rte[o] = to_pos(o, m04.predict(net, Xte)); rtr[o] = to_pos(o, m04.predict(net, Xtr))
        for o, k in [('a', 'a_route'), ('v', 'v_route'), ('p', 'p_direct')]:
            acc[k].append(per_trial_r(rte[o], Ytrue).mean())

        zte = {o: ztime(rte[o]) for o in ORDERS}
        w = np.array([max(0.0, per_trial_r(rtr[o], Yr_tr['p']).mean()) for o in ORDERS]); w /= (w.sum()+1e-12)
        acc['E_cross'].append(per_trial_r(sum(w[i]*zte[o] for i, o in enumerate(ORDERS)), Ytrue).mean())

        # p-anchored bag (script 04's reference) and a-anchored bag (matched to the carrier)
        for tag, order in [('E_bag', 'p'), ('E_bagA', 'a')]:
            bag = []
            for b in range(N_BAG):
                idx = brng.choice(len(tr), size=int(BAG_FRAC*len(tr)), replace=False)
                nb = m04.train_lstm(Xtr[idx], Yr_tr[order][idx], seed=b+1)
                bag.append(ztime(to_pos(order, m04.predict(nb, Xte))))
            acc[tag].append(per_trial_r(np.mean(bag, 0), Ytrue).mean())

        # acceleration-anchored bandwidth-matched pseudo-fusion (zero cross-order content)
        Lp = mag_filter(rtr['p'], rtr['a']); Lv = mag_filter(rtr['v'], rtr['a'])
        ste = {'a': rte['a'], 'v': apply_mag(rte['a'], Lv), 'p': apply_mag(rte['a'], Lp)}
        str_ = {'a': rtr['a'], 'v': apply_mag(rtr['a'], Lv), 'p': apply_mag(rtr['a'], Lp)}
        zs = {o: ztime(ste[o]) for o in ORDERS}
        ws = np.array([max(0.0, per_trial_r(str_[o], Yr_tr['p']).mean()) for o in ORDERS]); ws /= (ws.sum()+1e-12)
        acc['E_specA'].append(per_trial_r(sum(ws[i]*zs[o] for i, o in enumerate(ORDERS)), Ytrue).mean())

    return {k: float(np.mean(v)) for k, v in acc.items()}


def _st(recs, a, b):
    da = np.array([r[a] for r in recs]); db = np.array([r[b] for r in recs]); d = da-db
    sj = np.array([r['subj'] for r in recs]); ds = np.array([d[sj == s].mean() for s in np.unique(sj)])
    return (da.mean(), db.mean(), d.mean(), int((d > 0).sum()), len(d),
            wilcoxon(da, db).pvalue, int((ds > 0).sum()), len(ds), wilcoxon(ds).pvalue)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kin = dict(np.load(f"{SIB}/features/kinematics.npz", allow_pickle=True))
    common = m04.common_channels(); print(f"common grad: {len(common)}", flush=True)
    if mode == "test":
        t0 = time.time(); r = run_session(1, 1, common, kin)
        print(f"Sub1 ses1 ({time.time()-t0:.1f}s)")
        for k, v in r.items(): print(f"   {k:9s} {v:+.4f}")
        print(f"   E_cross-E_bagA  {r['E_cross']-r['E_bagA']:+.4f}")
        print(f"   E_specA-E_bagA  {r['E_specA']-r['E_bagA']:+.4f}")
        print(f"   E_cross-E_specA {r['E_cross']-r['E_specA']:+.4f}")
    else:
        recs = []
        for s in range(1, 10):
            for se in (1, 2):
                t0 = time.time(); r = run_session(s, se, common, kin)
                recs.append(dict(subj=s, sess=se, **r))
                print(f"Sub{s} ses{se} ({time.time()-t0:.1f}s) Ec={r['E_cross']:.3f} "
                      f"EbA={r['E_bagA']:.3f} EsA={r['E_specA']:.3f} "
                      f"res={r['E_cross']-r['E_specA']:+.3f}", flush=True)
        np.savez(f"{PROJ}/results/lstm_route_spectral.npz", recs=np.array(recs, dtype=object))
        L = ["# Route-anchored bandwidth control on the trained BiLSTM — script 37 (n = 18)\n",
             "Removes the BiLSTM's exclusion from the bandwidth decomposition (Supplementary S1, block C) by "
             "anchoring the controls on acceleration, the only route this backbone decodes. E_bagA = bagged "
             "acceleration decoders (T2-integrated); E_specA = {a, L_v|a a, L_p|a a}, zero cross-order "
             "content.\n",
             f"Single routes: a {np.mean([r['a_route'] for r in recs]):.3f} · "
             f"v {np.mean([r['v_route'] for r in recs]):.3f} · p {np.mean([r['p_direct'] for r in recs]):.3f}; "
             f"E_cross {np.mean([r['E_cross'] for r in recs]):.3f}, p-anchored E_bag "
             f"{np.mean([r['E_bag'] for r in recs]):.3f}.\n",
             "| contrast | mean A | mean B | diff | sessions | subjects |", "|---|---|---|---|---|---|"]
        for a, b, lab in [('E_cross', 'E_bagA', 'E_cross − E_bagA (vs same-order acceleration bagging)'),
                          ('E_specA', 'E_bagA', 'E_specA − E_bagA (a-anchored bandwidth-only)'),
                          ('E_cross', 'E_specA', '**E_cross − E_specA (bandwidth-clean residual)**'),
                          ('E_cross', 'E_bag', 'E_cross − E_bag (reference, script 04)')]:
            ma, mb, md, sp, sn, ps, jp, jn, pj = _st(recs, a, b)
            L.append(f"| {lab} | {ma:.3f} | {mb:.3f} | **{md:+.4f}** | {sp}/{sn}, p={ps:.4f} | {jp}/{jn}, p={pj:.4f} |")
        open(f"{PROJ}/results/lstm_route_spectral.md", "w").write("\n".join(L)+"\n")
        print("\n".join(L)); print("SAVED results/lstm_route_spectral.{npz,md}")
