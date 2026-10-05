"""
36 — Does a STRONGER estimator recover cross-order information beyond bandwidth?

PAPER.md §5 (Limitations) concedes the one escape hatch left by the central de-claiming result:
"a fixed-weight fusion in the position-residual metric cannot exclude a *weak* cross-order signal that
a noise-covariance-aware (e.g. GLS/learned) estimator might extract". This script closes it, in the
zero-phase full-lag frame where the central claim lives (§3.9).

The logic: apply the SAME estimator to two triples and compare.
  cross-order triple : {a_route, v_route, p_direct}            (real cross-order neural content)
  bandwidth triple   : {p_direct, L_v*p_direct, L_a*p_direct}  (script 17's E_spec: ZERO cross-order
                                                                content, same bandwidth diversity)
Estimators, in increasing strength (all fitted without touching test data):
  1. perf_wt      — the paper's fixed performance-weighted mean (reference; reproduces §3.9)
  2. stack_ridge  — learned static linear weights, fitted by NESTED out-of-fold CV (script 07's scheme)
  3. band_wiener  — a per-frequency-band least-squares combiner (six bands over 0.5-12 Hz), i.e. a
                    crude Wiener/GLS filter that can weight each route differently at each frequency.
                    This is precisely the estimator class that could exploit band-limited cross-order
                    information which a single scalar weight per route cannot.

If the cross-order triple still fails to beat the bandwidth-matched triple under the stronger
estimators, the "a better estimator might find it" hedge is closed for this dataset and metric.

CONDITIONING CONTROL (added after the first run). The band-wise combiner was the one estimator under
which the cross-order triple came out ahead — but E_spec's three members are deterministic filterings
of ONE decoder and are therefore near-collinear, which makes a per-band least-squares fit unstable in a
way that has nothing to do with neural content. We therefore add a third, decorrelated bandwidth-only
triple, `speci`: {p_b1, L_v * p_b2, L_a * p_b3}, where p_b1..3 are three INDEPENDENTLY bagged position
decoders. It carries zero cross-order content like E_spec but has member-to-member independence like
the cross-order routes, so it isolates conditioning from information.

Run:  .../.venv_meggpt/bin/python scripts/36_estimator_strength_control.py <test|full>
"""
import sys, time, importlib.util, numpy as np
from scipy.stats import wilcoxon

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"
def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod); return mod
e01 = _load("ens01", f"{PROJ}/scripts/01_ensemble_distance.py")
s17 = _load("spec17", f"{PROJ}/scripts/17_spectral_control.py")

ORDERS, NFOLD = e01.ORDERS, e01.NFOLD
per_trial_r, ztime, to_pos, ridge_coef = e01.per_trial_r, e01.ztime, e01.to_pos, e01.ridge_coef
mag_filter, apply_mag = s17.mag_filter, s17.apply_mag
SIB = "/mnt/hdd/MEG_Motor/motor_comparison_claude"
NINNER = 5
SF = 100.0
BANDS = [(0.5, 1), (1, 2), (2, 4), (4, 6), (6, 8), (8, 12)]
N_BAG = 10


def band_masks(nt):
    f = np.fft.rfftfreq(nt, d=1.0/SF)
    return [((f >= lo) & (f < hi)).astype(float) for lo, hi in BANDS]


def band_split(x, masks):
    """x: (ntr, nt) -> list of band-limited copies (zero-phase)."""
    R = np.fft.rfft(x, axis=-1)
    return [np.fft.irfft(R*mk, n=x.shape[-1], axis=-1) for mk in masks]


def fit_band_wiener(F_oof, y_oof, masks, ridge=1.0):
    """F_oof: (n, nt, K) OOF member predictions; y_oof: (n, nt). Returns per-band weight vectors."""
    W = []
    yb = band_split(y_oof, masks)
    Fb = [np.stack([band_split(F_oof[:, :, k], masks)[b] for k in range(F_oof.shape[2])], axis=-1)
          for b in range(len(masks))]
    for b in range(len(masks)):
        A = Fb[b].reshape(-1, F_oof.shape[2]); yv = yb[b].reshape(-1)
        G = A.T @ A; G[np.diag_indices_from(G)] += ridge
        W.append(np.linalg.solve(G, A.T @ yv))
    return W


def apply_band_wiener(F, W, masks):
    out = np.zeros(F.shape[:2])
    for b, mk in enumerate(masks):
        Fb = np.stack([band_split(F[:, :, k], masks)[b] for k in range(F.shape[2])], axis=-1)
        out += Fb @ W[b]
    return out


def run_session(subj, sess, common, kin):
    X, t = e01.load_meg(subj, sess, common)
    D, tw = e01.build_design(X, t)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]) for o in ORDERS}
    ntr_all, nt, m = D.shape
    masks = band_masks(nt)
    folds = np.array_split(np.random.default_rng(1000*subj+sess).permutation(ntr_all), NFOLD)
    srng = np.random.default_rng(7000 + 100*subj + sess)
    keys = ['perf_cross', 'perf_spec', 'stack_cross', 'stack_spec',
            'wien_cross', 'wien_spec', 'perf_speci', 'stack_speci', 'wien_speci',
            'E_bag', 'p_direct', 'rho_cross', 'rho_spec', 'rho_speci']
    acc = {k: [] for k in keys}

    for fi in range(NFOLD):
        te = folds[fi]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != fi])
        Dm = D[tr].mean(0); flat = (D[tr]-Dm).reshape(-1, m); mu = flat.mean(0); sd = flat.std(0)+1e-9
        Ztr = (flat-mu)/sd; Ztr3 = Ztr.reshape(len(tr), nt, m); Zte = (D[te]-Dm-mu)/sd
        Ym = {o: Ys[o][tr].mean(0) for o in ORDERS}
        Yr_tr = {o: Ys[o][tr]-Ym[o] for o in ORDERS}
        Ytrue = Ys['p'][te]-Ym['p']

        B = ridge_coef(Ztr, np.stack([Yr_tr[o].reshape(-1) for o in ORDERS], axis=1))
        cf = {o: B[:, i] for i, o in enumerate(ORDERS)}
        rte = {o: to_pos(o, Zte @ cf[o]) for o in ORDERS}
        rtr = {o: to_pos(o, Ztr3 @ cf[o]) for o in ORDERS}
        acc['p_direct'].append(per_trial_r(rte['p'], Ytrue).mean())

        Lv = mag_filter(rtr['v'], rtr['p']); La = mag_filter(rtr['a'], rtr['p'])
        ste = {'p': rte['p'], 'v': apply_mag(rte['p'], Lv), 'a': apply_mag(rte['p'], La)}
        str_ = {'p': rtr['p'], 'v': apply_mag(rtr['p'], Lv), 'a': apply_mag(rtr['p'], La)}

        def perf(rt_te, rt_tr):
            z = {k: ztime(rt_te[k]) for k in ORDERS}
            w = np.array([max(0.0, per_trial_r(rt_tr[k], Yr_tr['p']).mean()) for k in ORDERS])
            w = w/(w.sum()+1e-12)
            return sum(w[i]*z[k] for i, k in enumerate(ORDERS))
        acc['perf_cross'].append(per_trial_r(perf(rte, rtr), Ytrue).mean())
        acc['perf_spec'].append(per_trial_r(perf(ste, str_), Ytrue).mean())

        # ---- nested OOF member predictions on the training trials (leakage-free) ----
        inner = np.array_split(np.random.default_rng(31 + fi).permutation(len(tr)), NINNER)
        oofC = np.zeros((len(tr), nt, 3)); oofS = np.zeros((len(tr), nt, 3))
        for g in range(NINNER):
            iva = inner[g]; itr = np.concatenate([inner[k] for k in range(NINNER) if k != g])
            Dm_i = D[tr[itr]].mean(0); fl = (D[tr[itr]]-Dm_i).reshape(-1, m)
            mui = fl.mean(0); sdi = fl.std(0)+1e-9
            Zi = (fl-mui)/sdi; Zi3 = Zi.reshape(len(itr), nt, m)
            Zv = (D[tr[iva]]-Dm_i-mui)/sdi
            Ymi = {o: Ys[o][tr[itr]].mean(0) for o in ORDERS}
            Bi = ridge_coef(Zi, np.stack([(Ys[o][tr[itr]]-Ymi[o]).reshape(-1) for o in ORDERS], axis=1))
            cfi = {o: Bi[:, j] for j, o in enumerate(ORDERS)}
            rv = {o: to_pos(o, Zv @ cfi[o]) for o in ORDERS}
            ri = {o: to_pos(o, Zi3 @ cfi[o]) for o in ORDERS}
            Lvi = mag_filter(ri['v'], ri['p']); Lai = mag_filter(ri['a'], ri['p'])
            for j, o in enumerate(ORDERS):
                oofC[iva, :, j] = ztime(rv[o])
            for j, o in enumerate(ORDERS):
                oofS[iva, :, j] = ztime({'a': apply_mag(rv['p'], Lai),
                                         'v': apply_mag(rv['p'], Lvi), 'p': rv['p']}[o])
        yoof = Yr_tr['p']
        FteC = np.stack([ztime(rte[o]) for o in ORDERS], axis=-1)
        FteS = np.stack([ztime(ste[o]) for o in ORDERS], axis=-1)

        # ---- estimator 2: learned static linear stack ----
        for tag, Foof, Fte in [('cross', oofC, FteC), ('spec', oofS, FteS)]:
            A = Foof.reshape(-1, 3); G = A.T @ A; G[np.diag_indices_from(G)] += 1.0
            c = np.linalg.solve(G, A.T @ yoof.reshape(-1))
            acc[f'stack_{tag}'].append(per_trial_r((Fte.reshape(-1, 3) @ c).reshape(len(te), nt), Ytrue).mean())

        # ---- estimator 3: per-band Wiener/GLS combiner ----
        for tag, Foof, Fte in [('cross', oofC, FteC), ('spec', oofS, FteS)]:
            W = fit_band_wiener(Foof, yoof, masks)
            acc[f'wien_{tag}'].append(per_trial_r(apply_band_wiener(Fte, W, masks), Ytrue).mean())

        bag = []
        for _ in range(N_BAG):
            idx = srng.choice(len(tr), size=int(0.8*len(tr)), replace=False)
            cb = ridge_coef(Ztr3[idx].reshape(-1, m), Yr_tr['p'][idx].reshape(-1))
            bag.append(ztime(Zte @ cb))
        acc['E_bag'].append(per_trial_r(np.mean(bag, axis=0), Ytrue).mean())

        # ---- conditioning control: bandwidth-only triple from THREE independent bagged p decoders ----
        irng = np.random.default_rng(9100 + 100*subj + sess + fi)
        pb_te, pb_tr = [], []
        for _ in range(3):
            idx = irng.choice(len(tr), size=int(0.8*len(tr)), replace=False)
            cb = ridge_coef(Ztr3[idx].reshape(-1, m), Yr_tr['p'][idx].reshape(-1))
            pb_te.append(Zte @ cb); pb_tr.append(Ztr3 @ cb)
        ite = {'p': pb_te[0], 'v': apply_mag(pb_te[1], Lv), 'a': apply_mag(pb_te[2], La)}
        itr = {'p': pb_tr[0], 'v': apply_mag(pb_tr[1], Lv), 'a': apply_mag(pb_tr[2], La)}
        acc['perf_speci'].append(per_trial_r(perf(ite, itr), Ytrue).mean())
        FteI = np.stack([ztime(ite[o]) for o in ORDERS], axis=-1)
        oofI = np.zeros_like(oofS)
        jrng = np.random.default_rng(9500 + 100*subj + sess + fi)
        for g in range(NINNER):
            iva = inner[g]; itr_i = np.concatenate([inner[k] for k in range(NINNER) if k != g])
            Dm_i = D[tr[itr_i]].mean(0); fl = (D[tr[itr_i]]-Dm_i).reshape(-1, m)
            mui = fl.mean(0); sdi = fl.std(0)+1e-9
            Zi3 = ((D[tr[itr_i]]-Dm_i-mui)/sdi); Zv = (D[tr[iva]]-Dm_i-mui)/sdi
            Ymi = {o: Ys[o][tr[itr_i]].mean(0) for o in ORDERS}
            Bi = ridge_coef(Zi3.reshape(-1, m),
                            np.stack([(Ys[o][tr[itr_i]]-Ymi[o]).reshape(-1) for o in ORDERS], axis=1))
            cfi = {o: Bi[:, j] for j, o in enumerate(ORDERS)}
            ri = {o: to_pos(o, Zi3 @ cfi[o]) for o in ORDERS}
            Lvi = mag_filter(ri['v'], ri['p']); Lai = mag_filter(ri['a'], ri['p'])
            members = []
            for _ in range(3):
                jdx = jrng.choice(len(itr_i), size=int(0.8*len(itr_i)), replace=False)
                cbj = ridge_coef(Zi3[jdx].reshape(-1, m),
                                 (Ys['p'][tr[itr_i]][jdx]-Ymi['p']).reshape(-1))
                members.append(Zv @ cbj)
            for j, o in enumerate(ORDERS):
                sigj = {'p': members[0], 'v': apply_mag(members[1], Lvi), 'a': apply_mag(members[2], Lai)}[o]
                oofI[iva, :, j] = ztime(sigj)
        A = oofI.reshape(-1, 3); G = A.T @ A; G[np.diag_indices_from(G)] += 1.0
        c = np.linalg.solve(G, A.T @ yoof.reshape(-1))
        acc['stack_speci'].append(per_trial_r((FteI.reshape(-1, 3) @ c).reshape(len(te), nt), Ytrue).mean())
        W = fit_band_wiener(oofI, yoof, masks)
        acc['wien_speci'].append(per_trial_r(apply_band_wiener(FteI, W, masks), Ytrue).mean())

        # member-to-member correlation of each triple (conditioning diagnostic)
        for tag, F in [('rho_cross', FteC), ('rho_spec', FteS), ('rho_speci', FteI)]:
            rr = [per_trial_r(F[:, :, i], F[:, :, j]).mean() for i in range(3) for j in range(i+1, 3)]
            acc[tag].append(float(np.mean(rr)))

    return {k: float(np.mean(v)) for k, v in acc.items()}


def _st(recs, a, b):
    da = np.array([r[a] for r in recs]); db = np.array([r[b] for r in recs]); d = da-db
    sj = np.array([r['subj'] for r in recs])
    ds = np.array([d[sj == s].mean() for s in np.unique(sj)])
    return (da.mean(), db.mean(), d.mean(), int((d > 0).sum()), len(d),
            wilcoxon(da, db).pvalue, int((ds > 0).sum()), len(ds), wilcoxon(ds).pvalue)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kin = dict(np.load(f"{SIB}/features/kinematics.npz", allow_pickle=True))
    common = e01.common_channels(); print(f"common grad: {len(common)}", flush=True)
    if mode == "test":
        t0 = time.time(); r = run_session(1, 1, common, kin)
        print(f"Sub1 ses1 ({time.time()-t0:.1f}s)")
        for k, v in r.items(): print(f"   {k:12s} {v:+.4f}")
        for e in ['perf', 'stack', 'wien']:
            print(f"   {e}: cross-spec = {r[f'{e}_cross']-r[f'{e}_spec']:+.4f}")
    else:
        recs = []
        for s in range(1, 10):
            for se in (1, 2):
                t0 = time.time(); r = run_session(s, se, common, kin)
                recs.append(dict(subj=s, sess=se, **r))
                print(f"Sub{s} ses{se} ({time.time()-t0:.1f}s) " +
                      " ".join(f"{e}:{r[f'{e}_cross']-r[f'{e}_spec']:+.4f}" for e in ['perf','stack','wien']),
                      flush=True)
        np.savez(f"{PROJ}/results/estimator_strength.npz", recs=np.array(recs, dtype=object))
        L = ["# Estimator-strength control for the bandwidth-clean residual — script 36 (n = 18)\n",
             "Closes the Limitations hedge that a stronger, noise-covariance-aware estimator might extract "
             "cross-order information a fixed-weight fusion misses. The SAME estimator is applied to the "
             "cross-order triple {a,v,p} and to the bandwidth-matched triple {p, L_v p, L_a p} (zero "
             "cross-order content); the contrast is cross - spec. Zero-phase, full-lag frame (§3.9).\n",
             "| estimator | cross | spec | cross − spec | sessions | subjects |", "|---|---|---|---|---|---|"]
        for e, lab in [('perf', 'fixed performance weights (paper)'),
                       ('stack', 'learned static stack (nested OOF)'),
                       ('wien', 'per-band Wiener/GLS combiner (6 bands)')]:
            for ref, rlab in [('spec', ''), ('speci', ' — vs DECORRELATED bandwidth triple')]:
                ma, mb, md, sp, sn, ps, jp, jn, pj = _st(recs, f'{e}_cross', f'{e}_{ref}')
                L.append(f"| {lab}{rlab} | {ma:.3f} | {mb:.3f} | **{md:+.4f}** | {sp}/{sn}, p={ps:.4f} | "
                         f"{jp}/{jn}, p={pj:.4f} |")
        L.append("")
        L.append("Mean member-to-member correlation of each triple: cross {:.3f} · spec {:.3f} · "
                 "decorrelated-spec {:.3f}.".format(*[np.mean([r[k] for r in recs])
                 for k in ('rho_cross','rho_spec','rho_speci')]))
        L += ["", "Reference: E_bag {:.3f}, p_direct {:.3f}.".format(
            np.mean([r['E_bag'] for r in recs]), np.mean([r['p_direct'] for r in recs])), ""]
        for e in ['stack', 'wien']:
            ma, mb, md, sp, sn, ps, jp, jn, pj = _st(recs, f'{e}_cross', 'E_bag')
            L.append(f"- {e}_cross − E_bag = {md:+.4f} ({sp}/{sn}, p={ps:.4f}; {jp}/{jn} subjects, p={pj:.4f})")
        open(f"{PROJ}/results/estimator_strength.md", "w").write("\n".join(L)+"\n")
        print("\n".join(L)); print("SAVED results/estimator_strength.{npz,md}")
