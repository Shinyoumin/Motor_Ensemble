"""
45 — Implementation robustness of the bandwidth-matched pseudo-fusion (revision, reviewer 2.1 / 1).

The reviewers asked for the estimation details of E_spec and, implicitly, for evidence that the
central null does not depend on them. The published filter is

    |L_o(f)| = sqrt( mean_trials |FFT(route_tr[o])|^2 / ( mean_trials |FFT(route_tr[p])|^2 + eps ) )

estimated on TRAIN-fold route predictions only and applied by multiplication in the frequency domain,
i.e. a zero-phase circular convolution over the 2.8 s analysis window with eps = 1e-30.

Two implementation choices in that line are arbitrary and are varied here:

  (A) REGULARISATION of the denominator at frequencies with little power. eps is swept as a fraction
      of the mean denominator power: eps = c * mean(den), c in {0 (published), 1e-3, 1e-2, 1e-1}.
      A frequency-domain floor on the gain (|L| clipped to <= g_max) is also tested.
  (B) BOUNDARY handling. Circular convolution wraps the window edges. We repeat the control with
      zero-padding to twice the window length (linear convolution) and with a Tukey taper applied
      before the transform and undone after.

Reported quantity: the bandwidth-clean residual E_cross - E_spec, at the participant level (n = 9),
which is the paper's central null. Everything else (folds, ridge, routes, weights) is identical to
script 17 and uses the same deterministic per-session seeds.

Run:  python 45_espec_implementation_controls.py test   # sub 1 ses 1
      python 45_espec_implementation_controls.py full   # 9x2 -> results/espec_implementation.{npz,md}
"""
import sys, time, importlib.util, numpy as np
from scipy.stats import wilcoxon
from scipy.signal import windows

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"
_s = importlib.util.spec_from_file_location("ens01", f"{PROJ}/scripts/01_ensemble_distance.py")
ens01 = importlib.util.module_from_spec(_s); _s.loader.exec_module(ens01)
_s2 = importlib.util.spec_from_file_location("sc17", f"{PROJ}/scripts/17_spectral_control.py")
sc17 = importlib.util.module_from_spec(_s2); _s2.loader.exec_module(sc17)

load_meg, build_design, to_pos = ens01.load_meg, ens01.build_design, ens01.to_pos
per_trial_r, ztime, common_channels = ens01.per_trial_r, ens01.ztime, ens01.common_channels
ORDERS, NFOLD = ens01.ORDERS, ens01.NFOLD
ridge_coef = sc17.ridge_coef if hasattr(sc17, "ridge_coef") else ens01.ridge_coef

EPS_FRACS = [0.0, 1e-3, 1e-2, 1e-1]          # denominator regularisation, as a fraction of mean power
GMAX = 4.0                                    # gain ceiling for the clipped variant


def mag_filter(train_route, train_p, eps_frac=0.0, gmax=None, pad=1, taper=None):
    """Magnitude filter with explicit regularisation and boundary handling."""
    n = train_route.shape[-1] * pad
    r, p = train_route, train_p
    if taper is not None:
        r, p = r * taper, p * taper
    Rr = np.fft.rfft(r, n=n, axis=-1); Rp = np.fft.rfft(p, n=n, axis=-1)
    num = (np.abs(Rr) ** 2).mean(0); den = (np.abs(Rp) ** 2).mean(0)
    den = den + (eps_frac * den.mean() if eps_frac else 1e-30)
    L = np.sqrt(num / den)
    if gmax is not None:
        L = np.minimum(L, gmax)
    return L


def apply_mag(x, L, pad=1):
    """The taper is a spectral-estimation device only; the filter itself is applied untapered."""
    n = x.shape[-1] * pad
    return np.fft.irfft(np.fft.rfft(x, n=n, axis=-1) * L, n=n, axis=-1)[..., :x.shape[-1]]


VARIANTS = (
    [("published (eps=1e-30, circular)", dict(eps_frac=0.0, pad=1, taper=False, gmax=None))]
    + [(f"eps = {c:g} x mean power", dict(eps_frac=c, pad=1, taper=False, gmax=None)) for c in EPS_FRACS[1:]]
    + [(f"gain clipped at {GMAX:g}", dict(eps_frac=0.0, pad=1, taper=False, gmax=GMAX)),
       ("zero-padded (linear conv.)", dict(eps_frac=0.0, pad=2, taper=False, gmax=None)),
       ("Tukey taper in the spectral estimate (alpha = 0.1)", dict(eps_frac=0.0, pad=1, taper=True, gmax=None)),
       ("zero-pad + eps = 1e-2", dict(eps_frac=1e-2, pad=2, taper=False, gmax=None))]
)


def run_session(subj, sess, common, kin, X=None, t=None):
    if X is None:
        X, t = load_meg(subj, sess, common)
    D, tw = build_design(X, t)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]) for o in ORDERS}
    ntr_all, nt, m = D.shape
    frng = np.random.default_rng(1000 * subj + sess)
    folds = np.array_split(frng.permutation(ntr_all), NFOLD)
    taper = windows.tukey(nt, 0.1)

    out = {nm: [] for nm, _ in VARIANTS}; out['E_cross'] = []
    for f in range(NFOLD):
        te = folds[f]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != f])
        Dmean = D[tr].mean(0)
        flat = (D[tr] - Dmean).reshape(-1, m); mu = flat.mean(0); sd = flat.std(0) + 1e-9
        Ztr = (flat - mu) / sd; Ztr3 = Ztr.reshape(len(tr), nt, m)
        Zte = (D[te] - Dmean - mu) / sd
        Ym = {o: Ys[o][tr].mean(0) for o in ORDERS}
        Yr_tr = {o: Ys[o][tr] - Ym[o] for o in ORDERS}
        Ytrue = Ys['p'][te] - Ym['p']

        Ystack = np.stack([Yr_tr[o].reshape(-1) for o in ORDERS], axis=1)
        B = ens01.ridge_coef(Ztr, Ystack)
        coef = {o: B[:, i] for i, o in enumerate(ORDERS)}
        rte = {o: to_pos(o, Zte @ coef[o]) for o in ORDERS}
        rtr = {o: to_pos(o, Ztr3 @ coef[o]) for o in ORDERS}

        w = np.array([max(per_trial_r(rtr[o], Yr_tr['p']).mean(), 0) for o in ORDERS])
        w = w / (w.sum() + 1e-30)
        out['E_cross'].append(per_trial_r(
            sum(w[i] * ztime(rte[o]) for i, o in enumerate(ORDERS)), Ytrue).mean())

        for nm, kw in VARIANTS:
            tp = taper if kw['taper'] else None
            Lv = mag_filter(rtr['v'], rtr['p'], kw['eps_frac'], kw['gmax'], kw['pad'], tp)
            La = mag_filter(rtr['a'], rtr['p'], kw['eps_frac'], kw['gmax'], kw['pad'], tp)
            spec_te = {'p': rte['p'],
                       'v': apply_mag(rte['p'], Lv, kw['pad']),
                       'a': apply_mag(rte['p'], La, kw['pad'])}
            spec_tr = {'p': rtr['p'],
                       'v': apply_mag(rtr['p'], Lv, kw['pad']),
                       'a': apply_mag(rtr['p'], La, kw['pad'])}
            ws = np.array([max(per_trial_r(spec_tr[o], Yr_tr['p']).mean(), 0) for o in ORDERS])
            ws = ws / (ws.sum() + 1e-30)
            out[nm].append(per_trial_r(
                sum(ws[i] * ztime(spec_te[o]) for i, o in enumerate(ORDERS)), Ytrue).mean())
    return {k: float(np.mean(v)) for k, v in out.items()}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    common = common_channels()
    kin = dict(np.load(f"{PROJ}/../motor_comparison_claude/features/kinematics.npz", allow_pickle=True))
    if mode == "test":
        t0 = time.time(); r = run_session(1, 1, common, kin)
        print(f"sub1 ses1 ({time.time()-t0:.1f}s)")
        for k, v in r.items(): print(f"   {k:34s} {v:+.4f}")
        sys.exit()

    recs = []
    for s in range(1, 10):
        for se in (1, 2):
            t0 = time.time(); r = run_session(s, se, common, kin)
            r.update(subj=s, sess=se); recs.append(r)
            print(f"Sub{s} ses{se} ({time.time()-t0:.1f}s) E_cross={r['E_cross']:+.4f}", flush=True)
    np.savez(f"{PROJ}/results/espec_implementation.npz", recs=np.array(recs, dtype=object))

    def subj(key):
        sj = np.array([r['subj'] for r in recs]); v = np.array([r[key] for r in recs])
        return np.array([v[sj == u].mean() for u in np.unique(sj)])

    ec = subj('E_cross')
    lines = ["# E_spec implementation robustness — script 45 (n = 18 sessions / 9 participants)", "",
             "Participant-level bandwidth-clean residual E_cross − E_spec under variations of the",
             "denominator regularisation and the boundary handling of the magnitude-filter estimate.",
             "Folds, ridge and weighting are identical to the main analysis.", "",
             "| E_spec variant | E_spec | E_cross − E_spec | participants > 0 | *p* |",
             "|---|---|---|---|---|"]
    for nm, _ in VARIANTS:
        es = subj(nm); d = ec - es
        lines.append(f"| {nm} | {es.mean():.4f} | **{d.mean():+.4f}** | {int((d>0).sum())}/9 | "
                     f"{wilcoxon(d).pvalue:.3f} |")
    lines += ["", f"E_cross = {ec.mean():.4f} (participant mean)."]
    open(f"{PROJ}/results/espec_implementation.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[-len(VARIANTS)-3:]))
