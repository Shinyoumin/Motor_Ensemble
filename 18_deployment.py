"""
18 — Deployment-realism arm: cross-session transfer, calibration learning curve, per-subject reliability.
[CLINICAL: calibrate-once, few-trial calibration, per-patient benefit]

The published CV never crosses a session boundary, so its numbers are a within-session ceiling, not a
deployment estimate. Three clinically load-bearing questions, all answerable from the 9x2 design:

  CROSS-SESSION TRANSFER (calibrate-once). Train on ALL trials of one session, test on the OTHER
  session of the same subject, no refit (both directions). Does the cross-order fusion advantage over
  same-order bagging survive a session/day change, and what fraction of within-session accuracy is
  retained? This is the honest deployable number.

  CALIBRATION LEARNING CURVE (few-trial calibration). Within session, subsample the training set to
  n_cal in {10,20,40,80,all} trials against a fixed test fold. How many calibration trials does the
  fusion decoder need to reach its plateau, and is the fusion margin already present at small n_cal?

  PER-SUBJECT RELIABILITY (per-patient benefit). Report E_cross - E_bag for EACH subject and EACH of
  its two sessions, with a bootstrap CI over test trials, instead of a single group p-value. A method
  that helps on average but is null/negative in some individuals is not deployable.

Same ridge pipeline, z-scored position-residual per-trial r. Cross-session uses TRAIN-session
standardisation applied to the TEST session (what a deployed decoder actually has).

Run:  python 18_deployment.py test        # sub1 sanity
      python 18_deployment.py full         # -> results/deployment.npz + deployment.md
"""
import sys, time, importlib.util, numpy as np
from scipy.stats import wilcoxon

_spec = importlib.util.spec_from_file_location(
    "ens01", "/mnt/hdd/MEG_Motor/motor_ensemble_claude/scripts/01_ensemble_distance.py")
ens01 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ens01)
load_meg = ens01.load_meg; build_design = ens01.build_design; ridge_coef = ens01.ridge_coef
to_pos = ens01.to_pos; per_trial_r = ens01.per_trial_r; ztime = ens01.ztime
common_channels = ens01.common_channels; ORDERS = ens01.ORDERS; NFOLD = ens01.NFOLD

ROOT = "/mnt/hdd/MEG_Motor"; SIB = f"{ROOT}/motor_comparison_claude"
PROJ = f"{ROOT}/motor_ensemble_claude"
N_BAG = 10; N_CAL = [10, 20, 40, 80, 0]      # 0 = all


def prep(subj, sess, common, kin):
    X, t = load_meg(subj, sess, common)
    D, tw = build_design(X, t)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]) for o in ORDERS}
    return D, Ys


def fit_predict(Dtr, Ytr, Dte, srng, n_cal=0, do_bag=True):
    """Fit routes on (a subsample of) train; predict test. Returns per-trial-r means dict."""
    m = Dtr.shape[2]
    if n_cal and n_cal < Dtr.shape[0]:
        idx = srng.choice(Dtr.shape[0], size=n_cal, replace=False)
        Dtr = Dtr[idx]; Ytr = {o: Ytr[o][idx] for o in ORDERS}
    Dmean = Dtr.mean(0)
    flat = (Dtr - Dmean).reshape(-1, m); mur = flat.mean(0); sdr = flat.std(0) + 1e-9
    Ztr = (flat - mur) / sdr; Ztr3 = Ztr.reshape(len(Dtr), Dtr.shape[1], m)
    Zte = (Dte - Dmean - mur) / sdr
    Ym = {o: Ytr[o].mean(0) for o in ORDERS}
    Yr_tr = {o: Ytr[o] - Ym[o] for o in ORDERS}
    Ystack = np.stack([Yr_tr[o].reshape(-1) for o in ORDERS], axis=1)
    B = ridge_coef(Ztr, Ystack); cf = {o: B[:, i] for i, o in enumerate(ORDERS)}
    rte = {o: to_pos(o, Zte @ cf[o]) for o in ORDERS}
    rtr = {o: to_pos(o, Ztr3 @ cf[o]) for o in ORDERS}
    z = {o: ztime(rte[o]) for o in ORDERS}
    w = np.array([max(0.0, per_trial_r(rtr[o], Yr_tr['p']).mean()) for o in ORDERS])
    w = w / (w.sum() + 1e-12)
    E_cross = w[0]*z['a'] + w[1]*z['v'] + w[2]*z['p']
    out = {'E_cross': E_cross, 'p_direct': rte['p'], 'a_route': rte['a'], 'v_route': rte['v'],
           'Ym_p': Ym['p']}
    if do_bag:
        bag = []
        for _ in range(N_BAG):
            bi = srng.choice(len(Dtr), size=max(2, int(0.8*len(Dtr))), replace=False)
            cb = ridge_coef(Ztr3[bi].reshape(-1, m), Yr_tr['p'][bi].reshape(-1))
            bag.append(ztime(Zte @ cb))
        out['E_bag'] = np.mean(bag, axis=0)
    return out


def eval_r(pred, Yte_p, Ym_p):
    return per_trial_r(pred, Yte_p - Ym_p).mean()


# ---------- cross-session ----------
def cross_session(subj, common, kin):
    D1, Y1 = prep(subj, 1, common, kin); D2, Y2 = prep(subj, 2, common, kin)
    srng = np.random.default_rng(500 + subj)
    res = {}
    for (Dtr, Ytr, Dte, Yte, tag) in [(D1, Y1, D2, Y2, '1to2'), (D2, Y2, D1, Y1, '2to1')]:
        o = fit_predict(Dtr, Ytr, Dte, srng)
        for k in ['E_cross', 'E_bag', 'p_direct', 'a_route', 'v_route']:
            res[f'{k}_{tag}'] = eval_r(o[k], Yte['p'], o['Ym_p'])
    # average the two directions
    for k in ['E_cross', 'E_bag', 'p_direct', 'a_route', 'v_route']:
        res[k] = 0.5*(res[f'{k}_1to2'] + res[f'{k}_2to1'])
    return res


# ---------- within-session (for retention baseline + per-subject + calib) ----------
def within_session(subj, sess, common, kin, calib=False):
    D, Ys = prep(subj, sess, common, kin)
    folds = np.array_split(np.random.default_rng(1000*subj+sess).permutation(D.shape[0]), NFOLD)
    srng = np.random.default_rng(9000 + 100*subj + sess)
    acc = {'E_cross': [], 'E_bag': [], 'p_direct': []}
    ec_te, eb_te = [], []                                   # pooled per-trial for bootstrap CI
    calib_acc = {n: {'E_cross': [], 'E_bag': []} for n in N_CAL} if calib else None
    for fi in range(NFOLD):
        te = folds[fi]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != fi])
        Dtr = D[tr]; Ytr = {o: Ys[o][tr] for o in ORDERS}; Dte = D[te]
        Yte_p = Ys['p'][te]
        o = fit_predict(Dtr, Ytr, Dte, srng)
        acc['E_cross'].append(eval_r(o['E_cross'], Yte_p, o['Ym_p']))
        acc['E_bag'].append(eval_r(o['E_bag'], Yte_p, o['Ym_p']))
        acc['p_direct'].append(eval_r(o['p_direct'], Yte_p, o['Ym_p']))
        ec_te.append(per_trial_r(o['E_cross'], Yte_p - o['Ym_p']))
        eb_te.append(per_trial_r(o['E_bag'], Yte_p - o['Ym_p']))
        if calib:
            for n in N_CAL:
                oc = fit_predict(Dtr, Ytr, Dte, srng, n_cal=n)
                calib_acc[n]['E_cross'].append(eval_r(oc['E_cross'], Yte_p, oc['Ym_p']))
                calib_acc[n]['E_bag'].append(eval_r(oc['E_bag'], Yte_p, oc['Ym_p']))
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out['_ec_trials'] = np.concatenate(ec_te); out['_eb_trials'] = np.concatenate(eb_te)
    if calib:
        out['calib'] = {n: {kk: float(np.mean(vv)) for kk, vv in d.items()} for n, d in calib_acc.items()}
    return out


def _paired(vals_a, vals_b):
    da = np.array(vals_a); db = np.array(vals_b); d = da - db
    try: p = wilcoxon(da, db).pvalue
    except Exception: p = float('nan')
    return da.mean(), db.mean(), d.mean(), int((d > 0).sum()), len(da), p


def _write_report(cs, ws):
    L = ["# Deployment-realism arm — script 18 (9 subjects x 2 sessions)\n"]
    # cross-session
    L += ["## Cross-session transfer (train one session, test the other; no refit)",
          "| metric | within-session | cross-session | retention |",
          "|---|---|---|---|"]
    for k in ['E_cross', 'E_bag', 'p_direct']:      # keys present in BOTH within & cross
        wv = np.mean([w[k] for w in ws])
        cv = np.mean([c[k] for c in cs])
        ret = cv/wv if wv else float('nan')
        L.append(f"| {k} | {wv:.3f} | {cv:.3f} | {ret*100:.0f}% |")
    ecs = [c['E_cross'] for c in cs]; ebs = [c['E_bag'] for c in cs]
    ma, mb, md, pos, n, p = _paired(ecs, ebs)
    _, _, mdp, posp, _, pp = _paired(ecs, [c['p_direct'] for c in cs])
    L += ["",
          f"- **Cross-session E_cross vs E_bag: {ma:.3f} vs {mb:.3f}, {md:+.3f} ({pos}/{n} subjects, "
          f"p = {p:.4f}).**",
          f"- Cross-session E_cross vs p_direct: {mdp:+.3f} ({posp}/{n}, p = {pp:.4f}).",
          f"- Fusion margin retention (cross/within): "
          f"{(md/np.mean([_paired([w['E_cross']],[w['E_bag']])[2] for w in ws])) if False else ''}"]
    within_margin = np.mean([w['E_cross']-w['E_bag'] for w in ws])
    L[-1] = (f"- Fusion margin: within-session {within_margin:+.3f} -> cross-session {md:+.3f} "
             f"({md/within_margin*100:.0f}% retained).")

    # per-subject reliability (within-session margin per session; bootstrap CI if trial arrays present)
    has_ci = '_ec_trials' in ws[0]
    L += ["", "## Per-subject reliability (E_cross - E_bag, each subject, both sessions)",
          "| subject | ses1 margin | ses2 margin | mean |" + (" bootstrap 95% CI |" if has_ci else "")
          + " both>0 |",
          "|---|---|---|---|" + ("---|" if has_ci else "")]
    n_pos_both = 0
    for s in range(1, 10):
        w1 = ws[(s-1)*2]; w2 = ws[(s-1)*2+1]
        m1 = w1['E_cross']-w1['E_bag']; m2 = w2['E_cross']-w2['E_bag']
        both = (m1 > 0) and (m2 > 0); n_pos_both += int(both)
        ci = ""
        if has_ci:
            pool = np.concatenate([w1['_ec_trials']-w1['_eb_trials'], w2['_ec_trials']-w2['_eb_trials']])
            bs = np.array([np.random.default_rng(s*100+b).choice(pool, len(pool)).mean() for b in range(2000)])
            lo, hi = np.percentile(bs, [2.5, 97.5]); ci = f" [{lo:+.3f}, {hi:+.3f}] |"
        L.append(f"| S{s} | {m1:+.3f} | {m2:+.3f} | {(m1+m2)/2:+.3f} |{ci} {'yes' if both else 'NO'} |")
    L.append(f"\n- Fusion (within-session) helps in BOTH sessions of **{n_pos_both}/9** subjects.")

    # calibration curve
    if 'calib' in ws[0]:
        L += ["", "## Calibration learning curve (within-session, fixed test fold)",
              "| n_cal trials | E_cross | E_bag | margin |", "|---|---|---|---|"]
        for n in N_CAL:
            label = 'all(~96)' if n == 0 else str(n)
            ec = np.mean([w['calib'][n]['E_cross'] for w in ws])
            eb = np.mean([w['calib'][n]['E_bag'] for w in ws])
            L.append(f"| {label} | {ec:.3f} | {eb:.3f} | {ec-eb:+.3f} |")
        L.append("\n- Read off how few calibration trials reach most of the plateau and whether the "
                 "fusion margin is already present at small n_cal.")

    L += ["", "## Bottom line",
          ("**Calibrate-once viable.**" if (md > 0 and p < 0.05) else "**Needs per-session recalibration.**")
          + f" Cross-order fusion {'retains' if md>0 else 'loses'} its advantage over same-order bagging "
            f"across a session change ({md:+.3f}, {pos}/{n}, p = {p:.4f}), "
          + (f"and helps both sessions of {n_pos_both}/9 subjects." if md>0 else "and is unreliable per-subject.")]
    txt = "\n".join(L) + "\n"
    with open(f"{PROJ}/results/deployment.md", "w") as fh:
        fh.write(txt)
    print("\n" + txt)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kin = dict(np.load(f"{SIB}/features/kinematics.npz", allow_pickle=True))
    common = common_channels(); print(f"common grad: {len(common)}", flush=True)
    if mode == "test":
        t0 = time.time(); c = cross_session(1, common, kin)
        w = within_session(1, 1, common, kin, calib=True)
        print(f"Sub1 ({time.time()-t0:.1f}s)")
        print(f"  cross-session: E_cross={c['E_cross']:.3f} E_bag={c['E_bag']:.3f} "
              f"p_direct={c['p_direct']:.3f}")
        print(f"  within ses1:   E_cross={w['E_cross']:.3f} E_bag={w['E_bag']:.3f}")
        print(f"  calib margins:", {('all' if n==0 else n): round(w['calib'][n]['E_cross']-w['calib'][n]['E_bag'],3) for n in N_CAL})
    elif mode == "full":
        cs, ws = [], []
        for s in range(1, 10):
            t0 = time.time(); c = cross_session(s, common, kin); cs.append(dict(subj=s, **c))
            for se in (1, 2):
                ws.append(within_session(s, se, common, kin, calib=True))
            print(f"Sub{s} ({time.time()-t0:.1f}s) cross Ecross={c['E_cross']:.3f} "
                  f"Ebag={c['E_bag']:.3f}", flush=True)
        np.savez(f"{PROJ}/results/deployment.npz",
                 cs=np.array(cs, dtype=object),
                 ws=np.array([{k: v for k, v in w.items() if not k.startswith('_')} for w in ws], dtype=object))
        print("SAVED results/deployment.npz")
        _write_report(cs, ws)
