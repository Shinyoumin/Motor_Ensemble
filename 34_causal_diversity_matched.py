"""
34 — Causal decorrelation-matched control (the analogue of script 15 for the causal frame).

Script 33 showed that under a strictly causal, zero-look-ahead budget the cross-order margin is NOT
reproduced by a bandwidth-matched pseudo-fusion anchored on either route, leaving a positive residual
(+0.031 at tau=0). PAPER.md §3.10 states that this residual is unresolved between (i) genuine
cross-order information, (ii) route hedging, and (iii) generic estimator decorrelation, and that
"separating those would need a causal analogue of the decorrelation-matched control of §2.4(ii),
which we did not run". This script runs it.

Construction (mirrors script 15, but causal and anchored on the CARRIER route):
  * the same-order family decodes VELOCITY (the strong route at tau=0) on random feature subspaces
    (fraction f of the lag x sensor columns per member; Ho's random-subspace principle), each member
    causally integrated into position space and scaled by its TRAIN std;
  * sweeping f trades member strength against member diversity, measured as the mean pairwise
    per-trial correlation of member predictions (rho_same);
  * the decisive comparison is at f* where rho_same matches the cross-order routes' own rho_cross:
    if E_cross still beats the diversity-matched same-order ensemble there, the causal residual is
    not generic decorrelation.

Everything else (causal band-pass, causal integration + 0.1 Hz high-pass, train-fixed scales, folds,
fusion weights) is reused unchanged from script 25 via import, so the numbers are paired with
scripts 25/33.

Run:  .../.venv_meggpt/bin/python scripts/34_causal_diversity_matched.py test
      .../.venv_meggpt/bin/python scripts/34_causal_diversity_matched.py full
"""
import sys, time, importlib.util, numpy as np
from scipy.stats import wilcoxon

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"
_s = importlib.util.spec_from_file_location("c25", f"{PROJ}/scripts/25_causal_pipeline.py")
c25 = importlib.util.module_from_spec(_s); _s.loader.exec_module(c25)
per_trial_r = c25.per_trial_r; ORDERS = c25.ORDERS; NFOLD = c25.NFOLD; SIB = c25.SIB
N_MEM = 10
FRACS = [1.0, 0.5, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01]  # extended: causal rho_cross is much lower than zero-phase, so the sweep must reach it


def mean_pairwise_corr(preds):
    vals = [per_trial_r(preds[i], preds[j]).mean()
            for i in range(len(preds)) for j in range(i + 1, len(preds))]
    return float(np.mean(vals))


def run_session(subj, sess, common, kin, lags_ms):
    X, t, sf = c25.load_meg_causal(subj, sess, common)
    D, tw = c25.build_design_lags(X, t, lags_ms, sf)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]) for o in ORDERS}
    ntr_all, nt, m = D.shape
    folds = np.array_split(np.random.default_rng(1000*subj + sess).permutation(ntr_all), NFOLD)
    srng = np.random.default_rng(7000 + 100*subj + sess)

    acc = {k: [] for k in ['E_cross', 'E_bagV', 'rho_cross', 'rho_bagV',
                           'a_route', 'v_route', 'p_direct']}
    for f in FRACS:
        for k in (f'same_acc@{f}', f'same_rho@{f}', f'same_mem@{f}'):
            acc[k] = []

    for fi in range(NFOLD):
        te = folds[fi]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != fi])
        Dmean = D[tr].mean(0)
        flat_r = (D[tr]-Dmean).reshape(-1, m); mur = flat_r.mean(0); sdr = flat_r.std(0)+1e-9
        Ztr = (flat_r-mur)/sdr; Ztr3 = Ztr.reshape(len(tr), nt, m)
        Zte = (D[te]-Dmean-mur)/sdr
        Ym = {o: Ys[o][tr].mean(0) for o in ORDERS}
        Yr_tr = {o: (Ys[o][tr]-Ym[o]) for o in ORDERS}
        Ytrue = Ys['p'][te]-Ym['p']

        Ystack = np.stack([Yr_tr[o].reshape(-1) for o in ORDERS], axis=1)
        B = c25.ridge_coef(Ztr, Ystack); cf = {o: B[:, i] for i, o in enumerate(ORDERS)}
        rte = {o: c25.to_pos_causal(o, Zte @ cf[o]) for o in ORDERS}
        rtr = {o: c25.to_pos_causal(o, Ztr3 @ cf[o]) for o in ORDERS}
        for o, key in [('a', 'a_route'), ('v', 'v_route'), ('p', 'p_direct')]:
            acc[key].append(per_trial_r(rte[o], Ytrue).mean())
        acc['E_cross'].append(per_trial_r(
            c25.fuse_fixedscale(rte, rtr, ORDERS, Yr_tr['p']), Ytrue).mean())
        # cross-order diversity, on train-fixed-scaled predictions (causal analogue of 15's ztime)
        sc = {o: rtr[o].std()+1e-12 for o in ORDERS}
        acc['rho_cross'].append(mean_pairwise_corr([rte[o]/sc[o] for o in ORDERS]))

        # reference: velocity bagging (script 33's E_bagV)
        bagV = []
        for _ in range(N_MEM):
            idx = srng.choice(len(tr), size=int(0.8*len(tr)), replace=False)
            cb = c25.ridge_coef(Ztr3[idx].reshape(-1, m), Yr_tr['v'][idx].reshape(-1))
            vte = c25.to_pos_causal('v', Zte @ cb); vtr = c25.to_pos_causal('v', Ztr3[idx] @ cb)
            bagV.append(vte/(vtr.std()+1e-12))
        acc['E_bagV'].append(per_trial_r(np.mean(bagV, axis=0), Ytrue).mean())
        acc['rho_bagV'].append(mean_pairwise_corr(bagV))

        # same-order (velocity) random-subspace ensembles across the diversity sweep
        for f in FRACS:
            k = max(1, int(round(f*m)))
            members, mem_acc = [], []
            for _ in range(N_MEM):
                cols = srng.choice(m, size=k, replace=False)
                cb = c25.ridge_coef(Ztr3[:, :, cols].reshape(-1, k), Yr_tr['v'].reshape(-1))
                vte = c25.to_pos_causal('v', Zte[:, :, cols] @ cb)
                vtr = c25.to_pos_causal('v', Ztr3[:, :, cols] @ cb)
                members.append(vte/(vtr.std()+1e-12))
                mem_acc.append(per_trial_r(vte, Ytrue).mean())
            acc[f'same_acc@{f}'].append(per_trial_r(np.mean(members, axis=0), Ytrue).mean())
            acc[f'same_rho@{f}'].append(mean_pairwise_corr(members))
            acc[f'same_mem@{f}'].append(float(np.mean(mem_acc)))

    return {k: float(np.mean(v)) for k, v in acc.items()}


def _paired(recs, a, b):
    da = np.array([r[a] for r in recs]); db = np.array([r[b] for r in recs]); d = da-db
    subj = np.array([r['subj'] for r in recs])
    ds = np.array([d[subj == s].mean() for s in np.unique(subj)])
    ps = wilcoxon(da, db).pvalue if np.any(d) else 1.0
    pj = wilcoxon(ds).pvalue if np.any(ds) else 1.0
    return d.mean(), int((d > 0).sum()), len(d), ps, int((ds > 0).sum()), len(ds), pj


def report(tag, recs):
    rho_cross = np.mean([r['rho_cross'] for r in recs])
    curve = [(f, np.mean([r[f'same_rho@{f}'] for r in recs]),
              np.mean([r[f'same_acc@{f}'] for r in recs]),
              np.mean([r[f'same_mem@{f}'] for r in recs])) for f in FRACS]
    f_star = min(curve, key=lambda c: abs(c[1]-rho_cross))[0]
    L = [f"### {tag}", "",
         f"Cross-order routes: rho_cross = **{rho_cross:+.3f}**, E_cross = "
         f"**{np.mean([r['E_cross'] for r in recs]):.3f}**. Velocity bag: rho = "
         f"{np.mean([r['rho_bagV'] for r in recs]):+.3f}, acc = {np.mean([r['E_bagV'] for r in recs]):.3f}. "
         f"Single routes: a {np.mean([r['a_route'] for r in recs]):.3f} · "
         f"v {np.mean([r['v_route'] for r in recs]):.3f} · p {np.mean([r['p_direct'] for r in recs]):.3f}.", "",
         "| feature frac f | member acc | mean pairwise rho | ensemble acc | vs E_cross |",
         "|---|---|---|---|---|"]
    for f, rho, a_, mem in curve:
        mark = "  **<- matched**" if f == f_star else ""
        L.append(f"| {f:.2f} | {mem:.3f} | {rho:+.3f} | {a_:.3f} | "
                 f"{a_-np.mean([r['E_cross'] for r in recs]):+.3f}{mark} |")
    L.append("")
    for a, b, lab in [('E_cross', f'same_acc@{f_star}', f'E_cross − diversity-matched same-order (f={f_star})'),
                      ('E_cross', 'E_bagV', 'E_cross − E_bagV (reference, script 33)')]:
        md, sp, sn, ps, jp, jn, pj = _paired(recs, a, b)
        L.append(f"- **{lab}: {md:+.4f}** ({sp}/{sn} sessions, p = {ps:.4f}; {jp}/{jn} participants, p = {pj:.4f})")
    best = max(curve, key=lambda c: c[2])
    md, sp, sn, ps, jp, jn, pj = _paired(recs, 'E_cross', f'same_acc@{best[0]}')
    L.append(f"- Envelope (best same-order ensemble at ANY diversity, f = {best[0]}, acc {best[2]:.3f}): "
             f"E_cross − best = {md:+.4f} ({sp}/{sn}, p = {ps:.4f}; {jp}/{jn} participants, p = {pj:.4f})")
    return "\n".join(L)+"\n"


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kin = dict(np.load(f"{SIB}/features/kinematics.npz", allow_pickle=True))
    common = c25.common_channels(); print(f"common grad: {len(common)}", flush=True)
    CAUSAL0 = [-100, -50, 0]
    if mode == "test":
        t0 = time.time(); r = run_session(1, 1, common, kin, CAUSAL0)
        print(f"[tau=0] Sub1 ses1 ({time.time()-t0:.1f}s) E_cross={r['E_cross']:.3f} "
              f"rho_cross={r['rho_cross']:+.3f} E_bagV={r['E_bagV']:.3f}")
        for f in FRACS:
            print(f"   f={f:.2f} rho={r[f'same_rho@{f}']:+.3f} acc={r[f'same_acc@{f}']:.3f} "
                  f"mem={r[f'same_mem@{f}']:.3f}")
    else:
        recs = []
        for s in range(1, 10):
            for se in (1, 2):
                t0 = time.time(); r = run_session(s, se, common, kin, CAUSAL0)
                recs.append(dict(subj=s, sess=se, **r))
                print(f"[tau0] Sub{s} ses{se} ({time.time()-t0:.1f}s) Ec={r['E_cross']:.3f} "
                      f"rho_cross={r['rho_cross']:+.3f} EbV={r['E_bagV']:.3f}", flush=True)
        np.savez(f"{PROJ}/results/causal_diversity_matched.npz", recs=np.array(recs, dtype=object))
        md = ["# Causal decorrelation-matched control — script 34 (n = 18 sessions / 9 subjects)\n",
              "Causal analogue of script 15, anchored on the velocity route (the carrier at tau=0). Same "
              "causal operators, folds and fusion as scripts 25/33. Same-order members decode velocity on "
              "random feature subspaces and are causally integrated into position space; the decisive "
              "comparison is at the feature fraction whose member-prediction diversity matches the "
              "cross-order routes' own.\n",
              report("Causal operators, tau = 0 (zero look-ahead)", recs)]
        open(f"{PROJ}/results/causal_diversity_matched.md", "w").write("\n".join(md))
        print("\n".join(md)); print("SAVED results/causal_diversity_matched.{npz,md}")
