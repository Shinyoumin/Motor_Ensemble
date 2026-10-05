"""
33 — Velocity-anchored causal bandwidth control (closes the deferred causal-mechanism question).

Script 25 showed the fusion-over-bagging margin survives a fully causal, zero-look-ahead pipeline
(tau=0: E_cross - E_bag = +0.044), but its bandwidth control E_spec could NOT re-adjudicate the
mechanism there, because E_spec is built entirely from p_direct and under a tight causal budget
p_direct is a WEAK route while the velocity route becomes the carrier (v 0.133 > p 0.120 > a 0.062).
A p-anchored control cannot reproduce a velocity-route advantage, so the positive causal residual it
leaves is uninformative. PAPER.md (§3.10, Supplementary S1) states this openly and defers the proper
test: a bandwidth control anchored on the STRONG causal route.

This script runs that test. Everything is reused from script 25 (same causal operators, same folds,
same fixed-scale fusion), and two velocity-anchored controls are added:

  E_bagV   — same-order bagging of TEN velocity decoders (each integrated causally into position
             space). The correct same-order variance control when velocity is the carrier.
  E_specV  — velocity-anchored bandwidth-matched pseudo-fusion: {v_route, F_p|v(v_route),
             F_a|v(v_route)} where F_x|v is a minimum-phase causal FIR matching route x's magnitude
             spectrum relative to the velocity route (estimated on TRAIN). All three members are
             deterministic filterings of ONE decoder, so it carries ZERO cross-order neural content
             while holding the ensemble's bandwidth diversity.
  E_specV_d — additionally bulk-delay-matched (magnitude + causal delay). NOTE the honest asymmetry:
             velocity is read out LATER than position, so reproducing p's timing from v would need a
             negative delay, i.e. an advance, which a causal operator cannot do (delays are clipped at
             0). A v-anchored control can therefore match bandwidth but not p's lead.

Decisive contrast: E_cross - E_specV at tau=0.
  ~0  -> the causal margin is operator (bandwidth) diversity, exactly as in the zero-phase frame.
  >0  -> under causality there IS a cross-order (route-importance) benefit beyond bandwidth, and the
         paper must say so.

Run:  .../.venv_meggpt/bin/python scripts/33_causal_velocity_anchor.py test
      .../.venv_meggpt/bin/python scripts/33_causal_velocity_anchor.py full
"""
import sys, time, importlib.util, numpy as np
from scipy.stats import wilcoxon

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"
_s = importlib.util.spec_from_file_location("c25", f"{PROJ}/scripts/25_causal_pipeline.py")
c25 = importlib.util.module_from_spec(_s); _s.loader.exec_module(c25)

np_r = c25.per_trial_r
ORDERS = c25.ORDERS
NFOLD, N_BAG, SIB = c25.NFOLD, c25.N_BAG, c25.SIB


def run_session(subj, sess, common, kin, lags_ms):
    """Script 25's causal session, plus the velocity-anchored controls."""
    X, t, sf = c25.load_meg_causal(subj, sess, common)
    D, tw = c25.build_design_lags(X, t, lags_ms, sf)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]) for o in ORDERS}
    ntr_all, nt, m = D.shape
    folds = np.array_split(np.random.default_rng(1000*subj+sess).permutation(ntr_all), NFOLD)
    srng = np.random.default_rng(7000 + 100*subj + sess)          # same bag seeds as script 25
    keys = ['E_cross', 'E_bag', 'E_spec', 'E_bagV', 'E_specV', 'E_specV_d',
            'p_direct', 'a_route', 'v_route', 'tau_pv', 'tau_av']
    acc = {k: [] for k in keys}

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
            acc[key].append(np_r(rte[o], Ytrue).mean())
        acc['E_cross'].append(np_r(c25.fuse_fixedscale(rte, rtr, ORDERS, Yr_tr['p']), Ytrue).mean())

        # --- reference controls (script 25, p-anchored) ---
        bag = []
        for _ in range(N_BAG):
            idx = srng.choice(len(tr), size=int(0.8*len(tr)), replace=False)
            Zb = Ztr3[idx].reshape(-1, m); cb = c25.ridge_coef(Zb, Yr_tr['p'][idx].reshape(-1))
            bag.append((Zte @ cb) / ((Ztr3[idx] @ cb).std()+1e-12))
        acc['E_bag'].append(np_r(np.mean(bag, axis=0), Ytrue).mean())
        hv = c25.causal_minphase_fir(rtr['v'], rtr['p'], sf); ha = c25.causal_minphase_fir(rtr['a'], rtr['p'], sf)
        ps_te = {'p': rte['p'], 'v': c25.apply_causal_fir(rte['p'], hv), 'a': c25.apply_causal_fir(rte['p'], ha)}
        ps_tr = {'p': rtr['p'], 'v': c25.apply_causal_fir(rtr['p'], hv), 'a': c25.apply_causal_fir(rtr['p'], ha)}
        acc['E_spec'].append(np_r(c25.fuse_fixedscale(ps_te, ps_tr, ORDERS, Yr_tr['p']), Ytrue).mean())

        # --- NEW (1): same-order VELOCITY bagging, integrated causally into position space ---
        bagV = []
        srngV = np.random.default_rng(9000 + 100*subj + sess + fi)
        for _ in range(N_BAG):
            idx = srngV.choice(len(tr), size=int(0.8*len(tr)), replace=False)
            Zb = Ztr3[idx].reshape(-1, m); cb = c25.ridge_coef(Zb, Yr_tr['v'][idx].reshape(-1))
            vb_te = c25.to_pos_causal('v', Zte @ cb); vb_tr = c25.to_pos_causal('v', Ztr3[idx] @ cb)
            bagV.append(vb_te / (vb_tr.std()+1e-12))
        acc['E_bagV'].append(np_r(np.mean(bagV, axis=0), Ytrue).mean())

        # --- NEW (2): velocity-anchored bandwidth-matched pseudo-fusion ---
        hp_v = c25.causal_minphase_fir(rtr['p'], rtr['v'], sf)     # magnitude of p relative to v
        ha_v = c25.causal_minphase_fir(rtr['a'], rtr['v'], sf)     # magnitude of a relative to v
        qv_te = {'v': rte['v'], 'p': c25.apply_causal_fir(rte['v'], hp_v),
                 'a': c25.apply_causal_fir(rte['v'], ha_v)}
        qv_tr = {'v': rtr['v'], 'p': c25.apply_causal_fir(rtr['v'], hp_v),
                 'a': c25.apply_causal_fir(rtr['v'], ha_v)}
        acc['E_specV'].append(np_r(c25.fuse_fixedscale(qv_te, qv_tr, ORDERS, Yr_tr['p']), Ytrue).mean())

        # --- NEW (3): magnitude + bulk causal delay (delays clipped at 0; see docstring) ---
        t_pv = c25.est_delay(rtr['p'], rtr['v'], sf); t_av = c25.est_delay(rtr['a'], rtr['v'], sf)
        acc['tau_pv'].append(t_pv/sf*1000); acc['tau_av'].append(t_av/sf*1000)
        qd_te = {'v': rte['v'], 'p': c25.causal_delay(qv_te['p'], t_pv), 'a': c25.causal_delay(qv_te['a'], t_av)}
        qd_tr = {'v': rtr['v'], 'p': c25.causal_delay(qv_tr['p'], t_pv), 'a': c25.causal_delay(qv_tr['a'], t_av)}
        acc['E_specV_d'].append(np_r(c25.fuse_fixedscale(qd_te, qd_tr, ORDERS, Yr_tr['p']), Ytrue).mean())

    return {k: float(np.mean(v)) for k, v in acc.items()}


def _paired(recs, a, b):
    da = np.array([r[a] for r in recs]); db = np.array([r[b] for r in recs]); d = da-db
    subj = np.array([r['subj'] for r in recs])
    dss = np.array([d[subj == s].mean() for s in np.unique(subj)])
    ps = wilcoxon(da, db).pvalue if np.any(d) else 1.0
    pj = wilcoxon(dss).pvalue if np.any(dss) else 1.0
    return da.mean(), db.mean(), d.mean(), int((d > 0).sum()), len(recs), ps, int((dss > 0).sum()), len(dss), pj


CONTRASTS = [
    ('E_cross', 'E_bagV', 'E_cross − E_bagV (fusion vs SAME-ORDER VELOCITY bagging)'),
    ('E_specV', 'E_bagV', 'E_specV − E_bagV (v-anchored bandwidth-only)'),
    ('E_cross', 'E_specV', '**E_cross − E_specV (bandwidth-clean residual, v-anchored)**'),
    ('E_cross', 'E_specV_d', 'E_cross − E_specV(mag+delay) (residual vs operator-match)'),
    ('E_cross', 'E_bag', 'E_cross − E_bag (reference: p-anchored bagging, script 25)'),
    ('E_cross', 'E_spec', 'E_cross − E_spec (reference: p-anchored bandwidth, script 25)'),
]


def _block(title, recs):
    L = [f"### {title}", "",
         "Single routes (mean per-trial r): " + " · ".join(
             f"{k} {np.mean([r[k] for r in recs]):.3f}" for k in ['a_route', 'v_route', 'p_direct']) +
         f". Ensembles: E_cross {np.mean([r['E_cross'] for r in recs]):.3f} · "
         f"E_bagV {np.mean([r['E_bagV'] for r in recs]):.3f} · "
         f"E_specV {np.mean([r['E_specV'] for r in recs]):.3f}. "
         f"Bulk delay match (clipped causal): tau_p|v {np.mean([r['tau_pv'] for r in recs]):.0f} ms, "
         f"tau_a|v {np.mean([r['tau_av'] for r in recs]):.0f} ms.", "",
         "| contrast | mean A | mean B | diff | sessions | subjects |", "|---|---|---|---|---|---|"]
    for a, b, lab in CONTRASTS:
        ma, mb, md, sp, sn, ps, jp, jn, pj = _paired(recs, a, b)
        L.append(f"| {lab} | {ma:.3f} | {mb:.3f} | **{md:+.4f}** | {sp}/{sn}, p={ps:.4f} | {jp}/{jn}, p={pj:.4f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kin = dict(np.load(f"{SIB}/features/kinematics.npz", allow_pickle=True))
    common = c25.common_channels(); print(f"common grad: {len(common)}", flush=True)
    FULL = list(range(-100, 175, 50)); CAUSAL0 = [-100, -50, 0]

    if mode == "test":
        t0 = time.time(); r = run_session(1, 1, common, kin, CAUSAL0)
        print(f"[tau=0] Sub1 ses1 ({time.time()-t0:.1f}s)")
        for k, v in r.items(): print(f"   {k:10s} {v:+.4f}")
        print(f"   E_cross-E_bagV  {r['E_cross']-r['E_bagV']:+.4f}")
        print(f"   E_specV-E_bagV  {r['E_specV']-r['E_bagV']:+.4f}")
        print(f"   E_cross-E_specV {r['E_cross']-r['E_specV']:+.4f}")
    else:
        out = {"C": [], "B": []}
        for tag, lags in [("C", CAUSAL0), ("B", FULL)]:
            for s in range(1, 10):
                for se in (1, 2):
                    t0 = time.time(); r = run_session(s, se, common, kin, lags)
                    out[tag].append(dict(subj=s, sess=se, **r))
                    print(f"[{tag}] Sub{s} ses{se} ({time.time()-t0:.1f}s) "
                          f"Ec={r['E_cross']:.3f} EbV={r['E_bagV']:.3f} EsV={r['E_specV']:.3f} "
                          f"tot={r['E_cross']-r['E_bagV']:+.3f} resV={r['E_cross']-r['E_specV']:+.3f}",
                          flush=True)
        np.savez(f"{PROJ}/results/causal_velocity_anchor.npz",
                 recs_tau0=np.array(out["C"], dtype=object), recs_full=np.array(out["B"], dtype=object))
        md = ["# Velocity-anchored causal bandwidth control — script 33 (n = 18 sessions / 9 subjects)\n",
              "Closes the causal-mechanism question deferred in PAPER.md §3.10 / Supplementary S1. Same "
              "causal pipeline, folds and fusion as script 25; the controls are re-anchored on the "
              "velocity route, which is the carrier under a tight causal budget. E_bagV = ten bagged "
              "velocity decoders (causally integrated); E_specV = {v, F_p|v(v), F_a|v(v)} with "
              "minimum-phase causal FIRs matching the p- and a-route magnitude spectra relative to v — "
              "zero cross-order neural content. Causal delays are clipped at 0, so a v-anchored control "
              "can match bandwidth but cannot reproduce position's temporal LEAD.\n",
              _block("C. Causal operators, tau = 0 (genuine real-time, zero look-ahead)", out["C"]),
              _block("B. Causal operators, full lags [-100..150]", out["B"])]
        open(f"{PROJ}/results/causal_velocity_anchor.md", "w").write("\n".join(md))
        print("\n".join(md))
        print("SAVED results/causal_velocity_anchor.{npz,md}")
