"""
29 — Physics-consistency λ sweep + auxiliary-supervision negative controls (reviewer point 6).

Script 08 tested a SINGLE consistency weight (λc=0.3) and a single aux weight; "physics consistency
does not help" cannot rest on one λ, and "auxiliary supervision helps" needs a control that the benefit
requires REAL cross-order labels (not just an extra output head regularising the trunk).

This script, on script 08's differentiable multi-order decoder (same trunk/heads/integration graph):

(A) λc SWEEP: cons − aux at λc ∈ {0, 0.01, 0.1, 0.3, 1, 3, 10} (λa fixed at 0.5). A flat non-positive
    curve shows the physics-consistency null is not a single-λ artifact.

(B) AUX NEGATIVE CONTROLS at λa=0.5: the auxiliary a/v targets are
      real    : true acceleration / velocity                (genuine cross-order supervision)
      shuffled: a/v targets from a permuted trial order      (destroys MEG↔a/v correspondence)
      noise   : Gaussian targets matched in variance         (pure extra-head regularisation)
    If only `real` improves the position head over base, the auxiliary benefit is genuine cross-order
    supervision; if `shuffled`/`noise` help as much, it is generic multi-head regularisation.

Single seed (the cons−aux effect is large, dz≈−1.1, and the aux effect robust 16/18); reported at
session and subject (n=9) level. Pre-SPECIFIED (not pre-registered — no timestamped protocol exists;
wording corrected in the paper accordingly).

Run (.venv_deep):  python 29_consistency_sweep.py <test|full> [gpu]
"""
import sys, time, importlib.util, numpy as np
from scipy.stats import wilcoxon
import torch, torch.nn as nn
b08 = importlib.util.spec_from_file_location("m08", "/mnt/hdd/MEG_Motor/motor_ensemble_claude/scripts/08_consistency_decoder.py")
m08 = importlib.util.module_from_spec(b08); b08.loader.exec_module(m08)

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"; SIB = "/mnt/hdd/MEG_Motor/motor_comparison_claude"
ORDERS = m08.ORDERS; NFOLD = m08.NFOLD; LAM_A = 0.5
per_trial_r = m08.per_trial_r; ztime = m08.ztime; to_pos = m08.to_pos
LAMBDAS = [0.0, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0]
GPU = int(sys.argv[2]) if len(sys.argv) > 2 else 2
DEV = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu'); m08.DEV = DEV


def train_cfg(Xtr, Ytr, Ysd, lam_c=0.0, aux=True, seed=0):
    """aux=True adds a/v supervision (λa); lam_c>0 adds the physics-consistency term. Ytr already the
    (possibly shuffled/noise) standardized a/v/p target stack."""
    torch.manual_seed(seed)
    net = m08.MultiHead(Xtr.shape[-1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=m08.LR, weight_decay=m08.WD)
    xb = torch.from_numpy(Xtr).to(DEV); yb = torch.from_numpy(Ytr).to(DEV)
    sd = torch.tensor([Ysd[o] for o in ORDERS], device=DEV, dtype=torch.float32)
    n = len(xb); g = torch.Generator(device=DEV).manual_seed(seed); net.train()
    for _ in range(m08.EPOCHS):
        perm = torch.randperm(n, generator=g, device=DEV)
        for i in range(0, n, m08.TRIAL_BS):
            idx = perm[i:i+m08.TRIAL_BS]; out = net(xb[idx]); yt = yb[idx]
            loss = ((out[..., 2]-yt[..., 2])**2).mean()
            if aux:
                loss = loss + LAM_A*(((out[..., 0]-yt[..., 0])**2).mean() + ((out[..., 1]-yt[..., 1])**2).mean())
            if lam_c > 0:
                a_ph = out[..., 0]*sd[0]; v_ph = out[..., 1]*sd[1]; p_ph = out[..., 2]*sd[2]
                za = m08.t_z(m08.t_T2(a_ph)); zv = m08.t_z(m08.t_T1(v_ph)); zp = m08.t_z(p_ph)
                loss = loss + lam_c*(((za-zp)**2).mean() + ((zv-zp)**2).mean() + ((za-zv)**2).mean())
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net


def run_session(subj, sess, common, kin, rng):
    X, t = m08.load_meg(subj, sess, common); D, tw = m08.build_design(X, t)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]).astype(np.float32) for o in ORDERS}
    ntr, nt, m = D.shape
    folds = np.array_split(m08.RNG.permutation(ntr), NFOLD)
    keys = ['base', 'aux'] + [f'cons{lc}' for lc in LAMBDAS if lc > 0] + ['aux_shuf', 'aux_noise']
    acc = {k: [] for k in keys}
    for fi in range(NFOLD):
        te = folds[fi]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != fi])
        Dmean = D[tr].mean(0); flat = (D[tr]-Dmean).reshape(-1, m); mu = flat.mean(0); sd = flat.std(0)+1e-9
        Xtr = (D[tr]-Dmean-mu)/sd; Xte = (D[te]-Dmean-mu)/sd
        Ym = {o: Ys[o][tr].mean(0) for o in ORDERS}; Yr_tr = {o: Ys[o][tr]-Ym[o] for o in ORDERS}
        Ysd = {o: float(Yr_tr[o].std()+1e-9) for o in ORDERS}
        Ystd = np.stack([Yr_tr[o]/Ysd[o] for o in ORDERS], axis=-1).astype(np.float32)   # (ntr,nt,3)
        Ytrue = Ys['p'][te]-Ym['p']

        def evalp(net):
            pr = m08.predict(net, Xte); return per_trial_r(to_pos('p', pr[:, :, 2]*Ysd['p']), Ytrue).mean()
        # base (no aux), aux (real), cons sweep
        acc['base'].append(evalp(train_cfg(Xtr, Ystd, Ysd, lam_c=0.0, aux=False)))
        acc['aux'].append(evalp(train_cfg(Xtr, Ystd, Ysd, lam_c=0.0, aux=True)))
        for lc in LAMBDAS:
            if lc > 0:
                acc[f'cons{lc}'].append(evalp(train_cfg(Xtr, Ystd, Ysd, lam_c=lc, aux=True)))
        # aux negative controls: shuffle a/v across trials; noise a/v
        Yshuf = Ystd.copy(); pe = rng.permutation(len(tr))
        Yshuf[:, :, 0] = Ystd[pe, :, 0]; Yshuf[:, :, 1] = Ystd[pe, :, 1]
        acc['aux_shuf'].append(evalp(train_cfg(Xtr, Yshuf, Ysd, lam_c=0.0, aux=True)))
        Ynoise = Ystd.copy()
        Ynoise[:, :, 0] = rng.standard_normal(Ystd[:, :, 0].shape).astype(np.float32)
        Ynoise[:, :, 1] = rng.standard_normal(Ystd[:, :, 1].shape).astype(np.float32)
        acc['aux_noise'].append(evalp(train_cfg(Xtr, Ynoise, Ysd, lam_c=0.0, aux=True)))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _paired(recs, a, b):
    da = np.array([r[a] for r in recs]); db = np.array([r[b] for r in recs]); d = da-db
    subj = np.array([r['subj'] for r in recs]); dss = np.array([d[subj == s].mean() for s in np.unique(subj)])
    ps = wilcoxon(da, db).pvalue if np.any(d) else float('nan')
    pj = wilcoxon(dss).pvalue if np.any(dss) else float('nan')
    return da.mean(), db.mean(), d.mean(), int((d > 0).sum()), len(recs), ps, int((dss > 0).sum()), len(dss), pj


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kin = dict(np.load(f"{SIB}/features/kinematics.npz", allow_pickle=True))
    common = m08.common_channels(); print(f"[cons_sweep] dev {DEV} EPOCHS {m08.EPOCHS}", flush=True)
    if mode == "test":
        t0 = time.time(); r = run_session(1, 1, common, kin, np.random.default_rng(101))
        print(f"Sub1 ses1 ({time.time()-t0:.1f}s)")
        print(f"  base={r['base']:.4f} aux={r['aux']:.4f} aux_shuf={r['aux_shuf']:.4f} aux_noise={r['aux_noise']:.4f}")
        print("  cons−aux by λ:", {lc: round(r[f'cons{lc}']-r['aux'], 4) for lc in LAMBDAS if lc > 0})
    else:
        recs = []
        for s in range(1, 10):
            for se in (1, 2):
                t0 = time.time(); r = run_session(s, se, common, kin, np.random.default_rng(100*s+se))
                recs.append(dict(subj=s, sess=se, **r))
                print(f"[cons_sweep] Sub{s} ses{se} ({time.time()-t0:.1f}s) aux-base={r['aux']-r['base']:+.3f} "
                      f"cons0.3-aux={r['cons0.3']-r['aux']:+.3f}", flush=True)
        np.savez(f"{PROJ}/results/consistency_sweep.npz", recs=np.array(recs, dtype=object))
        L = ["# Consistency λ sweep + auxiliary negative controls — script 29 (n=18)\n",
             "Position-head per-trial r, single seed. λa = 0.5 throughout.\n",
             "## (A) Physics-consistency λc sweep: cons − aux (should stay ≤ 0 across λ)",
             "| λc | cons − aux | sess pos/n, p | subj pos/9, p |", "|---|---|---|---|"]
        for lc in LAMBDAS:
            if lc > 0:
                ma, mb, md, sp, sn, ps, jp, jn, pj = _paired(recs, f'cons{lc}', 'aux')
                L.append(f"| {lc} | **{md:+.4f}** | {sp}/{sn}, {ps:.3f} | {jp}/{jn}, {pj:.3f} |")
        L += ["", "## (B) Auxiliary supervision negative controls: X − base",
              "| aux target | mean X | X − base | sess pos/n, p | subj pos/9, p |", "|---|---|---|---|---|"]
        for k, nm in [('aux','real a/v labels'),('aux_shuf','trial-shuffled a/v'),('aux_noise','Gaussian noise')]:
            ma, mb, md, sp, sn, ps, jp, jn, pj = _paired(recs, k, 'base')
            L.append(f"| {nm} | {ma:.4f} | **{md:+.4f}** | {sp}/{sn}, {ps:.3f} | {jp}/{jn}, {pj:.3f} |")
        L += ["", "## Reading",
              "- (A) If cons − aux stays ≤ 0 (n.s.) across the whole λc sweep, 'physics consistency does not "
              "add over auxiliary supervision' is not a single-λ artifact.",
              "- (B) If `real` a/v labels improve the position head but `shuffled`/`noise` do not, the "
              "auxiliary benefit is genuine cross-order supervision, not generic multi-head regularisation. "
              "If shuffled/noise help as much, downgrade the claim to extra-head regularisation."]
        open(f"{PROJ}/results/consistency_sweep.md", "w").write("\n".join(L))
        print("\n".join(L)); print("\nSAVED results/consistency_sweep.{npz,md}")
