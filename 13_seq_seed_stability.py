"""
13 — MULTI-SEED STABILITY of the sequential (BiLSTM) attention-MTL arm (script 12).

Script 12 found that on a BiLSTM backbone the attention-over-no-attention gain
(seq_latent_att_p vs seq_latent_p) was only +0.013, 11/18, p=0.21 at a single seed — i.e. it did
NOT reach significance, in contrast to the static-MLP arm where it was robust (+0.026, 98/108
across seeds). Before concluding "attention's marginal gain is backbone-dependent" we must check
that the LSTM 11/18 is not just an unlucky seed. This mirrors script 10 exactly, but for the
sequential models of script 12, so the MLP and BiLSTM arms are compared on equal (multi-seed)
footing.

Only the training seed (weight init + minibatch order) is varied; per-session folds are
seed-independent (default_rng(1000*subj+sess)).

Run (env motor_comparison_claude/.venv_deep):
  python 13_seq_seed_stability.py test <gpu>
  python 13_seq_seed_stability.py full <gpu> <seed_start> <seed_end>
     -> results/seq_seed_stability/seed_{s}.npz  (one per seed)
"""
import sys, time, os, numpy as np
import torch
import importlib.util
_SPEC = importlib.util.spec_from_file_location(
    "m12", os.path.join(os.path.dirname(os.path.abspath(__file__)), "12_seq_attn_mtl.py"))
m12 = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(m12)

ORDERS, NFOLD, MODELS = m12.ORDERS, m12.NFOLD, m12.MODELS
PROJ, WIN = m12.PROJ, m12.WIN
OUTDIR = f"{PROJ}/results/seq_seed_stability"
os.makedirs(OUTDIR, exist_ok=True)


def run_session_seed(subj, sess, common, kin, seed):
    """Mirror of m12.run_session but (a) seed-independent per-session folds, (b) train(seed=seed)."""
    X, t = m12.load_meg(subj, sess, common)
    tt = np.where((t >= WIN[0]) & (t <= WIN[1]))[0]; tw = t[tt]
    Xw = X[:, :, tt].transpose(0, 2, 1).astype(np.float32)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]).astype(np.float32)
          for o in ORDERS}
    ntr_all, nt, nch = Xw.shape
    fold_rng = np.random.default_rng(1000*subj + sess)          # seed-independent, per-session
    folds = np.array_split(fold_rng.permutation(ntr_all), NFOLD)
    keys = [f'{k}_p' for k in MODELS] + [f'{k}_fuse' for k in MODELS]
    acc = {k: [] for k in keys}

    for f in range(NFOLD):
        te = folds[f]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != f])
        Xmean = Xw[tr].mean(0)
        mu = (Xw[tr]-Xmean).reshape(-1, nch).mean(0); sd = (Xw[tr]-Xmean).reshape(-1, nch).std(0)+1e-9
        Xtr = ((Xw[tr]-Xmean-mu)/sd).astype(np.float32); Xte = ((Xw[te]-Xmean-mu)/sd).astype(np.float32)
        Ym = {o: Ys[o][tr].mean(0) for o in ORDERS}
        Yr_tr = {o: Ys[o][tr]-Ym[o] for o in ORDERS}
        Ysd = {o: Yr_tr[o].std()+1e-9 for o in ORDERS}
        Ytr_stack = np.stack([(Yr_tr[o]/Ysd[o]) for o in ORDERS], axis=2).astype(np.float32)
        Ytrue = Ys['p'][te] - Ym['p']

        for mi, (name, ctor) in enumerate(MODELS.items()):
            # Seed WEIGHT INIT explicitly (see script 10) so init+dropout+batch order are all
            # controlled by `seed`; distinct per (seed,fold,model), fully reproducible.
            torch.manual_seed(1000*seed + 10*f + mi)
            net = m12.train(ctor(nch), Xtr, Ytr_stack, seed=seed)
            pr = m12.predict(net, Xte)
            route = {o: m12.to_pos(o, pr[:, :, i]*Ysd[o]) for i, o in enumerate(ORDERS)}
            acc[f'{name}_p'].append(m12.per_trial_r(route['p'], Ytrue).mean())
            zt = {o: m12.ztime(route[o]) for o in ORDERS}
            acc[f'{name}_fuse'].append(m12.per_trial_r((zt['a']+zt['v']+zt['p'])/3.0, Ytrue).mean())
    return {k: float(np.mean(v)) for k, v in acc.items()}


def run_seed(seed, common, kin):
    recs = []
    for s in range(1, 10):
        for se in (1, 2):
            t0 = time.time(); r = run_session_seed(s, se, common, kin, seed)
            recs.append(dict(subj=s, sess=se, seed=seed, **r))
            print(f"[seq seed {seed}] Sub{s} ses{se} ({time.time()-t0:.1f}s) "
                  f"lat={r['seq_latent_p']:+.3f} latAtt={r['seq_latent_att_p']:+.3f} "
                  f"d={r['seq_latent_att_p']-r['seq_latent_p']:+.3f}", flush=True)
    d = np.array([r['seq_latent_att_p']-r['seq_latent_p'] for r in recs])
    print(f"[seq seed {seed}] DONE seq_latent_att_p - seq_latent_p: mean={d.mean():+.4f} "
          f"pos={int((d>0).sum())}/{len(d)}", flush=True)
    np.savez(f"{OUTDIR}/seed_{seed}.npz", recs=np.array(recs, dtype=object))
    print(f"[seq seed {seed}] SAVED {OUTDIR}/seed_{seed}.npz", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kin = dict(np.load(f"{m12.m06.SIB}/features/kinematics.npz", allow_pickle=True))
    common = m12.common_channels()
    print(f"[seq_seed] common grad {len(common)} dev {m12.DEV} EPOCHS {m12.EPOCHS}", flush=True)
    if mode == "test":
        t0 = time.time(); r = run_session_seed(1, 1, common, kin, seed=1)
        print(f"[seq_seed] Sub1 ses1 seed1 ({time.time()-t0:.1f}s) "
              f"lat={r['seq_latent_p']:+.4f} latAtt={r['seq_latent_att_p']:+.4f} "
              f"d={r['seq_latent_att_p']-r['seq_latent_p']:+.4f}")
    else:
        s0 = int(sys.argv[3]); s1 = int(sys.argv[4])
        for sd in range(s0, s1+1):
            run_seed(sd, common, kin)
