"""
10 — MULTI-SEED STABILITY of the script-06 attention-MTL gain.

Script 06 reported latent_att_p vs latent_p = +0.0304, 18/18, p<1e-4 (the strongest new
result of the model-development arm), but at a SINGLE training seed (seed=0). Neural-net
init + minibatch order can, in principle, manufacture an 18/18 by luck. This script re-runs
the *identical* 06 pipeline across several seeds and asks: does the +0.030 attention gain and
its per-session sign consistency survive re-seeding?

Design. We reuse 06's exact data pipeline, architectures (STL / shared / latent / latent_att),
loss, metric and fair position-residual space by importing the module. The ONLY thing varied
across runs is the training seed (weight init + minibatch shuffle). To make the comparison
clean, the 5-fold split is made deterministic PER SESSION and seed-independent
(default_rng(1000*subj+sess)), so every seed sees identical folds and the only source of
variation is the seed itself. (Note: this decouples the fold split from loop order, so seed 0
here is not bit-identical to 06's original run — 06 advanced one shared RNG across sessions —
but the pipeline, folds-per-session recipe and metric are otherwise the same.)

Each seed writes its own npz so partial progress survives, and seeds can be sharded across GPUs.

Run (env motor_comparison_claude/.venv_deep):
  python 10_seed_stability.py test <gpu>
  python 10_seed_stability.py full <gpu> <seed_start> <seed_end>   # seeds [start, end] inclusive
     -> results/seed_stability/seed_{s}.npz  (one per seed)
"""
import sys, time, os, numpy as np
import torch

# 06 reads GPU from sys.argv[2]; our CLI puts <gpu> there too, so importing 06 binds DEV
# to the requested GPU. Import the whole module and reuse its building blocks verbatim.
import importlib.util
_SPEC = importlib.util.spec_from_file_location(
    "m06", os.path.join(os.path.dirname(os.path.abspath(__file__)), "06_attn_mtl.py"))
m06 = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(m06)

ORDERS, NFOLD, MODELS = m06.ORDERS, m06.NFOLD, m06.MODELS
PROJ = m06.PROJ
OUTDIR = f"{PROJ}/results/seed_stability"
os.makedirs(OUTDIR, exist_ok=True)


def run_session_seed(subj, sess, common, kin, seed):
    """Mirror of m06.run_session but (a) seed-independent per-session folds, (b) train(seed=seed)."""
    X, t = m06.load_meg(subj, sess, common)
    D, tw = m06.build_design(X, t)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]).astype(np.float32)
          for o in ORDERS}
    ntr_all, nt, m = D.shape
    fold_rng = np.random.default_rng(1000*subj + sess)          # seed-independent, per-session
    folds = np.array_split(fold_rng.permutation(ntr_all), NFOLD)
    keys = [f'{k}_p' for k in MODELS] + [f'{k}_fuse' for k in MODELS]
    acc = {k: [] for k in keys}

    for f in range(NFOLD):
        te = folds[f]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != f])
        Dmean = D[tr].mean(0)
        flat = (D[tr]-Dmean).reshape(-1, m); mu = flat.mean(0); sd = flat.std(0)+1e-9
        Xtr = ((D[tr]-Dmean-mu)/sd).reshape(-1, m)
        Xte = ((D[te]-Dmean-mu)/sd).reshape(-1, m)
        Ym = {o: Ys[o][tr].mean(0) for o in ORDERS}
        Yr_tr = {o: Ys[o][tr]-Ym[o] for o in ORDERS}
        Ysd = {o: Yr_tr[o].std()+1e-9 for o in ORDERS}
        Ytr_stack = np.stack([(Yr_tr[o]/Ysd[o]).reshape(-1) for o in ORDERS], axis=1)
        Ytrue = Ys['p'][te] - Ym['p']

        for mi, (name, ctor) in enumerate(MODELS.items()):
            # Seed the WEIGHT INIT explicitly (ctor builds+inits the net here, before m06.train's
            # own manual_seed governs dropout/minibatch order) so init, dropout and batch order are
            # ALL controlled by `seed` and runs are reproducible. Distinct per (seed,fold,model).
            torch.manual_seed(1000*seed + 10*f + mi)
            net = m06.train(ctor(m), Xtr, Ytr_stack, seed=seed)
            pr = m06.predict(net, Xte).reshape(len(te), nt, 3)
            route = {o: m06.to_pos(o, pr[:, :, i]*Ysd[o]) for i, o in enumerate(ORDERS)}
            acc[f'{name}_p'].append(m06.per_trial_r(route['p'], Ytrue).mean())
            zt = {o: m06.ztime(route[o]) for o in ORDERS}
            acc[f'{name}_fuse'].append(m06.per_trial_r((zt['a']+zt['v']+zt['p'])/3.0, Ytrue).mean())
    return {k: float(np.mean(v)) for k, v in acc.items()}


def run_seed(seed, common, kin):
    recs = []
    for s in range(1, 10):
        for se in (1, 2):
            t0 = time.time(); r = run_session_seed(s, se, common, kin, seed)
            recs.append(dict(subj=s, sess=se, seed=seed, **r))
            print(f"[seed {seed}] Sub{s} ses{se} ({time.time()-t0:.1f}s) "
                  f"lat={r['latent_p']:+.3f} latAtt={r['latent_att_p']:+.3f} "
                  f"d={r['latent_att_p']-r['latent_p']:+.3f}", flush=True)
    d = np.array([r['latent_att_p']-r['latent_p'] for r in recs])
    print(f"[seed {seed}] DONE latent_att_p - latent_p: mean={d.mean():+.4f} pos={int((d>0).sum())}/{len(d)}",
          flush=True)
    np.savez(f"{OUTDIR}/seed_{seed}.npz", recs=np.array(recs, dtype=object))
    print(f"[seed {seed}] SAVED {OUTDIR}/seed_{seed}.npz", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kin = dict(np.load(f"{m06.SIB}/features/kinematics.npz", allow_pickle=True))
    common = m06.common_channels()
    print(f"[seed_stability] common grad {len(common)} dev {m06.DEV} EPOCHS {m06.EPOCHS}", flush=True)
    if mode == "test":
        t0 = time.time(); r = run_session_seed(1, 1, common, kin, seed=1)
        print(f"[seed_stability] Sub1 ses1 seed1 ({time.time()-t0:.1f}s) "
              f"lat={r['latent_p']:+.4f} latAtt={r['latent_att_p']:+.4f} "
              f"d={r['latent_att_p']-r['latent_p']:+.4f}")
    else:
        s0 = int(sys.argv[3]); s1 = int(sys.argv[4])
        for sd in range(s0, s1+1):
            run_seed(sd, common, kin)
