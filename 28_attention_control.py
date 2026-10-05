"""
28 — Is the latent-attention gain SELECTIVE CROSS-ORDER COMMUNICATION or just extra capacity? (pt 5)

Script 06's headline contrast latent_att − latent adds a MultiheadAttention + FFN + 2 LayerNorms on
top of `latent`, so it confounds cross-order token MIXING with added parameters/depth. This script adds
the parameter- and depth-matched control the reviewer asks for:

  latent      : per-order latents, three heads, NO cross-talk                         (06's C)
  latent_att  : + cross-latent multihead self-attention over the 3 order tokens        (06's D, proposed)
  latent_ffn  : + a PER-TOKEN MLP block in the attention slot (same param budget/depth, token mixing
                DISABLED) then the same shared FFN — capacity/depth matched, communication removed.

Decisive contrast: latent_att − latent_ffn. If ~0, the "attention gain" is capacity, and the claim must
soften to "adding a cross-latent module improved performance." If > 0, selective cross-order
communication is licensed. We also report latent_att − latent (06's original) and latent_ffn − latent
(pure capacity effect). Parameter counts are printed to confirm the match.

Multi-seed (default 3) with seeding BEFORE model construction (the init-seeding fix from scripts 10-13),
per-session seed-averaged, reported at session and subject (n=9) level — the attention gain was shown
seed-fragile (scripts 11/13), so single-seed is not enough.

Run (.venv_deep):  python 28_attention_control.py <test|full> [gpu] [nseed]
"""
import sys, time, importlib.util, numpy as np
from scipy.stats import wilcoxon
import torch, torch.nn as nn
b06 = importlib.util.spec_from_file_location("m06", "/mnt/hdd/MEG_Motor/motor_ensemble_claude/scripts/06_attn_mtl.py")
m06 = importlib.util.module_from_spec(b06); b06.loader.exec_module(m06)

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"; SIB = "/mnt/hdd/MEG_Motor/motor_comparison_claude"
ORDERS = m06.ORDERS; NFOLD = m06.NFOLD; H = m06.H; DZ = m06.DZ; DROP = m06.DROP
per_trial_r = m06.per_trial_r; ztime = m06.ztime; to_pos = m06.to_pos
GPU = int(sys.argv[2]) if len(sys.argv) > 2 else 1
DEV = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu'); m06.DEV = DEV


class LatentFFN(nn.Module):
    """Capacity/depth-matched control: per-token MLP in the attention slot (NO cross-token mixing)."""
    def __init__(self, nin):
        super().__init__()
        self.trunk = m06.Trunk(nin)
        self.zproj = nn.ModuleList([nn.Linear(H, DZ) for _ in ORDERS])
        # per-token block matching attention's param budget (in/out proj ~ 2*DZ*2DZ), applied per token
        self.mix = nn.Sequential(nn.Linear(DZ, 2*DZ), nn.ReLU(), nn.Linear(2*DZ, DZ))
        self.ln1 = nn.LayerNorm(DZ); self.ln2 = nn.LayerNorm(DZ)
        self.ff = nn.Sequential(nn.Linear(DZ, 2*DZ), nn.ReLU(), nn.Dropout(DROP), nn.Linear(2*DZ, DZ))
        self.heads = nn.ModuleList([nn.Linear(DZ, 1) for _ in ORDERS])
    def forward(self, x):
        h = self.trunk(x)
        z = torch.stack([torch.relu(p(h)) for p in self.zproj], dim=1)     # (N,3,DZ)
        z = self.ln1(z + self.mix(z))                                      # per-token, no mixing
        z = self.ln2(z + self.ff(z))
        return torch.cat([self.heads[i](z[:, i, :]) for i in range(3)], dim=-1)

MODELS = {'latent': lambda m: m06.LatentMTL(m, attn=False),
          'latent_att': lambda m: m06.LatentMTL(m, attn=True),
          'latent_ffn': lambda m: LatentFFN(m)}


def nparams(net): return sum(p.numel() for p in net.parameters())


def run_session(subj, sess, common, kin, seeds):
    X, t = m06.load_meg(subj, sess, common); D, tw = m06.build_design(X, t)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]).astype(np.float32) for o in ORDERS}
    ntr, nt, m = D.shape
    folds = np.array_split(m06.RNG.permutation(ntr), NFOLD)
    acc = {k: [] for k in MODELS}
    for fi in range(NFOLD):
        te = folds[fi]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != fi])
        Dmean = D[tr].mean(0); flat = (D[tr]-Dmean).reshape(-1, m); mu = flat.mean(0); sd = flat.std(0)+1e-9
        Xtr = ((D[tr]-Dmean-mu)/sd).reshape(-1, m); Xte = ((D[te]-Dmean-mu)/sd).reshape(-1, m)
        Ym = {o: Ys[o][tr].mean(0) for o in ORDERS}; Yr_tr = {o: Ys[o][tr]-Ym[o] for o in ORDERS}
        Ysd = {o: Yr_tr[o].std()+1e-9 for o in ORDERS}
        Ytr_stack = np.stack([(Yr_tr[o]/Ysd[o]).reshape(-1) for o in ORDERS], axis=1)
        Ytrue = Ys['p'][te]-Ym['p']
        for name, ctor in MODELS.items():
            rs = []
            for sd_ in seeds:
                net = m06.train(ctor(m), Xtr, Ytr_stack, seed=sd_)       # train() seeds torch before use
                pr = m06.predict(net, Xte).reshape(len(te), nt, 3)
                rs.append(per_trial_r(to_pos('p', pr[:, :, 2]*Ysd['p']), Ytrue).mean())
            acc[name].append(np.mean(rs))                                # seed-avg within fold
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _paired(recs, a, b):
    da = np.array([r[a] for r in recs]); db = np.array([r[b] for r in recs]); d = da-db
    subj = np.array([r['subj'] for r in recs]); dss = np.array([d[subj == s].mean() for s in np.unique(subj)])
    ps = wilcoxon(da, db).pvalue if np.any(d) else float('nan')
    pj = wilcoxon(dss).pvalue if np.any(dss) else float('nan')
    return da.mean(), db.mean(), d.mean(), int((d > 0).sum()), len(recs), ps, int((dss > 0).sum()), len(dss), pj


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    nseed = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    seeds = list(range(nseed))
    kin = dict(np.load(f"{SIB}/features/kinematics.npz", allow_pickle=True))
    common = m06.common_channels()
    pc = {k: nparams(MODELS[k](1152)) for k in MODELS}
    print(f"[attn_ctrl] dev {DEV} nseed {nseed} params: {pc}", flush=True)
    if mode == "test":
        t0 = time.time(); r = run_session(1, 1, common, kin, seeds)
        print(f"Sub1 ses1 ({time.time()-t0:.1f}s): " + " ".join(f"{k}={r[k]:+.4f}" for k in MODELS))
        print(f"  latent_att-latent     = {r['latent_att']-r['latent']:+.4f}")
        print(f"  latent_ffn-latent     = {r['latent_ffn']-r['latent']:+.4f}  (pure capacity)")
        print(f"  latent_att-latent_ffn = {r['latent_att']-r['latent_ffn']:+.4f}  (communication-specific)")
    else:
        recs = []
        for s in range(1, 10):
            for se in (1, 2):
                t0 = time.time(); r = run_session(s, se, common, kin, seeds); recs.append(dict(subj=s, sess=se, **r))
                print(f"[attn_ctrl] Sub{s} ses{se} ({time.time()-t0:.1f}s) "
                      f"att-lat={r['latent_att']-r['latent']:+.3f} att-ffn={r['latent_att']-r['latent_ffn']:+.3f}", flush=True)
        np.savez(f"{PROJ}/results/attention_control.npz", recs=np.array(recs, dtype=object), params=pc)
        L = ["# Attention: communication vs capacity — parameter-matched control (script 28, n=18)\n",
             f"Seed-averaged over {nseed} seeds (seeded before construction), position-head per-trial r. "
             f"Parameter counts (nin=1152): latent {pc['latent']}, latent_att {pc['latent_att']}, "
             f"latent_ffn {pc['latent_ffn']} (att vs ffn matched to "
             f"{100*pc['latent_ffn']/pc['latent_att']:.0f}%).\n",
             "| contrast | mean A | mean B | diff | sess pos/n, p | subj pos/9, p | reads as |",
             "|---|---|---|---|---|---|---|"]
        for a, b, nm in [('latent_att','latent','latent_att − latent (06 original)'),
                         ('latent_ffn','latent','latent_ffn − latent (pure capacity/depth)'),
                         ('latent_att','latent_ffn','latent_att − latent_ffn (communication-specific)')]:
            ma, mb, md, sp, sn, ps, jp, jn, pj = _paired(recs, a, b)
            L.append(f"| {nm} | {ma:.4f} | {mb:.4f} | **{md:+.4f}** | {sp}/{sn}, {ps:.3f} | {jp}/{jn}, {pj:.3f} | |")
        L += ["", "## Reading",
              "- If **latent_att − latent_ffn > 0** (subject-level), the gain is specifically cross-order "
              "communication, not capacity — the 'selective cross-order communication' wording is licensed.",
              "- If it is ~0 while latent_ffn − latent > 0, the '06 attention gain' is a capacity/depth "
              "effect; soften to 'adding a cross-latent module (attention or per-token MLP) improved "
              "performance', and note latent_att is not specifically the mechanism."]
        open(f"{PROJ}/results/attention_control.md", "w").write("\n".join(L))
        print("\n".join(L)); print("\nSAVED results/attention_control.{npz,md}")
