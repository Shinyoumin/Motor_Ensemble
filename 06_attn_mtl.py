"""
06 — Multi-task with PER-ORDER latents and (optional) CROSS-LATENT ATTENTION.

Motivation. Script 01 showed a forced single shared low-rank bottleneck (RRR rank-1/2) HURTS
distance reconstruction in 0/18 sessions. That penalises *forced* sharing, not sharing per se.
Here we test the complementary architectural claim (user's hypothesis): give each kinematic order
its OWN latent, then let the three latents share information *selectively* via a small cross-latent
self-attention block. If selective sharing helps where forced sharing hurt, then attention > no-
attention, and either can recover the complementary single-trial information that E_cross exploits
at the output level — but now internalised at the latent level.

Four architectures, identical fair per-timepoint residual space as script 01 (lagged MEG design
-> per-order residual targets), trained with a joint multi-task MSE loss (a+v+p), evaluated on the
POSITION head in position-residual per-trial r (directly comparable to p_direct / mtl_r1 of 01):

  A stl        : three fully independent trunks+latents+heads (no sharing)            -> reference
  B shared     : one trunk -> ONE shared latent -> three heads (forced shared code)  -> 01's RRR analog
  C latent     : one trunk -> three per-order latents -> three heads (NO cross-talk)
  D latent+att : C + cross-latent multi-head self-attention over the 3 order-tokens   -> proposed

Key contrasts (paired Wilcoxon over 18 sessions): D vs C (does attention help), C vs B (per-order
latent vs forced-shared), B vs A (reproduce forced-sharing-hurts), D vs A (selective sharing vs
independent). We also report each MTL model's own output-level fusion (fuse its 3 heads into
distance) as `*_fuse`, to see whether latent attention makes explicit output fusion unnecessary.

Run (env .venv_deep):
  python 06_attn_mtl.py test [gpu]
  python 06_attn_mtl.py full [gpu]   -> results/ensemble_attn_mtl.npz, results/attn_mtl.md
"""
import sys, time, glob, mne, numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import detrend
import torch, torch.nn as nn
mne.set_log_level('ERROR')

ROOT = "/mnt/hdd/MEG_Motor"; SIB = f"{ROOT}/motor_comparison_claude"; PROJ = f"{ROOT}/motor_ensemble_claude"
SF_DEC = 100.0; DX = 1.0/SF_DEC; LP = (0.5, 12.0); WIN = (3.5, 6.3)
LAGS_MS = np.arange(-100, 175, 50)
NFOLD = 5; ORDERS = ['a', 'v', 'p']; RNG = np.random.default_rng(0)
import os
H = 128; DZ = 32; DROP = 0.3; LR = 2e-3; WD = 1e-3
EPOCHS = int(os.environ.get('EPOCHS', 70)); BS = int(os.environ.get('BS', 1024))
GPU = int(sys.argv[2]) if len(sys.argv) > 2 else 1
DEV = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')
_norm = lambda s: s.replace(' ', '')


# ---------------- data / metric helpers (mirror scripts 01/04) ----------------
def common_channels():
    common = None
    for fp in glob.glob(f"{ROOT}/00_preprocessed_data/Sub_*_grad_epoched.fif"):
        ep = mne.read_epochs(fp, preload=False)
        g = set(_norm(c) for c, t in zip(ep.ch_names, ep.get_channel_types()) if t == 'grad')
        common = g if common is None else (common & g)
    return sorted(common)

def load_meg(subj, sess, common):
    ep = mne.read_epochs(f"{ROOT}/00_preprocessed_data/Sub_{subj}_ses_{sess}_grad_epoched.fif", preload=True)
    ep.pick(mne.pick_types(ep.info, meg='grad'))
    ep.filter(LP[0], LP[1], picks='all', verbose='ERROR'); ep.resample(SF_DEC, verbose='ERROR')
    nm2idx = {_norm(c): i for i, c in enumerate(ep.ch_names)}
    return ep.get_data(picks=[nm2idx[c] for c in common]), ep.times

def build_design(X, t):
    lags = np.round(LAGS_MS/1000.0*SF_DEC).astype(int)
    tt = np.where((t >= WIN[0]) & (t <= WIN[1]))[0]
    feats = [X[:, :, np.clip(tt+L, 0, X.shape[2]-1)] for L in lags]
    D = np.stack(feats, axis=2).transpose(0, 3, 1, 2).reshape(X.shape[0], len(tt), -1)
    return D.astype(np.float32), t[tt]

def per_trial_r(pred, true):
    pc = pred - pred.mean(-1, keepdims=True); tc = true - true.mean(-1, keepdims=True)
    return (pc*tc).sum(-1) / (np.sqrt((pc**2).sum(-1)*(tc**2).sum(-1)) + 1e-30)
def ztime(x): return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-12)
def T1(x): return detrend(cumulative_trapezoid(x, dx=DX, axis=-1, initial=0.0), axis=-1, type='linear')
def T2(x): return T1(T1(x))
def to_pos(o, pred): return T2(pred) if o == 'a' else (T1(pred) if o == 'v' else pred)


# ---------------- architectures ----------------
class Trunk(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nin, H), nn.ReLU(), nn.Dropout(DROP))
    def forward(self, x): return self.net(x)

class STL(nn.Module):
    """Three fully independent order-specific networks (no shared params)."""
    def __init__(self, nin):
        super().__init__()
        self.branch = nn.ModuleList([nn.Sequential(Trunk(nin), nn.Linear(H, DZ), nn.ReLU(),
                                                   nn.Linear(DZ, 1)) for _ in ORDERS])
    def forward(self, x): return torch.cat([b(x) for b in self.branch], dim=-1)   # (N,3)

class SharedBottleneck(nn.Module):
    """One trunk -> ONE shared latent -> three heads (forced shared code; RRR analog)."""
    def __init__(self, nin):
        super().__init__()
        self.trunk = Trunk(nin); self.z = nn.Linear(H, DZ)
        self.heads = nn.ModuleList([nn.Linear(DZ, 1) for _ in ORDERS])
    def forward(self, x):
        z = torch.relu(self.z(self.trunk(x)))
        return torch.cat([hd(z) for hd in self.heads], dim=-1)

class LatentMTL(nn.Module):
    """One trunk -> three per-order latents -> three heads. attn=True adds cross-latent
    self-attention over the 3 order-tokens (residual, so it can learn identity == no-attn)."""
    def __init__(self, nin, attn):
        super().__init__()
        self.attn_on = attn
        self.trunk = Trunk(nin)
        self.zproj = nn.ModuleList([nn.Linear(H, DZ) for _ in ORDERS])
        if attn:
            self.att = nn.MultiheadAttention(DZ, num_heads=2, dropout=DROP, batch_first=True)
            self.ln1 = nn.LayerNorm(DZ); self.ln2 = nn.LayerNorm(DZ)
            self.ff = nn.Sequential(nn.Linear(DZ, 2*DZ), nn.ReLU(), nn.Dropout(DROP), nn.Linear(2*DZ, DZ))
        self.heads = nn.ModuleList([nn.Linear(DZ, 1) for _ in ORDERS])
    def forward(self, x):
        h = self.trunk(x)
        z = torch.stack([torch.relu(p(h)) for p in self.zproj], dim=1)       # (N,3,DZ) order-tokens
        if self.attn_on:
            a, _ = self.att(z, z, z); z = self.ln1(z + a)
            z = self.ln2(z + self.ff(z))
        return torch.cat([self.heads[i](z[:, i, :]) for i in range(3)], dim=-1)   # (N,3)

MODELS = {'stl': lambda m: STL(m), 'shared': lambda m: SharedBottleneck(m),
          'latent': lambda m: LatentMTL(m, attn=False), 'latent_att': lambda m: LatentMTL(m, attn=True)}


def train(model, Xtr, Ytr, seed=0):
    torch.manual_seed(seed)
    net = model.to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    xb = torch.from_numpy(Xtr).to(DEV); yb = torch.from_numpy(Ytr).to(DEV)   # (Nsamp,m),(Nsamp,3)
    n = len(xb); g = torch.Generator(device=DEV).manual_seed(seed); net.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g, device=DEV)
        for i in range(0, n, BS):
            idx = perm[i:i+BS]
            opt.zero_grad(); loss = ((net(xb[idx]) - yb[idx])**2).mean(); loss.backward(); opt.step()
    net.eval()
    return net

def predict(net, X):
    with torch.no_grad():
        out = []
        for i in range(0, len(X), 8192):
            out.append(net(torch.from_numpy(X[i:i+8192]).to(DEV)).cpu().numpy())
    return np.concatenate(out, 0)     # (Nsamp,3)


def run_session(subj, sess, common, kin):
    X, t = load_meg(subj, sess, common)
    D, tw = build_design(X, t)                                    # (ntr,nt,m)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]).astype(np.float32)
          for o in ORDERS}
    ntr_all, nt, m = D.shape
    folds = np.array_split(RNG.permutation(ntr_all), NFOLD)
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
        Ysd = {o: Yr_tr[o].std()+1e-9 for o in ORDERS}          # per-order scale -> balanced MTL loss
        Ytr_stack = np.stack([(Yr_tr[o]/Ysd[o]).reshape(-1) for o in ORDERS], axis=1)   # (ntr*nt,3)
        Ytrue = Ys['p'][te] - Ym['p']                            # position residual, eval target

        for name, ctor in MODELS.items():
            net = train(ctor(m), Xtr, Ytr_stack, seed=0)
            pr = predict(net, Xte).reshape(len(te), nt, 3)       # heads in standardized order-resid
            # rescale heads back to physical residual, transform to position space
            route = {o: to_pos(o, pr[:, :, i]*Ysd[o]) for i, o in enumerate(ORDERS)}
            acc[f'{name}_p'].append(per_trial_r(route['p'], Ytrue).mean())
            zt = {o: ztime(route[o]) for o in ORDERS}
            acc[f'{name}_fuse'].append(per_trial_r((zt['a']+zt['v']+zt['p'])/3.0, Ytrue).mean())
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _summ(recs, a, b):
    from scipy.stats import wilcoxon
    da = np.array([r[a] for r in recs]); db = np.array([r[b] for r in recs]); d = da-db
    try: p = wilcoxon(da, db).pvalue
    except Exception: p = float('nan')
    return da.mean(), db.mean(), d.mean(), int((d > 0).sum()), len(recs), p

CONTRASTS = [('latent_att_p', 'latent_p'), ('latent_p', 'shared_p'), ('shared_p', 'stl_p'),
             ('latent_att_p', 'stl_p'), ('latent_att_fuse', 'latent_att_p'),
             ('latent_att_p', 'latent_att_fuse')]


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kin = dict(np.load(f"{SIB}/features/kinematics.npz", allow_pickle=True))
    common = common_channels(); print(f"[attn_mtl] common grad {len(common)} dev {DEV}", flush=True)
    if mode == "test":
        t0 = time.time(); r = run_session(1, 1, common, kin)
        print(f"[attn_mtl] Sub1 ses1 ({time.time()-t0:.1f}s)")
        for k in [f'{n}_p' for n in MODELS] + [f'{n}_fuse' for n in MODELS]:
            print(f"   {k:16s} {r[k]:+.4f}")
    else:
        recs = []
        for s in range(1, 10):
            for se in (1, 2):
                t0 = time.time(); r = run_session(s, se, common, kin)
                recs.append(dict(subj=s, sess=se, **r))
                print(f"[attn_mtl] Sub{s} ses{se} ({time.time()-t0:.1f}s) "
                      f"stl={r['stl_p']:+.3f} shared={r['shared_p']:+.3f} "
                      f"lat={r['latent_p']:+.3f} latAtt={r['latent_att_p']:+.3f}", flush=True)
        np.savez(f"{PROJ}/results/ensemble_attn_mtl.npz", recs=np.array(recs, dtype=object))
        lines = ["# Attention-gated per-order latent MTL (script 06, n=18)\n",
                 "Per-trial r in position residual space; POSITION head (`_p`) unless noted `_fuse`.\n",
                 "| model | mean p-head r | mean fused r |", "|---|---|---|"]
        for n in MODELS:
            mp = np.mean([r[f'{n}_p'] for r in recs]); mf = np.mean([r[f'{n}_fuse'] for r in recs])
            lines.append(f"| {n} | {mp:+.4f} | {mf:+.4f} |")
        lines += ["\n## Paired contrasts (mean_a, mean_b, diff, pos/n, p)", "| contrast | A | B | diff | pos/n | p |", "|---|---|---|---|---|---|"]
        print("\nSAVED results/ensemble_attn_mtl.npz\n=== group contrasts (attn_mtl) ===")
        for a, b in CONTRASTS:
            ma, mb, md, pos, n, p = _summ(recs, a, b)
            print(f"  {a:16s} vs {b:16s}: {ma:+.4f} {mb:+.4f} d={md:+.4f} {pos}/{n} p={p:.4f}")
            lines.append(f"| {a} vs {b} | {ma:+.4f} | {mb:+.4f} | {md:+.4f} | {pos}/{n} | {p:.4f} |")
        with open(f"{PROJ}/results/attn_mtl.md", "w") as fh: fh.write("\n".join(lines)+"\n")
        print("SAVED results/attn_mtl.md")
