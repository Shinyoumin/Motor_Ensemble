"""
08 — Physics-informed CONSISTENCY-regularised multi-order decoder.

The three kinematic orders are deterministically linked by integration (v = ∫a, p = ∫v). Scripts
01/06 exploit that only at read-out (transform each decoded order into position, then fuse). Here we
bake the relationship into TRAINING as a differentiable self-consistency penalty: one network with
three heads (â, v̂, p̂); besides supervised MSE on each order we add a label-free term that forces the
network's own internal estimates to AGREE in position space — T2(â) ≈ T1(v̂) ≈ p̂ (integration done
with differentiable cumulative-trapezoid + linear detrend inside the graph). The question: does
using the known differential structure as a regulariser improve single-trial DISTANCE reconstruction
over (i) a plain position decoder and (ii) plain auxiliary-task supervision without the consistency
term?

Three models (same trunk+heads; differ only in the loss), evaluated on the POSITION head in
position-residual per-trial r (comparable to script 01's p_direct):
  base : L = MSE(p̂,p)                                  (plain distance decoder)
  aux  : L = MSE(p̂,p) + λa·[MSE(â,a)+MSE(v̂,v)]         (auxiliary-task supervision, no consistency)
  cons : L = aux + λc·[‖zT2(â)-zp̂‖ + ‖zT1(v̂)-zp̂‖ + ‖zT2(â)-zT1(v̂)‖]   (physics-informed)

Also reports each model's output-level fusion (`_fuse`, mean of z{T2 â, T1 v̂, p̂}) to see whether
consistency makes the p-head alone match the fused estimate.

Run (env .venv_deep):
  python 08_consistency_decoder.py test [gpu]
  python 08_consistency_decoder.py full [gpu]  -> results/ensemble_consistency.npz, results/consistency.md
"""
import sys, os, time, glob, mne, numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import detrend
import torch, torch.nn as nn
mne.set_log_level('ERROR')

ROOT = "/mnt/hdd/MEG_Motor"; SIB = f"{ROOT}/motor_comparison_claude"; PROJ = f"{ROOT}/motor_ensemble_claude"
SF_DEC = 100.0; DX = 1.0/SF_DEC; LP = (0.5, 12.0); WIN = (3.5, 6.3)
LAGS_MS = np.arange(-100, 175, 50)
NFOLD = 5; ORDERS = ['a', 'v', 'p']; KINDS = ['base', 'aux', 'cons']; RNG = np.random.default_rng(0)
H = 128; DROP = 0.3; LR = 2e-3; WD = 1e-3
EPOCHS = int(os.environ.get('EPOCHS', 120)); TRIAL_BS = int(os.environ.get('TRIAL_BS', 32))
LAM_A = 0.5; LAM_C = 0.3
GPU = int(sys.argv[2]) if len(sys.argv) > 2 else 2
DEV = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')
_norm = lambda s: s.replace(' ', '')


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

# ---- differentiable integration / detrend / ztime along time (dim=1 of (B,T)) ----
def t_cumtrapz(x):
    seg = (x[:, 1:] + x[:, :-1]) * 0.5 * DX
    c = torch.cumsum(seg, dim=1)
    return torch.cat([torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype), c], dim=1)
def t_detrend(x):
    T = x.shape[1]; tt = torch.linspace(0.0, 1.0, T, device=x.device, dtype=x.dtype); tt = tt - tt.mean()
    slope = (x * tt).sum(1, keepdim=True) / (tt * tt).sum()
    return x - slope * tt - x.mean(1, keepdim=True)
def t_T1(x): return t_detrend(t_cumtrapz(x))
def t_T2(x): return t_T1(t_T1(x))
def t_z(x): return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-8)


class MultiHead(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(nin, H), nn.ReLU(), nn.Dropout(DROP), nn.Linear(H, H), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(H, 1) for _ in ORDERS])
    def forward(self, x):                                   # x:(B,T,m) -> (B,T,3)
        B, T, m = x.shape
        h = self.trunk(x.reshape(B*T, m))
        return torch.cat([hd(h) for hd in self.heads], dim=-1).reshape(B, T, 3)


def train_model(kind, Xtr, Ytr, Ysd, seed=0):
    """Xtr:(ntr,nt,m) f32; Ytr:(ntr,nt,3) standardized order residuals; Ysd: per-order phys std."""
    torch.manual_seed(seed)
    net = MultiHead(Xtr.shape[-1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    xb = torch.from_numpy(Xtr).to(DEV); yb = torch.from_numpy(Ytr).to(DEV)
    sd = torch.tensor([Ysd[o] for o in ORDERS], device=DEV, dtype=torch.float32)   # (3,)
    n = len(xb); g = torch.Generator(device=DEV).manual_seed(seed); net.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g, device=DEV)
        for i in range(0, n, TRIAL_BS):
            idx = perm[i:i+TRIAL_BS]
            out = net(xb[idx])                              # (b,T,3) standardized
            yt = yb[idx]
            loss = ((out[..., 2] - yt[..., 2])**2).mean()   # task_p
            if kind in ('aux', 'cons'):
                loss = loss + LAM_A*(((out[..., 0]-yt[..., 0])**2).mean() + ((out[..., 1]-yt[..., 1])**2).mean())
            if kind == 'cons':
                a_ph = out[..., 0]*sd[0]; v_ph = out[..., 1]*sd[1]; p_ph = out[..., 2]*sd[2]
                za = t_z(t_T2(a_ph)); zv = t_z(t_T1(v_ph)); zp = t_z(p_ph)
                loss = loss + LAM_C*(((za-zp)**2).mean() + ((zv-zp)**2).mean() + ((za-zv)**2).mean())
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


def predict(net, X):
    with torch.no_grad():
        return net(torch.from_numpy(X).to(DEV)).cpu().numpy()      # (ntr,nt,3) standardized


def run_session(subj, sess, common, kin):
    X, t = load_meg(subj, sess, common); D, tw = build_design(X, t)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]).astype(np.float32)
          for o in ORDERS}
    ntr_all, nt, m = D.shape
    folds = np.array_split(RNG.permutation(ntr_all), NFOLD)
    acc = {f'{k}_p': [] for k in KINDS}; acc.update({f'{k}_fuse': [] for k in KINDS})

    for f in range(NFOLD):
        te = folds[f]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != f])
        Dmean = D[tr].mean(0); flat = (D[tr]-Dmean).reshape(-1, m); mu = flat.mean(0); sd = flat.std(0)+1e-9
        Xtr = (D[tr]-Dmean-mu)/sd; Xte = (D[te]-Dmean-mu)/sd
        Ym = {o: Ys[o][tr].mean(0) for o in ORDERS}
        Yr_tr = {o: Ys[o][tr]-Ym[o] for o in ORDERS}
        Ysd = {o: float(Yr_tr[o].std()+1e-9) for o in ORDERS}
        Ytr_std = np.stack([Yr_tr[o]/Ysd[o] for o in ORDERS], axis=-1).astype(np.float32)   # (ntr,nt,3)
        Ytrue = Ys['p'][te]-Ym['p']

        for kind in KINDS:
            net = train_model(kind, Xtr, Ytr_std, Ysd, seed=0)
            pr = predict(net, Xte)                                    # (nte,nt,3) standardized
            route = {o: to_pos(o, pr[:, :, i]*Ysd[o]) for i, o in enumerate(ORDERS)}
            acc[f'{kind}_p'].append(per_trial_r(route['p'], Ytrue).mean())
            zt = {o: ztime(route[o]) for o in ORDERS}
            acc[f'{kind}_fuse'].append(per_trial_r((zt['a']+zt['v']+zt['p'])/3.0, Ytrue).mean())
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _summ(recs, a, b):
    from scipy.stats import wilcoxon
    da = np.array([r[a] for r in recs]); db = np.array([r[b] for r in recs]); d = da-db
    try: p = wilcoxon(da, db).pvalue
    except Exception: p = float('nan')
    return da.mean(), db.mean(), d.mean(), int((d > 0).sum()), len(recs), p

CONTRASTS = [('aux_p', 'base_p'), ('cons_p', 'aux_p'), ('cons_p', 'base_p'),
             ('cons_fuse', 'cons_p'), ('cons_p', 'cons_fuse')]


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kin = dict(np.load(f"{SIB}/features/kinematics.npz", allow_pickle=True))
    common = common_channels(); print(f"[cons] common grad {len(common)} dev {DEV}", flush=True)
    if mode == "test":
        t0 = time.time(); r = run_session(1, 1, common, kin)
        print(f"[cons] Sub1 ses1 ({time.time()-t0:.1f}s)")
        for k in ['base_p', 'aux_p', 'cons_p', 'base_fuse', 'aux_fuse', 'cons_fuse']:
            print(f"   {k:12s} {r[k]:+.4f}")
    else:
        recs = []
        for s in range(1, 10):
            for se in (1, 2):
                t0 = time.time(); r = run_session(s, se, common, kin)
                recs.append(dict(subj=s, sess=se, **r))
                print(f"[cons] Sub{s} ses{se} ({time.time()-t0:.1f}s) "
                      f"base={r['base_p']:+.3f} aux={r['aux_p']:+.3f} cons={r['cons_p']:+.3f} "
                      f"consFuse={r['cons_fuse']:+.3f}", flush=True)
        np.savez(f"{PROJ}/results/ensemble_consistency.npz", recs=np.array(recs, dtype=object))
        lines = ["# Physics-informed consistency-regularised decoder (script 08, n=18)\n",
                 "Per-trial r in position residual space; POSITION head (`_p`) and own fusion (`_fuse`).\n",
                 "| model | mean p-head r | mean fused r |", "|---|---|---|"]
        for k in KINDS:
            lines.append(f"| {k} | {np.mean([r[f'{k}_p'] for r in recs]):+.4f} | {np.mean([r[f'{k}_fuse'] for r in recs]):+.4f} |")
        lines += ["\n## Paired contrasts", "| contrast | A | B | diff | pos/n | p |", "|---|---|---|---|---|---|"]
        print("\nSAVED results/ensemble_consistency.npz\n=== group contrasts (consistency) ===")
        for a, b in CONTRASTS:
            ma, mb, md, pos, n, p = _summ(recs, a, b)
            print(f"  {a:12s} vs {b:12s}: {ma:+.4f} {mb:+.4f} d={md:+.4f} {pos}/{n} p={p:.4f}")
            lines.append(f"| {a} vs {b} | {ma:+.4f} | {mb:+.4f} | {md:+.4f} | {pos}/{n} | {p:.4f} |")
        with open(f"{PROJ}/results/consistency.md", "w") as fh: fh.write("\n".join(lines)+"\n")
        print("SAVED results/consistency.md")
