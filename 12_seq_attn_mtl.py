"""
12 — Is the attention-MTL gain ARCHITECTURE-INDEPENDENT? (sequential BiLSTM backbone)

Script 06 established that per-order latents + cross-latent self-attention beat a no-attention
per-order-latent MTL (+0.030, 18/18) on a *static* MLP backbone applied to lag-augmented MEG.
The obvious worry: is "selective latent sharing helps" a property of the task/representation, or
an artefact of the MLP + lag design? This arm ports the SAME four-way comparison onto a TRAINED
sequential backbone (BiLSTM over the raw residual MEG sequence, as in script 04), where temporal
structure is modelled explicitly rather than baked into lag features.

Backbone: shared BiLSTM -> per-timepoint hidden -> per-order latent tokens -> per-order heads,
predicting each order's residual SEQUENCE (multi-task MSE). Cross-latent attention runs over the
3 order tokens at each timepoint (residual, so it can learn identity == no-attention).

  A seq_stl        : 3 independent BiLSTMs + heads (no sharing)                       -> reference
  B seq_latent     : one shared BiLSTM -> 3 per-order latents -> 3 heads (no crosstalk)
  C seq_latent_att : B + cross-latent multi-head self-attention over the 3 order tokens -> proposed

Same fair position-residual per-trial r space, to_pos transforms and output fusion (`_fuse`) as 06.
Key contrast: seq_latent_att_p vs seq_latent_p (does attention still help on a sequential net).

Run (env motor_comparison_claude/.venv_deep):
  python 12_seq_attn_mtl.py test [gpu]
  python 12_seq_attn_mtl.py full [gpu]   -> results/ensemble_seq_attn.npz, results/seq_attn.md
"""
import sys, time, os, numpy as np
import importlib.util
_SPEC = importlib.util.spec_from_file_location(
    "m06", os.path.join(os.path.dirname(os.path.abspath(__file__)), "06_attn_mtl.py"))
m06 = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(m06)
import torch, torch.nn as nn

ORDERS, NFOLD, PROJ, DEV = m06.ORDERS, m06.NFOLD, m06.PROJ, m06.DEV
SF_DEC, WIN = m06.SF_DEC, m06.WIN
load_meg, common_channels = m06.load_meg, m06.common_channels
per_trial_r, ztime, to_pos = m06.per_trial_r, m06.ztime, m06.to_pos
RNG = np.random.default_rng(0)
HID = 64; DZ = 32; DROP = 0.3; LR = 5e-3; WD = 1e-3
EPOCHS = int(os.environ.get('EPOCHS', 120)); BS = int(os.environ.get('BS', 64))


# ---------------- sequential architectures ----------------
class SeqSTL(nn.Module):
    """Three fully independent BiLSTM decoders (no shared params)."""
    def __init__(self, nin):
        super().__init__()
        self.lstm = nn.ModuleList([nn.LSTM(nin, HID, batch_first=True, bidirectional=True) for _ in ORDERS])
        self.drop = nn.Dropout(DROP)
        self.head = nn.ModuleList([nn.Linear(2*HID, 1) for _ in ORDERS])
    def forward(self, x):
        outs = [self.head[i](self.drop(self.lstm[i](x)[0])) for i in range(3)]   # each (N,nt,1)
        return torch.cat(outs, dim=-1)                                           # (N,nt,3)

class SeqLatentMTL(nn.Module):
    """Shared BiLSTM -> per-order latent tokens -> heads. attn=True adds cross-latent
    self-attention over the 3 order tokens at each timepoint (residual)."""
    def __init__(self, nin, attn):
        super().__init__()
        self.attn_on = attn
        self.lstm = nn.LSTM(nin, HID, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(DROP)
        self.zproj = nn.ModuleList([nn.Linear(2*HID, DZ) for _ in ORDERS])
        if attn:
            self.att = nn.MultiheadAttention(DZ, num_heads=2, dropout=DROP, batch_first=True)
            self.ln1 = nn.LayerNorm(DZ); self.ln2 = nn.LayerNorm(DZ)
            self.ff = nn.Sequential(nn.Linear(DZ, 2*DZ), nn.ReLU(), nn.Dropout(DROP), nn.Linear(2*DZ, DZ))
        self.heads = nn.ModuleList([nn.Linear(DZ, 1) for _ in ORDERS])
    def forward(self, x):
        N, nt, _ = x.shape
        h = self.drop(self.lstm(x)[0])                                   # (N,nt,2H)
        z = torch.stack([torch.relu(p(h)) for p in self.zproj], dim=2)   # (N,nt,3,DZ)
        z = z.reshape(N*nt, 3, DZ)                                       # order tokens per timepoint
        if self.attn_on:
            a, _ = self.att(z, z, z); z = self.ln1(z + a)
            z = self.ln2(z + self.ff(z))
        out = torch.cat([self.heads[i](z[:, i, :]) for i in range(3)], dim=-1)   # (N*nt,3)
        return out.reshape(N, nt, 3)

MODELS = {'seq_stl': lambda m: SeqSTL(m),
          'seq_latent': lambda m: SeqLatentMTL(m, attn=False),
          'seq_latent_att': lambda m: SeqLatentMTL(m, attn=True)}


def train(model, Xtr, Ytr, seed=0):
    torch.manual_seed(seed)
    net = model.to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    xb = torch.from_numpy(Xtr).to(DEV); yb = torch.from_numpy(Ytr).to(DEV)      # (ntr,nt,nin),(ntr,nt,3)
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
        for i in range(0, len(X), 256):
            out.append(net(torch.from_numpy(X[i:i+256]).to(DEV)).cpu().numpy())
    return np.concatenate(out, 0)      # (ntr,nt,3)


def run_session(subj, sess, common, kin):
    X, t = load_meg(subj, sess, common)
    tt = np.where((t >= WIN[0]) & (t <= WIN[1]))[0]; tw = t[tt]
    Xw = X[:, :, tt].transpose(0, 2, 1).astype(np.float32)            # (ntr,nt,192)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]).astype(np.float32)
          for o in ORDERS}
    ntr_all, nt, nch = Xw.shape
    folds = np.array_split(RNG.permutation(ntr_all), NFOLD)
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
        Ytr_stack = np.stack([(Yr_tr[o]/Ysd[o]) for o in ORDERS], axis=2).astype(np.float32)  # (ntr,nt,3)
        Ytrue = Ys['p'][te] - Ym['p']

        for name, ctor in MODELS.items():
            net = train(ctor(nch), Xtr, Ytr_stack, seed=0)
            pr = predict(net, Xte)                                    # (te,nt,3) standardized order-resid
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

CONTRASTS = [('seq_latent_att_p', 'seq_latent_p'), ('seq_latent_att_p', 'seq_stl_p'),
             ('seq_latent_att_fuse', 'seq_latent_att_p')]


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kin = dict(np.load(f"{m06.SIB}/features/kinematics.npz", allow_pickle=True))
    common = common_channels(); print(f"[seq_attn] common grad {len(common)} dev {DEV} EPOCHS {EPOCHS}", flush=True)
    if mode == "test":
        t0 = time.time(); r = run_session(1, 1, common, kin)
        print(f"[seq_attn] Sub1 ses1 ({time.time()-t0:.1f}s)")
        for k in [f'{n}_p' for n in MODELS] + [f'{n}_fuse' for n in MODELS]:
            print(f"   {k:20s} {r[k]:+.4f}")
    else:
        recs = []
        for s in range(1, 10):
            for se in (1, 2):
                t0 = time.time(); r = run_session(s, se, common, kin)
                recs.append(dict(subj=s, sess=se, **r))
                print(f"[seq_attn] Sub{s} ses{se} ({time.time()-t0:.1f}s) "
                      f"stl={r['seq_stl_p']:+.3f} lat={r['seq_latent_p']:+.3f} "
                      f"latAtt={r['seq_latent_att_p']:+.3f} "
                      f"d={r['seq_latent_att_p']-r['seq_latent_p']:+.3f}", flush=True)
        np.savez(f"{PROJ}/results/ensemble_seq_attn.npz", recs=np.array(recs, dtype=object))
        lines = ["# Sequential (BiLSTM) attention-MTL (script 12, n=18)\n",
                 "Per-trial r in position residual space; POSITION head (`_p`) unless noted `_fuse`.\n",
                 "| model | mean p-head r | mean fused r |", "|---|---|---|"]
        for n in MODELS:
            mp = np.mean([r[f'{n}_p'] for r in recs]); mf = np.mean([r[f'{n}_fuse'] for r in recs])
            lines.append(f"| {n} | {mp:+.4f} | {mf:+.4f} |")
        lines += ["\n## Paired contrasts (mean_a, mean_b, diff, pos/n, p)",
                  "| contrast | A | B | diff | pos/n | p |", "|---|---|---|---|---|---|"]
        print("\nSAVED results/ensemble_seq_attn.npz\n=== group contrasts (seq_attn) ===")
        for a, b in CONTRASTS:
            ma, mb, md, pos, n, p = _summ(recs, a, b)
            print(f"  {a:20s} vs {b:20s}: {ma:+.4f} {mb:+.4f} d={md:+.4f} {pos}/{n} p={p:.4f}")
            lines.append(f"| {a} vs {b} | {ma:+.4f} | {mb:+.4f} | {md:+.4f} | {pos}/{n} | {p:.4f} |")
        with open(f"{PROJ}/results/seq_attn.md", "w") as fh: fh.write("\n".join(lines)+"\n")
        print("SAVED results/seq_attn.md")
