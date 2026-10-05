"""
04 — Does the cross-order fusion gain survive a TRAINED nonlinear sequential backbone? (LSTM arm)

A small per-order LSTM decodes the residual MEG sequence (mean-removed, 100 Hz, 192 grad over the
window) into each order's residual target sequence; predictions are transformed to distance space
(T1/T2) exactly as in script 01, then the three routes are fused and compared to a same-order
bagging control. Scoped deliberately: small LSTM (hidden 64), strong weight decay, fixed epochs,
N_BAG=3 (LSTM training is expensive and the signal is weak — this arm tests robustness of the
DIRECTION of the fusion gain, not a squeeze for absolute performance). MTL-vs-STL is left to the
ridge/SSL arms (script 01/03), where it is cheap and unambiguous.

Run (env .venv_deep):
  python 04_lstm_ensemble.py test [gpu]
  python 04_lstm_ensemble.py full [gpu]   -> results/ensemble_lstm.npz
"""
import sys, time, glob, mne, numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import detrend
import torch, torch.nn as nn
mne.set_log_level('ERROR')

ROOT = "/mnt/hdd/MEG_Motor"; SIB = f"{ROOT}/motor_comparison_claude"; PROJ = f"{ROOT}/motor_ensemble_claude"
SF_DEC = 100.0; DX = 1.0/SF_DEC; LP = (0.5, 12.0); WIN = (3.5, 6.3)
NFOLD = 5; N_BAG = 3; BAG_FRAC = 0.8; ORDERS = ['a', 'v', 'p']; RNG = np.random.default_rng(0)
HID = 64; EPOCHS = 120; BS = 64; LR = 5e-3; WD = 1e-3
GPU = int(sys.argv[2]) if len(sys.argv) > 2 else 1
DEV = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')
_norm = lambda s: s.replace(' ', '')


class LSTMDec(nn.Module):
    def __init__(self, nin=192, hid=HID):
        super().__init__()
        self.lstm = nn.LSTM(nin, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3); self.head = nn.Linear(2*hid, 1)
    def forward(self, x): return self.head(self.drop(self.lstm(x)[0])).squeeze(-1)


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

def per_trial_r(pred, true):
    pc = pred - pred.mean(-1, keepdims=True); tc = true - true.mean(-1, keepdims=True)
    return (pc*tc).sum(-1) / (np.sqrt((pc**2).sum(-1)*(tc**2).sum(-1)) + 1e-30)
def ztime(x): return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-12)
def T1(x): return detrend(cumulative_trapezoid(x, dx=DX, axis=-1, initial=0.0), axis=-1, type='linear')
def T2(x): return T1(T1(x))
def to_pos(o, pred): return T2(pred) if o == 'a' else (T1(pred) if o == 'v' else pred)


def train_lstm(Xtr, Ytr, seed=0):
    torch.manual_seed(seed)
    net = LSTMDec(Xtr.shape[-1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    xb = torch.from_numpy(Xtr.astype(np.float32)).to(DEV)
    yb = torch.from_numpy(Ytr.astype(np.float32)).to(DEV)
    n = len(xb); g = torch.Generator(device=DEV).manual_seed(seed)
    net.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g, device=DEV)
        for i in range(0, n, BS):
            idx = perm[i:i+BS]
            opt.zero_grad(); loss = ((net(xb[idx]) - yb[idx])**2).mean(); loss.backward(); opt.step()
    net.eval()
    return net

def predict(net, X):
    with torch.no_grad():
        return net(torch.from_numpy(X.astype(np.float32)).to(DEV)).cpu().numpy()


def run_session(subj, sess, common, kin):
    X, t = load_meg(subj, sess, common)
    tt = np.where((t >= WIN[0]) & (t <= WIN[1]))[0]; tw = t[tt]
    Xw = X[:, :, tt].transpose(0, 2, 1)                         # (ntr, nt, 192)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]) for o in ORDERS}
    ntr_all, nt, nch = Xw.shape
    frng = np.random.default_rng(1000 * subj + sess)
    brng = np.random.default_rng(7_000 + 100 * subj + sess)
    folds = np.array_split(frng.permutation(ntr_all), NFOLD)
    keys = ['a_route', 'v_route', 'p_direct', 'E_mean', 'E_wt', 'E_bag']
    acc = {k: [] for k in keys}

    for f in range(NFOLD):
        te = folds[f]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != f])
        Xmean = Xw[tr].mean(0)
        mu = (Xw[tr]-Xmean).reshape(-1, nch).mean(0); sd = (Xw[tr]-Xmean).reshape(-1, nch).std(0)+1e-9
        Xtr = (Xw[tr]-Xmean-mu)/sd; Xte = (Xw[te]-Xmean-mu)/sd
        Ym = {o: Ys[o][tr].mean(0) for o in ORDERS}
        Yr_tr = {o: Ys[o][tr]-Ym[o] for o in ORDERS}; Yr_te = {o: Ys[o][te]-Ym[o] for o in ORDERS}
        Ytrue = Yr_te['p']

        route_te, route_tr = {}, {}
        for o in ORDERS:
            net = train_lstm(Xtr, Yr_tr[o], seed=0)
            route_te[o] = to_pos(o, predict(net, Xte))
            route_tr[o] = to_pos(o, predict(net, Xtr))
        acc['a_route'].append(per_trial_r(route_te['a'], Ytrue).mean())
        acc['v_route'].append(per_trial_r(route_te['v'], Ytrue).mean())
        acc['p_direct'].append(per_trial_r(route_te['p'], Ytrue).mean())

        zte = {o: ztime(route_te[o]) for o in ORDERS}
        acc['E_mean'].append(per_trial_r((zte['a']+zte['v']+zte['p'])/3, Ytrue).mean())
        w = np.array([max(0.0, per_trial_r(route_tr[o], Yr_tr['p']).mean()) for o in ORDERS]); w /= (w.sum()+1e-12)
        acc['E_wt'].append(per_trial_r(w[0]*zte['a']+w[1]*zte['v']+w[2]*zte['p'], Ytrue).mean())

        bag = []
        for b in range(N_BAG):
            idx = brng.choice(len(tr), size=int(BAG_FRAC*len(tr)), replace=False)
            nb = train_lstm(Xtr[idx], Yr_tr['p'][idx], seed=b+1)
            bag.append(ztime(predict(nb, Xte)))
        acc['E_bag'].append(per_trial_r(np.mean(bag, 0), Ytrue).mean())
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _summ(recs, a, b):
    from scipy.stats import wilcoxon
    da = np.array([r[a] for r in recs]); db = np.array([r[b] for r in recs]); d = da-db
    try: p = wilcoxon(da, db).pvalue
    except Exception: p = float('nan')
    return da.mean(), db.mean(), d.mean(), int((d > 0).sum()), len(recs), p


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    kin = dict(np.load(f"{SIB}/features/kinematics.npz", allow_pickle=True))
    common = common_channels(); print(f"[lstm] common grad {len(common)} dev {DEV}", flush=True)
    if mode == "test":
        t0 = time.time(); r = run_session(1, 1, common, kin)
        print(f"[lstm] Sub1 ses1 ({time.time()-t0:.1f}s)")
        for k in ['a_route', 'v_route', 'p_direct', 'E_mean', 'E_wt', 'E_bag']:
            print(f"   {k:10s} {r[k]:+.4f}")
    else:
        recs = []
        for s in range(1, 10):
            for se in (1, 2):
                t0 = time.time(); r = run_session(s, se, common, kin)
                recs.append(dict(subj=s, sess=se, **r))
                print(f"[lstm] Sub{s} ses{se} ({time.time()-t0:.1f}s) "
                      f"a={r['a_route']:+.3f} p={r['p_direct']:+.3f} Ecross={r['E_wt']:+.3f} Ebag={r['E_bag']:+.3f}", flush=True)
        np.savez(f"{PROJ}/results/ensemble_lstm.npz", recs=np.array(recs, dtype=object))
        print("SAVED results/ensemble_lstm.npz\n=== group contrasts (lstm) ===")
        for a, b in [('E_wt', 'p_direct'), ('E_wt', 'a_route'), ('E_wt', 'E_bag'), ('E_mean', 'E_bag'),
                     ('a_route', 'p_direct')]:
            ma, mb, md, pos, n, p = _summ(recs, a, b)
            print(f"  {a:9s} vs {b:9s}: {ma:+.4f} {mb:+.4f} d={md:+.4f} {pos}/{n} p={p:.4f}")
