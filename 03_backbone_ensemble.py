"""
03 — Is the cross-order fusion gain backbone-independent? (frozen-embedding arm)

Re-runs the script-01 distance-reconstruction ensemble on top of a FROZEN, label-blind feature
map phi(Z), with only a linear ridge probe touching the kinematics — mirroring the sibling study's
Track B. Two backbones:
  ssl  : the self-supervised masked-reconstruction MEG encoder (../motor_comparison_claude/
         results/ssl_encoder.pt, from sibling script 44) — a domain-matched, foundation-model-style
         embedding pretrained label-blind on this MEG.
  rand : the SAME architecture with fixed random weights — a task-blind deep-random-feature control.

Everything else (three routes to distance via T1/T2, z-scored fusion, same-order bagging control,
rank-1/2 MTL) is identical to script 01, just with phi(Z) (512-dim) in place of the 1152-dim design.
If E_cross > E_bag survives here, the fusion gain is a property of the MEG representation, not of a
ridge fit to the raw lagged design.

Run (env .venv_deep, has torch+mne+scipy):
  python 03_backbone_ensemble.py <ssl|rand> test
  python 03_backbone_ensemble.py <ssl|rand> full   -> results/ensemble_<kind>.npz
"""
import sys, time, glob, mne, numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.integrate import cumulative_trapezoid
from scipy.signal import detrend
import torch, torch.nn as nn
mne.set_log_level('ERROR')

ROOT = "/mnt/hdd/MEG_Motor"; SIB = f"{ROOT}/motor_comparison_claude"; PROJ = f"{ROOT}/motor_ensemble_claude"
SF_DEC = 100.0; DX = 1.0/SF_DEC; LP = (0.5, 12.0); WIN = (3.5, 6.3)
LAGS_MS = np.arange(-100, 175, 50); ALPHA = 100.0; NFOLD = 5
N_BAG = 10; BAG_FRAC = 0.8; ORDERS = ['a', 'v', 'p']; RNG = np.random.default_rng(0)
M_IN, HID, EMB = 1152, 768, 512
DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_norm = lambda s: s.replace(' ', '')


class Encoder(nn.Module):
    def __init__(self, m=M_IN, hid=HID, emb=EMB):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(m, hid), nn.GELU(), nn.Linear(hid, emb), nn.GELU())
    def forward(self, x): return self.net(x)


def build_encoder(kind):
    if kind == 'ssl':
        ck = torch.load(f"{SIB}/results/ssl_encoder.pt", map_location='cpu', weights_only=False)
        a = ck['arch']; enc = Encoder(a['m'], a['hid'], a['emb']); enc.load_state_dict(ck['state_dict'])
    elif kind == 'rand':
        torch.manual_seed(0); enc = Encoder()
    else:
        raise SystemExit("kind must be ssl|rand")
    return enc.to(DEV).eval()

ENC = None
def embed(Z):
    """Z:(n,1152) residual-standardized -> phi:(n,512) numpy float64."""
    out = []
    with torch.no_grad():
        for i in range(0, len(Z), 16384):
            xb = torch.from_numpy(Z[i:i+16384].astype(np.float32)).to(DEV)
            out.append(ENC(xb).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float64)


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
    return D.astype(np.float64), t[tt]

def per_trial_r(pred, true):
    pc = pred - pred.mean(-1, keepdims=True); tc = true - true.mean(-1, keepdims=True)
    return (pc*tc).sum(-1) / (np.sqrt((pc**2).sum(-1)*(tc**2).sum(-1)) + 1e-30)
def ztime(x): return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-12)
def T1(x): return detrend(cumulative_trapezoid(x, dx=DX, axis=-1, initial=0.0), axis=-1, type='linear')
def T2(x): return T1(T1(x))
def to_pos(o, pred): return T2(pred) if o == 'a' else (T1(pred) if o == 'v' else pred)
def ridge_coef(Z, Y):
    A = Z.T @ Z; A[np.diag_indices_from(A)] += ALPHA
    return cho_solve(cho_factor(A, lower=True), Z.T @ Y)


def run_session(subj, sess, common, kin):
    X, t = load_meg(subj, sess, common); D, tw = build_design(X, t)
    sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
    Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]) for o in ORDERS}
    ntr_all, nt, m = D.shape
    frng = np.random.default_rng(1000 * subj + sess)
    brng = np.random.default_rng(7_000 + 100 * subj + sess)
    folds = np.array_split(frng.permutation(ntr_all), NFOLD)
    keys = ['a_route', 'v_route', 'p_direct', 'E_mean', 'E_wt', 'E_bag', 'mtl_r1', 'mtl_r2']
    acc = {k: [] for k in keys}

    for f in range(NFOLD):
        te = folds[f]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != f])
        Dmean = D[tr].mean(0)
        flat_r = (D[tr] - Dmean).reshape(-1, m); mur = flat_r.mean(0); sdr = flat_r.std(0) + 1e-9
        Ztr_raw = (flat_r - mur) / sdr                              # (ntr*nt,1152)
        Zte_raw = ((D[te] - Dmean - mur) / sdr).reshape(-1, m)      # (nte*nt,1152)
        # frozen embedding
        Ptr = embed(Ztr_raw); Pte = embed(Zte_raw)
        muP = Ptr.mean(0); sdP = Ptr.std(0) + 1e-9
        Ptr = (Ptr - muP) / sdP; Pte = (Pte - muP) / sdP
        e = Ptr.shape[1]
        Ptr3 = Ptr.reshape(len(tr), nt, e); Pte3 = Pte.reshape(len(te), nt, e)

        Ym = {o: Ys[o][tr].mean(0) for o in ORDERS}
        Yr_tr = {o: (Ys[o][tr] - Ym[o]) for o in ORDERS}
        Yr_te = {o: (Ys[o][te] - Ym[o]) for o in ORDERS}
        Ytrue = Yr_te['p']

        Ystack = np.stack([Yr_tr[o].reshape(-1) for o in ORDERS], axis=1)
        B = ridge_coef(Ptr, Ystack)                                # (e,3)
        coef = {o: B[:, i] for i, o in enumerate(ORDERS)}
        route_te = {o: to_pos(o, Pte3 @ coef[o]) for o in ORDERS}
        route_tr = {o: to_pos(o, Ptr3 @ coef[o]) for o in ORDERS}
        acc['a_route'].append(per_trial_r(route_te['a'], Ytrue).mean())
        acc['v_route'].append(per_trial_r(route_te['v'], Ytrue).mean())
        acc['p_direct'].append(per_trial_r(route_te['p'], Ytrue).mean())

        zte = {o: ztime(route_te[o]) for o in ORDERS}
        E_mean = (zte['a'] + zte['v'] + zte['p']) / 3.0
        acc['E_mean'].append(per_trial_r(E_mean, Ytrue).mean())
        w = np.array([max(0.0, per_trial_r(route_tr[o], Yr_tr['p']).mean()) for o in ORDERS]); w /= (w.sum()+1e-12)
        acc['E_wt'].append(per_trial_r(w[0]*zte['a']+w[1]*zte['v']+w[2]*zte['p'], Ytrue).mean())

        bag = []
        for b in range(N_BAG):
            idx = brng.choice(len(tr), size=int(BAG_FRAC*len(tr)), replace=False)
            cb = ridge_coef(Ptr3[idx].reshape(-1, e), Yr_tr['p'][idx].reshape(-1))
            bag.append(ztime(Pte3 @ cb))
        acc['E_bag'].append(per_trial_r(np.mean(bag, 0), Ytrue).mean())

        Yhat = Ptr @ B; sc = Yhat.std(0) + 1e-9
        Vt = np.linalg.svd(Yhat/sc, full_matrices=False)[2]
        for k, key in [(1, 'mtl_r1'), (2, 'mtl_r2')]:
            Bk = B @ (Vt[:k].T @ Vt[:k])
            acc[key].append(per_trial_r((Pte @ Bk[:, 2]).reshape(len(te), nt), Ytrue).mean())
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _summ(recs, a, b):
    from scipy.stats import wilcoxon
    da = np.array([r[a] for r in recs]); db = np.array([r[b] for r in recs]); d = da-db
    try: p = wilcoxon(da, db).pvalue
    except Exception: p = float('nan')
    return da.mean(), db.mean(), d.mean(), int((d > 0).sum()), len(recs), p


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "ssl"
    mode = sys.argv[2] if len(sys.argv) > 2 else "test"
    ENC = build_encoder(kind)
    kin = dict(np.load(f"{SIB}/features/kinematics.npz", allow_pickle=True))
    common = common_channels(); print(f"[{kind}] common grad {len(common)} dev {DEV}", flush=True)
    if mode == "test":
        t0 = time.time(); r = run_session(1, 1, common, kin)
        print(f"[{kind}] Sub1 ses1 ({time.time()-t0:.1f}s)")
        for k in ['a_route', 'v_route', 'p_direct', 'E_mean', 'E_wt', 'E_bag', 'mtl_r1', 'mtl_r2']:
            print(f"   {k:10s} {r[k]:+.4f}")
    else:
        recs = []
        for s in range(1, 10):
            for se in (1, 2):
                t0 = time.time(); r = run_session(s, se, common, kin)
                recs.append(dict(subj=s, sess=se, **r))
                print(f"[{kind}] Sub{s} ses{se} ({time.time()-t0:.1f}s) "
                      f"a={r['a_route']:+.3f} p={r['p_direct']:+.3f} Ecross={r['E_wt']:+.3f} Ebag={r['E_bag']:+.3f}", flush=True)
        np.savez(f"{PROJ}/results/ensemble_{kind}.npz", recs=np.array(recs, dtype=object))
        print(f"SAVED results/ensemble_{kind}.npz\n=== group contrasts ({kind}) ===")
        for a, b in [('E_wt', 'p_direct'), ('E_wt', 'a_route'), ('E_wt', 'E_bag'),
                     ('E_mean', 'E_bag'), ('mtl_r1', 'p_direct'), ('a_route', 'p_direct')]:
            ma, mb, md, pos, n, p = _summ(recs, a, b)
            print(f"  {a:9s} vs {b:9s}: {ma:+.4f} {mb:+.4f} d={md:+.4f} {pos}/{n} p={p:.4f}")
