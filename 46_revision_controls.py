"""
46 — Three controls requested at revision (Referee 1), computed in one pass.

(A) OCULAR / ANTERIOR-SENSOR CONTROL. The referee notes that no artefact handling is described and
    asks whether the fusion and bandwidth results survive ocular correction. Only subject 1 has an
    EOG channel in this derivative of the dataset, so a uniform EOG regression is impossible; instead
    the whole analysis is repeated with the anterior third of the gradiometer array removed — the
    sensors over which ocular fields are largest. If the decomposition is ocular in origin it should
    not survive.

(B) BANDWIDTH-INSENSITIVE METRIC. Per-trial correlation rewards smoothing, which is precisely the
    mechanism under test. We therefore add a residual-space R² in which the prediction is rescaled by
    a single least-squares gain estimated on the TRAINING trials (not per test trial), so that a
    prediction with the wrong amplitude is penalised rather than normalised away:
        R² = 1 − Σ(y − g·ŷ)² / Σ y² ,  g = argmin over training trials.
    E_bag, E_spec and E_cross are compared on this metric as well as on r.

(C) NON-POSITIVE LAGS ONLY. The published design uses MEG from −100 to +150 ms relative to each
    kinematic sample. The referee asks for the offline margin when only non-positive lags
    (−100, −50, 0 ms) are admitted, i.e. no MEG sampled after the kinematic sample.

Folds, ridge, routes and weighting are identical to the main analysis (deterministic per-session
seeds). Run:  python 46_revision_controls.py test | full
"""
import sys, time, importlib.util, numpy as np, mne
from scipy.stats import wilcoxon

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"; ROOT = "/mnt/hdd/MEG_Motor"
_s = importlib.util.spec_from_file_location("ens01", f"{PROJ}/scripts/01_ensemble_distance.py")
ens01 = importlib.util.module_from_spec(_s); _s.loader.exec_module(ens01)
_s2 = importlib.util.spec_from_file_location("sc17", f"{PROJ}/scripts/17_spectral_control.py")
sc17 = importlib.util.module_from_spec(_s2); _s2.loader.exec_module(sc17)
mne.set_log_level("ERROR")

to_pos, per_trial_r, ztime = ens01.to_pos, ens01.per_trial_r, ens01.ztime
ORDERS, NFOLD, SF_DEC, WIN, LP = ens01.ORDERS, ens01.NFOLD, ens01.SF_DEC, ens01.WIN, ens01.LP
ridge_coef, common_channels = ens01.ridge_coef, ens01.common_channels
mag_filter, apply_mag = sc17.mag_filter, sc17.apply_mag

LAG_SETS = {"published (-100..+150 ms)": np.arange(-100, 175, 50),
            "non-positive only (-100..0 ms)": np.arange(-100, 50, 50)}


def anterior_mask(common, frac=1/3):
    """True for the posterior 2/3 of the array; anterior `frac` is dropped."""
    info = mne.io.read_info(f"{ROOT}/00_preprocessed_data/Sub_1_ses_1_grad_epoched.fif",
                            verbose="ERROR")
    nm2idx = {ens01._norm(c): i for i, c in enumerate(info["ch_names"])}
    pos = np.array([info["chs"][nm2idx[c]]["loc"][:3] for c in common])
    y = pos[:, 1]                                     # +y = anterior
    return y < np.quantile(y, 1 - frac)


def design(X, t, lags_ms):
    lags = np.round(lags_ms / 1000.0 * SF_DEC).astype(int)
    tt = np.where((t >= WIN[0]) & (t <= WIN[1]))[0]
    feats = [X[:, :, np.clip(tt + L, 0, X.shape[2] - 1)] for L in lags]
    D = np.stack(feats, axis=2).transpose(0, 3, 1, 2).reshape(X.shape[0], len(tt), -1)
    return D.astype(np.float64), t[tt]


def r2_fixed_gain(pred_te, true_te, pred_tr, true_tr):
    """Residual-space R² with one train-fitted least-squares gain (no per-trial renormalisation)."""
    g = float((pred_tr * true_tr).sum() / ((pred_tr ** 2).sum() + 1e-30))
    ss_res = ((true_te - g * pred_te) ** 2).sum(-1)
    ss_tot = (true_te ** 2).sum(-1) + 1e-30
    return (1.0 - ss_res / ss_tot).mean()


def run_session(subj, sess, common, kin):
    X_full, t = ens01.load_meg(subj, sess, common)
    keep = anterior_mask(common)
    out = {}
    for arr_name, Xa in (("all sensors", X_full), ("anterior third removed", X_full[:, keep, :])):
        for lag_name, lags in LAG_SETS.items():
            if arr_name != "all sensors" and "non-positive" in lag_name:
                continue                                    # 2x2 is unnecessary; keep one factor at a time
            D, tw = design(Xa, t, lags)
            sel = (kin['subj'] == subj) & (kin['sess'] == sess); kt = kin['times']
            Ys = {o: np.stack([np.interp(tw, kt, kin[o][sel][i]) for i in range(sel.sum())]) for o in ORDERS}
            ntr, nt, m = D.shape
            folds = np.array_split(np.random.default_rng(1000 * subj + sess).permutation(ntr), NFOLD)
            brng = np.random.default_rng(7_000 + 100 * subj + sess)
            acc = {k: [] for k in ("E_cross", "E_spec", "E_bag",
                                   "R2_cross", "R2_spec", "R2_bag", "R2_pdir")}
            for f in range(NFOLD):
                te = folds[f]; tr = np.concatenate([folds[k] for k in range(NFOLD) if k != f])
                Dm = D[tr].mean(0)
                flat = (D[tr] - Dm).reshape(-1, m); mu = flat.mean(0); sd = flat.std(0) + 1e-9
                Ztr = (flat - mu) / sd; Ztr3 = Ztr.reshape(len(tr), nt, m)
                Zte = (D[te] - Dm - mu) / sd
                Ym = {o: Ys[o][tr].mean(0) for o in ORDERS}
                Ytr = {o: Ys[o][tr] - Ym[o] for o in ORDERS}
                Yte = Ys['p'][te] - Ym['p']
                B = ridge_coef(Ztr, np.stack([Ytr[o].reshape(-1) for o in ORDERS], 1))
                coef = {o: B[:, i] for i, o in enumerate(ORDERS)}
                rte = {o: to_pos(o, Zte @ coef[o]) for o in ORDERS}
                rtr = {o: to_pos(o, Ztr3 @ coef[o]) for o in ORDERS}

                w = np.array([max(per_trial_r(rtr[o], Ytr['p']).mean(), 0) for o in ORDERS])
                w /= (w.sum() + 1e-30)
                Ec_te = sum(w[i] * ztime(rte[o]) for i, o in enumerate(ORDERS))
                Ec_tr = sum(w[i] * ztime(rtr[o]) for i, o in enumerate(ORDERS))

                Lv = mag_filter(rtr['v'], rtr['p']); La = mag_filter(rtr['a'], rtr['p'])
                ste = {'p': rte['p'], 'v': apply_mag(rte['p'], Lv), 'a': apply_mag(rte['p'], La)}
                str_ = {'p': rtr['p'], 'v': apply_mag(rtr['p'], Lv), 'a': apply_mag(rtr['p'], La)}
                ws = np.array([max(per_trial_r(str_[o], Ytr['p']).mean(), 0) for o in ORDERS])
                ws /= (ws.sum() + 1e-30)
                Es_te = sum(ws[i] * ztime(ste[o]) for i, o in enumerate(ORDERS))
                Es_tr = sum(ws[i] * ztime(str_[o]) for i, o in enumerate(ORDERS))

                bag_te, bag_tr = [], []
                for _ in range(10):
                    idx = brng.choice(len(tr), size=int(0.8 * len(tr)), replace=False)
                    cb = ridge_coef(Ztr3[idx].reshape(-1, m), Ytr['p'][idx].reshape(-1))
                    bag_te.append(ztime(Zte @ cb)); bag_tr.append(ztime(Ztr3 @ cb))
                Eb_te = np.mean(bag_te, 0); Eb_tr = np.mean(bag_tr, 0)

                acc['E_cross'].append(per_trial_r(Ec_te, Yte).mean())
                acc['E_spec'].append(per_trial_r(Es_te, Yte).mean())
                acc['E_bag'].append(per_trial_r(Eb_te, Yte).mean())
                acc['R2_cross'].append(r2_fixed_gain(Ec_te, Yte, Ec_tr, Ytr['p']))
                acc['R2_spec'].append(r2_fixed_gain(Es_te, Yte, Es_tr, Ytr['p']))
                acc['R2_bag'].append(r2_fixed_gain(Eb_te, Yte, Eb_tr, Ytr['p']))
                acc['R2_pdir'].append(r2_fixed_gain(ztime(rte['p']), Yte, ztime(rtr['p']), Ytr['p']))
            tag = arr_name if "non-positive" not in lag_name else lag_name
            for k, v in acc.items():
                out[f"{tag} | {k}"] = float(np.mean(v))
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    common = common_channels()
    kin = dict(np.load(f"{PROJ}/../motor_comparison_claude/features/kinematics.npz", allow_pickle=True))
    if mode == "test":
        t0 = time.time(); r = run_session(1, 1, common, kin)
        print(f"sub1 ses1 ({time.time()-t0:.1f}s)")
        for k, v in r.items(): print(f"   {k:46s} {v:+.4f}")
        sys.exit()
    recs = []
    for s in range(1, 10):
        for se in (1, 2):
            t0 = time.time(); r = run_session(s, se, common, kin); r.update(subj=s, sess=se)
            recs.append(r); print(f"Sub{s} ses{se} ({time.time()-t0:.1f}s)", flush=True)
    np.savez(f"{PROJ}/results/revision_controls.npz", recs=np.array(recs, dtype=object))

    def sj(key):
        s = np.array([r['subj'] for r in recs]); v = np.array([r[key] for r in recs])
        return np.array([v[s == u].mean() for u in np.unique(s)])

    L = ["# Revision controls — script 46 (n = 18 sessions / 9 participants)", ""]
    for tag, title in [("all sensors", "A. Full array (reference)"),
                       ("anterior third removed", "A. Anterior third of the gradiometer array removed"),
                       ("non-positive only (-100..0 ms)", "C. Non-positive MEG lags only")]:
        ec, es, eb = sj(f"{tag} | E_cross"), sj(f"{tag} | E_spec"), sj(f"{tag} | E_bag")
        tot, bw, res = ec - eb, es - eb, ec - es
        L += [f"## {title}", "",
              "| contrast | mean | participants > 0 | *p* |", "|---|---|---|---|",
              f"| E_cross − E_bag (total) | {tot.mean():+.4f} | {int((tot>0).sum())}/9 | {wilcoxon(tot).pvalue:.4f} |",
              f"| E_spec − E_bag (bandwidth) | {bw.mean():+.4f} | {int((bw>0).sum())}/9 | {wilcoxon(bw).pvalue:.4f} |",
              f"| **E_cross − E_spec (residual)** | **{res.mean():+.4f}** | {int((res>0).sum())}/9 | {wilcoxon(res).pvalue:.4f} |",
              "", f"Absolutes: E_cross {ec.mean():.4f} · E_spec {es.mean():.4f} · E_bag {eb.mean():.4f}; "
              f"bandwidth explains {100*bw.mean()/tot.mean():.1f}% of the margin.", ""]
    L += ["## B. Bandwidth-insensitive metric: residual-space R² with a train-fitted gain", "",
          "| model | per-trial *r* | R² (train-fitted gain) |", "|---|---|---|"]
    for nm, rk, r2k in [("p_direct", None, "R2_pdir"), ("E_bag", "E_bag", "R2_bag"),
                        ("E_spec", "E_spec", "R2_spec"), ("E_cross", "E_cross", "R2_cross")]:
        rv = f"{sj('all sensors | ' + rk).mean():.4f}" if rk else "—"
        L.append(f"| {nm} | {rv} | {sj('all sensors | ' + r2k).mean():+.4f} |")
    d = sj("all sensors | R2_cross") - sj("all sensors | R2_spec")
    L += ["", f"E_cross − E_spec on this metric: {d.mean():+.4f} "
          f"({int((d>0).sum())}/9, *p* = {wilcoxon(d).pvalue:.3f})."]
    open(f"{PROJ}/results/revision_controls.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
