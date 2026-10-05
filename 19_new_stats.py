"""
19 — Subject-level (n=9) re-test of the NEW clinical-review contrasts, closing the 9-subject x 2-session
pseudo-replication gap (mirrors script 14's philosophy: report the honest n=9 alongside n=18).

Loads the saved npz from scripts 16/17 and, for each headline contrast, reports:
  - session-level (n=18) mean diff, sign, Wilcoxon p   [what 16/17 .md already show]
  - subject-level (n=9) collapse (mean of the 2 sessions per subject): mean diff, sign, Wilcoxon p, sign test
No re-decoding; statistics only.
"""
import numpy as np
from scipy.stats import wilcoxon, binomtest

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"


def load(fn):
    return list(np.load(f"{PROJ}/results/{fn}", allow_pickle=True)['recs'])


def subj_collapse(recs, key):
    """Return length-9 array: per-subject mean of the metric across its 2 sessions."""
    by = {}
    for r in recs:
        by.setdefault(r['subj'], []).append(r[key])
    return np.array([np.mean(by[s]) for s in sorted(by)])


def contrast(recs, a, b, label):
    da = np.array([r[a] for r in recs]); db = np.array([r[b] for r in recs])
    d18 = da - db
    p18 = wilcoxon(da, db).pvalue if np.any(d18 != 0) else float('nan')
    sa = subj_collapse(recs, a); sb = subj_collapse(recs, b); d9 = sa - sb
    p9 = wilcoxon(sa, sb).pvalue if np.any(d9 != 0) else float('nan')
    pos9 = int((d9 > 0).sum())
    psign = binomtest(pos9, 9).pvalue
    return (f"| {label} | {d18.mean():+.4f} | {int((d18>0).sum())}/18 | {p18:.4f} "
            f"| {d9.mean():+.4f} | {pos9}/9 | {p9:.4f} | {psign:.3f} |")


def main():
    L = ["# Subject-level (n=9) re-test of clinical-review contrasts — script 19\n",
         "| contrast | Δ (n=18) | sign18 | p18 | Δ (n=9) | sign9 | p9(Wilcoxon) | p(sign test) |",
         "|---|---|---|---|---|---|---|---|"]

    sp = load("spectral_control.npz")
    L.append("### Spectral control (does complementarity survive bandwidth matching?)")
    L.append(contrast(sp, 'E_cross', 'E_bag', "E_cross − E_bag (total gain)"))
    L.append(contrast(sp, 'E_spec', 'E_bag', "E_spec − E_bag (bandwidth-only)"))
    L.append(contrast(sp, 'E_cross', 'E_spec', "**E_cross − E_spec (residual neural)**"))
    L.append("### Calibrated clinical metrics")
    L.append(contrast(sp, 'cal_r2_cross', 'cal_r2_dir', "per-trial R² (cross−direct)"))
    L.append(contrast(sp, 'end_r_cross', 'end_r_dir', "endpoint decode r (cross−direct)"))
    L.append(contrast(sp, 'end_nrmse_dir', 'end_nrmse_cross', "endpoint nRMSE (direct−cross; +=fusion better)"))

    cz = load("causal_latency.npz")
    L.append("### Causal / real-time (fusion vs bagging at each latency budget)")
    for tau in [0, 50, 100, 150]:
        L.append(contrast(cz, f'E_cross@{tau}', f'E_bag@{tau}', f"E_cross − E_bag @ τ={tau}ms"))

    txt = "\n".join(L) + "\n"
    with open(f"{PROJ}/results/new_stats.md", "w") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
