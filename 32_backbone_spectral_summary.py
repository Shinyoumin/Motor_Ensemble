"""
32 — Summarise the backbone-general bandwidth control (script 27) into a result table.

Script 27 wrote per-session archives (results/backbone_spectral_{raw,ssl,rand}.npz) and printed its
summary to the run logs only; this script regenerates the summary as results/backbone_spectral.md so
that Supplementary Table S1 (block C) has a citable source file alongside every other arm.

Statistics only — no re-decoding.

Run: python 32_backbone_spectral_summary.py   ->  results/backbone_spectral.md
"""
import numpy as np
from scipy.stats import wilcoxon

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"
BACKBONES = [('raw', 'ridge (raw lagged sensors)'), ('ssl', 'frozen self-supervised encoder'),
             ('rand', 'frozen random projection')]
CONTRASTS = [('total (E_cross − E_bag)', 'E_cross', 'E_bag'),
             ('bandwidth-only (E_spec − E_bag)', 'E_spec', 'E_bag'),
             ('bandwidth-clean residual (E_cross − E_spec)', 'E_cross', 'E_spec'),
             ('vs optimally-smoothed direct (E_cross − p_smooth)', 'E_cross', 'p_smooth')]


def stats(recs, a, b):
    d = np.array([r[a] - r[b] for r in recs])
    subj = sorted(set(int(r['subj']) for r in recs))
    ds = np.array([np.mean([r[a] - r[b] for r in recs if int(r['subj']) == s]) for s in subj])
    return (d.mean(), int((d > 0).sum()), len(d), wilcoxon(d).pvalue,
            int((ds > 0).sum()), len(ds), wilcoxon(ds).pvalue)


def main():
    out = ["# Backbone-general bandwidth control + optimally-smoothed decoder — scripts 27/32 (n = 18)",
           "",
           "Per-trial r in the fair position-residual space. E_spec = bandwidth-matched pseudo-fusion "
           "(zero cross-order neural content); p_smooth = the direct position decoder low-passed at a "
           "nested-CV-selected cutoff (no cross-order decoding at all). Subject level = 2 sessions "
           "collapsed per subject, Wilcoxon over 9 subjects.", ""]
    for tag, name in BACKBONES:
        recs = list(np.load(f"{PROJ}/results/backbone_spectral_{tag}.npz", allow_pickle=True)['recs'])
        out += [f"## {tag} — {name}", "",
                "| contrast | mean diff | sessions | session p | subjects | subject p |",
                "|---|---|---|---|---|---|"]
        for lab, a, b in CONTRASTS:
            m, pos, n, p, spos, sn, sp = stats(recs, a, b)
            out.append(f"| {lab} | **{m:+.4f}** | {pos}/{n} | {p:.4f} | {spos}/{sn} | {sp:.4f} |")
        mean_abs = {k: np.mean([r[k] for r in recs]) for k in ('E_cross', 'E_bag', 'E_spec', 'p_smooth', 'p_direct')}
        out += ["", "Absolute means: " + " · ".join(f"{k} {v:.3f}" for k, v in mean_abs.items()), ""]
    out += ["## Reading", "",
            "On every backbone the bandwidth-only pseudo-fusion reproduces essentially the whole "
            "fusion-over-bagging gain, the bandwidth-clean residual is null at the subject level, and an "
            "optimally-smoothed single position decoder matches or exceeds the cross-order ensemble — so the "
            "bandwidth account of the fusion margin is not specific to the ridge backbone, and cross-order "
            "decoding is not necessary to obtain the gain. The trained BiLSTM is excluded because its "
            "direct-position route is ~0, which leaves the p-anchored E_spec and p_smooth undefined.", ""]
    path = f"{PROJ}/results/backbone_spectral.md"
    open(path, "w").write("\n".join(out))
    print("\n".join(out))
    print(f"SAVED {path}")


if __name__ == "__main__":
    main()
