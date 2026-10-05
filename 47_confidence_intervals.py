"""
47 — Participant-level confidence intervals for every headline contrast (revision, Referee 1).

The referee notes that p = 3.9e-3 is the exact floor of the signed-rank test at n = 9 and conveys
only that all nine participants were positive, and asks for confidence intervals alongside it. This
script computes, for each headline contrast, the participant-level mean, the percentile and BCa
bootstrap 95% intervals over 20,000 resamples of the nine participants, the exact sign-flip p-value
over all 2^9 sign assignments, and the signed-rank p.

Run:  python 47_confidence_intervals.py   ->  results/confidence_intervals.{npz,md}
"""
import numpy as np, itertools
from scipy.stats import wilcoxon

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"
R = f"{PROJ}/results"
NBOOT = 20000
RNG = np.random.default_rng(20261001)


def recs(stem, key="recs"):
    return list(np.load(f"{R}/{stem}.npz", allow_pickle=True)[key])


def sj(rs, k):
    s = np.array([r["subj"] for r in rs]); v = np.array([r[k] for r in rs])
    return np.array([v[s == u].mean() for u in np.unique(s)])


def boot_ci(d, alpha=0.05):
    n = len(d); idx = RNG.integers(0, n, size=(NBOOT, n))
    bs = d[idx].mean(1)
    lo, hi = np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # BCa
    z0 = np.quantile(np.sign(bs - d.mean()), 0.5)  # placeholder if degenerate
    p0 = (bs < d.mean()).mean()
    from scipy.stats import norm
    z0 = norm.ppf(np.clip(p0, 1e-6, 1 - 1e-6))
    jack = np.array([np.delete(d, i).mean() for i in range(n)])
    jm = jack.mean()
    a = ((jm - jack) ** 3).sum() / (6 * (((jm - jack) ** 2).sum() ** 1.5) + 1e-30)
    def adj(q):
        zq = norm.ppf(q)
        return norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq)))
    blo, bhi = np.percentile(bs, [100 * adj(alpha / 2), 100 * adj(1 - alpha / 2)])
    return (lo, hi), (blo, bhi), bs.std(ddof=1)


def signflip_p(d):
    n = len(d); obs = abs(d.mean()); cnt = 0
    for signs in itertools.product([-1, 1], repeat=n):
        if abs((np.array(signs) * d).mean()) >= obs - 1e-15:
            cnt += 1
    return cnt / 2 ** n


CONTRASTS = []
core = recs("ensemble_distance"); sp = recs("spectral_control")
CONTRASTS += [("E_cross − E_bag (fusion vs bagging, ridge)", sj(core, "E_wt") - sj(core, "E_bag")),
              ("E_spec − E_bag (bandwidth-only)", sj(sp, "E_spec") - sj(sp, "E_bag")),
              ("E_cross − E_spec (bandwidth-clean residual)", sj(sp, "E_cross") - sj(sp, "E_spec")),
              ("p_direct − rank-1 shared bottleneck", sj(core, "p_direct") - sj(core, "mtl_r1"))]
for nm, st in [("frozen SSL", "ensemble_ssl"), ("random projection", "ensemble_rand"),
               ("trained LSTM", "ensemble_lstm")]:
    try:
        rs = recs(st); CONTRASTS.append((f"E_cross − E_bag ({nm})", sj(rs, "E_wt") - sj(rs, "E_bag")))
    except Exception:
        pass
dv = recs("diversity_matched")
CONTRASTS.append(("E_cross − decorrelation-matched same-order", sj(dv, "E_cross") - sj(dv, "same_acc@0.08")))
cp = list(np.load(f"{R}/causal_pipeline.npz", allow_pickle=True)["recs_tau0"])
va = list(np.load(f"{R}/causal_velocity_anchor.npz", allow_pickle=True)["recs_tau0"])
CONTRASTS += [("E_cross − E_bag (fully causal, τ = 0)", sj(cp, "E_cross") - sj(cp, "E_bag")),
              ("E_cross − E_specV (causal bandwidth-clean)", sj(va, "E_cross") - sj(va, "E_specV"))]
at = recs("attention_control")
CONTRASTS += [("latent_ffn − latent (added capacity)", sj(at, "latent_ffn") - sj(at, "latent")),
              ("latent_att − latent_ffn (token mixing)", sj(at, "latent_att") - sj(at, "latent_ffn"))]
dep = np.load(f"{R}/deployment.npz", allow_pickle=True); cs = list(dep["cs"])
CONTRASTS.append(("between-run E_cross − E_bag",
                  np.array([c["E_cross"] - c["E_bag"] for c in cs])))

L = ["# Participant-level confidence intervals — script 47 (n = 9)", "",
     f"Percentile and BCa 95% bootstrap intervals over {NBOOT:,} resamples of the nine participants;",
     "exact sign-flip *p* enumerates all 2⁹ = 512 sign assignments; *p*_SR is the two-sided Wilcoxon",
     "signed-rank test, whose attainable floor at *n* = 9 is 3.9×10⁻³.", "",
     "| contrast | mean | median | 95% CI (percentile) | 95% CI (BCa) | sign | *p*_SR | *p*_signflip |",
     "|---|---|---|---|---|---|---|---|"]
out = {}
for name, d in CONTRASTS:
    d = np.asarray(d, float)
    (lo, hi), (blo, bhi), se = boot_ci(d)
    L.append(f"| {name} | {d.mean():+.4f} | {np.median(d):+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
             f"[{blo:+.4f}, {bhi:+.4f}] | {int((d>0).sum())}/9 | {wilcoxon(d).pvalue:.4f} | "
             f"{signflip_p(d):.4f} |")
    out[name] = d
np.savez(f"{R}/confidence_intervals.npz", **{k: v for k, v in out.items()})
open(f"{R}/confidence_intervals.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
