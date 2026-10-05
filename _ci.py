"""Participant-level bootstrap confidence intervals for figure error bars (revision, Referee 1)."""
import numpy as np

NBOOT = 10000


def subj_means(recs, key):
    """Collapse the two runs of each participant -> the independent unit (n = 9)."""
    s = np.array([r["subj"] for r in recs])
    v = np.array([r[key] for r in recs], dtype=float)
    return np.array([v[s == u].mean() for u in np.unique(s)])


def ci(values, nboot=NBOOT, seed=20261001, alpha=0.05):
    """Percentile bootstrap CI of the mean over participants; returns (mean, lo, hi)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(nboot, len(v)))].mean(1)
    lo, hi = np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def err(values, **kw):
    """Asymmetric yerr array shaped for matplotlib errorbar/bar."""
    m, lo, hi = ci(values, **kw)
    return m, np.array([[m - lo], [hi - m]])
