"""
14 — SUBJECT-RANDOM-INTERCEPT MIXED-MODEL re-analysis of every headline contrast.

The whole study's primary tests are paired Wilcoxon over the 18 sessions, which are 9 subjects x 2
sessions — i.e. subject-level PSEUDO-REPLICATION (effective n closer to 9). The robustness arms
already added a subject-level Wilcoxon over the 9 subjects; this script closes the gap the proper
way, with a linear MIXED-EFFECTS model that keeps all 18 sessions but models the within-subject
correlation with a per-subject random intercept:

    d_ij = beta0 + u_i + e_ij ,   u_i ~ N(0, s_u^2)  (subject),  e_ij ~ N(0, s_e^2)  (session)

where d_ij is the paired per-session difference (condition A - condition B) in the SAME fair
position-residual per-trial-r space each arm already reports. beta0 is the population effect; its
p-value is the mixed-model test of "effect != 0" after accounting for subject clustering. This is
the reviewer-defensible replacement for the pooled 18-session Wilcoxon (which assumes 18 independent
units) and is reported next to it, plus the fully-independent subject-level test, for every contrast.

We re-analyse the SAVED per-session records of every arm (no re-decoding): 01 ensemble_distance,
06 attn_mtl, 07 fusion, 08 consistency, and the seed-AVERAGED 10/13 sweeps for the two attention
headlines. Nothing is re-fit on the MEG; this is a statistics-only pass over results/*.npz.

Run (env analysis/.venv_meggpt):  python 14_mixed_model.py  -> results/mixed_model.md
"""
import glob, os, warnings, numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon, ttest_rel
warnings.filterwarnings("ignore")

PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"
RES = f"{PROJ}/results"


def load_recs(npz):
    return list(np.load(f"{RES}/{npz}", allow_pickle=True)['recs'])


def seed_avg_recs(subdir):
    """Average each session's metrics over all seed_*.npz in results/<subdir> -> one rec/session."""
    per_sess = {}
    for fp in sorted(glob.glob(f"{RES}/{subdir}/seed_*.npz")):
        for r in np.load(fp, allow_pickle=True)['recs']:
            k = (int(r['subj']), int(r['sess']))
            per_sess.setdefault(k, []).append(r)
    out = []
    for (s, se), rs in sorted(per_sess.items()):
        keys = [k for k in rs[0].keys() if k not in ('subj', 'sess', 'seed')]
        rec = dict(subj=s, sess=se)
        for k in keys:
            rec[k] = float(np.mean([rr[k] for rr in rs]))
        out.append(rec)
    return out


def mixed_effect(recs, a, b):
    """Random-intercept LMM on paired differences d=a-b, grouped by subject.
    Returns dict of the population effect (beta0), its se/p, the ICC, and the naive/subject tests."""
    df = pd.DataFrame([dict(subj=int(r['subj']), sess=int(r['sess']),
                            d=float(r[a]) - float(r[b])) for r in recs])
    n = len(df)
    # --- primary: subject random-intercept mixed model on the differences ---
    # Scale d up for optimizer conditioning (the intercept test is scale-equivariant, so beta/se
    # scale back exactly and p is unchanged). If the subject variance component collapses to ~0 the
    # LMM Hessian goes singular; that IS the "no between-subject structure" case, where the correct
    # inference is an OLS intercept with subject-CLUSTER-robust SE -> use it as the fallback.
    C = 100.0
    dfc = df.assign(dc=df["d"] * C)
    try:
        md = smf.mixedlm("dc ~ 1", dfc, groups=dfc["subj"]).fit(reml=True, method="lbfgs")
        beta = float(md.params["Intercept"]) / C; se = float(md.bse["Intercept"]) / C
        p_lmm = float(md.pvalues["Intercept"])
        s_u = float(md.cov_re.iloc[0, 0]); s_e = float(md.scale)
        icc = s_u / (s_u + s_e) if (s_u + s_e) > 0 else 0.0
        method = "lmm"
    except np.linalg.LinAlgError:
        ols = smf.ols("d ~ 1", df).fit(cov_type="cluster", cov_kwds={"groups": df["subj"]})
        beta = float(ols.params["Intercept"]); se = float(ols.bse["Intercept"])
        p_lmm = float(ols.pvalues["Intercept"]); icc = 0.0; method = "ols_cluster"
    # --- references ---
    d = df["d"].values
    try: p_wil = wilcoxon(d).pvalue
    except Exception: p_wil = float("nan")
    p_t = float(ttest_rel([r[a] for r in recs], [r[b] for r in recs]).pvalue)
    # subject-level: average the two sessions within subject, then Wilcoxon over subjects
    subj_d = df.groupby("subj")["d"].mean().values
    try: p_subj = wilcoxon(subj_d).pvalue
    except Exception: p_subj = float("nan")
    return dict(mean=d.mean(), pos=int((d > 0).sum()), n=n, beta=beta, se=se, p_lmm=p_lmm,
                icc=icc, method=method, p_wil=p_wil, p_t=p_t, n_subj=len(subj_d),
                subj_pos=int((subj_d > 0).sum()), p_subj=p_subj)


# (arm label, loader, [(A,B,human-readable claim), ...])
ARMS = [
    ("01 routes/fusion/MTL (ensemble_distance)", lambda: load_recs("ensemble_distance.npz"),
     [("E_wt", "E_bag", "cross-order fusion > same-order bagging"),
      ("E_wt", "a_route", "fusion > best single route"),
      ("a_route", "v_route", "accel route > velocity route"),
      ("p_direct", "mtl_r1", "STL(p_direct) > rank-1 shared-bottleneck MTL"),
      ("p_direct", "mtl_r2", "STL(p_direct) > rank-2 shared-bottleneck MTL")]),
    ("06 attention-MTL, single seed (attn_mtl)", lambda: load_recs("ensemble_attn_mtl.npz"),
     [("latent_att_p", "latent_p", "cross-latent attention > per-order latent (no attn)"),
      ("latent_att_p", "stl_p", "attention-MTL > independent STL"),
      ("latent_att_fuse", "latent_att_p", "output fusion still adds on top of latent attention")]),
    ("07 learned fusion (ensemble_fusion)", lambda: load_recs("ensemble_fusion.npz"),
     [("stack_ridge", "perf_wt", "learned linear stack > fixed perf-weights"),
      ("stack_ridge", "E_bag", "learned stack > same-order bagging")]),
    ("08 consistency decoder (ensemble_consistency)", lambda: load_recs("ensemble_consistency.npz"),
     [("aux_p", "base_p", "auxiliary multi-task supervision > plain distance decoder"),
      ("cons_p", "aux_p", "physics-consistency term adds over aux (expected NULL)")]),
    ("10 attention-MTL, seed-averaged (seed_stability)", lambda: seed_avg_recs("seed_stability"),
     [("latent_att_p", "latent_p", "attention gain, seed-averaged (MLP backbone)"),
      ("latent_att_fuse", "latent_att_p", "output fusion complementarity, seed-averaged")]),
    ("13 seq attention-MTL, seed-averaged (seq_seed_stability)", lambda: seed_avg_recs("seq_seed_stability"),
     [("seq_latent_att_p", "seq_latent_p", "attention gain, seed-averaged (BiLSTM backbone)"),
      ("seq_latent_att_p", "seq_stl_p", "structured sharing > independent, seed-averaged (BiLSTM)")]),
]


def main():
    lines = ["# Subject random-intercept mixed-model re-analysis (script 14)\n",
             "Every headline contrast re-tested with a linear mixed model `d ~ 1 + (1|subject)` on the "
             "paired per-session differences (same fair position-residual per-trial-r space each arm "
             "reports). `beta` is the population effect (== mean diff), `p_lmm` its mixed-model p-value "
             "(accounts for the 9-subject x 2-session clustering); shown next to the pooled 18-session "
             "Wilcoxon (`p_wil`, assumes 18 independent — anti-conservative) and the fully-independent "
             "subject-level Wilcoxon (`p_subj`, n=9). ICC = between-subject share of the difference "
             "variance.\n",
             "| arm | contrast | claim | beta (mean) | pos/n | ICC | p_lmm | p_wil(n18) | p_subj(n9) |",
             "|---|---|---|---|---|---|---|---|---|"]
    print("=== mixed-model re-analysis ===")
    for label, loader, contrasts in ARMS:
        recs = loader()
        if not recs:
            print(f"[skip] {label}: no records"); continue
        for a, b, claim in contrasts:
            r = mixed_effect(recs, a, b)
            flag = "†" if r["method"] == "ols_cluster" else ""   # dagger = RE var collapsed -> cluster-OLS
            row = (f"| {label} | {a} vs {b} | {claim} | {r['beta']:+.4f} | {r['pos']}/{r['n']} | "
                   f"{r['icc']:.2f} | {r['p_lmm']:.2e}{flag} | {r['p_wil']:.2e} | "
                   f"{r['p_subj']:.2e} ({r['subj_pos']}/{r['n_subj']}) |")
            lines.append(row)
            print(f"  {label[:28]:28s} {a:16s} vs {b:16s}: beta={r['beta']:+.4f} "
                  f"p_lmm={r['p_lmm']:.2e}  p_wil={r['p_wil']:.2e}  p_subj={r['p_subj']:.2e}", flush=True)

    lines += ["\n## Reading",
              "- Where `p_lmm` stays significant, the effect survives proper accounting of subject "
              "clustering — not an artefact of counting 18 correlated sessions as independent.",
              "- A large gap between `p_wil(n18)` and `p_lmm`/`p_subj` flags a contrast whose apparent "
              "strength was inflated by pseudo-replication; the mixed model is the honest number.",
              "- High ICC means the difference is consistent within subject (between-subject variance "
              "dominates) — the mixed model then behaves close to the n=9 subject-level test.",
              "- `†` = the subject random-effect variance collapsed to ~0 (ICC 0); the mixed model is "
              "then singular and we report the equivalent OLS intercept with subject-cluster-robust SE "
              "(same population effect, honest clustered inference)."]
    with open(f"{RES}/mixed_model.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nSAVED {RES}/mixed_model.md")


if __name__ == "__main__":
    main()
