"""
50 — The three remaining referee items that needed computing (revision).

(A) ACCELEROMETER PC1 EXPLAINED VARIANCE vs the per-run fusion margin. Referee 1 asked whether the
    results vary with the explained variance of the principal component that defines the acceleration
    target. We recompute the per-run explained-variance ratio from the released auxiliary channels
    (the same window and sign convention as the target construction) and relate it to the per-run
    fusion margin and bandwidth-clean residual.

(B) PER-ROUTE ACCURACY AT EVERY LATENCY BUDGET. Referee 1 asked why the fusion margin is
    non-monotonic across latency budgets. The explanation rests on how each route behaves as the
    budget tightens, so the per-route accuracies at each tau are tabulated.

(C) UNROUNDED PARTICIPANT- AND RUN-LEVEL VALUES. Referee 2 asked for unrounded results so that the
    relationships among reported comparisons can be verified. Every headline quantity is exported at
    full precision, per run and per participant, as CSV.

Run:  python 50_remaining_revision_items.py  ->  results/remaining_items.md + results/unrounded_*.csv
"""
import numpy as np, mne, csv, warnings
from scipy.stats import wilcoxon, spearmanr, pearsonr

warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")
PROJ = "/mnt/hdd/MEG_Motor/motor_ensemble_claude"; ROOT = "/mnt/hdd/MEG_Motor"
R = f"{PROJ}/results"
BASE, MOVE = (2.0, 3.8), (4.0, 5.8)
ACC_CH = ["EMG062", "EMG063", "EMG064"]


def L(stem, key="recs"): return list(np.load(f"{R}/{stem}.npz", allow_pickle=True)[key])
def sj(rs, k):
    s = np.array([r["subj"] for r in rs]); v = np.array([r[k] for r in rs], float)
    return np.array([v[s == u].mean() for u in np.unique(s)])


# ---------------- (A) accelerometer PC1 explained variance ----------------
def pc1_evr(subj, sess):
    ep = mne.read_epochs(f"{ROOT}/00_preprocessed_data/Sub_{subj}_ses_{sess}_grad_epoched.fif",
                         preload=True, verbose="ERROR")
    # Channel names are inconsistent across the released files ("EMG062" vs "EMG 062"),
    # so match on the whitespace-stripped name as the rest of the pipeline does.
    norm = {c.replace(" ", ""): c for c in ep.ch_names}
    picks = [norm[c] for c in ACC_CH if c in norm]
    assert len(picks) == 3, f"accelerometer channels not found: {picks}"
    X = ep.get_data(picks=picks)                                  # (trials, 3, time)
    t = ep.times
    b = (t >= BASE[0]) & (t <= BASE[1]); m = (t >= MOVE[0]) & (t <= MOVE[1])
    X = X - X[:, :, b].mean(-1, keepdims=True)                    # baseline correction
    M = X[:, :, m]                                                # movement window
    F = M.transpose(0, 2, 1).reshape(-1, M.shape[1])              # (trials*time, 3)
    F = F - F.mean(0)
    ev = np.linalg.svd(F, full_matrices=False)[1] ** 2
    return float(ev[0] / ev.sum())


core = L("ensemble_distance"); sp = L("spectral_control")
rows = []
for r in core:
    s, se = int(r["subj"]), int(r["sess"])
    m = next(x for x in sp if x["subj"] == s and x["sess"] == se)
    rows.append(dict(subj=s, sess=se, evr=pc1_evr(s, se),
                     margin=r["E_wt"] - r["E_bag"],
                     residual=m["E_cross"] - m["E_spec"],
                     E_cross=r["E_wt"], E_bag=r["E_bag"], p_direct=r["p_direct"],
                     a_route=r["a_route"], v_route=r["v_route"],
                     E_spec=m["E_spec"]))
evr = np.array([x["evr"] for x in rows]); marg = np.array([x["margin"] for x in rows])
res = np.array([x["residual"] for x in rows])
rho_m, p_m = spearmanr(evr, marg); rho_r, p_r = spearmanr(evr, res)
r_m, pp_m = pearsonr(evr, marg)

# ---------------- (B) per-route accuracy at every latency budget ----------------
cz = L("causal_latency"); taus = [0, 50, 100, 150]

# ---------------- (C) unrounded export ----------------
with open(f"{R}/unrounded_run_level.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
subs = np.unique([x["subj"] for x in rows])
part = []
for u in subs:
    rr = [x for x in rows if x["subj"] == u]
    part.append({k: (u if k == "subj" else np.mean([x[k] for x in rr]))
                 for k in rows[0].keys() if k != "sess"})
with open(f"{R}/unrounded_participant_level.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(part[0].keys())); w.writeheader(); w.writerows(part)

# ---------------- report ----------------
out = ["# Remaining revision items — script 50", "",
       "## A. Accelerometer PC1 explained variance and the fusion margin", "",
       "The acceleration target is the first principal component of the three-axis accelerometer over",
       "the 4.0–5.8 s movement window. Referee 1 asked whether results vary with its explained variance.",
       "", "| participant | run 1 EVR | run 2 EVR |", "|---|---|---|"]
for u in subs:
    e = [x["evr"] for x in rows if x["subj"] == u]
    out.append(f"| {u} | {e[0]:.3f} | {e[1]:.3f} |")
out += ["", f"Across the 18 runs the explained variance is {evr.mean():.3f} on average "
        f"(range {evr.min():.3f}–{evr.max():.3f}).", "",
        "| association | Spearman ρ | *p* | Pearson *r* | *p* |", "|---|---|---|---|---|",
        f"| EVR vs per-run fusion margin (E_cross − E_bag) | {rho_m:+.3f} | {p_m:.3f} | {r_m:+.3f} | {pp_m:.3f} |",
        f"| EVR vs per-run bandwidth-clean residual | {rho_r:+.3f} | {p_r:.3f} | — | — |", "",
        "## B. Per-route accuracy at each output-latency budget", "",
        "Participant-level means (n = 9). This is the basis of the non-monotonicity explanation: the",
        "acceleration route loads early and is retained as the budget tightens, whereas the velocity",
        "route erodes, and the bagging baseline — built entirely from the direct route — loses most.",
        "", "| τ (ms) | a_route | v_route | p_direct | E_bag | E_cross | E_cross − E_bag |",
        "|---|---|---|---|---|---|---|"]
for t_ in taus:
    a_ = sj(cz, f"a_route@{t_}").mean(); v_ = sj(cz, f"v_route@{t_}").mean()
    p_ = sj(cz, f"p_direct@{t_}").mean(); eb = sj(cz, f"E_bag@{t_}").mean()
    ec = sj(cz, f"E_cross@{t_}").mean()
    out.append(f"| {t_} | {a_:.3f} | {v_:.3f} | {p_:.3f} | {eb:.3f} | {ec:.3f} | {ec-eb:+.3f} |")
out += ["", "## C. Unrounded results", "",
        "`results/unrounded_run_level.csv` (18 rows) and `results/unrounded_participant_level.csv`",
        "(9 rows) give every headline quantity at full precision, so the relationships among the",
        "reported comparisons can be verified directly."]
open(f"{R}/remaining_items.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
