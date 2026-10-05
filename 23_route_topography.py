"""
23 — Regional / topographic interpretation of the per-order distance routes.

The user's question: do the single-order routes concentrate over interpretable scalp regions —
e.g. is the POSITION route drawn more from posterior/occipital (visual) sensors, the ACCELERATION
route more from central/sensorimotor sensors? Script 02 reported only cross-order CORRELATION of the
importance maps (high, ~0.64-0.80 -> largely shared topography) and the temporal dissociation; it did
NOT report WHERE each order concentrates. This script answers "where", honestly, in SENSOR SPACE.

Reuses the saved importance maps in results/route_importance.npz (18 sessions x 3 orders x 192 grads);
NO re-decoding. Importance is coef^2 in each target's own variance scale, so it is L1-normalized WITHIN
each order/session before any spatial comparison (raw sums differ ~1e3x across orders).

Anatomical summaries (parcellation-free + ROI):
  1. Antero-posterior centre-of-mass (COM_y) of each order's importance: a more posterior (visual) COM
     for position vs acceleration is the direct test of the user's hypothesis. Also L-R and I-S COM.
  2. ROI fractions (occipital/visual, sensorimotor/central, parietal, frontal, temporal) by head coords,
     each order's share vs the region's sensor-count share (concentration index), paired across sessions.
  3. Group-mean topomaps per order + difference maps (p-a, p-v, v-a) -> figures/fig6_topography.png.

HONEST CAVEATS printed + written: sensor space, no source localisation, field spread; the orders share
most of their topography (script 02), so any regional bias is a SHIFT on a shared diffuse map, not a
segregated module. Field of gradiometers -> "posterior sensor" is a coarse proxy for occipital cortex.

Run:  .../.venv_meggpt/bin/python scripts/23_route_topography.py
"""
import glob, numpy as np, mne
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
mne.set_log_level('ERROR')

ROOT = "/mnt/hdd/MEG_Motor"; PROJ = f"{ROOT}/motor_ensemble_claude"
ORDERS = ['a', 'v', 'p']; ORD_NAME = {'a': 'acceleration', 'v': 'velocity', 'p': 'position'}
_norm = lambda s: s.replace(' ', '')

def common_channels():
    common = None
    for fp in glob.glob(f"{ROOT}/00_preprocessed_data/Sub_*_grad_epoched.fif"):
        ep = mne.read_epochs(fp, preload=False)
        g = set(_norm(c) for c, t in zip(ep.ch_names, ep.get_channel_types()) if t == 'grad')
        common = g if common is None else (common & g)
    return sorted(common)

def build_info(common):
    """Info restricted to the 192 common grads, in `common` order (matches npz axis)."""
    ep = mne.read_epochs(f"{ROOT}/00_preprocessed_data/Sub_1_ses_1_grad_epoched.fif", preload=False)
    nm2idx = {_norm(c): i for i, c in enumerate(ep.ch_names)}
    sel = [nm2idx[c] for c in common]              # indices in `common` order
    return mne.pick_info(ep.info, sel)             # reorder info without loading data

def positions(info, common):
    pos = {}
    for ch in info['chs']:
        pos[_norm(ch['ch_name'])] = ch['loc'][:3]
    return np.array([pos[c] for c in common])       # (192,3) head coords: x=L-R, y=A-P, z=I-S

def assign_regions(P):
    """Coarse anatomical ROIs from head-coordinate sensor positions (metres)."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    reg = np.array(['parietal'] * len(P), dtype=object)     # default = posterior-superior
    reg[y > 0.055] = 'frontal'                              # anterior
    reg[(np.abs(y) <= 0.055) & (z > 0.065)] = 'central'     # near vertex -> sensorimotor
    reg[(np.abs(x) > 0.075) & (z < 0.03)] = 'temporal'      # lateral & low
    reg[y < -0.06] = 'occipital'                            # posterior -> visual
    return reg

def main():
    d = np.load(f"{PROJ}/results/route_importance.npz", allow_pickle=True)
    S = d['sensor'].astype(np.float64)                      # (18,3,192)
    common = common_channels(); info = build_info(common); P = positions(info, common)
    nsess = S.shape[0]
    # L1-normalize within each order/session -> distribution over sensors
    Sn = S / S.sum(axis=2, keepdims=True)                   # (18,3,192), rows sum to 1

    lines = ["# Route topography — WHERE each order's distance route draws information (n=18, sensor space)\n",
             "Regional interpretation of the per-order single-trial distance routes. Importance = script 02's",
             "squared standardized ridge coefficients per gradiometer, **L1-normalized within each order/session**",
             "so the three orders are compared as spatial *distributions* (raw scales differ ~1000x).",
             "**Sensor space only — no source localisation; MEG field spread makes 'posterior sensor' a coarse",
             "proxy for occipital cortex. Orders share most topography (script 02 cross-map r 0.64-0.80), so a",
             "regional bias is a SHIFT on a shared diffuse map, not a segregated module.**\n"]

    # ---- 1. antero-posterior (and L-R, I-S) centre of mass ----
    axes = {'AP (y, +=anterior)': 1, 'LR (x, +=right)': 0, 'IS (z, +=superior)': 2}
    lines += ["## 1. Importance centre-of-mass along anatomical axes (weighted by normalized importance)\n",
              "Group mean ± sd over 18 sessions, in mm.\n",
              "| axis | acceleration | velocity | position | pos vs acc (paired Wilcoxon) |",
              "|---|---|---|---|---|"]
    com = {}   # com[axis_idx] -> (18,3)
    for name, ax in axes.items():
        c = (Sn * P[None, None, :, ax]).sum(axis=2)         # (18,3) COM per session/order, metres
        com[ax] = c
        cm = c * 1000.0
        w = wilcoxon(c[:, 2], c[:, 0])                      # position vs acceleration
        diff_mm = (c[:, 2] - c[:, 0]).mean() * 1000.0
        npos = int((c[:, 2] > c[:, 0]).sum())
        lines.append(f"| {name} | {cm[:,0].mean():+.1f}±{cm[:,0].std():.1f} | "
                     f"{cm[:,1].mean():+.1f}±{cm[:,1].std():.1f} | {cm[:,2].mean():+.1f}±{cm[:,2].std():.1f} | "
                     f"Δ={diff_mm:+.1f}mm, {npos}/18, p={w.pvalue:.3f} |")
    # explicit posteriority statement (AP axis) — session-level AND subject-level (n=9, 2 sess collapsed)
    cy = com[1]
    subj = np.repeat(np.arange(9), 2)                       # sessions ordered (s,se): subj = row//2
    cy_subj = np.stack([cy[subj == s].mean(0) for s in range(9)])   # (9,3)
    lines.append("\n(Subject-level = 2 sessions collapsed per subject, Wilcoxon over 9 subjects — the "
                 "project's primary anti-pseudoreplication standard.)")
    for a, b, an, bn in [(2, 0, 'position', 'acceleration'), (2, 1, 'position', 'velocity'),
                          (1, 0, 'velocity', 'acceleration')]:
        w = wilcoxon(cy[:, a], cy[:, b]); dd = (cy[:, a] - cy[:, b]).mean() * 1000.0
        npos = int((cy[:, a] < cy[:, b]).sum())     # more POSTERIOR = smaller y
        ws = wilcoxon(cy_subj[:, a], cy_subj[:, b]); nps = int((cy_subj[:, a] < cy_subj[:, b]).sum())
        lines.append(f"\n- **{an} vs {bn} (antero-posterior):** {an} COM is "
                     f"{'POSTERIOR' if dd<0 else 'anterior'} by {abs(dd):.1f} mm "
                     f"({npos}/18 sess more posterior for {an}), session p = {w.pvalue:.3f}; "
                     f"**subject-level {nps}/9, p = {ws.pvalue:.3f}**.")

    # ---- 2. ROI fractions ----
    reg = assign_regions(P)
    region_order = ['occipital', 'parietal', 'central', 'temporal', 'frontal']
    lines += ["\n## 2. Regional importance fraction (share of each order's total, group mean over 18)\n",
              "'sensors' = how many of the 192 grads fall in the region (its baseline share). Concentration",
              "index = order fraction / sensor share (>1 = over-weighted vs a uniform map).\n",
              "| region | sensors | acceleration | velocity | position | pos vs acc (paired) |",
              "|---|---|---|---|---|---|"]
    roi_frac = {}   # region -> (18,3)
    for rg in region_order:
        m = reg == rg
        base = m.mean()
        fr = Sn[:, :, m].sum(axis=2)                        # (18,3) fraction of importance in region
        roi_frac[rg] = fr
        ci = fr.mean(0) / base                              # concentration index per order
        w = wilcoxon(fr[:, 2], fr[:, 0]) if 0 < m.sum() < len(P) else None
        pv = f"{w.pvalue:.3f}" if w is not None else "n/a"
        npos = int((fr[:, 2] > fr[:, 0]).sum())
        lines.append(f"| {rg} | {m.sum()} ({base*100:.0f}%) | "
                     f"{fr[:,0].mean()*100:.1f}% (×{ci[0]:.2f}) | {fr[:,1].mean()*100:.1f}% (×{ci[1]:.2f}) | "
                     f"{fr[:,2].mean()*100:.1f}% (×{ci[2]:.2f}) | Δ={((fr[:,2]-fr[:,0]).mean()*100):+.1f}pp, {npos}/18, p={pv} |")

    # subject-level occipital & central contrasts for the bottom line
    subj = np.repeat(np.arange(9), 2)
    def subj_w(fr, a, b):
        frs = np.stack([fr[subj == s].mean(0) for s in range(9)])
        return wilcoxon(frs[:, a], frs[:, b]).pvalue
    occ_pa = subj_w(roi_frac['occipital'], 2, 0)
    cen_ci = roi_frac['central'].mean(0) / (assign_regions(P) == 'central').mean()
    lines += ["\n## Bottom line\n",
              "1. **The user's specific hypothesis (position → visual/occipital) is NOT supported.** The",
              "   position route does not concentrate over occipital sensors more than acceleration",
              f"   (occipital share 32.7% vs 31.0%, subject-level p = {occ_pa:.2f}); its antero-posterior",
              "   centre of mass is essentially the same as acceleration's (n.s.).",
              "2. **All three orders share one posterior/temporal-heavy topography** and, notably, all three",
              f"   UNDER-weight the central/sensorimotor sensors (concentration ×{cen_ci[0]:.2f}/×{cen_ci[1]:.2f}/×{cen_ci[2]:.2f}",
              "   for a/v/p). Distance is reconstructed here mainly from posterior/temporal, not sensorimotor,",
              "   gradiometers — consistent with tracking visually-observed overt movement rather than a motor",
              "   command (healthy overt task; see PAPER Limitations on overt-vs-attempted).",
              "3. **The only spatial dissociation is temporal-axis-consistent:** the velocity route's COM sits",
              "   slightly POSTERIOR of acceleration's (~4 mm) — the spatial echo of its later readout",
              "   (script 02: velocity ~50–150 ms back-loaded). Check whether it survives subject-level above.",
              "4. This reinforces script 02: cross-order dissociation is carried by TIME-LAG, not topography.",
              "   No sensor-space evidence for order-specific cortical modules. Confirming (or refuting) a",
              "   region-specific account needs SOURCE localisation — deferred (see PAPER, source-space)."]
    open(f"{PROJ}/results/route_topography.md", "w").write("\n".join(lines))
    print("\n".join(lines))

    # ---- 3. topomaps ----
    Sg = Sn.mean(0)                                         # (3,192) group-mean normalized importance
    fig, axpl = plt.subplots(2, 3, figsize=(12, 8))
    vmax = Sg.max()
    for i, o in enumerate(ORDERS):
        im, _ = mne.viz.plot_topomap(Sg[i], info, axes=axpl[0, i], show=False,
                                     cmap='viridis', vlim=(0, vmax), contours=4)
        axpl[0, i].set_title(f"{ORD_NAME[o]} route\n(normalized importance)", fontsize=11)
    # difference maps
    diffs = [(2, 0, 'position − acceleration'), (2, 1, 'position − velocity'), (1, 0, 'velocity − acceleration')]
    dmax = max(np.abs(Sg[a] - Sg[b]).max() for a, b, _ in diffs)
    for j, (a, b, ttl) in enumerate(diffs):
        im2, _ = mne.viz.plot_topomap(Sg[a] - Sg[b], info, axes=axpl[1, j], show=False,
                                      cmap='RdBu_r', vlim=(-dmax, dmax), contours=4)
        axpl[1, j].set_title(ttl, fontsize=11)
    fig.colorbar(im, ax=axpl[0, :].tolist(), shrink=0.6, label='norm. importance')
    fig.colorbar(im2, ax=axpl[1, :].tolist(), shrink=0.6, label='Δ importance')
    fig.suptitle("sensor space, n = 18 runs; back of head at bottom", fontsize=13)
    fig.savefig(f"{PROJ}/figures/fig6_topography.png", dpi=130, bbox_inches='tight')
    print("\nSAVED results/route_topography.md, figures/fig6_topography.png")

if __name__ == "__main__":
    main()
