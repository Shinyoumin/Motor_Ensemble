# Bandwidth-matched control for cross-order decoder fusion in single-trial MEG

Analysis code and per-run result archives for

> **Spectral smoothing explains the offline gain from cross-order decoder fusion in single-trial MEG
> movement reconstruction: implications for non-invasive BCI**
> Youmin Shin, Seoul AI School, aSSIST University — *Journal of Neural Engineering*, under revision
> (JNE-111541)

Everything needed to reproduce every number, table and figure in the manuscript is here, including
the cross-validation seeds, the spectral-filter estimation and the multi-seed training scripts. This
repository is public from the revision stage, before acceptance, because the paper's contribution is
a reproducibility-oriented control procedure.

---

## Data

The MEG recordings are **not redistributed here**. They are the publicly available dataset of

> Yeom HG, Kim JS, Chung CK. *A magnetoencephalography dataset during three-dimensional reaching
> movements for brain-computer interfaces.* Scientific Data 2023;10:552.
> doi:10.1038/s41597-023-02454-y · data: doi:10.6084/m9.figshare.c.6431021.v1 (CC-BY 4.0)

Download the raw (tSSS-filtered) recordings and place the gradiometer epochs as
`00_preprocessed_data/Sub_{1..9}_ses_{1,2}_grad_epoched.fif`. The scripts read from that layout and
write to `results/`.

What the analysis uses from the released files: the 192 gradiometers common to all 18 recordings
(individual runs contain 195–204), all 120 trials of every run (no trials are rejected), and the
three auxiliary channels carrying the three-axis accelerometer. Magnetometers are not present in the
released derivative. The stored epochs are band-limited at acquisition to 1–40 Hz, so the effective
passband of every decoder here is **1–12 Hz**.

---

## Reproducing the paper

```bash
python scripts/01_ensemble_distance.py full      # core ridge arm  -> Table 1 (ridge row)
python scripts/03_backbone_ensemble.py ssl full  # frozen SSL backbone
python scripts/03_backbone_ensemble.py rand full # random-projection backbone
python scripts/04_lstm_ensemble.py full          # trained LSTM (supplement)
python scripts/17_spectral_control.py full       # bandwidth-matched pseudo-fusion  -> the central result
python scripts/45_espec_implementation_controls.py full   # filter-implementation sweep
python scripts/46_revision_controls.py full      # ocular, R2, non-positive-lag controls
python scripts/47_confidence_intervals.py        # bootstrap CIs + exact sign-flip tests
```

Figures:

```bash
python scripts/48_fig1_schematic.py     # Fig. 1  schematic of routes, operators, pseudo-fusion
python scripts/38_fig1_benchmark.py     # Fig. 2  fusion vs same-order bagging
python scripts/39_fig2_bandwidth.py     # Fig. 3  bandwidth decomposition
python scripts/40_fig3_architecture.py  # Fig. 4  architecture controls
python scripts/41_fig4_realtime.py      # Fig. 5  causal operation and practical bounds
python scripts/42_figS1_supplement.py   # Fig. S1 descriptive analyses
```

Each figure script writes a working copy to `figures/` and a 300 dpi submission copy plus one file
per subplot to `figures_submission_300dpi/` (see `figures/` in this repository).

---

## Determinism: seeds and splits

Every analysis in the paper uses the same splits. This was not true of the originally submitted
version, and fixing it is one of the revisions.

| stream | seed | used for |
|---|---|---|
| cross-validation folds | `1000 * participant + run` | the 5-fold split of each run |
| bagging resamples | `7000 + 100 * participant + run` | the 80% resamples of E_bag and its controls |
| training seeds | `0 … 5` | network initialisation and minibatch order in the multi-seed sweeps |

Folds are seed-independent of the training seed, so the multi-seed sweeps vary only initialisation
and batch order. No analysis draws from a stream shared across runs. The pre-fix archives, whose
splits depended on execution order, are kept in `results/_archive_globalrng/` and are reported in the
paper as a sensitivity analysis only.

---

## Layout

```
scripts/     all analysis and figure code (numbered in the order they were written)
results/     per-run archives (.npz) and the report for each analysis (.md)
figures/     300 dpi figures (PNG + vector PDF) and panels/ with each subplot separately
docs/        the additional-analysis report prepared for the revision
```

### The control itself

The bandwidth-matched pseudo-fusion is implemented in `scripts/17_spectral_control.py`
(`mag_filter`, `apply_mag`) and varied in `scripts/45_espec_implementation_controls.py`. In brief:
for each training trial the route predictions are transformed by a real FFT over the 2.8 s analysis
window; the squared magnitudes are averaged over **training trials only**; the filter magnitude is
the square root of the ratio of that average to the direct route's; and the filters are applied to
the held-out direct-route prediction, which is then fused with the same weighting rule as the real
ensemble. The resulting object has the ensemble's bandwidth diversity and no cross-order neural
content.

---

## Environment

Python 3.10 with MNE-Python 1.12.1, NumPy, SciPy, scikit-learn, statsmodels and Matplotlib; the
deep-network arms additionally require PyTorch 2.5.1 (CUDA 12.1). Results in the paper were produced
on an NVIDIA RTX 3090. See `requirements.txt`.

## Licence

Code released under the MIT licence. The MEG data are the originators' and are distributed by them
under CC-BY 4.0; cite Yeom et al. (2023) if you use them.
