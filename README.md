# ML4SCI – HEPSIM GSoC 2026 Evaluation
### Quark vs. Gluon Jet Analysis · Aditya Patwa

This repo contains my solution to the ML4SCI HEPSIM evaluation task for Google Summer of Code 2026. The task covers data loading, jet physics observables, Lorentz boosts, and a binary classifier — all on the [Pythia 8 Quark and Gluon Jets dataset](https://zenodo.org/records/3164691).

---

## What's in the notebook

The notebook is structured to match the four parts of the evaluation exactly.

**Part (a) — Data loading and exploration**  
Loads up to 500k jets (5 files × 100k jets) from the Zenodo dataset. The tricky part here is that each `.npz` file zero-pads to its *own* maximum constituent dimension — so you can't naively `np.concatenate` across files. I re-pad everything to the global maximum first, then build a boolean mask wherever `pT > 0` to exclude padding from all downstream computations. Part (a) reports total constituent counts per class, constituent multiplicity histograms, and the pT/η distributions of the leading constituent in each jet.

**Part (b) — Jet observables**  
Computes three standard HEP jet observables from constituent four-momenta: jet invariant mass, jet width (pT-weighted mean ΔR), and pT dispersion. Each is plotted separately for quark and gluon jets. The four-vector conversion from (pT, rapidity, φ, pdgid) to (E, px, py, pz) is done assuming massless constituents, which is the standard approximation in this dataset.

**Part (c) — Lorentz boost to the jet rest frame**  
Implements a full general-direction Lorentz boost (not just along z). Given a jet with 4-momentum p_J, the boost vector is β = p⃗_J / E_J, with γ = 1/√(1−β²). The boost is applied to every constituent in every jet in one vectorised pass. Verification: the total 3-momentum in the rest frame is numerically zero to floating-point precision (~1e-10 GeV). The visualisation shows constituent momenta in the (px, py) plane for example quark and gluon jets side by side, with marker size proportional to energy.

**Part (d) — Classification**  
Trains a histogram gradient boosted decision tree (sklearn's `HistGradientBoostingClassifier`) on six features — four lab-frame and two rest-frame quantities. Reports ROC curve and AUC, a confusion matrix at a physics-motivated working point (70% quark efficiency), and permutation importance to identify the most discriminating feature. Also runs a three-way comparison: lab-frame only vs. rest-frame only vs. combined, to directly answer whether the Lorentz boost actually helps.

---

## Key results

| | |
|---|---|
| **Classifier AUC** | ~0.87 |
| **Most discriminating feature** | Multiplicity |
| **Rest-frame gain over lab-frame** | Marginal (~0.005 AUC) |
| **Boost verification** | Mean residual 3-momentum < 1e-10 GeV |

The dominant discriminator being multiplicity makes sense physically: gluons carry a larger QCD color charge (C_A = 3 vs. C_F = 4/3 for quarks), so they radiate more and produce denser jets on average. This is consistent with the established literature on quark/gluon jet tagging.

The rest-frame boost adds a small amount of information on top of lab-frame features, but most of the discriminating power is already captured by the lab-frame observables. The gain is real but not dramatic, which is also what you'd expect — the physics that separates quarks from gluons (color charge, radiation pattern) is frame-independent.

---

## Running the notebook

The notebook downloads the dataset automatically from Zenodo — no manual download needed. Just run it from top to bottom.

```bash
pip install numpy matplotlib scikit-learn
jupyter notebook Agentic_Lagrangian_Extraction_GSoC_Test.ipynb
```

Requires Python 3.9+. The download step fetches ~500MB across 5 files and caches them in a local `data/` directory, so subsequent runs are fast. Total runtime is roughly 5–10 minutes depending on your machine.

---

## Dataset

[Pythia8 Quark and Gluon Jets for Energy Flow](https://zenodo.org/records/3164691) — Komiske, Metodiev, Thaler (2019).

Each file has 100k jets (50k quark, 50k gluon), stored as variable-length constituent arrays zero-padded to a common width. I used the standard `QG_jets*.npz` files only, not the `withbc` variants.

---

## Context

This notebook is my submission for the ML4SCI HEPSIM evaluation task as part of my GSoC 2026 application. The associated project proposal is for **Agentic Lagrangian Extraction from the Literature** — building an agentic system inside the HEPTAPOD framework that takes a BSM scenario description and automatically searches the literature, extracts the Lagrangian, generates a FeynRules `.fr` model file, and validates it.

---

*Aditya Patwa · adityapatwa.tech@gmail.com · [github.com/Aditya-Patwa](https://github.com/Aditya-Patwa)*
