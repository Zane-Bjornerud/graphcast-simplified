# GraphCast Simplified: Weather Forecasting with Graph Neural Networks

A simplified, reproducible reimplementation of [GraphCast](https://www.science.org/doi/10.1126/science.adi2336) (Lam et al., 2023), DeepMind's graph-neural-network weather model, built to run on a single consumer GPU. This project reproduces the core **encode–process–decode** architecture at a fraction of the original's scale, evaluates it honestly against a persistence baseline, and provides a detailed failure-mode analysis of where and why a heavily simplified model breaks down.

> **TL;DR:** This is a *negative-result* reproduction. The simplified model does **not** beat the trivial persistence baseline. The value of the project is in the careful, honest analysis of *why*: systematic spatial biases, MSE-induced smoothing, and the compute/data trade-offs that separate research prototypes from production systems.

> **Note on attribution:** This repository is a **fork of [DeepMind's original GraphCast repository](https://github.com/google-deepmind/graphcast)**. All of my own work, the simplified mesh, model, data pipeline, training, and evaluation, lives in [`ZaneFileAdditions/`](ZaneFileAdditions/). The `graphcast/`, `docs/`, `build/` directories, `setup.py`, and the original demo notebooks are DeepMind's code, retained for reference and reuse under the original Apache 2.0 license.

---

## Overview

Traditional numerical weather prediction (NWP) solves atmospheric equations on discretized grids at high computational cost. GraphCast showed that a graph neural network could match ECMWF's operational HRES system while running orders of magnitude faster. However, the full architecture (36M+ parameters, 40,962-node multi-mesh, TPU-scale training) is out of reach for this academic reproduction.

This project asks a narrower question: **what happens to forecast skill when you aggressively simplify GraphCast to fit on one GPU?** By stripping the model down and measuring where it fails, the work identifies which components are essential and which simplifications are most damaging.

## Key details

| Aspect | This project | Original GraphCast |
| --- | --- | --- |
| Parameters | 3.5M | 36M+ |
| Mesh | Single icosahedral, 162 nodes | Multi-mesh, 40,962 finest nodes |
| Message-passing layers | 6 | 16+ |
| Embedding dimension | 128 | 512 |
| Spatial resolution | 32×64 grid (5.625°) | 0.25° |
| Variables | 2 (Z500, T2m) | 227 |
| Hardware | 1× NVIDIA RTX 3090 | TPU pods |
| Framework | JAX · Haiku · Jraph | JAX |

## Results

The model was evaluated on 500 test samples across four lead times (6h–24h) against a **persistence baseline** (predict that tomorrow equals today), the minimum bar any useful forecaster must clear.

- The simplified model **underperforms persistence** at all T2m lead times and at short Z500 lead times, reaching parity with persistence only at t+4 for Z500.
- Predictions capture large-scale atmospheric structure (troughs, ridges, latitudinal gradients) but are **excessively smooth**, a signature of MSE-trained models regressing toward conditional means when uncertain about fine-scale features.
- Error analysis reveals **structured spatial biases** rather than random noise, including a pronounced cold bias in subtropical regions that worsens over the forecast horizon.

See the paper (`/paper`) for full RMSE curves, prediction-vs-target maps, and error visualizations.

## Why the model underperforms

The write-up attributes the gap to four interacting factors, each a deliberate simplification:

1. **Insufficient capacity** — a 10× parameter reduction limits the model's ability to learn complex dynamics.
2. **Limited training data** — subsampled data (every 4th timestep, 1979–2015) yields only ~1,460 sequences vs. the original's 40+ years at full resolution.
3. **Coarse resolution** — the 5.625° grid (~625 km spacing) discards fine-scale features that matter for surface variables like T2m.
4. **Single-scale mesh** — one mesh level (vs. multi-mesh) limits both local interactions and efficient long-range information flow.

## Architecture

The model follows the encode–process–decode pattern:

- **Encoder** — a 2-layer MLP (SiLU) maps grid features to 128-dim embeddings, pooled onto mesh nodes via k-nearest-neighbor inverse-distance weighting, then combined with mesh node coordinates.
- **Processor** — 6 message-passing layers (Jraph `GraphNetwork`) that update edge features from sender/receiver/distance, then aggregate messages to update nodes.
- **Decoder** — mesh embeddings are unpooled back to the 32×64 grid and passed through an MLP to predict residuals, which are added to the input state.

The mesh is a single-level icosahedral grid (2 subdivisions → 162 nodes, 480 edges) using 3D Cartesian coordinates to avoid the pole singularities of lat–lon grids.

## Repository structure

All original contributions live in `ZaneFileAdditions/`. The rest of the repository is DeepMind's GraphCast code (see the attribution note above).

```
graphcast-simplified/
├── ZaneFileAdditions/              # ← my simplified reimplementation
│   ├── simple_mesh.py              # icosahedral mesh construction (162 nodes)
│   ├── simple_grid_mesh_mapping.py # bidirectional grid ↔ mesh kNN mappings
│   ├── simple_graphcast.py         # encode–process–decode GNN model
│   ├── simple_model.py             # network / MLP components
│   ├── data_loading.py             # WeatherBench2 loading
│   ├── explore_data.py             # data inspection utilities
│   ├── prepare_data_for_jax.py     # preprocessing into JAX-ready arrays
│   ├── train.py                    # autoregressive training loop
│   ├── eval.py                     # evaluation vs. persistence baseline
│   ├── simpleWeatherBenchTest.py   # WeatherBench data sanity checks
│   └── test_graphcast.py           # model tests
├── graphcast/                      # DeepMind original (forked)
├── docs/  ·  build/  ·  setup.py   # DeepMind original (forked)
└── README.md
```

## Setup

```bash
git clone https://github.com/Zane-Bjornerud/graphcast-simplified.git
cd graphcast-simplified
pip install -e .          # installs the forked graphcast package + dependencies
```

Core dependencies: `jax`, `dm-haiku`, `jraph`, plus the standard scientific stack (`numpy`, `xarray`) for handling [WeatherBench2](https://weatherbench2.readthedocs.io/) data. My additions in `ZaneFileAdditions/` reuse the JAX/Haiku/Jraph stack pulled in by the original `setup.py`.

## Data

This project uses [WeatherBench2](https://weatherbench2.readthedocs.io/) (ERA5 reanalysis preprocessed for ML), specifically geopotential at 500 hPa (Z500) and 2m temperature (T2m). The data is downsampled to a 32×64 grid at 6-hour intervals.

- **Train:** 1979–2015 · **Validation:** 2016–2017 · **Test:** 2018–2019
- Every 4th timestep is kept (25% of data) to fit training in memory.

## Usage

All scripts live in `ZaneFileAdditions/`. A typical run goes: prepare data → train → evaluate.

```bash
# 1. Preprocess WeatherBench2 data into JAX-ready arrays
python ZaneFileAdditions/prepare_data_for_jax.py

# 2. Train the simplified model (autoregressive, 4-step targets)
python ZaneFileAdditions/train.py

# 3. Evaluate against the persistence baseline
python ZaneFileAdditions/eval.py
```

> Check each script's arguments/paths before running, some expect the WeatherBench2 files to be downloaded locally first (see `data_loading.py` and `explore_data.py`).

## Future work

Before attempting the originally planned autoregressive-training extension, the priority is getting the baseline to actually beat persistence:

1. Longer training (50–100 epochs) with proper learning-rate scheduling and no data subsampling.
2. Higher-resolution mesh (3–4 subdivisions → 642–2,562 nodes) and a true multi-mesh architecture.
3. Full spatial resolution (no downsampling from 5.625°).
4. Multi-step autoregressive training once the single-step baseline is competitive.
5. Additional input variables (humidity, wind components).

## References

1. Lam, R. et al. *Learning skillful medium-range global weather forecasting.* Science 382(6677), 2023.
2. Bi, K. et al. *Accurate medium-range global weather forecasting with 3D neural networks.* Nature 619(7970), 2023.
3. Pathak, J. et al. *FourCastNet: A global data-driven high-resolution weather model using adaptive Fourier neural operators.* arXiv:2202.11214, 2022.
4. Keisler, R. *Forecasting global weather with graph neural networks.* arXiv:2202.07575, 2022.
5. Rasp, S. et al. *WeatherBench: a benchmark data set for data-driven weather forecasting.* JAMES 12(11), 2020.

## Citation

If you reference this work:

```bibtex
@misc{bjornerud_graphcast_simplified,
  author = {Bjornerud, Zane},
  title  = {GraphCast Simplified: Weather Forecasting with Graph Neural Networks},
  year   = {2025},
  howpublished = {\url{https://github.com/Zane-Bjornerud/graphcast-simplified}}
}
```
