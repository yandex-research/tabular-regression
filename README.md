# Revisiting Tabular Deep Regression

Code, search spaces, selected configurations, and experimental results for
**Revisiting Tabular Deep Regression: Objectives and the Limits of Global Early Stopping**.

We compare 18 regression objectives on 41 BeyondArena datasets and study selected
objectives with MLP, TabM, TabPack, and RealMLP. We also examine
uncertainty-dependent early stopping and Adaptive Quantile Regression (AdaQR),
which learns input-dependent smoothing jointly with quantile predictions.

## Repository layout

- `bin/` — tuning, evaluation, ensembling, and objective-specific entrypoints.
- `lib/` — shared data processing and experiment utilities.
- `exp/` — published configurations and machine-readable reports.
- `vendor/` — third-party sources and research adaptations; see [VENDOR.md](VENDOR.md).

Datasets, checkpoints, and prediction arrays are stored separately.

### Experiments

| Path | Experiments |
| --- | --- |
| `exp/mlp/muon/` | The full objective comparison and AdaQR experiments. |
| `exp/mlp/adamw/` | MSE, HL-Gaussian, and quantile loss with AdamW. |
| `exp/tabm/` | TabM-mini with PLE-Q feature embeddings. |
| `exp/tabpack/` | TabPack with cosine feature embeddings and greedy online ensembles. |
| `exp/realmlp/` | RealMLP with the selected regression objectives. |
| `exp/cohorts/` | Quantile and AdaQR early stopping, with 5 or 10 cohorts. |
| `exp/catboost-uncertainty/` | CatBoost uncertainty estimation. |

Objective directories use the paper's names: `mse`, `mae`, `huber`,
`hard-binning`, `soft-binning`, `joint-end-to-end`, `softmax-mse`,
`hl-gaussian`, `quantile`, `decoder-normalized` (Decoder[N]),
`decoder-unnormalized` (Decoder[U]), `nll`, `beta-nll`, `mdn`,
`gar-mae`, `gar-mse`, `ordinal-entropy`, `rot-reg`, and `adaqr`.
`adaqr-scale-reg` denotes the scale-regularization variant.

`exp/realmlp/adaqr-global-sigma/` contains the fixed global-smoothing ablation:
`loss_sigma` is selected by tuning and does not depend on the input or change
during training. This ablation covers all 28 non-Tiny datasets in the article:
12 Small, 10 Medium, and 6 Large datasets. Each retains tuning configurations
and reports, evaluation results for all 15 seeds, and five-model ensemble results.
Tiny datasets are omitted from this ablation. The tuning reports retain their
actual `n_completed_trials` counts.

Most methods follow this layout:

```text
exp/<backbone>/<objective>/beyond/<dataset>/
  tuning/
    config.toml           # Search space and tuning budget.
    report.json           # Selected configuration and its results.
  evaluation/
    config.toml           # Selected hyperparameters and evaluation seeds.
    report.json           # Results for individual seeds.
  ensemble-5/
    config.toml
    report.json           # Five-model ensemble results, where available.
```

MLP additionally separates the `muon` and `adamw` optimizers. The usual evaluation
budget is 15 seeds (`0` through `14`); each saved configuration specifies its budget.
Cooking Time, Maps Router ETA, and Delivery ETA are excluded from the article
comparison. Dataset identifiers retain their original spelling, including
`beyond-5g_energy_consumption-0`.

The four main TabPack comparisons each retain 41 `main/config.toml` files with
their hyperparameter search spaces under `sampler.space`. The scale-regularization
ablation mostly stores the explicit sampled pack members under `configs`, rather
than a resampling search specification. The selected online ensemble is evaluated under
`eval-online-ensembles/greedy/evaluation/`; `evaluation/report.json` exposes these
results in the same format as the other backbones. Cohort experiments reuse the
selected global hyperparameters and have no separate tuning stage.

## Environment

The experiments use Python 3.12.9 and PyTorch 2.7.1. Dependencies and their
versions are recorded in `pyproject.toml` and `uv.lock`.

From the repository root, install the environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra pytabkit
```

The modified RealMLP implementation is included in `vendor/pytabkit/`; no
additional checkout is needed. Run all commands from the repository root and
select the accelerator with `CUDA_VISIBLE_DEVICES`.

## Data

We use BeyondArena Lite (outer split `0`), with the three dataset exclusions
listed above. We keep the official test partitions unchanged and carve validation
sets from the official training partitions. For i.i.d. tasks, we hold out **20% of
the original training rows** for validation and use the remaining 80% for training
(`train_test_split(test_size=0.2, random_state=0)`). Grouped and temporal tasks use
the official task-aware holdout procedure, preserving group boundaries and
chronological order; their exact validation fractions can differ from 20%.

Download the file named [`data.zip` from Figshare](https://figshare.com/s/1ef375b0f5e628f35058), then extract it in this repository's root:

```bash
unzip /path/to/data.zip
```

It creates `data/`, for example `data/beyond-miami_housing-0`. The archive already
contains these prepared train/validation/test partitions and the files required
by the experiment loader; no additional splitting is needed.

Cohort evaluations additionally consume CatBoost uncertainty arrays at
`data/cohort-uncertainty/<dataset>.npz`, containing `val` and `test` arrays in the
same row order as the dataset splits.

## Reproducing experiments

Published reports are retained for inspection. To run a new experiment, copy its
configuration into a fresh directory under `local/`; copying a completed report
would cause `--resume` to reuse the recorded result.

### Selected hyperparameters

For example, evaluate RealMLP AdaQR on Miami Housing and build five-model ensembles:

```bash
mkdir -p local/reproduce/realmlp-adaqr/evaluation
cp exp/realmlp/adaqr/beyond/beyond-miami_housing-0/evaluation/config.toml \
  local/reproduce/realmlp-adaqr/evaluation/config.toml

CUDA_VISIBLE_DEVICES=0 uv run --extra pytabkit bin/go.py \
  local/reproduce/realmlp-adaqr/evaluation \
  --n-seeds 15 --resume --ensemble-sizes 5
```

The runner writes per-seed results under `evaluation/` and ensemble results under
`ensemble-5/`. When starting from `evaluation`, the seed count comes from its
`config.toml`; the command-line value does not overwrite it.

### Hyperparameter tuning

Start from a fresh tuning configuration to repeat the search and then evaluate
the selected model:

```bash
mkdir -p local/reproduce/mlp-adaqr/tuning
cp exp/mlp/muon/adaqr/beyond/beyond-miami_housing-0/tuning/config.toml \
  local/reproduce/mlp-adaqr/tuning/config.toml

CUDA_VISIBLE_DEVICES=0 uv run bin/go.py \
  local/reproduce/mlp-adaqr/tuning \
  --n-seeds 15 --resume --ensemble-sizes 5
```

### TabPack

TabPack combines hyperparameter sampling with training:

```bash
mkdir -p local/reproduce/tabpack-adaqr/main
cp exp/tabpack/adaqr/beyond/beyond-miami_housing-0/main/config.toml \
  local/reproduce/tabpack-adaqr/main/config.toml

CUDA_VISIBLE_DEVICES=0 uv run vendor/tabpack/run_tabpack_experiment.py \
  local/reproduce/tabpack-adaqr \
  --eval-online-ensembles greedy --eval-online-ensembles-n-seeds 15
```

This repeats the search, selects the greedy online ensemble, retrains its members
across 15 seeds, and exports a compact evaluation report. Some archived TabPack
AdaQR reports were recovered from text summaries; these reports record their
provenance and omit rounded total durations.

### Cohort early stopping

Copy a cohort `evaluation/config.toml` to a fresh evaluation directory and run
`bin/go.py` with `--resume`. The uncertainty and random controls use matched
global hyperparameters. The reported cohort metrics refer to predictions stitched
from the validation-selected checkpoint for each cohort, and are distinct from
the global checkpoint's metrics. The random control resamples cohorts for each seed.

Reports can be read directly as JSON: individual evaluations are in
`experiments[*].report.metrics`, and tuning reports contain `best.config` and
`best.report`. Numerical reproducibility also depends on the data, hardware, and
the pinned software environment.
