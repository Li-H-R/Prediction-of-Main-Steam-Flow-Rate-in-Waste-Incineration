# Prediction of Main Steam Flow Rate in Waste Incineration

Experiment 1 — **multi-model × multi-input-configuration comparison** for main-steam-flow
prediction in a waste-incineration plant, evaluated at two sampling rates.

This repository contains the core training / evaluation code for the **minute-level**
experiment of a larger private research project. The comparison is also run on
**second-level (5 s)** data; its configuration is documented in
[Second-level experiment](#second-level-5-s-experiment).

Both levels share the same **11 deep models** and architecture hyperparameters.

> **Note:** the raw plant data and the trained
> model checkpoints are **not** included. The code also expects one sibling module
> from the parent project (`extract_column.py`) — see [Dependencies](#dependencies).

## Task

Predict the future `pred_len` steps of the **main steam flow** (主蒸汽流量, channel 0) from a
window of `seq_len` past steps of `n_vars` process variables.

| Setting | Value |
|---|---|
| Target | Main steam flow (channel 0) |
| History window `seq_len` | 60 (minutes) |
| Forecast horizon `pred_len` | 5 (minutes) |
| Data | minute-level |
| Train / Val / Test split | 70% / 15% / 15% (chronological, no shuffle on val/test) |
| Normalization | per-variable z-score, statistics from the training set only |

## Input-variable configurations

| Key | Variables | `n_vars` |
|---|---|---|
| `univariate` | main steam flow only | 1 |
| `univariate_lag` | main steam flow + `K = 4` lag channels (`step = 5`) | 4 |
| `paper` | main steam flow + 6 state variables | 7 |
| `cluster_A` | main steam flow + retained features (from Exp. 2 `retained_features.csv`) | variable |
| `cluster_A_state` | `cluster_A` + 6 state variables (de-duplicated) | variable |
| `paper_plus_primary` *(optional, not default)* | `paper` + 6 primary-air variables | 13 |

The 6 **state variables**: drum pressure, evaporator-I inlet flue-gas temp, superheated-steam
header pressure, medium-temp superheater inlet flue-gas temp, first-flue front flue-gas temp
(T11), economizer-outlet flue-gas temp.

The 6 **primary-air variables**: primary-air fan motor speed / current (combustion section),
primary-air outlet pressure, and calculated primary-air flow for the drying / combustion /
burnout sections.

## Models

All models share a unified interface: `forward(x) -> (B, pred_len, n_vars)` for the
channel-independent class, or `-> (B, pred_len)` for the single-target class. Evaluation
always reports the main steam flow (channel 0).

| Model | Family | Output mode | Trained by default |
|---|---|---|---|
| `PatchTST` | channel-independent patch transformer | multi | ✅ |
| `CTPatchTST` | PatchTST + 1-layer cross-variable channel attention | multi | ✅ |
| `LSTM_mv` | multivariate-input single-target LSTM | single | ✅ |
| `LSTM_ci` | channel-independent LSTM | multi | ❌ (excluded) |
| `iTransformer` | inverted transformer (variable as token) | single | ✅ |
| `Transformer` | classic transformer (temporal attention + mean pool) | single | ✅ |
| `Crossformer` | DSW embedding + two-stage attention | multi | ✅ |
| `Crossformer_full` | full Crossformer (`cross_models` original) | multi | ✅ |
| `RTF` | Scaleformer + Autoformer (AS core) | single | ✅ |
| `MLP` | fully-connected (last `n_steps` flattened) | single | ✅ |
| `LSTM_iTransformer` | shared LSTM + variable-dim attention | single | ❌ (excluded) |

- `multi` = predicts every channel, channel 0 taken as the target.
- `single` = outputs the main steam flow directly.
- `LSTM_ci` and `LSTM_iTransformer` are excluded from the default training list
  (`EXCLUDE_MODELS`); they remain defined in `models.py`.

## Training configuration

Common to all model × config × seed runs:

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | `5e-4` |
| Epochs (max) | 100 |
| Early stopping | patience = 20 (on validation loss) |
| Batch size (train) | 256 |
| Gradient clipping | `max_norm = 1.0` |
| Mixed precision | AMP on CUDA; **RTF forced to fp32** (Autoformer FFT constraint) |
| Loss (`multi` mode) | channel-weighted MSE, weight 1.0 (target) / 0.2 (auxiliary channels) |
| Loss (`single` mode) | MSE on channel 0 |
| Seeds | 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 |

## Model architecture hyperparameters

Defined in `models.py` as `ARCH_DEFAULTS`:

| Model | Hyperparameters |
|---|---|
| `PatchTST` | patch_len=6, stride=6, d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1 |
| `CTPatchTST` | same as PatchTST + channel_heads=1 |
| `LSTM_mv` | hidden_size=64, num_layers=2, dropout=0.1 |
| `LSTM_ci` | hidden_size=64, num_layers=2, dropout=0.1 |
| `iTransformer` | d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1 |
| `Transformer` | d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1 |
| `Crossformer` | seg_len=6, d_model=64, n_heads=4, e_layers=2, d_ff=256, dropout=0.1 |
| `Crossformer_full` | seg_len=6, win_size=2, factor=10, d_model=64, n_heads=4, e_layers=2, d_ff=256, dropout=0.1 |
| `RTF` | d_model=64, n_heads=8, d_ff=256, encoder_layers=2, decoder_layers=1, dropout=0.1, moving_avg_kernel=25, factor=1, scales=(1, 2) |
| `MLP` | n_steps=10, hidden_dims=(256, 128), dropout=0.1 |
| `LSTM_iTransformer` | hidden_size=64, lstm_layers=2, d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1 |

## Evaluation

`test.py` reports **MAE / RMSE / R² / TCR** (trend consistency rate), computed on
z-score standardized values and averaged across seeds per `(model, config)`.

- `RTF` is evaluated with **Adaptive Time Matching**: the forecast period is split into
  segments, and per segment it picks the lower validation MAE between the offline AS model
  and an online Ridge model fitted on the preceding history (`RTF_HISTORY=180`,
  `RTF_FIT_SAMPLES=90`, `RTF_SEGMENT=180`, `RTF_RIDGE_ALPHA=1.0`).

## Second-level (5 s) experiment

The same multi-model × multi-configuration comparison is replicated on **second-level data**
(`filtered_data_clean.npy`, ~5 s sampling, 17 days, 99 process variables). The second-level
code lives in a sibling directory of the parent project and is **not** included in this
repository; this section documents its configuration.

| Setting | Value |
|---|---|
| Target | Main steam flow (channel 0) |
| History window `seq_len` | 120 (~10 min at 5 s/point) |
| Forecast horizon `pred_len` | 50 (~4.2 min) |
| Data | `filtered_data_clean.npy`, ~5 s sampling, 17 days, 99 variables |
| Train / Val / Test split | 70% / 15% / 15% (chronological) |
| Normalization | per-variable z-score, statistics from the training set only |

### Input-variable configurations (seconds)

| Key | Variables | `n_vars` |
|---|---|---|
| `univariate` | main steam flow only | 1 |
| `cluster_A` | main steam flow + low-correlation variables (`LOW_CORRELATION`) | variable |
| `paper` | main steam flow + leading variables (`POTENTIAL_LEADING`) + minute-level state variables present in the seconds data | variable |
| `cluster_A_paper` | `cluster_A` + `paper` (de-duplicated) | variable |

### Training configuration (seconds)

Identical to the minute-level, except:

| Hyperparameter | Value |
|---|---|
| Batch size (train) | 1024 |
| Seeds | 10, 20, 30, 42, 50, 60, 70, 80, 90, 100 |

The model list (`MODEL_NAMES`), architecture hyperparameters (`ARCH_DEFAULTS`), output modes,
channel-weighted loss, and the RTF fp32 / Adaptive-Time-Matching settings are **identical** to
the minute-level experiment (see the sections above).

## Repository layout

| File / dir | Purpose |
|---|---|
| `models.py` | all model definitions + registry (`MODEL_NAMES`, `ARCH_DEFAULTS`) |
| `train.py` | unified training entry (model × config × seed) |
| `test.py` | evaluation entry (MAE / RMSE / R² / TCR) |
| `cross_models/` | full `Crossformer` implementation (depends on `einops`) |

## Usage

```bash
# Train (default: 9 models × 5 configs × 10 seeds)
python train.py

# Train a subset
python train.py --models PatchTST,LSTM_mv --configs univariate,paper --seeds 10,20

# Shape self-check without training
python train.py --smoke

# Evaluate all checkpoints
python test.py
```

## Dependencies

- Python, **PyTorch**, NumPy, scikit-learn (Ridge, RTF only), matplotlib
- `einops` (for the full `Crossformer`)
- Sibling module from the parent project (not in this repo): `extract_column.py`
- Data files (not in this repo): `data_array.npy`, `var_names.npy`
- For the `cluster_A` / `cluster_A_state` configs: `实验2-…/results/retained_features.csv`
