# diffsmile

This repository contains the code and thesis material for diffusion-based
modeling of implied volatility surfaces. The project focuses on three related
tasks:

- forecasting next-day implied volatility surfaces,
- inpainting partially observed surfaces,
- conditional scenario generation under market shocks.

The runnable code lives in `code/`, while the manuscript lives in `thesis/`.

## Repository layout

- `code/` – training, evaluation, notebooks, and helper modules
- `code/diffsmile/` – Python package with configs, model code, evaluation, and helpers
- `code/diffsmile/notebooks/setup/` – data-preparation notebooks
- `code/train.sh` – Slurm runner for training the GNOT diffusion model
- `thesis/` – LaTeX thesis sources

## Setup

### 1) Install `uv`

Use one of the official installation methods from Astral:
<https://docs.astral.sh/uv/getting-started/installation/>

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Homebrew
brew install uv
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2) Sync dependencies

From the `code/` directory:

```bash
cd code
uv sync
```

## Data preparation

The model is trained on processed SPX implied-volatility surfaces, not directly
on raw vendor extracts. The raw options data should come from OptionMetrics and
must be joined with forward prices before running the training pipeline.

This expectation is stated directly in `code/diffsmile/config.py`:

> join forward price from "OptionMetrics - Forward Price" as the dataset does
> not contain it anymore!

The main dataset paths are defined in `code/diffsmile/config.py` under
`DatasetConfig`. In particular, you will likely need to adjust these paths on
your machine or cluster:

- `CORE_DATA_PATH`
- `option_metrics_dataset`
- `output_path_surfaces`
- `output_conditioning_scalars`
- `merged_surfaces_path`
- `merged_conditioning_scalars_path`

By default, these point to cluster-specific locations under:

```text
/cluster/project/math/andtran/develop/masters_thesis/code/data/
```

### Recommended preprocessing order

Run the two setup notebooks in this order:

1. `code/diffsmile/notebooks/setup/1_create_spx_dataset.ipynb`
2. `code/diffsmile/notebooks/setup/2_conditioning_scalars.ipynb`

#### Notebook 1: create the surface dataset

`1_create_spx_dataset.ipynb`:

- reads the processed OptionMetrics parquet file,
- expects a `forward_price` column to already be available,
- filters for liquid OTM quotes,
- constructs the 32 × 24 implied-volatility surface grid,
- saves the surface payload with `torch.save(...)`.

The notebook reads from:

- `dataset_config.option_metrics_dataset`

and saves to:

- `dataset_config.output_path_surfaces`

#### Notebook 2: create the conditioning scalars

`2_conditioning_scalars.ipynb`:

- loads the merged surface dates,
- downloads SPX and VIX close data,
- computes next-day return, EWMA return features, EWMA squared-return features,
  and VIX returns,
- saves the conditioning tensor.

The notebook reads from:

- `dataset_config.merged_surfaces_path`

and saves to:

- `dataset_config.output_conditioning_scalars`

## Training a model

The main training entrypoint is the Lightning-based script:

- `code/diffsmile/gnot_lightning.py`

The default Slurm launcher is:

- `code/train.sh`

From the `code/` directory, submit training with:

```bash
sbatch train.sh
```

Optional monitoring:

```bash
squeue -u $USER
```

### What the runner does

`code/train.sh` changes into the cluster code directory and runs:

```bash
uv run python diffsmile/gnot_lightning.py \
    --embed_dim=256 \
    --lr=0.001 \
    --n_heads=4 \
    --n_layers=6 \
    --n_experts=4 \
    --mlp_layers=4 \
    --batch_size=32 \
    --epochs=500 \
    --calendar_loss_weight=1e-5 \
    --butterfly_loss_weight=1e-5 \
    --smoothness_loss_weight=0.005 \
    --smoothness_weight_mode=inverse_sqrt
```

If you want to run the trainer directly, you can do so from `code/` with:

```bash
uv run python diffsmile/gnot_lightning.py
```

## Save paths and configuration caveats

Several paths are currently cluster-specific or hard-coded, so they may need to
be changed before the workflow runs on another machine.

The most important ones are:

- dataset paths in `code/diffsmile/config.py`
- `trained_model_path` in `code/diffsmile/config.py`
- `DATA_DIR` in `code/diffsmile/gnot_lightning.py`
- the cluster working directory in `code/train.sh`

The Lightning training script currently expects these files in its data directory:

- `spx_iv_dataset_full_365.pt`
- `conditioning_vectors_w_ret.pt`

## Evaluation

Useful follow-up notebooks and scripts include:

- `code/diffsmile/notebooks/eval_gnot_lightning.ipynb`
- `code/run_eval_aggregate_job.py`
- `code/eval.sh`

These are useful once a checkpointed model is available.

## Model release

The pretrained checkpoint is available in GitHub Releases.

Download the `.ckpt` file from the latest release and place it at the path
configured by `DatasetConfig.trained_model_path` in
`code/diffsmile/config.py`.

By default, that path is:

```text
code/trained_model.ckpt
```

If you want to store the checkpoint somewhere else, either update
`trained_model_path` in `code/diffsmile/config.py` or pass the checkpoint
explicitly to the evaluation script:

```bash
cd code
uv run python run_eval_aggregate_job.py --checkpoint-path /path/to/trained_model.ckpt
```

This lets users run evaluation with the released pretrained model without
having to retrain from scratch.

## Thesis

The thesis manuscript is in `thesis/`. If you only want to work on the paper,
you do not need the training workflow above.
