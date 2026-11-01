# Installation and data preparation

[README](../README.md) · [Training and evaluation](reproduction.md)

## Installation

Use Linux, Python 3.9 and an NVIDIA GPU. The original environment uses PyTorch 2.1.
Install the PyTorch build appropriate for your CUDA environment before the remaining
dependencies. See `requirements.txt` and `environment.yml` for package information.

```bash
git clone https://github.com/Rtwotwo/DriveCoFi.git
cd DriveCoFi
conda create -n cofidrive python=3.9 -y
conda activate cofidrive

# Install a compatible CUDA-enabled PyTorch build first.
python -m pip install -r requirements.txt
python -m pip install -e ./nuplan-devkit
python -m pip install -e .
```

## Data and weights

Prepare the NAVSIM images and logs together with the nuPlan maps using the
[dataset installation guide](install.md) and [split definitions](splits.md).
Download the [DINOv2 ViT-S/14 register weights](https://huggingface.co/timm/vit_small_patch14_reg4_dinov2.lvd142m).
Place `model.safetensors` at
`weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors`, or set its path in
[more_ccs.yaml](../navsim/planning/script/config/common/agent/more_ccs.yaml).
The pretrained backbone is separate from a trained CoFiDrive checkpoint.

Run these commands from the repository root, replacing the data paths:

```bash
export OPENSCENE_DATA_ROOT=/path/to/navsim/dataset
export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_DEVKIT_ROOT="$PWD"
export NAVSIM_EXP_ROOT=/path/to/experiments
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
```

## Evaluation metric cache

Generate the navtest cache before NAVSIM-v1 evaluation:

```bash
python navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  cache.cache_path="$NAVSIM_EXP_ROOT/navtest_metric_cache"
```

The evaluation configuration defaults to a directory named `metric_cache`.
Pass `metric_cache_path="$NAVSIM_EXP_ROOT/navtest_metric_cache"` when using the
cache generated above, as shown in the [evaluation instructions](reproduction.md#evaluate-on-navtest).
NavHard requires its own split, synthetic continuations and matching metric cache.

## Repository layout

```text
navsim/agents/more_ccs/   Full model: TC, CCS and CSA
navsim/agents/more_dual/  Dual-memory variant
navsim/agents/more_tc/    TC variant
navsim/planning/script/  Training, evaluation and Hydra configurations
scripts/release/         Portable training and evaluation launchers
docs/                    Setup, reproduction and experiment documentation
viz/                     Scene and attention visualization tools
```

The original launchers under `scripts/` may contain experiment-specific paths.
Start with `scripts/release/` and supply your own checkpoint and dataset locations.
