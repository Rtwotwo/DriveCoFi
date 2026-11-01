#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# MoRE-CCS training on navtrain (train 85k), 10 epochs.
# All paths are resolved relative to this script so the repository stays
# portable and free of machine-specific absolute paths.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVSIM_DEVKIT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"   # MoRE repository root
WORKSPACE_ROOT="$(cd "$NAVSIM_DEVKIT_ROOT/.." && pwd)"  # parent workspace (holds dataset/, exp/)

# Activate the conda environment by name (override with CONDA_ENV=<name|path>).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-drivoR}"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$WORKSPACE_ROOT/dataset/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$WORKSPACE_ROOT/dataset}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$NAVSIM_DEVKIT_ROOT/exp}"
export NAVSIM_DEVKIT_ROOT
# Reset PYTHONPATH (do not inherit): some servers inject unrelated paths that
# shadow python modules used by nuplan/navsim.
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT"
# Reset LD_LIBRARY_PATH so torch uses its bundled CUDA libraries through RPATH.
export LD_LIBRARY_PATH=""
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$NAVSIM_DEVKIT_ROOT"
python navsim/planning/script/run_training.py \
    agent=more_ccs \
    experiment_name=more_ccs/more_ccs_train85k_10epochs \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.max_epochs=10 \
    trainer.params.check_val_every_n_epoch=3 \
    dataloader.params.num_workers=8 \
    dataloader.params.prefetch_factor=1 \
    +dataloader.params.persistent_workers=true \
    dataloader.params.batch_size=16 \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.num_gpus=4 \
    agent.pdm_workers_per_rank=16 \
    agent.pdm_proposal_chunk_size=0 \
    agent.progress_bar=false \
    seed=2 \
    "$@"
