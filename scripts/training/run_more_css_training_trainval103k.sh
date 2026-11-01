#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# MoRE-CCS training on trainval (103k), 25 epochs.
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
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HYDRA_FULL_ERROR=1
# Cap per-process math-library threads. With 4 DDP ranks x many ray workers on a
# many-core node, unset OMP/MKL thread pools spawn a huge number of threads each
# and blow up the total thread count, causing fork() to fail with EAGAIN.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

AGENT=more_ccs
EXPERIMENT=more_ccs/more_ccs_trainval103k_25epochs

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py" \
    agent=$AGENT \
    experiment_name=$EXPERIMENT \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.max_epochs=25 \
    trainer.params.limit_val_batches=0 \
    trainer.params.num_sanity_val_steps=0 \
    trainer.params.check_val_every_n_epoch=1000000 \
    skip_val=true \
    dataloader.params.prefetch_factor=1 \
    dataloader.params.batch_size=16 \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.num_gpus=4 \
    agent.progress_bar=false \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.loss.prev_weight=0.0 \
    agent.config.long_trajectory_additional_poses=2 \
    agent.scheduler_args.dataset_size=103288 \
    seed=2
