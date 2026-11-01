#!/usr/bin/env bash
set -euo pipefail

# Run from an activated Python environment. Required paths are supplied by the user.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the NAVSIM data directory}"
: "${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT to the nuPlan maps directory}"
export NAVSIM_DEVKIT_ROOT="$ROOT"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$ROOT/exp}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"
python navsim/planning/script/run_training.py \
  agent=more_ccs \
  experiment_name="${EXPERIMENT_NAME:-cofidrive/train85k_10ep}" \
  train_test_split=navtrain \
  cache_path="${CACHE_PATH:-null}" \
  use_cache_without_dataset=false \
  trainer.params.max_epochs=10 \
  dataloader.params.batch_size="${BATCH_SIZE:-16}" \
  agent.lr_args.name=AdamW \
  agent.lr_args.base_lr=0.0002 \
  agent.num_gpus="${NUM_GPUS:-1}" \
  seed="${SEED:-2}" \
  "$@"
