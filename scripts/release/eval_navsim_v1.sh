#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${CHECKPOINT:?Set CHECKPOINT to a trained CoFiDrive .ckpt file}"
: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the NAVSIM data directory}"
: "${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT to the nuPlan maps directory}"
export NAVSIM_DEVKIT_ROOT="$ROOT"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$ROOT/exp}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"
python navsim/planning/script/run_pdm_score_multi_gpu.py \
  train_test_split=navtest \
  agent=more_ccs \
  agent.checkpoint_path="$CHECKPOINT" \
  experiment_name="${EXPERIMENT_NAME:-cofidrive/eval_navtest}" \
  agent.config.proposal_num=64 \
  agent.config.traj_expansion.inference_enabled=false \
  agent.num_gpus="${NUM_GPUS:-1}" \
  "$@"
