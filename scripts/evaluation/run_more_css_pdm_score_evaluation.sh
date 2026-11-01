#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# MoRE-CCS PDM-score evaluation on navtest.
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
export LD_LIBRARY_PATH=""

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"

cd "$NAVSIM_DEVKIT_ROOT"

AGENT="more_ccs"
EXPERIMENT="${EXPERIMENT:-more_ccs/eval_more_ccs_trainval103k_25epochs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$NAVSIM_EXP_ROOT/more_ccs/more_ccs_trainval103k_25epochs/checkpoints}"

CHECKPOINT_NAMES=(
    "epoch=23-step=31896.ckpt"
    "epoch=24-step=33225.ckpt"
)

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Checkpoint directory not found: $CHECKPOINT_DIR" >&2
    exit 1
fi

for checkpoint_name in "${CHECKPOINT_NAMES[@]}"; do
    if [[ ! -f "$CHECKPOINT_DIR/$checkpoint_name" ]]; then
        echo "Checkpoint file not found: $CHECKPOINT_DIR/$checkpoint_name" >&2
        exit 1
    fi
done

has_completed_score_csv() {
    local output_dir="$1"
    local csv_file
    [[ -d "$output_dir" ]] || return 1
    while IFS= read -r csv_file; do
        if [[ ",$(head -n 1 "$csv_file")," == *",score,"* ]] || [[ ",$(head -n 1 "$csv_file")," == *",scorer_selected_real_pdms,"* ]]; then
            return 0
        fi
    done < <(find "$output_dir" -maxdepth 1 -type f -name '*.csv' | sort)
    return 1
}

checkpoint_tag_from_name() {
    local checkpoint_name="$1"
    local checkpoint_stem
    checkpoint_stem="${checkpoint_name%.ckpt}"
    echo "${checkpoint_stem//[^[:alnum:]]/_}"
}

echo "Agent: $AGENT"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Experiment: $EXPERIMENT"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Checkpoint files requested: ${#CHECKPOINT_NAMES[@]}"
echo "Checkpoint evaluation order: ${CHECKPOINT_NAMES[*]}"
echo "Inference trajectory expansion: disabled; score head receives only the original 64 proposals."

PENDING_CHECKPOINT_NAMES=()
for checkpoint_name in "${CHECKPOINT_NAMES[@]}"; do
    checkpoint_tag="$(checkpoint_tag_from_name "$checkpoint_name")"
    output_dir="$NAVSIM_EXP_ROOT/ke/$EXPERIMENT/$checkpoint_tag"
    if has_completed_score_csv "$output_dir"; then
        echo "Skipping completed $checkpoint_name"
        continue
    fi
    PENDING_CHECKPOINT_NAMES+=("$checkpoint_name")
    echo "  $checkpoint_name -> $output_dir"
done

echo "Checkpoints pending: ${#PENDING_CHECKPOINT_NAMES[@]}"
if [[ ${#PENDING_CHECKPOINT_NAMES[@]} -eq 0 ]]; then
    echo "All requested checkpoint files have already been evaluated."
    exit 0
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "Dry run completed successfully."
    exit 0
fi

for checkpoint_name in "${PENDING_CHECKPOINT_NAMES[@]}"; do
    checkpoint_path="$CHECKPOINT_DIR/$checkpoint_name"
    checkpoint_tag="$(checkpoint_tag_from_name "$checkpoint_name")"
    output_dir="$NAVSIM_EXP_ROOT/ke/$EXPERIMENT/$checkpoint_tag"

    mkdir -p "$output_dir"
    echo "Evaluating $checkpoint_name on GPUs $CUDA_VISIBLE_DEVICES"
    echo "Output directory: $output_dir"

    python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_multi_gpu.py" \
        train_test_split=navtest \
        agent="$AGENT" \
        agent.checkpoint_path=\"$checkpoint_path\" \
        experiment_name="$EXPERIMENT" \
        experiment_uid="$checkpoint_tag" \
        output_dir="$output_dir" \
        agent.config.proposal_num=64 \
        agent.config.traj_expansion.inference_enabled=false \
        agent.config.refiner_ls_values=0.0 \
        agent.config.image_backbone.focus_front_cam=false \
        agent.config.one_token_per_traj=true \
        agent.config.refiner_num_heads=1 \
        agent.config.tf_d_model=256 \
        agent.config.tf_d_ffn=1024 \
        agent.config.area_pred=false \
        agent.config.agent_pred=false \
        agent.config.ref_num=4 \
        agent.config.noc=1 \
        agent.config.dac=1 \
        agent.config.ddc=0.0 \
        agent.config.ttc=5 \
        agent.config.ep=5 \
        agent.config.comfort=2 \
        agent.num_gpus=4 \
        agent.progress_bar=false
done
