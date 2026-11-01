# Training and evaluation guide

[Installation](setup.md) · [Method and configuration](method.md) · [Reported results](experiments.md)

## Prepare the environment

Install the dependencies and the vendored nuPlan package as described in the installation guide.
Download the NAVSIM images/logs, nuPlan maps and DINOv2 register weights separately.
Point the pretrained model setting in
[more_ccs.yaml](../navsim/planning/script/config/common/agent/more_ccs.yaml) to the local
`model.safetensors` file.

```bash
export OPENSCENE_DATA_ROOT=/path/to/navsim/dataset
export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps
export NAVSIM_EXP_ROOT=/path/to/experiments
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

Prepare evaluation metric caches with the matching benchmark configuration.
[docs/install.md](install.md), [docs/splits.md](splits.md) and the
[metric-caching launcher](../scripts/evaluation/run_metric_caching.sh) describe the
framework setup. Review paths in original launchers before using them.

## Train85k: a 10-epoch run

```bash
SEED=2 NUM_GPUS=4 BATCH_SIZE=16 \
  EXPERIMENT_NAME=cofidrive/train85k_seed2 \
  bash scripts/release/train_navsim_v1.sh
```

The wrapper uses `agent=more_ccs`, `train_test_split=navtrain`, AdamW with base
learning rate `2e-4`, and 10 epochs. The paper's training settings use four GPUs
and batch size 16 per GPU. The launcher defaults to one GPU for a smaller local
run; set the GPU count and visible devices explicitly for your intended setting.

Additional arguments are forwarded to Hydra. Save each run's resolved configuration
and checkpoint together with its seed, data split and evaluator version. Use separate
experiment names for separate runs.

## Evaluate on navtest

```bash
CHECKPOINT=/path/to/checkpoint.ckpt NUM_GPUS=1 \
  EXPERIMENT_NAME=cofidrive/navtest_seed2 \
  bash scripts/release/eval_navsim_v1.sh \
  metric_cache_path="$NAVSIM_EXP_ROOT/navtest_metric_cache"
```

The evaluation wrapper fixes `proposal_num=64` and
`traj_expansion.inference_enabled=false`. Use the checkpoint and caches associated
with the desired training and evaluation splits. CSA is a training procedure;
enabling test-time expansion changes the reported inference protocol.

## Other settings

- **Trainval103k:** the paper uses an additional resampled long-trajectory regression
  target. The default release configuration has `long_trajectory_additional_poses=-1`.
  Reproducing this setting requires the actual long-target configuration associated
  with its checkpoint; changing only the split does not reproduce it.
- **NavHard:** use the two-stage evaluator
  [run_pdm_score_gpu_v2.py](../navsim/planning/script/run_pdm_score_gpu_v2.py) with its
  matching scene split and caches. A NAVSIM-v1 PDMS evaluation is not a NavHard EPDMS
  evaluation.
- **Repeated training:** only the train85k main result in the paper is a three-run
  mean. Aggregate per-run scores for the same evaluation set and preserve the
  individual results when reporting a new mean or standard deviation.

## Configuration checks

| Purpose | Setting |
| --- | --- |
| Full method | TC enabled; patch memory enabled; CSA enabled during training |
| Camera registers | 16 per view, four views |
| Active experts | Six, scene-level soft mixture |
| CCS patch grid | 12 × 21 per view |
| CSA auxiliary list | 16 seeds × 12 slots + 64 originals |
| Inference | 64 original proposals; expansion disabled |

If a checkpoint fails to load, first compare its hidden width, register count,
expert count and enabled branches with the resolved agent configuration. The
internal `more_ccs` agent key is retained for these interfaces.
