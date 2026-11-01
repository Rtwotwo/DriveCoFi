<div align="center">

# CoFiDrive

### Scene-Conditioned Registers and Coarse-to-Fine Trajectory Scoring for End-to-End Driving

[Project page](https://rtwotwo.github.io/DriveCoFi/) · [Method](#method) · [Installation](#installation) · [Training and evaluation](#training-and-evaluation)

<img src="site/assets/images/Figure1.webp" alt="CoFiDrive architecture: register refinement, dual-channel scoring and training-only control-space augmentation" width="100%">

</div>

CoFiDrive is a vision-based trajectory planner built on NAVSIM. It uses four camera views,
each with 16 learned registers, to form a fixed 64-token scene memory. The semantic
token compressor (TC) refines this memory with a scene-conditioned soft mixture of
six SwiGLU experts; all experts remain active. A Cross-Compression Scorer (CCS)
combines the scene memory with pooled spatial patch features when ranking generated
trajectories. During training, control-space augmentation (CSA) adds perturbed,
bicycle-model-rolled-out candidates to the scorer's auxiliary supervision. CSA is
disabled at inference: CCS ranks only the generator's original 64 proposals.

The camera-register bottleneck provides the token reduction. TC **does not** reduce
the number of tokens further. Coarse-to-fine refers to the visual evidence used for
scoring, **not** to a cascade that removes trajectory candidates.

## Results

| Evaluation | Training split | Epochs | Result |
| --- | --- | ---: | ---: |
| NAVSIM-v1 navtest | train85k | 10 | 92.4 PDMS |
| NAVSIM-v1 navtest | trainval103k | 10 | 92.9 PDMS |
| NAVSIM-v2 NavHard | matched 10-epoch protocol | 10 | 48.0 EPDMS |

The train85k headline result is the mean of three independently seeded training
runs. The other rows and the ablations should not be read as three-seed averages.
These are benchmark scores, not closed-loop or on-road safety measurements.

## Method

1. **TC — scene-conditioned register refinement.** A scene-level router assigns
   weights to six active experts; 64 input registers remain 64 output scene tokens.
2. **CCS — two granularities of visual evidence.** Each trajectory query attends
   to the refined scene tokens and 1,008 pooled image-patch tokens. A
   candidate-dependent gate combines the two responses. The patch-backbone
   output is detached on the CCS patch path.
3. **CSA — training-only scorer supervision.** The top 16 PDM-ranked proposals
   seed 12 control-space variants each. Acceleration and yaw-rate perturbations
   are rolled out with a bicycle model, checked for validity, and scored by the
   PDM evaluator alongside the original 64 proposals. Invalid augmentation
   slots fall back to their seeds. Neither CSA nor PDM target computation runs
   during inference.

For the figures and qualitative comparison, see the [project page](https://rtwotwo.github.io/DriveCoFi/).
The static page source is in [`site/`](site/); GitHub Pages can deploy it using
[the included workflow](.github/workflows/pages.yml).

## Repository layout

```text
navsim/             NAVSIM framework and CoFiDrive agents/configs
  agents/more_ccs/  Full TC + CCS + CSA implementation (agent=more_ccs)
  agents/more_dual/ Dual-channel variant
  agents/more_tc/   TC variant
nuplan-devkit/      Vendored nuPlan dependency, with its own license
scripts/release/    Portable train/eval wrappers for new users
scripts/           Original experiment launchers; some retain local GPU or checkpoint defaults
download/          Benchmark data download helpers
viz/               Visualization code (generated outputs omitted)
site/              Static project page and selected paper figures
```

The internal Hydra key remains `agent=more_ccs` for checkpoint compatibility;
the paper and public-facing method name are **CoFiDrive**. The original launchers
under `scripts/training/` and `scripts/evaluation/` record experiment-specific
settings, including fixed GPU IDs and checkpoint names. Use `scripts/release/`
for a fresh run.

## Installation

The code expects Linux, a CUDA-capable PyTorch installation, Python 3.9, NAVSIM
datasets and nuPlan maps. The benchmark data and pretrained model weights are not
redistributed here. Review package versions against your CUDA driver before
installing; the original environment was developed with PyTorch 2.1.

```bash
conda create -n cofidrive python=3.9 -y
conda activate cofidrive
# Install a CUDA-compatible PyTorch build for your host first.
pip install -r requirements.txt
pip install -e ./nuplan-devkit
pip install -e .
```

Download NAVSIM according to the [official instructions](https://github.com/autonomousvision/navsim)
and provide the pretrained DINOv2 `vit_small_patch14_reg4_dinov2.lvd142m`
weights from [timm/Hugging Face](https://huggingface.co/timm/vit_small_patch14_reg4_dinov2.lvd142m)
at `weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors`, or
adjust the path in [`more_ccs.yaml`](navsim/planning/script/config/common/agent/more_ccs.yaml).
Never commit the weights or the dataset to this repository.

```bash
export OPENSCENE_DATA_ROOT=/path/to/navsim/dataset
export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps
export NAVSIM_EXP_ROOT=/path/to/experiment/output
export CUDA_VISIBLE_DEVICES=0
```

The metric caches required by NAVSIM must be prepared before scoring. Refer to
[`docs/install.md`](docs/install.md), [`docs/splits.md`](docs/splits.md), and
the original [`scripts/evaluation/run_metric_caching.sh`](scripts/evaluation/run_metric_caching.sh)
for benchmark-specific paths.

## Training and evaluation

```bash
# train85k, 10 epochs; override SEED, NUM_GPUS or Hydra settings as needed
SEED=2 NUM_GPUS=1 bash scripts/release/train_navsim_v1.sh

# evaluation uses the same 64 inference proposals and disables CSA
CHECKPOINT=/path/to/checkpoint.ckpt NUM_GPUS=1 \
  bash scripts/release/eval_navsim_v1.sh
```

The wrappers forward additional arguments to the underlying Hydra entry points.
For example, `SEED=3 NUM_GPUS=4 bash scripts/release/train_navsim_v1.sh`
changes the training seed and GPU count. The wrappers have
been syntax-checked; full training/evaluation still requires the data, model
weights and a suitable GPU environment.

NAVSIM-v2 NavHard uses a separate two-stage synthetic-scene evaluation protocol.
Do not evaluate a NavHard checkpoint with the NAVSIM-v1 wrapper and call the
result EPDMS. The relevant entry point is
[`run_pdm_score_gpu_v2.py`](navsim/planning/script/run_pdm_score_gpu_v2.py).

## Qualitative material

The project page displays the manuscript's architecture and selected NAVSIM
figures. Figure 2 compares two planners on the same scene, but they generate
their **own** candidate sets; the images do not establish scorer accuracy on a
shared candidate set. Figure 3 visualizes selected register attention maps.

## Attribution and license

This repository includes NAVSIM and a vendored nuPlan devkit. See
[`THIRD_PARTY.md`](THIRD_PARTY.md), the top-level [`LICENSE`](LICENSE), and
[`nuplan-devkit/LICENSE.txt`](nuplan-devkit/LICENSE.txt). Dataset images,
map assets and pretrained weights remain subject to their publishers' terms.
