<h1 align="center">🚗 CoFiDrive: Scene-Conditioned Registers and<br>Coarse-to-Fine Trajectory Scoring for End-to-End Driving</h1>

<p align="center">
  <img src="https://img.shields.io/badge/arXiv-B31B1B?logo=arxiv&amp;logoColor=white" alt="arXiv" title="Paper link forthcoming" height="24">
  <a href="https://github.com/Rtwotwo/DriveCoFi"><img src="https://img.shields.io/badge/GitHub-181717?logo=github&amp;logoColor=white" alt="GitHub repository" height="24"></a>
  <a href="https://rtwotwo.github.io/DriveCoFi/"><img src="https://img.shields.io/badge/GitHub.io-2563EB?logo=githubpages&amp;logoColor=white" alt="GitHub Pages project website" height="24"></a>
</p>

<p align="center">
  <a href="#-overview">Overview</a> ·
  <a href="#-results">Results</a> ·
  <a href="#-visualizations">Visualizations</a> ·
  <a href="#-getting-started">Getting started</a> ·
  <a href="#-training-and-evaluation">Training &amp; evaluation</a>
</p>

## 📰 News

- **2026.11.01** — Released the code and expanded project website: 16 paired scenes, four-camera attention, saved CSA trajectories and interactive ablation results.

## 🔍 Overview

<p align="center">
  <a href="site/assets/source/Figure1.svg"><img src="site/assets/images/Figure1.webp" alt="CoFiDrive architecture: TC refines camera registers, CCS scores trajectories with scene and patch memories, and CSA augments scorer training" width="100%"></a>
</p>

**CoFiDrive combines scene-conditioned visual features, dual-memory trajectory scoring, and control-space augmentation for end-to-end driving.** Generating a maneuver requires road and traffic context; choosing among nearby trajectories also depends on local spatial details. CoFiDrive uses a compact scene memory for trajectory generation and gives the scorer access to both that memory and spatial image features. Training adds motion alternatives around promising proposals to improve the scorer's ability to compare candidates.

The method has three components:

- **TC · Semantic Token Compressor.** Four cameras contribute 16 learned registers each. A scene-level router mixes six SwiGLU experts to refine the resulting 64 scene tokens, which serve both the generator and scorer. The camera registers provide the compact representation; TC adapts their features to the scene while retaining all 64 tokens.
- **CCS · Cross-Compression Scorer.** Each proposal attends to the refined scene memory and 1,008 pooled image-patch tokens. A candidate-dependent gate combines the two responses to predict trajectory quality. Coarse-to-fine describes these two levels of visual detail.
- **CSA · Control-Space Augmentation.** During training, the top 16 proposals ranked by the PDM evaluator each seed 12 acceleration and yaw-rate perturbations. A bicycle model converts them into trajectories for auxiliary CCS supervision. The training list contains 192 augmentation slots and the original 64 proposals; unsuccessful slots retain their seed trajectories.

**At inference, CCS scores the original 64 generated proposals.** CSA expands training supervision without adding inference candidates or model parameters.

<p align="center">
  <a href="https://rtwotwo.github.io/DriveCoFi/#method">Method details</a> ·
  <a href="site/assets/source/Figure1.svg">Architecture SVG</a>
</p>

## 📊 Results

| Benchmark | Training split / setting | Training epochs | Metric | CoFiDrive |
| :--- | :--- | :---: | :---: | ---: |
| NAVSIM-v1 · navtest | train85k | 10 | PDMS ↑ | **92.4** |
| NAVSIM-v1 · navtest | trainval103k | 10 | PDMS ↑ | **92.9** |
| NAVSIM-v2 · NavHard | NavHard training protocol | 10 | EPDMS ↑ | **48.0** |

The train85k main result is averaged over three independently seeded training runs. NAVSIM-v1 and NavHard use their respective benchmark evaluation protocols. See the [project page](https://rtwotwo.github.io/DriveCoFi/#results) for results and the sequential component ablation.

## 🎬 Visualizations

The [project website](https://rtwotwo.github.io/DriveCoFi/) provides **16 paired planning scenes**, a **four-camera attention explorer**, **five saved CSA trajectory sets**, and **six ablation views**. The gallery opens images directly; the trajectory and result explorers are rendered in the browser.

### Straight driving, left turns and right turns

Each pair shows the same scene for DrivoR and CoFiDrive. Blue curves are proposals, yellow marks the selected trajectory, and white denotes the reference where shown. Each planner generates and scores its own candidates.

<table align="center">
  <tr><th>Maneuver</th><th>DrivoR</th><th>CoFiDrive</th></tr>
  <tr>
    <td>Straight</td>
    <td><a href="https://rtwotwo.github.io/DriveCoFi/#evidence"><img src="site/assets/gallery/straight-1-drivor.webp" alt="DrivoR straight-driving example" width="360"></a></td>
    <td><a href="https://rtwotwo.github.io/DriveCoFi/#evidence"><img src="site/assets/gallery/straight-1-cofidrive.webp" alt="CoFiDrive straight-driving example" width="360"></a></td>
  </tr>
  <tr>
    <td>Left turn</td>
    <td><a href="https://rtwotwo.github.io/DriveCoFi/#evidence"><img src="site/assets/gallery/left-turn-1-drivor.webp" alt="DrivoR left-turn example" width="360"></a></td>
    <td><a href="https://rtwotwo.github.io/DriveCoFi/#evidence"><img src="site/assets/gallery/left-turn-1-cofidrive.webp" alt="CoFiDrive left-turn example" width="360"></a></td>
  </tr>
  <tr>
    <td>Right turn</td>
    <td><a href="https://rtwotwo.github.io/DriveCoFi/#evidence"><img src="site/assets/gallery/right-turn-1-drivor.webp" alt="DrivoR right-turn example" width="360"></a></td>
    <td><a href="https://rtwotwo.github.io/DriveCoFi/#evidence"><img src="site/assets/gallery/right-turn-1-cofidrive.webp" alt="CoFiDrive right-turn example" width="360"></a></td>
  </tr>
</table>

<p align="center"><a href="https://rtwotwo.github.io/DriveCoFi/#evidence"><strong>Browse all 16 matched scenes →</strong></a></p>

### Camera-register attention

Choose a camera and an individual register in the [attention explorer](https://rtwotwo.github.io/DriveCoFi/#attention). It includes front, left, right and rear views, with RGB, registers 0/4/8/12 and mean attention: 48 independent image panels in total.

<table align="center">
  <tr><th>DrivoR · front mean attention</th><th>CoFiDrive · front mean attention</th></tr>
  <tr>
    <td><a href="https://rtwotwo.github.io/DriveCoFi/#attention"><img src="site/assets/gallery/attention-f0-drivor-5.webp" alt="DrivoR front-camera mean attention" width="413"></a></td>
    <td><a href="https://rtwotwo.github.io/DriveCoFi/#attention"><img src="site/assets/gallery/attention-f0-cofidrive-5.webp" alt="CoFiDrive front-camera mean attention" width="413"></a></td>
  </tr>
</table>

### CSA: inspect the saved trajectory candidates

The explorer draws the original 64 and augmented 192 entries from five saved caches. Toggle trajectory layers and reveal the eight future poses. The plotted coordinates are stored outputs; the time control reveals waypoints rather than playing driving footage.

<p align="center"><a href="https://rtwotwo.github.io/DriveCoFi/#augmentation"><img src="site/assets/images/csa-preview.svg" alt="Original candidate trajectories and the larger CSA training list, drawn from the same saved scene" width="100%"></a></p>

<p align="center"><a href="https://rtwotwo.github.io/DriveCoFi/#augmentation"><strong>Open the interactive CSA explorer →</strong></a></p>

### Ablations and NavHard metrics

Switch between sequential additions, register processing, fusion, supervision, token counts and expert counts. The NavHard view compares all nine component metrics for both evaluation stages against the local DrivoR reproduction.

<p align="center"><a href="https://rtwotwo.github.io/DriveCoFi/#results"><img src="site/assets/images/ablation-summary.svg" alt="Sequential component ablation from 91.70 to 92.40 PDMS" width="100%"></a></p>

## 📚 Documentation

| Guide | Contents |
| --- | --- |
| [Method and implementation](docs/method.md) | TC / CCS / CSA, memory shapes, gradient paths, scoring and code locations |
| [Experiments and results](docs/experiments.md) | Main results, all ablation groups, NavHard breakdown and comparison settings |
| [Visualization guide](docs/visualizations.md) | Gallery legends, saved CSA data, attention commands and asset provenance |
| [Training and evaluation](docs/reproduction.md) | Data setup, train85k launchers, checkpoint evaluation and configuration checks |

## 🛠️ Getting started

### Installation

Use Linux with Python 3.9 and a CUDA-compatible PyTorch installation. The original environment uses PyTorch 2.1; dependency versions are listed in [requirements.txt](requirements.txt) and [environment.yml](environment.yml).

```bash
git clone https://github.com/Rtwotwo/DriveCoFi.git
cd DriveCoFi

conda create -n cofidrive python=3.9 -y
conda activate cofidrive

# Install the PyTorch build appropriate for your CUDA environment first.
pip install -r requirements.txt
pip install -e ./nuplan-devkit
pip install -e .
```

### Data and pretrained weights

1. Prepare the NAVSIM data and nuPlan maps using the [NAVSIM instructions](https://github.com/autonomousvision/navsim) and the local [installation guide](docs/install.md).
2. Download the [DINOv2 ViT-S/14 register weights](https://huggingface.co/timm/vit_small_patch14_reg4_dinov2.lvd142m). Place `model.safetensors` at `weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors`, or update the path in [more_ccs.yaml](navsim/planning/script/config/common/agent/more_ccs.yaml).
3. Set the data, map, and experiment paths:

```bash
export OPENSCENE_DATA_ROOT=/path/to/navsim/dataset
export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps
export NAVSIM_EXP_ROOT=/path/to/experiment/output
export CUDA_VISIBLE_DEVICES=0
```

Prepare the metric cache before evaluation. See [dataset splits](docs/splits.md) and the [metric-caching launcher](scripts/evaluation/run_metric_caching.sh) for the required configuration. Datasets, pretrained weights, and trained checkpoints are obtained separately.

## 🚀 Training and evaluation

### NAVSIM-v1 training

The release launcher trains on train85k for 10 epochs:

```bash
SEED=2 NUM_GPUS=1 bash scripts/release/train_navsim_v1.sh
```

Environment variables control the seed, batch size, number of GPUs, and output name. Additional Hydra options can be appended to the command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 SEED=3 NUM_GPUS=4 BATCH_SIZE=16 \
  EXPERIMENT_NAME=cofidrive/train85k_seed3 \
  bash scripts/release/train_navsim_v1.sh
```

### NAVSIM-v1 evaluation

Set `CHECKPOINT` to the trained model and run navtest evaluation:

```bash
CHECKPOINT=/path/to/checkpoint.ckpt NUM_GPUS=1 \
  bash scripts/release/eval_navsim_v1.sh
```

The evaluation launcher keeps 64 proposals and disables trajectory expansion. The Hydra agent key remains `agent=more_ccs` to preserve the existing configuration and checkpoint interfaces.

### NAVSIM-v2 evaluation

NavHard uses a separate two-stage EPDMS protocol. Its evaluation entry point is [run_pdm_score_gpu_v2.py](navsim/planning/script/run_pdm_score_gpu_v2.py); configure the NavHard data, caches, and checkpoint for that evaluator.

## 📁 Repository structure

```text
navsim/
  agents/more_ccs/       Full CoFiDrive model: TC + CCS + CSA
  agents/more_dual/      Dual-channel variant
  agents/more_tc/        TC variant
  planning/script/      Training, evaluation, and Hydra configurations
scripts/release/        Portable NAVSIM-v1 launchers
scripts/                Original experiment launchers
docs/                   Installation and dataset documentation
download/               Dataset download helpers
nuplan-devkit/          nuPlan dependency
viz/                    Planning visualization tools
site/                   Project website and paper figures
```

The original experiment launchers retain experiment-specific GPU and checkpoint settings. Use `scripts/release/` as the starting point for a new environment.

## 🙏 Acknowledgements

CoFiDrive builds on [DrivoR](https://github.com/valeoai/DrivoR) for register-based driving, [NAVSIM](https://github.com/autonomousvision/navsim) for training and evaluation, [nuPlan](https://github.com/motional/nuplan-devkit) for maps and planning utilities, and [DINOv2](https://github.com/facebookresearch/dinov2) for visual representations. We thank their authors for making these resources available.

For source-code licenses and attribution, see [LICENSE](LICENSE), [THIRD_PARTY.md](THIRD_PARTY.md), and the [nuPlan license](nuplan-devkit/LICENSE.txt).
