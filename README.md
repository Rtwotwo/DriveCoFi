<h1 align="center">CoFiDrive: Scene-Conditioned Registers and<br>Coarse-to-Fine Trajectory Scoring for End-to-End Driving</h1>

<p align="center">
  <a href="https://github.com/Rtwotwo/DriveCoFi"><img src="site/assets/badges/code.svg" alt="Code" height="22"></a>
  <img src="site/assets/badges/arxiv.svg" alt="arXiv (link forthcoming)" title="Paper link forthcoming" height="22">
  <img src="site/assets/badges/huggingface.svg" alt="Hugging Face (weights forthcoming)" title="Weights link forthcoming" height="22">
</p>

## News

- [ ] **2026.11.01** — Training, evaluation and visualization code released.

## Overview

<img src="site/assets/images/Figure1.webp" alt="CoFiDrive architecture: TC refines camera registers, CCS uses scene and patch memories, and CSA expands scorer training" width="100%">

CoFiDrive is a camera-based driving planner that combines scene-conditioned register features, dual-memory trajectory scoring and training-only control-space augmentation. **TC (Semantic Token Compressor)** refines 64 camera registers through scene-conditioned mixing of six experts, and the refined tokens support both trajectory generation and scoring. **CCS (Cross-Compression Scorer)** combines scene tokens with spatial patch features through candidate-dependent gated attention to score motion alternatives. **CSA (Control-Space Augmentation)** perturbs the acceleration and yaw rate of selected proposals to expand scorer supervision, and is used only during training. At inference, the planner generates and scores 64 proposals; CSA adds no inference candidates or model parameters.

## Results

### NAVSIM-v1 navtest

Results follow the manuscript's main comparison. CoFiDrive uses 10 training epochs; other methods retain their reported protocols. All metrics are on a 0–100 scale.

<div align="center">

<table>
  <thead>
    <tr>
      <th align="left">Method</th>
      <th align="right">NC ↑</th>
      <th align="right">DAC ↑</th>
      <th align="right">TTC ↑</th>
      <th align="right">Comfort ↑</th>
      <th align="right">EP ↑</th>
      <th align="right">PDMS ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="left">Ego-stat. MLP</td><td align="right">93.0</td><td align="right">77.3</td><td align="right">83.6</td><td align="right">100.0</td><td align="right">62.8</td><td align="right">65.6</td></tr>
    <tr><td align="left">UniVLA (base)</td><td align="right">96.9</td><td align="right">91.1</td><td align="right">91.7</td><td align="right">96.7</td><td align="right">76.8</td><td align="right">81.7</td></tr>
    <tr><td align="left">UniAD</td><td align="right">97.8</td><td align="right">91.9</td><td align="right">92.9</td><td align="right">100.0</td><td align="right">78.8</td><td align="right">83.4</td></tr>
    <tr><td align="left">LTF</td><td align="right">97.4</td><td align="right">92.8</td><td align="right">92.4</td><td align="right">100.0</td><td align="right">79.0</td><td align="right">83.8</td></tr>
    <tr><td align="left">PARA-Drive</td><td align="right">97.9</td><td align="right">92.4</td><td align="right">93.0</td><td align="right">99.8</td><td align="right">79.3</td><td align="right">84.0</td></tr>
    <tr><td align="left">DriveX-S</td><td align="right">97.5</td><td align="right">94.0</td><td align="right">93.0</td><td align="right">100.0</td><td align="right">79.7</td><td align="right">84.5</td></tr>
    <tr><td align="left">World4Drive</td><td align="right">97.4</td><td align="right">94.3</td><td align="right">92.8</td><td align="right">100.0</td><td align="right">79.9</td><td align="right">85.1</td></tr>
    <tr><td align="left">DRAMA</td><td align="right">98.0</td><td align="right">93.1</td><td align="right">94.8</td><td align="right">100.0</td><td align="right">80.1</td><td align="right">85.5</td></tr>
    <tr><td align="left">PRIX</td><td align="right">98.1</td><td align="right">96.3</td><td align="right">94.1</td><td align="right">100.0</td><td align="right">82.3</td><td align="right">87.8</td></tr>
    <tr><td align="left">DiffusionDrive</td><td align="right">98.2</td><td align="right">96.2</td><td align="right">94.7</td><td align="right">100.0</td><td align="right">82.2</td><td align="right">88.1</td></tr>
    <tr><td align="left">DIVER</td><td align="right">98.5</td><td align="right">96.5</td><td align="right">94.9</td><td align="right">100.0</td><td align="right">82.6</td><td align="right">88.3</td></tr>
    <tr><td align="left">AutoVLA (Post-RFT)</td><td align="right">98.4</td><td align="right">95.6</td><td align="right">98.0</td><td align="right">99.9</td><td align="right">81.9</td><td align="right">89.1</td></tr>
    <tr><td align="left">VADv2</td><td align="right">98.3</td><td align="right">97.4</td><td align="right">95.7</td><td align="right">100.0</td><td align="right">82.3</td><td align="right">89.3</td></tr>
    <tr><td align="left">DriveVLA-W0 (query)</td><td align="right">98.7</td><td align="right">99.1</td><td align="right">95.3</td><td align="right">99.3</td><td align="right">83.3</td><td align="right">90.2</td></tr>
    <tr><td align="left">ReCogDrive</td><td align="right">97.9</td><td align="right">97.3</td><td align="right">94.9</td><td align="right">100.0</td><td align="right">87.3</td><td align="right">90.8</td></tr>
    <tr><td align="left">Hydra-MDP++ (V2-99)</td><td align="right">98.6</td><td align="right">98.6</td><td align="right">95.1</td><td align="right">100.0</td><td align="right">85.7</td><td align="right">91.0</td></tr>
    <tr><td align="left">iPad</td><td align="right">98.6</td><td align="right">98.3</td><td align="right">94.9</td><td align="right">100.0</td><td align="right">88.0</td><td align="right">91.7</td></tr>
    <tr><td align="left"><strong>CoFiDrive (train85k)</strong></td><td align="right"><strong>99.0</strong></td><td align="right"><strong>98.2</strong></td><td align="right"><strong>96.2</strong></td><td align="right"><strong>100.0</strong></td><td align="right"><strong>88.3</strong></td><td align="right"><strong>92.4</strong></td></tr>
    <tr><td align="left"><strong>CoFiDrive (trainval103k)</strong></td><td align="right"><strong>99.0</strong></td><td align="right"><strong>98.4</strong></td><td align="right"><strong>96.0</strong></td><td align="right"><strong>100.0</strong></td><td align="right"><strong>89.6</strong></td><td align="right"><strong>92.9</strong></td></tr>
  </tbody>
</table>

</div>

The train85k main result is the mean of three independent training seeds. This aggregation does not extend to trainval103k or NavHard. The trainval103k configuration also uses a resampled long-trajectory target, so its difference from train85k does not isolate dataset size.

### NAVSIM-v2 NavHard

Stage 1 evaluates original observations and Stage 2 evaluates synthetic continuations. CoFiDrive and the local DrivoR baseline use the same 10-epoch training and evaluation protocol. RAP-DINO and GTRS-DrivoR report EPDMS only, so their per-metric cells are left blank. The local 47.3 DrivoR result is distinct from DrivoR's published 48.3 EPDMS.

<div align="center">

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Method</th>
      <th colspan="9">Stage 1</th>
      <th colspan="9">Stage 2</th>
      <th rowspan="2" align="right">EPDMS ↑</th>
    </tr>
    <tr>
      <th align="right">NC</th>
      <th align="right">DAC</th>
      <th align="right">DDC</th>
      <th align="right">TLC</th>
      <th align="right">EP</th>
      <th align="right">TTC</th>
      <th align="right">LK</th>
      <th align="right">HC</th>
      <th align="right">EC</th>
      <th align="right">NC</th>
      <th align="right">DAC</th>
      <th align="right">DDC</th>
      <th align="right">TLC</th>
      <th align="right">EP</th>
      <th align="right">TTC</th>
      <th align="right">LK</th>
      <th align="right">HC</th>
      <th align="right">EC</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="left">RAP-DINO</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">39.6</td></tr>
    <tr><td align="left">GTRS-DrivoR</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">45.8</td></tr>
    <tr><td align="left">DrivoR (local)</td><td align="right">98.9</td><td align="right">95.3</td><td align="right">99.1</td><td align="right">99.8</td><td align="right">73.0</td><td align="right">99.1</td><td align="right">93.8</td><td align="right">97.6</td><td align="right">64.0</td><td align="right">91.6</td><td align="right">87.3</td><td align="right">92.3</td><td align="right">98.2</td><td align="right">68.6</td><td align="right">89.2</td><td align="right">50.8</td><td align="right">98.5</td><td align="right">76.8</td><td align="right">47.3</td></tr>
    <tr><td align="left"><strong>CoFiDrive</strong></td><td align="right"><strong>99.1</strong></td><td align="right"><strong>95.3</strong></td><td align="right"><strong>99.0</strong></td><td align="right"><strong>99.6</strong></td><td align="right"><strong>73.9</strong></td><td align="right"><strong>98.7</strong></td><td align="right"><strong>93.6</strong></td><td align="right"><strong>97.6</strong></td><td align="right"><strong>69.3</strong></td><td align="right"><strong>89.6</strong></td><td align="right"><strong>89.3</strong></td><td align="right"><strong>91.8</strong></td><td align="right"><strong>98.7</strong></td><td align="right"><strong>69.7</strong></td><td align="right"><strong>87.6</strong></td><td align="right"><strong>53.3</strong></td><td align="right"><strong>98.7</strong></td><td align="right"><strong>73.9</strong></td><td align="right"><strong>48.0</strong></td></tr>
  </tbody>
</table>

</div>

Stage 2 mixes component changes: DAC and LK rise by 2.0 and 2.5 points, while NC, TTC and EC fall by 2.0, 1.6 and 2.9. The aggregate score should therefore not be read as a uniform improvement across safety and compliance measures.

## Visualizations

### Trajectory planning

Four scenes from the paper. Each column compares DrivoR and CoFiDrive on the same scene; each planner generates and scores its own candidates. Blue curves show proposals, yellow marks the selected trajectory, and white is the reference where shown.

<table>
  <tr><th width="25%">Straight</th><th width="25%">Left turn</th><th width="25%">Right turn</th><th width="25%">Straight · Scene 2</th></tr>
  <tr><td><img src="site/assets/gallery/straight-1-drivor.webp" alt="DrivoR: Straight" width="100%"><br>DrivoR</td><td><img src="site/assets/gallery/left-turn-1-drivor.webp" alt="DrivoR: Left turn" width="100%"><br>DrivoR</td><td><img src="site/assets/gallery/right-turn-1-drivor.webp" alt="DrivoR: Right turn" width="100%"><br>DrivoR</td><td><img src="site/assets/gallery/straight-2-drivor.webp" alt="DrivoR: Straight · Scene 2" width="100%"><br>DrivoR</td></tr>
  <tr><td><img src="site/assets/gallery/straight-1-cofidrive.webp" alt="CoFiDrive: Straight" width="100%"><br>CoFiDrive</td><td><img src="site/assets/gallery/left-turn-1-cofidrive.webp" alt="CoFiDrive: Left turn" width="100%"><br>CoFiDrive</td><td><img src="site/assets/gallery/right-turn-1-cofidrive.webp" alt="CoFiDrive: Right turn" width="100%"><br>CoFiDrive</td><td><img src="site/assets/gallery/straight-2-cofidrive.webp" alt="CoFiDrive: Straight · Scene 2" width="100%"><br>CoFiDrive</td></tr>
</table>

### Register attention

Mean attention in the four camera views. DrivoR is shown above CoFiDrive; warmer colors indicate higher displayed attention.

<table>
  <tr><th width="25%">Front</th><th width="25%">Left</th><th width="25%">Right</th><th width="25%">Rear</th></tr>
  <tr><td><img src="site/assets/gallery/attention-f0-drivor-5.webp" alt="DrivoR: Front" width="100%"><br>DrivoR</td><td><img src="site/assets/gallery/attention-l0-drivor-5.webp" alt="DrivoR: Left" width="100%"><br>DrivoR</td><td><img src="site/assets/gallery/attention-r0-drivor-5.webp" alt="DrivoR: Right" width="100%"><br>DrivoR</td><td><img src="site/assets/gallery/attention-b0-drivor-5.webp" alt="DrivoR: Rear" width="100%"><br>DrivoR</td></tr>
  <tr><td><img src="site/assets/gallery/attention-f0-cofidrive-5.webp" alt="CoFiDrive: Front" width="100%"><br>CoFiDrive</td><td><img src="site/assets/gallery/attention-l0-cofidrive-5.webp" alt="CoFiDrive: Left" width="100%"><br>CoFiDrive</td><td><img src="site/assets/gallery/attention-r0-cofidrive-5.webp" alt="CoFiDrive: Right" width="100%"><br>CoFiDrive</td><td><img src="site/assets/gallery/attention-b0-cofidrive-5.webp" alt="CoFiDrive: Rear" width="100%"><br>CoFiDrive</td></tr>
</table>

### Control-space augmentation

One saved scene before and after CSA: 64 original proposals and 192 additional training slots. The curves use stored trajectory coordinates. Failed augmentation slots retain their seeds.

<img src="site/assets/images/csa-preview.svg" alt="Original proposals and the expanded CSA training list for one saved scene" width="100%">

## Documentation

| Installation | Training | Evaluation | Method |
| :---: | :---: | :---: | :---: |
| [Environment and data](docs/setup.md) | [Train CoFiDrive](docs/reproduction.md#train85k-a-10-epoch-run) | [Evaluate checkpoints](docs/reproduction.md#evaluate-on-navtest) | [TC, CCS and CSA](docs/method.md) |

Detailed experiment tables and visualization instructions are in `docs/experiments.md` and `docs/visualizations.md`. Datasets, backbone weights and trained checkpoints are not bundled with the code. The configuration key remains `agent=more_ccs` for checkpoint compatibility.

## Acknowledgements

CoFiDrive builds on [DrivoR](https://github.com/valeoai/DrivoR), [NAVSIM](https://github.com/autonomousvision/navsim), [nuPlan](https://github.com/motional/nuplan-devkit) and [DINOv2](https://github.com/facebookresearch/dinov2). We thank their authors for sharing their code, data and evaluation tools. The NAVSIM/DrivoR-derived code retains its Apache-2.0 license in [`LICENSE`](LICENSE), and the vendored nuPlan dependency retains its license in [`nuplan-devkit/LICENSE.txt`](nuplan-devkit/LICENSE.txt).
