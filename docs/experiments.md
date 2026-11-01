# Experiments and reported results

[README](../README.md#results) · [Method](method.md)

The values below reproduce the current paper's tables. All CoFiDrive configurations
use 10 training epochs. They are displayed without recomputing scores from component means.

## Main results

| Evaluation | Training setting | NC | DAC | TTC | Comfort | EP | PDMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NAVSIM-v1 navtest | train85k | 99.0 | 98.2 | 96.2 | 100.0 | 88.3 | 92.4 |
| NAVSIM-v1 navtest | trainval103k | 99.0 | 98.4 | 96.0 | 100.0 | 89.6 | 92.9 |

The 92.4 train85k main result averages three independent training seeds. This
aggregation does not extend to trainval103k, NavHard or the ablation tables.
The trainval103k configuration also uses a resampled long-trajectory target, so
its difference from train85k does not isolate dataset size.

## Sequential component ablation

Train85k, navtest, 10 epochs and 64 inference candidates are fixed. Parameter counts
are in millions. Each row adds a component to the previous configuration.

| Configuration | Total parameters | Scorer parameters | PDMS |
| --- | ---: | ---: | ---: |
| Register-based base planner | 40.76 | 5.08 | 91.70 |
| + TC | 45.51 | 5.80 | 91.80 |
| + Dual-memory CCS | 47.52 | 7.81 | 91.90 |
| + CSA | 47.52 | 7.81 | 92.40 |

The successive gains are 0.10, 0.10 and 0.50 points. Their sum is 0.70 over the
base configuration. CSA produces the largest increase in this sequence while
keeping the model and inference candidate counts unchanged. These are conditional
additions, rather than independent main effects.

## Routing, fusion and supervision

**Register processing (Table B.4).** Dual-memory CCS and CSA remain fixed.

| Register processing | PDMS |
| --- | ---: |
| Bypass TC | 92.05 |
| Single SwiGLU | 92.12 |
| Uniform 6-expert mix | 92.18 |
| Scene-routed 6-expert mix | 92.40 |

Scene routing improves the reported score by 0.22 over uniform mixing and by 0.28
over the single-SwiGLU control.

**Fusion and supervision (Table B.5).** TC and both visual memories remain fixed.

| Fusion | Auxiliary supervision | PDMS |
| --- | --- | ---: |
| Candidate-dependent gate | None | 91.90 |
| Equal average | Original 64 + CSA 192 | 92.24 |
| Candidate-dependent gate | Original 64 repeated four times | 92.00 |
| Candidate-dependent gate | Original 64 + CSA 192 | 92.40 |

With CSA fixed, gating improves the score by 0.16 over averaging. With the auxiliary
list size and loss weight fixed, CSA improves it by 0.40 over repeated originals.
The tables report point differences, not significance tests.

## Token and expert sweeps

| Registers per camera | PDMS | Expert count | PDMS |
| ---: | ---: | ---: | ---: |
| 16 | 91.85 | 4 | 91.54 |
| 32 | 91.66 | 6 | 91.85 |
| 64 | 91.66 | 8 | 91.59 |

These sweeps form a separate run group from the sequential ablation. The available
summaries do not establish the cause of the 91.85 versus 92.40 default-score gap.
The website therefore compares count settings within each sweep.

## NavHard: two evaluation stages

CoFiDrive reaches **48.0 EPDMS**, compared with **47.3** for the local DrivoR
reproduction under the same 10-epoch protocol. These are the local comparison
values; DrivoR's published NavHard result is 48.3.

| Stage | Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | DrivoR (local) | 98.9 | 95.3 | 99.1 | 99.8 | 73.0 | 99.1 | 93.8 | 97.6 | 64.0 |
| 1 | CoFiDrive | 99.1 | 95.3 | 99.0 | 99.6 | 73.9 | 98.7 | 93.6 | 97.6 | 69.3 |
| 2 | DrivoR (local) | 91.6 | 87.3 | 92.3 | 98.2 | 68.6 | 89.2 | 50.8 | 98.5 | 76.8 |
| 2 | CoFiDrive | 89.6 | 89.3 | 91.8 | 98.7 | 69.7 | 87.6 | 53.3 | 98.7 | 73.9 |

Stage 1 evaluates original observations; Stage 2 evaluates synthetic continuations.
In Stage 2, DAC and LK rise by 2.0 and 2.5 points, while NC, TTC and EC fall by 2.0,
1.6 and 2.9. The aggregate score captures this mixture of component changes.

| Abbreviation | Metric |
| --- | --- |
| NC | No at-fault collision |
| DAC | Drivable-area compliance |
| DDC | Driving-direction compliance |
| TLC | Traffic-light compliance |
| EP | Ego progress |
| TTC | Time to collision |
| LK | Lane keeping |
| HC | History comfort |
| EC | Extended comfort |

The machine-readable website values are in [results.json](../site/data/results.json).
The benchmark protocols measure planning performance under their specified
rollouts; the gallery presents selected examples from those evaluations.
