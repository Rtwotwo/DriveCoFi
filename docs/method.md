# CoFiDrive: method and implementation

[README](../README.md) · [Experiments](experiments.md) · [Reproduction](reproduction.md)

CoFiDrive generates 64 candidate trajectories from a compact scene representation,
then scores them using scene context and spatial image features. TC changes the
scene representation, CCS combines the two visual memories, and CSA expands the
candidate list used to train CCS.

## 1. From camera images to scene memory

Four cameras share a DINOv2 ViT-S/14 encoder with rank-32 Q/V LoRA adapters.
Each view contributes 16 learned registers. Projecting their 384-dimensional
features to width 256 gives a scene memory with shape **64 × 256**.

The camera-register bottleneck provides the initial token reduction. TC preserves
the 64-token sequence. Its scene-level router pools the input tokens, predicts six
softmax weights, and applies the same weights across the scene. All six SwiGLU
experts process the input. Their weighted output is added through a scaled residual.
Each expert has hidden width 1,024.

Implementation: [image encoder](../navsim/agents/more_ccs/layers/image_encoder/dinov2_lora.py)
and [scene-adaptive experts](../navsim/agents/more_ccs/layers/moe/moe_scene_adaptive.py).

## 2. Candidate generation

The decoder combines 64 learned trajectory queries with the ego-state embedding.
Four decoder blocks use self-attention, cross-attention to the refined scene tokens,
and feed-forward layers. Each proposal has eight future poses, with coordinates
`(x, y, heading)`, at 0.5-second intervals over four seconds.

The generator's visual memory stays at 64 scene tokens. The patch memory described
below is consumed by the scorer.

## 3. CCS: Cross-Compression Scorer

CCS embeds each candidate and reads two memories:

| Memory | Shape | Construction |
| --- | --- | --- |
| Scene | 64 × 256 | TC output plus camera/type embeddings and normalization |
| Patch | 1,008 × 256 | Pool each view to 12 × 21; project and add position/camera/type embeddings |

The patch-backbone output is detached on this scoring path. A candidate-dependent
gate combines the responses of the two cross-attention branches. The four scorer
blocks predict six component scores: NC, DAC, TTC, EP, DDC and comfort. NC is named
NOC in parts of the implementation.

The configured inference aggregation multiplies NC and DAC by a weighted sum of
TTC, EP and comfort, evaluated in log space. TTC and EP have weight 5; comfort has
weight 2. DDC receives training supervision but is omitted from this aggregation.
The selected trajectory is the highest-ranked member of the original 64 proposals.

Here, **coarse-to-fine describes the two granularities of visual evidence** read for
each candidate. All 64 proposals are scored; there is no successive candidate-pruning
cascade.

Implementation: [`_build_scorer_kv`, `_score_candidate_set` and `_predicted_pdm_score`](../navsim/agents/more_ccs/drivor_model.py),
and the [scorer blocks](../navsim/agents/more_ccs/score_module/scorer.py).

## 4. CSA: Control-Space Augmentation

CSA trains the same CCS used at inference. The training sequence is:

1. Evaluate the 64 generated proposals with the PDM oracle and select the top 16.
2. Estimate acceleration and yaw rate from each seed's poses. Allocate 12 augmentation
   slots per seed.
3. Add a per-channel bias and temporally smoothed noise to the controls. Clip the
   controls and roll them out with the bicycle model from the current speed.
4. Check velocity and deviations from the seed. Try at most four samples per slot;
   a slot with no accepted sample retains its seed.
5. Combine the 192 retained slots with all 64 original proposals. Compute PDM component
   targets for this 256-entry auxiliary list and apply the auxiliary CCS loss.

These geometric checks do not establish that a candidate is collision-free. The PDM
component targets provide the supervision for its quality. Fallbacks can create duplicates.

The main scoring loss remains on the original proposals. The auxiliary scoring loss
has weight 0.5. At inference, CSA and PDM target computation are disabled, so CSA adds
neither learned parameters nor inference candidates.

Implementation: [`_expand_topk_proposals` and `score_auxiliary_proposals`](../navsim/agents/more_ccs/drivor_model.py),
[bicycle rollout](../navsim/agents/more_ccs/ccs/bicycle.py), and
[loss computation](../navsim/agents/more_ccs/layers/losses/drivor_loss.py).

## Configuration map

The complete defaults are in [more_ccs.yaml](../navsim/planning/script/config/common/agent/more_ccs.yaml).

| Configuration key | Role | Default |
| --- | --- | --- |
| `num_scene_tokens` | Registers per camera | 16 |
| `proposal_num` | Generated candidates | 64 |
| `num_poses` | Future poses per candidate | 8 |
| `moe.num_experts` | Active TC experts | 6 |
| `scorer_raw_tokens.enabled` | Patch-memory branch | `true` |
| `scorer_raw_tokens.detach_patch_backbone` | Detach patch path | `true` |
| `traj_expansion.seed_topk` | CSA seeds | 16 |
| `traj_expansion.num_augment_per_seed` | Slots per seed | 12 |
| `traj_expansion.inference_enabled` | Expand inference candidates | `false` |

The internal `more_ccs` agent name and `DrivoRModel` class names are retained for
compatibility with the existing configuration and checkpoint interfaces.
