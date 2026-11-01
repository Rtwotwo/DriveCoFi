"""
# For NavTest
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 viz/viz_scene2patch_attention_single.py \
    --checkpoint_path /path/to/your/checkpoint.ckpt \
    --agent_name more_ccs \
    --experiment_name map_scene_attention \
    --split navtest \
    --csv /path/to/your/tokens.csv \
    --output_dir /path/to/your/output_dir \
    --dpi 300

# For NavHard Test
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 viz/viz_scene2patch_attention_single.py \
    --checkpoint_path /path/to/your/checkpoint.ckpt \
    --agent_name more_ccs \
    --experiment_name map_scene_attention \
    --split navhard_two_stage \
    --csv /path/to/your/tokens.csv \
    --output_dir /path/to/your/output_dir \
    --dpi 300
"""
import os
import sys
import argparse
import csv
import io
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Map normalized values in [0, 1] through the blue-to-bright-yellow portion of jet.
_JET_BLUE_YELLOW = matplotlib.colors.LinearSegmentedColormap.from_list(
    "jet_blue_yellow", plt.get_cmap("jet")(np.linspace(0.0, 0.65, 256))
)
import cv2
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate


CAMERA_TOKEN_ORDER = [
    "cam_f0", "cam_b0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2"
]
DEFAULT_VIZ_CAMS = ["cam_f0", "cam_b0", "cam_l0", "cam_r0"]

# Keep all visualization implementations available, but only persist the final
# stacked front/back attention PDF by default.
SAVE_INTERMEDIATE_OUTPUTS = False


def parse_args():
    parser = argparse.ArgumentParser(description="DrivoR Scene Token → Patch Attention Visualization")

    parser.add_argument("--repo_root", type=str,
                        default=str(Path(__file__).resolve().parents[1]),
                        help="Path to drivoR repository root")
    parser.add_argument("--dataset_root", type=str,
                        default=str(Path(__file__).resolve().parents[2]),
                        help="Path to dataset root directory")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint file")
    parser.add_argument("--agent_name", type=str, default="drivoR_moetoken",
                        help="Agent config name for Hydra")
    parser.add_argument("--experiment_name", type=str, default="attention_map_viz",
                        help="Experiment name for Hydra config")
    parser.add_argument("--split", type=str, default="navtrain",
                        help="Train/test split name")
    parser.add_argument("--num_scene_tokens", type=int, default=None,
                        help="Override agent.config.num_scene_tokens to match the checkpoint (e.g. 32 for scenetoken32)")

    parser.add_argument("--token", type=str, default=None,
                        help="Specific scene token to visualize (random if not set)")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV file with a 'token' column; visualizes all its tokens (viz_bev_traj style)")
    parser.add_argument("--aggregation", type=str, default="besthead",
                        choices=["last", "shallow", "besthead"],
                        help="Attention aggregation method")
    parser.add_argument("--viz_types", type=str, nargs="+",
                        default=["attention_maps"],
                        choices=["attention_maps", "trajectory_camera", "patch_energy",
                                 "scene_similarity", "layer_divergence"],
                        help="Which visualizations to run")

    parser.add_argument("--H_p", type=int, default=48,
                        help="Patch grid height")
    parser.add_argument("--W_p", type=int, default=82,
                        help="Patch grid width")
    parser.add_argument("--shallow_last_k", type=int, default=3,
                        help="Number of last layers for shallow rollout")
    parser.add_argument("--shallow_alpha", type=float, default=0.9,
                        help="Alpha for shallow rollout residual connection")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save figures")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for saved figures")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")

    parser.add_argument("--heatmap_norm", type=str, default="global",
                        choices=["global", "camera", "token"],
                        help="Heatmap normalization: global is comparable across cameras/tokens; token is legacy per-map min-max")
    parser.add_argument("--heatmap_percentile", type=float, default=99.5,
                        help="Upper percentile used for global/camera heatmap scaling when --heatmap_vmax is not set")
    parser.add_argument("--heatmap_vmin", type=float, default=0.0,
                        help="Fixed heatmap lower bound for global/camera scaling")
    parser.add_argument("--heatmap_vmax", type=float, default=None,
                        help="Fixed heatmap upper bound; use the same value across model runs for strict comparability")
    parser.add_argument("--heatmap_alpha", type=float, default=0.5,
                        help="Overlay alpha for attention heatmaps")

    parser.add_argument("--group_mode", action="store_true", default=False,
                        help="Group scene tokens into G groups and sum within each group (one heatmap per group)")
    parser.add_argument("--group_count", type=int, default=4,
                        help="Number of groups for group_mode (default: 4)")
    parser.add_argument("--selected_scene_tokens", type=int, nargs="+", default=[0, 4, 8, 12],
                        help="Scene token indices to visualize in non-group mode (default: 0 4 8 12)")

    parser.add_argument("--inline_helpers", type=str, default="1",
                        help="Dummy arg")

    return parser.parse_args()


# ==============================================================================
# Environment & Agent Loading
# ==============================================================================

def setup_environment(args):
    repo_root = Path(args.repo_root)
    dataset_root = Path(args.dataset_root)

    os.environ["NUPLAN_MAPS_ROOT"] = str(dataset_root / "dataset/maps")
    os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
    os.environ["OPENSCENE_DATA_ROOT"] = str(dataset_root / "dataset")
    os.environ["NAVSIM_EXP_ROOT"] = str(dataset_root / "exp")
    os.environ["NAVSIM_DEVKIT_ROOT"] = str(repo_root)

    # Prioritize this repo's navsim over any other navsim on sys.path/PYTHONPATH
    # (e.g. when launched from another repo's cwd), so the imported agent class
    # and composed config match the checkpoint's model/config schema.
    repo_root_str = str(repo_root)
    while repo_root_str in sys.path:
        sys.path.remove(repo_root_str)
    sys.path.insert(0, repo_root_str)

    assert Path(os.environ["NUPLAN_MAPS_ROOT"]).exists(), \
        f"Maps root not found: {os.environ['NUPLAN_MAPS_ROOT']}"

    return repo_root, dataset_root


def build_two_stage_token_remap(logs_dir, anchor_idx):
    """Map every frame token in a two-stage scene to that scene's anchor token.

    For two-stage metadata, SceneLoader is keyed by the anchor token while CSV
    tokens may reference any frame in the short synthetic sequence. This lets us
    resolve CSV/user tokens to the anchor token the loader actually exposes.
    """
    import pickle

    remap = {}
    for pkl_path in Path(logs_dir).glob("*.pkl"):
        try:
            with open(pkl_path, "rb") as f:
                frames = pickle.load(f)
        except Exception:
            continue
        if not isinstance(frames, list) or len(frames) <= anchor_idx:
            continue
        anchor_frame = frames[anchor_idx]
        if not isinstance(anchor_frame, dict) or "token" not in anchor_frame:
            continue
        anchor_token = str(anchor_frame["token"])
        remap[anchor_token] = anchor_token
        for frame in frames:
            if isinstance(frame, dict) and "token" in frame:
                remap[str(frame["token"])] = anchor_token
    return remap


def load_agent_and_scenes(repo_root, dataset_root, checkpoint_path, agent_name, experiment_name, split, num_scene_tokens=None):
    from navsim.agents.abstract_agent import AbstractAgent
    from navsim.common.dataloader import SceneLoader, SceneFilter, SensorConfig, MetricCacheLoader

    config_dir = repo_root / "navsim/planning/script/config/training"
    agent_config_dir = repo_root / "navsim/planning/script/config/common/agent"

    two_stage_splits = ("navhard_two_stage", "private_test_hard_two_stage")
    is_two_stage = split in two_stage_splits
    base_split = "navtest" if is_two_stage else split

    overrides = [
        f"train_test_split={base_split}",
        f"experiment_name={experiment_name}",
    ]

    dinov2_weights = str(repo_root / "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors")

    agent_overrides = [
        f"checkpoint_path='{str(checkpoint_path)}'",
        "config.shared_refiner=false",
        "lr_args=null",
        "scheduler_args=null",
        "progress_bar=false",
        "config.refiner_ls_values=0.0",
        "config.image_backbone.focus_front_cam=false",
        "config.one_token_per_traj=true",
        "config.refiner_num_heads=1",
        "config.tf_d_model=256",
        "config.tf_d_ffn=1024",
        "config.area_pred=false",
        "config.agent_pred=false",
        "config.ref_num=4",
        f"config.image_backbone.model_weights={dinov2_weights}",
        f"config.lidar_backbone.model_weights={dinov2_weights}",
        "config.noc=1", "config.dac=1", "config.ddc=0",
        "config.ttc=5", "config.ep=5", "config.comfort=2",
        "loss.prev_weight=0.0",
        "batch_size=null",
    ]
    if num_scene_tokens is not None:
        agent_overrides.append(f"config.num_scene_tokens={int(num_scene_tokens)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=str(agent_config_dir)):
        agent_cfg = compose(config_name=agent_name, overrides=agent_overrides)

    agent: AbstractAgent = instantiate(agent_cfg)
    agent.initialize()

    cache_path = Path(os.environ["NAVSIM_EXP_ROOT"]) / "train_metric_cache"
    if cache_path.exists():
        agent.test_metric_cache_paths = MetricCacheLoader(cache_path).metric_cache_paths

    agent.b2d = False
    agent.ray = False

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="default_training", overrides=overrides)

    token_remap = {}
    num_future_override = None
    if is_two_stage:
        ts_dir = Path(dataset_root) / "dataset" / split
        cfg.navsim_log_path = str(ts_dir / "openscene_meta_datas")
        cfg.sensor_blobs_path = str(ts_dir / "sensor_blobs")
        num_future_override = 1
        token_remap = build_two_stage_token_remap(Path(cfg.navsim_log_path), anchor_idx=3)
        print(f"[two-stage] Using data path: logs={cfg.navsim_log_path}")
        print(f"[two-stage] Using data path: sensor={cfg.sensor_blobs_path}")
        print(f"[two-stage] Token remap entries: {len(token_remap)} (num_future_frames=1)")

    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if num_future_override is not None:
        scene_filter.num_future_frames = num_future_override
    if is_two_stage:
        # navtest scene_filter carries fixed log_names AND tokens whitelists that
        # match none of the token-named two-stage pkls; clearing both (plus
        # max_scenes/has_route) lets SceneLoader load all two-stage scenes keyed
        # by their anchor (index-3) token.
        scene_filter.log_names = None
        scene_filter.tokens = None
        scene_filter.max_scenes = None
        scene_filter.has_route = False

    inference_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config()
    )

    viz_sensor_config = SensorConfig(
        cam_f0=[3], cam_l0=[3], cam_l1=[3], cam_l2=[3],
        cam_r0=[3], cam_r1=[3], cam_r2=[3], cam_b0=[3],
        lidar_pc=False,
    )
    viz_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=viz_sensor_config
    )

    return agent, inference_loader, viz_loader, token_remap


# ==============================================================================
# Model Inspection Helpers
# ==============================================================================

def _get_drivor_model(agent):
    """Extract DrivoRModel from agent, handling various wrapping patterns."""
    import torch.nn as nn
    if hasattr(agent, '_drivor_model'):
        return agent._drivor_model
    if hasattr(agent, 'model'):
        return agent.model
    for attr in dir(agent):
        child = getattr(agent, attr)
        if isinstance(child, nn.Module) and hasattr(child, 'image_backbone'):
            return child
    raise RuntimeError(
        "Cannot find DrivoRModel in agent. Available attrs: " +
        str([a for a in dir(agent) if not a.startswith('__')])
    )


def _get_vit_model(drivor_model):
    """Extract raw ViT backbone from DrivoRModel."""
    if hasattr(drivor_model, 'image_backbone'):
        encoder = drivor_model.image_backbone
        if hasattr(encoder, 'model'):
            inner = encoder.model
            if hasattr(inner, 'lora_vit'):
                return inner.lora_vit
            elif hasattr(inner, 'blocks'):
                return inner
    return None


def get_active_camera_order(drivor_model):
    """Return active camera names in the same order used by feature extraction."""
    active = []
    cfg = getattr(drivor_model, "_config", None)
    if cfg is None:
        return active
    for cam_name in CAMERA_TOKEN_ORDER:
        cam_cfg = cfg.get(cam_name, [])
        if cam_cfg is not None and len(cam_cfg) > 0:
            active.append(cam_name)
    return active


def detect_token_layout(vit_model, N_total, H_p, W_p):
    """Auto-detect token layout: [scene_tokens | CLS | registers | patches].

    Returns dict with prefix, num_scene_tokens, num_cls_tokens, num_reg_tokens,
    reg_start, num_patches.
    """
    num_patches = H_p * W_p
    prefix = N_total - num_patches

    num_reg_tokens = 0
    if hasattr(vit_model, 'reg_token') and vit_model.reg_token is not None:
        num_reg_tokens = vit_model.reg_token.shape[1]

    num_cls_tokens = 0
    if hasattr(vit_model, 'cls_token') and vit_model.cls_token is not None:
        num_cls_tokens = 1

    num_scene_tokens = prefix - num_cls_tokens - num_reg_tokens
    reg_start = num_scene_tokens + num_cls_tokens

    layout = {
        'prefix': prefix,
        'num_scene_tokens': num_scene_tokens,
        'num_cls_tokens': num_cls_tokens,
        'num_reg_tokens': num_reg_tokens,
        'reg_start': reg_start,
        'num_patches': num_patches,
    }
    return layout


def extract_vit_attentions(model, device="cuda"):
    """Register hooks on ViT attention blocks to capture raw attention weights.

    Returns (hooks_list, attn_maps_dict).
    """
    hooks = []
    attn_maps = {}

    def hook_fn(name):
        def hook(module, input, output):
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(
                B, N, 3, module.num_heads, module.head_dim
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            if hasattr(module, 'q_norm'):
                q = module.q_norm(q)
            if hasattr(module, 'k_norm'):
                k = module.k_norm(k)
            q = q * module.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn_maps[name] = attn.detach()
        return hook

    vit_model = _get_vit_model(model)
    if vit_model is None:
        raise RuntimeError("Cannot find ViT backbone in model")

    for i, blk in enumerate(vit_model.blocks):
        name = f"layer_{i}"
        h = blk.attn.register_forward_hook(hook_fn(name))
        hooks.append(h)

    return hooks, attn_maps


def extract_trajectory_attentions(model):
    """Register hooks on trajectory decoder cross-attention layers."""
    hooks = []
    attn_maps = {}

    def hook_fn(name):
        def hook(module, input, output):
            q = input[0]
            kv = input[1] if len(input) > 1 else None
            with torch.no_grad():
                if kv is not None:
                    _, attn_w = module.attn(query=q, key=kv, value=kv, need_weights=True)
                else:
                    _, attn_w = module.attn(query=q, key=q, value=q, need_weights=True)
            attn_maps[name] = attn_w.detach()
        return hook

    if hasattr(model, 'trajectory_decoder') and hasattr(model.trajectory_decoder, 'layers'):
        for i, layer in enumerate(model.trajectory_decoder.layers):
            name = f"traj_cross_layer_{i}"
            h = layer.cross_attn.register_forward_hook(hook_fn(name))
            hooks.append(h)

    return hooks, attn_maps


# ==============================================================================
# Attention Aggregation Methods
# ==============================================================================

def avg_heads(A):
    """Average attention over heads and normalize."""
    A = A.mean(1)
    return A / (A.sum(-1, keepdim=True) + 1e-9)


@torch.no_grad()
def scene_to_patch_last_layer(attn_last, scene_idx, patch_idx, H_p, W_p):
    """Extract scene→patch attention from last layer only."""
    A = avg_heads(attn_last)
    V = A[:, scene_idx][:, :, patch_idx]
    return V.view(V.shape[0], V.shape[1], H_p, W_p)


@torch.no_grad()
def scene_to_patch_shallow_rollout(attn_list, scene_idx, patch_idx, H_p, W_p,
                                   last_k=3, alpha=0.9):
    """Attention rollout over last K layers with residual."""
    mats = []
    for A in attn_list[-last_k:]:
        M = avg_heads(A).clone()
        I = torch.eye(M.size(-1), device=M.device, dtype=M.dtype).expand_as(M)
        M = alpha * M + (1 - alpha) * I
        mats.append(M)

    R = mats[0]
    for M in mats[1:]:
        R = R @ M

    R = R / (R.sum(-1, keepdim=True) + 1e-9)
    V = R[:, scene_idx][:, :, patch_idx]
    return V.view(V.shape[0], V.shape[1], H_p, W_p)


@torch.no_grad()
def scene_to_patch_low_entropy_head(attn_last, scene_idx, patch_idx, H_p, W_p):
    """Select lowest-entropy head per sample (most focused attention)."""
    B, H, N, _ = attn_last.shape
    S2P = attn_last[:, :, scene_idx][:, :, :, patch_idx]
    S2P = S2P / (S2P.sum(-1, keepdim=True) + 1e-9)

    ent = -(S2P.clamp_min(1e-12) * S2P.clamp_min(1e-12).log()).sum(-1)

    best_h = ent.argmin(1)
    S = len(scene_idx)
    out = torch.zeros((B, S, H_p, W_p), device=attn_last.device, dtype=attn_last.dtype)

    for b in range(B):
        sel = S2P[b, best_h[b], torch.arange(S, device=attn_last.device)]
        out[b] = sel.view(S, H_p, W_p)
    return out


# ==============================================================================
# Visualization Functions
# ==============================================================================

def _scene_token_grid(num_tokens):
    """Determine grid layout for scene token visualization.

    Returns (nrows, ncols):
        - 16 tokens → 4×4
        - 32 tokens → 4×8
        - 64 tokens → 8×8
        - other → auto square-ish
    """
    if num_tokens == 16:
        return 4, 4
    elif num_tokens == 32:
        return 4, 8
    elif num_tokens == 64:
        return 8, 8
    else:
        ncols = int(np.ceil(np.sqrt(num_tokens)))
        nrows = int(np.ceil(num_tokens / ncols))
        return nrows, ncols



def _as_numpy_attention(attn_maps):
    if torch.is_tensor(attn_maps):
        return attn_maps.detach().float().cpu().numpy()
    return np.asarray(attn_maps, dtype=np.float32)


def _heatmap_scale(attn_np, mode="global", cam_index=None, percentile=99.5, vmin=0.0, vmax=None):
    if mode == "token":
        return None
    values = attn_np if mode == "global" else attn_np[cam_index]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo = float(np.min(values) if vmin is None else vmin)
    hi = float(vmax) if vmax is not None else float(np.percentile(values, percentile))
    if not np.isfinite(hi) or hi <= lo:
        hi = float(np.max(values)) if values.size else lo + 1.0
    if hi <= lo:
        hi = lo + 1e-9
    return lo, hi


def _normalize_heatmap(hm, scale=None):
    if scale is None:
        lo, hi = float(np.min(hm)), float(np.max(hm))
    else:
        lo, hi = scale
    return np.clip((hm - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _write_attention_stats(attn_maps, output_dir, prefix, ordered_cam_names=None):
    attn_np = _as_numpy_attention(attn_maps)
    B, T = attn_np.shape[:2]
    if ordered_cam_names is None:
        ordered_cam_names = [f"cam_{i}" for i in range(B)]
    path = output_dir / f"{prefix}_attention_stats.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "camera", "token_index", "sum", "mean", "std", "min", "max",
            "p95", "p99", "entropy", "entropy_norm", "top1_mass", "top5pct_mass",
        ])
        for b in range(B):
            cam_name = ordered_cam_names[b] if b < len(ordered_cam_names) else f"cam_{b}"
            for t in range(T):
                flat = attn_np[b, t].reshape(-1).astype(np.float64)
                total = float(flat.sum())
                prob = flat / (total + 1e-12)
                entropy = float(-(prob * np.log(prob + 1e-12)).sum())
                k = max(1, int(np.ceil(flat.size * 0.05)))
                topk = np.partition(flat, -k)[-k:]
                writer.writerow([
                    cam_name, t, total, float(flat.mean()), float(flat.std()),
                    float(flat.min()), float(flat.max()), float(np.percentile(flat, 95)),
                    float(np.percentile(flat, 99)), entropy,
                    entropy / np.log(max(flat.size, 2)), float(flat.max() / (total + 1e-12)),
                    float(topk.sum() / (total + 1e-12)),
                ])
    print(f"Saved stats: {path}")


def visualize_attention_maps(scene_maps, cams, output_dir, S=16, H_img=672, W_img=1148, dpi=150,
                             group_mode=False, group_count=4, selected_scene_tokens=None,
                             heatmap_norm="global", heatmap_percentile=99.5,
                             heatmap_vmin=0.0, heatmap_vmax=None, heatmap_alpha=0.5):
    """Overlay scene-token attention maps on camera images.

    Supports two modes:
      - group_mode=False: each scene token gets its own heatmap, grid auto-arranged
      - group_mode=True:  scene tokens evenly split into `group_count` consecutive groups,
                          summed within each group, producing one heatmap per group.

    Args:
        scene_maps: [num_cams, S, H_p, W_p]  scene-to-patch attention
        cams: dict of camera_name → image tensor
        S: number of scene tokens per camera
        group_mode: if True, group tokens and sum
        group_count: number of groups (only used when group_mode=True)
    """
    PATCH_SIZE = 14
    H_p, W_p = scene_maps.shape[-2:]
    scene_np = _as_numpy_attention(scene_maps)
    global_scale = _heatmap_scale(
        scene_np, mode="global", percentile=heatmap_percentile,
        vmin=heatmap_vmin, vmax=heatmap_vmax,
    ) if heatmap_norm == "global" else None

    mode_suffix = "grouped" if group_mode else "sampled"

    if group_mode:
        # --- Grouped mode: split tokens into groups, sum within each group ---
        G = group_count
        assert S % G == 0, f"Scene token count {S} must be divisible by group_count {G}"
        group_size = S // G
        grouped_scene_maps = torch.stack([
            scene_maps[:, g * group_size : (g + 1) * group_size].sum(dim=1)
            for g in range(G)
        ], dim=1)  # [B, G, H_p, W_p]
        grouped_np = _as_numpy_attention(grouped_scene_maps)
        grouped_global_scale = _heatmap_scale(
            grouped_np, mode="global", percentile=heatmap_percentile,
            vmin=heatmap_vmin, vmax=heatmap_vmax,
        ) if heatmap_norm == "global" else None

        nrows, ncols = 1, G

        for b, (title, cam_img) in enumerate(cams.items()):
            group_maps = grouped_scene_maps[b]

            if torch.is_tensor(cam_img):
                cam_img = cam_img.detach().cpu().permute(1, 2, 0).numpy()
            if cam_img.max() > 1.1:
                cam_img = cam_img / 255.0

            img_resized = cv2.resize(cam_img, (W_img, H_img), interpolation=cv2.INTER_AREA)
            H_crop, W_crop = H_p * PATCH_SIZE, W_p * PATCH_SIZE
            img_cropped = img_resized[:H_crop, :W_crop]

            fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.5))
            if ncols == 1:
                axes = [axes]

            for g in range(G):
                ax = axes[g]
                hm = group_maps[g].detach().float().cpu().numpy()
                scale = grouped_global_scale
                if heatmap_norm == "camera":
                    scale = _heatmap_scale(
                        grouped_np, mode="camera", cam_index=b,
                        percentile=heatmap_percentile, vmin=heatmap_vmin,
                        vmax=heatmap_vmax,
                    )
                elif heatmap_norm == "token":
                    scale = None
                hm = _normalize_heatmap(hm, scale)
                hm_resized = cv2.resize(hm, (W_crop, H_crop), interpolation=cv2.INTER_NEAREST)

                ax.imshow(np.clip(img_cropped, 0, 1))
                ax.imshow(hm_resized, alpha=heatmap_alpha, cmap=_JET_BLUE_YELLOW, vmin=0, vmax=1)
                ax.set_title(f"{title} Group {g} (tokens {g*group_size}-{(g+1)*group_size-1})", fontsize=8)
                ax.axis('off')

            fig.suptitle(f"{title.upper()} — Grouped Scene Token Attention ({G} groups × {group_size} tokens)", fontsize=13)
            fig.tight_layout()
            save_path = output_dir / f"attention_maps_{mode_suffix}_{title}.png"
            if SAVE_INTERMEDIATE_OUTPUTS:
                fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
                print(f"Saved: {save_path}")
            plt.close(fig)

    else:
        # --- Selected-token mode: ONE combined grid ---
        # Layout: col 0 = original RGB image, cols 1.. = grayscale bg + attention overlay,
        #         far right = shared colorbar. Rows = cameras.
        if selected_scene_tokens is None or len(selected_scene_tokens) == 0:
            selected_scene_tokens = list(range(S))
        selected_scene_tokens = [int(i) for i in selected_scene_tokens if 0 <= int(i) < S]
        if len(selected_scene_tokens) == 0:
            selected_scene_tokens = list(range(S))

        cam_items = list(cams.items())
        n_rows = len(cam_items)
        n_tok = len(selected_scene_tokens)
        if n_rows == 0 or n_tok == 0:
            return
        n_cols = n_tok + 1  # +1 for the original RGB column
        H_crop, W_crop = H_p * PATCH_SIZE, W_p * PATCH_SIZE
        cell_w = 2.6
        cell_h = cell_w * H_crop / max(W_crop, 1)
        cbar_w = 0.55  # inches reserved on the right for the colorbar
        fig_w = cell_w * n_cols + cbar_w
        fig_h = cell_h * n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
        axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
        _gray_w = np.array([0.299, 0.587, 0.114], dtype=np.float32)

        for r, (title, cam_img) in enumerate(cam_items):
            maps_b = scene_maps[r]
            if torch.is_tensor(cam_img):
                cam_img = cam_img.detach().cpu().permute(1, 2, 0).numpy()
            cam_img = np.asarray(cam_img, dtype=np.float32)
            if cam_img.max() > 1.1:
                cam_img = cam_img / 255.0
            img_resized = cv2.resize(cam_img, (W_crop, H_crop), interpolation=cv2.INTER_AREA)
            img_resized = np.clip(img_resized, 0, 1)
            gray = np.clip(img_resized @ _gray_w, 0, 1)

            # Column 0: original RGB image (no overlay)
            ax0 = axes[r][0]
            ax0.imshow(img_resized)
            ax0.set_xticks([])
            ax0.set_yticks([])

            for c, token_id in enumerate(selected_scene_tokens):
                ax = axes[r][c + 1]
                hm = maps_b[token_id].detach().float().cpu().numpy()
                scale = global_scale
                if heatmap_norm == "camera":
                    scale = _heatmap_scale(
                        scene_np, mode="camera", cam_index=r,
                        percentile=heatmap_percentile, vmin=heatmap_vmin,
                        vmax=heatmap_vmax,
                    )
                elif heatmap_norm == "token":
                    scale = None
                hm = _normalize_heatmap(hm, scale)
                hm_resized = cv2.resize(hm, (W_crop, H_crop), interpolation=cv2.INTER_LINEAR)
                ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
                ax.imshow(hm_resized, alpha=heatmap_alpha, cmap=_JET_BLUE_YELLOW, vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])

        # Force white subplot borders so no default black frames appear.
        for _ax in axes.reshape(-1):
            for _sp in _ax.spines.values():
                _sp.set_visible(True)
                _sp.set_color("white")
                _sp.set_linewidth(1.0)

        right_edge = 1.0 - (cbar_w + 0.12) / fig_w
        fig.subplots_adjust(left=0.005, right=right_edge, top=0.995, bottom=0.005,
                            wspace=0.03, hspace=0.03)
        cbar_h = 0.88
        cbar_w = (cbar_h * fig_h / 16.0) / fig_w
        cax = fig.add_axes([right_edge + 0.012, 0.06, cbar_w, cbar_h])
        sm = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap=_JET_BLUE_YELLOW)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_visible(True)
        cbar.outline.set_edgecolor("white")
        cbar.outline.set_linewidth(1.0)

        save_path = output_dir / "attention_maps_sampled_all"
        png_path = save_path.with_suffix(".png")
        pdf_path = save_path.with_suffix(".pdf")
        # dpi high enough that each 2.6-inch panel keeps >= native pixels (no downscale)
        save_dpi = int(max(dpi, np.ceil(W_crop / cell_w)))
        if SAVE_INTERMEDIATE_OUTPUTS:
            fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=save_dpi)
            fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02, dpi=save_dpi)
            print(f"Saved: {png_path}")
            print(f"Saved: {pdf_path}")
        plt.close(fig)

        # --- Also emit one PDF per camera at native resolution. Each figure is
        #     a single row: [RGB | scene-token overlays ... | 16-token energy]. ---
        save_per_camera_attention_maps(
            scene_maps, cams, output_dir, selected_scene_tokens,
            heatmap_norm=heatmap_norm, heatmap_percentile=heatmap_percentile,
            heatmap_vmin=heatmap_vmin, heatmap_vmax=heatmap_vmax,
            heatmap_alpha=heatmap_alpha, dpi=dpi,
        )


def save_per_camera_attention_maps(scene_maps, cams, output_dir, selected_scene_tokens,
                                   heatmap_norm="global", heatmap_percentile=99.5,
                                   heatmap_vmin=0.0, heatmap_vmax=None,
                                   heatmap_alpha=0.5, dpi=300):
    """Split the horizontal attention comparison into one PDF per camera.

    Each camera figure is a single row:
        [ original RGB | scene-token overlays ... | mean-over-tokens energy ]
    The original photo/grayscale resolution is preserved (images are drawn at
    the source pixel size; heatmaps are resized up to match). No titles or axis
    ticks are drawn; a 0..1 colorbar is added on the right. The energy panel is
    the mean attention over all scene tokens (the total distribution).
    """
    S = scene_maps.shape[1]
    scene_np = _as_numpy_attention(scene_maps)
    global_scale = _heatmap_scale(
        scene_np, mode="global", percentile=heatmap_percentile,
        vmin=heatmap_vmin, vmax=heatmap_vmax,
    ) if heatmap_norm == "global" else None

    if selected_scene_tokens is None or len(selected_scene_tokens) == 0:
        selected_scene_tokens = list(range(S))
    selected_scene_tokens = [int(i) for i in selected_scene_tokens if 0 <= int(i) < S]
    if len(selected_scene_tokens) == 0:
        selected_scene_tokens = list(range(S))

    energy_maps = scene_maps.mean(1)
    energy_np = _as_numpy_attention(energy_maps)
    energy_scale = _heatmap_scale(
        energy_np[:, None], mode="global", percentile=heatmap_percentile,
        vmin=heatmap_vmin, vmax=heatmap_vmax,
    )

    _gray_w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    cam_items = list(cams.items())
    n_tok = len(selected_scene_tokens)
    n_cols = 1 + n_tok + 1  # RGB + token overlays + total energy
    rendered_panels = {}

    for b, (cam_name, cam_img) in enumerate(cam_items):
        if torch.is_tensor(cam_img):
            cam_img = cam_img.detach().cpu().permute(1, 2, 0).numpy()
        cam_img = np.asarray(cam_img, dtype=np.float32)
        if cam_img.max() > 1.1:
            cam_img = cam_img / 255.0
        cam_img = np.clip(cam_img, 0, 1)
        H_src, W_src = cam_img.shape[:2]
        gray = np.clip(cam_img @ _gray_w, 0, 1)

        cell_w = 2.6
        cell_h = cell_w * H_src / max(W_src, 1)
        cbar_w = 0.55
        fig_w = cell_w * n_cols + cbar_w
        fig_h = cell_h
        fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h))
        axes = np.atleast_1d(axes).reshape(-1)

        # Column 0: original RGB image (native resolution).
        axes[0].imshow(cam_img)
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        maps_b = scene_maps[b]
        for c, token_id in enumerate(selected_scene_tokens):
            ax = axes[1 + c]
            hm = maps_b[token_id].detach().float().cpu().numpy()
            scale = global_scale
            if heatmap_norm == "camera":
                scale = _heatmap_scale(
                    scene_np, mode="camera", cam_index=b,
                    percentile=heatmap_percentile, vmin=heatmap_vmin,
                    vmax=heatmap_vmax,
                )
            elif heatmap_norm == "token":
                scale = None
            hm = _normalize_heatmap(hm, scale)
            hm_resized = cv2.resize(hm, (W_src, H_src), interpolation=cv2.INTER_LINEAR)
            ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
            ax.imshow(hm_resized, alpha=heatmap_alpha, cmap=_JET_BLUE_YELLOW, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

        # Last column: total energy over all scene tokens (like patch-energy fig).
        e = _normalize_heatmap(
            energy_maps[b].detach().float().cpu().numpy(), energy_scale)
        e_resized = cv2.resize(e, (W_src, H_src), interpolation=cv2.INTER_LINEAR)
        ax_e = axes[n_cols - 1]
        ax_e.imshow(e_resized, cmap=_JET_BLUE_YELLOW, vmin=0, vmax=1)
        ax_e.set_xticks([])
        ax_e.set_yticks([])

        # Draw white axes frames so subplot borders are explicit and never black.
        for _ax in axes:
            for _sp in _ax.spines.values():
                _sp.set_visible(True)
                _sp.set_color("white")
                _sp.set_linewidth(1.0)

        right_edge = 1.0 - (cbar_w + 0.03) / fig_w
        fig.subplots_adjust(left=0.005, right=right_edge, top=0.995, bottom=0.005,
                            wspace=0.004, hspace=0.004)
        cbar_h = 0.88
        cbar_w = (cbar_h * fig_h / 16.0) / fig_w
        cax = fig.add_axes([right_edge + 0.004, 0.06, cbar_w, cbar_h])
        sm = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap=_JET_BLUE_YELLOW)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_visible(True)
        cbar.outline.set_edgecolor("white")
        cbar.outline.set_linewidth(1.0)

        save_path = output_dir / f"attention_maps_sampled_{cam_name}"
        pdf_path = save_path.with_suffix(".pdf")
        png_path = save_path.with_suffix(".png")
        # Choose a dpi that keeps each cell at least the source width in pixels
        # so the raster output is not downsampled below native resolution.
        save_dpi = int(max(dpi, np.ceil(W_src / cell_w)))
        panel_buffer = io.BytesIO()
        fig.savefig(panel_buffer, format="png", bbox_inches="tight", pad_inches=0.02, dpi=save_dpi)
        panel_data = np.frombuffer(panel_buffer.getvalue(), dtype=np.uint8)
        rendered_panels[cam_name] = cv2.imdecode(panel_data, cv2.IMREAD_COLOR)
        panel_buffer.close()
        # Always persist a standalone per-view attention PDF (native vector) for each of the 4 views.
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02, dpi=save_dpi)
        print(f"Saved: {pdf_path}")
        if SAVE_INTERMEDIATE_OUTPUTS:
            fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=save_dpi)
            print(f"Saved: {png_path}")
        plt.close(fig)

    # --- Vertically stack the front (cam_f0, top) and back (cam_b0, bottom)
    #     entirely in memory; only the final PDF is persisted. ---
    top = rendered_panels.get("cam_f0")
    bot = rendered_panels.get("cam_b0")
    if top is not None and bot is not None:
        W_stack = max(top.shape[1], bot.shape[1])

        def _pad_w(img, W):
            h, w = img.shape[:2]
            if w >= W:
                return img
            pad = W - w
            left = pad // 2
            right = pad - left
            return cv2.copyMakeBorder(img, 0, 0, left, right,
                                      cv2.BORDER_CONSTANT, value=(255, 255, 255))

        top = _pad_w(top, W_stack)
        bot = _pad_w(bot, W_stack)
        stacked = np.vstack([top, bot])
        if SAVE_INTERMEDIATE_OUTPUTS:
            stack_path = output_dir / "attention_maps_sampled_cam_f0_b0.png"
            cv2.imwrite(str(stack_path), stacked, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"Saved: {stack_path}")

        stack_pdf = output_dir / "attention_maps_sampled_cam_f0_b0.pdf"
        try:
            _rgb = cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB)
            _h, _w = _rgb.shape[:2]
            _pdf_dpi = 300
            _fig = plt.figure(figsize=(_w / _pdf_dpi, _h / _pdf_dpi), dpi=_pdf_dpi)
            _ax = _fig.add_axes([0.0, 0.0, 1.0, 1.0])
            _ax.imshow(_rgb, interpolation="none")
            _ax.set_axis_off()
            _fig.savefig(str(stack_pdf), dpi=_pdf_dpi)
            plt.close(_fig)
            print(f"Saved: {stack_pdf}")
        except Exception as _e:
            print(f"[WARN] Failed to write stacked PDF: {_e}")


def plot_trajectories_by_camera(proposals, A_tr2scene, output_dir, num_cams=4,
                                camera_labels=None,
                                ego_yaw=0.0, title="Trajectory Attention", dpi=150):
    """Color trajectories by dominant camera based on cross-attention to scene tokens."""
    if A_tr2scene.ndim == 3:
        A_tr2scene = A_tr2scene[0]

    num_proposals, seq_len = A_tr2scene.shape
    tokens_per_cam = seq_len // num_cams

    A_tr2scene_cams = A_tr2scene.view(num_proposals, num_cams, tokens_per_cam)
    cam_strength = A_tr2scene_cams.sum(-1)
    cam_strength = cam_strength / (cam_strength.sum(-1, keepdim=True) + 1e-9)
    dominant_cam = cam_strength.argmax(dim=-1).cpu().numpy()
    cam_confidence = cam_strength.max(-1).values.cpu().numpy()

    rot = np.array([
        [np.cos(ego_yaw), -np.sin(ego_yaw)],
        [np.sin(ego_yaw),  np.cos(ego_yaw)],
    ])
    xy = proposals[..., :2].cpu().numpy() @ rot.T

    if camera_labels is None:
        camera_labels = [f"Cam{i}" for i in range(num_cams)]
    camera_colors = [matplotlib.colors.to_hex(plt.cm.tab10(i % 10)) for i in range(num_cams)]
    colors = np.array(camera_colors)[dominant_cam]

    fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(xy.shape[0]):
        x, y = xy[i, :, 0], xy[i, :, 1]
        alpha = 0.4 + 0.6 * cam_confidence[i]
        ax.plot(x, y, color=colors[i], alpha=alpha, lw=2, zorder=2)
        ax.scatter(x[-1], y[-1], color=colors[i], s=10, zorder=3)

    ego_shape = np.array([[-1.0, -0.5], [1.0, -0.5], [1.0, 0.5], [-1.0, 0.5]]) @ rot.T
    ax.add_patch(plt.Polygon(ego_shape, closed=True, color="gray", alpha=0.5, zorder=4))

    handles = [plt.Line2D([0], [0], color=c, lw=3, label=l) for c, l in zip(camera_colors, camera_labels)]
    ax.legend(handles=handles, title="Dominant Camera", loc="upper right")

    pad = 5.0
    ax.set_xlim(xy[..., 0].min() - pad, xy[..., 0].max() + pad)
    ax.set_ylim(xy[..., 1].min() - pad, xy[..., 1].max() + pad)
    ax.set_aspect('equal')
    ax.set_title(title)
    fig.tight_layout()
    save_path = output_dir / "trajectory_camera_alignment.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_patch_energy(scene_maps, cams, output_dir, dpi=150,
                      heatmap_percentile=99.5, heatmap_vmin=0.0, heatmap_vmax=None,
                      heatmap_alpha=0.5):
    """All cameras' scene-token patch energy overlaid on the real camera image,
    in one compact row (+ shared colorbar)."""
    import matplotlib as _mpl
    PATCH_SIZE = 14
    H_p, W_p = scene_maps.shape[-2:]
    energy_maps = scene_maps.mean(1)
    energy_np = _as_numpy_attention(energy_maps)
    scale = _heatmap_scale(
        energy_np[:, None], mode="global", percentile=heatmap_percentile,
        vmin=heatmap_vmin, vmax=heatmap_vmax,
    )
    H_crop, W_crop = H_p * PATCH_SIZE, W_p * PATCH_SIZE
    panels = []
    for b, (cam_name, cam_img) in enumerate(cams.items()):
        if torch.is_tensor(cam_img):
            cam_img = cam_img.detach().cpu().permute(1, 2, 0).numpy()
        cam_img = np.asarray(cam_img, dtype=np.float32)
        if cam_img.max() > 1.1:
            cam_img = cam_img / 255.0
        img_resized = cv2.resize(cam_img, (W_crop, H_crop), interpolation=cv2.INTER_AREA)
        img_resized = np.clip(img_resized, 0, 1)
        hm = energy_maps[b].detach().float().cpu().numpy()
        hm = _normalize_heatmap(hm, scale)
        hm_resized = cv2.resize(hm, (W_crop, H_crop), interpolation=cv2.INTER_LINEAR)
        panels.append((cam_name, img_resized, hm_resized))
    if not panels:
        return
    n = len(panels)
    fig_w = 3.4 * n + 1.0
    fig_h = 3.0
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h))
    if n == 1:
        axes = [axes]
    for ax, (cam_name, img_resized, hm_resized) in zip(axes, panels):
        gray = np.clip(img_resized @ np.array([0.299, 0.587, 0.114], dtype=np.float32), 0, 1)
        ax.imshow(gray, cmap='gray', vmin=0, vmax=1)
        ax.imshow(hm_resized, alpha=heatmap_alpha, cmap=_JET_BLUE_YELLOW, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for _sp in ax.spines.values():
            _sp.set_visible(True)
            _sp.set_color("white")
            _sp.set_linewidth(1.0)
    fig.subplots_adjust(left=0.015, right=0.9, top=0.99, bottom=0.14, wspace=0.06)
    cbar_h = 0.83
    cbar_w = (cbar_h * fig_h / 16.0) / fig_w
    cax = fig.add_axes([0.915, 0.14, cbar_w, cbar_h])
    sm = _mpl.cm.ScalarMappable(norm=_mpl.colors.Normalize(vmin=0, vmax=1), cmap=_JET_BLUE_YELLOW)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_edgecolor("white")
    cbar.outline.set_linewidth(1.0)
    save_path = output_dir / "patch_energy_all"
    png_path = save_path.with_suffix(".png")
    pdf_path = save_path.with_suffix(".pdf")
    # each panel is ~3.4 inch wide; keep >= native pixels
    save_dpi = int(max(dpi, np.ceil(W_crop / 3.4)))
    fig.savefig(png_path, bbox_inches="tight", dpi=save_dpi)
    fig.savefig(pdf_path, bbox_inches="tight", dpi=save_dpi)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


_CAM_DISPLAY = {
    "cam_f0": "Front cam.",
    "cam_b0": "Back cam.",
    "cam_l0": "Front-left cam.",
    "cam_l1": "Left cam.",
    "cam_l2": "Rear-left cam.",
    "cam_r0": "Front-right cam.",
    "cam_r1": "Right cam.",
    "cam_r2": "Rear-right cam.",
}


def _pretty_cam_name(cam_name):
    return _CAM_DISPLAY.get(cam_name, cam_name)


def plot_scene_similarity(scene_maps, cams, output_dir, dpi=150):
    """All cameras' scene-token cosine-similarity matrices in one compact row (+ shared colorbar)."""
    panels = []
    for b, (cam_name, _) in enumerate(cams.items()):
        sf = scene_maps[b].flatten(1)
        sf = sf / (sf.norm(dim=-1, keepdim=True) + 1e-9)
        sim = (sf @ sf.T).cpu()
        panels.append((cam_name, sim))
    if not panels:
        return
    n = len(panels)
    fig_w = 3.0 * n + 1.0
    fig_h = 3.6
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h))
    if n == 1:
        axes = [axes]
    im = None
    for ax, (cam_name, sim) in zip(axes, panels):
        im = ax.imshow(sim, cmap=_JET_BLUE_YELLOW, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for _sp in ax.spines.values():
            _sp.set_visible(True)
            _sp.set_color("white")
            _sp.set_linewidth(1.0)
        mean_sim = sim.triu(1).mean().item()
        print(f"{cam_name} mean inter-scene similarity: {mean_sim:.4f}")
    fig.subplots_adjust(left=0.015, right=0.9, top=0.99, bottom=0.12, wspace=0.06)
    cbar_h = 0.87
    cbar_w = (cbar_h * fig_h / 16.0) / fig_w
    cax = fig.add_axes([0.915, 0.12, cbar_w, cbar_h])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([])
    cbar.outline.set_edgecolor("white")
    cbar.outline.set_linewidth(1.0)
    save_path = output_dir / "scene_similarity_all"
    png_path = save_path.with_suffix(".png")
    pdf_path = save_path.with_suffix(".pdf")
    sim_dpi = int(max(dpi, 300))
    fig.savefig(png_path, bbox_inches="tight", dpi=sim_dpi)
    fig.savefig(pdf_path, bbox_inches="tight", dpi=sim_dpi)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

def plot_layer_divergence(attentions_list, cams, output_dir, cam_attn_indices,
                          scene_idx=None, S=16, dpi=150):
    """Inter-scene-token cosine similarity vs. ViT layer depth.

    Lower similarity in deeper layers → scene tokens are differentiating.
    """
    for b, (title, _) in enumerate(cams.items()):
        cam_attn_idx = cam_attn_indices[b]
        divergence = []
        for l, A in enumerate(attentions_list):
            if scene_idx is not None:
                scenes = A.mean(1)[cam_attn_idx, scene_idx, :]
            else:
                scenes = A.mean(1)[cam_attn_idx, :S, :]
            sim = F.cosine_similarity(scenes[:, None, :], scenes[None, :, :], dim=-1)
            mean_offdiag = sim.triu(1).mean().item()
            divergence.append(mean_offdiag)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(divergence, marker='o')
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Mean inter-scene cosine sim")
        ax.set_title(f"{title.upper()} — Scene Token Differentiation vs Depth")
        ax.grid(True)
        fig.tight_layout()
        save_path = output_dir / f"layer_divergence_{title}.png"
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        print(f"Saved: {save_path}")


# ==============================================================================
# Main
# ==============================================================================

def init_distributed():
    """Read torchrun env vars. Returns (rank, world_size, local_rank).

    No process group is created: each rank independently processes a shard of
    tokens and writes its own image files, so no collective ops are needed.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def load_tokens_from_csv(csv_path):
    """Read a CSV (viz_bev_traj style) and return (tokens, token_to_score_text)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "token" not in df.columns:
        raise ValueError(f"CSV is missing required 'token' column: {csv_path}")
    df["token"] = df["token"].astype(str).str.strip()
    df = df[df["token"] != ""]
    tokens = list(dict.fromkeys(df["token"].astype(str).tolist()))
    token_to_score_text = {}
    if "score" in df.columns:
        for _, row in df.iterrows():
            tok = str(row["token"])
            if tok not in token_to_score_text:
                token_to_score_text[tok] = str(row["score"])
    return tokens, token_to_score_text


def resolve_token_output_dir(base_output_dir, token):
    """Return per-token folder in map_*_attention/<agent>/<token> style."""
    base = Path(base_output_dir)
    if base.name == token:
        return base
    return base / token


def build_token_list(args, inference_loader, token_remap=None):
    """Resolve the list of tokens to visualize from --token, --csv, or random."""
    token_remap = token_remap or {}

    def _remap(toks):
        if not token_remap:
            return list(toks)
        return list(dict.fromkeys(token_remap.get(t, t) for t in toks))

    if args.token is not None:
        return _remap([args.token])
    if getattr(args, "csv", None):
        tokens, _ = load_tokens_from_csv(args.csv)
        tokens = _remap(tokens)
        available = set(inference_loader.tokens)
        missing = [t for t in tokens if t not in available]
        if missing:
            print(f"[WARN] {len(missing)}/{len(tokens)} CSV tokens not found in split; skipping them.")
        return [t for t in tokens if t in available]
    return [str(np.random.choice(inference_loader.tokens))]


def visualize_single_token(token, output_dir, args, agent, inference_loader, viz_loader):
    print(f"Visualizing Token: {token}")

    scene = viz_loader.get_scene_from_token(token)
    agent_input = inference_loader.get_agent_input_from_token(token)

    features = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(agent_input))
    features = {k: v.unsqueeze(0).to(args.device) for k, v in features.items()}

    drivor_model = _get_drivor_model(agent)

    vit_hooks, vit_attn_maps = extract_vit_attentions(drivor_model, device=args.device)
    traj_hooks, traj_attn_maps = extract_trajectory_attentions(drivor_model)

    with torch.no_grad():
        predictions = agent.forward(features)

    img_attentions = []
    num_layers = len(vit_attn_maps)
    for i in range(num_layers):
        key = f"layer_{i}"
        if key in vit_attn_maps:
            img_attentions.append(vit_attn_maps[key])

    for h in vit_hooks:
        h.remove()
    for h in traj_hooks:
        h.remove()

    if len(img_attentions) == 0:
        raise RuntimeError("No attention maps captured from ViT backbone")

    print(f"Captured {len(img_attentions)} ViT attention layers, shape: {img_attentions[0].shape}")

    proposals = predictions["proposals"][0]

    vit_model = _get_vit_model(drivor_model)
    if vit_model is None:
        raise RuntimeError("Cannot find ViT model for grid-size inference")

    inferred_grid = getattr(vit_model.patch_embed, "grid_size", None)
    if inferred_grid is None:
        H_p, W_p = args.H_p, args.W_p
        print(f"[WARN] Cannot infer patch grid from ViT, fallback to args: H_p={H_p}, W_p={W_p}")
    else:
        H_p, W_p = int(inferred_grid[0]), int(inferred_grid[1])
        if (H_p, W_p) != (args.H_p, args.W_p):
            print(f"[INFO] Overriding --H_p/--W_p ({args.H_p}, {args.W_p}) with ViT grid ({H_p}, {W_p})")

    N_total = img_attentions[0].shape[-1]
    layout = detect_token_layout(vit_model, N_total, H_p, W_p)

    prefix = layout['prefix']
    S = layout['num_scene_tokens']
    R = layout['num_reg_tokens']
    reg_start = layout['reg_start']

    if S <= 0:
        raise RuntimeError(
            f"Invalid scene token count inferred: S={S}. "
            f"N_total={N_total}, H_p={H_p}, W_p={W_p}, prefix={prefix}"
        )

    print(f"Token layout (auto-detected): "
          f"scene_tokens={S} (indices 0-{S-1}), "
          f"cls={layout['num_cls_tokens']}, "
          f"registers={R} (indices {reg_start}-{reg_start+R-1}), "
          f"patches={layout['num_patches']} (indices {prefix}-{N_total-1}), "
          f"total={N_total}")

    # Scene tokens are always at positions 0..S-1
    scene_idx = torch.arange(0, S, device=img_attentions[0].device)
    patch_idx = torch.arange(prefix, N_total, device=img_attentions[0].device)

    # Extract scene → patch attention
    if args.aggregation == "last":
        scene_maps = scene_to_patch_last_layer(img_attentions[-1], scene_idx, patch_idx, H_p, W_p)
    elif args.aggregation == "shallow":
        scene_maps = scene_to_patch_shallow_rollout(
            img_attentions, scene_idx, patch_idx, H_p, W_p,
            last_k=args.shallow_last_k, alpha=args.shallow_alpha
        )
    else:
        scene_maps = scene_to_patch_low_entropy_head(img_attentions[-1], scene_idx, patch_idx, H_p, W_p)

    print(f"Scene maps shape (all active cams): {scene_maps.shape}")

    active_cam_order = get_active_camera_order(drivor_model)
    if len(active_cam_order) == 0:
        raise RuntimeError("No active cameras found in drivor_model._config")

    cam_to_attn_idx = {cam_name: i for i, cam_name in enumerate(active_cam_order)}
    preferred_viz_cams = [cam for cam in DEFAULT_VIZ_CAMS if cam in cam_to_attn_idx]
    viz_cam_names = preferred_viz_cams if len(preferred_viz_cams) > 0 else active_cam_order[:4]

    cams = {}
    scene_maps_viz_list = []
    cam_attn_indices = []
    for cam_name in viz_cam_names:
        cam_obj = getattr(scene.frames[3].cameras, cam_name, None)
        if cam_obj is None or cam_obj.image is None:
            print(f"[WARN] Missing image for {cam_name}, skip this camera")
            continue
        cams[cam_name] = cam_obj.image
        cam_idx = cam_to_attn_idx[cam_name]
        cam_attn_indices.append(cam_idx)
        scene_maps_viz_list.append(scene_maps[cam_idx])

    if len(scene_maps_viz_list) == 0:
        raise RuntimeError("No valid camera images found for visualization")

    scene_maps_viz = torch.stack(scene_maps_viz_list, dim=0)
    print(f"Visualizing cameras: {list(cams.keys())} with attn indices: {cam_attn_indices}")

    if "attention_maps" in args.viz_types:
        mode_str = f"grouped ({args.group_count} groups)" if args.group_mode else "individual"
        print(f"Generating scene attention maps (method={args.aggregation}, mode={mode_str}, norm={args.heatmap_norm})...")
        if SAVE_INTERMEDIATE_OUTPUTS:
            _write_attention_stats(scene_maps_viz, output_dir, "scene2patch", ordered_cam_names=list(cams.keys()))
        visualize_attention_maps(
            scene_maps_viz,
            cams,
            output_dir,
            S=S,
            dpi=args.dpi,
            group_mode=args.group_mode,
            group_count=args.group_count,
            selected_scene_tokens=args.selected_scene_tokens,
            heatmap_norm=args.heatmap_norm,
            heatmap_percentile=args.heatmap_percentile,
            heatmap_vmin=args.heatmap_vmin,
            heatmap_vmax=args.heatmap_vmax,
            heatmap_alpha=args.heatmap_alpha,
        )

    if SAVE_INTERMEDIATE_OUTPUTS and "trajectory_camera" in args.viz_types:
        print("Generating trajectory-camera alignment...")
        traj_cross_keys = sorted([k for k in traj_attn_maps if k.startswith("traj_cross_")])
        if len(traj_cross_keys) > 0:
            last_traj_attn = traj_attn_maps[traj_cross_keys[-1]]
            A_tr2scene = last_traj_attn[0]
            plot_trajectories_by_camera(proposals, A_tr2scene, output_dir,
                                        num_cams=drivor_model.num_cams,
                                        camera_labels=active_cam_order,
                                        dpi=args.dpi)
        else:
            print("WARNING: No trajectory attention captured, skipping trajectory_camera viz")

    if SAVE_INTERMEDIATE_OUTPUTS and "patch_energy" in args.viz_types:
        print("Generating patch energy maps...")
        plot_patch_energy(
            scene_maps_viz, cams, output_dir, dpi=args.dpi,
            heatmap_percentile=args.heatmap_percentile,
            heatmap_vmin=args.heatmap_vmin,
            heatmap_vmax=args.heatmap_vmax,
        )

    if SAVE_INTERMEDIATE_OUTPUTS and "scene_similarity" in args.viz_types:
        print("Generating scene token similarity matrices...")
        plot_scene_similarity(scene_maps_viz, cams, output_dir, dpi=args.dpi)

    if SAVE_INTERMEDIATE_OUTPUTS and "layer_divergence" in args.viz_types:
        print("Generating layer divergence plots...")
        plot_layer_divergence(
            img_attentions,
            cams,
            output_dir,
            cam_attn_indices=cam_attn_indices,
            scene_idx=scene_idx,
            S=S,
            dpi=args.dpi,
        )

    print(f"\nAll visualizations saved to: {output_dir}")


def main():
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        args.device = f"cuda:{local_rank}"

    repo_root, dataset_root = setup_environment(args)
    checkpoint_path = Path(args.checkpoint_path)

    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "results/scene_attention_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repo root:      {repo_root}")
    print(f"Dataset root:   {dataset_root}")
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Agent name:     {args.agent_name}")
    print(f"Output dir:     {output_dir}")
    print(f"Aggregation:    {args.aggregation}")
    print(f"Viz types:      {args.viz_types}")

    print("Loading model...")
    agent, inference_loader, viz_loader, token_remap = load_agent_and_scenes(
        repo_root, dataset_root, checkpoint_path, args.agent_name, args.experiment_name, args.split,
        num_scene_tokens=args.num_scene_tokens
    )
    agent.cuda(args.device).eval()
    print("Model loaded.")

    tokens = build_token_list(args, inference_loader, token_remap=token_remap)
    if world_size > 1:
        _all_n = len(tokens)
        tokens = tokens[rank::world_size]
        if rank == 0:
            print(f"[INFO] Sharding {_all_n} tokens across {world_size} ranks")
        print(f"[RANK {rank}] local tokens: {len(tokens)}")
    if len(tokens) == 0:
        print("[WARN] No tokens to visualize.")
        return
    base_output_dir = output_dir
    print(f"Total tokens to visualize: {len(tokens)}")
    for i, token in enumerate(tokens):
        tok_out = resolve_token_output_dir(base_output_dir, token)
        tok_out.mkdir(parents=True, exist_ok=True)
        print(f"\n[{i+1}/{len(tokens)}] Token: {token} -> {tok_out}")
        try:
            visualize_single_token(token, tok_out, args, agent, inference_loader, viz_loader)
        except Exception as e:
            import traceback
            print(f"[WARN] token {token} failed: {e}")
            traceback.print_exc()
    print(f"\nAll visualizations saved under: {base_output_dir}")



if __name__ == "__main__":
    main()
