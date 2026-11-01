"""
# For NavTest
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 viz/viz_reg2patch_attention_single.py \
    --checkpoint_path /path/to/your/checkpoint.ckpt \
    --agent_name more_ccs \
    --experiment_name map_register_attention \
    --split navtest \
    --csv /path/to/your/tokens.csv \
    --output_dir /path/to/your/output_dir \
    --dpi 400

# For NavHard Test
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 viz/viz_reg2patch_attention_single.py \
    --checkpoint_path /path/to/your/checkpoint.ckpt \
    --agent_name more_ccs \
    --experiment_name map_register_attention \
    --num_scene_tokens 32 \
    --split navhard_two_stage \
    --csv /path/to/your/tokens.csv \
    --output_dir /path/to/your/output_dir \
    --dpi 300
"""
import os
import sys
import argparse
import csv
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Jet-style colormap truncated so values in [0, 1] map from blue to bright yellow
# (without entering the red tail of full jet).
_JET_BLUE_YELLOW = matplotlib.colors.LinearSegmentedColormap.from_list(
    "jet_blue_yellow", plt.get_cmap("jet")(np.linspace(0.0, 0.74, 256))
)
import cv2
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate


def parse_args():
    parser = argparse.ArgumentParser(description="DrivoR Attention Map Visualization")

    parser.add_argument("--repo_root", type=str,
                        default=str(Path(__file__).resolve().parents[1]),
                        help="Path to drivoR repository root")
    parser.add_argument("--dataset_root", type=str,
                        default=str(Path(__file__).resolve().parents[2]),
                        help="Path to dataset root directory")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint file")
    parser.add_argument("--agent_name", type=str, default="drivoR",
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
                        choices=["attention_maps"],
                        help="Which visualizations to run (only the combined reg2patch grid)")

    parser.add_argument("--num_registers", type=int, default=16,
                        help="Number of register tokens (R)")
    parser.add_argument("--prefix", type=int, default=21,
                        help="Number of prefix tokens (registers + specials)")
    parser.add_argument("--H_p", type=int, default=48,
                        help="Patch grid height")
    parser.add_argument("--W_p", type=int, default=82,
                        help="Patch grid width")
    parser.add_argument("--shallow_last_k", type=int, default=3,
                        help="Number of last layers for shallow rollout")
    parser.add_argument("--shallow_alpha", type=float, default=0.9,
                        help="Alpha for shallow rollout residual connection")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save figures (if not set, uses repo_root/results/attention_viz)")
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
    parser.add_argument("--selected_registers", type=int, nargs="+", default=[0, 4, 8, 12],
                        help="Register-token indices shown as columns in the combined grid")

    return parser.parse_args()


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
    # matches the composed config and the checkpoint's model architecture.
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

    # Two-stage splits (navhard_two_stage / private_test_hard_two_stage) have no
    # Hydra train_test_split config. Load a standard split for config structure,
    # then point the data paths at the two-stage dataset directory.
    two_stage_splits = ("navhard_two_stage", "private_test_hard_two_stage")
    is_two_stage = split in two_stage_splits
    base_split = "navtest" if is_two_stage else split

    overrides = [
        f"train_test_split={base_split}",
        f"experiment_name={experiment_name}",
    ]

    dinov2_weights = str(repo_root / "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors")

    agent_overrides = [
        f"checkpoint_path={str(checkpoint_path)}",
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
    # Some two-stage synthetic scenes have roadblock_ids=None; disable route filtering.
    scene_filter.has_route = False
    if is_two_stage:
        # navtest scene_filter carries fixed log_names AND tokens whitelists that
        # match none of the token-named two-stage pkls. Without clearing both,
        # SceneLoader loads 0 logs (log_names) or filters every scene out
        # (tokens), so every CSV token is reported missing. Clear both to load
        # all two-stage scenes keyed by their anchor (index-3) token.
        scene_filter.log_names = None
        scene_filter.tokens = None

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


def avg_heads(A):
    A = A.mean(1)
    return A / (A.sum(-1, keepdim=True) + 1e-9)


@torch.no_grad()
def reg_to_patch_last_layer(attn_last, reg_idx, patch_idx, H_p, W_p):
    A = avg_heads(attn_last)
    V = A[:, reg_idx][:, :, patch_idx]
    return V.view(V.shape[0], V.shape[1], H_p, W_p)


@torch.no_grad()
def reg_to_patch_shallow_rollout(attn_list, reg_idx, patch_idx, H_p, W_p,
                                 last_k=3, alpha=0.9):
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
    V = R[:, reg_idx][:, :, patch_idx]
    return V.view(V.shape[0], V.shape[1], H_p, W_p)


@torch.no_grad()
def reg_to_patch_low_entropy_head(attn_last, reg_idx, patch_idx, H_p, W_p):
    B, H, N, _ = attn_last.shape
    R2P = attn_last[:, :, reg_idx][:, :, :, patch_idx]
    R2P = R2P / (R2P.sum(-1, keepdim=True) + 1e-9)

    ent = -(R2P.clamp_min(1e-12) * R2P.clamp_min(1e-12).log()).sum(-1)

    best_h = ent.argmin(1)
    out = torch.zeros((B, len(reg_idx), H_p, W_p), device=attn_last.device, dtype=attn_last.dtype)

    for b in range(B):
        sel = R2P[b, best_h[b], torch.arange(len(reg_idx), device=attn_last.device)]
        out[b] = sel.view(len(reg_idx), H_p, W_p)
    return out


# Feature builder camera stacking order (from drivor_features.py _get_camera_feature)
FEATURE_CAMERA_ORDER = ["cam_f0", "cam_b0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2"]



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


def _save_clean(fig, save_path, dpi):
    """Save a figure as text-free PNG + vector PDF (for LaTeX/Overleaf)."""
    st = getattr(fig, "_suptitle", None)
    if st is not None:
        st.set_text("")
    for _txt in list(fig.texts):
        _txt.set_visible(False)
    save_path = Path(save_path)
    png_path = save_path.with_suffix(".png")
    pdf_path = save_path.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def visualize_attention_maps(reg_maps, cams, output_dir, R=16, H_img=672, W_img=1148, dpi=150,
                             ordered_cam_names=None, selected_registers=None,
                             heatmap_norm="global", heatmap_percentile=99.5, heatmap_vmin=0.0,
                             heatmap_vmax=None, heatmap_alpha=0.5):
    """Overlay register-token attention on camera images as ONE combined grid.

    Layout mirrors the scene2patch "sampled_all" figure: rows = cameras,
    column 0 = original RGB, columns 1.. = grayscale background + register-token
    attention overlay, far right = one shared colorbar. Uses the truncated-hot
    (_HOT_YELLOW) colormap for consistency with scene2patch.

    Only the combined grid is written (reg2patch_attention_maps_sampled_all.{png,pdf}).
    No per-camera / per-token images are produced.

    Args:
        reg_maps: attention maps of shape (B, R, H_p, W_p) where B = num_cams
        cams: dict of camera images keyed by cam name
        ordered_cam_names: list of camera names matching reg_maps batch order
        selected_registers: register-token indices shown as columns
    """
    PATCH_SIZE = 14
    H_p, W_p = reg_maps.shape[-2:]
    B = reg_maps.shape[0]

    if ordered_cam_names is None:
        ordered_cam_names = [f"cam_{i}" for i in range(B)]

    reg_np = _as_numpy_attention(reg_maps)
    global_scale = _heatmap_scale(
        reg_np, mode="global", percentile=heatmap_percentile,
        vmin=heatmap_vmin, vmax=heatmap_vmax,
    ) if heatmap_norm == "global" else None

    if selected_registers is None or len(selected_registers) == 0:
        selected_registers = list(range(R))
    selected_registers = [int(i) for i in selected_registers if 0 <= int(i) < R]
    if len(selected_registers) == 0:
        selected_registers = list(range(R))

    # Cameras present in both the attention batch order and the image dict.
    cam_items = [(b, cn) for b, cn in enumerate(ordered_cam_names) if cn in cams]
    n_cams = len(cam_items)
    n_tok = len(selected_registers)
    if n_cams == 0 or n_tok == 0:
        print("[WARN] No cameras/registers available for the combined grid")
        return

    # Horizontal layout: rows = register tokens, columns = cameras.
    n_rows = n_tok
    n_cols = n_cams
    H_crop, W_crop = H_p * PATCH_SIZE, W_p * PATCH_SIZE
    cell_w = 2.6
    cell_h = cell_w * H_crop / max(W_crop, 1)
    cbar_w = 0.55  # inches reserved on the right for the colorbar
    fig_w = cell_w * n_cols + cbar_w
    fig_h = cell_h * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
    for r, reg_id in enumerate(selected_registers):
        for c, (b, cam_name) in enumerate(cam_items):
            ax = axes[r][c]
            hm = reg_maps[b][reg_id].detach().float().cpu().numpy()
            scale = global_scale
            if heatmap_norm == "camera":
                scale = _heatmap_scale(
                    reg_np, mode="camera", cam_index=b,
                    percentile=heatmap_percentile, vmin=heatmap_vmin,
                    vmax=heatmap_vmax,
                )
            elif heatmap_norm == "token":
                scale = None
            hm = _normalize_heatmap(hm, scale)
            hm_resized = cv2.resize(hm, (W_crop, H_crop), interpolation=cv2.INTER_LINEAR)
            ax.imshow(hm_resized, cmap=_JET_BLUE_YELLOW, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

    right_edge = 1.0 - (cbar_w + 0.12) / fig_w
    fig.subplots_adjust(left=0.005, right=right_edge, top=0.995, bottom=0.005,
                        wspace=0.03, hspace=0.03)
    cax = fig.add_axes([right_edge + 0.012, 0.06, 0.10 / fig_w * cell_w, 0.88])
    sm = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap=_JET_BLUE_YELLOW)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(True)

    save_path = output_dir / "reg2patch_attention_maps_sampled_all"
    png_path = save_path.with_suffix(".png")
    pdf_path = save_path.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=dpi)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

def plot_trajectories_by_camera(proposals, A_tr2reg, output_dir, num_cams=4,
                                ego_yaw=0.0, title="Trajectory Attention", dpi=150):
    if A_tr2reg.ndim == 3:
        A_tr2reg = A_tr2reg[0]

    num_proposals, seq_len = A_tr2reg.shape
    tokens_per_cam = seq_len // num_cams

    A_tr2reg_cams = A_tr2reg.view(num_proposals, num_cams, tokens_per_cam)

    cam_strength = A_tr2reg_cams.sum(-1)
    cam_strength = cam_strength / (cam_strength.sum(-1, keepdim=True) + 1e-9)
    dominant_cam = cam_strength.argmax(dim=-1).cpu().numpy()
    cam_confidence = cam_strength.max(-1).values.cpu().numpy()

    rot = np.array([
        [np.cos(ego_yaw), -np.sin(ego_yaw)],
        [np.sin(ego_yaw),  np.cos(ego_yaw)],
    ])
    xy = proposals[..., :2].cpu().numpy() @ rot.T

    camera_labels = ["Front", "Back", "Left", "Right"]
    camera_colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3"]
    colors = np.array(camera_colors)[dominant_cam]

    fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(xy.shape[0]):
        x, y = xy[i, :, 0], xy[i, :, 1]
        alpha = 0.4 + 0.6 * cam_confidence[i]
        ax.plot(x, y, color=colors[i], alpha=alpha, lw=2, zorder=2)
        ax.scatter(x[-1], y[-1], color=colors[i], s=10, zorder=3)

    ego_shape = np.array([[-1.0, -0.5], [1.0, -0.5], [1.0, 0.5], [-1.0, 0.5]]) @ rot.T
    ax.add_patch(plt.Polygon(ego_shape, closed=True, color="gray", alpha=0.5, zorder=4))

    pad = 5.0
    ax.set_xlim(xy[..., 0].min() - pad, xy[..., 0].max() + pad)
    ax.set_ylim(xy[..., 1].min() - pad, xy[..., 1].max() + pad)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_path = output_dir / "trajectory_camera_alignment"
    _save_clean(fig, save_path, dpi)
    plt.close(fig)


def plot_patch_energy(reg_maps, cams, output_dir, dpi=150, ordered_cam_names=None,
                      heatmap_percentile=99.5, heatmap_vmin=0.0, heatmap_vmax=None):
    """All cameras' patch energy maps in one compact row (+ shared colorbar)."""
    B = reg_maps.shape[0]
    if ordered_cam_names is None:
        ordered_cam_names = [f"cam_{i}" for i in range(B)]
    energy_maps = reg_maps.mean(1)
    energy_np = _as_numpy_attention(energy_maps)
    scale = _heatmap_scale(
        energy_np[:, None], mode="global", percentile=heatmap_percentile,
        vmin=heatmap_vmin, vmax=heatmap_vmax,
    )
    panels = []
    for b, cam_name in enumerate(ordered_cam_names):
        if cam_name not in cams:
            continue
        panels.append((cam_name, energy_maps[b].cpu()))
    if not panels:
        return
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n + 1.0, 3.0))
    if n == 1:
        axes = [axes]
    im = None
    for ax, (cam_name, pe) in zip(axes, panels):
        im = ax.imshow(pe, cmap=_JET_BLUE_YELLOW, vmin=scale[0], vmax=scale[1])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(left=0.015, right=0.9, top=0.99, bottom=0.14, wspace=0.06)
    cax = fig.add_axes([0.915, 0.14, 0.013, 0.83])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([])
    save_path = output_dir / "patch_energy_all"
    png_path = save_path.with_suffix(".png")
    pdf_path = save_path.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    fig.savefig(pdf_path, bbox_inches="tight")
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


def plot_register_similarity(reg_maps, cams, output_dir, dpi=150, ordered_cam_names=None):
    """All cameras' register cosine-similarity matrices in one compact row (+ shared colorbar)."""
    B = reg_maps.shape[0]
    if ordered_cam_names is None:
        ordered_cam_names = [f"cam_{i}" for i in range(B)]
    panels = []
    for b, cam_name in enumerate(ordered_cam_names):
        if cam_name not in cams:
            continue
        rf = reg_maps[b].flatten(1)
        rf = rf / (rf.norm(dim=-1, keepdim=True) + 1e-9)
        sim = (rf @ rf.T).cpu()
        panels.append((cam_name, sim))
    if not panels:
        return
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n + 1.0, 3.6))
    if n == 1:
        axes = [axes]
    im = None
    for ax, (cam_name, sim) in zip(axes, panels):
        im = ax.imshow(sim, cmap=_JET_BLUE_YELLOW, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        mean_sim = sim.triu(1).mean().item()
        print(f"{cam_name} mean inter-register similarity: {mean_sim:.4f}")
    fig.subplots_adjust(left=0.015, right=0.9, top=0.99, bottom=0.12, wspace=0.06)
    cax = fig.add_axes([0.915, 0.12, 0.013, 0.87])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([])
    save_path = output_dir / "register_similarity_all"
    png_path = save_path.with_suffix(".png")
    pdf_path = save_path.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

def plot_layer_divergence(attentions_list, cams, output_dir, reg_idx=None, R=16, dpi=150,
                          ordered_cam_names=None):
    B = attentions_list[0].shape[0] if len(attentions_list) > 0 else 0
    if ordered_cam_names is None:
        ordered_cam_names = [f"cam_{i}" for i in range(B)]
    for b, cam_name in enumerate(ordered_cam_names):
        if cam_name not in cams:
            continue
        divergence = []
        for l, A in enumerate(attentions_list):
            if reg_idx is not None:
                regs = A.mean(1)[b, reg_idx, :]
            else:
                regs = A.mean(1)[b, :R, :]
            sim = F.cosine_similarity(regs[:, None, :], regs[None, :, :], dim=-1)
            mean_offdiag = sim.triu(1).mean().item()
            divergence.append(mean_offdiag)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(divergence, marker='o')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True)
        fig.tight_layout()
        save_path = output_dir / f"layer_divergence_{cam_name}"
        _save_clean(fig, save_path, dpi)
        plt.close(fig)


def _get_drivor_model(agent):
    if hasattr(agent, '_drivor_model'):
        return agent._drivor_model
    if hasattr(agent, 'model'):
        return agent.model
    for attr in dir(agent):
        child = getattr(agent, attr)
        if isinstance(child, nn.Module) and hasattr(child, 'image_backbone'):
            return child
    raise RuntimeError("Cannot find DrivoRModel in agent. Available attrs: " + str([a for a in dir(agent) if not a.startswith('__')]))


def _get_vit_model(drivor_model):
    if hasattr(drivor_model, 'image_backbone'):
        encoder = drivor_model.image_backbone
        if hasattr(encoder, '_get_vit_model'):
            return encoder._get_vit_model()
        elif hasattr(encoder, 'model'):
            inner = encoder.model
            if hasattr(inner, 'lora_vit'):
                return inner.lora_vit
            elif hasattr(inner, 'blocks'):
                return inner
    return None


def detect_token_layout(vit_model, N_total, H_p, W_p):
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
    hooks = []
    attn_maps = {}

    def hook_fn(name):
        def hook(module, input, output):
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
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
    """Resolve the list of tokens to visualize from --token, --csv, or random.

    For two-stage splits, token_remap maps CSV/user frame tokens to the anchor
    token the SceneLoader is keyed by.
    """
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

    # Try to use model-returned attentions (more reliable than hook-based recomputation)
    if "image_backbone_attentions" in predictions:
        print("Using model-returned image_backbone_attentions (preferred over hooks)")
        img_attentions = predictions["image_backbone_attentions"]
    else:
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

    N_total = img_attentions[0].shape[-1]
    H_p, W_p = args.H_p, args.W_p

    vit_model = _get_vit_model(drivor_model)
    layout = detect_token_layout(vit_model, N_total, H_p, W_p)

    prefix = layout['prefix']
    R = layout['num_reg_tokens']
    reg_start = layout['reg_start']

    print(f"Token layout (auto-detected): "
          f"scene_tokens={layout['num_scene_tokens']}, "
          f"cls={layout['num_cls_tokens']}, "
          f"registers={R} (indices {reg_start}-{reg_start+R-1}), "
          f"patches={layout['num_patches']} (indices {prefix}-{N_total-1}), "
          f"total={N_total}")

    if args.prefix != prefix:
        print(f"NOTE: --prefix={args.prefix} overridden by auto-detected prefix={prefix}")
    if args.num_registers != R:
        print(f"NOTE: --num_registers={args.num_registers} overridden by auto-detected num_registers={R}")

    reg_idx = torch.arange(reg_start, reg_start + R, device=img_attentions[0].device)
    patch_idx = torch.arange(prefix, N_total, device=img_attentions[0].device)

    if args.aggregation == "last":
        reg_maps = reg_to_patch_last_layer(img_attentions[-1], reg_idx, patch_idx, H_p, W_p)
    elif args.aggregation == "shallow":
        reg_maps = reg_to_patch_shallow_rollout(
            img_attentions, reg_idx, patch_idx, H_p, W_p,
            last_k=args.shallow_last_k, alpha=args.shallow_alpha
        )
    else:
        reg_maps = reg_to_patch_low_entropy_head(img_attentions[-1], reg_idx, patch_idx, H_p, W_p)

    # Build cams dict from ALL available cameras in the viz loader scene
    ALL_CAM_NAMES = ["cam_f0", "cam_b0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2"]
    cams = {}
    for cam_name in ALL_CAM_NAMES:
        try:
            cam_obj = getattr(scene.frames[3].cameras, cam_name, None)
            if cam_obj is not None and cam_obj.image is not None:
                cams[cam_name] = cam_obj.image
        except Exception:
            pass

    # Determine actual camera order from the agent's sensor_config.
    # The ViT batch order follows the feature stacking order, which is the sequence
    # of cameras in ALL_CAM_NAMES that have non-empty frame indices in sensor_config.
    B_cam = img_attentions[0].shape[0]
    try:
        sensor_cfg = agent.get_sensor_config()
        ordered_names = [cn for cn in ALL_CAM_NAMES if len(getattr(sensor_cfg, cn, []) or []) > 0]
    except Exception:
        ordered_names = ALL_CAM_NAMES[:B_cam]  # fallback

    if len(ordered_names) != B_cam:
        print(f"[WARN] sensor_config has {len(ordered_names)} cameras but attention batch has {B_cam}")
        # Align to B: if sensor says more cameras, trim; if fewer, fill from ALL_CAM_NAMES
        if len(ordered_names) < B_cam:
            for cn in ALL_CAM_NAMES:
                if cn not in ordered_names:
                    ordered_names.append(cn)
                if len(ordered_names) >= B_cam:
                    break
        else:
            ordered_names = ordered_names[:B_cam]

    # Diagnostic: show camera batch mapping
    cams_in_viz = list(cams.keys())
    print(f"Camera batch mapping (B={B_cam}):")
    for bi in range(B_cam):
        cam_name = ordered_names[bi] if bi < len(ordered_names) else "???"
        in_viz = cam_name in cams
        status = "✓ matched" if in_viz else "✗ NOT in viz cams"
        print(f"  reg_maps[{bi}] = {cam_name} {status}")
    for cam in cams_in_viz:
        if cam not in ordered_names[:B_cam]:
            print(f"  [INFO] {cam} has image but is NOT in model's attention batch (model doesn't process this camera)")

    if "attention_maps" in args.viz_types:
        print(f"Generating attention maps (method={args.aggregation}, norm={args.heatmap_norm})...")
        _write_attention_stats(reg_maps, output_dir, "reg2patch", ordered_cam_names=ordered_names)
        visualize_attention_maps(reg_maps, cams, output_dir, R=R, dpi=args.dpi,
                                 ordered_cam_names=ordered_names,
                                 selected_registers=args.selected_registers,
                                 heatmap_norm=args.heatmap_norm,
                                 heatmap_percentile=args.heatmap_percentile,
                                 heatmap_vmin=args.heatmap_vmin,
                                 heatmap_vmax=args.heatmap_vmax,
                                 heatmap_alpha=args.heatmap_alpha)

    if "trajectory_camera" in args.viz_types:
        print("Generating trajectory-camera alignment...")
        traj_cross_keys = sorted([k for k in traj_attn_maps if k.startswith("traj_cross_")])
        if len(traj_cross_keys) > 0:
            last_traj_attn = traj_attn_maps[traj_cross_keys[-1]]
            A_tr2reg = last_traj_attn[0]
            plot_trajectories_by_camera(proposals, A_tr2reg, output_dir,
                                        num_cams=drivor_model.num_cams, dpi=args.dpi)
        else:
            print("WARNING: No trajectory attention captured, skipping trajectory_camera viz")

    if "patch_energy" in args.viz_types:
        print("Generating patch energy maps...")
        plot_patch_energy(reg_maps, cams, output_dir, dpi=args.dpi,
                          ordered_cam_names=ordered_names,
                          heatmap_percentile=args.heatmap_percentile,
                          heatmap_vmin=args.heatmap_vmin,
                          heatmap_vmax=args.heatmap_vmax)

    if "register_similarity" in args.viz_types:
        print("Generating register similarity matrices...")
        plot_register_similarity(reg_maps, cams, output_dir, dpi=args.dpi,
                                 ordered_cam_names=ordered_names)

    if "layer_divergence" in args.viz_types:
        print("Generating layer divergence plots...")
        plot_layer_divergence(img_attentions, cams, output_dir, reg_idx=reg_idx, R=R, dpi=args.dpi,
                              ordered_cam_names=ordered_names)

    print(f"\nAll visualizations saved to: {output_dir}")


def main():
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        args.device = f"cuda:{local_rank}"

    repo_root, dataset_root = setup_environment(args)
    checkpoint_path = Path(args.checkpoint_path)

    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "results/attention_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repo root: {repo_root}")
    print(f"Dataset root: {dataset_root}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Viz types: {args.viz_types}")

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
