"""
# For NavTest
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 viz/viz_bev_traj.py \
    --csv /path/to/your/tokens.csv \
    --checkpoint /path/to/your/checkpoint.ckpt \
    --output-dir /path/to/your/output_dir \
    --agent-name more_ccs \
    --data-split navtest \
    --save-format gif \
    --vector-format pdf

# For NavHard Test
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 viz/viz_bev_traj.py \
    --csv /path/to/your/tokens.csv \
    --checkpoint /path/to/your/checkpoint.ckpt \
    --output-dir /path/to/your/output_dir \
    --agent-name more_ccs \
    --data-split navhard_two_stage \
    --save-format gif \
    --vector-format pdf
"""
import argparse
import io
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _bootstrap_repo_path_from_argv() -> None:
	raw_args = sys.argv[1:]
	repo_root_arg: Optional[str] = None
	for idx, arg in enumerate(raw_args):
		if arg == "--repo-root" and idx + 1 < len(raw_args):
			repo_root_arg = raw_args[idx + 1]
			break
		if arg.startswith("--repo-root="):
			repo_root_arg = arg.split("=", 1)[1]
			break

	workspace_root = Path(__file__).resolve().parents[2]
	candidates: List[Path] = []
	if repo_root_arg:
		candidates.append(Path(repo_root_arg).expanduser())
	env_repo_root = os.getenv("NAVSIM_DEVKIT_ROOT")
	if env_repo_root:
		candidates.append(Path(env_repo_root).expanduser())
	candidates.extend([
		workspace_root / "ominidrive",
		workspace_root / "MoRE",
		workspace_root / "multidrive",
		workspace_root / "simscale",
	])

	for candidate in candidates:
		navsim_pkg = candidate / "navsim"
		if not navsim_pkg.is_dir():
			continue
		candidate_str = str(candidate.resolve())
		try:
			sys.path.remove(candidate_str)
		except ValueError:
			pass
		sys.path.insert(0, candidate_str)
		break


_bootstrap_repo_path_from_argv()

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import to_hex
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from navsim.common.dataclasses import Trajectory
from navsim.visualization.plots import configure_ax, configure_bev_ax
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.config import TRAJECTORY_CONFIG

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "viz" / "csv" / "drivoR_navhard_two_stage.csv"
DEFAULT_CKPT = WORKSPACE_ROOT / "exp" / "drivoR_exp" / "ke" / "training_drivoR_Nav2_10epochs" / "03.14_12.57" / "lightning_logs" / "version_0" / "checkpoints" / "best-epoch=9-step=13290.ckpt"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "viz" / "output" / "map_bev_traj" / "drivoR"
DEFAULT_DATA_SPLIT = "navhard_two_stage"

# Keep all rendering helpers available, but only persist the final combined PDF.
SAVE_INTERMEDIATE_OUTPUTS = False

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FEATURE_CAMERA_ORDER = [
	"CAM_F0",
	"CAM_B0",
	"CAM_L0",
	"CAM_L1",
	"CAM_L2",
	"CAM_R0",
	"CAM_R1",
	"CAM_R2",
]

# 3x3 layout: center is BEV, surrounding tiles are 8 cameras.
GRID_LAYOUT = {
	"CAM_L0": (0, 0),
	"CAM_F0": (0, 1),
	"CAM_R0": (0, 2),
	"CAM_L1": (1, 0),
	"CAM_R1": (1, 2),
	"CAM_L2": (2, 0),
	"CAM_B0": (2, 1),
	"CAM_R2": (2, 2),
}

PRED_TRAJ_STYLE = {
	"line_color": "#FFD400",  # yellow (model-selected)
	"line_color_alpha": 0.7,
	"line_width": 3.2,
	"line_style": "-",
	"marker": "o",
	"marker_size": 6,
	"marker_edge_color": "#222222",
	"zorder": 5,
}

GT_TRAJ_STYLE = {
	"line_color": "#FFFFFF",  # white (ground truth)
	"line_color_alpha": 0.7,
	"line_width": 3.0,
	"line_style": "--",
	"marker": "o",
	"marker_size": 5,
	"marker_edge_color": "#111111",
	"zorder": 3,
}

PROPOSAL_TRAJ_STYLE = {
	"line_color": "#0000FF",  # blue candidate spray
	"line_color_alpha": 0.5,
	"line_width": 0.6,
	"line_style": "-",
	"marker": "o",
	"marker_size": 2.0,
	"marker_edge_color": "#1971C2",
	"zorder": 1,
}


def resolve_repo_root(agent_name: str, repo_root_override: Optional[Path] = None) -> Path:
	if repo_root_override is not None:
		return Path(repo_root_override)

	env_repo_root = os.getenv("NAVSIM_DEVKIT_ROOT")
	if env_repo_root:
		candidate = Path(env_repo_root)
		if candidate.exists():
			return candidate

	search_order = [
		WORKSPACE_ROOT / "ominidrive",
		WORKSPACE_ROOT / "drivoR",
		WORKSPACE_ROOT / "multidrive",
		WORKSPACE_ROOT / "simscale",
	]
	for candidate in search_order:
		agent_cfg = candidate / "navsim" / "planning" / "script" / "config" / "common" / "agent" / f"{agent_name}.yaml"
		if agent_cfg.exists():
			return candidate

	return REPO_ROOT


def set_default_env(repo_root: Path) -> None:
	os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
	os.environ.setdefault("NUPLAN_MAPS_ROOT", str(WORKSPACE_ROOT / "dataset" / "maps"))
	os.environ.setdefault("OPENSCENE_DATA_ROOT", str(WORKSPACE_ROOT / "dataset"))
	os.environ.setdefault("NAVSIM_EXP_ROOT", str(WORKSPACE_ROOT / "exp" / "drivoR_exp"))
	os.environ.setdefault("NAVSIM_DEVKIT_ROOT", str(repo_root))
	# Scoring agents (GTRS-aug) interpolate ${oc.env:NAVSIM_TRAJPDM_ROOT} in their
	# config; provide a default so instantiate() can resolve it even though the
	# referenced pickles are never loaded at inference time.
	os.environ.setdefault("NAVSIM_TRAJPDM_ROOT", str(WORKSPACE_ROOT / "dataset" / "traj_pdm_v2"))

	# navsim.common.dataclasses caches env vars at import time, so refresh them at runtime.
	try:
		import navsim.common.dataclasses as navsim_dataclasses
		navsim_dataclasses.OPENSCENE_DATA_ROOT = os.environ.get("OPENSCENE_DATA_ROOT")
		navsim_dataclasses.NUPLAN_MAPS_ROOT = os.environ.get("NUPLAN_MAPS_ROOT")
	except Exception:
		pass


def setup_import_path(repo_root: Path) -> None:
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))


def load_cfg_and_agent(args: argparse.Namespace, repo_root: Path):
	config_dir = repo_root / "navsim" / "planning" / "script" / "config" / "pdm_scoring"
	common_config_dir = repo_root / "navsim" / "planning" / "script" / "config" / "common"
	overrides = [
		f"hydra.searchpath=[file://{common_config_dir.as_posix()},pkg://navsim.planning.script.config.common]",
		f"agent={args.agent_name}",
		"dataloader.params.batch_size=1",
		"dataloader.params.num_workers=0",
		"dataloader.params.pin_memory=false",
		"dataloader.params.prefetch_factor=1",
	]
	if args.checkpoint is not None:
		overrides.append(f"agent.checkpoint_path='{args.checkpoint}'")
	if getattr(args, 'num_scene_tokens', None) is not None:
		overrides.append(f"agent.config.num_scene_tokens={args.num_scene_tokens}")

	if GlobalHydra.instance().is_initialized():
		GlobalHydra.instance().clear()

	with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
		cfg = compose(config_name=args.config_name, overrides=overrides)

	# Force local absolute weight paths to avoid cwd-dependent path issues.
	for branch in ["image_backbone", "lidar_backbone"]:
		if branch in cfg.agent.config and "model_weights" in cfg.agent.config[branch]:
			local_weight = Path(str(cfg.agent.config[branch].model_weights))
			if not local_weight.is_absolute():
				local_weight = repo_root / local_weight
			cfg.agent.config[branch].model_weights = str(local_weight)

	# Inference/benchmark safety: neutralize training-only heavy loads so that
	# scoring agents (ZTRS / GTRS) can be constructed without loading multi-GB
	# target pickles (e.g. navtrain_16384.pkl ~30GB via pdm_gt_path) which would
	# OOM-kill the process. Guarded by key existence -> DrivoR configs unaffected.
	try:
		if "pdm_gt_path" in cfg.agent:
			cfg.agent.pdm_gt_path = None
		if "config" in cfg.agent and cfg.agent.config is not None and "training" in cfg.agent.config:
			cfg.agent.config.training = False
	except Exception as _exc:
		print(f"[WARN] could not apply inference-safety agent overrides: {_exc}")

	agent = instantiate(cfg.agent)
	try:
		# Newer code + matching checkpoint: keep original strict load path.
		agent.initialize()
	except RuntimeError as exc:
		# Backward-compatible load for older checkpoints that miss newly added heads.
		if "Missing key(s) in state_dict" not in str(exc):
			raise
		if args.checkpoint is None:
			raise

		ckpt = torch.load(str(args.checkpoint), map_location="cpu")
		state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
		mapped_state = {
			k.replace("agent._drivor_model", "_drivor_model"): v for k, v in state_dict.items()
		}
		incompatible = agent.load_state_dict(mapped_state, strict=False)
		print(
			"[WARN] strict checkpoint load failed, fallback to strict=False. "
			f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}"
		)

	_cfg_obj = getattr(agent, "_config", None)
	if _cfg_obj is not None:
		try:
			_has_momentum_key = "momentum_use_agent_cache" in _cfg_obj
		except TypeError:
			# dataclass-style configs (e.g. ZTRS HydraRLConfig) are not iterable
			_has_momentum_key = hasattr(_cfg_obj, "momentum_use_agent_cache")
		if _has_momentum_key:
			try:
				_cfg_obj.momentum_use_agent_cache = False
			except Exception:
				pass
	return cfg, agent


def load_tokens_and_scores(
	csv_path: Path,
) -> Tuple[List[str], Dict[str, float], Dict[str, str], Dict[str, Dict[str, str]], List[str]]:
	df = pd.read_csv(csv_path)
	if "token" not in df.columns:
		raise ValueError(f"CSV is missing required 'token' column: {csv_path}")
	df["token"] = df["token"].astype(str).str.strip()
	df = df[df["token"] != ""]

	# Keep order and deduplicate tokens to avoid repeated inference.
	tokens = list(dict.fromkeys(df["token"].astype(str).tolist()))
	token_to_score: Dict[str, float] = {}
	token_to_score_text: Dict[str, str] = {}
	raw_metric_cols = [c for c in df.columns if c != "token"]
	raw_metric_out_cols = [f"csv_{c}" for c in raw_metric_cols]
	token_to_raw_metrics: Dict[str, Dict[str, str]] = {}
	if "score" in df.columns:
		for _, row in df.iterrows():
			token = str(row["token"])
			raw_score = row["score"]
			try:
				token_to_score[token] = float(raw_score)
			except Exception:
				token_to_score[token] = float("nan")
			token_to_score_text[token] = str(raw_score)
			if token not in token_to_raw_metrics:
				token_to_raw_metrics[token] = {
					f"csv_{c}": ("" if pd.isna(row[c]) else str(row[c])) for c in raw_metric_cols
				}
	else:
		for token in tokens:
			token_to_score[token] = float("nan")
			token_to_score_text[token] = "nan"
		for _, row in df.iterrows():
			token = str(row["token"])
			if token not in token_to_raw_metrics:
				token_to_raw_metrics[token] = {
					f"csv_{c}": ("" if pd.isna(row[c]) else str(row[c])) for c in raw_metric_cols
				}

	return tokens, token_to_score, token_to_score_text, token_to_raw_metrics, raw_metric_out_cols


def build_dataloader(
	cfg,
	agent,
	tokens: List[str],
	num_future_frames_override: Optional[int] = None,
	include_targets: bool = True,
):
	from navsim.common.dataloader import SceneLoader
	from navsim.planning.training.dataset import Dataset

	scene_filter = instantiate(cfg.train_test_split.scene_filter)
	if num_future_frames_override is not None:
		scene_filter.num_future_frames = int(num_future_frames_override)
	# Clear split-level constraints and filter strictly by CSV tokens.
	scene_filter.log_names = None
	scene_filter.max_scenes = None
	# Some splits may have roadblock_ids=None; disable route filtering to avoid len(None) crashes.
	scene_filter.has_route = False
	scene_filter.tokens = tokens

	scene_loader = SceneLoader(
		data_path=Path(cfg.navsim_log_path),
		sensor_blobs_path=Path(cfg.sensor_blobs_path),
		scene_filter=scene_filter,
		sensor_config=agent.get_sensor_config(),
	)

	dataset = Dataset(
		scene_loader=scene_loader,
		feature_builders=agent.get_feature_builders(),
		target_builders=agent.get_target_builders() if include_targets else [],
		cache_path=None,
		force_cache_computation=False,
		append_token_to_batch=True,
	)

	dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
	return scene_loader, dataloader


def build_viz_scene_loader(cfg, tokens: List[str], num_future_frames_override: Optional[int] = None):
	from navsim.common.dataloader import SceneLoader
	from navsim.common.dataclasses import SensorConfig

	scene_filter = instantiate(cfg.train_test_split.scene_filter)
	if num_future_frames_override is not None:
		scene_filter.num_future_frames = int(num_future_frames_override)
	scene_filter.log_names = None
	scene_filter.max_scenes = None
	scene_filter.has_route = False
	scene_filter.tokens = tokens
	num_history = int(scene_filter.num_history_frames)
	num_future = int(scene_filter.num_future_frames) if scene_filter.num_future_frames is not None else 0
	camera_frame_indices = list(range(max(1, num_history + num_future)))

	viz_sensor_config = SensorConfig(
		cam_f0=camera_frame_indices,
		cam_l0=camera_frame_indices,
		cam_l1=camera_frame_indices,
		cam_l2=camera_frame_indices,
		cam_r0=camera_frame_indices,
		cam_r1=camera_frame_indices,
		cam_r2=camera_frame_indices,
		cam_b0=camera_frame_indices,
		lidar_pc=[],
	)

	return SceneLoader(
		data_path=Path(cfg.navsim_log_path),
		sensor_blobs_path=Path(cfg.sensor_blobs_path),
		scene_filter=scene_filter,
		sensor_config=viz_sensor_config,
	)


def denormalize_image(image_chw: torch.Tensor) -> Image.Image:
	img = image_chw.detach().cpu().numpy().transpose(1, 2, 0)
	img = np.clip((img * STD + MEAN) * 255.0, 0, 255).astype(np.uint8)
	return Image.fromarray(img)


def extract_camera_images(image_tensor: torch.Tensor) -> Dict[str, Image.Image]:
	# [N, C, H, W] or [T, N, C, H, W] (take latest timestep)
	if image_tensor.ndim == 5:
		image_tensor = image_tensor[-1]

	cams: Dict[str, Image.Image] = {}
	num_cam = min(image_tensor.shape[0], len(FEATURE_CAMERA_ORDER))
	for i in range(num_cam):
		cams[FEATURE_CAMERA_ORDER[i]] = denormalize_image(image_tensor[i])
	return cams


def wrap_to_pi(angle_rad: float) -> float:
	return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def extract_ego_heading(scene) -> float:
	anchor_idx = scene.scene_metadata.num_history_frames - 1
	return float(scene.frames[anchor_idx].ego_status.ego_pose[2])


def find_nearest_centerline_heading(scene, search_radius_m: float = 35.0) -> float:
	anchor_idx = scene.scene_metadata.num_history_frames - 1
	ego_pose = scene.frames[anchor_idx].ego_status.ego_pose
	ego_x = float(ego_pose[0])
	ego_y = float(ego_pose[1])

	layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
	proximal = scene.map_api.get_proximal_map_objects(
		layers=layers,
		point=Point2D(ego_x, ego_y),
		radius=search_radius_m,
	)

	best_dist2 = float("inf")
	best_heading = None
	for layer in layers:
		for obj in proximal.get(layer, []):
			if not hasattr(obj, "baseline_path") or obj.baseline_path is None:
				continue
			if not hasattr(obj.baseline_path, "discrete_path"):
				continue
			nodes = obj.baseline_path.discrete_path
			if len(nodes) < 2:
				continue
			for i in range(len(nodes) - 1):
				x0, y0 = float(nodes[i].x), float(nodes[i].y)
				x1, y1 = float(nodes[i + 1].x), float(nodes[i + 1].y)
				dx = x1 - x0
				dy = y1 - y0
				if abs(dx) < 1e-6 and abs(dy) < 1e-6:
					continue
				mx = 0.5 * (x0 + x1)
				my = 0.5 * (y0 + y1)
				dist2 = (mx - ego_x) ** 2 + (my - ego_y) ** 2
				if dist2 < best_dist2:
					best_dist2 = dist2
					best_heading = float(np.arctan2(dy, dx))

	if best_heading is None:
		return float("nan")
	return best_heading


def compute_heading_centerline_angle_deg(scene, search_radius_m: float = 35.0) -> float:
	"""Return |ego heading - nearest lane centerline heading| in degrees."""
	ego_heading = extract_ego_heading(scene)
	centerline_heading = find_nearest_centerline_heading(scene, search_radius_m=search_radius_m)
	if not np.isfinite(centerline_heading):
		return float("nan")
	return float(abs(np.degrees(wrap_to_pi(ego_heading - centerline_heading))))


def compute_centerline_relative_dynamics(
	scene,
	ego_status_last: np.ndarray,
	search_radius_m: float = 35.0,
) -> Tuple[float, float, float, float]:
	centerline_heading = find_nearest_centerline_heading(scene, search_radius_m=search_radius_m)
	if not np.isfinite(centerline_heading):
		return float("nan"), float("nan"), float("nan"), float("nan")

	vx, vy = float(ego_status_last[3]), float(ego_status_last[4])
	ax, ay = float(ego_status_last[5]), float(ego_status_last[6])
	ch = float(np.cos(centerline_heading))
	sh = float(np.sin(centerline_heading))
	v_lon = vx * ch + vy * sh
	v_lat = -vx * sh + vy * ch
	a_lon = ax * ch + ay * sh
	a_lat = -ax * sh + ay * ch
	return float(v_lon), float(v_lat), float(a_lon), float(a_lat)


def get_gif_frame_index(scene, step_limit: int) -> int:
	anchor_idx = scene.scene_metadata.num_history_frames - 1
	return min(anchor_idx + max(0, int(step_limit) - 1), len(scene.frames) - 1)


def extract_camera_images_from_scene(scene, frame_idx: Optional[int] = None) -> Dict[str, Image.Image]:
	anchor_idx = scene.scene_metadata.num_history_frames - 1
	if frame_idx is None:
		frame_idx = anchor_idx
	frame_idx = max(0, min(int(frame_idx), len(scene.frames) - 1))
	frame_cameras = scene.frames[frame_idx].cameras
	cams: Dict[str, Image.Image] = {}

	for cam_name in FEATURE_CAMERA_ORDER:
		attr = cam_name.lower()
		cam_obj = getattr(frame_cameras, attr)
		if cam_obj.image is not None:
			img = cam_obj.image
			if img.dtype != np.uint8:
				img = np.clip(img, 0, 255).astype(np.uint8)
			cams[cam_name] = Image.fromarray(img)

	return cams


def global_to_local_xy(x: float, y: float, ego_x: float, ego_y: float, ego_yaw: float) -> Tuple[float, float]:
	dx = x - ego_x
	dy = y - ego_y
	cos_yaw = float(np.cos(ego_yaw))
	sin_yaw = float(np.sin(ego_yaw))
	x_local = cos_yaw * dx + sin_yaw * dy
	y_local = -sin_yaw * dx + cos_yaw * dy
	return x_local, y_local


def local_to_global_xy(x_local: float, y_local: float, ego_x: float, ego_y: float, ego_yaw: float) -> Tuple[float, float]:
	cos_yaw = float(np.cos(ego_yaw))
	sin_yaw = float(np.sin(ego_yaw))
	dx = cos_yaw * x_local - sin_yaw * y_local
	dy = sin_yaw * x_local + cos_yaw * y_local
	return ego_x + dx, ego_y + dy


def reproject_traj_between_frames(scene, traj_xyh: np.ndarray, from_frame_idx: int, to_frame_idx: int) -> np.ndarray:
	if traj_xyh.ndim != 2 or traj_xyh.shape[0] == 0:
		return traj_xyh
	from_frame = scene.frames[int(from_frame_idx)]
	to_frame = scene.frames[int(to_frame_idx)]

	from_pose = from_frame.ego_status.ego_pose
	to_pose = to_frame.ego_status.ego_pose
	from_x, from_y, from_yaw = float(from_pose[0]), float(from_pose[1]), float(from_pose[2])
	to_x, to_y, to_yaw = float(to_pose[0]), float(to_pose[1]), float(to_pose[2])

	out = np.asarray(traj_xyh, dtype=np.float32).copy()
	for i in range(out.shape[0]):
		x_rel, y_rel = float(out[i, 0]), float(out[i, 1])
		x_global, y_global = local_to_global_xy(x_rel, y_rel, from_x, from_y, from_yaw)
		x_new, y_new = global_to_local_xy(x_global, y_global, to_x, to_y, to_yaw)
		out[i, 0] = x_new
		out[i, 1] = y_new
		if out.shape[1] >= 3:
			heading_global = float(out[i, 2]) + from_yaw
			out[i, 2] = wrap_to_pi(heading_global - to_yaw)
	return out


def get_overlay_font(size: int) -> ImageFont.FreeTypeFont:
	font_candidates = [
		"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
		"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
	]
	for font_path in font_candidates:
		if Path(font_path).exists():
			return ImageFont.truetype(font_path, size=size)
	return ImageFont.load_default()


def get_local_map_polylines(scene, radius_m: float = 55.0) -> List[np.ndarray]:
	anchor_idx = scene.scene_metadata.num_history_frames - 1
	ego_pose = scene.frames[anchor_idx].ego_status.ego_pose
	ego_x, ego_y, ego_yaw = float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2])

	layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
	proximal = scene.map_api.get_proximal_map_objects(
		layers=layers,
		point=Point2D(ego_x, ego_y),
		radius=radius_m,
	)

	polylines: List[np.ndarray] = []
	for layer in layers:
		for obj in proximal.get(layer, []):
			if not hasattr(obj, "baseline_path") or obj.baseline_path is None:
				continue
			if not hasattr(obj.baseline_path, "discrete_path"):
				continue

			pts_local: List[List[float]] = []
			for node in obj.baseline_path.discrete_path:
				x_local, y_local = global_to_local_xy(float(node.x), float(node.y), ego_x, ego_y, ego_yaw)
				pts_local.append([x_local, y_local])

			if len(pts_local) >= 2:
				polylines.append(np.asarray(pts_local, dtype=np.float32))

	return polylines


def render_trajectory_zoom_inset(
	scene,
	traj_xyh: np.ndarray,
	gt_traj_xyh: np.ndarray,
	proposals: Optional[np.ndarray],
	step_limit: int,
	anchor_idx: int,
	frame_idx: int,
	margin_m: float = 2.5,
) -> Optional[Image.Image]:
	"""Render a zoomed-in BEV crop tightly framed around the trajectory region."""
	traj_for_frame = reproject_traj_between_frames(scene, traj_xyh, anchor_idx, frame_idx)
	gt_for_frame = reproject_traj_between_frames(scene, gt_traj_xyh, anchor_idx, frame_idx)
	pred_poses = np.asarray(traj_for_frame[:step_limit], dtype=np.float32)
	gt_poses = np.asarray(gt_for_frame[:step_limit], dtype=np.float32)

	fwd_vals = [0.0]
	side_vals = [0.0]
	for arr in (pred_poses, gt_poses):
		if arr.ndim == 2 and arr.shape[0] > 0:
			fwd_vals.extend(arr[:, 0].tolist())
			side_vals.extend(arr[:, 1].tolist())
	if len(fwd_vals) <= 1 and len(side_vals) <= 1:
		return None

	fwd_lo, fwd_hi = min(fwd_vals), max(fwd_vals)
	side_lo, side_hi = min(side_vals), max(side_vals)
	cx = 0.5 * (side_lo + side_hi)
	cy = 0.5 * (fwd_lo + fwd_hi)
	span = max(fwd_hi - fwd_lo, side_hi - side_lo) + 2.0 * margin_m
	span = max(span, 8.0)
	half = min(0.5 * span, 18.0)
	x_lo, x_hi = cx - half, cx + half
	y_lo, y_hi = cy - half, cy + half

	fig, ax = plt.subplots(1, 1, figsize=(5, 5))
	add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])

	prop_cfg = dict(TRAJECTORY_CONFIG["agent"])
	prop_cfg.update(PROPOSAL_TRAJ_STYLE)
	if proposals is not None and proposals.ndim == 3 and proposals.shape[0] > 0:
		for i in range(proposals.shape[0]):
			prop_for_frame = reproject_traj_between_frames(scene, proposals[i], anchor_idx, frame_idx)
			prop_poses = np.asarray(prop_for_frame[:step_limit], dtype=np.float32)
			if prop_poses.shape[0] > 1:
				add_trajectory_to_bev_ax(
					ax,
					Trajectory(prop_poses, trajectory_sampling=TrajectorySampling(num_poses=int(prop_poses.shape[0]), interval_length=0.5)),
					prop_cfg,
				)

	pred_cfg = dict(TRAJECTORY_CONFIG["agent"])
	pred_cfg.update(PRED_TRAJ_STYLE)
	if pred_poses.shape[0] > 1:
		add_trajectory_to_bev_ax(
			ax,
			Trajectory(pred_poses, trajectory_sampling=TrajectorySampling(num_poses=int(pred_poses.shape[0]), interval_length=0.5)),
			pred_cfg,
		)

	gt_cfg = dict(TRAJECTORY_CONFIG["human"])
	gt_cfg.update(GT_TRAJ_STYLE)
	if gt_poses.ndim == 2 and gt_poses.shape[0] > 0:
		add_trajectory_to_bev_ax(
			ax,
			Trajectory(gt_poses, trajectory_sampling=TrajectorySampling(num_poses=int(gt_poses.shape[0]), interval_length=0.5)),
			gt_cfg,
		)

	ax.set_aspect("equal")
	ax.set_xlim(x_hi, x_lo)
	ax.set_ylim(y_lo, y_hi)
	configure_ax(ax)
	fig.tight_layout(pad=0.02)

	buf = io.BytesIO()
	fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", pad_inches=0.01)
	buf.seek(0)
	inset = Image.open(buf).convert("RGB")
	buf.close()
	plt.close(fig)
	return inset


def render_paper_style_bev(
	scene,
	traj_xyh: np.ndarray,
	gt_traj_xyh: np.ndarray,
	proposals: Optional[np.ndarray],
	token: str,
	score_text: str,
	ego_status_last: np.ndarray,
	heading_centerline_angle_deg: Optional[float],
	step_limit: int,
	frame_idx: Optional[int] = None,
	vector_path: Optional[Path] = None,
) -> Image.Image:
	anchor_idx = scene.scene_metadata.num_history_frames - 1
	if frame_idx is None:
		frame_idx = anchor_idx
	frame_idx = max(0, min(int(frame_idx), len(scene.frames) - 1))
	fig, ax = plt.subplots(1, 1, figsize=(5, 5))

	add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])

	# Reproject anchor-frame local trajectory into current frame local coordinates.
	traj_for_frame = reproject_traj_between_frames(scene, traj_xyh, anchor_idx, frame_idx)
	gt_for_frame = reproject_traj_between_frames(scene, gt_traj_xyh, anchor_idx, frame_idx)

	pred_poses = np.asarray(traj_for_frame[:step_limit], dtype=np.float32)
	gt_poses = np.asarray(gt_for_frame[:step_limit], dtype=np.float32)

	# 1. Draw all proposal trajectories (faint, behind everything)
	prop_cfg = dict(TRAJECTORY_CONFIG["agent"])
	prop_cfg.update(PROPOSAL_TRAJ_STYLE)
	num_proposals = 0
	if proposals is not None and proposals.ndim == 3 and proposals.shape[0] > 0:
		for i in range(proposals.shape[0]):
			prop_traj = proposals[i]
			prop_for_frame = reproject_traj_between_frames(scene, prop_traj, anchor_idx, frame_idx)
			prop_poses = np.asarray(prop_for_frame[:step_limit], dtype=np.float32)
			if prop_poses.shape[0] > 1:
				prop_traj_obj = Trajectory(
					prop_poses,
					trajectory_sampling=TrajectorySampling(num_poses=int(prop_poses.shape[0]), interval_length=0.5),
				)
				add_trajectory_to_bev_ax(ax, prop_traj_obj, prop_cfg)
				num_proposals += 1

	# 2. Draw selected final trajectory (prominent)
	pred_traj = Trajectory(
		pred_poses,
		trajectory_sampling=TrajectorySampling(num_poses=int(pred_poses.shape[0]), interval_length=0.5),
	)
	pred_cfg = dict(TRAJECTORY_CONFIG["agent"])
	pred_cfg.update(PRED_TRAJ_STYLE)
	add_trajectory_to_bev_ax(ax, pred_traj, pred_cfg)

	# 3. Draw GT trajectory (on top, most visible)
	gt_cfg = dict(TRAJECTORY_CONFIG["human"])
	gt_cfg.update(GT_TRAJ_STYLE)
	if gt_poses.ndim == 2 and gt_poses.shape[0] > 0:
		gt_traj = Trajectory(
			gt_poses,
			trajectory_sampling=TrajectorySampling(num_poses=int(gt_poses.shape[0]), interval_length=0.5),
		)
		add_trajectory_to_bev_ax(ax, gt_traj, gt_cfg)

	configure_bev_ax(ax)
	configure_ax(ax)
	fig.tight_layout(pad=0.02)

	if vector_path is not None:
		fig.savefig(str(vector_path), format=vector_path.suffix.lstrip("."), bbox_inches="tight", pad_inches=0.01)

	buf = io.BytesIO()
	fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", pad_inches=0.01)
	buf.seek(0)
	bev = Image.open(buf).convert("RGB")
	buf.close()
	plt.close(fig)

	inset = render_trajectory_zoom_inset(
		scene=scene,
		traj_xyh=traj_xyh,
		gt_traj_xyh=gt_traj_xyh,
		proposals=proposals,
		step_limit=step_limit,
		anchor_idx=anchor_idx,
		frame_idx=frame_idx,
	)
	if inset is not None:
		bev_w, bev_h = bev.size
		# Keep the zoom inset small so it only shows the magnified trajectory
		# without covering the main BEV field of view.
		target = max(120, int(min(bev_w, bev_h) * 0.24))
		inset = inset.resize((target, target), Image.Resampling.BILINEAR)
		border = 3
		framed = Image.new("RGB", (target + 2 * border, target + 2 * border), (255, 180, 80))
		framed.paste(inset, (border, border))
		margin = 8
		paste_x = bev_w - framed.width - margin
		paste_y = bev_h - framed.height - margin
		bev.paste(framed, (paste_x, paste_y))

	return bev


def draw_bev(
	scene,
	traj_xyh: np.ndarray,
	gt_traj_xyh: np.ndarray,
	proposals: Optional[np.ndarray],
	ego_status_last: np.ndarray,
	heading_centerline_angle_deg: Optional[float],
	token: str,
	score: float,
	score_text: str,
	size: Tuple[int, int],
	step_limit: int,
	frame_idx: Optional[int] = None,
	vector_path: Optional[Path] = None,
) -> Image.Image:
	bev = render_paper_style_bev(
		scene=scene,
		traj_xyh=traj_xyh,
		gt_traj_xyh=gt_traj_xyh,
		proposals=proposals,
		token=token,
		score_text=score_text if score_text else (str(score) if not np.isnan(score) else "nan"),
		ego_status_last=ego_status_last,
		heading_centerline_angle_deg=heading_centerline_angle_deg,
		step_limit=step_limit,
		frame_idx=frame_idx,
		vector_path=vector_path,
	)
	return bev


PROPOSALS_ONLY_STYLE = {
	"line_color": "#0000FF",  # blue candidate spray
	"line_color_alpha": 0.55,
	"line_width": 0.5,
	"line_style": "-",
	"marker": "o",
	"marker_size": 1.8,
	"marker_edge_color": "#1971C2",
	"zorder": 1,
}

PROPOSALS_ONLY_SEL_STYLE = {
	"line_color": "#FFD400",  # yellow (model-selected)
	"line_color_alpha": 0.7,
	"line_width": 3.0,
	"line_style": "-",
	"marker": "o",
	"marker_size": 5,
	"marker_edge_color": "#222222",
	"zorder": 5,
}

PROPOSALS_ONLY_GT_STYLE = {
	"line_color": "#FFFFFF",  # white (ground truth)
	"line_color_alpha": 0.7,
	"line_width": 2.5,
	"line_style": "--",
	"marker": "o",
	"marker_size": 4,
	"marker_edge_color": "#111111",
	"zorder": 3,
}


def project_ego_points_to_camera(points_xy, camera, ground_z, img_shape, eps=1e-3, allow_below=False):
	"""Project ego/lidar-frame ground points (x-forward, y-left) into camera pixels.

	Returns (uv[N,2], in_fov[N]) using the same lidar2img math as navsim.visualization.camera.
	"""
	pts = np.asarray(points_xy, dtype=np.float64)
	if pts.ndim != 2 or pts.shape[0] == 0:
		return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)
	xyz = np.concatenate([pts[:, :2], np.full((pts.shape[0], 1), float(ground_z))], axis=1)
	R = np.asarray(camera.sensor2lidar_rotation, dtype=np.float64)
	t = np.asarray(camera.sensor2lidar_translation, dtype=np.float64).reshape(3)
	K = np.asarray(camera.intrinsics, dtype=np.float64)
	lidar2cam_r = np.linalg.inv(R)
	lidar2cam_t = t @ lidar2cam_r.T
	lidar2cam_rt = np.eye(4)
	lidar2cam_rt[:3, :3] = lidar2cam_r.T
	lidar2cam_rt[3, :3] = -lidar2cam_t
	viewpad = np.eye(4)
	viewpad[: K.shape[0], : K.shape[1]] = K
	lidar2img = viewpad @ lidar2cam_rt.T
	hom = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
	cam = (lidar2img @ hom.T).T
	depth = cam[:, 2]
	in_front = depth > eps
	uv = cam[:, :2] / np.maximum(cam[:, 2:3], eps)
	img_h, img_w = img_shape
	in_fov = in_front & (uv[:, 0] > 0) & (uv[:, 0] < img_w - 1) & (uv[:, 1] > 0)
	if not allow_below:
		in_fov = in_fov & (uv[:, 1] < img_h - 1)
	return uv, in_fov


def densify_polyline_xy(xy, prepend_origin=True, points_per_meter=4.0, min_points=2):
	"""Resample a trajectory polyline to roughly uniform spacing for smooth projection."""
	arr = np.asarray(xy, dtype=np.float64)
	if arr.ndim != 2 or arr.shape[0] == 0:
		return np.zeros((0, 2), dtype=np.float64)
	arr = arr[:, :2]
	if prepend_origin:
		arr = np.vstack([np.zeros((1, 2), dtype=np.float64), arr])
	if arr.shape[0] < 2:
		return arr
	seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
	cum = np.concatenate([[0.0], np.cumsum(seg)])
	total = float(cum[-1])
	if total < 1e-6:
		return arr
	n = max(int(min_points), int(total * points_per_meter))
	s_new = np.linspace(0.0, total, n)
	x = np.interp(s_new, cum, arr[:, 0])
	y = np.interp(s_new, cum, arr[:, 1])
	return np.stack([x, y], axis=1)


def build_front_panorama(cams, f_pan_cap=4000):
	"""Rotation-only cylindrical stitch of front cameras. cams: list of (name, cam_obj).

	Returns (pano_uint8[H,W,3], params) or (None, None).
	"""
	import cv2

	metas = []
	for name, cam in cams:
		R = np.asarray(cam.sensor2lidar_rotation, dtype=np.float64)
		K = np.asarray(cam.intrinsics, dtype=np.float64)
		img = cam.image
		if img.dtype != np.uint8:
			img = np.clip(img, 0, 255).astype(np.uint8)
		h, w = img.shape[:2]
		axis = R @ np.array([0.0, 0.0, 1.0])  # camera optical axis in lidar frame
		az = float(np.arctan2(axis[1], axis[0]))
		fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
		half_l = float(np.arctan2(cx, fx))
		half_r = float(np.arctan2(w - cx, fx))
		metas.append(dict(name=name, R=R, img=img, h=h, w=w, az=az,
			fx=fx, fy=fy, cx=cx, cy=cy, az_min=az - half_l, az_max=az + half_r))
	if len(metas) == 0:
		return None, None

	center_meta = min(metas, key=lambda m: abs(m["az"]))
	f_pan = float(center_meta["fx"])
	theta_max = max(m["az_max"] for m in metas) + np.deg2rad(2.0)
	theta_min = min(m["az_min"] for m in metas) - np.deg2rad(2.0)
	W = int(round(f_pan * (theta_max - theta_min)))
	if W > f_pan_cap:
		f_pan *= f_pan_cap / float(W)
		W = f_pan_cap
	half_up = float(np.arctan2(center_meta["cy"], center_meta["fy"]))
	half_dn = float(np.arctan2(center_meta["h"] - center_meta["cy"], center_meta["fy"]))
	H = int(round(f_pan * (half_up + half_dn)))
	cx_pan = f_pan * theta_max
	cy_pan = f_pan * half_up

	us = np.arange(W, dtype=np.float32)
	rows = np.arange(H, dtype=np.float32)
	theta = (cx_pan - us) / f_pan            # (W,)
	m = (cy_pan - rows) / f_pan              # (H,)
	ray_x = (np.cos(theta)[None, :] * np.ones((H, 1), np.float32)).astype(np.float32)
	ray_y = (np.sin(theta)[None, :] * np.ones((H, 1), np.float32)).astype(np.float32)
	ray_z = (m[:, None] * np.ones((1, W), np.float32)).astype(np.float32)
	ray = np.stack([ray_x, ray_y, ray_z], axis=-1)  # (H,W,3)

	acc = np.zeros((H, W, 3), np.float32)
	wsum = np.zeros((H, W), np.float32)
	# --- pass 1: rotation-only warp of every camera + coverage/feather ---
	sampled_list, inb_list, feather_list = [], [], []
	for md in metas:
		v_cam = ray @ md["R"].astype(np.float32)  # R.T @ ray -> camera frame
		zc = v_cam[..., 2]
		valid = zc > 1e-3
		zc_safe = np.where(valid, zc, 1.0).astype(np.float32)
		px = md["fx"] * (v_cam[..., 0] / zc_safe) + md["cx"]
		py = md["fy"] * (v_cam[..., 1] / zc_safe) + md["cy"]
		inb = valid & (px >= 0) & (px <= md["w"] - 1) & (py >= 0) & (py <= md["h"] - 1)
		map_x = np.where(inb, px, -1).astype(np.float32)
		map_y = np.where(inb, py, -1).astype(np.float32)
		sampled = cv2.remap(md["img"], map_x, map_y, interpolation=cv2.INTER_LINEAR,
			borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
		# smoothstep feather that fades toward each image's left/right border
		d = np.clip(np.minimum(px, md["w"] - 1 - px) / (0.30 * md["w"]), 0.0, 1.0)
		fw = (d * d * (3.0 - 2.0 * d)).astype(np.float32)
		feather = np.where(inb, np.maximum(fw, 1e-3), 0.0).astype(np.float32)
		sampled_list.append(sampled.astype(np.float32))
		inb_list.append(inb)
		feather_list.append(feather)
	# --- pass 2: exposure/colour compensation vs. the centre camera ---
	center_i = int(np.argmin([abs(m["az"]) for m in metas]))
	for i in range(len(metas)):
		if i == center_i:
			continue
		ov_mask = inb_list[i] & inb_list[center_i]
		if int(ov_mask.sum()) > 200:
			for c in range(3):
				ref = sampled_list[center_i][..., c][ov_mask]
				cur = sampled_list[i][..., c][ov_mask]
				mcur = float(cur.mean())
				if mcur > 1e-3:
					gain = float(np.clip(float(ref.mean()) / mcur, 0.6, 1.7))
					sampled_list[i][..., c] *= gain
	# --- pass 3: feather-weighted blend ---
	for i in range(len(metas)):
		w = feather_list[i]
		acc += sampled_list[i] * w[..., None]
		wsum += w
	pano = np.where(wsum[..., None] > 1e-6, acc / np.maximum(wsum[..., None], 1e-6), 30.0)
	pano = np.clip(pano, 0, 255).astype(np.uint8)
	coverage = wsum > 1e-6
	params = dict(f_pan=f_pan, cx_pan=cx_pan, cy_pan=cy_pan, W=W, H=H, metas=metas, coverage=coverage)
	return pano, params


def _camera_px_to_panorama_uv(px, py, meta, params):
	"""Map a camera pixel (px,py) to panorama pixel (u,row) via rotation-only ray direction."""
	dir_cam = np.stack([(px - meta["cx"]) / meta["fx"], (py - meta["cy"]) / meta["fy"], np.ones_like(px)], axis=-1)
	dir_lidar = dir_cam @ meta["R"].T  # R @ dir_cam
	x, y, z = dir_lidar[..., 0], dir_lidar[..., 1], dir_lidar[..., 2]
	horiz = np.hypot(x, y)
	theta = np.arctan2(y, x)
	mm = np.where(horiz > 1e-6, z / np.maximum(horiz, 1e-6), 0.0)
	u = params["cx_pan"] - params["f_pan"] * theta
	row = params["cy_pan"] - params["f_pan"] * mm
	return np.stack([u, row], axis=-1)


def project_polyline_to_panorama(dense_xy, ground_z, ordered_pairs, params, allow_below=False):
	"""Project a trajectory polyline onto the panorama. ordered_pairs: [(meta, cam_obj)] priority order.

	Each point is rendered by the camera whose optical axis points most directly
	at it (smallest azimuth difference) among the cameras that actually see it.
	This keeps the projection spatially consistent and avoids the "bend" that a
	first-match / priority assignment produces near camera FOV boundaries.
	"""
	dense_xy = np.asarray(dense_xy, dtype=np.float64)
	n = dense_xy.shape[0]
	uv = np.full((n, 2), np.nan, dtype=np.float64)
	if n == 0:
		return uv, np.zeros((0,), dtype=bool)
	point_az = np.arctan2(dense_xy[:, 1], dense_xy[:, 0])
	best_ang = np.full(n, np.inf, dtype=np.float64)
	for meta, cam_obj in ordered_pairs:
		pxpy, in_fov = project_ego_points_to_camera(dense_xy, cam_obj, ground_z, (meta["h"], meta["w"]), allow_below=allow_below)
		if pxpy.shape[0] == 0:
			continue
		pano_uv = _camera_px_to_panorama_uv(pxpy[:, 0], pxpy[:, 1], meta, params)
		# Angular distance between each point and this camera's optical axis.
		d_az = np.abs(np.arctan2(np.sin(point_az - meta["az"]), np.cos(point_az - meta["az"])))
		better = in_fov & (d_az < best_ang)
		uv[better] = pano_uv[better]
		best_ang[better] = d_az[better]
	in_view = ~np.isnan(uv[:, 0])
	return uv, in_view


CAMERA_OVERLAY_STYLE = {
    "candidate_color": "#0000FF",
    "candidate_width": 0.8,
    "candidate_alpha": 0.6,
    "candidate_dot_size": 6,
    "selected_color": "#FFD400",
    "selected_width": 3.4,
    "selected_dot_size": 30,
    "selected_alpha": 0.7,
    "gt_color": "#FFFFFF",
    "gt_width": 3.4,
    "gt_dot_size": 30,
    "gt_alpha": 0.7,
    "arrow_scale": 24,
    "ground_fill": "#3b3b3b",
}


def _overlay_direction_arrow(ax, line_uv, valid, color, zorder=6, mutation=24):
    """Draw a forward-pointing arrowhead at the far end of a projected trajectory."""
    idx = np.where(valid)[0]
    if idx.shape[0] < 2:
        return
    end = int(idx[-1])
    start = int(idx[max(0, idx.shape[0] - 4)])
    if start == end:
        return
    x0, y0 = float(line_uv[start, 0]), float(line_uv[start, 1])
    x1, y1 = float(line_uv[end, 0]), float(line_uv[end, 1])
    if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1)):
        return
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=0,
                        mutation_scale=mutation, shrinkA=0, shrinkB=0),
        zorder=zorder,
    )


def _project_vertices_to_camera(verts_xy, cam, ground_z):
    if (cam is None or cam.image is None or cam.intrinsics is None
            or getattr(cam, "sensor2lidar_rotation", None) is None):
        return None, None
    uv, in_fov = project_ego_points_to_camera(
        verts_xy, cam, ground_z, cam.image.shape[:2], allow_below=False)
    return uv, in_fov


def _history_vertex_sources(scene, anchor_idx, verts_xy_anchor, ground_z=0.0, max_back=6):
    """For anchor-ego ground vertices, project them into recent history-frame cameras.
    Returns a list (most recent first) of dicts with the source image and per-vertex uv/in-fov."""
    a_pose = scene.frames[anchor_idx].ego_status.ego_pose
    ax0, ay0, ayaw = float(a_pose[0]), float(a_pose[1]), float(a_pose[2])
    ca, sa = np.cos(ayaw), np.sin(ayaw)
    gx = ax0 + ca * verts_xy_anchor[:, 0] - sa * verts_xy_anchor[:, 1]
    gy = ay0 + sa * verts_xy_anchor[:, 0] + ca * verts_xy_anchor[:, 1]
    sources = []
    start = anchor_idx - 1
    stop = max(-1, anchor_idx - 1 - max_back)
    for hist in range(start, stop, -1):
        frame = scene.frames[hist]
        h_pose = frame.ego_status.ego_pose
        hx0, hy0, hyaw = float(h_pose[0]), float(h_pose[1]), float(h_pose[2])
        ch, sh = np.cos(hyaw), np.sin(hyaw)
        dx = gx - hx0
        dy = gy - hy0
        lx = ch * dx + sh * dy
        ly = -sh * dx + ch * dy
        pts = np.stack([lx, ly], axis=1)
        for cam_name in ("cam_f0", "cam_l0", "cam_r0"):
            cam = getattr(frame.cameras, cam_name, None)
            uv, in_fov = _project_vertices_to_camera(pts, cam, ground_z)
            if uv is None:
                continue
            img = np.asarray(cam.image)
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            sources.append({"img": img, "uv": uv, "in": in_fov})
    return sources


def _warp_triangle_into(src_img, src_tri, dst_tri, canvas, alpha, w_canvas, h_canvas):
    """Affine texture-map one source-image triangle into the destination canvas triangle."""
    import cv2
    dst = np.asarray(dst_tri, dtype=np.float32)
    x, y, w, h = cv2.boundingRect(dst)
    if w <= 0 or h <= 0:
        return
    if x < 0 or y < 0 or x + w > w_canvas or y + h > h_canvas:
        return
    src = np.asarray(src_tri, dtype=np.float32)
    dst_local = dst - np.array([x, y], dtype=np.float32)
    try:
        M = cv2.getAffineTransform(src, dst_local)
    except cv2.error:
        return
    warped = cv2.warpAffine(src_img, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
    tmask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(tmask, np.int32(np.round(dst_local)), 1)
    m = tmask.astype(bool)
    if not m.any():
        return
    canvas[y:y + h, x:x + w][m] = warped[m][:, :3]
    alpha[y:y + h, x:x + w][m] = True


def build_history_ground_fill(scene, anchor_idx, ordered_pairs, params,
                              crop_x0, crop_y0, h_crop, bottom, w_pan, ground_z=0.0):
    """Texture-map history-frame road imagery into the region below the current FOV.
    Returns (rgba_uint8[Hout, W, 4], extent) for imshow, or (None, None)."""
    if anchor_idx <= 0:
        return None, None
    nx, ny = 90, 80
    t = np.linspace(0.0, 1.0, nx + 1)
    xs = 0.3 + (t ** 2) * (45.0 - 0.3)           # denser sampling close to the ego
    ys = np.linspace(-16.0, 16.0, ny + 1)
    vx, vy = np.meshgrid(xs, ys, indexing="ij")
    verts_xy = np.stack([vx.ravel(), vy.ravel()], axis=1)
    uv, in_view = project_polyline_to_panorama(
        verts_xy, ground_z, ordered_pairs, params, allow_below=True)
    uv = uv.copy()
    uv[:, 0] -= crop_x0
    uv[:, 1] -= crop_y0
    u_grid = uv[:, 0].reshape(nx + 1, ny + 1)
    r_grid = uv[:, 1].reshape(nx + 1, ny + 1)
    seen = in_view.reshape(nx + 1, ny + 1)

    sources = _history_vertex_sources(scene, anchor_idx, verts_xy, ground_z=ground_z)
    if len(sources) == 0:
        return None, None
    for s in sources:
        s["in_g"] = s["in"].reshape(nx + 1, ny + 1)
        s["uv_g"] = s["uv"].reshape(nx + 1, ny + 1, 2)

    w_c = int(round(w_pan))
    h_out = int(np.ceil(bottom)) + 2
    canvas = np.zeros((h_out, w_c, 3), dtype=np.uint8)
    alpha = np.zeros((h_out, w_c), dtype=bool)

    for i in range(nx):
        for j in range(ny):
            if not (seen[i, j] and seen[i, j + 1] and seen[i + 1, j + 1] and seen[i + 1, j]):
                continue
            uu = np.array([u_grid[i, j], u_grid[i, j + 1], u_grid[i + 1, j + 1], u_grid[i + 1, j]])
            rr = np.array([r_grid[i, j], r_grid[i, j + 1], r_grid[i + 1, j + 1], r_grid[i + 1, j]])
            if not (np.all(np.isfinite(uu)) and np.all(np.isfinite(rr))):
                continue
            rmin, rmax = float(rr.min()), float(rr.max())
            if rmax < h_crop - 2 or rmin > bottom + 5:
                continue
            if (uu.max() - uu.min()) > 0.5 * w_pan or (rmax - rmin) > 0.6 * (bottom + 1.0):
                continue
            # most recent source seeing all four corners
            chosen = None
            for s in sources:
                ing = s["in_g"]
                if ing[i, j] and ing[i, j + 1] and ing[i + 1, j + 1] and ing[i + 1, j]:
                    chosen = s
                    break
            if chosen is None:
                continue
            su = chosen["uv_g"]
            src = np.array([su[i, j], su[i, j + 1], su[i + 1, j + 1], su[i + 1, j]], dtype=np.float32)
            dst = np.stack([uu, rr], axis=1).astype(np.float32)
            img = chosen["img"]
            _warp_triangle_into(img, src[[0, 1, 2]], dst[[0, 1, 2]], canvas, alpha, w_c, h_out)
            _warp_triangle_into(img, src[[0, 2, 3]], dst[[0, 2, 3]], canvas, alpha, w_c, h_out)

    lo = max(0, int(h_crop))
    if lo < h_out:
        alpha[:lo] = False              # never overwrite the real camera image
    if not alpha.any():
        return None, None
    rgba = np.dstack([canvas, (alpha.astype(np.uint8) * 255)])
    extent = (0.0, float(w_c), float(h_out), 0.0)
    return rgba, extent


def _largest_true_rectangle(mask, max_cells=400000):
    """Largest all-True axis-aligned rectangle in a boolean mask.

    Returns (y0, y1, x0, x1) with y1/x1 exclusive. Large masks are conservatively
    AND-pooled before the histogram search for speed, which guarantees the returned
    rectangle contains no False (uncovered) pixel.
    """
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    sy = sx = 1
    while (H // sy) * (W // sx) > max_cells:
        if (W // sx) >= (H // sy):
            sx += 1
        else:
            sy += 1
    if sy > 1 or sx > 1:
        Hs, Ws = H // sy, W // sx
        small = mask[:Hs * sy, :Ws * sx].reshape(Hs, sy, Ws, sx).all(axis=(1, 3))
    else:
        Hs, Ws, small = H, W, mask
    heights = np.zeros(Ws, dtype=np.int64)
    best_area = 0
    best = (0, 0, 0, 0)
    for r in range(Hs):
        heights = np.where(small[r], heights + 1, 0)
        stack = []  # (start_col, height)
        for c in range(Ws + 1):
            cur = int(heights[c]) if c < Ws else 0
            start = c
            while stack and stack[-1][1] > cur:
                idx, height = stack.pop()
                area = height * (c - idx)
                if area > best_area:
                    best_area = area
                    best = (r - height + 1, r + 1, idx, c)
                start = idx
            stack.append((start, cur))
    y0, y1, x0, x1 = best
    y0 *= sy
    x0 *= sx
    y1 = min(H, y1 * sy)
    x1 = min(W, x1 * sx)
    return y0, y1, x0, x1


# Map scores in [0, 1] through the blue-to-bright-yellow portion of jet.
PDM_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "jet_blue_yellow", plt.get_cmap("jet")(np.linspace(0.0, 0.65, 256))
)


def _pdm_proposal_colors(pdm_scores, n_prop):
    """Map PDM scores to colors via the PDM colormap.

    Returns (colors, vals): colors is a list of hex strings (len n_prop) or
    None when scores are unavailable; vals are the [0, 1] scores used for the
    draw order.
    """
    if pdm_scores is not None and np.asarray(pdm_scores).reshape(-1).shape[0] == n_prop:
        vals = np.clip(np.exp(np.asarray(pdm_scores, dtype=np.float64).reshape(-1)), 0.0, 1.0)
        colors = [to_hex(PDM_CMAP(float(v))) for v in vals]
        return colors, vals
    return None, None


def _save_combined_projection_bev(proj_img, bev_img, pdf_path, png_path, render_dpi=400):
    """Compose projection (left) and BEV (right) into one figure. Saved as PDF
    (vector, native raster embedded) and PNG at native resolution. Either image
    may be None.
    """
    panels = []
    if proj_img is not None:
        panels.append(np.asarray(proj_img))
    if bev_img is not None:
        panels.append(np.asarray(bev_img))
    if len(panels) == 0:
        return
    target_h = max(pl.shape[0] for pl in panels)
    widths_px = [pl.shape[1] * target_h / pl.shape[0] for pl in panels]
    fig_w_in = sum(widths_px) / render_dpi
    fig_h_in = target_h / render_dpi
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    gs = fig.add_gridspec(1, len(widths_px), width_ratios=widths_px, wspace=0.02)
    for k, arr in enumerate(panels):
        ax = fig.add_subplot(gs[k])
        ax.imshow(arr, interpolation="none")
        ax.set_axis_off()
    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004, wspace=0.02)
    fig.savefig(str(pdf_path), bbox_inches="tight", pad_inches=0.02, dpi=render_dpi)
    if SAVE_INTERMEDIATE_OUTPUTS:
        fig.savefig(str(png_path), bbox_inches="tight", pad_inches=0.02, dpi=render_dpi)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    if SAVE_INTERMEDIATE_OUTPUTS:
        print(f"Saved: {png_path}")


def render_proposals_on_panorama(scene, cams, proposals, pdm_scores, traj_xyh, gt_traj_xyh, token, score_text, ground_z=0.0, vector_path=None, fill_anchor_idx=None, prepend_origin=True):
    """Stitch front cameras into one wide panorama and project candidate trajectories."""
    ov = CAMERA_OVERLAY_STYLE
    pano, params = build_front_panorama(cams)
    if pano is None:
        return None
    # Crop to the largest fully-covered rectangle so no uncovered (gray) region remains.
    crop_x0, crop_y0 = 0, 0
    coverage = params.get("coverage")
    if coverage is not None and coverage.any():
        y0, y1, x0, x1 = _largest_true_rectangle(coverage)
        if (y1 - y0) >= 2 and (x1 - x0) >= 2:
            crop_x0, crop_y0 = x0, y0
            pano = pano[y0:y1, x0:x1]
            # panorama tends to be too wide: trim a little extra off both sides
            cw = pano.shape[1]
            dx = int(round(cw * 0.07))
            if dx > 0 and (cw - 2 * dx) >= 8:
                pano = pano[:, dx:cw - dx]
                crop_x0 += dx
    metas = params["metas"]
    name2obj = {name: obj for name, obj in cams}
    ordered = sorted(metas, key=lambda m: abs(m["az"]))
    ordered_pairs = [(md, name2obj[md["name"]]) for md in ordered]

    props = np.asarray(proposals, dtype=np.float64)
    if props.ndim != 3 or props.shape[0] == 0:
        return None
    n_prop = props.shape[0]
    prop_colors, vals = _pdm_proposal_colors(pdm_scores, n_prop)
    order = np.argsort(vals) if vals is not None else np.arange(n_prop)

    H, W = pano.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(max(8.0, W / 200.0), max(3.2, H / 200.0)))
    ax.imshow(pano)

    def _proj_pano(xy, allow_below=False):
        uv, in_view = project_polyline_to_panorama(xy, ground_z, ordered_pairs, params, allow_below=allow_below)
        p = uv.copy()
        p[:, 0] -= crop_x0
        p[:, 1] -= crop_y0
        return p, in_view

    max_row = float(H)

    # 1) candidate proposals: thin, faint fan
    for i in order:
        dxy = densify_polyline_xy(props[i], prepend_origin=prepend_origin)
        p, in_view = _proj_pano(dxy)
        if int(in_view.sum()) < 2:
            continue
        pp = p.copy()
        pp[~in_view] = np.nan
        cand_color = ov["candidate_color"]
        ax.plot(pp[:, 0], pp[:, 1], "-", color=cand_color,
                linewidth=ov["candidate_width"], alpha=ov["candidate_alpha"],
                solid_capstyle="round", zorder=2)
        wp_c, wp_c_view = _proj_pano(np.asarray(props[i], dtype=np.float64)[:, :2])
        if wp_c_view.any():
            cvv = wp_c[wp_c_view]
            ax.scatter(cvv[:, 0], cvv[:, 1], s=ov["candidate_dot_size"], c=cand_color,
                       alpha=ov["candidate_alpha"], edgecolors="none", zorder=3)

    # 2) selected trajectory: bold line + waypoint dots + arrow
    dsel = densify_polyline_xy(traj_xyh, prepend_origin=prepend_origin)
    p_sel, sel_view = _proj_pano(dsel)
    if int(sel_view.sum()) >= 2:
        pp = p_sel.copy()
        pp[~sel_view] = np.nan
        ax.plot(pp[:, 0], pp[:, 1], "-", color=ov["selected_color"],
                linewidth=ov["selected_width"], alpha=ov["selected_alpha"], solid_capstyle="round", zorder=6)
    wp_sel, wp_sel_view = _proj_pano(np.asarray(traj_xyh, dtype=np.float64)[:, :2])
    if wp_sel_view.any():
        vv = wp_sel[wp_sel_view]
        ax.scatter(vv[:, 0], vv[:, 1], s=ov["selected_dot_size"], c=ov["selected_color"],
                   edgecolors="#222222", linewidths=0.6, alpha=ov["selected_alpha"], zorder=7)
        max_row = max(max_row, float(np.nanmax(vv[:, 1])))

    # 3) ground-truth trajectory: prominent line + waypoint dots + arrow
    if gt_traj_xyh is not None and np.asarray(gt_traj_xyh).size > 0:
        dgt = densify_polyline_xy(gt_traj_xyh, prepend_origin=prepend_origin)
        p_gt, gt_view = _proj_pano(dgt)
        if int(gt_view.sum()) >= 2:
            pp = p_gt.copy()
            pp[~gt_view] = np.nan
            ax.plot(pp[:, 0], pp[:, 1], "-", color=ov["gt_color"],
                    linewidth=ov["gt_width"], alpha=ov["gt_alpha"], solid_capstyle="round", zorder=4)
        wp_gt, wp_gt_view = _proj_pano(np.asarray(gt_traj_xyh, dtype=np.float64)[:, :2])
        if wp_gt_view.any():
            vv = wp_gt[wp_gt_view]
            ax.scatter(vv[:, 0], vv[:, 1], s=ov["gt_dot_size"], c=ov["gt_color"],
                       edgecolors="#111111", linewidths=0.6, alpha=ov["gt_alpha"], zorder=5)
            max_row = max(max_row, float(np.nanmax(vv[:, 1])))

    # Waypoints occluded by the ego (below the image) are simply clipped; keep the
    # view within the real image so there is no gray no-image band at the bottom.
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()

    if vector_path is not None:
        fig.savefig(str(vector_path), format=vector_path.suffix.lstrip("."), bbox_inches="tight", pad_inches=0.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=400, bbox_inches="tight", pad_inches=0.0)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    buf.close()
    plt.close(fig)
    return out


def render_proposals_on_camera(
    scene,
    proposals,
    pdm_scores,
    traj_xyh,
    gt_traj_xyh,
    token,
    score_text,
    ego_status_last,
    camview_mode="front",
    ground_z: float = 0.0,
    viewpoint_frame_idx=None,
    vector_path=None,
):
    """Project candidate trajectories onto the front camera(s)."""
    ov = CAMERA_OVERLAY_STYLE
    anchor_idx = scene.scene_metadata.num_history_frames - 1
    anchor_idx = max(0, min(int(anchor_idx), len(scene.frames) - 1))
    view_idx = anchor_idx if viewpoint_frame_idx is None else max(0, min(int(viewpoint_frame_idx), len(scene.frames) - 1))
    if view_idx != anchor_idx:
        # Render from an earlier history frame so the near waypoints of the current
        # trajectory fall inside the field of view; reproject trajectories accordingly.
        _props = np.asarray(proposals, dtype=np.float64)
        if _props.ndim == 3 and _props.shape[0] > 0:
            proposals = np.stack([
                reproject_traj_between_frames(scene, _props[i], anchor_idx, view_idx)
                for i in range(_props.shape[0])
            ])
        traj_xyh = reproject_traj_between_frames(scene, np.asarray(traj_xyh, dtype=np.float64), anchor_idx, view_idx)
        if gt_traj_xyh is not None and np.asarray(gt_traj_xyh).size > 0:
            gt_traj_xyh = reproject_traj_between_frames(scene, np.asarray(gt_traj_xyh, dtype=np.float64), anchor_idx, view_idx)
    # When rendering from an earlier frame, do not connect the first waypoint back to
    # the (history) ego origin, which would draw a spurious straight link.
    prepend_origin_flag = (view_idx == anchor_idx)
    frame_cameras = scene.frames[view_idx].cameras
    if camview_mode == "wide":
        cam_attrs = ["cam_l0", "cam_f0", "cam_r0"]
    else:
        cam_attrs = ["cam_f0"]
    cams = []
    for attr in cam_attrs:
        cam_obj = getattr(frame_cameras, attr, None)
        if cam_obj is not None and cam_obj.image is not None and cam_obj.intrinsics is not None and cam_obj.sensor2lidar_rotation is not None:
            cams.append((attr.upper(), cam_obj))
    if len(cams) == 0:
        return None

    if camview_mode == "wide" and len(cams) >= 2:
        return render_proposals_on_panorama(
            scene, cams, proposals, pdm_scores, traj_xyh, gt_traj_xyh,
            token, score_text, ground_z=ground_z, vector_path=vector_path,
            fill_anchor_idx=view_idx, prepend_origin=prepend_origin_flag,
        )

    props = np.asarray(proposals, dtype=np.float64)
    if props.ndim != 3 or props.shape[0] == 0:
        return None
    n_prop = props.shape[0]
    prop_colors, vals = _pdm_proposal_colors(pdm_scores, n_prop)
    order = np.argsort(vals) if vals is not None else np.arange(n_prop)

    dense_props = [densify_polyline_xy(props[i], prepend_origin=prepend_origin_flag) for i in range(n_prop)]
    dense_sel = densify_polyline_xy(traj_xyh, prepend_origin=prepend_origin_flag)
    raw_sel = np.asarray(traj_xyh, dtype=np.float64)[:, :2]
    dense_gt = None
    raw_gt = None
    if gt_traj_xyh is not None and np.asarray(gt_traj_xyh).size > 0:
        dense_gt = densify_polyline_xy(gt_traj_xyh, prepend_origin=prepend_origin_flag)
        raw_gt = np.asarray(gt_traj_xyh, dtype=np.float64)[:, :2]

    ncam = len(cams)
    fig, axes = plt.subplots(1, ncam, figsize=(6.4 * ncam, 3.6), squeeze=False)
    axes = list(axes[0])
    for ax, (cam_name, cam_obj) in zip(axes, cams):
        img = cam_obj.image
        ih, iw = img.shape[:2]
        ax.imshow(img)
        max_row = float(ih)
        # candidate proposals
        for i in order:
            uv, in_fov = project_ego_points_to_camera(dense_props[i], cam_obj, ground_z, (ih, iw), allow_below=False)
            if int(in_fov.sum()) < 2:
                continue
            p = uv.copy()
            p[~in_fov] = np.nan
            cand_color = ov["candidate_color"]
            ax.plot(p[:, 0], p[:, 1], "-", color=cand_color,
                    linewidth=ov["candidate_width"], alpha=ov["candidate_alpha"],
                    solid_capstyle="round", zorder=2)
            wp_uv_c, wp_fov_c = project_ego_points_to_camera(props[i][:, :2], cam_obj, ground_z, (ih, iw), allow_below=False)
            if wp_fov_c.any():
                cvv = wp_uv_c[wp_fov_c]
                ax.scatter(cvv[:, 0], cvv[:, 1], s=ov["candidate_dot_size"], c=cand_color,
                           alpha=ov["candidate_alpha"], edgecolors="none", zorder=3)
        # selected trajectory
        uv, in_fov = project_ego_points_to_camera(dense_sel, cam_obj, ground_z, (ih, iw), allow_below=False)
        if int(in_fov.sum()) >= 2:
            p = uv.copy()
            p[~in_fov] = np.nan
            ax.plot(p[:, 0], p[:, 1], "-", color=ov["selected_color"],
                    linewidth=ov["selected_width"], alpha=ov["selected_alpha"], solid_capstyle="round", zorder=6)
        wp_uv, wp_fov = project_ego_points_to_camera(raw_sel, cam_obj, ground_z, (ih, iw), allow_below=False)
        if wp_fov.any():
            vv = wp_uv[wp_fov]
            ax.scatter(vv[:, 0], vv[:, 1], s=ov["selected_dot_size"], c=ov["selected_color"],
                       edgecolors="#222222", linewidths=0.6, alpha=ov["selected_alpha"], zorder=7)
            max_row = max(max_row, float(np.nanmax(vv[:, 1])))
        # ground-truth trajectory
        if dense_gt is not None:
            uv, in_fov = project_ego_points_to_camera(dense_gt, cam_obj, ground_z, (ih, iw), allow_below=False)
            if int(in_fov.sum()) >= 2:
                p = uv.copy()
                p[~in_fov] = np.nan
                ax.plot(p[:, 0], p[:, 1], "-", color=ov["gt_color"],
                        linewidth=ov["gt_width"], alpha=ov["gt_alpha"], solid_capstyle="round", zorder=4)
            wp_uv, wp_fov = project_ego_points_to_camera(raw_gt, cam_obj, ground_z, (ih, iw), allow_below=False)
            if wp_fov.any():
                vv = wp_uv[wp_fov]
                ax.scatter(vv[:, 0], vv[:, 1], s=ov["gt_dot_size"], c=ov["gt_color"],
                           edgecolors="#111111", linewidths=0.6, alpha=ov["gt_alpha"], zorder=5)
                max_row = max(max_row, float(np.nanmax(vv[:, 1])))
        ax.set_xlim(0, iw)
        ax.set_ylim(ih, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()

    if vector_path is not None:
        fig.savefig(str(vector_path), format=vector_path.suffix.lstrip("."), bbox_inches="tight", pad_inches=0.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=400, bbox_inches="tight", pad_inches=0.0)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    buf.close()
    plt.close(fig)
    return out


def render_proposals_zoom_inset(
	scene,
	proposals: np.ndarray,
	traj_xyh: np.ndarray,
	gt_traj_xyh: np.ndarray,
	step_limit: int,
	anchor_idx: int,
	frame_idx: int,
	margin_m: float = 2.5,
) -> Optional[Image.Image]:
	"""Zoomed-in BEV crop around the trajectory, styled like the proposals-only view."""
	traj_for_frame = reproject_traj_between_frames(scene, traj_xyh, anchor_idx, frame_idx)
	gt_for_frame = reproject_traj_between_frames(scene, gt_traj_xyh, anchor_idx, frame_idx)
	pred_poses = np.asarray(traj_for_frame[:step_limit], dtype=np.float32)
	gt_poses = np.asarray(gt_for_frame[:step_limit], dtype=np.float32)

	fwd_vals = [0.0]
	side_vals = [0.0]
	for arr in (pred_poses, gt_poses):
		if arr.ndim == 2 and arr.shape[0] > 0:
			fwd_vals.extend(arr[:, 0].tolist())
			side_vals.extend(arr[:, 1].tolist())
	if len(fwd_vals) <= 1 and len(side_vals) <= 1:
		return None

	fwd_lo, fwd_hi = min(fwd_vals), max(fwd_vals)
	side_lo, side_hi = min(side_vals), max(side_vals)
	cx = 0.5 * (side_lo + side_hi)
	cy = 0.5 * (fwd_lo + fwd_hi)
	span = max(fwd_hi - fwd_lo, side_hi - side_lo) + 2.0 * margin_m
	span = max(span, 8.0)
	half = min(0.5 * span, 18.0)
	x_lo, x_hi = cx - half, cx + half
	y_lo, y_hi = cy - half, cy + half

	fig, ax = plt.subplots(1, 1, figsize=(5, 5))
	add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])

	prop_cfg = dict(TRAJECTORY_CONFIG["agent"])
	prop_cfg.update(PROPOSALS_ONLY_STYLE)
	if proposals is not None and proposals.ndim == 3 and proposals.shape[0] > 0:
		for i in range(proposals.shape[0]):
			prop_for_frame = reproject_traj_between_frames(scene, proposals[i], anchor_idx, frame_idx)
			prop_poses = np.asarray(prop_for_frame[:step_limit], dtype=np.float32)
			if prop_poses.shape[0] > 1:
				add_trajectory_to_bev_ax(
					ax,
					Trajectory(prop_poses, trajectory_sampling=TrajectorySampling(num_poses=int(prop_poses.shape[0]), interval_length=0.5)),
					prop_cfg,
				)

	sel_cfg = dict(TRAJECTORY_CONFIG["agent"])
	sel_cfg.update(PROPOSALS_ONLY_SEL_STYLE)
	if pred_poses.shape[0] > 1:
		add_trajectory_to_bev_ax(
			ax,
			Trajectory(pred_poses, trajectory_sampling=TrajectorySampling(num_poses=int(pred_poses.shape[0]), interval_length=0.5)),
			sel_cfg,
		)

	gt_cfg = dict(TRAJECTORY_CONFIG["human"])
	gt_cfg.update(PROPOSALS_ONLY_GT_STYLE)
	if gt_poses.ndim == 2 and gt_poses.shape[0] > 0:
		add_trajectory_to_bev_ax(
			ax,
			Trajectory(gt_poses, trajectory_sampling=TrajectorySampling(num_poses=int(gt_poses.shape[0]), interval_length=0.5)),
			gt_cfg,
		)

	ax.set_aspect("equal")
	ax.set_xlim(x_hi, x_lo)
	ax.set_ylim(y_lo, y_hi)
	configure_ax(ax)
	fig.tight_layout(pad=0.02)

	buf = io.BytesIO()
	fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", pad_inches=0.01)
	buf.seek(0)
	inset = Image.open(buf).convert("RGB")
	buf.close()
	plt.close(fig)
	return inset


def _add_trajectory_zoom_inset(
	ax,
	scene,
	proposals,
	traj_xyh: np.ndarray,
	gt_traj_xyh: np.ndarray,
	step_limit: int,
	anchor_idx: int,
	frame_idx: int,
	margin_m: float = 1.0,
) -> None:
	"""Draw a zoomed-in crop of the trajectory region as an inset in the
	bottom-right corner of ``ax``. The inset is rendered into the same
	matplotlib axes, so it is embedded in the vector (PDF) output as well as
	the raster PNG, keeping the vehicle trajectory clearly visible."""
	traj_for_frame = reproject_traj_between_frames(scene, traj_xyh, anchor_idx, frame_idx)
	gt_for_frame = reproject_traj_between_frames(scene, gt_traj_xyh, anchor_idx, frame_idx)
	pred_poses = np.asarray(traj_for_frame[:step_limit], dtype=np.float32)
	gt_poses = np.asarray(gt_for_frame[:step_limit], dtype=np.float32)

	fwd_vals = [0.0]
	side_vals = [0.0]
	for arr in (pred_poses, gt_poses):
		if arr.ndim == 2 and arr.shape[0] > 0:
			fwd_vals.extend(arr[:, 0].tolist())
			side_vals.extend(arr[:, 1].tolist())
	if len(fwd_vals) <= 1 and len(side_vals) <= 1:
		return

	fwd_lo, fwd_hi = min(fwd_vals), max(fwd_vals)
	side_lo, side_hi = min(side_vals), max(side_vals)
	cx = 0.5 * (side_lo + side_hi)
	cy = 0.5 * (fwd_lo + fwd_hi)
	span = max(fwd_hi - fwd_lo, side_hi - side_lo) + 2.0 * margin_m
	span = max(span, 4.0)
	half = min(0.5 * span, 8.0)
	x_lo, x_hi = cx - half, cx + half
	y_lo, y_hi = cy - half, cy + half

	axins = ax.inset_axes([0.62, 0.02, 0.36, 0.36])
	add_configured_bev_on_ax(axins, scene.map_api, scene.frames[frame_idx])

	prop_cfg = dict(TRAJECTORY_CONFIG["agent"])
	prop_cfg.update(PROPOSALS_ONLY_STYLE)
	if proposals is not None and getattr(proposals, "ndim", 0) == 3 and proposals.shape[0] > 0:
		for i in range(proposals.shape[0]):
			prop_for_frame = reproject_traj_between_frames(scene, proposals[i], anchor_idx, frame_idx)
			prop_poses = np.asarray(prop_for_frame[:step_limit], dtype=np.float32)
			if prop_poses.shape[0] > 1:
				add_trajectory_to_bev_ax(
					axins,
					Trajectory(prop_poses, trajectory_sampling=TrajectorySampling(num_poses=int(prop_poses.shape[0]), interval_length=0.5)),
					prop_cfg,
				)

	sel_cfg = dict(TRAJECTORY_CONFIG["agent"])
	sel_cfg.update(PROPOSALS_ONLY_SEL_STYLE)
	if pred_poses.shape[0] > 1:
		add_trajectory_to_bev_ax(
			axins,
			Trajectory(pred_poses, trajectory_sampling=TrajectorySampling(num_poses=int(pred_poses.shape[0]), interval_length=0.5)),
			sel_cfg,
		)

	gt_cfg = dict(TRAJECTORY_CONFIG["human"])
	gt_cfg.update(PROPOSALS_ONLY_GT_STYLE)
	if gt_poses.ndim == 2 and gt_poses.shape[0] > 0:
		add_trajectory_to_bev_ax(
			axins,
			Trajectory(gt_poses, trajectory_sampling=TrajectorySampling(num_poses=int(gt_poses.shape[0]), interval_length=0.5)),
			gt_cfg,
		)

	axins.set_aspect("equal")
	axins.set_xlim(x_hi, x_lo)
	axins.set_ylim(y_lo, y_hi)
	axins.set_xticks([])
	axins.set_yticks([])
	axins.set_xlabel("")
	axins.set_ylabel("")
	for spine in axins.spines.values():
		spine.set_visible(True)
		spine.set_edgecolor("#FF8C00")
		spine.set_linewidth(2.0)


def render_proposals_only_bev(
	scene,
	proposals: np.ndarray,
	traj_xyh: np.ndarray,
	gt_traj_xyh: np.ndarray,
	token: str,
	score_text: str,
	ego_status_last: np.ndarray,
	heading_centerline_angle_deg: Optional[float],
	step_limit: int,
	stage_idx: int = 0,
	vector_path: Optional[Path] = None,
	pdm_scores: Optional[np.ndarray] = None,
) -> Image.Image:
	"""Render BEV with ALL proposals prominently, plus selected trajectory and GT."""
	anchor_idx = scene.scene_metadata.num_history_frames - 1
	frame_idx = anchor_idx
	fig, ax = plt.subplots(1, 1, figsize=(6, 6))

	add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])

	# Draw all proposals (solid red candidate spray)
	prop_cfg = dict(TRAJECTORY_CONFIG["agent"])
	prop_cfg.update(PROPOSALS_ONLY_STYLE)
	for i in range(proposals.shape[0]):
		prop_traj = proposals[i]
		prop_for_frame = reproject_traj_between_frames(scene, prop_traj, anchor_idx, frame_idx)
		prop_poses = np.asarray(prop_for_frame[:step_limit], dtype=np.float32)
		if prop_poses.shape[0] > 1:
			prop_traj_obj = Trajectory(
				prop_poses,
				trajectory_sampling=TrajectorySampling(num_poses=int(prop_poses.shape[0]), interval_length=0.5),
			)
			add_trajectory_to_bev_ax(ax, prop_traj_obj, prop_cfg)

	# Draw selected trajectory (magenta, thicker)
	traj_for_frame = reproject_traj_between_frames(scene, traj_xyh, anchor_idx, frame_idx)
	pred_poses = np.asarray(traj_for_frame[:step_limit], dtype=np.float32)
	pred_traj = Trajectory(
		pred_poses,
		trajectory_sampling=TrajectorySampling(num_poses=int(pred_poses.shape[0]), interval_length=0.5),
	)
	sel_cfg = dict(TRAJECTORY_CONFIG["agent"])
	sel_cfg.update(PROPOSALS_ONLY_SEL_STYLE)
	add_trajectory_to_bev_ax(ax, pred_traj, sel_cfg)

	# Draw GT (green dashed, on top)
	gt_for_frame = reproject_traj_between_frames(scene, gt_traj_xyh, anchor_idx, frame_idx)
	gt_poses = np.asarray(gt_for_frame[:step_limit], dtype=np.float32)
	if gt_poses.ndim == 2 and gt_poses.shape[0] > 0:
		gt_traj = Trajectory(
			gt_poses,
			trajectory_sampling=TrajectorySampling(num_poses=int(gt_poses.shape[0]), interval_length=0.5),
		)
		gt_cfg = dict(TRAJECTORY_CONFIG["human"])
		gt_cfg.update(PROPOSALS_ONLY_GT_STYLE)
		add_trajectory_to_bev_ax(ax, gt_traj, gt_cfg)

	configure_bev_ax(ax)
	configure_ax(ax)
	stage_label = "Initial" if stage_idx == 0 else f"Refined stage {stage_idx}"
	# ax.set_title(f"[{stage_label}] {proposals.shape[0]} Proposals (orange) + Selected (magenta) + GT (green)", fontsize=11)
	fig.tight_layout(pad=0.02)

	_add_trajectory_zoom_inset(
		ax,
		scene,
		proposals,
		traj_xyh,
		gt_traj_xyh,
		step_limit,
		anchor_idx,
		frame_idx,
	)

	if vector_path is not None:
		fig.savefig(str(vector_path), format=vector_path.suffix.lstrip("."), bbox_inches="tight", pad_inches=0.01)

	buf = io.BytesIO()
	fig.savefig(buf, format="png", dpi=400, bbox_inches="tight", pad_inches=0.01)
	buf.seek(0)
	bev = Image.open(buf).convert("RGB")
	buf.close()
	plt.close(fig)

	return bev


def extract_ego_metrics(ego_status_last: np.ndarray) -> Tuple[float, float]:
	vx, vy = float(ego_status_last[3]), float(ego_status_last[4])
	ax, ay = float(ego_status_last[5]), float(ego_status_last[6])
	speed = float(np.hypot(vx, vy))
	accel = float(np.hypot(ax, ay))
	return speed, accel


def compute_pred_gt_l2_mean(traj_xyh: np.ndarray, gt_traj_xyh: np.ndarray) -> float:
	steps = min(traj_xyh.shape[0], gt_traj_xyh.shape[0])
	if steps <= 0:
		return float("nan")
	delta = traj_xyh[:steps, :2] - gt_traj_xyh[:steps, :2]
	l2 = np.linalg.norm(delta, axis=-1)
	return float(np.mean(l2))


def format_float4(value: float) -> str:
	if value is None or not np.isfinite(value):
		return "nan"
	return f"{float(value):.4f}"


def sanitize_tag(text: str) -> str:
	if text is None:
		return "na"
	clean = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in str(text))
	clean = clean.strip("-")
	return clean if clean else "na"


def extract_command_tag(features: Dict[str, torch.Tensor]) -> Optional[str]:
	candidates = [
		"command",
		"cmd",
		"driving_command",
		"navigation_command",
		"planner_command",
		"high_level_command",
	]
	for key in candidates:
		if key not in features:
			continue
		value = features[key]
		if torch.is_tensor(value):
			t = value.detach().cpu()
			if t.numel() == 0:
				continue
			if t.numel() == 1:
				raw = t.reshape(-1)[0].item()
				if isinstance(raw, (int, np.integer)):
					return f"cmd{int(raw)}"
				if isinstance(raw, (float, np.floating)):
					return f"cmd{float(raw):.4f}"
				return sanitize_tag(str(raw))
			return f"cmd{int(torch.argmax(t.reshape(-1)).item())}"
		return sanitize_tag(str(value))
	return None


def fit_image_to_tile(image: Image.Image, tile_size: Tuple[int, int], bg_color: Tuple[int, int, int]) -> Image.Image:
	tile_w, tile_h = tile_size
	src_w, src_h = image.size
	if src_w <= 0 or src_h <= 0:
		return Image.new("RGB", tile_size, bg_color)

	scale = min(tile_w / float(src_w), tile_h / float(src_h))
	new_w = max(1, int(src_w * scale))
	new_h = max(1, int(src_h * scale))
	resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

	tile = Image.new("RGB", tile_size, bg_color)
	offset_x = (tile_w - new_w) // 2
	offset_y = (tile_h - new_h) // 2
	tile.paste(resized, (offset_x, offset_y))
	return tile


def build_front_pano_image(scene, frame_idx: Optional[int] = None) -> Optional[Image.Image]:
	"""Cylindrically stitch the three front cameras (L0/F0/R0) into one wide panorama.

	Widens the horizontal field of view by merging overlapping views into a single
	seamless image instead of showing three disjoint tiles. Returns None on failure.
	"""
	try:
		if frame_idx is None:
			frame_idx = scene.scene_metadata.num_history_frames - 1
		frame_idx = max(0, min(int(frame_idx), len(scene.frames) - 1))
		frame_cameras = scene.frames[frame_idx].cameras
		cams = []
		for name in ("CAM_L0", "CAM_F0", "CAM_R0"):
			cam_obj = getattr(frame_cameras, name.lower(), None)
			if cam_obj is None or getattr(cam_obj, "image", None) is None:
				continue
			cams.append((name, cam_obj))
		if len(cams) == 0:
			return None
		pano, _ = build_front_panorama(cams)
		if pano is None:
			return None
		return Image.fromarray(pano)
	except Exception:
		return None


def compose_panel(
	cam_images: Dict[str, Image.Image],
	bev_image: Image.Image,
	tile_size: Tuple[int, int] = (640, 360),
	front_pano: Optional[Image.Image] = None,
) -> Image.Image:
	tile_w, tile_h = tile_size
	canvas = Image.new("RGB", (tile_w * 3, tile_h * 3), (16, 18, 22))
	draw = ImageDraw.Draw(canvas)

	# When a stitched front panorama is available, the three front cameras
	# (CAM_L0 / CAM_F0 / CAM_R0) are merged into a single wide top row instead
	# of being shown as three separate tiles.
	top_row_cams = {"CAM_L0", "CAM_F0", "CAM_R0"}

	for cam_name, (r, c) in GRID_LAYOUT.items():
		if front_pano is not None and cam_name in top_row_cams:
			continue
		x0, y0 = c * tile_w, r * tile_h
		if cam_name in cam_images:
			img = cam_images[cam_name].resize((tile_w, tile_h), Image.Resampling.BILINEAR)
		else:
			img = Image.new("RGB", (tile_w, tile_h), (60, 60, 60))
			ph = ImageDraw.Draw(img)
			ph.text((20, 20), f"Missing {cam_name}", fill=(240, 240, 240))
		canvas.paste(img, (x0, y0))
		draw.rectangle([(x0, y0), (x0 + tile_w - 1, y0 + tile_h - 1)], outline=(90, 90, 90), width=2)
		draw.text((x0 + 12, y0 + 10), cam_name, fill=(255, 255, 255))

	# Stitched front panorama spanning the full width of the top row.
	if front_pano is not None:
		pano_tile = fit_image_to_tile(front_pano, (tile_w * 3, tile_h), (16, 18, 22))
		canvas.paste(pano_tile, (0, 0))
		draw.rectangle([(0, 0), (tile_w * 3 - 1, tile_h - 1)], outline=(90, 90, 90), width=2)
		draw.text((12, 10), "Front panorama (CAM_L0 + CAM_F0 + CAM_R0)", fill=(255, 255, 255))

	cx, cy = tile_w, tile_h
	center = fit_image_to_tile(bev_image, (tile_w, tile_h), (255, 255, 255))
	canvas.paste(center, (cx, cy))
	draw.rectangle([(cx, cy), (cx + tile_w - 1, cy + tile_h - 1)], outline=(255, 180, 80), width=3)
	return canvas


def plot_pred_dynamics(traj_xyh: np.ndarray, out_path: Path, dt: float = 0.5) -> Optional[Path]:
	"""Plot per-point speed and acceleration curves of the PREDICTED trajectory (not GT)."""
	xy = np.asarray(traj_xyh, dtype=np.float64)
	if xy.ndim != 2 or xy.shape[0] < 2:
		return None
	xy = xy[:, :2]
	# Prepend ego origin (0,0) at t=0 since the prediction starts at t=dt.
	xy_full = np.vstack([np.zeros((1, 2), dtype=np.float64), xy])
	t = np.arange(xy_full.shape[0]) * float(dt)
	vel = np.gradient(xy_full, float(dt), axis=0)
	speed = np.linalg.norm(vel, axis=1)
	accel = np.gradient(speed, float(dt))
	
	fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
	ax_v.plot(t, speed, "-o", color="#D81B60", linewidth=2.0, markersize=5)
	ax_v.set_ylabel("Speed (m/s)")
	ax_v.set_title("Predicted trajectory dynamics")
	ax_v.grid(True, alpha=0.3)
	ax_a.plot(t, accel, "-o", color="#1E88E5", linewidth=2.0, markersize=5)
	ax_a.set_ylabel("Accel (m/s^2)")
	ax_a.set_xlabel("Time (s)")
	ax_a.grid(True, alpha=0.3)
	fig.tight_layout()
	fig.savefig(str(out_path), dpi=150)
	plt.close(fig)
	return out_path


def _pick_camview_viewpoint(scene, mode, traj_xyh=None, ground_z=0.0):
    """Return the frame index used as the camera viewpoint for trajectory projection.

    Start from the frame immediately before the current one (anchor-1) and step
    further back only when the first predicted waypoint cannot be projected inside
    that frame's front-camera image (i.e. it falls below/out of view).
    """
    anchor_idx = scene.scene_metadata.num_history_frames - 1
    anchor_idx = max(0, min(int(anchor_idx), len(scene.frames) - 1))
    if mode in (None, "current", "off"):
        return anchor_idx

    def _usable_cam(idx):
        cam = getattr(scene.frames[idx].cameras, "cam_f0", None)
        if (cam is not None and cam.image is not None and cam.intrinsics is not None
                and getattr(cam, "sensor2lidar_rotation", None) is not None):
            return cam
        return None

    have_wp = traj_xyh is not None and np.asarray(traj_xyh).size > 0
    traj_arr = np.asarray(traj_xyh, dtype=np.float64) if have_wp else None

    for idx in range(anchor_idx - 1, -1, -1):
        cam = _usable_cam(idx)
        if cam is None:
            continue
        if not have_wp:
            return idx
        wp_view = reproject_traj_between_frames(scene, traj_arr, anchor_idx, idx)[:1, :2]
        ih, iw = cam.image.shape[:2]
        _, in_fov = project_ego_points_to_camera(wp_view, cam, ground_z, (ih, iw), allow_below=False)
        if bool(np.asarray(in_fov).any()):
            return idx
    # Fallback: closest usable frame regardless of waypoint visibility, else current.
    for idx in range(anchor_idx - 1, -1, -1):
        if _usable_cam(idx) is not None:
            return idx
    return anchor_idx


def save_visualization(
	output_dir: Path,
	token: str,
	score: float,
	speed: float,
	accel: float,
	heading_centerline_angle_deg: float,
	else_tag: Optional[str],
	cam_images: Dict[str, Image.Image],
	scene,
	traj_xyh: np.ndarray,
	gt_traj_xyh: np.ndarray,
	proposals: Optional[np.ndarray],
	proposal_stages: Optional[List[np.ndarray]],
	ego_status_last: np.ndarray,
	save_format: str,
	pdm_scores: Optional[np.ndarray] = None,
	camview_mode: str = "front",
	camview_viewpoint: str = "current",
	ground_z: float = 0.0,
	vector_format: str = "none",
) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)
	score_tag = format_float4(score)
	speed_tag = format_float4(speed)
	accel_tag = format_float4(accel)
	heading_tag = format_float4(heading_centerline_angle_deg)
	base_name = f"token{token}_score{score_tag}_speed{speed_tag}_accel{accel_tag}_headingCenterDeg{heading_tag}"
	if else_tag:
		extra_tag = sanitize_tag(else_tag)
		base_name = f"{base_name}_{extra_tag}"

	vec = None if vector_format in (None, "none", "") else vector_format

	# Helper to save per-stage proposals BEV images
	def _save_proposal_stages():
		stages = proposal_stages
		if stages is None and proposals is not None:
			stages = [proposals]
		if stages is None:
			return
		for stage_idx, stage_props in enumerate(stages):
			if stage_idx != 4:
				continue
			if stage_props.ndim != 3 or stage_props.shape[0] == 0:
				continue
			prop_bev_path = output_dir / f"{base_name}bev_proposals[{stage_idx}].jpg"
			vector_path = output_dir / f"{base_name}bev_proposals[{stage_idx}].{vec}" if vec else None
			prop_bev = render_proposals_only_bev(
				scene, stage_props, traj_xyh, gt_traj_xyh,
				token, score_tag, ego_status_last, heading_centerline_angle_deg,
				traj_xyh.shape[0], stage_idx=stage_idx, vector_path=vector_path,
			)
			prop_bev.save(prop_bev_path, quality=95)

	def _save_camview():
		if camview_mode in (None, "off"):
			return
		stages = proposal_stages
		props = None
		if stages is not None and len(stages) > 0:
			props = stages[-1]
		elif proposals is not None:
			props = proposals
		if props is None or getattr(props, 'ndim', 0) != 3 or props.shape[0] == 0:
			return
		vector_path = output_dir / f"{base_name}camview_proposals.{vec}" if vec else None
		view_idx = _pick_camview_viewpoint(scene, camview_viewpoint, traj_xyh=traj_xyh, ground_z=ground_z)
		cam_img = render_proposals_on_camera(
			scene, props, pdm_scores, traj_xyh, gt_traj_xyh,
			token, score_tag, ego_status_last,
			camview_mode=camview_mode, ground_z=ground_z, vector_path=vector_path,
			viewpoint_frame_idx=view_idx,
		)
		if cam_img is not None:
			cam_img.save(output_dir / f"{base_name}camview_proposals.jpg", quality=95)

	# ---- Combined figure only: left = projection, right = BEV. GIF and
	#      per-stage / camview / dynamics / panel images are intentionally not
	#      produced. Native resolution preserved. ----
	view_idx = _pick_camview_viewpoint(scene, camview_viewpoint, traj_xyh=traj_xyh, ground_z=ground_z)
	proj_img = None
	if camview_mode not in (None, "off"):
		props_cam = proposal_stages[-1] if (proposal_stages is not None and len(proposal_stages) > 0) else proposals
		if props_cam is not None and getattr(props_cam, "ndim", 0) == 3 and props_cam.shape[0] > 0:
			proj_img = render_proposals_on_camera(
				scene, props_cam, pdm_scores, traj_xyh, gt_traj_xyh,
				token, score_tag, ego_status_last,
				camview_mode=camview_mode, ground_z=ground_z,
				viewpoint_frame_idx=view_idx,
			)
	bev_stage = proposal_stages[-1] if (proposal_stages is not None and len(proposal_stages) > 0) else proposals
	bev_img = None
	# Always export a standalone BEV figure as a separate PDF (native vector) per scene.
	bev_only_pdf = output_dir / f"{base_name}_bev.pdf"
	if bev_stage is not None and getattr(bev_stage, "ndim", 0) == 3 and bev_stage.shape[0] > 0:
		bev_img = render_proposals_only_bev(
			scene, bev_stage, traj_xyh, gt_traj_xyh,
			token, score_tag, ego_status_last, heading_centerline_angle_deg,
			traj_xyh.shape[0], stage_idx=0, pdm_scores=pdm_scores,
			vector_path=bev_only_pdf,
		)
	out_pdf = output_dir / f"{base_name}.pdf"
	out_png = output_dir / f"{base_name}.png"
	_save_combined_projection_bev(proj_img, bev_img, out_pdf, out_png)
	return out_pdf


def _move_to_device(value, device):
	"""Recursively move tensors inside nested lists/tuples/dicts to `device`,
	preserving structure and leaving non-tensor values (None, numpy, floats)
	intact. Scoring agents such as ZTRS / GTRS emit list-valued features
	(multi-frame `camera_feature` / `status_feature`) that must not be dropped."""
	if torch.is_tensor(value):
		return value.to(device, non_blocking=True)
	if isinstance(value, list):
		return [_move_to_device(v, device) for v in value]
	if isinstance(value, tuple):
		return tuple(_move_to_device(v, device) for v in value)
	if isinstance(value, dict):
		return {k: _move_to_device(v, device) for k, v in value.items()}
	return value


def tensor_features_to_device(features: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
	return {key: _move_to_device(value, device) for key, value in features.items()}


def _build_forward_caller(agent):
	"""Return f(features, targets, tokens) -> model output, adapting to agents
	whose `forward` expects either a features dict (DrivoR / ZTRS / GTRS-dense)
	or a (features, targets, tokens) batch tuple (GTRS-aug distillation arch)."""
	import inspect
	src = ""
	try:
		src = inspect.getsource(agent.forward)
	except Exception:
		pass
	if "= batch" in src:
		return lambda f, t, tok: agent.forward((f, t, tok))
	return lambda f, t, tok: agent.forward(f)


def init_distributed() -> Tuple[int, int, int]:
	if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
		rank = int(os.environ["RANK"])
		world_size = int(os.environ["WORLD_SIZE"])
		local_rank = int(os.environ.get("LOCAL_RANK", 0))
		if torch.cuda.is_available():
			torch.cuda.set_device(local_rank)
		return rank, world_size, local_rank
	return 0, 1, 0


def resolve_data_paths(args: argparse.Namespace) -> Tuple[Optional[Path], Optional[Path]]:
	"""Resolve data paths for non-standard splits. Returns (None, None) when Hydra config paths should be used."""
	if args.log_path is not None and args.sensor_path is not None:
		return args.log_path, args.sensor_path

	if args.log_path is not None or args.sensor_path is not None:
		raise ValueError("--log-path and --sensor-path must be provided together")

	if args.data_split == "navhard_two_stage":
		return (
			WORKSPACE_ROOT / "dataset" / "navhard_two_stage" / "openscene_meta_datas",
			WORKSPACE_ROOT / "dataset" / "navhard_two_stage" / "sensor_blobs",
		)

	if args.data_split == "private_test_hard_two_stage":
		return (
			WORKSPACE_ROOT / "dataset" / "private_test_hard_two_stage" / "openscene_meta_datas",
			WORKSPACE_ROOT / "dataset" / "private_test_hard_two_stage" / "sensor_blobs",
		)

	# For standard splits (navhard_two_stage, navtrain, etc.), use Hydra config paths
	return None, None


def resolve_token_output_dir(base_output_dir: Path, agent_name: str, token: str) -> Path:
        """Resolve per-token output folder to match map_bev_traj/<agent>/<token> style."""
        base = Path(base_output_dir)
        if base.name == token:
                return base
        if base.name == agent_name:
                return base / token
        if base.name == "map_bev_traj":
                return base / agent_name / token
        return base / token


def build_two_stage_token_remap(logs_dir: Path, anchor_idx: int) -> Dict[str, str]:
	"""
	For two-stage metadata, SceneLoader consumes the anchor token while CSV tokens may come
	from any frame in the short sequence. Map every frame token to the same anchor token.
	"""
	import pickle

	remap: Dict[str, str] = {}
	for pkl_path in logs_dir.glob("*.pkl"):
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
				frame_token = str(frame["token"])
				remap[frame_token] = anchor_token

	return remap


def estimate_forward_flops(run_forward, device: torch.device) -> Optional[float]:
	"""Estimate total FLOPs for a single agent.forward call using torch.profiler.

	Returns the total FLOPs (float) or None if profiling is unavailable/failed.
	Note: torch.profiler only counts FLOPs for recognized ops (matmul/conv/etc.),
	so this is a lower-bound estimate of the true model FLOPs.
	"""
	try:
		from torch.profiler import ProfilerActivity, profile
	except Exception as exc:  # pragma: no cover - depends on torch build
		print(f"[WARN] torch.profiler unavailable, skipping FLOPs: {exc}")
		return None

	use_cuda = device.type == "cuda"
	activities = [ProfilerActivity.CPU]
	if use_cuda:
		activities.append(ProfilerActivity.CUDA)

	try:
		with torch.no_grad():
			with profile(activities=activities, with_flops=True) as prof:
				_ = run_forward()
				if use_cuda:
					torch.cuda.synchronize(device)
		total_flops = 0.0
		for evt in prof.key_averages():
			flops = getattr(evt, "flops", 0) or 0
			total_flops += float(flops)
		return total_flops
	except Exception as exc:
		print(f"[WARN] FLOPs profiling failed: {exc}")
		return None


def benchmark_inference(
	agent,
	dataloader,
	device: torch.device,
	anchor_to_sources: Dict[str, List[str]],
	rank: int,
	world_size: int,
	warmup: int,
	measure_flops: bool,
	output_path: Path,
	agent_name: str,
) -> None:
	"""Measure ONLY the model inference latency (agent.forward) per scene.

	All visualization / rendering / metric extraction is skipped so the reported
	times reflect pure model inference. GPU timing uses CUDA events with explicit
	synchronization; CPU timing falls back to time.perf_counter.
	"""
	import time

	use_cuda = device.type == "cuda"
	per_scene: List[Dict[str, float]] = []
	flops_value: Optional[float] = None
	warmup_done = False

	agent.eval()
	forward_call = _build_forward_caller(agent)
	with torch.no_grad():
		for features, _targets, batch_tokens in tqdm(dataloader, desc="Benchmark-Infer"):
			loader_token = batch_tokens[0]
			source_tokens = anchor_to_sources.get(loader_token, [loader_token])
			model_features = tensor_features_to_device(features, device)

			# One-time warmup + optional FLOPs on the first scene (not timed).
			if not warmup_done:
				for _ in range(max(0, warmup)):
					_ = forward_call(model_features, _targets, batch_tokens)
					if use_cuda:
						torch.cuda.synchronize(device)
				if measure_flops:
					flops_value = estimate_forward_flops(lambda: forward_call(model_features, _targets, batch_tokens), device)
				warmup_done = True

			# Time the pure forward pass for this scene.
			if use_cuda:
				torch.cuda.synchronize(device)
				start_event = torch.cuda.Event(enable_timing=True)
				end_event = torch.cuda.Event(enable_timing=True)
				start_event.record()
				_ = forward_call(model_features, _targets, batch_tokens)
				end_event.record()
				torch.cuda.synchronize(device)
				latency_ms = float(start_event.elapsed_time(end_event))
			else:
				t0 = time.perf_counter()
				_ = forward_call(model_features, _targets, batch_tokens)
				latency_ms = (time.perf_counter() - t0) * 1000.0

			# One loader token may map back to several CSV (source) tokens.
			for token in source_tokens:
				per_scene.append({"token": token, "latency_ms": latency_ms})

	# Gather per-scene results from all ranks onto rank 0.
	if world_size > 1:
		if not dist.is_initialized():
			dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
		dist.barrier()
		gathered: List[Optional[List[Dict[str, float]]]] = [None for _ in range(world_size)]
		dist.all_gather_object(gathered, per_scene)
		gathered_flops: List[Optional[float]] = [None for _ in range(world_size)]
		dist.all_gather_object(gathered_flops, flops_value)
		if rank != 0:
			if dist.is_initialized():
				dist.destroy_process_group()
			return
		merged: List[Dict[str, float]] = []
		for part in gathered:
			if part:
				merged.extend(part)
		per_scene = merged
		flops_value = next((f for f in gathered_flops if f is not None), None)

	if not per_scene:
		print("[WARN] No scenes were benchmarked (empty dataloader?).")
		if world_size > 1 and dist.is_initialized():
			dist.destroy_process_group()
		return

	latencies = np.array([row["latency_ms"] for row in per_scene], dtype=np.float64)
	summary = {
		"agent": agent_name,
		"num_scenes": int(latencies.size),
		"mean_ms": float(np.mean(latencies)),
		"median_ms": float(np.median(latencies)),
		"std_ms": float(np.std(latencies)),
		"min_ms": float(np.min(latencies)),
		"max_ms": float(np.max(latencies)),
		"p90_ms": float(np.percentile(latencies, 90)),
		"p95_ms": float(np.percentile(latencies, 95)),
		"p99_ms": float(np.percentile(latencies, 99)),
		"throughput_fps": float(1000.0 / np.mean(latencies)) if np.mean(latencies) > 0 else float("nan"),
	}
	if flops_value is not None:
		summary["flops"] = float(flops_value)
		summary["gflops"] = float(flops_value) / 1e9

	# Parameter counts (total + trainable), reported in millions for readability.
	num_params = sum(p.numel() for p in agent.parameters())
	num_trainable = sum(p.numel() for p in agent.parameters() if p.requires_grad)
	summary["num_params"] = int(num_params)
	summary["params_M"] = float(num_params) / 1e6
	summary["num_trainable_params"] = int(num_trainable)
	summary["trainable_params_M"] = float(num_trainable) / 1e6

	# Write per-scene CSV plus an aggregated summary CSV next to it.
	output_path.parent.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(per_scene).to_csv(output_path, index=False)
	summary_path = output_path.with_name(output_path.stem + "_summary.csv")
	pd.DataFrame([summary]).to_csv(summary_path, index=False)

	print("=" * 60)
	print(f"[BENCHMARK] agent={agent_name}")
	print(f"[BENCHMARK] scenes           : {summary['num_scenes']}")
	print(f"[BENCHMARK] mean latency     : {summary['mean_ms']:.3f} ms")
	print(f"[BENCHMARK] median latency   : {summary['median_ms']:.3f} ms")
	print(f"[BENCHMARK] std latency      : {summary['std_ms']:.3f} ms")
	print(f"[BENCHMARK] min / max        : {summary['min_ms']:.3f} / {summary['max_ms']:.3f} ms")
	print(f"[BENCHMARK] p90 / p95 / p99  : {summary['p90_ms']:.3f} / {summary['p95_ms']:.3f} / {summary['p99_ms']:.3f} ms")
	print(f"[BENCHMARK] throughput       : {summary['throughput_fps']:.3f} scenes/s")
	print(f"[BENCHMARK] parameters       : {summary['params_M']:.3f} M total ({summary['num_params']}), {summary['trainable_params_M']:.3f} M trainable")
	if flops_value is not None:
		print(f"[BENCHMARK] FLOPs (1 scene)  : {summary['gflops']:.3f} GFLOPs (torch.profiler lower bound)")
	print(f"[BENCHMARK] per-scene CSV     : {output_path}")
	print(f"[BENCHMARK] summary CSV       : {summary_path}")
	print("=" * 60)

	if world_size > 1 and dist.is_initialized():
		dist.destroy_process_group()


def run(args: argparse.Namespace) -> None:
	repo_root = resolve_repo_root(args.agent_name, args.repo_root)
	print(f"[INFO] Using NAVSIM devkit root: {repo_root}")
	set_default_env(repo_root)
	setup_import_path(repo_root)
	rank, world_size, local_rank = init_distributed()
	if rank == 0:
		print(f"[INFO] Distributed world size: {world_size}")

	cfg, agent = load_cfg_and_agent(args, repo_root)
	resolved_log_path, resolved_sensor_path = resolve_data_paths(args)
	if resolved_log_path is not None:
		cfg.navsim_log_path = str(resolved_log_path)
		cfg.sensor_blobs_path = str(resolved_sensor_path)
		print(f"Using explicit data path: logs={cfg.navsim_log_path}")
		print(f"Using explicit data path: sensor={cfg.sensor_blobs_path}")
	else:
		print(f"Using Hydra config data path: logs={cfg.navsim_log_path}")
		print(f"Using Hydra config data path: sensor={cfg.sensor_blobs_path}")
	is_two_stage = (
		resolved_log_path is not None and (
			"navhard_two_stage" in str(resolved_log_path)
			or "private_test_hard_two_stage" in str(resolved_log_path)
		)
	)
	num_future_override = 1 if is_two_stage else None
	include_targets = (not is_two_stage) and (not args.benchmark)
	if is_two_stage:
		print("Detected two-stage data: using num_future_frames=1 and disabling GT target builders.")
	if args.token is not None:
		# When --token is specified, skip CSV entirely
		tokens = [args.token]
		token_to_score: Dict[str, float] = {}
		token_to_score_text: Dict[str, str] = {}
		token_to_raw_metrics: Dict[str, Dict[str, str]] = {}
		raw_metric_cols: List[str] = []
	else:
		tokens, token_to_score, token_to_score_text, token_to_raw_metrics, raw_metric_cols = load_tokens_and_scores(args.csv)
	if args.max_tokens > 0:
		tokens = tokens[: args.max_tokens]
	requested_tokens = list(tokens)
	anchor_to_sources: Dict[str, List[str]] = {}
	if is_two_stage:
		anchor_idx = 3
		remap = build_two_stage_token_remap(resolved_log_path, anchor_idx=anchor_idx)
		mapped_tokens = [remap.get(t, t) for t in requested_tokens]
		for src_token, mapped_token in zip(requested_tokens, mapped_tokens):
			if mapped_token not in anchor_to_sources:
				anchor_to_sources[mapped_token] = []
			anchor_to_sources[mapped_token].append(src_token)
		mapped_hits = sum(1 for src in requested_tokens if src in remap)
		print(f"two-stage token remap hits: {mapped_hits}/{len(requested_tokens)}")
		tokens = list(dict.fromkeys(mapped_tokens))
	else:
		for t in requested_tokens:
			anchor_to_sources[t] = [t]

	all_tokens = list(tokens)
	if world_size > 1:
		tokens = tokens[rank::world_size]
		if rank == 0:
			print(f"[INFO] Sharding {len(all_tokens)} tokens across {world_size} ranks")
		print(f"[RANK {rank}] local tokens: {len(tokens)}")

	scene_loader, dataloader = build_dataloader(
		cfg,
		agent,
		tokens,
		num_future_frames_override=num_future_override,
		include_targets=include_targets,
	)
	viz_scene_loader = None if args.benchmark else build_viz_scene_loader(cfg, tokens, num_future_frames_override=num_future_override)

	device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
	agent = agent.to(device).eval()

	if args.benchmark:
		if args.benchmark_output is not None:
			benchmark_output = Path(args.benchmark_output)
		else:
			benchmark_output = Path(args.output_dir) / f"inference_latency_{args.agent_name}.csv"
		print(f"[BENCHMARK] Running inference-only latency benchmark for agent={args.agent_name}")
		print(f"[BENCHMARK] warmup={args.benchmark_warmup}, measure_flops={args.measure_flops}")
		benchmark_inference(
			agent=agent,
			dataloader=dataloader,
			device=device,
			anchor_to_sources=anchor_to_sources,
			rank=rank,
			world_size=world_size,
			warmup=args.benchmark_warmup,
			measure_flops=args.measure_flops,
			output_path=benchmark_output,
			agent_name=args.agent_name,
		)
		return

	found_set = set(scene_loader.tokens)
	if is_two_stage:
		found_source_tokens = set()
		for anchor in scene_loader.tokens:
			for src in anchor_to_sources.get(anchor, [anchor]):
				found_source_tokens.add(src)
		missing_tokens = [t for t in requested_tokens if t not in found_source_tokens]
		print(
			f"CSV requested tokens: {len(requested_tokens)}, unique mapped anchor tokens: {len(tokens)}, "
			f"available anchor tokens in scene loader: {len(scene_loader.tokens)}"
		)
	else:
		missing_tokens = [t for t in tokens if t not in found_set]
		print(f"CSV tokens after dedup: {len(tokens)}, available tokens in scene loader: {len(scene_loader.tokens)}")
	if len(scene_loader.tokens) == 0:
		print("[WARN] Scene loader has 0 tokens. CSV tokens do not intersect with local dataset content.")
		print("[WARN] Check dataset version/split and data root paths.")
	if missing_tokens:
		print(f"[WARN] {len(missing_tokens)} tokens were not found in scenes and will be skipped.")

	print(f"[RANK {rank}] Starting inference and visualization, local scene tokens: {len(scene_loader.tokens)}")

	output_root = Path(args.output_dir)

	saved_paths: List[Path] = []
	per_token_info: Dict[str, Dict[str, str]] = {}
	has_any_command = False
	with torch.no_grad():
		for features, targets, batch_tokens in tqdm(dataloader, desc="Infer+Viz"):
			loader_token = batch_tokens[0]
			source_tokens = anchor_to_sources.get(loader_token, [loader_token])
			model_features = tensor_features_to_device(features, device)

			pred = agent.forward(model_features)
			trajectory = pred["trajectory"][0].detach().cpu().numpy()
			# Extract all proposal refinement stages for visualization
			proposal_list = pred.get("proposal_list", None)
			if proposal_list is not None:
				# Convert each stage to numpy: proposal_list[stage_idx][0] -> (num_proposals, num_poses, 3)
				proposals = proposal_list[0][0].detach().cpu().numpy()
				proposal_stages = [p[0].detach().cpu().numpy() for p in proposal_list]
				print(f"[INFO] Extracted {len(proposal_stages)} proposal stages "
				      f"(stage 0: {proposal_stages[0].shape[0]} initial, "
				      f"stage {len(proposal_stages)-1}: {proposal_stages[-1].shape[0]} refined)")
			else:
				proposals = pred.get("proposals", None)
				proposal_stages = None
				if proposals is not None:
					proposals = proposals[0].detach().cpu().numpy()
					proposal_stages = [proposals]
					print(f"[INFO] Extracted {proposals.shape[0]} proposals (single stage)")
				else:
					proposals = None
					print("[WARN] No proposals found in pred, skipping proposal visualization")
			pdm_scores_np = None
			if isinstance(pred, dict) and pred.get("pdm_score", None) is not None:
				try:
					pdm_scores_np = pred["pdm_score"][0].detach().cpu().numpy()
				except Exception:
					pdm_scores_np = None
			if include_targets and "trajectory" in targets and targets["trajectory"].numel() > 0:
				gt_trajectory = targets["trajectory"][0].detach().cpu().numpy()
			else:
				gt_trajectory = np.empty((0, trajectory.shape[-1]), dtype=trajectory.dtype)

			scene = viz_scene_loader.get_scene_from_token(loader_token)
			cams = extract_camera_images_from_scene(scene)

			# Camera fallback: if scene camera images are unavailable, use model input images.
			if len(cams) == 0:
				image_tensor = features["image"][0]
				cams = extract_camera_images(image_tensor)

			ego_status_last = features["ego_status"][0, -1].detach().cpu().numpy()
			speed, accel = extract_ego_metrics(ego_status_last)
			ego_heading = extract_ego_heading(scene)
			heading_centerline_angle_deg = compute_heading_centerline_angle_deg(scene)
			v_lon_centerline, v_lat_centerline, a_lon_centerline, a_lat_centerline = compute_centerline_relative_dynamics(
				scene,
				ego_status_last,
			)
			pred_gt_l2_mean = compute_pred_gt_l2_mean(trajectory, gt_trajectory)
			cmd_tag = extract_command_tag(features)
			has_any_command = has_any_command or (cmd_tag is not None)

			for token in source_tokens:
				score = token_to_score.get(token, float("nan"))
				token_output_dir = resolve_token_output_dir(output_root, args.agent_name, token)
				out_path = save_visualization(
					output_dir=token_output_dir,
					token=token,
					score=score,
					speed=speed,
					accel=accel,
					heading_centerline_angle_deg=heading_centerline_angle_deg,
					else_tag=cmd_tag,
					cam_images=cams,
					scene=scene,
					traj_xyh=trajectory,
					gt_traj_xyh=gt_trajectory,
					proposals=proposals,
					proposal_stages=proposal_stages,
					ego_status_last=ego_status_last,
					save_format=args.save_format,
					pdm_scores=pdm_scores_np,
					camview_mode=args.camview_mode,
					camview_viewpoint=args.camview_viewpoint,
					ground_z=args.ground_z,
					vector_format=args.vector_format,
				)
				saved_paths.append(out_path)
				per_token_info[token] = {
					"token": token,
					"score": format_float4(score),
					"ego_heading_deg": format_float4(np.degrees(ego_heading)),
					"ego_vx": format_float4(float(ego_status_last[3])),
					"ego_vy": format_float4(float(ego_status_last[4])),
					"ego_ax": format_float4(float(ego_status_last[5])),
					"ego_ay": format_float4(float(ego_status_last[6])),
					"speed": format_float4(speed),
					"accel": format_float4(accel),
					"heading_centerline_angle_deg": format_float4(heading_centerline_angle_deg),
					"ego_v_lon_centerline": format_float4(v_lon_centerline),
					"ego_v_lat_centerline": format_float4(v_lat_centerline),
					"ego_a_lon_centerline": format_float4(a_lon_centerline),
					"ego_a_lat_centerline": format_float4(a_lat_centerline),
					"pred_gt_l2_mean": format_float4(pred_gt_l2_mean),
					"command": cmd_tag if cmd_tag is not None else "",
					"output_file": out_path.name,
				}
				for col in raw_metric_cols:
					per_token_info[token][col] = token_to_raw_metrics.get(token, {}).get(col, "")

	if world_size > 1:
		if not dist.is_initialized():
			dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
		dist.barrier()
		gathered_token_info = [None for _ in range(world_size)]
		dist.all_gather_object(gathered_token_info, per_token_info)
		gathered_saved_counts = [None for _ in range(world_size)]
		dist.all_gather_object(gathered_saved_counts, len(saved_paths))
		if rank != 0:
			if dist.is_initialized():
				dist.destroy_process_group()
			return
		merged_token_info: Dict[str, Dict[str, str]] = {}
		for part in gathered_token_info:
			merged_token_info.update(part)
		per_token_info = merged_token_info
		saved_paths = [None] * sum(int(x) for x in gathered_saved_counts)

	metrics_rows: List[Dict[str, str]] = []
	for token in requested_tokens:
		if token in per_token_info:
			metrics_rows.append(per_token_info[token])
		else:
			score = token_to_score.get(token, float("nan"))
			metrics_rows.append(
				{
					"token": token,
					"score": format_float4(score),
					"ego_heading_deg": "nan",
					"ego_vx": "nan",
					"ego_vy": "nan",
					"ego_ax": "nan",
					"ego_ay": "nan",
					"speed": "nan",
					"accel": "nan",
					"heading_centerline_angle_deg": "nan",
					"ego_v_lon_centerline": "nan",
					"ego_v_lat_centerline": "nan",
					"ego_a_lon_centerline": "nan",
					"ego_a_lat_centerline": "nan",
					"pred_gt_l2_mean": "nan",
					"command": "",
					"output_file": "",
				}
			)
			for col in raw_metric_cols:
				metrics_rows[-1][col] = token_to_raw_metrics.get(token, {}).get(col, "")

	if not has_any_command:
		for row in metrics_rows:
			if "command" in row:
				del row["command"]

	if SAVE_INTERMEDIATE_OUTPUTS:
		output_root.mkdir(parents=True, exist_ok=True)
		metrics_path = output_root / args.metrics_filename
		pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
		print(f"Ego state + metric table saved to: {metrics_path}")

	print(f"Done. Saved {len(saved_paths)} files to: {output_root}")
	if world_size > 1 and dist.is_initialized():
		dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Token-level trajectory inference + 8-camera/BEV visualization export")
	parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV file containing token and scoring columns")
	parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT, help="Model checkpoint path")
	parser.add_argument("--num-scene-tokens", type=int, default=None, help="Override agent.config.num_scene_tokens to match the checkpoint (e.g. 32)")
	parser.add_argument("--agent-name", type=str, default="drivoR", help="Hydra agent name, e.g. drivoR")
	parser.add_argument("--config-name", type=str, default="default_run_pdm_score_gpu", help="Hydra config name under pdm_scoring config dir")
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
	parser.add_argument("--repo-root", type=Path, default=None, help="Override NAVSIM devkit root path")
	parser.add_argument("--data-split", type=str, default=DEFAULT_DATA_SPLIT, help="Data split name (default: navhard_two_stage, also supports navhard_two_stage, private_test_hard_two_stage, etc.)")
	parser.add_argument("--log-path", type=Path, default=None, help="Explicit log pkl directory (must be used with --sensor-path)")
	parser.add_argument("--sensor-path", type=Path, default=None, help="Explicit sensor_blobs directory (must be used with --log-path)")
	parser.add_argument("--max-tokens", type=int, default=0, help="Maximum number of tokens to process, 0 means all")
	parser.add_argument("--metrics-filename", type=str, default="ego_metrics.csv", help="Output table filename")
	parser.add_argument("--token", type=str, default=None, help="Specific scene token to visualize (processes all CSV tokens if not set)")
	parser.add_argument("--save-format", choices=["jpg", "gif"], default="jpg", help="Visualization format")
	parser.add_argument("--camview-mode", choices=["off", "front", "wide"], default="wide", help="Project candidate trajectories onto front camera(s): off, front (CAM_F0), or wide (CAM_L0+F0+R0 stitched panorama)")
	parser.add_argument("--camview-viewpoint", choices=["current", "first_history"], default="current", help="Viewpoint frame for camview trajectory projection: 'current' anchor frame, or 'first_history' (earliest history frame) so near waypoints are visible")
	parser.add_argument("--vector-format", choices=["none", "pdf", "svg", "eps"], default="none", help="Also export BEV/proposal/camview figures as a vector format (pdf/svg/eps) for LaTeX/Overleaf")
	parser.add_argument("--ground-z", type=float, default=0.0, help="Ground-plane height in ego/lidar frame used for trajectory-to-camera projection (default 0.0)")
	parser.add_argument("--benchmark", action="store_true", help="Benchmark mode: measure ONLY the model inference latency (agent.forward) per scene, skipping all visualization/rendering")
	parser.add_argument("--benchmark-output", type=Path, default=None, help="CSV path to write per-scene inference latency. Defaults to <output-dir>/inference_latency_<agent>.csv")
	parser.add_argument("--benchmark-warmup", type=int, default=5, help="Number of warmup forward passes (not timed) before benchmarking, to stabilize CUDA kernels")
	parser.add_argument("--measure-flops", action="store_true", help="Also estimate model FLOPs for a single scene via torch.profiler (benchmark mode only)")
	return parser.parse_args()


if __name__ == "__main__":
	run(parse_args())
