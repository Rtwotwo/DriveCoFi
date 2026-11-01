"""
Scene-to-Scene cosine similarity map builder. Computes the mean
Scene-to-Scene cosine similarity across the entire validation set.

# Example (multi-GPU DDP)
torchrun --nproc_per_node=4 viz/viz_scene2patch_attention_all.py \
    --checkpoint_path /path/to/your/checkpoint.ckpt \
    --agent_name more_ccs \
    --experiment_name scene_similarity_analysis \
    --split navtest \
    --num_workers 4 \
    --batch_size 16 \
    --output_dir /path/to/your/output_dir \
    --dpi 150
"""
import os
import sys
import argparse
import torch
import torch.distributed as dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from torch.utils.data import DataLoader, DistributedSampler

# ============================================================
# CONFIGURATION — change these to match your model
# ============================================================
REPO_ROOT = str(Path(__file__).resolve().parents[1])
DATASET_ROOT = str(Path(__file__).resolve().parents[2])
CHECKPOINT_PATH = None  # REQUIRED — e.g. "/path/to/model.ckpt"

AGENT_NAME = "drivoR_moetoken"
EXPERIMENT_NAME = "scene_similarity_analysis"
SPLIT = "navtest"

H_P = 48
W_P = 82
NUM_WORKERS = 4
BATCH_SIZE = 8

OUTPUT_DIR = None
DPI = 150
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(description="DrivoR Scene Token Cosine Similarity Map Builder")

    parser.add_argument("--repo_root", type=str, default=REPO_ROOT)
    parser.add_argument("--dataset_root", type=str, default=DATASET_ROOT)
    parser.add_argument("--checkpoint_path", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--agent_name", type=str, default=AGENT_NAME)
    parser.add_argument("--experiment_name", type=str, default=EXPERIMENT_NAME)
    parser.add_argument("--split", type=str, default=SPLIT)

    parser.add_argument("--H_p", type=int, default=H_P)
    parser.add_argument("--W_p", type=int, default=W_P)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)

    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=DPI)

    parser.add_argument("--inline_helpers", type=str, default="1")

    args = parser.parse_args()

    if args.checkpoint_path is None:
        parser.error("--checkpoint_path is required (set it in CONFIGURATION section or via CLI)")

    return args


def setup_environment(repo_root, dataset_root):
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))

    os.environ.update({
        "NUPLAN_MAPS_ROOT": str(dataset_root / "dataset/maps"),
        "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
        "OPENSCENE_DATA_ROOT": str(dataset_root / "dataset"),
        "NAVSIM_EXP_ROOT": str(dataset_root / "exp"),
        "NAVSIM_DEVKIT_ROOT": str(repo_root),
    })


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        print("DDP variables not found. Running in single-GPU mode.")
        rank, world_size, local_rank = 0, 1, 0

    return rank, world_size, local_rank


def load_components(repo_root, checkpoint_path, agent_name, experiment_name,
                    split, num_workers, batch_size, world_size, rank):
    from navsim.agents.abstract_agent import AbstractAgent
    from navsim.common.dataloader import SceneLoader, SceneFilter
    from navsim.planning.training.dataset import Dataset

    config_dir = repo_root / "navsim/planning/script/config/training"
    agent_config_dir = repo_root / "navsim/planning/script/config/common/agent"

    overrides = [
        f"train_test_split={split}",
        f"experiment_name={experiment_name}",
        f"dataloader.params.num_workers={num_workers}",
        f"dataloader.params.batch_size={batch_size}",
    ]

    dinov2_weights = str(repo_root / "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors")

    agent_overrides = [
        f"checkpoint_path={str(checkpoint_path)}",
        "config.shared_refiner=false", "lr_args=null", "scheduler_args=null",
        "progress_bar=false", "config.refiner_ls_values=0.0",
        "config.image_backbone.focus_front_cam=false", "config.one_token_per_traj=true",
        "config.refiner_num_heads=1", "config.tf_d_model=256", "config.tf_d_ffn=1024",
        "config.area_pred=false", "config.agent_pred=false", "config.ref_num=4",
        f"config.image_backbone.model_weights={dinov2_weights}",
        f"config.lidar_backbone.model_weights={dinov2_weights}",
        "loss.prev_weight=0.0", "batch_size=null",
        "config.noc=10.2", "config.dac=12.5", "config.ddc=6.0",
        "config.ttc=14", "config.ep=15", "config.comfort=2.1",
    ]

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=str(agent_config_dir)):
        agent_cfg = compose(config_name=agent_name, overrides=agent_overrides)
    agent = instantiate(agent_cfg)
    agent.initialize()
    agent.b2d = agent.ray = False

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="default_training", overrides=overrides)

    scene_filter = instantiate(cfg.train_test_split.scene_filter)

    loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config()
    )

    dataset = Dataset(
        scene_loader=loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True
    )

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        **cfg.dataloader.params,
        drop_last=True,
    )

    return agent, dataloader


# ==============================================================================
# Model Inspection Helpers (inline)
# ==============================================================================

def _get_drivor_model(agent):
    import torch.nn as nn
    if hasattr(agent, '_drivor_model'):
        return agent._drivor_model
    if hasattr(agent, 'model'):
        return agent.model
    for attr in dir(agent):
        child = getattr(agent, attr)
        if isinstance(child, nn.Module) and hasattr(child, 'image_backbone'):
            return child
    raise RuntimeError("Cannot find DrivoRModel in agent.")


def _get_vit_model(drivor_model):
    if hasattr(drivor_model, 'image_backbone'):
        encoder = drivor_model.image_backbone
        if hasattr(encoder, 'model'):
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

    return {
        'prefix': prefix,
        'num_scene_tokens': num_scene_tokens,
        'num_cls_tokens': num_cls_tokens,
        'num_reg_tokens': num_reg_tokens,
        'reg_start': reg_start,
        'num_patches': num_patches,
    }


def extract_vit_attentions(model, device="cuda"):
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


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()

    repo_root = Path(args.repo_root)
    dataset_root = Path(args.dataset_root)
    checkpoint_path = Path(args.checkpoint_path)

    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "results/scene_similarity_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_environment(repo_root, dataset_root)
    rank, world_size, local_rank = init_distributed()

    if rank == 0:
        print(f"Repo root:    {repo_root}")
        print(f"Dataset root: {dataset_root}")
        print(f"Checkpoint:   {checkpoint_path}")
        print(f"Agent name:   {args.agent_name}")
        print(f"Output dir:   {output_dir}")
        print(f"World size:   {world_size}")

    print("Loading model...")
    agent, val_loader = load_components(
        repo_root, checkpoint_path, args.agent_name, args.experiment_name,
        args.split, args.num_workers, args.batch_size, world_size, rank
    )
    agent.cuda(local_rank).eval()
    print("Model loaded.")

    cams = ["cam_f0", "cam_b0", "cam_l0", "cam_r0"]
    H_p, W_p = args.H_p, args.W_p
    sum_sims = None
    sum_energies = None
    count_local = torch.tensor([0.0], device=f"cuda:{local_rank}")

    with torch.no_grad():
        iterator = tqdm(val_loader, disable=(rank != 0), desc="Computing Scene Similarity")
        for batch in iterator:
            features, _, _ = batch
            features = {k: v.cuda(local_rank, non_blocking=True) if hasattr(v, "cuda") else v
                        for k, v in features.items()}

            drivor_model = _get_drivor_model(agent)
            vit_hooks, vit_attn_maps = extract_vit_attentions(drivor_model, device=f"cuda:{local_rank}")
            preds = agent.forward(features)

            attn_last = vit_attn_maps[f"layer_{len(vit_attn_maps)-1}"]
            for h in vit_hooks:
                h.remove()

            B, H, N, _ = attn_last.shape

            vit_model = _get_vit_model(drivor_model)
            layout = detect_token_layout(vit_model, N, H_p, W_p)
            prefix_auto = layout['prefix']
            S_auto = layout['num_scene_tokens']

            if sum_sims is None:
                sum_sims = {cam: torch.zeros((S_auto, S_auto), device=f"cuda:{local_rank}", dtype=torch.float32) for cam in cams}
                sum_energies = {cam: torch.zeros((H_p, W_p), device=f"cuda:{local_rank}", dtype=torch.float32) for cam in cams}

            scene_idx = torch.arange(0, S_auto, device=attn_last.device)
            patch_idx = torch.arange(prefix_auto, N, device=attn_last.device)

            A = attn_last.mean(1)
            A = A / (A.sum(-1, keepdim=True) + 1e-9)
            S2P = A[:, scene_idx][:, :, patch_idx]

            for b, cam in enumerate(cams):
                sf = S2P[b].flatten(1)
                sf = sf / (sf.norm(dim=-1, keepdim=True) + 1e-9)
                sim = sf @ sf.T
                sum_sims[cam] += sim

                energy = S2P[b].mean(0)
                sum_energies[cam] += energy

            count_local += 1
            del preds, features

    if world_size > 1:
        if rank == 0:
            print("Aggregating results across GPUs...")

        count_global = count_local.clone()
        dist.all_reduce(count_global, op=dist.ReduceOp.SUM)

        mean_sims = {}
        mean_energies = {}
        for cam, sim_local in sum_sims.items():
            sim_global = sim_local.clone()
            dist.all_reduce(sim_global, op=dist.ReduceOp.SUM)
            mean_sims[cam] = (sim_global / count_global).cpu()

        for cam, e_local in sum_energies.items():
            e_global = e_local.clone()
            dist.all_reduce(e_global, op=dist.ReduceOp.SUM)
            mean_energies[cam] = (e_global / count_global).cpu()
    else:
        count_global = count_local
        mean_sims = {cam: (sim / count_local).cpu() for cam, sim in sum_sims.items()}
        mean_energies = {cam: (e / count_local).cpu() for cam, e in sum_energies.items()}

    if rank == 0:
        # Save raw data
        torch.save({"mean_sims": mean_sims, "mean_energies": mean_energies},
                   output_dir / f"mean_scene_scene_cosine_{args.split}.pt")

        S = list(mean_sims.values())[0].shape[0]
        print(f"\nResults summary ({int(count_global.item())} batches, {S} scene tokens per camera):")
        print("=" * 50)

        # ---- Scene-Scene Cosine Similarity ----
        for cam, sim in mean_sims.items():
            fig, ax = plt.subplots(figsize=(5, 4.5))
            im = ax.imshow(sim, cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f"{cam.upper()} — Mean Scene↔Scene Cosine")
            fig.colorbar(im, label="Cosine Similarity")
            ax.set_xlabel("Scene Token j")
            ax.set_ylabel("Scene Token i")
            fig.tight_layout()

            save_path = output_dir / f"scene_scene_cosine_{cam}_{args.split}.png"
            fig.savefig(save_path, bbox_inches="tight", dpi=args.dpi)
            plt.close(fig)

            off_diag_mean = sim.triu(1).mean().item()
            print(f"  {cam}: mean inter-scene cosine = {off_diag_mean:.4f}")
            print(f"     Saved: {save_path.name}")

        # ---- Mean Patch Energy ----
        for cam, energy in mean_energies.items():
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(energy, cmap='hot')
            ax.set_title(f"{cam.upper()} — Mean Scene→Patch Energy")
            fig.colorbar(im, label="Mean Attention")
            fig.tight_layout()

            save_path = output_dir / f"mean_patch_energy_{cam}_{args.split}.png"
            fig.savefig(save_path, bbox_inches="tight", dpi=args.dpi)
            plt.close(fig)
            print(f"     Saved: {save_path.name}")

        print(f"\nAll results saved to: {output_dir}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
