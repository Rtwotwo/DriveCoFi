import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import matplotlib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneFilter, SceneLoader
from navsim.planning.training.dataset import Dataset


matplotlib.use("Agg")
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"


class GateAnalysisLightningModule(pl.LightningModule):
    """Collect scorer branch outputs without changing the model computation."""

    def __init__(self, agent: AbstractAgent):
        super().__init__()
        self.agent = agent
        self._captured: Dict[Tuple[int, str], Tensor] = {}

        model = getattr(agent, "_drivor_model", None)
        scorer_attention = getattr(model, "scorer_attention", None)
        layers = getattr(scorer_attention, "layers", None)
        if layers is None:
            raise TypeError("The selected agent does not expose scorer_attention.layers")

        self.num_scorer_layers = len(layers)
        for layer_index, layer in enumerate(layers):
            if not hasattr(layer, "scene_cross_attn") or not hasattr(layer, "patch_cross_attn"):
                raise TypeError(
                    f"Scorer layer {layer_index} is not a dual-branch gated block"
                )
            layer.scene_cross_attn.register_forward_hook(
                self._make_capture_hook(layer_index, "scene")
            )
            layer.patch_cross_attn.register_forward_hook(
                self._make_capture_hook(layer_index, "patch")
            )

    def _make_capture_hook(self, layer_index: int, branch: str):
        def hook(_module, _inputs, output):
            self._captured[(layer_index, branch)] = output.detach()

        return hook

    def predict_step(self, batch, batch_idx: int):
        features, _targets, tokens = batch
        self.agent.eval()
        self._captured.clear()

        with torch.no_grad():
            predictions = self.agent.forward(features)

        gates = predictions.get("scorer_patch_gates")
        if gates is None:
            raise KeyError("Model output does not contain scorer_patch_gates")
        gates = gates.squeeze(-1).permute(1, 0, 2).float()

        scene_deltas = []
        patch_deltas = []
        for layer_index in range(self.num_scorer_layers):
            scene_key = (layer_index, "scene")
            patch_key = (layer_index, "patch")
            if scene_key not in self._captured or patch_key not in self._captured:
                raise RuntimeError(f"Missing branch capture for scorer layer {layer_index}")
            scene_deltas.append(self._captured[scene_key].float())
            patch_deltas.append(self._captured[patch_key].float())

        scene_deltas = torch.stack(scene_deltas, dim=1)
        patch_deltas = torch.stack(patch_deltas, dim=1)
        if scene_deltas.shape[:3] != gates.shape or patch_deltas.shape[:3] != gates.shape:
            raise RuntimeError(
                "Gate and branch shapes disagree: "
                f"gate={tuple(gates.shape)}, scene={tuple(scene_deltas.shape)}, "
                f"patch={tuple(patch_deltas.shape)}"
            )

        scene_norm = torch.linalg.vector_norm(scene_deltas, dim=-1)
        patch_norm = torch.linalg.vector_norm(patch_deltas, dim=-1)
        branch_cosine = F.cosine_similarity(scene_deltas, patch_deltas, dim=-1)
        scene_contribution = (1.0 - gates) * scene_norm
        patch_contribution = gates * patch_norm
        effective_patch_ratio = patch_contribution / (
            scene_contribution + patch_contribution + 1e-8
        )

        pdm_score = predictions["pdm_score"].float()
        selected_index = pdm_score.argmax(dim=-1)

        return {
            "tokens": list(tokens),
            "gates": gates.cpu().numpy(),
            "scene_norm": scene_norm.cpu().numpy(),
            "patch_norm": patch_norm.cpu().numpy(),
            "effective_patch_ratio": effective_patch_ratio.cpu().numpy(),
            "branch_cosine": branch_cosine.cpu().numpy(),
            "pdm_score": pdm_score.cpu().numpy(),
            "selected_index": selected_index.cpu().numpy(),
        }


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _merge_predictions(all_predictions: List[List[Dict[str, Any]]]) -> Dict[str, np.ndarray]:
    batches = [batch for rank_predictions in all_predictions for batch in rank_predictions]
    if not batches:
        raise RuntimeError("No gate-analysis predictions were produced")

    tokens = np.asarray([token for batch in batches for token in batch["tokens"]])
    array_keys = [
        "gates",
        "scene_norm",
        "patch_norm",
        "effective_patch_ratio",
        "branch_cosine",
        "pdm_score",
        "selected_index",
    ]
    merged = {key: np.concatenate([batch[key] for batch in batches], axis=0) for key in array_keys}

    first_index: Dict[str, int] = {}
    unique_indices = []
    duplicate_differences = []
    for index, token in enumerate(tokens.tolist()):
        if token not in first_index:
            first_index[token] = index
            unique_indices.append(index)
            continue
        previous = first_index[token]
        duplicate_differences.append(
            float(np.max(np.abs(merged["gates"][previous] - merged["gates"][index])))
        )

    unique_indices_array = np.asarray(unique_indices, dtype=np.int64)
    output = {key: value[unique_indices_array] for key, value in merged.items()}
    output["tokens"] = tokens[unique_indices_array]
    output["duplicate_count"] = np.asarray(len(tokens) - len(unique_indices), dtype=np.int64)
    output["max_duplicate_gate_difference"] = np.asarray(
        max(duplicate_differences, default=0.0), dtype=np.float32
    )
    return output


def _describe(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"{prefix}_count": 0}
    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_p05": float(quantiles[0]),
        f"{prefix}_p25": float(quantiles[1]),
        f"{prefix}_median": float(quantiles[2]),
        f"{prefix}_p75": float(quantiles[3]),
        f"{prefix}_p95": float(quantiles[4]),
        f"{prefix}_fraction_lt_025": float((values < 0.25).mean()),
        f"{prefix}_fraction_gt_050": float((values > 0.50).mean()),
        f"{prefix}_fraction_gt_075": float((values > 0.75).mean()),
    }


def _take_proposals(values: np.ndarray, proposal_indices: np.ndarray) -> np.ndarray:
    batch_indices = np.arange(values.shape[0])[:, None]
    layer_indices = np.arange(values.shape[1])[None, :]
    if proposal_indices.ndim == 1:
        return values[batch_indices, layer_indices, proposal_indices[:, None]]
    return values[
        batch_indices[:, :, None],
        layer_indices[:, :, None],
        proposal_indices[:, None, :],
    ]


def _build_layer_summary(data: Dict[str, np.ndarray]) -> pd.DataFrame:
    gates = data["gates"]
    selected_index = data["selected_index"]
    top5_indices = np.argpartition(data["pdm_score"], -5, axis=1)[:, -5:]
    selected = {
        key: _take_proposals(data[key], selected_index)
        for key in ("gates", "scene_norm", "patch_norm", "effective_patch_ratio", "branch_cosine")
    }
    top5 = {
        key: _take_proposals(data[key], top5_indices)
        for key in ("gates", "scene_norm", "patch_norm", "effective_patch_ratio", "branch_cosine")
    }

    rows = []
    for layer_index in range(gates.shape[1]):
        for scope, source in (("all", data), ("selected", selected), ("predicted_top5", top5)):
            row: Dict[str, Any] = {"layer": layer_index, "scope": scope}
            row.update(_describe(source["gates"][:, layer_index], "gate"))
            row.update(
                _describe(
                    source["effective_patch_ratio"][:, layer_index],
                    "effective_patch_ratio",
                )
            )
            row.update(_describe(source["scene_norm"][:, layer_index], "scene_norm"))
            row.update(_describe(source["patch_norm"][:, layer_index], "patch_norm"))
            row.update(_describe(source["branch_cosine"][:, layer_index], "branch_cosine"))
            rows.append(row)
    return pd.DataFrame(rows)


def _build_per_scene_summary(data: Dict[str, np.ndarray]) -> pd.DataFrame:
    tokens = data["tokens"]
    selected_index = data["selected_index"]
    pdm_score = data["pdm_score"]
    top5_indices = np.argpartition(pdm_score, -5, axis=1)[:, -5:]

    selected = {
        key: _take_proposals(data[key], selected_index)
        for key in ("gates", "scene_norm", "patch_norm", "effective_patch_ratio", "branch_cosine")
    }
    top5 = {
        key: _take_proposals(data[key], top5_indices)
        for key in ("gates", "effective_patch_ratio")
    }

    frame = pd.DataFrame(
        {
            "token": tokens,
            "selected_index": selected_index,
            "selected_predicted_pdm_score": pdm_score[np.arange(len(tokens)), selected_index],
        }
    )
    for layer_index in range(data["gates"].shape[1]):
        frame[f"layer_{layer_index}_gate_all_mean"] = data["gates"][:, layer_index].mean(axis=1)
        frame[f"layer_{layer_index}_gate_selected"] = selected["gates"][:, layer_index]
        frame[f"layer_{layer_index}_gate_top5_mean"] = top5["gates"][:, layer_index].mean(axis=1)
        frame[f"layer_{layer_index}_effective_patch_ratio_all_mean"] = data[
            "effective_patch_ratio"
        ][:, layer_index].mean(axis=1)
        frame[f"layer_{layer_index}_effective_patch_ratio_selected"] = selected[
            "effective_patch_ratio"
        ][:, layer_index]
        frame[f"layer_{layer_index}_effective_patch_ratio_top5_mean"] = top5[
            "effective_patch_ratio"
        ][:, layer_index].mean(axis=1)
        frame[f"layer_{layer_index}_scene_norm_selected"] = selected["scene_norm"][:, layer_index]
        frame[f"layer_{layer_index}_patch_norm_selected"] = selected["patch_norm"][:, layer_index]
        frame[f"layer_{layer_index}_branch_cosine_selected"] = selected["branch_cosine"][:, layer_index]

    gate_selected_columns = [
        column for column in frame if column.endswith("_gate_selected")
    ]
    effective_selected_columns = [
        column for column in frame if column.endswith("_effective_patch_ratio_selected")
    ]
    frame["gate_selected_layers_mean"] = frame[gate_selected_columns].mean(axis=1)
    frame["effective_patch_ratio_selected_layers_mean"] = frame[
        effective_selected_columns
    ].mean(axis=1)
    return frame


def _combine_stage_metric(
    frame: pd.DataFrame, metric: str, aliases: Tuple[str, ...] = ()
) -> pd.Series:
    metric_candidates = (metric, *aliases)
    for candidate in metric_candidates:
        if candidate in frame:
            return frame[candidate]
    columns = [
        column
        for candidate in metric_candidates
        for column in (f"{candidate}_stage_one", f"{candidate}_stage_two")
        if column in frame
    ]
    if not columns:
        return pd.Series(np.nan, index=frame.index)
    return frame[columns].bfill(axis=1).iloc[:, 0]


def _infer_pdms_stage(frame: pd.DataFrame) -> pd.Series:
    stage_one_columns = [column for column in frame if column.endswith("_stage_one")]
    stage_two_columns = [column for column in frame if column.endswith("_stage_two")]
    if not stage_one_columns and not stage_two_columns:
        return pd.Series("combined", index=frame.index)

    stage_one_present = (
        frame[stage_one_columns].notna().any(axis=1)
        if stage_one_columns
        else pd.Series(False, index=frame.index)
    )
    stage_two_present = (
        frame[stage_two_columns].notna().any(axis=1)
        if stage_two_columns
        else pd.Series(False, index=frame.index)
    )
    return pd.Series(
        np.select(
            [stage_one_present & ~stage_two_present, stage_two_present & ~stage_one_present],
            ["stage_one", "stage_two"],
            default="combined",
        ),
        index=frame.index,
    )


def _add_pdms_analysis(
    per_scene: pd.DataFrame, pdms_csv: Path, output_dir: Path
) -> Dict[str, Any]:
    pdms = pd.read_csv(pdms_csv)
    if "valid" in pdms:
        pdms = pdms[pdms["valid"].astype(str).str.lower().eq("true")]
    pdms = pdms[pdms["token"].isin(per_scene["token"])]

    metric_aliases = {
        "no_at_fault_collisions": (),
        "drivable_area_compliance": (),
        "driving_direction_compliance": (),
        "traffic_light_compliance": (),
        "ego_progress": (),
        "time_to_collision_within_bound": (),
        "lane_keeping": (),
        "history_comfort": ("comfort",),
        "two_frame_extended_comfort": (),
    }
    compact = pdms[["token", "score"]].copy()
    for metric, aliases in metric_aliases.items():
        compact[metric] = _combine_stage_metric(pdms, metric, aliases)
    compact["stage"] = _infer_pdms_stage(pdms)

    merged = per_scene.merge(compact, on="token", how="left", validate="one_to_one")
    matched = merged[merged["score"].notna()].copy()
    merged.to_csv(output_dir / "per_scene_gate_with_pdms.csv", index=False)

    matched["score_bin"] = np.select(
        [
            matched["score"].eq(0.0),
            matched["score"].lt(0.5),
            matched["score"].lt(0.9),
            matched["score"].lt(1.0),
        ],
        ["score_0", "score_0_to_0.5", "score_0.5_to_0.9", "score_0.9_to_1"],
        default="score_1",
    )

    layer_count = len([column for column in matched if column.endswith("_gate_selected")])
    group_rows = []
    masks = {
        "all": pd.Series(True, index=matched.index),
        "score_zero": matched["score"].eq(0.0),
        "score_perfect": matched["score"].eq(1.0),
        "stage_one": matched["stage"].eq("stage_one"),
        "stage_two": matched["stage"].eq("stage_two"),
        "noc_failure": matched["no_at_fault_collisions"].lt(1.0),
        "dac_failure": matched["drivable_area_compliance"].lt(1.0),
        "ttc_failure": matched["time_to_collision_within_bound"].lt(1.0),
        "low_progress": matched["ego_progress"].lt(0.5),
    }
    for score_bin in matched["score_bin"].dropna().unique():
        masks[f"bin_{score_bin}"] = matched["score_bin"].eq(score_bin)

    for group_name, mask in masks.items():
        subset = matched[mask]
        if subset.empty:
            continue
        row: Dict[str, Any] = {
            "group": group_name,
            "count": len(subset),
            "score_mean": float(subset["score"].mean()),
        }
        for layer_index in range(layer_count):
            row[f"layer_{layer_index}_gate_selected_mean"] = float(
                subset[f"layer_{layer_index}_gate_selected"].mean()
            )
            row[f"layer_{layer_index}_effective_patch_ratio_selected_mean"] = float(
                subset[f"layer_{layer_index}_effective_patch_ratio_selected"].mean()
            )
        group_rows.append(row)
    group_frame = pd.DataFrame(group_rows)
    group_frame.to_csv(output_dir / "pdms_group_summary.csv", index=False)

    correlation_rows = []
    for layer_index in range(layer_count):
        for metric in ("gate_selected", "effective_patch_ratio_selected"):
            column = f"layer_{layer_index}_{metric}"
            correlation_rows.append(
                {
                    "layer": layer_index,
                    "metric": metric,
                    "pearson_with_pdms": float(matched[column].corr(matched["score"])),
                    "spearman_with_pdms": float(
                        matched[column].corr(matched["score"], method="spearman")
                    ),
                }
            )
    correlation_frame = pd.DataFrame(correlation_rows)
    correlation_frame.to_csv(output_dir / "gate_pdms_correlations.csv", index=False)

    return {
        "pdms_csv": str(pdms_csv),
        "matched_scenes": int(len(matched)),
        "pdms_mean": float(matched["score"].mean()),
        "groups": group_frame.to_dict(orient="records"),
        "correlations": correlation_frame.to_dict(orient="records"),
    }


def _save_plots(data: Dict[str, np.ndarray], per_scene: pd.DataFrame, output_dir: Path) -> None:
    layer_count = data["gates"].shape[1]
    selected_index = data["selected_index"]
    selected_gate = _take_proposals(data["gates"], selected_index)
    selected_effective = _take_proposals(data["effective_patch_ratio"], selected_index)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for layer_index in range(layer_count):
        axes[0, 0].hist(
            selected_gate[:, layer_index], bins=50, alpha=0.45, label=f"layer {layer_index}"
        )
        axes[0, 1].hist(
            selected_effective[:, layer_index],
            bins=50,
            alpha=0.45,
            label=f"layer {layer_index}",
        )
    axes[0, 0].set_title("Selected-proposal raw patch gate")
    axes[0, 1].set_title("Selected-proposal effective patch ratio")
    axes[0, 0].set_xlabel("gate")
    axes[0, 1].set_xlabel("effective patch ratio")
    axes[0, 0].legend()
    axes[0, 1].legend()

    layers = np.arange(layer_count)
    axes[1, 0].plot(layers, data["gates"].mean(axis=(0, 2)), marker="o", label="all")
    axes[1, 0].plot(layers, selected_gate.mean(axis=0), marker="o", label="selected")
    axes[1, 0].axhline(0.5, color="black", linewidth=1, linestyle="--")
    axes[1, 0].set_title("Raw gate by scorer layer")
    axes[1, 0].set_xticks(layers)
    axes[1, 0].legend()

    axes[1, 1].plot(
        layers,
        data["effective_patch_ratio"].mean(axis=(0, 2)),
        marker="o",
        label="all",
    )
    axes[1, 1].plot(layers, selected_effective.mean(axis=0), marker="o", label="selected")
    axes[1, 1].axhline(0.5, color="black", linewidth=1, linestyle="--")
    axes[1, 1].set_title("Effective patch contribution by scorer layer")
    axes[1, 1].set_xticks(layers)
    axes[1, 1].legend()
    figure.savefig(output_dir / "gate_overview.png", dpi=180)
    plt.close(figure)

    pdms_path = output_dir / "per_scene_gate_with_pdms.csv"
    if pdms_path.exists():
        merged = pd.read_csv(pdms_path)
        merged = merged[merged["score"].notna()]
        if not merged.empty:
            figure, axes = plt.subplots(1, layer_count, figsize=(4 * layer_count, 4), sharey=True)
            if layer_count == 1:
                axes = [axes]
            for layer_index, axis in enumerate(axes):
                axis.hexbin(
                    merged[f"layer_{layer_index}_effective_patch_ratio_selected"],
                    merged["score"],
                    gridsize=35,
                    mincnt=1,
                    cmap="viridis",
                )
                axis.set_title(f"Layer {layer_index}")
                axis.set_xlabel("effective patch ratio")
            axes[0].set_ylabel("PDMS")
            figure.savefig(output_dir / "effective_patch_ratio_vs_pdms.png", dpi=180)
            plt.close(figure)


def _save_markdown_report(
    summary: Dict[str, Any], layer_summary: pd.DataFrame, output_dir: Path
) -> None:
    lines = [
        "# Dual-Branch Gate Analysis",
        "",
        f"- Unique scenes: {summary['unique_scenes']}",
        f"- DDP padding duplicates removed: {summary['duplicate_count']}",
        f"- Max duplicate gate difference: {summary['max_duplicate_gate_difference']:.8f}",
        "",
        "## Layer Summary",
        "",
        "| Layer | Scope | Raw gate mean | Effective patch ratio | Scene norm | Patch norm | Cosine |",
        "|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for _, row in layer_summary.iterrows():
        lines.append(
            f"| {int(row['layer'])} | {row['scope']} | {row['gate_mean']:.4f} | "
            f"{row['effective_patch_ratio_mean']:.4f} | {row['scene_norm_mean']:.4f} | "
            f"{row['patch_norm_mean']:.4f} | {row['branch_cosine_mean']:.4f} |"
        )

    pdms_summary = summary.get("pdms")
    if pdms_summary:
        lines.extend(
            [
                "",
                "## PDMS Join",
                "",
                f"- Matched scenes: {pdms_summary['matched_scenes']}",
                f"- Mean PDMS: {pdms_summary['pdms_mean']:.9f}",
                "- See `pdms_group_summary.csv` and `gate_pdms_correlations.csv` for conditional statistics.",
            ]
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    dataset = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True,
    )
    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)
    trainer = pl.Trainer(**cfg.trainer.params)
    predictions = trainer.predict(
        GateAnalysisLightningModule(agent=agent), dataloader, return_predictions=True
    )

    if _dist_ready():
        dist.barrier()
    world_size = dist.get_world_size() if _dist_ready() else 1
    all_predictions: List[Any] = [None for _ in range(world_size)]
    if _dist_ready():
        dist.all_gather_object(all_predictions, predictions)
    else:
        all_predictions = [predictions]

    rank = dist.get_rank() if _dist_ready() else 0
    if rank != 0:
        return

    data = _merge_predictions(all_predictions)
    expected_scenes = len(scene_loader.tokens)
    allow_partial = bool(cfg.get("analysis_allow_partial", False))
    if len(data["tokens"]) != expected_scenes and not allow_partial:
        raise RuntimeError(
            f"Expected {expected_scenes} unique scenes, collected {len(data['tokens'])}"
        )

    arrays_to_save = {
        key: value
        for key, value in data.items()
        if key not in ("duplicate_count", "max_duplicate_gate_difference")
    }
    np.savez_compressed(output_dir / "gate_analysis_full.npz", **arrays_to_save)

    layer_summary = _build_layer_summary(data)
    layer_summary.to_csv(output_dir / "layer_gate_summary.csv", index=False)
    per_scene = _build_per_scene_summary(data)
    per_scene.to_csv(output_dir / "per_scene_gate_summary.csv", index=False)

    summary: Dict[str, Any] = {
        "checkpoint": str(cfg.agent.checkpoint_path),
        "expected_scenes": expected_scenes,
        "unique_scenes": int(len(data["tokens"])),
        "num_layers": int(data["gates"].shape[1]),
        "num_proposals": int(data["gates"].shape[2]),
        "duplicate_count": int(data["duplicate_count"]),
        "max_duplicate_gate_difference": float(data["max_duplicate_gate_difference"]),
        "layer_summary": layer_summary.to_dict(orient="records"),
    }

    pdms_csv_value = str(cfg.get("analysis_pdms_csv", ""))
    if pdms_csv_value:
        pdms_csv = Path(pdms_csv_value)
        if not pdms_csv.is_file():
            raise FileNotFoundError(f"PDMS CSV does not exist: {pdms_csv}")
        summary["pdms"] = _add_pdms_analysis(per_scene, pdms_csv, output_dir)

    with (output_dir / "gate_analysis_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    _save_plots(data, per_scene, output_dir)
    _save_markdown_report(summary, layer_summary, output_dir)

    logger.info("Gate analysis completed: %s", output_dir)
    logger.info("Unique scenes: %d / %d", len(data["tokens"]), expected_scenes)


if __name__ == "__main__":
    main()