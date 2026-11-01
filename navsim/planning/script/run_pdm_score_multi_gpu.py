# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import json
import os
import pickle
import traceback
import uuid
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union
from dataclasses import asdict

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.distributed as dist
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import PDMResults, SensorConfig, Trajectory
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.evaluate.pdm_score import get_trajectory_as_array, pdm_score, transform_trajectory
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"


def score_proposals_independently(
    metric_cache, proposals: np.ndarray, simulator: PDMSimulator, scorer: PDMScorer
) -> np.ndarray:
    """Score proposals as independent baseline/proposal pairs after one simulation pass."""
    initial_ego_state = metric_cache.ego_state
    baseline_states = get_trajectory_as_array(
        metric_cache.trajectory, simulator.proposal_sampling, initial_ego_state.time_point
    )
    proposal_states = [
        get_trajectory_as_array(
            transform_trajectory(Trajectory(proposal), initial_ego_state),
            simulator.proposal_sampling,
            initial_ego_state.time_point,
        )
        for proposal in proposals
    ]
    trajectory_states = np.stack([baseline_states, *proposal_states])
    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)
    scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )

    multiplicative_scores = scorer._multi_metrics.prod(axis=0)
    raw_progress = scorer._progress_raw * multiplicative_scores
    proposal_progress = raw_progress[1:]
    pairwise_max_progress = np.maximum(raw_progress[0], proposal_progress)
    normalized_progress = np.ones_like(proposal_progress)
    above_threshold = pairwise_max_progress > scorer._config.progress_distance_threshold
    normalized_progress[above_threshold] = (
        proposal_progress[above_threshold] / pairwise_max_progress[above_threshold]
    )
    normalized_progress[~above_threshold & (multiplicative_scores[1:] == 0.0)] = 0.0

    proposal_weighted_metrics = scorer._weighted_metrics[:, 1:].copy()
    proposal_weighted_metrics[WeightedMetricIndex.PROGRESS] = normalized_progress
    metric_weights = scorer._config.weighted_metrics_array
    weighted_scores = (proposal_weighted_metrics * metric_weights[:, None]).sum(axis=0)
    weighted_scores /= metric_weights.sum()
    return multiplicative_scores[1:] * weighted_scores

def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[pd.DataFrame]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    model_trajectory = args[0]['model_trajectory']

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    # print(cfg)
    print("!!!!!!!!!!!!!!!!Path(cfg.metric_cache_path) ", Path(cfg.metric_cache_path))
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[Dict[str, Any]] = []
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            prediction = model_trajectory[token]
            if "proposals" in prediction:
                scorer_scores = np.asarray(prediction["scorer_predicted_scores"])
                lookup_scores = np.asarray(prediction["lookup_scores"])
                proposals = np.asarray(prediction["proposals"])
                if scorer_scores.shape != lookup_scores.shape or proposals.shape[0] != len(scorer_scores):
                    raise ValueError(f"Inconsistent three-way score shapes for token {token}")
                real_scores = score_proposals_independently(metric_cache, proposals, simulator, scorer)
                scorer_selected_index = int(np.argmax(scorer_scores))
                score_row.update(
                    {
                        "scorer_selected_real_pdms": float(real_scores[scorer_selected_index]),
                        "lookup_selected_offline_score": float(np.max(lookup_scores)),
                        "real_pdms_oracle_best": float(np.max(real_scores)),
                    }
                )
            else:
                pdm_result = pdm_score(
                    metric_cache=metric_cache,
                    model_trajectory=prediction["trajectory"],
                    future_sampling=simulator.proposal_sampling,
                    simulator=simulator,
                    scorer=scorer,
                )
                score_row.update(asdict(pdm_result))
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)

    return pdm_results

def dist_ready():
    return dist.is_available() and dist.is_initialized()

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    dump_root = os.path.join(os.getenv('SUBSCORE_PATH'), "navsim1_pdm_scores", cfg.experiment_name)
    os.makedirs(dump_root, exist_ok=True)
    dump_path = os.path.join(dump_root, f"{timestamp}.pkl")
    print(f'Subscore/Trajectories saved to {dump_path}')
    # gpu inference
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    scene_filter = instantiate(cfg.train_test_split.scene_filter)

    scene_loader_inference = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    dataset = Dataset(
        scene_loader=scene_loader_inference,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True,
    )
    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)
    trainer = pl.Trainer(**cfg.trainer.params)
    predictions = trainer.predict(
        AgentLightningModule(agent=agent),
        dataloader,
        return_predictions=True
    )
    
    if dist_ready():
        dist.barrier()
    
    world_size = dist.get_world_size() if dist_ready() else 1
    all_predictions = [None for _ in range(world_size)]

    if dist_ready():
        dist.all_gather_object(all_predictions, predictions)
    else:
        all_predictions = [predictions]

    rank = dist.get_rank() if dist_ready() else 0
    if rank != 0:
        return None

    merged_predictions = {}
    for proc_prediction in all_predictions:
        for d in proc_prediction:
            merged_predictions.update(d)

    pickle.dump(merged_predictions, open(dump_path, 'wb'))

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
            "model_trajectory": merged_predictions
        }
        for log_file, tokens_list in scene_loader_inference.get_tokens_list_per_log().items()
    ]

    worker = build_worker(cfg)
    score_rows: List[pd.DataFrame] = worker_map(worker, run_pdm_score, data_points)

    pdm_score_df = pd.DataFrame(score_rows)
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    three_way_columns = [
        "scorer_selected_real_pdms",
        "lookup_selected_offline_score",
        "real_pdms_oracle_best",
    ]
    if all(column in pdm_score_df for column in three_way_columns):
        three_way_summary = {column: float(average_row[column]) for column in three_way_columns}
        summary_path = Path(dump_root) / f"{timestamp}.three_way_summary.json"
        with summary_path.open("w") as file:
            json.dump(three_way_summary, file, indent=2)
        logger.info(f"Three-way best-then-mean scores: {three_way_summary}")
        logger.info(f"Three-way summary is stored in: {summary_path}")

    final_score_column = "scorer_selected_real_pdms" if "scorer_selected_real_pdms" in pdm_score_df else "score"
    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df[final_score_column].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.

            All scores:
            {average_row}
        """
    )


if __name__ == "__main__":
    main()