from typing import Any, List, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from pathlib import Path
import pickle
from .drivor_model import DrivoRModel
from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.dataset import load_feature_target_from_pickle
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar, LearningRateMonitor
from navsim.planning.training.callbacks.memory_trace_callback import MemoryTraceCallback
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from .drivor_features import DrivoRTargetBuilder
from .drivor_features import DrivoRFeatureBuilder
import sys
from omegaconf import OmegaConf
import math

class LitProgressBar(ProgressBar):

    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True

    def disable(self):
        self.enable = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if batch_idx%100 == 0:
            print(f"Epoch {trainer.current_epoch} - train {batch_idx} / {self.total_train_batches} - {self.get_metrics(trainer, pl_module)}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if batch_idx%100 == 0:
            print(f"Epoch {trainer.current_epoch} - val {batch_idx} / {self.total_train_batches} - {self.get_metrics(trainer, pl_module)}")

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(self, pl_module)
        metrics = self.get_metrics(trainer, pl_module)
        train_metrics = dict()
        val_metrics = dict()
        other_metrics = dict()
        for k,v in metrics.items():
            if "train/" in k:
                train_metrics[k]=v
            elif "val/" in k:
                val_metrics[k]=v
            else:
                other_metrics[k]=v
        print(f"\n###########  Epoch {trainer.current_epoch} ##########")
        for k,v in train_metrics.items():
            print(f"{k},{v:.3f}")
        for k,v in val_metrics.items():
            print(f"{k},{v:.3f}")
        for k,v in other_metrics.items():
            print(f"{k},{v:.3f}")
        print(f"###########\n")

class DrivoRAgent(AbstractAgent):
    def __init__(
            self,
            config,
            lr_args: dict,
            checkpoint_path: str = None,
            loss: nn.Module = None,
            progress_bar: bool = True,
            scheduler_args: dict = None,
            batch_size: int = 64,
            num_gpus: int = 1,
            pdm_workers_per_rank: int = 16,
            pdm_proposal_chunk_size: int = 0,
    ):
        super().__init__()
        self._config = config
        self._lr_args = lr_args
        self._checkpoint_path = checkpoint_path
        self.progress_bar = progress_bar
        self.scheduler_args = scheduler_args
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.pdm_workers_per_rank = pdm_workers_per_rank
        if self.pdm_workers_per_rank < 1:
            raise ValueError("pdm_workers_per_rank must be at least 1")
        self.pdm_proposal_chunk_size = pdm_proposal_chunk_size
        if self.pdm_proposal_chunk_size < 0:
            raise ValueError("pdm_proposal_chunk_size must be non-negative")
        self.sync_train_metrics = False


        cache_data=False

        if not cache_data:
            self._drivor_model = DrivoRModel(config)

        if not cache_data and self._checkpoint_path == "": # only for training
            self.bce_logit_loss = nn.BCEWithLogitsLoss()
            self.b2d = config.b2d

            self.ray=True

            if self.ray:
                from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
                from nuplan.planning.utils.multithreading.worker_utils import worker_map
                self.worker = RayDistributedNoTorch(
                    threads_per_node=self.pdm_workers_per_rank,
                    log_to_driver=False,
                )
                self.worker_map=worker_map


            from .score_module.compute_navsim_score import get_scores

            metric_cache = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_cache"))
            try:
                # add synthetic metric_cache
                metric_cache_synthetic_0 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-0"))
                metric_cache_synthetic_1 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-1"))
                metric_cache_synthetic_2 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-2"))
                metric_cache_synthetic_3 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-3"))
                metric_cache_synthetic_4 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-4"))

                self.train_metric_cache_paths_synthetic = metric_cache_synthetic_0.metric_cache_paths
                self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_0.metric_cache_paths)
                self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_1.metric_cache_paths)
                self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_2.metric_cache_paths)
                self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_3.metric_cache_paths)
                self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_4.metric_cache_paths)

                self.test_metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic
            except:
                self.test_metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic = None

            self.test_metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic
            self.train_metric_cache_paths = metric_cache.metric_cache_paths
            self.test_metric_cache_paths = metric_cache.metric_cache_paths

            self.get_scores = get_scores

            self.loss = loss
            


    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""

        if self._checkpoint_path != "":
            if torch.cuda.is_available():
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
            else:
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                    "state_dict"]
            self.load_state_dict({k.replace("agent._drivor_model", "_drivor_model"): v for k, v in state_dict.items()})

    def get_sensor_config(self) :
        """Inherited, see superclass."""
        # return SensorConfig(
        #     cam_f0=[3],
        #     cam_l0=[3],
        #     cam_l1=[],
        #     cam_l2=[],
        #     cam_r0=[3],
        #     cam_r1=[],
        #     cam_r2=[],
        #     cam_b0=[3],
        #     lidar_pc=[],
        # )
        return SensorConfig(
            cam_f0=OmegaConf.to_object(self._config["cam_f0"]),
            cam_l0=OmegaConf.to_object(self._config["cam_l0"]),
            cam_l1=OmegaConf.to_object(self._config["cam_l1"]),
            cam_l2=OmegaConf.to_object(self._config["cam_l2"]),
            cam_r0=OmegaConf.to_object(self._config["cam_r0"]),
            cam_r1=OmegaConf.to_object(self._config["cam_r1"]),
            cam_r2=OmegaConf.to_object(self._config["cam_r2"]),
            cam_b0=OmegaConf.to_object(self._config["cam_b0"]),
            lidar_pc=OmegaConf.to_object(self._config["lidar_pc"]),
        )
    
    def get_target_builders(self) :
        return [DrivoRTargetBuilder(config=self._config)]

    def get_feature_builders(self) :
        return [DrivoRFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._drivor_model(features)

    def _compute_target_scores(self, targets, proposals):
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
            metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic
        else:
            metric_cache_paths = self.test_metric_cache_paths
            metric_cache_paths_synthetic = self.test_metric_cache_paths_synthetic

        proposals=proposals.detach()

        proposal_arrays = proposals.cpu().numpy()
        chunk_size = self.pdm_proposal_chunk_size

        if chunk_size <= 0:
            data_points = [
                {
                    "token": metric_cache_paths[token] if token in metric_cache_paths else metric_cache_paths_synthetic[token],
                    "poses": poses,
                }
                for token, poses in zip(targets["token"], proposal_arrays)
            ]

            if self.ray:
                all_res = self.worker_map(self.worker, self.get_scores, data_points)
            else:
                all_res = self.get_scores(data_points)

            return torch.FloatTensor(np.stack(all_res)).to(proposals.device)

        data_points = []
        chunk_meta = []
        for batch_idx, (token, poses) in enumerate(zip(targets["token"], proposal_arrays)):
            metric_cache_path = metric_cache_paths[token] if token in metric_cache_paths else metric_cache_paths_synthetic[token]
            for start_idx in range(0, poses.shape[0], chunk_size):
                end_idx = min(start_idx + chunk_size, poses.shape[0])
                data_points.append({"token": metric_cache_path, "poses": poses[start_idx:end_idx]})
                chunk_meta.append((batch_idx, start_idx, end_idx))

        if self.ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        sample_scores = [None] * len(proposal_arrays)
        for result, (batch_idx, start_idx, end_idx) in zip(all_res, chunk_meta):
            result = np.asarray(result)
            if sample_scores[batch_idx] is None:
                sample_scores[batch_idx] = np.empty(
                    (proposal_arrays[batch_idx].shape[0],) + result.shape[1:],
                    dtype=result.dtype,
                )
            sample_scores[batch_idx][start_idx:end_idx] = result

        return torch.FloatTensor(np.stack(sample_scores)).to(proposals.device)

    def compute_score(self, targets, proposals, test=True):
        target_trajectory = targets["trajectory"]
        proposals = proposals.detach()
        target_scores = self._compute_target_scores(targets, proposals)

        final_scores = target_scores[:, :, -1]

        best_scores = torch.amax(final_scores, dim=-1)

        if test:
            l2_2s = torch.linalg.norm(proposals[:, 0] - target_trajectory, dim=-1)[:, :4]

            return final_scores[:, 0].mean(), best_scores.mean(), final_scores, l2_2s.mean(), target_scores[:, 0]
        else:
            return final_scores, best_scores, target_scores, None, None, None

    def compute_validation_scores(self, targets, predictions):
        # pdm_score is produced over the EXPANDED proposal set (e.g. 256), while
        # predictions["proposals"] keeps only the original set (e.g. 64) for the
        # trajectory loss. Score the expanded set so argmax(pdm_score) stays in-bounds.
        proposals = predictions.get("expanded_proposals", predictions["proposals"]).detach()
        target_scores = self._compute_target_scores(targets, proposals)
        all_proposal_scores = target_scores[:, :, -1]

        selected_indices = torch.argmax(predictions["pdm_score"], dim=1)
        batch_indices = torch.arange(len(proposals), device=proposals.device)
        selected_scores = all_proposal_scores[batch_indices, selected_indices]
        selected_target_scores = target_scores[batch_indices, selected_indices]
        selected_proposals = proposals[batch_indices, selected_indices]
        l2_2s = torch.linalg.norm(
            selected_proposals - targets["trajectory"], dim=-1
        )[:, :4]

        return {
            "final_score": selected_scores.mean(),
            "selected_scores": selected_scores[:, None],
            "best_score": all_proposal_scores.amax(dim=-1).mean(),
            "all_proposal_scores": all_proposal_scores,
            "l2": l2_2s.mean(),
            "trajectory_scores": selected_target_scores,
        }

    def shutdown_pdm_workers(self) -> None:
        worker = getattr(self, "worker", None)
        if worker is not None:
            worker.shutdown()

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            pred: Dict[str, torch.Tensor],
    ) -> Dict:
        expansion_cfg = self._config.get("traj_expansion", {})
        use_aux_expansion = (
            self.training
            and expansion_cfg.get("enabled", False)
            and hasattr(self._drivor_model, "score_auxiliary_proposals")
            and "_ego_token" in pred
        )
        if use_aux_expansion:
            main_final_scores, main_best_scores, main_target_scores, _, _, _ = self.compute_score(
                targets, pred["proposals"], test=False)
            seed_topk = min(int(expansion_cfg.get("seed_topk", 16)), pred["proposals"].shape[1])
            seed_indices = torch.topk(main_final_scores, k=seed_topk, dim=1).indices
            pred.update({
                "main_final_scores": main_final_scores,
                "main_best_scores": main_best_scores,
                "main_target_scores": main_target_scores,
                "aux_seed_indices": seed_indices,
            })
            pred.update(self._drivor_model.score_auxiliary_proposals(features, pred, seed_indices))
            main_count = pred["proposals"].shape[1]
            aug_proposals = pred["aux_proposals"][:, main_count:]
            if aug_proposals.shape[1] > 0:
                aug_final_scores, _, aug_target_scores, _, _, _ = self.compute_score(
                    targets, aug_proposals, test=False)
                aux_final_scores = torch.cat([main_final_scores, aug_final_scores], dim=1)
                aux_target_scores = torch.cat([main_target_scores, aug_target_scores], dim=1)
            else:
                aux_final_scores = main_final_scores
                aux_target_scores = main_target_scores
            aux_best_scores = aux_final_scores.amax(dim=1)
            pred.update({
                "aux_final_scores": aux_final_scores,
                "aux_best_scores": aux_best_scores,
                "aux_target_scores": aux_target_scores,
            })
        return self.loss(targets, pred, self._config, self.compute_score)

    def get_optimizers(self):

        global_batchsize = self.batch_size * self.num_gpus
        if self._lr_args["name"] == "Adam":
            lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
            optimizer = torch.optim.Adam(self._drivor_model.parameters(), lr=lr)
        elif self._lr_args["name"] == "AdamW":
            lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
            optimizer = torch.optim.AdamW(self._drivor_model.parameters(), lr=lr)
        else:
            raise NotImplementedError

        if self.scheduler_args is not None:

            T_max = int(math.ceil(self.scheduler_args.dataset_size / global_batchsize) *  self.scheduler_args.num_epochs)
            warmup_fraction = float(os.getenv("NAVSIM_LR_WARMUP_FRACTION", "0.1"))
            if warmup_fraction < 0.0 or warmup_fraction >= 1.0:
                raise ValueError("NAVSIM_LR_WARMUP_FRACTION must be in [0, 1)")

            # classic cosine
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer,
            #     T_max=T_max, 
            #     eta_min=0.0, last_epoch=-1
            # )

            # Ramp + cosine
            T_max_ramp = int(T_max * warmup_fraction)
            if T_max_ramp == 0:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=T_max,
                    eta_min=0.0,
                    last_epoch=-1,
                )
            else:
                scheduler_ramp = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=T_max_ramp)
                T_max_cosine = T_max - T_max_ramp
                scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=T_max_cosine,
                    eta_min=0.0, last_epoch=-1
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[scheduler_ramp, scheduler_cosine],
                    milestones=[T_max_ramp],
                )           

            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
        else:
            return [optimizer]

    def get_training_callbacks(self):

        checkpoint_dir = os.getenv("NAVSIM_CHECKPOINT_DIR") or None
        checkpoint_filename_prefix = os.getenv("NAVSIM_CHECKPOINT_FILENAME_PREFIX", "")

        checkpoint_cb_best = ModelCheckpoint(save_top_k=1,
                                        monitor='val/score_epoch',
                                        filename=f'{checkpoint_filename_prefix}best-{{epoch}}-{{step}}',
                                        mode="max",
                                        dirpath=checkpoint_dir,
                                        )
        
        checkpoint_cb_save_all = ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_last=True,
            save_top_k=-1,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            filename=f'{checkpoint_filename_prefix}epoch={{epoch}}-step={{step}}',
            auto_insert_metric_name=False,
        )

        lr_monitor = LearningRateMonitor(logging_interval="step", 
                                            log_momentum=False,
                                            log_weight_decay=False)

        enable_memory_profile = os.getenv("NAVSIM_MEMORY_PROFILE", "0") == "1"
        memory_profile_interval = int(os.getenv("NAVSIM_MEMORY_PROFILE_INTERVAL", "10"))
        memory_trace_cb = MemoryTraceCallback(interval_steps=memory_profile_interval)
        
        if self.progress_bar:
            callbacks = [checkpoint_cb_best, checkpoint_cb_save_all, lr_monitor]
        else:
            progress_bar = LitProgressBar()
            callbacks = [checkpoint_cb_best, checkpoint_cb_save_all, progress_bar, lr_monitor]

        if enable_memory_profile:
            callbacks.append(memory_trace_cb)

        return callbacks
