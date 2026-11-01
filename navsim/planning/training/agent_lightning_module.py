from time import sleep

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import Dict, Tuple, Any, List
from navsim.common.dataclasses import Trajectory
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory

def _rowwise_isin(tensor_1: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    matches = (tensor_1[:, None] == target_tensor)
    
    return torch.sum(matches, dim=1, dtype=torch.bool)


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent, for_viz = False):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self.checkpoint_file=None
        self.for_viz = for_viz
        # GRPO multi-step PPO uses manual optimization (rollout reuse +
        # PPO inner loop over minibatches). Only enabled if the agent asks.
        if getattr(agent, "manual_grpo", False):
            self.automatic_optimization = False
        self._grpo_batch_count = 0

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch

        if hasattr(self.agent, "set_training_progress") and self.trainer is not None:
            self.agent.set_training_progress(self.global_step, self.trainer.max_steps)

        # Expose the ground-truth trajectory to the model during training so
        # the scorer can include it as an extra candidate (drivor agents only).
        if (
            self.agent.training
            and isinstance(targets, dict)
            and "trajectory" in targets
            and "gt_trajectory" not in features
        ):
            features = {**features, "gt_trajectory": targets["trajectory"]}

        prediction = self.agent.forward(features)
        loss_dict = self.agent.compute_loss(features, targets, prediction)

        if type(loss_dict) is dict:
            sync_train_metrics = getattr(self.agent, "sync_train_metrics", True)
            for key,value in loss_dict.items():
                self.log(
                    f"{logging_prefix}/" + key,
                    value,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    sync_dist=sync_train_metrics,
                )
            return loss_dict["loss"]
        else:
            return loss_dict

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        if not self.automatic_optimization:
            return self._grpo_manual_step(batch)
        return self._step(batch, "train")

    def _grpo_manual_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]]) -> Tensor:
        """Manual-optimization GRPO step: one PDM rollout reused for K PPO
        epochs of minibatch updates (ratio = pi_theta / pi_old)."""
        features, targets = batch
        if hasattr(self.agent, "set_training_progress"):
            # batch-based progress (manual opt makes global_step count inner steps)
            self.agent.set_training_progress(self._grpo_batch_count, 0)
        self._grpo_batch_count += 1

        opt = self.optimizers()
        sch = self.lr_schedulers()

        cache = self.agent.grpo_rollout(features, targets)

        last_loss = None
        monitor = {}
        for _ in range(self.agent.ppo_epochs):
            for mb in self.agent.iter_minibatches(cache):
                loss, monitor = self.agent.grpo_inner_loss(mb)
                opt.zero_grad()
                self.manual_backward(loss)
                opt.step()
                last_loss = loss.detach()

        # scheduler configured with interval="step" assumes one step per batch;
        # step it once per batch here (manual opt disables auto scheduler stepping).
        if sch is not None:
            sch.step()

        self.log("train/loss", last_loss, on_step=True, on_epoch=False,
                 prog_bar=True, sync_dist=True)
        for k, v in monitor.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False,
                     prog_bar=False, sync_dist=True)
        return last_loss

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        if 'drivor' in self.agent.name() or "DrivoR" in self.agent.name():
            features, targets = batch
            # score,best_score=self.agent.inference(features, targets)
            predictions = self.agent.forward(features)
            if hasattr(self.agent, "compute_validation_scores"):
                validation_scores = self.agent.compute_validation_scores(targets, predictions)
                final_score = validation_scores["final_score"]
                proposal_scores = validation_scores["selected_scores"]
                best_score = validation_scores["best_score"]
                all_proposal_scores = validation_scores["all_proposal_scores"]
                l2 = validation_scores["l2"]
                trajectoy_scores = validation_scores["trajectory_scores"]
            else:
                all_chosen_trajectories = predictions["trajectory"][:,None]
                all_proposed_trajectories = predictions.get("expanded_proposals", predictions["proposals"])
                final_score, _, proposal_scores, l2, trajectoy_scores = self.agent.compute_score(targets, all_chosen_trajectories)
                _, best_score, all_proposal_scores, _, _ = self.agent.compute_score(targets, all_proposed_trajectories)
            mean_score=proposal_scores.mean()

            logging_prefix="val"
            if "pdm_score" in predictions:
                pdm_score = predictions["pdm_score"]
                # Diagnostic metrics only: coerce to 2D (B, num_proposals) and
                # guard empty shapes so a degenerate 1D batch cannot raise
                # "too many indices for tensor of dimension 1" and abort
                # validation. Under DDP that would hang the other ranks on the
                # sync_dist barrier until the NCCL watchdog aborts the job.
                if pdm_score.dim() == 1:
                    pdm_score = pdm_score.unsqueeze(1)
                if all_proposal_scores.dim() == 1:
                    all_proposal_scores = all_proposal_scores.unsqueeze(1)

                # Safe defaults so every DDP rank logs the same keys.
                score_error = torch.zeros((), device=pdm_score.device, dtype=pdm_score.dtype)
                score_hit_rate = torch.zeros((), device=pdm_score.device, dtype=torch.float32)
                lost_score = torch.zeros((), device=pdm_score.device, dtype=pdm_score.dtype)
                top_5_score_hit_rate = torch.zeros((), device=pdm_score.device, dtype=torch.float32)

                score_count = min(pdm_score.shape[1], all_proposal_scores.shape[1]) if (pdm_score.dim() >= 2 and all_proposal_scores.dim() >= 2) else 0
                if score_count >= 1:
                    pdm_score = pdm_score[:, :score_count]
                    all_proposal_scores = all_proposal_scores[:, :score_count]
                    row_indices = torch.arange(len(pdm_score), device=pdm_score.device)

                    best_pred_score_values = pdm_score[row_indices, torch.argmax(pdm_score, dim=1)]
                    score_error_pred = best_pred_score_values.reshape(-1)
                    score_error_target = proposal_scores.reshape(-1)
                    score_error_count = min(score_error_pred.shape[0], score_error_target.shape[0])
                    score_error_pred = score_error_pred[:score_error_count]
                    score_error_target = score_error_target[:score_error_count]
                    valid_score_error = torch.isfinite(score_error_pred) & torch.isfinite(score_error_target)
                    if valid_score_error.any():
                        score_error = torch.abs(score_error_pred[valid_score_error] - score_error_target[valid_score_error]).mean()

                    best_pred_score_index = torch.argmax(pdm_score, dim=1)
                    best_real_score_index = torch.argmax(all_proposal_scores, dim=1)
                    score_hit_rate = torch.mean(best_pred_score_index == best_real_score_index, dtype=torch.float32)

                    best_possible_scores = all_proposal_scores[row_indices, best_real_score_index]
                    best_actual_scores = all_proposal_scores[row_indices, best_pred_score_index]
                    lost_score = torch.mean(best_possible_scores - best_actual_scores)

                    top_5_indices_real = torch.topk(all_proposal_scores, k=min(5, score_count), dim=1).indices
                    top_5_score_hit_rate = _rowwise_isin(best_pred_score_index, top_5_indices_real).mean(dtype=torch.float32)

                self.log(f"{logging_prefix}/score_error", score_error, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"{logging_prefix}/score_hit_rate", score_hit_rate, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"{logging_prefix}/lost_score", lost_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"{logging_prefix}/top_5_score_hit_rate", top_5_score_hit_rate, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
            self.log(f"{logging_prefix}/score", final_score, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/best_score", best_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/mean_score", mean_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/l2", l2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            collision=trajectoy_scores[:,0].mean()
            self.log(f"{logging_prefix}/collision", collision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            drivable_area_compliance=trajectoy_scores[:,1].mean()
            self.log(f"{logging_prefix}/dac", drivable_area_compliance, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            ego_progress=trajectoy_scores[:,2].mean()
            self.log(f"{logging_prefix}/progress", ego_progress, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            time_to_collision_within_bound=trajectoy_scores[:,3].mean()
            self.log(f"{logging_prefix}/ttc", time_to_collision_within_bound, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            comfort=trajectoy_scores[:,4].mean()
            self.log(f"{logging_prefix}/comfort", comfort, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            if "scene_selected_count" in predictions:
                selected_count = predictions["scene_selected_count"].float().mean()
                selected_ratio = predictions["scene_selected_count"].float().mean() / predictions["scene_token_capacity"].float().clamp_min(1.0)
                self.log(f"{logging_prefix}/scene_selected_tokens", selected_count, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log(f"{logging_prefix}/scene_selected_ratio", selected_ratio, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            if "scene_selector_temperature" in predictions:
                self.log(f"{logging_prefix}/scene_selector_temperature", predictions["scene_selector_temperature"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

            # ---- Query-aware 剪枝监控（路线 B） ----
            if "qa_keep_ratio_mean" in predictions:
                self.log(f"{logging_prefix}/qa_keep_ratio", predictions["qa_keep_ratio_mean"].detach(),
                         on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            if "qa_attn_entropy_norm" in predictions:
                self.log(f"{logging_prefix}/qa_attn_entropy", predictions["qa_attn_entropy_norm"].mean().detach(),
                         on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            if "qa_pruned_token_count" in predictions and "qa_total_token_count" in predictions:
                total = max(predictions["qa_total_token_count"], 1)
                ratio = float(predictions["qa_pruned_token_count"]) / total
                self.log(f"{logging_prefix}/qa_pruned_token_ratio", ratio,
                         on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

            return final_score
        else:
            return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

    def teardown(self, stage: str) -> None:
        shutdown = getattr(self.agent, "shutdown_pdm_workers", None)
        if callable(shutdown):
            shutdown()
    
    def predict_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Used during the multi-gpu proccessing to parallelize the prediction of trajectories.
        NOTE: requires append_token_to_batch=True in the dataset used to instantiate the trainer.
        """
        return self.predict_step_drivor(batch, batch_idx)

    def predict_step_drivor(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], List[str]], batch_idx: int):
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            poses = predictions["trajectory"]
            if self.for_viz:
                all_proposed_trajectories = predictions["proposal_list"]
                final_trajectories = predictions["proposals"]
                _, _, final_scores, _, _ = self.agent.compute_score(targets, final_trajectories)
                ego_status = features["ego_status"]
        result = {}
        for index, (pose, token) in enumerate(zip(poses.cpu().numpy(), tokens)):
            proposal = Trajectory(pose)
            if self.for_viz:
                proposal_list = [proposal_list[index].cpu().numpy() for proposal_list in all_proposed_trajectories]
                result[token] = {
                    'trajectory': proposal, 
                    'all_proposals': proposal_list, 
                    'all_proposal_scores': final_scores[index],
                    'high_level_command': ego_status[index]
                }
            else:
                result[token] = {'trajectory': proposal}
                if "lookup_pdm_score" in predictions:
                    result[token].update(
                        {
                            "proposals": predictions["proposals"][index].cpu().numpy(),
                            "scorer_predicted_scores": predictions["learned_pdm_score"][index]
                            .cpu()
                            .numpy(),
                            "lookup_scores": predictions["lookup_pdm_score"][index].cpu().numpy(),
                        }
                    )
        return result
