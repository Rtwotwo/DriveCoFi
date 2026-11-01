from typing import Any, List, Dict, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from scipy.optimize import linear_sum_assignment
from .load_balance_loss import compute_moe_auxiliary_loss

@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Function to calculate cross-entropy cost for cost matrix.
    :param gt_valid: tensor of binary ground-truth labels
    :param pred_logits: tensor of predicted logits of neural net
    :return: bce cost matrix as tensor
    """

    # NOTE: numerically stable BCE with logits
    # https://github.com/pytorch/pytorch/blob/c64e006fc399d528bb812ae589789d0365f3daf4/aten/src/ATen/native/Loss.cpp#L214
    gt_valid_expanded = gt_valid[:, :, None].detach().float()  # (b, n, 1)
    pred_logits_expanded = pred_logits[:, None, :].detach()  # (b, 1, n)

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(
        torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val)
    )
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term  # (b, n, n)
    ce_cost = ce_cost.permute(0, 2, 1)

    return ce_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor
) -> torch.Tensor:
    """
    Function to calculate L1 cost for cost matrix.
    :param gt_states: tensor of ground-truth bounding boxes
    :param pred_states: tensor of predicted bounding boxes
    :param gt_valid: mask of binary ground-truth labels
    :return: l1 cost matrix as tensor
    """

    gt_states_expanded = gt_states[:, :, None, :2].detach()  # (b, n, 1, 2)
    pred_states_expanded = pred_states[:, None, :, :2].detach()  # (b, 1, n, 2)
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(
        dim=-1
    )
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    """
    Helper function to align indices after matching
    :param indices: matched indices
    :return: permuted indices
    """
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx



def _agent_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], 
    agent_class_weight, agent_box_weight
):
    """
    Hungarian matching loss for agent detection
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: detection loss
    """

    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

    # if config.latent:
    #     # see transfuser loss

    # save constants
    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

    cost = agent_class_weight * ce_cost + agent_box_weight * l1_cost
    cost = cost.cpu()

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss

def three_to_two_classes(x):
    x[x == 0.5] = 0.0
    return x

class DrivoRLoss(torch.nn.Module):

    def __init__(self,
                        trajectory_weight: float = 1.0,
                        inter_weight: float = 1.0,
                        sub_score_weight: float = 1.0,
                        final_score_weight: float = 1.0,
                        pred_ce_weight: float = 1.0,
                        pred_l1_weight: float = 1.0,
                        pred_area_weight: float = 1.0,
                        prev_weight: float = 1.0,
                        agent_class_weight: float = 1.0,
                        agent_box_weight: float = 1.0,
                        bev_semantic_weight: float = 1.0,
                        load_balance_weight: float = 0.01,
                        entropy_weight: float = 0.001,
                        num_moe_experts: int = 6,
                        aux_score_weight: float = 0.5,
                        **kwargs):
        super().__init__()

        self.trajectory_weight = trajectory_weight
        self.inter_weight = inter_weight
        self.sub_score_weight = sub_score_weight
        self.final_score_weight = final_score_weight
        self.pred_ce_weight = pred_ce_weight
        self.pred_l1_weight = pred_l1_weight
        self.pred_area_weight = pred_area_weight
        self.prev_weight = prev_weight
        self.agent_class_weight = agent_class_weight
        self.agent_box_weight = agent_box_weight
        self.bev_semantic_weight = bev_semantic_weight
        self.load_balance_weight = load_balance_weight
        self.entropy_weight = entropy_weight
        self.num_moe_experts = num_moe_experts
        self.aux_score_weight = aux_score_weight


    def score_loss(self, pred_logit, pred_logit2, agents_state, pred_area_logits, target_scores, gt_states, gt_valid,
                   gt_ego_areas, l2_distance):


        pred_ce_loss = 0
        pred_l1_loss = 0
        pred_area_loss = 0

        ep_lw = 1.0

        trajectory_pdm_weight = {
            'no_at_fault_collisions': 1.0,
            'drivable_area_compliance': 1.0,
            'time_to_collision_within_bound': 1.0,
            'ego_progress': ep_lw,
            'driving_direction_compliance': 1.0,
            'lane_keeping': 1.0,
            'traffic_light_compliance': 1.0,
            'history_comfort': 1.0,
        }


        comfort = pred_logit['comfort']

        dtype = comfort.dtype

        no_at_fault_collisions, drivable_area_compliance, time_to_collision_within_bound, ego_progress = (
            pred_logit['no_at_fault_collisions'],
            pred_logit['drivable_area_compliance'],
            pred_logit['time_to_collision_within_bound'],
            pred_logit['ego_progress']
            )
        driving_direction_compliance = pred_logit['driving_direction_compliance']
        

        gt_no_at_fault_collisions, gt_drivable_area_compliance, gt_ego_progress, gt_time_to_collision_within_bound, gt_comfort, gt_driving_direction_compliance, _ = torch.split(target_scores, 1, dim=-1)
        gt_no_at_fault_collisions = gt_no_at_fault_collisions.squeeze(-1)
        gt_drivable_area_compliance = gt_drivable_area_compliance.squeeze(-1)
        gt_ego_progress = gt_ego_progress.squeeze(-1)
        gt_time_to_collision_within_bound = gt_time_to_collision_within_bound.squeeze(-1)
        gt_driving_direction_compliance = gt_driving_direction_compliance.squeeze(-1)
        gt_comfort = gt_comfort.squeeze(-1)

        da_loss = F.binary_cross_entropy_with_logits(drivable_area_compliance, gt_drivable_area_compliance.to(dtype))


        # print("gt_time_to_collision_within_bound:", gt_time_to_collision_within_bound.shape)
        # print("time_to_collision_within_bound:", time_to_collision_within_bound.shape)
        # print("gt_time_to_collision_within_bound:", gt_time_to_collision_within_bound)
        # time_to_collision_within_bound: torch.Size([16, 64])
        # gt_time_to_collision_within_bound: torch.Size([16, 64])
        # time_to_collision_within_bound: torch.Size([16, 64])

        mask_valid_ttc = (gt_time_to_collision_within_bound != 2.0).float()
        ttc_loss =F.binary_cross_entropy_with_logits(time_to_collision_within_bound, gt_time_to_collision_within_bound.to(dtype),  mask_valid_ttc, reduction='sum')/mask_valid_ttc.sum().clamp(min=1.0)
        # print("F.binary_cross_entropy_with_logits(time_to_collision_within_bound, gt_time_to_collision_within_bound.to(dtype), reduction='none').shape:", F.binary_cross_entropy_with_logits(time_to_collision_within_bound, gt_time_to_collision_within_bound.to(dtype), reduction='none').shape)
        # print("mask_valid_ttc.shape:", mask_valid_ttc.shape)

        # print("gt_no_at_fault_collisions:", gt_no_at_fault_collisions)
        noc_gt = three_to_two_classes(gt_no_at_fault_collisions.to(dtype))
        noc_loss = F.binary_cross_entropy_with_logits(no_at_fault_collisions, noc_gt)
        progress_loss = F.binary_cross_entropy_with_logits(ego_progress, gt_ego_progress.to(dtype))

        # print("gt_driving_direction_compliance:", gt_driving_direction_compliance)
        ddc_gt = three_to_two_classes(gt_driving_direction_compliance.to(dtype))

        ddc_loss = F.binary_cross_entropy_with_logits(driving_direction_compliance, ddc_gt)

        # imi_loss = F.cross_entropy(imi, l2_distance.sum((-2, -1)).softmax(1))
        comfort_loss = F.binary_cross_entropy_with_logits(comfort, gt_comfort.to(dtype))

        sub_score_loss = [da_loss, ttc_loss, noc_loss, progress_loss, ddc_loss, comfort_loss]

        final_score_loss = da_loss*trajectory_pdm_weight['drivable_area_compliance'] + \
                           ttc_loss*trajectory_pdm_weight['time_to_collision_within_bound'] + \
                           noc_loss*trajectory_pdm_weight['no_at_fault_collisions'] + \
                           progress_loss*trajectory_pdm_weight['ego_progress'] + \
                           ddc_loss*trajectory_pdm_weight['driving_direction_compliance'] + comfort_loss*trajectory_pdm_weight['history_comfort']

        return sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss

    def diversity_loss(self, proposals):
        dist = torch.linalg.norm(proposals[:, :, None] - proposals[:, None], dim=-1, ord=1).mean(-1)

        dist = dist + (dist == 0)

        inter_loss = -dist.amin(1).amin(1).mean()

        return inter_loss

    def forward(self,targets: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], config  , scoring_function=None):

        proposals = pred["proposals"]
        proposal_list = pred["proposal_list"]
        target_trajectory = targets["trajectory"]

        final_scores = pred.get("main_final_scores")
        best_scores = pred.get("main_best_scores")
        target_scores = pred.get("main_target_scores")
        gt_states = gt_valid = gt_ego_areas = None
        if final_scores is None or best_scores is None or target_scores is None:
            final_scores, best_scores, target_scores, gt_states, gt_valid, gt_ego_areas = scoring_function(
                targets, proposals, test=False)

        ########
        if "trajectory_long" in targets.keys():
            target_trajectory_long = targets["trajectory_long"]
        else:
            target_trajectory_long = None
        ########

        #############################
        ### TRAJECTORY AND DIVERSITY
        trajectory_loss = 0
        min_loss_list = []
        inter_loss_list = []
        for proposals_i in proposal_list:

            min_loss = torch.linalg.norm(proposals_i - target_trajectory[:, None], dim=-1, ord=1).mean(-1).amin(
                1).mean()

            #########
            if target_trajectory_long is not None:
                min_loss = min_loss + torch.linalg.norm(proposals_i - target_trajectory_long[:, None], dim=-1, ord=1).mean(-1).amin(
                    1).mean()
            #########

            inter_loss = self.diversity_loss(proposals_i)

            trajectory_loss = self.prev_weight * trajectory_loss  + min_loss + inter_loss * self.inter_weight

            min_loss_list.append(min_loss)
            inter_loss_list.append(inter_loss)
        min_loss0 = min_loss_list[0]
        inter_loss0 = inter_loss_list[0]
        # min_loss1 = min_loss_list[1]
        # inter_loss1 = inter_loss_list[1]
        l2_distance =  -((proposals.detach() - target_trajectory[:, None]) ** 2) / 0.5 # b,N,8,8
        
        if "pred_logit" in pred.keys():
            sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss = self.score_loss(
                pred["pred_logit"], pred["pred_logit2"],
                pred["pred_agents_states"], pred["pred_area_logit"]
                , target_scores, gt_states, gt_valid, gt_ego_areas, l2_distance.detach())
        else:
            sub_score_loss = final_score_loss = pred_ce_loss = pred_l1_loss = pred_area_loss = 0

        aux_score_loss = torch.tensor(0.0, device=proposals.device)
        aux_best_score = torch.tensor(0.0, device=proposals.device)
        aux_mean_score = torch.tensor(0.0, device=proposals.device)
        aux_positive_rate = torch.tensor(0.0, device=proposals.device)
        aux_dac0_rate = torch.tensor(0.0, device=proposals.device)
        aux_valid_rate = torch.tensor(1.0, device=proposals.device)
        if pred.get("aux_proposals") is not None and pred.get("aux_pred_logit") is not None:
            aux_proposals = pred["aux_proposals"]
            aux_target_scores = pred.get("aux_target_scores")
            aux_final_scores = pred.get("aux_final_scores")
            aux_best_scores = pred.get("aux_best_scores")
            if aux_target_scores is None or aux_final_scores is None or aux_best_scores is None:
                aux_final_scores, aux_best_scores, aux_target_scores, _, _, _ = scoring_function(
                    targets, aux_proposals, test=False)
            aux_l2_distance = -((aux_proposals.detach() - target_trajectory[:, None]) ** 2) / 0.5
            _, aux_score_loss, _, _, _ = self.score_loss(
                pred["aux_pred_logit"], pred.get("aux_pred_logit2"),
                pred.get("aux_pred_agents_states"), pred.get("aux_pred_area_logit"),
                aux_target_scores, gt_states, gt_valid, gt_ego_areas, aux_l2_distance.detach())
            aux_best_score = aux_best_scores.mean()
            aux_mean_score = aux_final_scores.mean()
            aux_positive_rate = (aux_final_scores > 0.8).float().mean()
            aux_dac0_rate = (aux_target_scores[:, :, 1] <= 0.0).float().mean()
            if pred.get("aux_expansion_valid_mask") is not None:
                aux_valid_rate = pred["aux_expansion_valid_mask"].float().mean()


        if pred["agent_states"] is not None:
            agent_class_loss, agent_box_loss = _agent_loss(targets, pred, config,
                                                           self.agent_class_weight, self.agent_box_weight)
        else:
            agent_class_loss = 0
            agent_box_loss = 0

        if pred["bev_semantic_map"] is not None:
            bev_semantic_loss = F.cross_entropy(pred["bev_semantic_map"], targets["bev_semantic_map"].long())
        else:
            bev_semantic_loss = 0

        # MoE auxiliary loss: load balance + entropy regularization
        moe_balance_loss = torch.tensor(0.0, device=proposals.device)
        if pred.get("moe_router_logits") is not None:
            moe_balance_loss = compute_moe_auxiliary_loss(
                router_logits=pred["moe_router_logits"],
                num_experts=self.num_moe_experts,
                load_balance_weight=self.load_balance_weight,
                entropy_weight=self.entropy_weight,
            )

        loss = (
                self.trajectory_weight * trajectory_loss
                # + self.sub_score_weight * sub_score_loss
                + self.final_score_weight * final_score_loss
            + self.aux_score_weight * aux_score_loss
                + self.pred_ce_weight * pred_ce_loss
                + self.pred_l1_weight * pred_l1_loss
                + self.pred_area_weight * pred_area_loss
                + self.agent_class_weight * agent_class_loss
                + self.agent_box_weight * agent_box_loss
                + self.bev_semantic_weight * bev_semantic_loss
                + moe_balance_loss

        )

        pdm_score = pred["pdm_score"].detach()
        top_proposals = torch.argmax(pdm_score, dim=1)
        score = final_scores[np.arange(len(final_scores)), top_proposals].mean()
        best_score = best_scores.mean()

        [da_loss, ttc_loss, noc_loss, progress_loss, ddc_loss, comfort_loss] = sub_score_loss

        loss_dict = {
            "loss": loss,
            "trajectory_loss": trajectory_loss,
            # 'sub_score_loss': sub_score_loss,
            "da_loss": da_loss,
            "ttc_loss": ttc_loss,
            "noc_loss": noc_loss,
            "progress_loss": progress_loss,
            "ddc_loss": ddc_loss,
            "comfort_loss": comfort_loss,
            'final_score_loss': final_score_loss,
            'aux_final_score_loss': aux_score_loss,
            'aux_best_score': aux_best_score,
            'aux_mean_score': aux_mean_score,
            'aux_positive_rate': aux_positive_rate,
            'aux_dac0_rate': aux_dac0_rate,
            'aux_valid_rate': aux_valid_rate,
            'pred_ce_loss': pred_ce_loss,
            'pred_l1_loss': pred_l1_loss,
            'pred_area_loss': pred_area_loss,
            "inter_loss0": inter_loss0,
            # "inter_loss1": inter_loss1,
            "inter_loss": inter_loss,
            "min_loss0": min_loss0,
            # "min_loss1": min_loss1,
            "min_loss": min_loss,
            "score": score,
            "best_score": best_score,
            "moe_balance_loss": moe_balance_loss.detach() if isinstance(moe_balance_loss, torch.Tensor) else moe_balance_loss,
        }

        # Add MoE monitoring metrics to loss_dict for TensorBoard
        if isinstance(pred.get("moe_monitor"), dict):
            for k, v in pred["moe_monitor"].items():
                loss_dict[f"moe/{k}"] = v

        return loss_dict