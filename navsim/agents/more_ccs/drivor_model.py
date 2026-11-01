from typing import Dict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import open_dict
from .score_module.scorer import Scorer
from .transformer_decoder import TransformerDecoder, TransformerDecoderScorer
from .ccs.bicycle import BicycleModel
from .layers.image_encoder.dinov2_lora import ImgEncoder
from .layers.utils.mlp import MLP
from .layers.moe.moe_scene_adaptive import SceneAdaptiveMoE
from .utils import pylogger
log = pylogger.get_pylogger(__name__)
import logging
# log.setLevel(logging.DEBUG)


def _build_2d_sincos_position_embedding(height: int, width: int, dim: int) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("2D sine-cosine position embedding dimension must be divisible by 4")

    quarter_dim = dim // 4
    omega = torch.arange(quarter_dim, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(quarter_dim - 1, 1)))
    y = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    x = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    x_angles = grid_x.reshape(-1, 1) * omega.reshape(1, -1) * (2 * math.pi)
    y_angles = grid_y.reshape(-1, 1) * omega.reshape(1, -1) * (2 * math.pi)
    return torch.cat(
        [x_angles.sin(), x_angles.cos(), y_angles.sin(), y_angles.cos()],
        dim=-1,
    ).reshape(height, width, dim)

class DrivoRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num=config.num_poses
        self.state_size=3
        self.embed_dims = self._config.tf_d_model

        ###########################################
        # camera embedding
        self.num_cams = 0
        if len(self._config["cam_f0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_b0"]) > 0:
            self.num_cams += 1

        # Scene-Adaptive MoE: placed between ViT output and trajectory decoder
        moe_cfg = self._config.get("moe", {})
        moe_enabled = moe_cfg.get("enabled", False)
        self.moe = None
        if moe_enabled:
            self.moe = SceneAdaptiveMoE(
                d_model=self._config["tf_d_model"],
                d_ffn=moe_cfg.get("d_ffn", 1024),
                num_experts=moe_cfg.get("num_experts", 6),
                dropout=moe_cfg.get("dropout", 0.1),
                jitter_noise=moe_cfg.get("jitter_noise", 0.1),
            )
            log.info(
                f"Scene-Adaptive MoE enabled: "
                f"num_experts={moe_cfg.get('num_experts', 6)}, "
                f"d_ffn={moe_cfg.get('d_ffn', 1024)}"
            )
        else:
            log.info("Scene-Adaptive MoE disabled")

        ############################################
        # lidar embedding
        self.num_lidar = 0
        if len(self._config["lidar_pc"]) > 0:
            self.num_lidar += 1

        # create the image backbone
        if self.num_cams > 0:
            config_image_backbone = config["image_backbone"]
            with open_dict(config_image_backbone):
                config_image_backbone["image_size"] = config["image_size"]
                config_image_backbone["num_scene_tokens"] = config["num_scene_tokens"]
                config_image_backbone["tf_d_model"] = config["tf_d_model"]
            self.image_backbone = ImgEncoder(config_image_backbone)
            self.scene_embeds = nn.Parameter(torch.randn(1, self.num_cams, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

            scorer_raw_cfg = self._config.get("scorer_raw_tokens", {})
            self.scorer_raw_tokens_enabled = scorer_raw_cfg.get("enabled", False)
            if self.scorer_raw_tokens_enabled:
                self.scorer_pool_height = scorer_raw_cfg.get("pool_height", 12)
                self.scorer_pool_width = scorer_raw_cfg.get("pool_width", 21)
                self.scorer_include_scene_tokens = scorer_raw_cfg.get("include_scene_tokens", True)
                self.scorer_detach_patch_backbone = scorer_raw_cfg.get("detach_patch_backbone", True)
                self.scorer_patch_projection = nn.Sequential(
                    nn.LayerNorm(self.image_backbone.num_features),
                    nn.Linear(self.image_backbone.num_features, config.tf_d_model),
                    nn.GELU(),
                    nn.Linear(config.tf_d_model, config.tf_d_model),
                )
                self.scorer_camera_embedding = nn.Embedding(self.num_cams, config.tf_d_model)
                self.scorer_token_type_embedding = nn.Embedding(2, config.tf_d_model)
                self.scorer_scene_kv_norm = nn.LayerNorm(config.tf_d_model)
                self.scorer_patch_kv_norm = nn.LayerNorm(config.tf_d_model)
                self.register_buffer(
                    "scorer_grid_position",
                    _build_2d_sincos_position_embedding(
                        self.scorer_pool_height,
                        self.scorer_pool_width,
                        config.tf_d_model,
                    ),
                    persistent=False,
                )
                nn.init.trunc_normal_(self.scorer_camera_embedding.weight, std=0.02)
                nn.init.trunc_normal_(self.scorer_token_type_embedding.weight, std=0.02)
        else:
            self.scorer_raw_tokens_enabled = False

            # print("self.scene_embeds ", self.scene_embeds)

        # create the lidar backbone
        if self.num_lidar > 0:
            config_lidar_backbone = config["lidar_backbone"]
            with open_dict(config_lidar_backbone):
                config_lidar_backbone["image_size"] = config["lidar_image_size"]
                config_lidar_backbone["num_scene_tokens"] = config["num_scene_tokens"]
                config_lidar_backbone["tf_d_model"] = config["tf_d_model"]
            self.lidar_backbone = ImgEncoder(config_lidar_backbone)
            self.lidar_scene_embeds = nn.Parameter(torch.randn(1, self.num_lidar, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

        # ego status encoder
        if self._config.full_history_status:
            self.hist_encoding = nn.Linear(11*4, config.tf_d_model)
        else:
            self.hist_encoding = nn.Linear(11, config.tf_d_model)

        # trajectory embdedding
        if self._config.one_token_per_traj:
            self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.poses_num*self.state_size
        else:
            self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)
            traj_head_output_size =self.state_size

        # trajectory decoder
        self.trajectory_decoder = TransformerDecoder(proj_drop=0.1, drop_path=0.2, config=config)

        # scorer decoder
        self.scorer_attention = TransformerDecoderScorer(num_layers=config.scorer_ref_num, d_model=config.tf_d_model, proj_drop=0.1, drop_path=0.2, config=config)

        self.pos_embed = nn.Sequential(
                nn.Linear(self.poses_num * 3, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, config.tf_d_model),
            )


        # get the trajectory decoders
        self.poses_num=config.num_poses
        self.state_size=3
        ref_num=config.ref_num
        self.traj_head = nn.ModuleList([MLP(config.tf_d_model, config.tf_d_ffn,  traj_head_output_size) for _ in range(ref_num+1)])

        # scorer
        self.scorer = Scorer(config)

        self.b2d=config.b2d

        # trajectory expansion via bicycle-model control-space perturbation
        expansion_cfg = self._config.get("traj_expansion", {})
        self.traj_expansion_enabled = expansion_cfg.get("enabled", False)
        self.traj_expansion_inference_enabled = expansion_cfg.get("inference_enabled", True)
        if self.traj_expansion_enabled:
            dt = float(self._config.trajectory_sampling.interval_length)
            self.bicycle = BicycleModel(dt=dt)
            self.exp_num_augment_per = int(expansion_cfg.get("num_augment_per", 3))
            self.exp_seed_topk = int(expansion_cfg.get("seed_topk", 16))
            self.exp_num_augment_per_seed = int(expansion_cfg.get("num_augment_per_seed", 12))
            self.exp_eps_accel = float(expansion_cfg.get("eps_accel", 0.3))
            self.exp_eps_yawrate = float(expansion_cfg.get("eps_yawrate", 0.1))
            self.exp_min_accel = float(expansion_cfg.get("min_accel", -4.05))
            self.exp_max_accel = float(expansion_cfg.get("max_accel", 2.40))
            self.exp_max_yawrate = float(expansion_cfg.get("max_yawrate", 0.95))
            self.exp_noise_alpha = float(expansion_cfg.get("noise_smoothing_alpha", 0.8))
            self.exp_noise_bias_scale = float(expansion_cfg.get("noise_bias_scale", 0.7))
            self.exp_noise_smooth_scale = float(expansion_cfg.get("noise_smooth_scale", 0.3))
            self.exp_max_resample_attempts = int(expansion_cfg.get("max_resample_attempts", 4))
            self.exp_min_velocity = float(expansion_cfg.get("min_velocity", -0.2))
            self.exp_max_lateral_deviation = float(expansion_cfg.get("max_lateral_deviation", 2.0))
            self.exp_max_xy_deviation = float(expansion_cfg.get("max_xy_deviation", 6.0))
            self.exp_max_heading_deviation = float(expansion_cfg.get("max_heading_deviation", 0.8))
            log.info(f"Trajectory expansion enabled: topk={self.exp_seed_topk}, n_aug_per_seed={self.exp_num_augment_per_seed}, dt={dt}, eps_accel={self.exp_eps_accel}, eps_yawrate={self.exp_eps_yawrate}")
        else:
            log.info("Trajectory expansion disabled")

    @torch.no_grad()
    def _expand_proposals(self, proposals: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
        """Expand proposals (B, N0, T, 3) into (B, N0*(1+n_aug), T, 3) via bicycle-model
        control-space perturbation. The first N0 entries are the original proposals."""
        B, N0, T, _ = proposals.shape
        n_aug = self.exp_num_augment_per
        if n_aug <= 0:
            return proposals
        v0_seed = v0[:, None].expand(B, N0)
        controls = self.bicycle.inverse(proposals, v0_seed)  # (B, N0, T, 2)
        controls_rep = controls[:, :, None, :, :].expand(B, N0, n_aug, T, 2).reshape(B, N0 * n_aug, T, 2)
        noise_a = torch.randn_like(controls_rep[..., 0]) * self.exp_eps_accel
        noise_w = torch.randn_like(controls_rep[..., 1]) * self.exp_eps_yawrate
        controls_aug = controls_rep + torch.stack([noise_a, noise_w], dim=-1)
        accel = controls_aug[..., 0].clamp(self.exp_min_accel, self.exp_max_accel)
        yaw_rate = controls_aug[..., 1].clamp(-self.exp_max_yawrate, self.exp_max_yawrate)
        controls_aug = torch.stack([accel, yaw_rate], dim=-1)
        v0_aug = v0[:, None].expand(B, N0 * n_aug)
        aug_proposals = self.bicycle.forward(controls_aug, v0_aug)  # (B, N0*n_aug, T, 3)
        return torch.cat([proposals, aug_proposals], dim=1)

    @torch.no_grad()
    def _smooth_control_noise(self, shape, device, dtype, std: float) -> torch.Tensor:
        B, M, T = shape
        white = torch.randn(B, M, T, device=device, dtype=dtype)
        smooth = torch.empty_like(white)
        smooth[..., 0] = white[..., 0]
        alpha = max(0.0, min(self.exp_noise_alpha, 0.99))
        innovation_scale = math.sqrt(max(1.0 - alpha * alpha, 1e-4))
        for step in range(1, T):
            smooth[..., step] = alpha * smooth[..., step - 1] + innovation_scale * white[..., step]
        smooth = smooth / smooth.std(dim=-1, keepdim=True).clamp_min(1e-4)
        bias = torch.randn(B, M, 1, device=device, dtype=dtype) * std * self.exp_noise_bias_scale
        return bias + smooth * std * self.exp_noise_smooth_scale

    @torch.no_grad()
    def _sample_seeded_aug_proposals(self, seed_proposals, seed_controls, v0_aug):
        B, M, T, _ = seed_controls.shape
        noise_a = self._smooth_control_noise((B, M, T), seed_controls.device, seed_controls.dtype, self.exp_eps_accel)
        noise_w = self._smooth_control_noise((B, M, T), seed_controls.device, seed_controls.dtype, self.exp_eps_yawrate)
        controls_aug = seed_controls + torch.stack([noise_a, noise_w], dim=-1)
        accel = controls_aug[..., 0].clamp(self.exp_min_accel, self.exp_max_accel)
        yaw_rate = controls_aug[..., 1].clamp(-self.exp_max_yawrate, self.exp_max_yawrate)
        controls_aug = torch.stack([accel, yaw_rate], dim=-1)
        aug_proposals = self.bicycle.forward(controls_aug, v0_aug)
        velocity = self.bicycle.rollout_velocity(controls_aug, v0_aug)

        xy_deviation = torch.linalg.norm(aug_proposals[..., :2] - seed_proposals[..., :2], dim=-1).amax(dim=-1)
        lateral_deviation = (aug_proposals[..., 1] - seed_proposals[..., 1]).abs().amax(dim=-1)
        heading_deviation = (aug_proposals[..., 2] - seed_proposals[..., 2]).abs().amax(dim=-1)
        valid = (
            (velocity.amin(dim=-1) >= self.exp_min_velocity)
            & (lateral_deviation <= self.exp_max_lateral_deviation)
            & (xy_deviation <= self.exp_max_xy_deviation)
            & (heading_deviation <= self.exp_max_heading_deviation)
        )
        return aug_proposals, valid

    @torch.no_grad()
    def _expand_topk_proposals(self, proposals: torch.Tensor, v0: torch.Tensor, seed_indices: torch.Tensor):
        B, _, T, S = proposals.shape
        K = seed_indices.shape[1]
        n_aug = self.exp_num_augment_per_seed
        if K <= 0 or n_aug <= 0:
            valid_mask = torch.ones(B, proposals.shape[1], device=proposals.device, dtype=torch.bool)
            return proposals, valid_mask

        gather_index = seed_indices[:, :, None, None].expand(B, K, T, S)
        selected_proposals = proposals.gather(1, gather_index)
        seed_controls = self.bicycle.inverse(selected_proposals, v0[:, None].expand(B, K))

        M = K * n_aug
        seed_proposals = selected_proposals[:, :, None].expand(B, K, n_aug, T, S).reshape(B, M, T, S)
        seed_controls = seed_controls[:, :, None].expand(B, K, n_aug, T, 2).reshape(B, M, T, 2)
        v0_aug = v0[:, None].expand(B, M)

        best_aug = seed_proposals.clone()
        valid_out = torch.zeros(B, M, device=proposals.device, dtype=torch.bool)
        for _ in range(max(self.exp_max_resample_attempts, 1)):
            aug_proposals, valid = self._sample_seeded_aug_proposals(seed_proposals, seed_controls, v0_aug)
            replace = (~valid_out) & valid
            best_aug = torch.where(replace[:, :, None, None], aug_proposals, best_aug)
            valid_out = valid_out | valid
            if bool(valid_out.all()):
                break

        aux_proposals = torch.cat([proposals, best_aug], dim=1)
        valid_mask = torch.cat(
            [torch.ones(B, proposals.shape[1], device=proposals.device, dtype=torch.bool), valid_out],
            dim=1,
        )
        return aux_proposals, valid_mask

    def _build_scorer_kv(self, scene_features, image_patch_tokens):
        if not self.scorer_raw_tokens_enabled:
            raise RuntimeError("Dual-branch gated scorer requires raw patch tokens")
        if image_patch_tokens is None:
            raise RuntimeError("Raw scorer tokens require image patch features")

        patch_tokens = image_patch_tokens.detach() if self.scorer_detach_patch_backbone else image_patch_tokens
        batch_size, num_cameras, _, _, feature_dim = patch_tokens.shape
        patch_tokens = patch_tokens.permute(0, 1, 4, 2, 3).reshape(
            batch_size * num_cameras, feature_dim, patch_tokens.shape[2], patch_tokens.shape[3]
        )
        patch_tokens = F.adaptive_avg_pool2d(
            patch_tokens,
            (self.scorer_pool_height, self.scorer_pool_width),
        )
        patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(
            batch_size,
            num_cameras,
            self.scorer_pool_height,
            self.scorer_pool_width,
            feature_dim,
        )
        patch_tokens = self.scorer_patch_projection(patch_tokens)

        camera_ids = torch.arange(num_cameras, device=patch_tokens.device)
        camera_position = self.scorer_camera_embedding(camera_ids)[None, :, None, None, :]
        raw_type = self.scorer_token_type_embedding.weight[1].view(1, 1, 1, 1, -1)
        patch_tokens = (
            patch_tokens
            + self.scorer_grid_position[None, None].to(patch_tokens.dtype)
            + camera_position
            + raw_type
        )
        patch_tokens = patch_tokens.reshape(batch_size, -1, self.embed_dims)

        if not self.scorer_include_scene_tokens:
            raise RuntimeError("Dual-branch gated scorer requires scene tokens")

        scene_camera_ids = torch.arange(num_cameras, device=scene_features.device)
        scene_camera_position = self.scorer_camera_embedding(scene_camera_ids)
        scene_camera_position = scene_camera_position.repeat_interleave(
            self._config.num_scene_tokens,
            dim=0,
        )[None]
        scene_type = self.scorer_token_type_embedding.weight[0].view(1, 1, -1)
        scene_tokens = scene_features + scene_camera_position + scene_type

        return (
            self.scorer_scene_kv_norm(scene_tokens),
            self.scorer_patch_kv_norm(patch_tokens),
        )

    def _predicted_pdm_score(self, pred_logit):
        additive_score = (
            self._config.ttc * pred_logit['time_to_collision_within_bound'].sigmoid()
            + self._config.ep * pred_logit['ego_progress'].sigmoid()
            + self._config.comfort * pred_logit['comfort'].sigmoid()
        ).clamp_min(1e-6)
        return (
            self._config.noc * F.logsigmoid(pred_logit['no_at_fault_collisions'])
            + self._config.dac * F.logsigmoid(pred_logit['drivable_area_compliance'])
            + self._config.ddc * F.logsigmoid(pred_logit['driving_direction_compliance'])
            + additive_score.log()
        )

    def _score_candidate_set(self, scoring_proposals, ego_token, scorer_scene_features, scorer_patch_features):
        B, N, _, _ = scoring_proposals.shape
        embedded_traj = self.pos_embed(scoring_proposals.reshape(B, N, -1).detach())
        tr_out, scorer_patch_gates = self.scorer_attention(
            embedded_traj,
            scorer_scene_features,
            scorer_patch_features,
        )
        tr_out = tr_out + ego_token
        pred_logit, pred_logit2, pred_agents_states, pred_area_logit, bev_semantic_map, agent_states, agent_labels = self.scorer(scoring_proposals, tr_out)
        return {
            "pred_logit": pred_logit,
            "pred_logit2": pred_logit2,
            "pred_agents_states": pred_agents_states,
            "pred_area_logit": pred_area_logit,
            "bev_semantic_map": bev_semantic_map,
            "agent_states": agent_states,
            "agent_labels": agent_labels,
            "scorer_patch_gates": scorer_patch_gates,
            "pdm_score": self._predicted_pdm_score(pred_logit),
        }

    def score_auxiliary_proposals(self, features, pred, seed_indices):
        v0 = features["ego_status"][:, -1, 3]
        aux_proposals, valid_mask = self._expand_topk_proposals(pred["proposals"], v0, seed_indices)
        aux = self._score_candidate_set(
            aux_proposals,
            pred["_ego_token"],
            pred["_scorer_scene_features"],
            pred["_scorer_patch_features"],
        )
        return {
            "aux_proposals": aux_proposals,
            "aux_expansion_valid_mask": valid_mask,
            "aux_pred_logit": aux["pred_logit"],
            "aux_pred_logit2": aux["pred_logit2"],
            "aux_pred_agents_states": aux["pred_agents_states"],
            "aux_pred_area_logit": aux["pred_area_logit"],
            "aux_pdm_score": aux["pdm_score"],
            "aux_scorer_patch_gates": aux["scorer_patch_gates"],
        }


    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # ego status and initial traj tokens
        if self._config.full_history_status:
            ego_status: torch.Tensor = features["ego_status"].flatten(-2)
        else:
            ego_status: torch.Tensor = features["ego_status"][:, -1]
        
        ego_token = self.hist_encoding(ego_status)[:, None]
        log.debug(f"Ego features - {ego_token.shape}")
        traj_tokens = ego_token + self.init_feature.weight[None]
        log.debug(f"Traj tokens initial - {traj_tokens.shape}")


        batch_size = ego_status.shape[0]



        scene_features = []
        image_patch_tokens = None
        # image features
        if self.num_cams > 0:
            
            if "image" in features :
                img = features["image"]
            elif "camera_feature" in features:
                img = features["camera_feature"]
            else:
                raise ValueError

            scene_tokens = self.scene_embeds.repeat(batch_size, 1, 1, 1)
            image_scene_tokens, image_patch_tokens = self.image_backbone(
                img,
                scene_tokens,
                return_patch_tokens=True,
            )

            log.debug(f"Backbone image - {image_scene_tokens.shape}")
            scene_features.append(image_scene_tokens)

        # lidar features
        if self.num_lidar > 0:
            img = features["lidar_feature"]
            scene_tokens = self.lidar_scene_embeds.repeat(batch_size, 1, 1, 1)
            lidar_scene_tokens = self.lidar_backbone(img, scene_tokens)
            log.debug(f"Backbone lidar - {lidar_scene_tokens.shape}")
            scene_features.append(lidar_scene_tokens)

        scene_features = torch.cat(scene_features, dim=1)
        log.debug(f"Scene features - {scene_features.shape}")

        # Pass scene features through Scene-Adaptive MoE
        router_logits = None
        router_weights = None
        moe_monitor = {}
        if self.moe is not None:
            scene_features, router_logits, router_weights = self.moe(scene_features)
            log.debug(f"Scene features after MoE - {scene_features.shape}")
            moe_monitor = self.moe.get_monitor_data(router_weights)

        scorer_scene_features, scorer_patch_features = self._build_scorer_kv(
            scene_features, image_patch_tokens)

        # initial trajectories
        proposals = self.traj_head[0](traj_tokens).reshape(traj_tokens.shape[0], -1, self.poses_num, self.state_size)
        proposal_list = [proposals]
        log.debug(f"Proposals initial - {proposals.shape}")

        # decode the trajectories at each step of the decoder
        token_list = self.trajectory_decoder(traj_tokens, scene_features)
        log.debug(f"Trajectory decoder - {len(token_list)}")
        for i in range(self._config.ref_num):
            tokens = token_list[i]
            proposals = self.traj_head[i+1](tokens).reshape(tokens.shape[0], -1, self.poses_num, self.state_size)
            proposal_list.append(proposals)
        
        traj_tokens = token_list[-1]
        proposals=proposal_list[-1]
        

        output={}
        output["proposals"] = proposals
        output["proposal_list"] = proposal_list

        if (not self.training) and self.traj_expansion_enabled and self.traj_expansion_inference_enabled:
            v0 = features["ego_status"][:, -1, 3]  # (B,) longitudinal speed vx
            scoring_proposals = self._expand_proposals(proposals, v0)
        else:
            scoring_proposals = proposals
        if scoring_proposals.shape[1] != proposals.shape[1]:
            output["expanded_proposals"] = scoring_proposals

        score_output = self._score_candidate_set(
            scoring_proposals,
            ego_token,
            scorer_scene_features,
            scorer_patch_features,
        )
        output.update(score_output)

        pdm_score = score_output["pdm_score"]

        token = torch.argmax(pdm_score, dim=1)
        trajectory = scoring_proposals[torch.arange(batch_size), token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        # Expose MoE router outputs for load balance loss and monitoring
        output["moe_router_logits"] = router_logits
        output["moe_router_weights"] = router_weights
        output["moe_monitor"] = moe_monitor
        if self.training and self.traj_expansion_enabled:
            output["_ego_token"] = ego_token
            output["_scorer_scene_features"] = scorer_scene_features
            output["_scorer_patch_features"] = scorer_patch_features

        return output



