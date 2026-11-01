from typing import Dict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import open_dict
from .score_module.scorer import Scorer
from .transformer_decoder import TransformerDecoder, TransformerDecoderScorer
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

        # scoring
        B,N,_,_=proposals.shape

        embedded_traj = self.pos_embed(proposals.reshape(B, N, -1).detach())  # (B, N, d_model)
        tr_out, scorer_patch_gates = self.scorer_attention(
            embedded_traj,
            scorer_scene_features,
            scorer_patch_features,
        )  # (B, N, d_model), (num_layers, B, N, 1)
        tr_out = tr_out+ego_token
        pred_logit,pred_logit2, pred_agents_states, pred_area_logit ,bev_semantic_map,agent_states,agent_labels= self.scorer(proposals, tr_out)

        output["pred_logit"]=pred_logit
        output["pred_logit2"]=pred_logit2
        output["pred_agents_states"]=pred_agents_states
        output["pred_area_logit"]=pred_area_logit
        output["bev_semantic_map"]=bev_semantic_map
        output["agent_states"]=agent_states
        output["agent_labels"]=agent_labels
        output["scorer_patch_gates"] = scorer_patch_gates

        pdm_score = (
        self._config.noc * pred_logit['no_at_fault_collisions'].sigmoid().log() +
        self._config.dac * pred_logit['drivable_area_compliance'].sigmoid().log() +
        self._config.ddc * pred_logit['driving_direction_compliance'].sigmoid().log() +    
        (self._config.ttc * pred_logit['time_to_collision_within_bound'].sigmoid() +
        self._config.ep * pred_logit['ego_progress'].sigmoid()  
        + self._config.comfort * pred_logit['comfort'].sigmoid()).log()
        )

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        # Expose MoE router outputs for load balance loss and monitoring
        output["moe_router_logits"] = router_logits
        output["moe_router_weights"] = router_weights
        output["moe_monitor"] = moe_monitor

        return output



