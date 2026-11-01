"""
Scene-Adaptive Soft MoE for DrivoR Scene Tokens.

Core design (different from Mixtral's token-level top-K routing):
  - Scene-level soft routing: pool all scene tokens → router → softmax weights
  - All experts always participate (weighted mixture), no hard top-K selection
  - Each expert processes ALL scene tokens, weighted by router probabilities
  - Collects router_logits / router_weights for load balance loss and monitoring

Shape convention:
  Input:  [B, N_total, D]   where N_total = num_cameras * num_scene_tokens_per_cam
  Output: [B, N_total, D]   same shape, transformed by expert mixture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SwiGLUExpert(nn.Module):
    """Single SwiGLU FFN expert that processes all scene tokens.

    Architecture: gate(x) * up(x) → down, matching DrivoR's ffn hidden dim.
    """

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.up_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        return self.down_proj(hidden)


class SceneSoftRouter(nn.Module):
    """Scene-level soft router: pool scene tokens → linear → softmax.

    Input:  [B, N_total, D]  (all scene tokens across cameras)
    Output: router_logits [B, num_experts], router_weights [B, num_experts]
    """

    def __init__(self, d_model: int, num_experts: int, jitter_noise: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.jitter_noise = jitter_noise
        self.weight = nn.Parameter(torch.empty(num_experts, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, scene_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Scene-level pooling: mean over N_total dimension
        scene_vector = scene_tokens.mean(dim=1)  # [B, D]

        # Router logits
        router_logits = F.linear(scene_vector, self.weight)  # [B, num_experts]

        # Add jitter noise during training to encourage exploration
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise

        router_weights = F.softmax(router_logits, dim=-1)  # [B, num_experts]

        return router_logits, router_weights


class SceneAdaptiveMoE(nn.Module):
    """Scene-Adaptive Mixture of Experts for DrivoR scene tokens.

    Design rationale:
      - Scene-level soft routing: avoids token-level top-K artifacts
      - All experts process all tokens, weighted by router → continuous scene adaptation
      - SwiGLU experts match DrivoR's FFN dimension convention
      - Returns router_logits / router_weights for load balance loss and monitoring

    Args:
        d_model: scene token feature dimension (256)
        d_ffn: expert hidden dimension (default 1024, matching tf_d_ffn)
        num_experts: number of scene experts (default 6)
        dropout: dropout in expert FFNs
        jitter_noise: std of Gaussian noise added to router logits during training
    """

    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        num_experts: int = 6,
        dropout: float = 0.1,
        jitter_noise: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model

        self.router = SceneSoftRouter(
            d_model=d_model,
            num_experts=num_experts,
            jitter_noise=jitter_noise,
        )

        self.experts = nn.ModuleList([
            SwiGLUExpert(d_model=d_model, d_ffn=d_ffn, dropout=dropout)
            for _ in range(num_experts)
        ])

        # Residual scaling: output is residual added to input after MoE
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(
        self, scene_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            scene_tokens: [B, N_total, D]  scene tokens from all cameras

        Returns:
            output:          [B, N_total, D]  transformed scene tokens
            router_logits:   [B, num_experts]  raw router logits (for loss)
            router_weights:  [B, num_experts]  softmax routing weights (for monitoring)
        """
        # 1. Router: scene-level pooling → softmax weights
        router_logits, router_weights = self.router(scene_tokens)
        # router_weights: [B, E]

        # 2. Each expert processes ALL scene tokens
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(scene_tokens)  # [B, N_total, D]
            expert_outputs.append(expert_out)
        expert_stack = torch.stack(expert_outputs, dim=0)  # [E, B, N_total, D]

        # 3. Weighted mixture: Σ_e weight_e * expert_e_output
        # router_weights: [B, E] → [E, B, 1, 1]
        weights = router_weights.permute(1, 0)[:, :, None, None]  # [E, B, 1, 1]
        mixed = (expert_stack * weights).sum(dim=0)  # [B, N_total, D]

        # 4. Residual connection with learnable scale
        output = scene_tokens + self.residual_scale * mixed

        return output, router_logits, router_weights

    def get_monitor_data(self, router_weights: torch.Tensor) -> dict:
        """Compute monitoring metrics from router_weights [B, num_experts].

        Returns dict with:
            - router_entropy: mean entropy of routing distribution
            - expert_usage_i: mean weight assigned to expert i
            - router_prob_mean_i: same as expert_usage for soft routing
        """
        B, E = router_weights.shape
        eps = 1e-8

        # Router entropy per sample
        entropy = -(router_weights * torch.log(router_weights + eps)).sum(dim=-1)  # [B]
        router_entropy = entropy.mean().item()

        # Expert usage: mean weight per expert
        expert_usage = router_weights.mean(dim=0)  # [E]

        monitor = {"router_entropy": router_entropy}
        for i in range(E):
            monitor[f"expert_usage_{i}"] = expert_usage[i].item()
            monitor[f"router_prob_mean_{i}"] = expert_usage[i].item()

        return monitor
