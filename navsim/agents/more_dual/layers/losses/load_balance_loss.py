"""
Load Balancing Loss for Scene-Adaptive MoE.

Prevents router collapse by penalizing concentrated expert usage.
For soft routing, the standard Switch Transformer formulation simplifies:
  L_balance = E * Σ_i (f_i * p_i)
where f_i = mean(router_weight[:, i]) and p_i = mean(router_weight[:, i]).

Also supports auxiliary entropy regularization:
  L_entropy = -Σ_i p_i * log(p_i)
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_load_balance_loss(
    router_logits: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Compute auxiliary load balancing loss (Switch Transformer style).

    For scene-adaptive soft routing:
      router_logits: [B, E] or stacked from multiple MoE layers
      Returns scalar loss penalizing non-uniform expert usage.

    Args:
        router_logits: raw router logits [B, num_experts] (or [L, B, E])
        num_experts: number of experts

    Returns:
        scalar load balance loss
    """
    if router_logits is None:
        return torch.tensor(0.0, device=router_logits.device if router_logits is not None else "cpu")

    # Handle single layer vs multi-layer
    if router_logits.dim() == 2:
        router_logits = router_logits.unsqueeze(0)  # [1, B, E]

    # Softmax probabilities
    router_probs = F.softmax(router_logits.float(), dim=-1)  # [L, B, E]

    # f_i: fraction of tokens (batch) routed to expert i
    # For soft routing, this is the mean weight per expert
    f_i = router_probs.mean(dim=1)  # [L, E]

    # p_i: mean router probability for expert i (same as f_i for soft routing)
    p_i = router_probs.mean(dim=1)  # [L, E]

    # L_balance = E * Σ_i f_i * p_i
    balance_loss = num_experts * (f_i * p_i).sum(dim=-1)  # [L]
    balance_loss = balance_loss.mean()  # scalar

    return balance_loss


def compute_router_entropy_loss(
    router_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute entropy regularization loss for router.

    Maximizes entropy of routing distribution to prevent collapse.
    L_entropy = -mean(Σ_i p_i * log(p_i))

    Args:
        router_logits: [B, num_experts] or [L, B, E]

    Returns:
        scalar entropy loss (negative → minimize to maximize entropy)
    """
    if router_logits is None:
        return torch.tensor(0.0)

    if router_logits.dim() == 2:
        router_logits = router_logits.unsqueeze(0)  # [1, B, E]

    router_probs = F.softmax(router_logits.float(), dim=-1)
    eps = 1e-8
    entropy = -(router_probs * torch.log(router_probs + eps)).sum(dim=-1)  # [L, B]
    # We want to MAXIMIZE entropy, so return negative as a loss
    return -entropy.mean()


def compute_moe_auxiliary_loss(
    router_logits: torch.Tensor,
    num_experts: int,
    load_balance_weight: float = 0.01,
    entropy_weight: float = 0.001,
) -> torch.Tensor:
    """Combined MoE auxiliary loss: load balance + entropy regularization.

    Args:
        router_logits: [B, E] raw router logits
        num_experts: number of experts
        load_balance_weight: weight for load balance loss (default 0.01)
        entropy_weight: weight for entropy regularization (default 0.001)

    Returns:
        scalar combined auxiliary loss
    """
    if router_logits is None:
        return torch.tensor(0.0)

    loss = torch.tensor(0.0, device=router_logits.device, dtype=torch.float32)

    if load_balance_weight > 0:
        loss = loss + load_balance_weight * compute_load_balance_loss(
            router_logits, num_experts
        )

    if entropy_weight > 0:
        loss = loss + entropy_weight * compute_router_entropy_loss(router_logits)

    return loss
