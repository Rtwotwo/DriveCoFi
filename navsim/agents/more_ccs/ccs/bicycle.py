"""Invertible kinematic control-space model used by CCS.

A trajectory ``tau = (x_t, y_t, theta_t)_{t=1..T}`` is expressed in the ego
frame at a fixed time step ``dt`` starting from the origin ``(0, 0, 0)`` with
initial longitudinal speed ``v0``.

The control sequence is ``u_t = (a_t, omega_t)`` where ``a_t`` is longitudinal
acceleration and ``omega_t`` is yaw-rate.  The forward integration is

    v_t     = v_{t-1} + a_t * dt          (v_0 = v0)
    theta_t = theta_{t-1} + omega_t * dt  (theta_0 = 0)
    x_t     = x_{t-1} + v_t * cos(theta_t) * dt
    y_t     = y_{t-1} + v_t * sin(theta_t) * dt

The inverse recovers ``u`` exactly from ``tau`` (see :meth:`BicycleModel.inverse`),
so ``forward(inverse(tau)) == tau`` up to numerical precision. Working in the
control space guarantees smooth, dynamically feasible samples during search.

All tensors are batched: leading ``...`` dims are preserved, the last two dims
are ``(T, 2)`` for controls and ``(T, 3)`` for poses.
"""

import torch


class BicycleModel:
    """Batched, differentiable, invertible control<->trajectory model."""

    def __init__(self, dt: float = 0.5):
        self.dt = float(dt)

    # ------------------------------------------------------------------ #
    def forward(self, u: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
        """Roll out controls into a pose trajectory.

        :param u: controls ``(..., T, 2)`` = (accel, yaw_rate).
        :param v0: initial longitudinal speed broadcastable to ``(...,)``.
        :return: poses ``(..., T, 3)`` = (x, y, heading) in the ego frame.
        """
        dt = self.dt
        a = u[..., 0]        # (..., T)
        omega = u[..., 1]    # (..., T)

        v = v0.unsqueeze(-1) + torch.cumsum(a, dim=-1) * dt          # (..., T)
        theta = torch.cumsum(omega, dim=-1) * dt                     # (..., T)

        dx = v * torch.cos(theta) * dt
        dy = v * torch.sin(theta) * dt

        x = torch.cumsum(dx, dim=-1)
        y = torch.cumsum(dy, dim=-1)

        return torch.stack([x, y, theta], dim=-1)                    # (..., T, 3)

    __call__ = forward

    # ------------------------------------------------------------------ #
    def inverse(self, tau: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
        """Recover the control sequence from a pose trajectory.

        :param tau: poses ``(..., T, 3)`` = (x, y, heading).
        :param v0: initial longitudinal speed broadcastable to ``(...,)``.
        :return: controls ``(..., T, 2)`` = (accel, yaw_rate).
        """
        dt = self.dt
        x = tau[..., 0]
        y = tau[..., 1]
        theta = tau[..., 2]

        x_prev = torch.cat([torch.zeros_like(x[..., :1]), x[..., :-1]], dim=-1)
        y_prev = torch.cat([torch.zeros_like(y[..., :1]), y[..., :-1]], dim=-1)
        theta_prev = torch.cat(
            [torch.zeros_like(theta[..., :1]), theta[..., :-1]], dim=-1
        )

        omega = (theta - theta_prev) / dt

        dx = x - x_prev
        dy = y - y_prev
        v = (dx * torch.cos(theta) + dy * torch.sin(theta)) / dt      # (..., T)

        v_prev = torch.cat([v0.unsqueeze(-1), v[..., :-1]], dim=-1)
        a = (v - v_prev) / dt

        return torch.stack([a, omega], dim=-1)                        # (..., T, 2)

    # ------------------------------------------------------------------ #
    def rollout_velocity(self, u: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
        """Return the per-step longitudinal speed ``v_t`` for a control seq."""
        return v0.unsqueeze(-1) + torch.cumsum(u[..., 0], dim=-1) * self.dt
