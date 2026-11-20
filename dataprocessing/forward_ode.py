"""Guidance-free noise codes: forward-time integration of a pre-trained flow matching model.

A clean image latent ``x_1 = E(x)`` is transported towards the Gaussian prior by integrating
the probability-flow ODE ``dx/dt = v_θ(x_t, t, y)`` backwards in time (t = 1 -> 0) with Euler steps
on the grid ``t_k = 1 - k / num_steps``. With guidance scale ω = 1 (the native conditional
velocity) the resulting codes ``x_t`` reconstruct their image without extra guidance.

Each requested ``t`` must lie on the grid; the state stored under ``t`` is exactly the state
*at* ``t`` (``x_1`` is the clean latent, ``x_0`` the fully encoded noise).
"""
from typing import Sequence

import torch
from torch import nn


class GuidanceFreeEncoder:
    def __init__(
        self,
        first_stage: nn.Module,
        flow: nn.Module,
        timesteps: Sequence[float],
        num_steps: int = 100,
        cfg_scale: float = 1.0,
        num_classes: int = 1000,
    ):
        self.first_stage = first_stage
        self.flow = flow
        self.num_steps = num_steps
        self.cfg_scale = cfg_scale
        self.null_label = num_classes                   # class-dropout token of the class embedding
        self.steps = {self._grid_index(t): float(t) for t in timesteps}

    def _grid_index(self, t: float) -> int:
        """Number of Euler steps from t=1 to reach ``t``; raises if ``t`` is not on the grid."""
        k = round((1.0 - t) * self.num_steps)
        if not 0 <= k <= self.num_steps or abs((1.0 - k / self.num_steps) - t) > 1e-6:
            raise ValueError(f"t={t} is not on the grid of {self.num_steps} Euler steps over [0, 1]")
        return k

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> dict[float, torch.Tensor]:
        """``images``: uint8 ``(B, 3, H, W)``; returns ``{t: x_t}`` with float32 ``(B, 4, h, w)`` latents."""
        x = images.float() / 127.5 - 1.0
        xt = self.first_stage.encode(x).float()
        states = {self.steps[0]: xt} if 0 in self.steps else {}
        uc_cond = torch.full_like(labels, self.null_label)

        for k in range(min(max(self.steps), self.num_steps)):
            t_curr, t_next = 1.0 - k / self.num_steps, 1.0 - (k + 1) / self.num_steps
            t = torch.full((xt.shape[0],), t_curr, dtype=xt.dtype, device=xt.device)
            velocity = self.flow(xt, t, y=labels, cfg_scale=self.cfg_scale, uc_cond=uc_cond)
            xt = xt + (t_next - t_curr) * velocity
            if k + 1 in self.steps:
                states[self.steps[k + 1]] = xt
        return states
