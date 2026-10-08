from __future__ import annotations

import numpy as np
import torch


class ConditionalFlowMatcher:
    """Minimal conditional flow matcher for action-sequence policies.

    This implements the sigma=0 path used by MARS*: sample a time t, linearly
    interpolate from the flow source x0 to target x1, and train the model to
    predict the constant velocity x1 - x0.
    """

    def __init__(self, sigma: float = 0.0, num_sampling_steps: int = 1):
        if float(sigma) != 0.0:
            raise NotImplementedError("Only sigma=0.0 is supported in this CUPID port.")
        self.sigma = float(sigma)
        self.num_sampling_steps = int(num_sampling_steps)

    def compute_loss(self, model, target: torch.Tensor, start: torch.Tensor | None = None, **kwargs):
        if start is None:
            start = torch.randn_like(target)
        t = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        t_view = t.view(-1, *([1] * (target.ndim - 1)))
        xt = (1.0 - t_view) * start + t_view * target
        ut = target - start
        vt = model(xt, t, **kwargs)
        loss = torch.mean((vt - ut) ** 2)
        return loss, {"loss": loss.item()}

    def sample(
        self,
        model,
        shape,
        device,
        num_steps: int | None = None,
        return_traces: bool = False,
        start: torch.Tensor | None = None,
        **kwargs,
    ):
        if num_steps is None:
            num_steps = self.num_sampling_steps
        num_steps = int(num_steps)
        if start is None:
            x = torch.randn(shape, device=device)
        else:
            x = start
        dt = 1.0 / max(num_steps, 1)

        if return_traces:
            traj_history = [x.detach().clone()]
            vel_history = [np.zeros_like(x.detach().cpu().numpy())]

        for i in range(num_steps):
            t = torch.full((x.shape[0],), i / max(num_steps, 1), device=x.device, dtype=x.dtype)
            vt = model(x, t, **kwargs)
            x = x + vt * dt
            if return_traces:
                traj_history.append(x.detach().clone().cpu())
                vel_history.append(vt.detach().clone().cpu())

        if return_traces:
            return x, (traj_history, vel_history)
        return x

