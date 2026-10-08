"""Expose the ``torch.func`` API on the paper's PyTorch 1.12 runtime."""

from __future__ import annotations

from types import SimpleNamespace

import torch


def ensure_torch_func() -> None:
    """Install the functorch 0.2 compatibility surface when torch.func is absent."""
    if hasattr(torch, "func"):
        return
    from functorch import grad, vmap
    from torch.nn.utils.stateless import functional_call as stateless_call

    def functional_call(module, parameter_and_buffer_dicts, args, kwargs=None):
        if isinstance(parameter_and_buffer_dicts, tuple):
            parameters, buffers = parameter_and_buffer_dicts
            state = dict(parameters)
            state.update(buffers)
        else:
            state = parameter_and_buffer_dicts
        return stateless_call(module, state, args, kwargs)

    torch.func = SimpleNamespace(  # type: ignore[attr-defined]
        grad=grad,
        vmap=vmap,
        functional_call=functional_call,
    )

