from types import SimpleNamespace

import torch


if not hasattr(torch, "func"):
    try:
        from functorch import grad, vmap
        from torch.nn.utils.stateless import functional_call as _stateless_call
    except ImportError as error:
        raise ImportError(
            "TRAK on PyTorch 1.12 requires functorch==0.2.1; install the "
            "release's environments/cupid-runtime.txt"
        ) from error

    def _functional_call(module, parameter_and_buffer_dicts, args, kwargs=None):
        if isinstance(parameter_and_buffer_dicts, tuple):
            parameters, buffers = parameter_and_buffer_dicts
            state = dict(parameters)
            state.update(buffers)
        else:
            state = parameter_and_buffer_dicts
        return _stateless_call(module, state, args, kwargs)

    torch.func = SimpleNamespace(  # type: ignore[attr-defined]
        grad=grad, vmap=vmap, functional_call=_functional_call
    )

from .traker import TRAKer
from .utils import test_install

__version__ = "0.3.2"
VERSION = __version__
