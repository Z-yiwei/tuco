"""Shared MLP-BC loader for the MuJoCo sim2sim co-training pipeline.

The pegHole BC checkpoint's output head is named `action_head` (train_mlp_bc.py),
NOT `mean_head` (train_mlp_gauss.py). The loader detects that distinction and
validates every model key before using a checkpoint.
"""
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        n_obs,
        hidden,
        head_key,
        architecture="legacy",
    ):
        super().__init__()
        self.n_obs = n_obs
        self.head_key = head_key
        self.architecture = architecture
        if architecture == "legacy":
            self.actor_hidden_dims = (1024, hidden, hidden, hidden)
            self.activation = "relu"
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim * n_obs, 1024), nn.ReLU(),
                nn.Linear(1024, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            setattr(self, head_key, nn.Linear(hidden, act_dim))
        elif architecture == "rsl_actor":
            self.actor_hidden_dims = (512, 256, 128, 64)
            self.activation = "elu"
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim * n_obs, 512), nn.ELU(),
                nn.Linear(512, 256), nn.ELU(),
                nn.Linear(256, 128), nn.ELU(),
                nn.Linear(128, 64), nn.ELU(),
            )
            setattr(self, head_key, nn.Linear(64, act_dim))
        else:
            raise ValueError(f"unknown MLP architecture: {architecture!r}")

    def forward(self, x):
        return getattr(self, self.head_key)(self.trunk(x.reshape(x.shape[0], -1)))


def load_mlp(path, device="cpu", use_ema=True):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["ema_state_dict"] if (use_ema and "ema_state_dict" in ck) else ck["model_state_dict"]
    action_head = "action_head.weight" in sd
    mean_head = "mean_head.weight" in sd
    if action_head == mean_head:
        raise RuntimeError(
            "MLP checkpoint must contain exactly one output head: "
            "action_head.weight or mean_head.weight"
        )
    head_key = "action_head" if action_head else "mean_head"

    architecture = ck.get("architecture", "legacy")
    m = MLP(
        ck["obs_dim"],
        ck["act_dim"],
        ck["n_obs_steps"],
        ck["hidden"],
        head_key,
        architecture=architecture,
    ).to(device)

    if "architecture" in ck:
        saved_dims = ck.get("actor_hidden_dims")
        saved_activation = ck.get("activation")
        if saved_dims is None or saved_activation is None:
            raise RuntimeError(
                "MLP checkpoint with architecture metadata must also contain "
                "actor_hidden_dims and activation"
            )
        if tuple(saved_dims) != m.actor_hidden_dims:
            raise RuntimeError(
                "MLP architecture metadata mismatch: "
                f"actor_hidden_dims={saved_dims}, expected={list(m.actor_hidden_dims)}"
            )
        if saved_activation != m.activation:
            raise RuntimeError(
                "MLP architecture metadata mismatch: "
                f"activation={saved_activation!r}, expected={m.activation!r}"
            )

    # Gaussian checkpoints have a separately consumed log_std parameter. Remove
    # only that explicitly supported non-policy key, then validate the policy
    # state dict exactly; never let strict=False hide a missing/random layer.
    model_sd = {
        key: value
        for key, value in sd.items()
        if key != "log_std" and not key.startswith("log_std.")
    }
    expected_sd = m.state_dict()
    missing = sorted(set(expected_sd) - set(model_sd))
    unexpected = sorted(set(model_sd) - set(expected_sd))
    if missing or unexpected:
        raise RuntimeError(
            f"MLP state-dict key mismatch: missing={missing}, unexpected={unexpected}"
        )
    shape_mismatches = [
        f"{key}: checkpoint={tuple(model_sd[key].shape)}, expected={tuple(value.shape)}"
        for key, value in expected_sd.items()
        if model_sd[key].shape != value.shape
    ]
    if shape_mismatches:
        raise RuntimeError(
            "MLP state-dict shape mismatch: " + "; ".join(shape_mismatches)
        )
    m.load_state_dict(model_sd, strict=True)
    m.eval()
    ns = ck["norm_stats"]
    norm = dict(
        s_mean=ns["s_mean"].cpu().numpy(), s_std=ns["s_std"].cpu().numpy(),
        a_center=ns["a_center"].cpu().numpy(), a_scale=ns["a_scale"].cpu().numpy(),
        n_obs=ck["n_obs_steps"], head_key=head_key,
        architecture=architecture,
        actor_hidden_dims=list(m.actor_hidden_dims),
        activation=m.activation,
    )
    return m, norm
