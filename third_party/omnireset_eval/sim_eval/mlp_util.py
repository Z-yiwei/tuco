"""Shared MLP-BC loader for the MuJoCo sim2sim co-training pipeline.

The pegHole BC checkpoint's output head is named `action_head` (train_mlp_bc.py),
NOT `mean_head` (train_mlp_gauss.py). Loading with the wrong head name + strict=False
silently leaves a RANDOM output layer -> deformed actions -> 0% SR. This loader detects
the head key from the state_dict and loads it correctly.
"""
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, obs_dim, act_dim, n_obs, hidden, head_key):
        super().__init__()
        self.n_obs = n_obs
        self.head_key = head_key
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim * n_obs, 1024), nn.ReLU(),
            nn.Linear(1024, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        setattr(self, head_key, nn.Linear(hidden, act_dim))

    def forward(self, x):
        return getattr(self, self.head_key)(self.trunk(x.reshape(x.shape[0], -1)))


def load_mlp(path, device="cpu", use_ema=True):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["ema_state_dict"] if (use_ema and "ema_state_dict" in ck) else ck["model_state_dict"]
    head_key = "action_head" if "action_head.weight" in sd else "mean_head"
    m = MLP(ck["obs_dim"], ck["act_dim"], ck["n_obs_steps"], ck["hidden"], head_key).to(device)
    miss, unexp = m.load_state_dict(sd, strict=False)
    miss = [k for k in miss if not k.startswith("log_std")]  # gauss ckpts have an extra std head
    assert not miss, f"MLP load: missing critical keys {miss}"
    assert f"{head_key}.weight" in sd, f"head {head_key} not in ckpt!"
    m.eval()
    ns = ck["norm_stats"]
    norm = dict(
        s_mean=ns["s_mean"].cpu().numpy(), s_std=ns["s_std"].cpu().numpy(),
        a_center=ns["a_center"].cpu().numpy(), a_scale=ns["a_scale"].cpu().numpy(),
        n_obs=ck["n_obs_steps"], head_key=head_key,
    )
    return m, norm
