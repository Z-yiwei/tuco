"""Stage 1: standalone torch loader for the Franka OmniReset RL expert.

Reconstructs the rsl_rl actor *exactly* — but in plain torch with NO IsaacLab /
rsl_rl dependency — so the policy can run inside the MuJoCo env (rv_dp_mujoco).

Faithful to rsl_rl `ActorCritic.act_inference` (modules/actor_critic.py:294-300):

    obs  -> actor_obs_normalizer(obs) = (obs - mean) / (std + eps)   # eps = 1e-2
         -> actor MLP [200->512->256->128->64->7], ELU between layers, none after
         -> 7-D mean   (6 arm OSC delta + 1 gripper; gripper component is IGNORED
                        downstream — the grasp-guard rule controls the fingers)

gSDE only affects the std (sampling); the deterministic mean we step with is just
the MLP output, so gSDE is irrelevant here.

The empirical obs normalizer stats (_mean/_std) live INSIDE the checkpoint, so
forgetting them = garbage in. They are loaded verbatim.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# rsl_rl EmpiricalNormalization default (networks/normalization.py:17)
NORM_EPS = 1e-2
ACT_HIDDEN_DIMS = [512, 256, 128, 64]
OBS_DIM = 200
ACT_DIM = 7


def _build_mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    """rsl_rl MLP: Linear, ELU, Linear, ELU, ..., Linear (no final activation).

    Layer indices match the checkpoint keys: actor.0/2/4/6/8 are the Linears,
    ELU sits at the odd indices.
    """
    layers: list[nn.Module] = []
    dims = [in_dim, *hidden]
    for i in range(len(hidden)):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ELU())
    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)


class FrankaPolicy(nn.Module):
    """Deterministic-mean actor for the Franka OmniReset peg task."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: list[int] | None = None, eps: float = NORM_EPS):
        super().__init__()
        hidden = hidden or ACT_HIDDEN_DIMS
        self.eps = eps
        self.actor = _build_mlp(obs_dim, hidden, act_dim)
        # normalizer stats (filled by load_from_checkpoint); shape (1, obs_dim)
        self.register_buffer("obs_mean", torch.zeros(1, obs_dim))
        self.register_buffer("obs_std", torch.ones(1, obs_dim))

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (..., 200) -> mean action (..., 7)."""
        x = (obs - self.obs_mean) / (self.obs_std + self.eps)
        return self.actor(x)

    @classmethod
    def load_from_checkpoint(cls, ckpt_path: str, device: str = "cpu") -> "FrankaPolicy":
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        # infer obs_dim from the first actor layer
        obs_dim = sd["actor.0.weight"].shape[1]
        act_dim = sd["actor.8.weight"].shape[0] if "actor.8.weight" in sd else ACT_DIM
        model = cls(obs_dim=obs_dim, act_dim=act_dim)
        actor_sd = {k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}
        missing, unexpected = model.actor.load_state_dict(actor_sd, strict=True)
        assert not missing and not unexpected, (missing, unexpected)
        model.obs_mean.copy_(sd["actor_obs_normalizer._mean"])
        model.obs_std.copy_(sd["actor_obs_normalizer._std"])
        model.to(device).eval()
        model.ckpt_iter = int(ck.get("iter", -1))
        return model


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    a = ap.parse_args()
    p = FrankaPolicy.load_from_checkpoint(a.checkpoint)
    print(f"[FrankaPolicy] loaded iter={p.ckpt_iter} obs_dim={p.obs_mean.shape[1]} "
          f"act_dim={p.actor[-1].out_features}")
    o = torch.zeros(3, p.obs_mean.shape[1])
    print("zero-obs action:", p(o)[0].numpy())
