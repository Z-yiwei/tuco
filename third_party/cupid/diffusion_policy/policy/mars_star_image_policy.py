from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.flow.flow_transformer import FlowTransformer
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class MarsStarImagePolicy(BaseImagePolicy):
    """MARS* image policy adapted for CUPID's 2-observation / 8-action DP setup."""

    requires_past_action = True

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        flow_matcher,
        source_action_steps=None,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        diffusion_step_embed_dim=256,
        latent_dim=512,
        flow_loss_weight=1.0,
        consistency_weight=1.0,
        router_entropy_weight=0.0,
        diversity_weight=1.0,
        allow_diversity_ablation=False,
        diversity_k=5,
        adaptive_steps=True,
        adaptive_max_steps=10,
        adaptive_cond=True,
        history_min=1,
        adaptive_exec=True,
        exec_min=1,
        router_input="image",
        history_source="past_action",
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            raise TypeError(f"Unexpected MarsStarImagePolicy kwargs: {sorted(kwargs.keys())}")
        if history_source != "past_action":
            raise ValueError("This CUPID MARS* port currently requires history_source='past_action'.")
        if router_input not in ("full", "image"):
            raise ValueError(f"router_input must be 'full' or 'image', got {router_input!r}.")
        if float(diversity_weight) <= 0.0 and not bool(allow_diversity_ablation):
            raise ValueError(
                "MARS* requires diversity_weight > 0 with KNN target_spread. "
                "Set allow_diversity_ablation=True only for an explicit ablation run."
            )

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        source_action_steps = int(source_action_steps or n_action_steps)
        if source_action_steps != int(n_action_steps):
            raise ValueError(
                "source_action_steps must equal n_action_steps for the current flow source/target shape; "
                f"got {source_action_steps} vs {n_action_steps}."
            )

        obs_feature_dim = obs_encoder.output_shape()[0]
        global_cond_dim = obs_feature_dim * int(n_obs_steps)

        self.obs_encoder = obs_encoder
        self.flow_net = FlowTransformer(
            input_dim=action_dim,
            condition_dim=global_cond_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            time_embed_dim=diffusion_step_embed_dim,
        )
        self.history_encoder = nn.Sequential(
            nn.Linear(action_dim * source_action_steps, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        router_hidden = 256
        self.router_input = router_input
        if router_input == "image":
            low_total = sum(int(np.prod(obs_encoder.key_shape_map[k])) for k in obs_encoder.low_dim_keys)
            self._rgb_dim = int(obs_feature_dim - low_total)
            self.router_obs_proj = nn.Sequential(
                nn.LayerNorm(self._rgb_dim),
                nn.Linear(self._rgb_dim, router_hidden),
                nn.GELU(),
            )
            self.router_hist_proj = nn.Identity()
            self.router_fusion = nn.Sequential(
                nn.Linear(router_hidden, router_hidden),
                nn.GELU(),
                nn.Linear(router_hidden, router_hidden),
                nn.GELU(),
                nn.Linear(router_hidden, action_dim),
            )
        else:
            self._rgb_dim = None
            self.router_obs_proj = nn.Sequential(
                nn.LayerNorm(global_cond_dim),
                nn.Linear(global_cond_dim, router_hidden),
                nn.GELU(),
            )
            self.router_hist_proj = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, router_hidden),
                nn.GELU(),
            )
            self.router_fusion = nn.Sequential(
                nn.Linear(router_hidden * 2, router_hidden),
                nn.GELU(),
                nn.Linear(router_hidden, router_hidden),
                nn.GELU(),
                nn.Linear(router_hidden, action_dim),
            )

        self.flow_matcher = flow_matcher
        self.num_sampling_steps = flow_matcher.num_sampling_steps
        self.flow_loss_weight = float(flow_loss_weight)
        self.consistency_weight = float(consistency_weight)
        self.router_entropy_weight = float(router_entropy_weight)
        self.diversity_weight = float(diversity_weight)
        self.diversity_k = int(diversity_k)
        self.adaptive_steps = bool(adaptive_steps)
        self.adaptive_max_steps = int(adaptive_max_steps)
        self.adaptive_cond = bool(adaptive_cond)
        self.history_min = int(history_min)
        self.adaptive_exec = bool(adaptive_exec)
        self.exec_min = int(exec_min)
        self.history_source = history_source

        self.normalizer = LinearNormalizer()
        self.horizon = int(horizon)
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.source_action_steps = source_action_steps
        self.latent_dim = latent_dim
        self._last_metrics = {}
        self._last_L_eff = None
        self.reset()

        print("MARS* flow params: %e" % sum(p.numel() for p in self.flow_net.parameters()))
        print("MARS* vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _encode_condition(self, nobs: Dict[str, torch.Tensor], batch_size: int):
        this_nobs = dict_apply(
            nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(batch_size, -1)

    def _get_history_actions(self, raw_obs=None, batch=None):
        if batch is not None:
            if "past_action" not in batch:
                raise KeyError("MARS* training requires batch['past_action'] from the dataset wrapper.")
            raw = batch["past_action"]
        else:
            if raw_obs is None or "past_action" not in raw_obs:
                raise KeyError("MARS* inference requires obs_dict['past_action'] from the eval runner.")
            raw = raw_obs["past_action"]
        raw = raw[:, : self.source_action_steps, :]
        if raw.shape[1] != self.source_action_steps:
            raise ValueError(f"past_action length {raw.shape[1]} != {self.source_action_steps}")
        return self.normalizer["action"].normalize(raw)

    def _encode_history_latent(self, history_actions):
        batch_size = history_actions.shape[0]
        return self.history_encoder(history_actions.reshape(batch_size, -1))

    def _predict_router_weight(self, obs_latents, history_latents):
        if self.router_input == "image":
            batch_size = obs_latents.shape[0]
            per_frame_dim = obs_latents.shape[-1] // self.n_obs_steps
            cur_img = obs_latents.reshape(batch_size, self.n_obs_steps, per_frame_dim)[:, -1, : self._rgb_dim]
            return torch.sigmoid(self.router_fusion(self.router_obs_proj(cur_img)))

        h_obs = self.router_obs_proj(obs_latents)
        h_hist = self.router_hist_proj(history_latents)
        fused = torch.cat([h_obs * h_hist, h_obs + h_hist], dim=-1)
        return torch.sigmoid(self.router_fusion(fused))

    def _adaptive_cond(self, obs_cond, router_weight):
        batch_size = obs_cond.shape[0]
        if not self.adaptive_cond:
            self._last_L_eff = torch.full((batch_size,), float(self.n_obs_steps), device=obs_cond.device)
            return obs_cond
        if self.n_obs_steps <= 1:
            self._last_L_eff = torch.ones(batch_size, device=obs_cond.device)
            return obs_cond

        per_frame_dim = obs_cond.shape[-1] // self.n_obs_steps
        feats = obs_cond.reshape(batch_size, self.n_obs_steps, per_frame_dim)
        w = router_weight.amax(dim=-1).detach().clamp(0.0, 1.0)
        history_min = min(max(self.history_min, 1), self.n_obs_steps)
        l_eff = history_min + (self.n_obs_steps - history_min) * (1.0 - w)
        age = torch.arange(
            self.n_obs_steps - 1,
            -1,
            -1,
            device=obs_cond.device,
            dtype=feats.dtype,
        )
        gate = (l_eff.unsqueeze(1).to(feats.dtype) - age.unsqueeze(0)).clamp(0.0, 1.0)
        self._last_L_eff = l_eff
        return (feats * gate.unsqueeze(-1)).reshape(batch_size, self.n_obs_steps * per_frame_dim)

    def _mix_start(self, history_actions, weight):
        w = weight.unsqueeze(1)
        noise = torch.randn_like(history_actions)
        return (1.0 - w) * history_actions + w * noise

    def _weight_to_steps(self, weight):
        w_scalar = weight.amax(dim=-1) if weight.dim() > 1 else weight
        steps = torch.ceil(w_scalar.clamp(0.0, 1.0) * self.adaptive_max_steps)
        return steps.clamp(min=1, max=self.adaptive_max_steps).long()

    def _sample_adaptive_steps(self, start, obs_cond, weight):
        if not self.adaptive_steps:
            return self.flow_matcher.sample(
                model=self.flow_net,
                shape=start.shape,
                device=start.device,
                start=start,
                global_cond=obs_cond,
            )

        steps_per_sample = self._weight_to_steps(weight)
        output = torch.empty_like(start)
        for step_count in steps_per_sample.unique().tolist():
            mask = steps_per_sample == step_count
            sub_start = start[mask]
            sub_cond = obs_cond[mask]
            output[mask] = self.flow_matcher.sample(
                model=self.flow_net,
                shape=sub_start.shape,
                device=sub_start.device,
                num_steps=int(step_count),
                start=sub_start,
                global_cond=sub_cond,
            )
        return output

    def _diversity_term(self, start, future_actions, history_actions, router_weight, batch):
        if "target_spread" not in batch or "nn_histories" not in batch:
            raise KeyError(
                "MARS* diversity_weight > 0 requires KNN fields: 'target_spread' and 'nn_histories'."
            )
        target_spread = batch["target_spread"].detach()
        nn_hist = batch["nn_histories"].detach()
        batch_size = start.shape[0]
        dim = nn_hist.shape[-1]
        nn_noise = torch.randn_like(nn_hist)
        nn_start = (1.0 - router_weight.view(batch_size, 1, 1, dim)) * nn_hist
        nn_start = nn_start + router_weight.view(batch_size, 1, 1, dim) * nn_noise
        per_pair = (start.unsqueeze(1) - nn_start).abs().mean(dim=2)
        nn_weights = batch.get("nn_weights")
        if nn_weights is None:
            start_spread = per_pair.mean(dim=1)
        else:
            start_spread = (nn_weights.detach().unsqueeze(-1) * per_pair).sum(dim=1)
        return F.relu(target_spread - start_spread).mean()

    def compute_loss(self, batch, return_batch_loss: bool = False):
        if return_batch_loss:
            raise NotImplementedError("MARS* currently returns only the reduced batch loss.")
        assert "valid_mask" not in batch

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        obs_cond = self._encode_condition(nobs, batch_size)
        history_actions = self._get_history_actions(batch=batch)
        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:, future_start:future_end, :]
        if future_actions.shape[1] != self.n_action_steps:
            raise ValueError(
                f"future action length {future_actions.shape[1]} != n_action_steps {self.n_action_steps}"
            )

        history_latent = self._encode_history_latent(history_actions)
        router_weight = self._predict_router_weight(obs_cond, history_latent)
        obs_cond_adapt = self._adaptive_cond(obs_cond, router_weight)
        start = self._mix_start(history_actions, router_weight)

        flow_loss, metrics = self.flow_matcher.compute_loss(
            self.flow_net,
            target=future_actions,
            start=start,
            global_cond=obs_cond_adapt,
        )
        loss = self.flow_loss_weight * flow_loss
        metrics["flow_loss"] = float(flow_loss.item())

        t_zero = torch.zeros(batch_size, device=start.device, dtype=start.dtype)
        pred_actions = start + self.flow_net(start, t_zero, global_cond=obs_cond_adapt)
        if self.consistency_weight > 0:
            per_dim_loss = F.l1_loss(pred_actions, future_actions, reduction="none").mean(dim=-2)
            consistency_weight = (1.0 - router_weight).detach()
            consistency_loss = (consistency_weight * per_dim_loss).mean()
            loss = loss + self.consistency_weight * consistency_loss
            metrics["consistency_loss"] = float(consistency_loss.item())
            metrics["consistency_weight_mean"] = float(consistency_weight.mean().item())

        if self.diversity_weight > 0:
            diversity_loss = self._diversity_term(
                start, future_actions, history_actions, router_weight, batch
            )
            loss = loss + self.diversity_weight * diversity_loss
            metrics["diversity_loss"] = float(diversity_loss.item())

        if self.router_entropy_weight > 0:
            w = router_weight
            entropy = -(w * torch.log(w + 1e-6) + (1.0 - w) * torch.log(1.0 - w + 1e-6)).mean()
            loss = loss - self.router_entropy_weight * entropy
            metrics["router_entropy"] = float(entropy.item())

        metrics["router_weight_mean"] = float(router_weight.mean().item())
        metrics["router_weight_std"] = float(router_weight.std().item())
        metrics["router_weight_max_dim"] = float(router_weight.mean(dim=0).max().item())
        metrics["router_weight_min_dim"] = float(router_weight.mean(dim=0).min().item())
        if self._last_L_eff is not None:
            metrics["hist_len_mean"] = float(self._last_L_eff.mean().item())
        self._last_metrics = metrics
        return loss

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        to_normalize = {k: v for k, v in obs_dict.items() if k != "past_action"}
        nobs = self.normalizer.normalize(to_normalize)
        batch_size = next(iter(nobs.values())).shape[0]

        obs_cond = self._encode_condition(nobs, batch_size)
        history_actions = self._get_history_actions(raw_obs=obs_dict)
        with torch.no_grad():
            history_latent = self._encode_history_latent(history_actions)
            router_weight = self._predict_router_weight(obs_cond, history_latent)

        obs_cond_adapt = self._adaptive_cond(obs_cond, router_weight)
        start = self._mix_start(history_actions, router_weight)
        action_pred = self._sample_adaptive_steps(start, obs_cond_adapt, router_weight)
        action_pred = self.normalizer["action"].unnormalize(action_pred)

        if self.adaptive_exec and self.n_action_steps > self.exec_min:
            w_bar = float(router_weight.amax().item())
            exec_len = round(self.exec_min + (self.n_action_steps - self.exec_min) * (1.0 - w_bar))
            exec_len = int(max(self.exec_min, min(exec_len, self.n_action_steps)))
        else:
            exec_len = self.n_action_steps
        action = action_pred[:, :exec_len]

        self._weight_history.append(float(router_weight[0].max().item()))
        if self.adaptive_steps:
            steps_used = int(self._weight_to_steps(router_weight)[0].item())
        else:
            steps_used = int(self.num_sampling_steps)
        self._steps_history.append(steps_used)
        if self._last_L_eff is not None:
            self._hist_len_history.append(float(self._last_L_eff[0].item()))
        self._exec_len_history.append(exec_len)

        return {
            "action": action,
            "action_pred": action_pred,
            "router_weight": router_weight,
            "score": router_weight.amax(dim=-1),
            "history_len": self._last_L_eff,
        }

    def reset(self):
        self._weight_history = []
        self._steps_history = []
        self._hist_len_history = []
        self._exec_len_history = []

    def get_weight_history(self):
        return list(self._weight_history)

    def get_score_history(self):
        return list(self._weight_history)

    def get_steps_history(self):
        return list(self._steps_history)

    def get_history_len_history(self):
        return list(self._hist_len_history)

    def get_exec_len_history(self):
        return list(self._exec_len_history)
