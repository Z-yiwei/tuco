"""Train a deterministic MLP behavior-cloning policy on Franka state demos.

This is the plain MLP BC baseline: same state/action normalization and MLP trunk
as train_mlp_gauss.py, but no Gaussian head and no teacher std. The model
directly regresses the recorded teacher mean action.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import zarr


class MLPBCPolicy(nn.Module):
    """state history -> deterministic action."""

    def __init__(self, obs_dim, act_dim=7, n_obs_steps=2, hidden=512):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_obs_steps = n_obs_steps
        in_dim = obs_dim * n_obs_steps
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(),
            nn.Linear(1024, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden, act_dim)

    def forward(self, obs_stack):
        h = self.trunk(obs_stack.reshape(obs_stack.shape[0], -1))
        return self.action_head(h)


class MLPBCDataset(torch.utils.data.Dataset):
    def __init__(self, zarr_path, n_obs_steps=2, num_demos=None):
        z = zarr.open(zarr_path, mode="r")
        state = torch.from_numpy(z["data/state"][:]).float()
        prev_state = (
            torch.from_numpy(z["data/prev_state"][:]).float()
            if "data/prev_state" in z
            else None
        )
        action = torch.from_numpy(z["data/action"][:]).float()
        ends = z["meta/episode_ends"][:].astype(np.int64)
        if num_demos is not None and num_demos < len(ends):
            cut = int(ends[num_demos - 1])
            state, action, ends = state[:cut], action[:cut], ends[:num_demos]
            if prev_state is not None:
                prev_state = prev_state[:cut]
            print(f"[INFO] limiting dataset to first {num_demos} demos ({cut} steps)")

        self.starts = np.concatenate([[0], ends[:-1]]).astype(np.int64)
        self.ends = ends
        self.n_obs = n_obs_steps
        self.s_mean = state.mean(dim=0)
        self.s_std = state.std(dim=0) + 1e-6
        a_max, a_min = action.max(dim=0).values, action.min(dim=0).values
        self.a_center = (a_max + a_min) / 2.0
        self.a_scale = (a_max - a_min) / 2.0 + 1e-6

        self.state_n = (state - self.s_mean) / self.s_std
        self.prev_state_n = (
            (prev_state - self.s_mean) / self.s_std if prev_state is not None else None
        )
        self.action_n = (action - self.a_center) / self.a_scale

        self.samples = []
        for ep in range(len(self.ends)):
            for t in range(int(self.ends[ep] - self.starts[ep])):
                self.samples.append((int(self.starts[ep]), int(self.ends[ep]), t))
        print(f"[INFO] dataset: {len(self.ends)} episodes, {len(self.samples)} samples")
        print(f"[INFO] obs_dim={state.shape[1]}, act_dim={action.shape[1]}, n_obs_steps={n_obs_steps}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_start, ep_end, t = self.samples[idx]
        if self.prev_state_n is not None and self.n_obs == 2:
            g = ep_start + t
            previous = self.prev_state_n[g] if t == 0 else self.state_n[g - 1]
            obs = torch.stack([previous, self.state_n[g]])
        else:
            obs_idxs = [min(max(ep_start, ep_start + t + i), ep_end - 1)
                        for i in range(-self.n_obs + 1, 1)]
            obs = self.state_n[obs_idxs]
        g = ep_start + t
        return obs, self.action_n[g]


class EMA:
    def __init__(self, model, inv_gamma=1.0, power=0.75, min_value=0.0,
                 max_value=0.9999, update_after_step=0):
        self.inv_gamma, self.power = inv_gamma, power
        self.min_value, self.max_value = min_value, max_value
        self.update_after_step = update_after_step
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def _decay(self, step):
        step = max(0, step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        return float(min(self.max_value, max(self.min_value, value)))

    @torch.no_grad()
    def update(self, model, step):
        d = self._decay(step)
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--output_dir", default="./checkpoints/mlp_bc_franka_task0_stage2")
    ap.add_argument("--n_obs_steps", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--loss", choices=["mse", "l1"], default="mse")
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument(
        "--save_start",
        type=int,
        default=0,
        help="Do not write periodic checkpoints before this training step.",
    )
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_demos", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    ds = MLPBCDataset(args.zarr, n_obs_steps=args.n_obs_steps, num_demos=args.num_demos)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0, drop_last=True,
        generator=loader_generator,
    )

    obs_dim = ds.state_n.shape[1]
    act_dim = ds.action_n.shape[1]
    model = MLPBCPolicy(obs_dim=obs_dim, act_dim=act_dim, n_obs_steps=args.n_obs_steps,
                        hidden=args.hidden).to(args.device)
    print(f"[INFO] model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"loss={args.loss}")

    ema = EMA(model)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              betas=(0.95, 0.999), eps=1e-8, weight_decay=args.weight_decay)

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    lr_sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    norm_stats = {
        "s_mean": ds.s_mean.cpu(), "s_std": ds.s_std.cpu(),
        "a_center": ds.a_center.cpu(), "a_scale": ds.a_scale.cpu(),
    }

    def save(tag):
        path = os.path.join(args.output_dir, f"{tag}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.shadow,
            "norm_stats": norm_stats,
            "step": step,
            "obs_dim": obs_dim, "act_dim": act_dim,
            "n_obs_steps": args.n_obs_steps, "hidden": args.hidden,
            "loss": args.loss, "seed": args.seed,
        }, path)
        print(f"  [SAVE] {path}")

    print(f"[INFO] training {args.steps} steps, batch={args.batch_size}, lr={args.lr}")
    t0, step, losses = time.time(), 0, []
    while step < args.steps:
        for obs, action_t in loader:
            obs = obs.to(args.device, non_blocking=True)
            action_t = action_t.to(args.device, non_blocking=True)
            pred = model(obs)
            if args.loss == "mse":
                loss = nn.functional.mse_loss(pred, action_t)
            else:
                loss = nn.functional.l1_loss(pred, action_t)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            lr_sched.step()
            ema.update(model, step)
            losses.append(loss.item())
            step += 1

            if step % args.log_every == 0:
                print(f"  step {step:6d}/{args.steps}  loss={loss.item():.6f}  "
                      f"mean{args.log_every}={np.mean(losses[-args.log_every:]):.6f}  "
                      f"lr={lr_sched.get_last_lr()[0]:.2e}  elapsed={time.time()-t0:.0f}s")
            if step >= args.save_start and step % args.save_every == 0:
                save(f"mlp_bc_step_{step}")
            if step >= args.steps:
                break

    save("mlp_bc_final")
    print("[DONE]")


if __name__ == "__main__":
    main()
