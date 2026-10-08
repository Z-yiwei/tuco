if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import copy
import contextlib
import random
import time
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.common.camera_mask import apply_training_camera_mask

OmegaConf.register_new_resolver("eval", eval, replace=True)


class _ComputeLossModule(torch.nn.Module):
    """Expose policy.compute_loss through forward so DDP can track gradients."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch):
        return self.policy.compute_loss(batch)


class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
        # ModuleAttrMixin uses zero-sized parameters only to expose device/dtype.
        # They are not part of the loss and must not enter DDP gradient buckets.
        for parameter in self.model.parameters():
            if parameter.numel() == 0:
                parameter.requires_grad_(False)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        distributed = world_size > 1
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        is_main = rank == 0
        if distributed:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
            cfg.training.device = f"cuda:{local_rank}"

        # Model initialization uses the same seed on every rank. Runtime
        # randomness must differ so ranks do not use identical crops and noise.
        runtime_seed = int(cfg.training.seed) + rank
        torch.manual_seed(runtime_seed)
        np.random.seed(runtime_seed)
        random.seed(runtime_seed)
        torch.backends.cudnn.benchmark = bool(
            OmegaConf.select(cfg, "training.cudnn_benchmark", default=False)
        )

        # resume training
        resumed = False
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                # Checkpoints are written after the last batch and validation,
                # immediately before these counters are advanced at epoch end.
                self.global_step += 1
                self.epoch += 1
                resumed = True

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_loader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
        train_sampler = None
        if distributed:
            global_batch_size = int(train_loader_cfg["batch_size"])
            if global_batch_size % world_size != 0:
                raise ValueError(
                    f"Global batch size {global_batch_size} must be divisible by "
                    f"world size {world_size}."
                )
            train_loader_cfg["batch_size"] = global_batch_size // world_size
            configured_workers = int(train_loader_cfg.get("num_workers", 0))
            if configured_workers > 0:
                train_loader_cfg["num_workers"] = max(
                    1, configured_workers // world_size
                )
            train_sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=bool(train_loader_cfg.pop("shuffle", False)),
                seed=int(cfg.training.seed),
            )
            train_loader_cfg["sampler"] = train_sampler
        train_dataloader = DataLoader(dataset, **train_loader_cfg)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataloader = None
        if is_main:
            val_dataset = dataset.get_validation_dataset()
            val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        keep_checkpoint_normalizer = bool(
            OmegaConf.select(
                cfg, "training.keep_checkpoint_normalizer", default=False
            )
        )
        if resumed and keep_checkpoint_normalizer:
            if is_main:
                print("TRAIN_NORMALIZER source=checkpoint")
        else:
            self.model.set_normalizer(normalizer)
            if cfg.training.use_ema:
                self.ema_model.set_normalizer(normalizer)
            if is_main:
                print("TRAIN_NORMALIZER source=dataset")

        # Keep the scheduler budget aligned with the capped training loop. This
        # matters when datasets have different lengths but max_train_steps is
        # used to give each run the same optimizer-step budget.
        scheduler_steps_per_epoch = len(train_dataloader)
        max_train_steps = OmegaConf.select(
            cfg, "training.max_train_steps", default=None
        )
        if max_train_steps is not None:
            scheduler_steps_per_epoch = min(
                scheduler_steps_per_epoch, int(max_train_steps)
            )
        if is_main:
            print(
                "TRAIN_SCHEDULER "
                f"steps_per_epoch={scheduler_steps_per_epoch} "
                f"loader_steps={len(train_dataloader)} "
                f"num_epochs={int(cfg.training.num_epochs)}"
            )

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                scheduler_steps_per_epoch * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
            # The checkpoint stores global_step and ema_model, but EMAModel is
            # local to run(). Restore its warmup counter instead of restarting
            # EMA decay from zero and overwriting the historical average.
            if resumed:
                ema.optimization_step = self.global_step
                ema.decay = ema.get_decay(ema.optimization_step)
                if is_main:
                    print(
                        "TRAIN_RESUME "
                        f"epoch={self.epoch} "
                        f"global_step={self.global_step} "
                        f"ema_optimization_step={ema.optimization_step} "
                        f"ema_decay={ema.decay:.8f}"
                    )

        # configure env
        env_runner = None
        if is_main:
            env_runner: BaseImageRunner
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)
            assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        wandb_run = None
        if is_main:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # configure checkpoint
        topk_manager = None
        if is_main:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        if cfg.training.freeze_encoder:
            self.model.obs_encoder.eval()
            self.model.obs_encoder.requires_grad_(False)
        train_model = self.model
        loss_model = _ComputeLossModule(self.model)
        if distributed:
            train_model = DistributedDataParallel(
                loss_model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
            )

        if is_main:
            print(
                "TRAIN_DISTRIBUTED "
                f"world_size={world_size} "
                f"global_batch_size={int(cfg.dataloader.batch_size)} "
                f"local_batch_size={int(train_loader_cfg['batch_size'])} "
                f"local_workers={int(train_loader_cfg.get('num_workers', 0))}"
            )

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        logger_context = (
            JsonLogger(log_path) if is_main else contextlib.nullcontext(None)
        )
        with logger_context as json_logger:
            for local_epoch_idx in range(self.epoch, cfg.training.num_epochs):
                if train_sampler is not None:
                    train_sampler.set_epoch(self.epoch)
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                camera_mask_mode = str(
                    OmegaConf.select(cfg, "training.camera_mask_mode", default="none")
                )
                camera_mask_keys = list(
                    OmegaConf.select(
                        cfg,
                        "training.camera_mask_keys",
                        default=["front_rgb", "side_rgb", "wrist_rgb"],
                    )
                )
                camera_mask_counts = torch.zeros(
                    len(camera_mask_keys) + 1, dtype=torch.int64, device=device
                )
                benchmark_only = bool(
                    OmegaConf.select(cfg, "training.benchmark_only", default=False)
                )
                benchmark_warmup_steps = int(
                    OmegaConf.select(cfg, "training.benchmark_warmup_steps", default=20)
                )
                benchmark_start = None
                benchmark_batches = 0
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec,
                        disable=not is_main) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        batch, batch_camera_mask_counts = apply_training_camera_mask(
                            batch,
                            mode=camera_mask_mode,
                            camera_keys=camera_mask_keys,
                            seed=int(cfg.training.seed),
                            global_step=self.global_step,
                            rank=rank,
                        )
                        camera_mask_counts += batch_camera_mask_counts

                        # compute loss
                        if distributed:
                            raw_loss = train_model(batch)
                        else:
                            raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        if benchmark_only and batch_idx == benchmark_warmup_steps:
                            torch.cuda.synchronize(device)
                            torch.cuda.reset_peak_memory_stats(device)
                            benchmark_start = time.perf_counter()
                        if benchmark_start is not None:
                            benchmark_batches += 1
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        extra_metrics = getattr(self.model, "_last_metrics", None)
                        if isinstance(extra_metrics, dict):
                            for key, value in extra_metrics.items():
                                try:
                                    step_log[f"train/{key}"] = float(value)
                                except (TypeError, ValueError):
                                    pass

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if is_main:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                if distributed:
                    reduced_train_loss = torch.tensor(train_loss, device=device)
                    dist.all_reduce(reduced_train_loss, op=dist.ReduceOp.SUM)
                    train_loss = reduced_train_loss.item() / world_size
                step_log['train_loss'] = train_loss
                camera_mask_total = int(camera_mask_counts.sum().item())
                camera_mask_labels = ["none", *camera_mask_keys]
                for index, label in enumerate(camera_mask_labels):
                    count = int(camera_mask_counts[index].item())
                    step_log[f"train/camera_mask_{label}_count"] = count
                    step_log[f"train/camera_mask_{label}_fraction"] = (
                        count / camera_mask_total if camera_mask_total else 0.0
                    )

                if benchmark_only:
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - benchmark_start
                    elapsed_tensor = torch.tensor(elapsed, device=device)
                    if distributed:
                        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
                    elapsed = elapsed_tensor.item()
                    global_batch_size = int(cfg.dataloader.batch_size)
                    if is_main:
                        print(
                            "TRAIN_BENCHMARK "
                            f"cudnn_benchmark={int(torch.backends.cudnn.benchmark)} "
                            f"world_size={world_size} "
                            f"global_batch_size={global_batch_size} "
                            f"local_batch_size={int(train_loader_cfg['batch_size'])} "
                            f"local_workers={int(train_loader_cfg.get('num_workers', 0))} "
                            f"timed_batches={benchmark_batches} "
                            f"elapsed_s={elapsed:.6f} "
                            f"batches_per_s={benchmark_batches / elapsed:.6f} "
                            f"samples_per_s={benchmark_batches * global_batch_size / elapsed:.6f} "
                            f"peak_memory_gib={torch.cuda.max_memory_allocated(device) / 2**30:.6f} "
                            f"train_loss={train_loss:.8f}"
                        )
                        wandb_run.finish()
                    if distributed:
                        dist.destroy_process_group()
                    return

                # ========= eval for this epoch ==========
                if is_main:
                    policy = self.model
                    if cfg.training.use_ema:
                        policy = self.ema_model
                    policy.eval()

                # run rollout
                if is_main and (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if is_main and (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if is_main and (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = dict(batch['obs'])
                        if getattr(policy, "requires_past_action", False) and "past_action" in batch:
                            obs_dict["past_action"] = batch["past_action"]
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        if pred_action.shape != gt_action.shape:
                            start = int(getattr(policy, "n_obs_steps", cfg.n_obs_steps)) - 1
                            end = start + pred_action.shape[1]
                            gt_action = gt_action[:, start:end, :pred_action.shape[-1]]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if is_main and (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                if is_main:
                    policy.train()

                if distributed:
                    dist.barrier()

                # end of epoch
                # log of last step is combined with validation and rollout
                if is_main:
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        if is_main:
            wandb_run.finish()
        if distributed:
            dist.destroy_process_group()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
