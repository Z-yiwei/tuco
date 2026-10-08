"""CupCake workspace that emits one unambiguous final checkpoint."""

from pathlib import Path

from diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
    TrainDiffusionUnetImageWorkspace,
)


class CupCakeCotrainWorkspace(TrainDiffusionUnetImageWorkspace):
    def run(self) -> None:
        super().run()
        if self._saving_thread is not None:
            self._saving_thread.join()
        method = str(self.cfg.curation_method)
        path = Path(self.output_dir) / "checkpoints" / f"{method}_epoch150.ckpt"
        if path.exists():
            raise FileExistsError(path)
        self.epoch -= 1
        self.global_step -= 1
        try:
            self.save_checkpoint(path=path, use_thread=False)
        finally:
            self.epoch += 1
            self.global_step += 1
