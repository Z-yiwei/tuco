"""Dummy image env runner that skips rollout evaluation.
Used when training data comes from a non-robosuite env (e.g. OmniReset vision).
Mirrors dummy_lowdim_runner.DummyLowdimRunner for the image policy path."""

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class DummyImageRunner(BaseImageRunner):
    def __init__(self, output_dir, **kwargs):
        super().__init__(output_dir)

    def run(self, policy):
        return {"test/mean_score": 0.0}
