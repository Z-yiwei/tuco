"""Dummy env runner that skips rollout evaluation.
Used when the training data comes from a non-robosuite environment (e.g., OmniReset)."""

from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class DummyLowdimRunner(BaseLowdimRunner):
    def __init__(self, output_dir, **kwargs):
        super().__init__(output_dir)

    def run(self, policy):
        return {"test/mean_score": 0.0}
