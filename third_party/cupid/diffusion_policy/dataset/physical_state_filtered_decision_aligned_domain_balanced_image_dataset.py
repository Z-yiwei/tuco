"""Mix selected physical-state sim episodes with decision-aligned real data."""

from diffusion_policy.dataset.decision_aligned_domain_balanced_image_dataset import (
    DecisionAlignedDomainBalancedImageDataset,
)
from diffusion_policy.dataset.physical_state_filtered_robomimic_replay_image_dataset import (
    PhysicalStateFilteredRobomimicReplayImageDataset,
)


class PhysicalStateFilteredDecisionAlignedDomainBalancedImageDataset(
    DecisionAlignedDomainBalancedImageDataset,
    PhysicalStateFilteredRobomimicReplayImageDataset,
):
    """Apply the physical-state mask before constructing the real/sim sampler.

    The cooperative MRO is intentional: ``DecisionAlignedDomainBalancedImageDataset``
    calls ``super().__init__``, which first executes the physical-state filtering
    initializer. Consequently its fixed-ratio real sampler is sized from only
    the selected simulation sequences. The paper-facing launcher also computes
    normalization from the selected simulation training split only.
    """
