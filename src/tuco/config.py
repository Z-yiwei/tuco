"""Method configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TucoConfig:
    """Hyperparameters shared by every experiment family.

    ``tau`` is the implementation name for the paper's cone threshold gamma.
    Values follow the default configuration highlighted in
    the paper appendix.
    """

    tau: float = 0.01
    lambda_cov: float = 0.2
    rho: float = 0.1
    eps: float = 1e-8

    def validate(self) -> None:
        if not 0.0 <= self.tau < 1.0:
            raise ValueError("tau must satisfy 0 <= tau < 1")
        if self.lambda_cov < 0.0:
            raise ValueError("lambda_cov must be non-negative")
        if self.rho <= 0.0:
            raise ValueError("rho must be positive")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


PAPER_AGGREGATION = "target_sum_candidate_sum"
PAPER_CURVATURE_RIDGE = 0.0
