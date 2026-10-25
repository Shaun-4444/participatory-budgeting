"""Threshold approval ballots: each voter approves alternatives with utility above a threshold."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pb_distortion.models import PBInstance


ThresholdApprovalBallot = frozenset[str]


@dataclass(frozen=True, slots=True)
class ThresholdApprovalBallotFormat:
    """Elicit approval for alternatives meeting or exceeding a utility threshold.

    When threshold is None (the default), threshold is dynamically set to 1/m,
    representing the voter's average project value under normalized utilities, as
    studied by Benadè et al. (Theorem 14).

    threshold can also be a mapping from project_count -> threshold float,
    enabling per-project-count learned thresholds from validation sets.
    """

    name: str = "threshold_approval"
    threshold: float | Mapping[int, float] | None = None
    relative: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.threshold, Mapping):
            for count, t in self.threshold.items():
                if not isinstance(count, int) or count <= 0:
                    raise ValueError("Project count in threshold mapping must be a positive integer.")
                if not math.isfinite(t) or t < 0:
                    raise ValueError("Threshold in mapping must be a finite non-negative number.")
        elif self.threshold is not None:
            if not math.isfinite(self.threshold) or self.threshold < 0:
                raise ValueError("Approval threshold must be a finite non-negative number.")

    def effective_threshold(
        self,
        utilities: NDArray[np.float64],
        instance: PBInstance,
    ) -> float:
        """Determine the cutoff threshold for the given voter and instance."""
        if isinstance(self.threshold, Mapping):
            base_threshold = self.threshold.get(
                instance.project_count,
                1.0 / instance.project_count,
            )
        elif self.threshold is not None:
            base_threshold = self.threshold
        else:
            base_threshold = 1.0 / instance.project_count

        if self.relative:
            total_utility = float(utilities.sum())
            return base_threshold * total_utility
        return base_threshold

    def elicit(
        self,
        utilities: NDArray[np.float64],
        instance: PBInstance,
    ) -> ThresholdApprovalBallot:
        values = np.asarray(utilities, dtype=np.float64)
        if values.ndim != 1 or values.shape[0] != instance.project_count:
            raise ValueError("A voter's utility vector must have one value per project.")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Utilities must be finite and non-negative.")

        cutoff = self.effective_threshold(values, instance)
        # Using a slight tolerance (1e-12) to avoid strict float inequality precision issues
        approved = [
            project_id
            for project_id, value in zip(instance.project_ids, values, strict=True)
            if value >= cutoff - 1e-12
        ]
        return frozenset(approved)
