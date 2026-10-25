"""Value ranking ballot format: voters rank projects by raw utility.

Each voter submits a strict ordinal ranking of candidate projects in descending
order of their raw perceived utility u_i(j) for all projects with positive utility
u_i(j) > 0. Projects with zero utility are unranked. Ties are broken deterministically
by ascending project ID. Only ordinal ordering is retained; cardinal magnitudes are
discarded.

Reference: Benadè et al. (2021), "Preference Elicitation for Participatory Budgeting",
           Management Science 67(5):2813–2827. Section 3.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pb_distortion.models import PBInstance


ValueRankingBallot = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ValueRankingFormat:
    """Elicit a ranking of projects by raw perceived utility, highest first.

    The returned ballot is an ordered tuple of project IDs in descending utility order
    for projects with positive utility (u_i(j) > 0). Ties are broken by ascending
    project ID for full determinism. Projects with zero utility are not ranked.
    """

    name: str = "value_ranking"

    def elicit(
        self,
        utilities: NDArray[np.float64],
        instance: PBInstance,
    ) -> ValueRankingBallot:
        values = np.asarray(utilities, dtype=np.float64)
        if values.ndim != 1 or values.shape[0] != instance.project_count:
            raise ValueError("A voter's utility vector must have one value per project.")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Utilities must be finite and non-negative.")

        project_ids = instance.project_ids
        positive_indices = [index for index, val in enumerate(values) if val > 0]
        ordering = sorted(
            positive_indices,
            key=lambda index: (-values[index], project_ids[index]),
        )
        return tuple(project_ids[index] for index in ordering)
