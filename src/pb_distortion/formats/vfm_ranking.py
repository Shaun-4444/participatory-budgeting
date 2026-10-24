"""Value-for-money ranking ballot format: voters rank all projects by u_i(j) / c_j.

Each voter submits a strict ordinal ranking of all candidate projects in descending
order of perceived value-for-money (utility per unit cost). Ties are broken
deterministically by ascending project ID. Only ordinal ordering is retained;
cardinal magnitudes are discarded.

Reference: Benadè et al. (2021), "Preference Elicitation for Participatory Budgeting",
           Management Science 67(5):2813–2827. Section 3.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pb_distortion.models import PBInstance


VFMRankingBallot = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VFMRankingFormat:
    """Elicit a ranking of all projects by value-for-money u_i(j)/c_j, highest first.

    The returned ballot is an ordered tuple of all project IDs in descending bang-per-buck order.
    Ties are broken by ascending project ID for full determinism.
    """

    name: str = "vfm_ranking"

    def elicit(
        self,
        utilities: NDArray[np.float64],
        instance: PBInstance,
    ) -> VFMRankingBallot:
        values = np.asarray(utilities, dtype=np.float64)
        if values.ndim != 1 or values.shape[0] != instance.project_count:
            raise ValueError("A voter's utility vector must have one value per project.")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Utilities must be finite and non-negative.")

        costs = np.array(instance.costs, dtype=np.float64)
        vfm_scores = values / costs
        project_ids = instance.project_ids
        ordering = sorted(
            range(len(project_ids)),
            key=lambda index: (-vfm_scores[index], project_ids[index]),
        )
        return tuple(project_ids[index] for index in ordering)

