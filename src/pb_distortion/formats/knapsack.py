"""Knapsack ballots: each voter submits their favorite feasible bundle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pb_distortion.models import Outcome, PBInstance
from pb_distortion.solvers import solve_knapsack


KnapsackBallot = frozenset[str]


@dataclass(frozen=True, slots=True)
class KnapsackBallotFormat:
    """Elicit an exact utility-maximizing bundle under the public budget."""

    name: str = "knapsack"

    def elicit(
        self,
        utilities: NDArray[np.float64],
        instance: PBInstance,
    ) -> KnapsackBallot:
        values = np.asarray(utilities, dtype=np.float64)
        if values.ndim != 1 or values.shape[0] != instance.project_count:
            raise ValueError("A voter's utility vector must have one value per project.")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Utilities must be finite and non-negative.")
        bundle: Outcome = solve_knapsack(
            instance.project_ids,
            instance.costs,
            values,
            instance.budget,
        )
        return frozenset(bundle)
