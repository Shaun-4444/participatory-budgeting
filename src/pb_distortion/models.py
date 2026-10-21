"""Core, ballot-format-neutral data models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PBInstance:
    """A participatory-budgeting instance with integer project costs."""

    project_ids: tuple[str, ...]
    costs: tuple[int, ...]
    budget: int

    def __post_init__(self) -> None:
        project_ids = tuple(self.project_ids)
        costs = tuple(self.costs)
        object.__setattr__(self, "project_ids", project_ids)
        object.__setattr__(self, "costs", costs)

        if not project_ids:
            raise ValueError("An instance must contain at least one project.")
        if len(set(project_ids)) != len(project_ids):
            raise ValueError("Project IDs must be unique.")
        if any(not isinstance(project_id, str) or not project_id for project_id in project_ids):
            raise ValueError("Project IDs must be non-empty strings.")
        if len(costs) != len(project_ids):
            raise ValueError("There must be exactly one cost per project.")
        if any(isinstance(cost, bool) or not isinstance(cost, int) or cost <= 0 for cost in costs):
            raise ValueError("Project costs must be positive integers.")
        if isinstance(self.budget, bool) or not isinstance(self.budget, int) or self.budget < 0:
            raise ValueError("The budget must be a non-negative integer.")

    @property
    def project_count(self) -> int:
        return len(self.project_ids)

    @property
    def total_cost(self) -> int:
        return sum(self.costs)

    def cost_by_project(self) -> dict[str, int]:
        return dict(zip(self.project_ids, self.costs, strict=True))


@dataclass(frozen=True, slots=True)
class UtilityProfile:
    """Voter-by-project cardinal utilities, aligned to an instance's project order."""

    values: NDArray[np.float64]
    project_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        project_ids = tuple(self.project_ids)
        values = np.array(self.values, dtype=np.float64, copy=True)
        if values.ndim != 2:
            raise ValueError("Utility values must be a two-dimensional voter-by-project matrix.")
        if values.shape[1] != len(project_ids):
            raise ValueError("Utility columns must match the number of project IDs.")
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("A utility profile must include at least one voter and one project.")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Utilities must be finite and non-negative.")
        if len(set(project_ids)) != len(project_ids):
            raise ValueError("Project IDs must be unique.")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "project_ids", project_ids)

    @property
    def voter_count(self) -> int:
        return self.values.shape[0]

    @property
    def project_count(self) -> int:
        return self.values.shape[1]


@dataclass(frozen=True, slots=True)
class Election:
    """A PB instance with voters' true and perceived utility profiles."""

    instance: PBInstance
    utilities: UtilityProfile
    perceived_utilities: UtilityProfile | None = None

    def __post_init__(self) -> None:
        if self.instance.project_ids != self.utilities.project_ids:
            raise ValueError("Instance and utility-profile project orders must match exactly.")
        perceived = (
            self.utilities
            if self.perceived_utilities is None
            else self.perceived_utilities
        )
        if perceived.project_ids != self.instance.project_ids:
            raise ValueError("Perceived utilities must use the instance project order.")
        if perceived.values.shape != self.utilities.values.shape:
            raise ValueError("True and perceived utility profiles must have the same shape.")
        object.__setattr__(self, "perceived_utilities", perceived)


Outcome = tuple[str, ...]
