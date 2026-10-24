"""Exact solvers for integer-cost knapsack problems."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pb_distortion.models import Outcome, PBInstance


def solve_knapsack(
    project_ids: Sequence[str],
    costs: Sequence[int],
    values: Sequence[float],
    budget: int,
) -> Outcome:
    """Return an exactly value-maximizing feasible subset using dynamic programming.

    Ties are resolved deterministically: projects are processed in ascending ID order,
    and equal-value decisions skip the later project. Returned IDs follow that order.
    Runtime is O(project_count * budget), vectorized over budget capacities with
    NumPy. This keeps exact tie-breaking while reducing Python-loop overhead.
    """

    ids = tuple(project_ids)
    item_costs = tuple(costs)
    item_values = tuple(float(value) for value in values)
    if not (len(ids) == len(item_costs) == len(item_values)):
        raise ValueError("Project IDs, costs, and values must have the same length.")
    if len(set(ids)) != len(ids):
        raise ValueError("Project IDs must be unique.")
    if any(isinstance(cost, bool) or not isinstance(cost, int) or cost <= 0 for cost in item_costs):
        raise ValueError("Project costs must be positive integers.")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("The budget must be a non-negative integer.")
    if any(not math.isfinite(value) or value < 0 for value in item_values):
        raise ValueError("Project values must be finite and non-negative.")

    ordered = sorted(zip(ids, item_costs, item_values, strict=True), key=lambda item: item[0])
    count = len(ordered)
    best_values = np.zeros(budget + 1, dtype=np.float64)
    take_project = np.zeros((count, budget + 1), dtype=np.bool_)
    for index, (_, cost, value) in enumerate(ordered):
        if cost > budget:
            continue
        include_values = best_values[:-cost] + value
        take = include_values > best_values[cost:]
        take_project[index, cost:] = take
        best_values[cost:] = np.maximum(best_values[cost:], include_values)

    selected: list[str] = []
    capacity = budget
    for index in range(count - 1, -1, -1):
        project_id, cost, _ = ordered[index]
        if take_project[index, capacity]:
            selected.append(project_id)
            capacity -= cost
    selected.reverse()
    return tuple(selected)


def optimal_feasible_bundle(instance: PBInstance, project_values: Sequence[float]) -> Outcome:
    """Solve the social-welfare knapsack problem exactly for an instance."""

    if len(project_values) != instance.project_count:
        raise ValueError("There must be one social value per project.")
    return solve_knapsack(instance.project_ids, instance.costs, project_values, instance.budget)


def bundle_cost(instance: PBInstance, project_ids: Sequence[str]) -> int:
    """Return bundle cost, rejecting unknown or duplicate IDs."""

    costs = instance.cost_by_project()
    if len(set(project_ids)) != len(project_ids):
        raise ValueError("A bundle cannot contain duplicate project IDs.")
    unknown = set(project_ids) - costs.keys()
    if unknown:
        raise ValueError(f"Unknown project IDs in bundle: {sorted(unknown)}")
    return sum(costs[project_id] for project_id in project_ids)
