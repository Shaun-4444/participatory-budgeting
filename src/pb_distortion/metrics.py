"""Welfare metrics for empirical experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pb_distortion.models import UtilityProfile


def social_welfare(profile: UtilityProfile, project_ids: Sequence[str]) -> float:
    """Sum all voters' additive utility for a selected bundle."""

    selected = set(project_ids)
    unknown = selected - set(profile.project_ids)
    if unknown:
        raise ValueError(f"Unknown project IDs in outcome: {sorted(unknown)}")
    columns = [index for index, project_id in enumerate(profile.project_ids) if project_id in selected]
    if not columns:
        return 0.0
    return float(profile.values[:, columns].sum())


def distortion(optimal_welfare: float, chosen_welfare: float) -> float:
    """Return optimal/chosen welfare for this sampled instance."""

    if optimal_welfare < 0 or chosen_welfare < 0:
        raise ValueError("Welfare values must be non-negative.")
    if optimal_welfare == 0:
        return 1.0
    if chosen_welfare == 0:
        return math.inf
    distortion_value = optimal_welfare / chosen_welfare
    if distortion_value < 1.0 and math.isclose(
        distortion_value, 1.0, rel_tol=1e-12, abs_tol=1e-12
    ):
        return 1.0
    if distortion_value < 1.0:
        raise ValueError("Chosen welfare cannot exceed the exact optimum.")
    return distortion_value
