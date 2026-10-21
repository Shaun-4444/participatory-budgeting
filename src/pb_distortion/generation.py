"""Seeded generation of synthetic instances and latent utilities."""

from __future__ import annotations

import math

import numpy as np

from pb_distortion.models import Election, PBInstance, UtilityProfile


def normalize_utility_rows(values: np.ndarray) -> np.ndarray:
    """Normalize each voter's nonzero utility vector to sum to one."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Utilities must be a two-dimensional matrix.")
    totals = matrix.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Cannot normalize a voter row whose total utility is zero.")
    return matrix / totals


def perturb_utilities(
    values: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one mean-corrected lognormal perturbation per voter-project pair."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("True utilities must be a finite, nonnegative matrix.")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("noise sigma must be finite and nonnegative.")
    if sigma == 0:
        return matrix.copy()

    epsilon = rng.standard_normal(size=matrix.shape)
    perturbed = matrix * np.exp(sigma * epsilon - (sigma**2) / 2.0)
    if not np.isfinite(perturbed).all():
        raise ValueError("Noise produced non-finite perceived utilities; use a smaller sigma.")
    return perturbed


def generate_synthetic_election(
    *,
    project_count: int,
    voter_count: int,
    cost_min: int,
    cost_max: int,
    budget_fraction: float,
    rng: np.random.Generator,
    utility_distribution: str = "uniform",
    utility_minimum: float = 0.0,
    utility_maximum: float = 1.0,
    normal_mean: float = 0.5,
    normal_std: float = 0.2,
    pareto_shape: float = 1.5,
    normalize_rows: bool = True,
    noise_sigma: float = 0.0,
) -> Election:
    """Generate integer costs and additive utilities from a configured distribution."""

    if project_count <= 0 or voter_count <= 0:
        raise ValueError("Project and voter counts must both be positive.")
    if cost_min <= 0 or cost_max < cost_min:
        raise ValueError("Costs must satisfy 0 < cost_min <= cost_max.")
    if not (0 < budget_fraction <= 1):
        raise ValueError("budget_fraction must be in (0, 1].")
    if utility_distribution not in {"uniform", "normal", "pareto"}:
        raise ValueError("utility_distribution must be 'uniform', 'normal', or 'pareto'.")
    if (
        not math.isfinite(utility_minimum)
        or not math.isfinite(utility_maximum)
        or utility_minimum < 0
        or utility_maximum <= utility_minimum
    ):
        raise ValueError("Utilities must satisfy 0 <= minimum < maximum.")
    if (
        not math.isfinite(normal_mean)
        or not math.isfinite(normal_std)
        or normal_mean < 0
        or normal_std <= 0
    ):
        raise ValueError("Truncated-normal utilities require mean >= 0 and std > 0.")
    if not math.isfinite(pareto_shape) or pareto_shape <= 1:
        raise ValueError("pareto_shape must be greater than 1 for a finite-mean Pareto draw.")
    if not math.isfinite(noise_sigma) or noise_sigma < 0:
        raise ValueError("noise_sigma must be finite and nonnegative.")

    project_ids = tuple(f"p{index:03d}" for index in range(project_count))
    costs = tuple(int(cost) for cost in rng.integers(cost_min, cost_max + 1, size=project_count))
    budget = math.ceil(budget_fraction * sum(costs))
    instance = PBInstance(project_ids=project_ids, costs=costs, budget=budget)

    utility_shape = (voter_count, project_count)
    if utility_distribution == "uniform":
        values = rng.uniform(utility_minimum, utility_maximum, size=utility_shape)
    elif utility_distribution == "normal":
        # Rejection sampling gives a normal distribution conditioned on nonnegative
        # values, as required by the utility model.
        values = rng.normal(normal_mean, normal_std, size=utility_shape)
        negative = values < 0
        while negative.any():
            values[negative] = rng.normal(normal_mean, normal_std, size=int(negative.sum()))
            negative = values < 0
    else:
        # NumPy's standard Pareto has support [0, inf); adding one gives the
        # conventional Pareto-I support [1, inf) and a heavy right tail.
        values = rng.pareto(pareto_shape, size=utility_shape) + 1.0
    if normalize_rows:
        values = normalize_utility_rows(values)
    perceived_values = perturb_utilities(values, noise_sigma, rng)
    utilities = UtilityProfile(values=values, project_ids=project_ids)
    perceived_utilities = UtilityProfile(values=perceived_values, project_ids=project_ids)
    return Election(
        instance=instance,
        utilities=utilities,
        perceived_utilities=perceived_utilities,
    )
