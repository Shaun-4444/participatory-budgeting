"""Model-based utility sampling consistent with observed knapsack ballots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

from pb_distortion.formats.knapsack import KnapsackBallot
from pb_distortion.models import PBInstance
from pb_distortion.solvers import bundle_cost, optimal_feasible_bundle


@dataclass(frozen=True, slots=True)
class UtilitySampleDiagnostics:
    """Work performed while finding one normalized consistent utility vector."""

    lp_solves: int
    separation_cuts: int
    consistency_gap: float


def sample_knapsack_consistent_utilities(
    instance: PBInstance,
    ballot: KnapsackBallot,
    rng: np.random.Generator,
    *,
    witness_mix: float = 0.75,
    max_cuts: int = 500,
    tolerance: float = 1e-9,
) -> tuple[NDArray[np.float64], UtilitySampleDiagnostics]:
    """Sample one nonnegative, normalized utility vector rationalizing ``ballot``.

    A random linear objective is optimized over the voter-utility polytope. The
    constraints ``u(ballot) >= u(S)`` are added lazily: an exact knapsack solve
    acts as a separation oracle and finds a violating feasible bundle ``S``.
    Thus the returned vector is consistent with the complete knapsack choice
    set without enumerating its exponentially many bundles.

    Random objectives induce a non-uniform distribution over vertices of the
    consistent-utility polytope. To avoid one-hot utility vectors, the returned
    point is blended with a random Dirichlet vector supported on the submitted
    bundle. That witness vector is itself consistent: it gives the reported
    bundle total utility one and every other project utility zero. Convexity
    preserves consistency. ``witness_mix`` controls the blend and is part of
    the assumed latent-utility model, not a parameter estimated from ballots.
    """

    known_projects = set(instance.project_ids)
    unknown = set(ballot) - known_projects
    if unknown:
        raise ValueError(f"Ballot contains unknown projects: {sorted(unknown)}.")
    if bundle_cost(instance, tuple(ballot)) > instance.budget:
        raise ValueError("The observed knapsack ballot is not budget-feasible.")
    if max_cuts <= 0:
        raise ValueError("max_cuts must be positive.")
    if not np.isfinite(witness_mix) or not 0 <= witness_mix <= 1:
        raise ValueError("witness_mix must be finite and between zero and one.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")

    project_index = {project_id: index for index, project_id in enumerate(instance.project_ids)}
    ballot_vector = np.zeros(instance.project_count, dtype=np.float64)
    for project_id in ballot:
        ballot_vector[project_index[project_id]] = 1.0
    if not ballot:
        raise ValueError(
            "An empty knapsack ballot cannot be normalized into the supported-bundle "
            "utility witness used by this sampler."
        )
    ballot_indices = np.flatnonzero(ballot_vector)
    witness = np.zeros(instance.project_count, dtype=np.float64)
    witness[ballot_indices] = rng.dirichlet(np.ones(len(ballot_indices)))
    objective = rng.normal(size=instance.project_count)

    cuts: list[NDArray[np.float64]] = []
    cut_signatures: set[tuple[float, ...]] = set()
    lp_solves = 0

    for _ in range(max_cuts + 1):
        lp_result = linprog(
            c=objective,
            A_ub=np.vstack(cuts) if cuts else None,
            b_ub=np.zeros(len(cuts)) if cuts else None,
            A_eq=np.ones((1, instance.project_count), dtype=np.float64),
            b_eq=np.array([1.0]),
            bounds=(0.0, None),
            method="highs",
        )
        lp_solves += 1
        if not lp_result.success or lp_result.x is None:
            raise ValueError(
                "The recorded ballot is not consistent with a nonnegative additive "
                f"utility vector under this budget (LP status {lp_result.status}: {lp_result.message})."
            )

        utilities = np.maximum(np.asarray(lp_result.x, dtype=np.float64), 0.0)
        row_sum = float(utilities.sum())
        if row_sum <= 0 or not np.isfinite(row_sum):
            raise RuntimeError("The LP returned a utility vector that cannot be normalized.")
        utilities /= row_sum

        best_bundle = optimal_feasible_bundle(instance, utilities)
        best_value = float(sum(utilities[project_index[project_id]] for project_id in best_bundle))
        reported_value = float(utilities @ ballot_vector)
        gap = best_value - reported_value
        if gap <= tolerance:
            utilities = witness_mix * witness + (1 - witness_mix) * utilities
            mixed_best_bundle = optimal_feasible_bundle(instance, utilities)
            mixed_best_value = float(
                sum(utilities[project_index[project_id]] for project_id in mixed_best_bundle)
            )
            mixed_reported_value = float(utilities @ ballot_vector)
            mixed_gap = mixed_best_value - mixed_reported_value
            if mixed_gap > 10 * tolerance:
                raise RuntimeError("The mixed utility sample failed the knapsack consistency check.")
            utilities.setflags(write=False)
            return utilities, UtilitySampleDiagnostics(
                lp_solves=lp_solves,
                separation_cuts=len(cuts),
                consistency_gap=max(0.0, mixed_gap),
            )

        best_vector = np.zeros(instance.project_count, dtype=np.float64)
        for project_id in best_bundle:
            best_vector[project_index[project_id]] = 1.0
        cut = best_vector - ballot_vector
        signature = tuple(cut.tolist())
        if signature in cut_signatures:
            raise RuntimeError(
                "The separation oracle found a previously added violated constraint; "
                "this indicates numerical instability."
            )
        cut_signatures.add(signature)
        cuts.append(cut)

    raise RuntimeError(f"Utility inference did not converge after {max_cuts} cuts.")
