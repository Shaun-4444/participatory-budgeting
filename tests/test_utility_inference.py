from __future__ import annotations

import numpy as np

from pb_distortion.models import PBInstance
from pb_distortion.solvers import optimal_feasible_bundle
from pb_distortion.utility_inference import sample_knapsack_consistent_utilities


def test_sampled_utilities_are_normalized_and_rationalize_ballot() -> None:
    instance = PBInstance(("a", "b", "c", "d"), (2, 1, 2, 3), 3)
    ballot = frozenset({"a", "b"})

    utilities, diagnostics = sample_knapsack_consistent_utilities(
        instance,
        ballot,
        np.random.default_rng(21),
    )

    optimum = optimal_feasible_bundle(instance, utilities)
    best_value = sum(utilities[instance.project_ids.index(project_id)] for project_id in optimum)
    ballot_value = sum(utilities[instance.project_ids.index(project_id)] for project_id in ballot)
    assert np.isclose(utilities.sum(), 1.0)
    assert (utilities >= 0).all()
    assert (utilities > 0).sum() >= len(ballot)
    assert best_value <= ballot_value + 1e-8
    assert diagnostics.lp_solves == diagnostics.separation_cuts + 1
    assert not utilities.flags.writeable


def test_utility_sampling_rejects_infeasible_ballot() -> None:
    instance = PBInstance(("a", "b"), (2, 2), 2)

    try:
        sample_knapsack_consistent_utilities(
            instance,
            frozenset({"a", "b"}),
            np.random.default_rng(1),
        )
    except ValueError as error:
        assert "not budget-feasible" in str(error)
    else:
        raise AssertionError("Expected an infeasible ballot to be rejected.")
