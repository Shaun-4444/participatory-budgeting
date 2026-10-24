import numpy as np

from pb_distortion.formats.knapsack import KnapsackBallotFormat
from pb_distortion.models import PBInstance
from pb_distortion.solvers import bundle_cost


def test_knapsack_ballot_is_feasible_and_utility_maximizing():
    instance = PBInstance(("a", "b", "c", "d"), (2, 3, 4, 2), 6)
    utilities = np.array([0.4, 0.5, 0.9, 0.35])
    ballot = KnapsackBallotFormat().elicit(utilities, instance)

    assert bundle_cost(instance, tuple(ballot)) <= instance.budget
    selected_value = sum(utilities[instance.project_ids.index(project_id)] for project_id in ballot)
    assert selected_value == 1.3
    assert ballot == frozenset({"a", "c"})


def test_tied_ballots_are_reproducible():
    instance = PBInstance(("b", "a"), (1, 1), 1)
    ballot = KnapsackBallotFormat().elicit(np.array([1.0, 1.0]), instance)
    assert ballot == frozenset({"a"})
