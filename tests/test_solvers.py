from itertools import combinations

import pytest

from pb_distortion.models import PBInstance
from pb_distortion.solvers import bundle_cost, optimal_feasible_bundle, solve_knapsack


def exhaustive_value(project_ids, costs, values, budget):
    best = 0.0
    for size in range(len(project_ids) + 1):
        for indices in combinations(range(len(project_ids)), size):
            if sum(costs[index] for index in indices) <= budget:
                best = max(best, sum(values[index] for index in indices))
    return best


@pytest.mark.parametrize(
    "costs,values,budget",
    [
        ((2, 3, 4), (4.0, 5.0, 8.0), 5),
        ((1, 1, 1, 1), (0.2, 0.9, 0.4, 0.1), 2),
        ((3, 2, 5), (1.0, 1.0, 1.0), 5),
        ((2, 4), (0.0, 3.0), 1),
    ],
)
def test_dynamic_program_matches_exhaustive_enumeration(costs, values, budget):
    ids = tuple(f"p{i}" for i in range(len(costs)))
    selected = solve_knapsack(ids, costs, values, budget)
    selected_value = sum(values[ids.index(project_id)] for project_id in selected)
    assert bundle_cost(PBInstance(ids, costs, budget), selected) <= budget
    assert selected_value == pytest.approx(exhaustive_value(ids, costs, values, budget))


def test_project_id_order_makes_ties_deterministic():
    assert solve_knapsack(("b", "a"), (1, 1), (1.0, 1.0), 1) == ("a",)


def test_optimal_bundle_checks_number_of_values():
    instance = PBInstance(("a", "b"), (1, 2), 2)
    with pytest.raises(ValueError, match="one social value"):
        optimal_feasible_bundle(instance, (1.0,))
