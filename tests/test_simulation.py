import math

import numpy as np

from pb_distortion.formats.knapsack import KnapsackBallotFormat
from pb_distortion.generation import generate_synthetic_election
from pb_distortion.rules.greedy_approval import GreedyApprovalRule
from pb_distortion.simulation import SimulationSettings, run_simulation


def test_generated_baseline_has_normalized_utilities_and_expected_budget():
    election = generate_synthetic_election(
        project_count=10,
        voter_count=20,
        cost_min=1,
        cost_max=10,
        budget_fraction=0.3,
        rng=np.random.default_rng(531),
    )
    assert np.allclose(election.utilities.values.sum(axis=1), 1.0)
    assert all(isinstance(cost, int) and 1 <= cost <= 10 for cost in election.instance.costs)
    assert election.instance.budget == math.ceil(0.3 * election.instance.total_cost)


def test_seeded_simulation_repeats_exactly():
    settings = SimulationSettings(
        trials=4,
        master_seed=12345,
        project_count=6,
        voter_count=12,
        cost_min=1,
        cost_max=5,
        budget_fraction=0.3,
    )
    first = run_simulation(settings, KnapsackBallotFormat(), GreedyApprovalRule())
    second = run_simulation(settings, KnapsackBallotFormat(), GreedyApprovalRule())

    assert first == second
    assert len({result.seed for result in first}) == settings.trials
    for result in first:
        assert result.benchmark_exact
        assert result.optimal_welfare >= result.chosen_welfare
        assert result.distortion >= 1.0


def test_noise_uses_perceived_utilities_but_scores_true_utilities():
    settings = SimulationSettings(
        trials=4,
        master_seed=987,
        project_count=8,
        voter_count=12,
        cost_min=1,
        cost_max=5,
        budget_fraction=0.3,
        noise_sigma=0.8,
    )
    format_ = KnapsackBallotFormat()
    rule = GreedyApprovalRule()

    first = run_simulation(settings, format_, rule)
    second = run_simulation(settings, format_, rule)

    assert first == second
    assert all(result.noise_sigma == 0.8 for result in first)
    assert all(result.optimal_welfare >= result.chosen_welfare for result in first)
    assert all(result.distortion >= 1.0 for result in first)


def test_noise_zero_preserves_baseline_simulation_results():
    settings = SimulationSettings(
        trials=3,
        master_seed=123,
        project_count=7,
        voter_count=10,
        cost_min=1,
        cost_max=5,
        budget_fraction=0.3,
    )
    zero_noise_settings = SimulationSettings(
        trials=3,
        master_seed=123,
        project_count=7,
        voter_count=10,
        cost_min=1,
        cost_max=5,
        budget_fraction=0.3,
        noise_sigma=0.0,
    )

    baseline = run_simulation(settings, KnapsackBallotFormat(), GreedyApprovalRule())
    zero_noise = run_simulation(zero_noise_settings, KnapsackBallotFormat(), GreedyApprovalRule())

    assert baseline == zero_noise
