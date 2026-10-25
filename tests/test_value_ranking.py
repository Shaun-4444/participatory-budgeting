"""Tests for value-ranking ballot format and aggregation rule."""

from __future__ import annotations

import math
import numpy as np
import pytest

from pb_distortion.formats.value_ranking import ValueRankingBallot, ValueRankingFormat
from pb_distortion.generation import generate_synthetic_election
from pb_distortion.metrics import distortion, social_welfare
from pb_distortion.models import PBInstance
from pb_distortion.rules.value_ranking import HarmonicValueRankingRule
from pb_distortion.simulation import SimulationSettings, evaluate_election, run_simulation
from pb_distortion.solvers import bundle_cost, optimal_feasible_bundle


def _instance(*args: tuple[str, int], budget: int) -> PBInstance:
    ids, costs = zip(*args)
    return PBInstance(project_ids=ids, costs=costs, budget=budget)


class TestValueRankingFormat:
    def test_ballot_contains_all_projects_when_all_positive(self):
        instance = _instance(("a", 2), ("b", 3), ("c", 1), budget=5)
        utilities = np.array([0.5, 0.8, 0.2])
        ballot = ValueRankingFormat().elicit(utilities, instance)
        assert set(ballot) == {"a", "b", "c"}
        assert len(ballot) == 3

    def test_ballot_is_sorted_descending_by_utility(self):
        instance = _instance(("a", 2), ("b", 3), ("c", 1), budget=5)
        utilities = np.array([0.5, 0.8, 0.2])
        ballot = ValueRankingFormat().elicit(utilities, instance)
        assert ballot == ("b", "a", "c")

    def test_tie_broken_by_ascending_project_id(self):
        instance = _instance(("b", 2), ("a", 2), budget=2)
        utilities = np.array([0.5, 0.5])
        ballot = ValueRankingFormat().elicit(utilities, instance)
        assert ballot == ("a", "b")

    def test_single_project(self):
        instance = _instance(("x", 3), budget=3)
        ballot = ValueRankingFormat().elicit(np.array([0.7]), instance)
        assert ballot == ("x",)

    def test_rejects_wrong_length(self):
        instance = _instance(("a", 1), ("b", 1), budget=2)
        with pytest.raises(ValueError):
            ValueRankingFormat().elicit(np.array([0.5]), instance)

    def test_rejects_negative_utility(self):
        instance = _instance(("a", 1), ("b", 1), budget=2)
        with pytest.raises(ValueError):
            ValueRankingFormat().elicit(np.array([-0.1, 0.5]), instance)

    def test_zero_utilities_produces_empty_ballot(self):
        instance = _instance(("c", 1), ("a", 1), ("b", 1), budget=3)
        ballot = ValueRankingFormat().elicit(np.zeros(3), instance)
        assert ballot == ()

    def test_partial_positive_utilities_only_ranks_positive(self):
        instance = _instance(("a", 1), ("b", 1), ("c", 1), budget=3)
        utilities = np.array([0.5, 0.0, 0.2])
        ballot = ValueRankingFormat().elicit(utilities, instance)
        assert ballot == ("a", "c")

    def test_ballot_is_deterministic(self):
        instance = _instance(("a", 2), ("b", 3), ("c", 1), budget=5)
        utilities = np.array([0.3, 0.6, 0.1])
        b1 = ValueRankingFormat().elicit(utilities, instance)
        b2 = ValueRankingFormat().elicit(utilities, instance)
        assert b1 == b2


class TestHarmonicValueRankingRule:
    def test_harmonic_scores_single_voter(self):
        instance = _instance(("a", 1), ("b", 2), ("c", 3), budget=6)
        ballot: ValueRankingBallot = ("b", "a", "c")
        scores = HarmonicValueRankingRule().harmonic_scores([ballot], instance)
        assert scores["b"] == pytest.approx(1.0)
        assert scores["a"] == pytest.approx(0.5)
        assert scores["c"] == pytest.approx(1.0 / 3)

    def test_harmonic_scores_multi_voters(self):
        instance = _instance(("a", 1), ("b", 2), budget=3)
        ballots: list[ValueRankingBallot] = [("a", "b"), ("b", "a")]
        scores = HarmonicValueRankingRule().harmonic_scores(ballots, instance)
        assert scores["a"] == pytest.approx(1.5)
        assert scores["b"] == pytest.approx(1.5)

    def test_rejects_unknown_project_in_ballot(self):
        instance = _instance(("a", 1), ("b", 2), ("c", 3), budget=6)
        bad_ballot: ValueRankingBallot = ("a", "unknown")
        with pytest.raises(ValueError, match="unique valid project IDs"):
            HarmonicValueRankingRule().aggregate([bad_ballot], instance)

    def test_rejects_duplicate_in_ballot(self):
        instance = _instance(("a", 1), ("b", 2), ("c", 3), budget=6)
        bad_ballot: ValueRankingBallot = ("a", "a")
        with pytest.raises(ValueError, match="unique valid project IDs"):
            HarmonicValueRankingRule().aggregate([bad_ballot], instance)

    def test_partial_ranking_ballots_supported(self):
        instance = _instance(("a", 1), ("b", 2), ("c", 3), budget=6)
        ballot: ValueRankingBallot = ("b", "a")
        scores = HarmonicValueRankingRule().harmonic_scores([ballot], instance)
        assert scores["b"] == pytest.approx(1.0)
        assert scores["a"] == pytest.approx(0.5)
        assert scores["c"] == pytest.approx(0.0)

    def test_outcome_is_budget_feasible(self):
        instance = _instance(("a", 3), ("b", 4), ("c", 2), budget=5)
        ballots: list[ValueRankingBallot] = [("a", "b", "c")] * 5
        outcome = HarmonicValueRankingRule().aggregate(ballots, instance)
        assert bundle_cost(instance, outcome) <= instance.budget

    def test_skip_nonfitting_project_and_continue(self):
        # costs: a=4, b=4, c=2, budget=6
        # Both voters rank: a 1st, b 2nd, c 3rd.
        # scores: a=2.0 (eff=2/4=0.5), b=1.0 (eff=1/4=0.25), c=2/3 (eff=(2/3)/2=0.33)
        # ordering: a (0.5), c (0.33), b (0.25)
        # spent: take a (spent 4, rem 2), b costs 4 -> skips b! take c (costs 2, spent 6).
        # outcome: ('a', 'c')
        instance = _instance(("a", 4), ("b", 4), ("c", 2), budget=6)
        ballots: list[ValueRankingBallot] = [("a", "b", "c")] * 2
        outcome = HarmonicValueRankingRule().aggregate(ballots, instance)
        assert outcome == ("a", "c")
        assert bundle_cost(instance, outcome) <= instance.budget

    def test_tie_break_ascending_project_id(self):
        # Same cost, same scores -> tie broken by project id
        instance = _instance(("b", 2), ("a", 2), budget=2)
        ballots: list[ValueRankingBallot] = [("a", "b"), ("b", "a")]
        outcome = HarmonicValueRankingRule().aggregate(ballots, instance)
        assert outcome == ("a",)

    def test_empty_ballots_returns_empty(self):
        instance = _instance(("a", 2), ("b", 3), budget=5)
        assert HarmonicValueRankingRule().aggregate([], instance) == ()

    def test_zero_budget_returns_empty(self):
        instance = _instance(("a", 2), ("b", 3), budget=0)
        ballots: list[ValueRankingBallot] = [("a", "b"), ("b", "a")]
        assert HarmonicValueRankingRule().aggregate(ballots, instance) == ()

    @pytest.mark.parametrize("distribution", ["uniform", "normal", "pareto"])
    def test_sampled_distributions_pipeline(self, distribution: str):
        """Sample utilities from each codebase distribution, elicit rankings, and aggregate."""
        rng = np.random.default_rng(2024)
        election = generate_synthetic_election(
            project_count=12,
            voter_count=25,
            cost_min=1,
            cost_max=10,
            budget_fraction=0.33,
            rng=rng,
            utility_distribution=distribution,
            normalize_rows=True,
            noise_sigma=0.0,
        )

        fmt = ValueRankingFormat()
        rule = HarmonicValueRankingRule()

        # Elicit ballots from perceived utilities
        ballots = [
            fmt.elicit(election.perceived_utilities.values[i], election.instance)
            for i in range(election.instance.project_count if False else election.utilities.voter_count)
        ]
        assert len(ballots) == election.utilities.voter_count
        for b in ballots:
            assert len(b) == election.instance.project_count
            assert set(b) == set(election.instance.project_ids)

        # Aggregate into chosen outcome
        outcome = rule.aggregate(ballots, election.instance)

        # 1. Budget feasibility
        assert bundle_cost(election.instance, outcome) <= election.instance.budget
        assert all(pid in election.instance.project_ids for pid in outcome)

        # 2. Evaluate social welfare under true utilities
        opt_bundle = optimal_feasible_bundle(
            election.instance,
            election.utilities.values.sum(axis=0),
        )
        opt_welfare = social_welfare(election.utilities, opt_bundle)
        chosen_welfare = social_welfare(election.utilities, outcome)
        trial_distortion = distortion(opt_welfare, chosen_welfare)

        # Distortion sanity checks
        assert math.isfinite(trial_distortion)
        assert trial_distortion >= 1.0
        assert opt_welfare >= chosen_welfare

    @pytest.mark.parametrize("distribution", ["uniform", "normal", "pareto"])
    def test_simulation_integration_across_distributions(self, distribution: str):
        """Verify full run_simulation execution with noise across all distributions."""
        settings = SimulationSettings(
            trials=3,
            master_seed=42,
            project_count=8,
            voter_count=15,
            cost_min=1,
            cost_max=8,
            budget_fraction=0.30,
            utility_distribution=distribution,
            noise_sigma=0.5,
        )
        results = run_simulation(settings, ValueRankingFormat(), HarmonicValueRankingRule())
        assert len(results) == 3
        for r in results:
            assert r.benchmark_exact
            assert r.distortion >= 1.0
            assert r.optimal_welfare >= r.chosen_welfare
            assert r.utility_distribution == distribution
            assert r.noise_sigma == 0.5
