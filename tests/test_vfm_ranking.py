"""Tests for value-for-money ranking ballot format and aggregation rule."""

from __future__ import annotations

import math
import numpy as np
import pytest

from pb_distortion.formats.value_ranking import ValueRankingFormat
from pb_distortion.formats.vfm_ranking import VFMRankingBallot, VFMRankingFormat
from pb_distortion.generation import generate_synthetic_election
from pb_distortion.metrics import distortion, social_welfare
from pb_distortion.models import PBInstance
from pb_distortion.rules.vfm_ranking import HarmonicVFMRankingRule
from pb_distortion.simulation import SimulationSettings, run_simulation
from pb_distortion.solvers import bundle_cost, optimal_feasible_bundle


def _instance(*args: tuple[str, int], budget: int) -> PBInstance:
    ids, costs = zip(*args)
    return PBInstance(project_ids=ids, costs=costs, budget=budget)


class TestVFMRankingFormat:
    def test_ballot_sorted_by_vfm(self):
        # costs: a=4, b=1, c=2
        # utilities: a=0.8, b=0.3, c=0.5
        # VFM: a=0.2, b=0.3, c=0.25 -> b > c > a
        instance = _instance(("a", 4), ("b", 1), ("c", 2), budget=5)
        utilities = np.array([0.8, 0.3, 0.5])
        ballot = VFMRankingFormat().elicit(utilities, instance)
        assert ballot == ("b", "c", "a")

    def test_vfm_differs_from_value_ranking(self):
        instance = _instance(("a", 4), ("b", 1), budget=5)
        utilities = np.array([0.8, 0.5])
        v_ballot = ValueRankingFormat().elicit(utilities, instance)
        vfm_ballot = VFMRankingFormat().elicit(utilities, instance)
        assert v_ballot == ("a", "b")
        assert vfm_ballot == ("b", "a")

    def test_tie_broken_by_ascending_project_id(self):
        instance = _instance(("b", 2), ("a", 2), budget=4)
        utilities = np.array([0.5, 0.5])
        ballot = VFMRankingFormat().elicit(utilities, instance)
        assert ballot == ("a", "b")

    def test_rejects_wrong_length(self):
        instance = _instance(("a", 1), ("b", 2), budget=3)
        with pytest.raises(ValueError):
            VFMRankingFormat().elicit(np.array([0.5]), instance)

    def test_rejects_negative_utility(self):
        instance = _instance(("a", 1), ("b", 2), budget=3)
        with pytest.raises(ValueError):
            VFMRankingFormat().elicit(np.array([-0.2, 0.5]), instance)

    def test_zero_utilities_produces_empty_ballot(self):
        instance = _instance(("c", 1), ("a", 1), ("b", 1), budget=3)
        ballot = VFMRankingFormat().elicit(np.zeros(3), instance)
        assert ballot == ()

    def test_partial_positive_utilities_only_ranks_positive(self):
        instance = _instance(("a", 1), ("b", 1), ("c", 1), budget=3)
        utilities = np.array([0.5, 0.0, 0.2])
        ballot = VFMRankingFormat().elicit(utilities, instance)
        assert ballot == ("a", "c")


class TestHarmonicVFMRankingRule:
    def test_harmonic_scores(self):
        instance = _instance(("a", 1), ("b", 2), ("c", 3), budget=6)
        ballot: VFMRankingBallot = ("b", "a", "c")
        scores = HarmonicVFMRankingRule().harmonic_scores([ballot], instance)
        assert scores["b"] == pytest.approx(1.0)
        assert scores["a"] == pytest.approx(0.5)
        assert scores["c"] == pytest.approx(1.0 / 3)

    def test_rejects_unknown_project_in_ballot(self):
        instance = _instance(("a", 1), ("b", 2), ("c", 3), budget=6)
        bad_ballot: VFMRankingBallot = ("a", "unknown")
        with pytest.raises(ValueError, match="unique valid project IDs"):
            HarmonicVFMRankingRule().aggregate([bad_ballot], instance)

    def test_rejects_duplicate_in_ballot(self):
        instance = _instance(("a", 1), ("b", 2), ("c", 3), budget=6)
        bad_ballot: VFMRankingBallot = ("a", "a")
        with pytest.raises(ValueError, match="unique valid project IDs"):
            HarmonicVFMRankingRule().aggregate([bad_ballot], instance)

    def test_partial_ranking_ballots_supported(self):
        instance = _instance(("a", 1), ("b", 2), ("c", 3), budget=6)
        ballot: VFMRankingBallot = ("b", "a")
        scores = HarmonicVFMRankingRule().harmonic_scores([ballot], instance)
        assert scores["b"] == pytest.approx(1.0)
        assert scores["a"] == pytest.approx(0.5)
        assert scores["c"] == pytest.approx(0.0)

    def test_outcome_is_budget_feasible(self):
        instance = _instance(("a", 3), ("b", 4), ("c", 2), budget=5)
        ballots: list[VFMRankingBallot] = [("c", "a", "b")] * 5
        outcome = HarmonicVFMRankingRule().aggregate(ballots, instance)
        assert bundle_cost(instance, outcome) <= instance.budget

    def test_skip_nonfitting_project_and_continue(self):
        # costs: a=4, b=4, c=2, budget=6
        # Ballots: b 1st (score 2.0), a 2nd (score 1.0), c 3rd (score 0.66)
        # ordering: b, a, c
        # take b (spent 4), a costs 4 -> skips a! take c (costs 2, spent 6).
        # outcome: ('b', 'c')
        instance = _instance(("a", 4), ("b", 4), ("c", 2), budget=6)
        ballots: list[VFMRankingBallot] = [("b", "a", "c")] * 2
        outcome = HarmonicVFMRankingRule().aggregate(ballots, instance)
        assert outcome == ("b", "c")
        assert bundle_cost(instance, outcome) <= instance.budget

    def test_tie_break_ascending_project_id(self):
        instance = _instance(("b", 2), ("a", 2), budget=2)
        ballots: list[VFMRankingBallot] = [("a", "b"), ("b", "a")]
        outcome = HarmonicVFMRankingRule().aggregate(ballots, instance)
        assert outcome == ("a",)

    def test_empty_ballots_returns_empty(self):
        instance = _instance(("a", 2), ("b", 3), budget=5)
        assert HarmonicVFMRankingRule().aggregate([], instance) == ()

    def test_zero_budget_returns_empty(self):
        instance = _instance(("a", 2), ("b", 3), budget=0)
        ballots: list[VFMRankingBallot] = [("a", "b"), ("b", "a")]
        assert HarmonicVFMRankingRule().aggregate(ballots, instance) == ()

    @pytest.mark.parametrize("distribution", ["uniform", "normal", "pareto"])
    def test_sampled_distributions_pipeline(self, distribution: str):
        """Sample utilities from each codebase distribution, elicit VFM rankings, and aggregate."""
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

        fmt = VFMRankingFormat()
        rule = HarmonicVFMRankingRule()

        ballots = [
            fmt.elicit(election.perceived_utilities.values[i], election.instance)
            for i in range(election.utilities.voter_count)
        ]
        assert len(ballots) == election.utilities.voter_count
        for b in ballots:
            assert len(b) == election.instance.project_count
            assert set(b) == set(election.instance.project_ids)

        outcome = rule.aggregate(ballots, election.instance)

        assert bundle_cost(election.instance, outcome) <= election.instance.budget
        assert all(pid in election.instance.project_ids for pid in outcome)

        opt_bundle = optimal_feasible_bundle(
            election.instance,
            election.utilities.values.sum(axis=0),
        )
        opt_welfare = social_welfare(election.utilities, opt_bundle)
        chosen_welfare = social_welfare(election.utilities, outcome)
        trial_distortion = distortion(opt_welfare, chosen_welfare)

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
        results = run_simulation(settings, VFMRankingFormat(), HarmonicVFMRankingRule())
        assert len(results) == 3
        for r in results:
            assert r.benchmark_exact
            assert r.distortion >= 1.0
            assert r.optimal_welfare >= r.chosen_welfare
            assert r.utility_distribution == distribution
            assert r.noise_sigma == 0.5
