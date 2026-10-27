"""Unit tests for strategic voting and manipulation degradation analysis."""

from __future__ import annotations

import numpy as np
import pytest

from pb_distortion.formats.knapsack import KnapsackBallotFormat
from pb_distortion.formats.threshold_approval import ThresholdApprovalBallotFormat
from pb_distortion.formats.value_ranking import ValueRankingFormat
from pb_distortion.formats.vfm_ranking import VFMRankingFormat
from pb_distortion.generation import generate_synthetic_election
from pb_distortion.models import PBInstance
from pb_distortion.rules.greedy_approval import GreedyApprovalRule
from pb_distortion.rules.threshold_approval import ThresholdApprovalRule
from pb_distortion.rules.value_ranking import HarmonicValueRankingRule
from pb_distortion.rules.vfm_ranking import HarmonicVFMRankingRule
from pb_distortion.simulation import SimulationSettings
from pb_distortion.strategic import (
    evaluate_strategic_election,
    generate_strategic_ballot,
    identify_favorite_project,
    run_strategic_simulation,
)


def test_identify_favorite_project():
    inst = PBInstance(project_ids=("p0", "p1", "p2"), costs=(1, 2, 3), budget=4)
    utils = np.array([0.2, 0.5, 0.3])
    assert identify_favorite_project(utils, inst) == "p1"

    # Tie breaking by lowest cost then pid
    utils_tie = np.array([0.4, 0.4, 0.2])
    # p0 has cost 1, p1 has cost 2 -> p0 wins tie
    assert identify_favorite_project(utils_tie, inst) == "p0"


def test_generate_strategic_ballot_knapsack():
    inst = PBInstance(project_ids=("p0", "p1", "p2"), costs=(1, 2, 3), budget=4)
    utils = np.array([0.2, 0.5, 0.3])
    fmt = KnapsackBallotFormat()

    # Honest knapsack for budget 4 would take p1 (cost 2) + p2 (cost 3) -> 5 (no), or p1 + p0 -> 3 (yes, util 0.7)
    honest = fmt.elicit(utils, inst)
    assert honest == frozenset({"p0", "p1"})

    # Strategic knapsack must be bullet vote {p1}
    strat = generate_strategic_ballot(fmt, utils, inst)
    assert strat == frozenset({"p1"})


def test_generate_strategic_ballot_threshold():
    inst = PBInstance(project_ids=("p0", "p1", "p2"), costs=(1, 2, 3), budget=4)
    utils = np.array([0.2, 0.5, 0.3])
    fmt = ThresholdApprovalBallotFormat(threshold=0.25)

    # Honest threshold would approve p1 (0.5) and p2 (0.3)
    honest = fmt.elicit(utils, inst)
    assert honest == frozenset({"p1", "p2"})

    # Strategic threshold must be bullet vote {p1}
    strat = generate_strategic_ballot(fmt, utils, inst)
    assert strat == frozenset({"p1"})


def test_generate_strategic_ballot_rankings():
    inst = PBInstance(project_ids=("p0", "p1", "p2"), costs=(1, 2, 3), budget=4)
    utils = np.array([0.2, 0.5, 0.3])

    # Value Ranking
    vr_fmt = ValueRankingFormat()
    vr_strat = generate_strategic_ballot(vr_fmt, utils, inst)
    assert vr_strat[0] == "p1"
    # Rivals buried: p0 (util 0.2) is less threatening than p2 (util 0.3), so p0 comes before p2
    assert vr_strat == ("p1", "p0", "p2")

    # VFM Ranking
    # p0: 0.2/1 = 0.2, p1: 0.5/2 = 0.25, p2: 0.3/3 = 0.1
    vfm_fmt = VFMRankingFormat()
    vfm_strat = generate_strategic_ballot(vfm_fmt, utils, inst)
    assert vfm_strat[0] == "p1"
    # Rivals buried by v/c: p2 (0.1) has lower v/c than p0 (0.2), so p2 comes before p0
    assert vfm_strat == ("p1", "p2", "p0")


def test_evaluate_strategic_election_all_formats():
    rng = np.random.default_rng(42)
    election = generate_synthetic_election(
        project_count=5,
        voter_count=10,
        cost_min=1,
        cost_max=5,
        budget_fraction=0.4,
        rng=rng,
    )

    format_rules = [
        (KnapsackBallotFormat(), GreedyApprovalRule()),
        (ThresholdApprovalBallotFormat(relative=True), ThresholdApprovalRule()),
        (ValueRankingFormat(), HarmonicValueRankingRule()),
        (VFMRankingFormat(), HarmonicVFMRankingRule()),
    ]

    for fmt, rule in format_rules:
        res = evaluate_strategic_election(election, fmt, rule, strategic_voter_index=0)
        assert res.honest_distortion >= 1.0
        assert res.strategic_distortion >= 1.0
        assert res.honest_welfare <= res.optimal_welfare + 1e-9
        assert res.strategic_welfare <= res.optimal_welfare + 1e-9
        assert isinstance(res.distortion_degradation, float)
        assert isinstance(res.manipulation_successful, bool)
        assert res.favorite_project_id in election.instance.project_ids


def test_run_strategic_simulation():
    settings = SimulationSettings(
        trials=15,
        master_seed=42,
        project_count=5,
        voter_count=10,
        cost_min=1,
        cost_max=5,
        budget_fraction=0.4,
        utility_distribution="uniform",
    )
    fmt = ThresholdApprovalBallotFormat(threshold=0.2)
    rule = ThresholdApprovalRule()

    results, summary = run_strategic_simulation(settings, fmt, rule, bootstrap_resamples=100)

    assert len(results) == 15
    assert summary.trial_count == 15
    assert summary.format_name == "threshold_approval"
    assert summary.mean_honest_distortion >= 1.0
    assert summary.mean_strategic_distortion >= 1.0
    assert 0.0 <= summary.manipulation_success_rate <= 1.0
    assert summary.distortion_ci_honest[0] <= summary.distortion_ci_honest[1]
    assert summary.distortion_ci_strategic[0] <= summary.distortion_ci_strategic[1]
