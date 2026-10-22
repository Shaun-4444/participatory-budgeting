"""Tests for threshold approval ballot format and aggregation rule."""

from __future__ import annotations

import numpy as np
import pytest

from pb_distortion.formats.threshold_approval import ThresholdApprovalBallotFormat
from pb_distortion.models import PBInstance
from pb_distortion.rules.threshold_approval import ThresholdApprovalRule
from pb_distortion.simulation import SimulationSettings, run_simulation
from pb_distortion.solvers import bundle_cost


def test_threshold_approval_default_threshold_is_one_over_m():
    instance = PBInstance(
        project_ids=("p0", "p1", "p2", "p3"),
        costs=(2, 2, 2, 2),
        budget=4,
    )
    # m = 4, so default threshold = 1/4 = 0.25
    format_ = ThresholdApprovalBallotFormat()
    utilities = np.array([0.1, 0.25, 0.4, 0.25])
    ballot = format_.elicit(utilities, instance)

    # p1 (0.25), p2 (0.4), and p3 (0.25) should be approved; p0 (0.1) should not
    assert ballot == frozenset({"p1", "p2", "p3"})


def test_threshold_approval_custom_threshold():
    instance = PBInstance(
        project_ids=("p0", "p1", "p2"),
        costs=(1, 2, 3),
        budget=3,
    )
    format_ = ThresholdApprovalBallotFormat(threshold=0.5)
    utilities = np.array([0.6, 0.49, 0.5])
    ballot = format_.elicit(utilities, instance)

    assert ballot == frozenset({"p0", "p2"})


def test_threshold_approval_relative_threshold():
    instance = PBInstance(
        project_ids=("p0", "p1", "p2"),
        costs=(1, 2, 3),
        budget=3,
    )
    # Utilities sum to 10.0; relative threshold 0.3 means cutoff is 3.0
    format_ = ThresholdApprovalBallotFormat(threshold=0.3, relative=True)
    utilities = np.array([2.5, 3.0, 4.5])
    ballot = format_.elicit(utilities, instance)

    assert ballot == frozenset({"p1", "p2"})


def test_threshold_approval_validation():
    instance = PBInstance(
        project_ids=("p0", "p1"),
        costs=(1, 2),
        budget=2,
    )
    with pytest.raises(ValueError, match="finite non-negative"):
        ThresholdApprovalBallotFormat(threshold=-0.1)
    with pytest.raises(ValueError, match="finite non-negative"):
        ThresholdApprovalBallotFormat(threshold=float("nan"))

    format_ = ThresholdApprovalBallotFormat()
    with pytest.raises(ValueError, match="one value per project"):
        format_.elicit(np.array([0.5]), instance)
    with pytest.raises(ValueError, match="finite and non-negative"):
        format_.elicit(np.array([-0.1, 0.5]), instance)
    with pytest.raises(ValueError, match="finite and non-negative"):
        format_.elicit(np.array([float("inf"), 0.5]), instance)


def test_threshold_approval_ballot_can_exceed_budget():
    """Unlike knapsack ballots, threshold approvals are not restricted by the budget."""
    instance = PBInstance(
        project_ids=("p0", "p1", "p2"),
        costs=(5, 5, 5),
        budget=5,
    )
    format_ = ThresholdApprovalBallotFormat(threshold=0.2)
    utilities = np.array([0.3, 0.4, 0.3])
    ballot = format_.elicit(utilities, instance)

    assert ballot == frozenset({"p0", "p1", "p2"})
    # Ballot total cost is 15, which exceeds budget 5
    assert bundle_cost(instance, tuple(ballot)) == 15
    assert bundle_cost(instance, tuple(ballot)) > instance.budget

    # Aggregation rule must accept this without raising ValueError
    rule = ThresholdApprovalRule()
    outcome = rule.aggregate([ballot], instance)
    assert bundle_cost(instance, outcome) <= instance.budget


def test_threshold_approval_rule_ordering_and_tie_breaking():
    instance = PBInstance(
        project_ids=("p0", "p1", "p2", "p3"),
        costs=(2, 2, 2, 2),
        budget=4,
    )
    # p1: 3 approvals, p0: 2 approvals, p2: 2 approvals, p3: 0 approvals
    ballots = [
        frozenset({"p0", "p1"}),
        frozenset({"p1", "p2"}),
        frozenset({"p0", "p1", "p2"}),
    ]
    rule = ThresholdApprovalRule(mode="count")
    outcome = rule.aggregate(ballots, instance)

    # p1 has 3 approvals (cost 2) -> funded
    # p0 and p2 are tied at 2 approvals; tie broken by ID -> p0 (cost 2) -> funded
    # Budget of 4 is now full (p1, p0).
    assert set(outcome) == {"p0", "p1"}
    assert bundle_cost(instance, outcome) <= instance.budget


def test_threshold_approval_rule_skips_non_fitting_projects():
    instance = PBInstance(
        project_ids=("p0", "p1", "p2"),
        costs=(5, 10, 3),
        budget=8,
    )
    # Approvals: p1 has 3, p0 has 2, p2 has 1
    ballots = [
        frozenset({"p0", "p1"}),
        frozenset({"p1"}),
        frozenset({"p1", "p2"}),
    ]
    rule = ThresholdApprovalRule(mode="count")
    outcome = rule.aggregate(ballots, instance)

    # p1 has 3 approvals but cost is 10 > budget 8 -> skipped!
    # p0 has 2 approvals, cost 5 <= 8 -> funded (remaining budget 3)
    # p2 has 1 approval, cost 3 <= 3 -> funded (remaining budget 0)
    assert outcome == ("p0", "p2")
    assert bundle_cost(instance, outcome) == 8


def test_threshold_approval_rule_modes():
    instance = PBInstance(
        project_ids=("p0", "p1"),
        costs=(1, 5),
        budget=5,
    )
    # p0: 2 approvals (cost 1, ratio 2.0)
    # p1: 3 approvals (cost 5, ratio 0.6)
    ballots = [
        frozenset({"p0", "p1"}),
        frozenset({"p0", "p1"}),
        frozenset({"p1"}),
    ]

    # 'count' mode: p1 has 3 approvals, fits into budget 5 -> funds ('p1',)
    rule_count = ThresholdApprovalRule(mode="count")
    assert rule_count.aggregate(ballots, instance) == ("p1",)

    # 'cost_ratio' mode: p0 has ratio 2.0 vs p1 ratio 0.6 -> funds ('p0',)
    rule_ratio = ThresholdApprovalRule(mode="cost_ratio")
    assert rule_ratio.aggregate(ballots, instance) == ("p0",)

    # 'knapsack' mode: finds exact knapsack maximizing total approvals
    # subset {p1} gives 3 approvals (cost 5 <= 5)
    # subset {p0} gives 2 approvals (cost 1 <= 5)
    # subset {p0, p1} cost 6 > 5
    # knapsack picks {p1}
    rule_knap = ThresholdApprovalRule(mode="knapsack")
    assert rule_knap.aggregate(ballots, instance) == ("p1",)


def test_threshold_approval_rule_rejects_unknown_projects():
    instance = PBInstance(
        project_ids=("p0", "p1"),
        costs=(1, 1),
        budget=2,
    )
    rule = ThresholdApprovalRule()
    with pytest.raises(ValueError, match="unknown projects"):
        rule.aggregate([frozenset({"p0", "unknown_project"})], instance)


def test_threshold_approval_end_to_end_simulation():
    settings = SimulationSettings(
        trials=10,
        master_seed=123,
        project_count=6,
        voter_count=10,
        cost_min=1,
        cost_max=5,
        budget_fraction=0.4,
        utility_distribution="uniform",
        noise_sigma=0.2,
    )
    format_ = ThresholdApprovalBallotFormat()
    rule = ThresholdApprovalRule()

    results = run_simulation(settings, format_, rule)
    assert len(results) == 10
    for result in results:
        # Check distortion is always >= 1.0
        assert result.distortion >= 1.0
        assert result.chosen_welfare <= result.optimal_welfare + 1e-9
        assert result.budget == int(result.budget)
