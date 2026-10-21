from pb_distortion.formats.knapsack import KnapsackBallot
from pb_distortion.models import PBInstance
from pb_distortion.rules.greedy_approval import GreedyApprovalRule
from pb_distortion.solvers import bundle_cost


def test_greedy_approval_counts_votes_skips_nonfitting_and_continues():
    instance = PBInstance(("a", "b", "c"), (4, 4, 2), 6)
    ballots = [
        KnapsackBallot({"a"}),
        KnapsackBallot({"a"}),
        KnapsackBallot({"b"}),
        KnapsackBallot({"b"}),
        KnapsackBallot({"c"}),
    ]
    rule = GreedyApprovalRule()
    outcome = rule.aggregate(ballots, instance)

    assert rule.approval_scores(ballots, instance) == {"a": 2, "b": 2, "c": 1}
    assert outcome == ("a", "c")
    assert bundle_cost(instance, outcome) <= instance.budget


def test_tie_break_is_ascending_project_id():
    instance = PBInstance(("b", "a"), (2, 2), 2)
    outcome = GreedyApprovalRule().aggregate(
        [KnapsackBallot({"b"}), KnapsackBallot({"a"})],
        instance,
    )
    assert outcome == ("a",)


def test_rule_rejects_infeasible_ballot():
    instance = PBInstance(("a", "b"), (2, 2), 2)
    try:
        GreedyApprovalRule().aggregate([KnapsackBallot({"a", "b"})], instance)
    except ValueError as error:
        assert "feasible" in str(error)
    else:
        raise AssertionError("Expected infeasible ballot to be rejected.")
