"""Greedy approval aggregation for knapsack ballots."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from pb_distortion.formats.knapsack import KnapsackBallot
from pb_distortion.models import Outcome, PBInstance
from pb_distortion.solvers import bundle_cost


@dataclass(frozen=True, slots=True)
class GreedyApprovalRule:
    """Rank by ballot appearances, then greedily fund projects that fit.

    A non-fitting project is skipped and later projects are still considered.
    Ties in approval counts use ascending project ID. Projects with zero approvals
    remain in the ordering, as specified by the baseline rule.
    """

    name: str = "greedy_approval"

    def approval_scores(
        self,
        ballots: Sequence[KnapsackBallot],
        instance: PBInstance,
    ) -> dict[str, int]:
        known_projects = set(instance.project_ids)
        scores = Counter({project_id: 0 for project_id in instance.project_ids})
        for ballot in ballots:
            unknown = set(ballot) - known_projects
            if unknown:
                raise ValueError(f"Ballot contains unknown projects: {sorted(unknown)}")
            if bundle_cost(instance, tuple(ballot)) > instance.budget:
                raise ValueError("Every knapsack ballot must be feasible under the budget.")
            scores.update(ballot)
        return {project_id: scores[project_id] for project_id in instance.project_ids}

    def aggregate(
        self,
        ballots: Sequence[KnapsackBallot],
        instance: PBInstance,
    ) -> Outcome:
        scores = self.approval_scores(ballots, instance)
        ordering = sorted(instance.project_ids, key=lambda project_id: (-scores[project_id], project_id))
        costs = instance.cost_by_project()
        selected: list[str] = []
        spent = 0
        for project_id in ordering:
            if spent + costs[project_id] <= instance.budget:
                selected.append(project_id)
                spent += costs[project_id]
        return tuple(selected)
