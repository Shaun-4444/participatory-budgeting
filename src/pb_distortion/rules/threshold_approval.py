"""Threshold approval aggregation rules."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from pb_distortion.formats.threshold_approval import ThresholdApprovalBallot
from pb_distortion.models import Outcome, PBInstance
from pb_distortion.solvers import solve_knapsack


@dataclass(frozen=True, slots=True)
class ThresholdApprovalRule:
    """Aggregate threshold approval ballots into a feasible project bundle.

    Modes:
    - 'count': Greedily select projects ordered by descending approval count,
      skipping projects that exceed remaining budget. Ties broken by project ID ascending.
    - 'cost_ratio': Greedily select projects ordered by descending approval/cost ratio,
      then approval count, then project ID ascending.
    - 'knapsack': Exact dynamic-programming knapsack maximizing total approvals
      subject to the budget constraint.
    """

    name: str = "threshold_approval"
    mode: str = "count"

    def __post_init__(self) -> None:
        if self.mode not in {"count", "cost_ratio", "knapsack"}:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be 'count', 'cost_ratio', or 'knapsack'."
            )

    def approval_scores(
        self,
        ballots: Sequence[ThresholdApprovalBallot],
        instance: PBInstance,
    ) -> dict[str, int]:
        known_projects = set(instance.project_ids)
        scores = Counter({project_id: 0 for project_id in instance.project_ids})
        for ballot in ballots:
            unknown = set(ballot) - known_projects
            if unknown:
                raise ValueError(f"Ballot contains unknown projects: {sorted(unknown)}")
            # Note: unlike knapsack ballots, threshold approval votes are not
            # constrained by individual budget feasibility.
            scores.update(ballot)
        return {project_id: scores[project_id] for project_id in instance.project_ids}

    def aggregate(
        self,
        ballots: Sequence[ThresholdApprovalBallot],
        instance: PBInstance,
    ) -> Outcome:
        scores = self.approval_scores(ballots, instance)

        if self.mode == "knapsack":
            values = [float(scores[project_id]) for project_id in instance.project_ids]
            return solve_knapsack(
                instance.project_ids,
                instance.costs,
                values,
                instance.budget,
            )

        costs = instance.cost_by_project()
        if self.mode == "count":
            ordering = sorted(
                instance.project_ids,
                key=lambda project_id: (-scores[project_id], project_id),
            )
        elif self.mode == "cost_ratio":
            ordering = sorted(
                instance.project_ids,
                key=lambda project_id: (
                    -scores[project_id] / costs[project_id],
                    -scores[project_id],
                    project_id,
                ),
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        selected: list[str] = []
        spent = 0
        for project_id in ordering:
            if spent + costs[project_id] <= instance.budget:
                selected.append(project_id)
                spent += costs[project_id]
        return tuple(selected)
