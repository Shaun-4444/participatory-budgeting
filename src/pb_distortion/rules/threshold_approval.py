"""Threshold approval aggregation rule: greedy cost-ratio selection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from pb_distortion.formats.threshold_approval import ThresholdApprovalBallot
from pb_distortion.models import Outcome, PBInstance


@dataclass(frozen=True, slots=True)
class ThresholdApprovalRule:
    """Aggregate threshold approval ballots using greedy approval-to-cost ratio.

    Projects are ordered descending by their approval-to-cost ratio (approvals / cost).
    Ties are broken first by descending approval count, then by ascending project ID.
    Projects are greedily funded under the budget constraint with skip logic.
    This corresponds to the canonical aggregation algorithm analyzed in Benadè et al. (Theorem 14).
    """

    name: str = "threshold_approval"

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
        costs = instance.cost_by_project()

        # Order by descending approval/cost ratio, then descending approvals, then ascending project ID
        ordering = sorted(
            instance.project_ids,
            key=lambda project_id: (
                -scores[project_id] / costs[project_id],
                -scores[project_id],
                project_id,
            ),
        )

        selected: list[str] = []
        spent = 0
        for project_id in ordering:
            if spent + costs[project_id] <= instance.budget:
                selected.append(project_id)
                spent += costs[project_id]
        return tuple(selected)
