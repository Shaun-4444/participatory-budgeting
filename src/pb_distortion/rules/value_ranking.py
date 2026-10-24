"""Greedy harmonic aggregation rule for value-ranking ballots.

Aggregates value-ranking ballots using harmonic positional scoring:
voter i's rank-k project receives weight 1/k, summed across all voters:
    S(j) = sum_{i=1}^n 1 / rank_{i}(j)

Because voters rank projects by raw perceived utility u_i(j) without regard to cost,
the aggregate score S(j) estimates the total social utility of project j.
To select a feasible bundle under the budget constraint, projects are ordered by
their estimated cost-efficiency S(j) / c_j (or raw score S(j) if cost_weighted=False),
breaking ties by ascending project ID. Projects are then greedily funded, skipping
non-fitting projects and continuing the scan until the budget is exhausted.

Reference: Benadè et al. (2021), "Preference Elicitation for Participatory Budgeting",
           Management Science 67(5):2813–2827.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pb_distortion.formats.value_ranking import ValueRankingBallot
from pb_distortion.models import Outcome, PBInstance


@dataclass(frozen=True, slots=True)
class HarmonicValueRankingRule:
    """Aggregate value rankings by harmonic scoring, then greedily fund by efficiency.

    A non-fitting project is skipped and later projects are still considered,
    matching the greedy scan behavior of GreedyApprovalRule.
    """

    name: str = "harmonic_value_ranking"
    cost_weighted: bool = True

    def harmonic_scores(
        self,
        ballots: Sequence[ValueRankingBallot],
        instance: PBInstance,
    ) -> dict[str, float]:
        """Compute aggregate harmonic score for each project across all ballots."""
        known = set(instance.project_ids)
        scores = {pid: 0.0 for pid in instance.project_ids}
        for ballot in ballots:
            if set(ballot) != known or len(ballot) != len(known):
                raise ValueError("A ranking ballot must contain every project ID exactly once.")
            for rank, project_id in enumerate(ballot, start=1):
                scores[project_id] += 1.0 / rank
        return scores

    def aggregate(
        self,
        ballots: Sequence[ValueRankingBallot],
        instance: PBInstance,
    ) -> Outcome:
        if not ballots or instance.budget <= 0:
            return ()

        scores = self.harmonic_scores(ballots, instance)
        costs = instance.cost_by_project()

        if self.cost_weighted:
            # Score per unit cost (bang-per-buck greedy knapsack heuristic)
            ordering = sorted(
                instance.project_ids,
                key=lambda pid: (-scores[pid] / costs[pid], pid),
            )
        else:
            # Pure score greedy
            ordering = sorted(
                instance.project_ids,
                key=lambda pid: (-scores[pid], pid),
            )

        selected: list[str] = []
        spent = 0
        for pid in ordering:
            if spent + costs[pid] <= instance.budget:
                selected.append(pid)
                spent += costs[pid]

        return tuple(selected)
