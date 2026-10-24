"""Greedy harmonic aggregation rule for value-for-money ranking ballots.

Aggregates value-for-money ranking ballots using harmonic positional scoring:
voter i's rank-k project receives weight 1/k, summed across all voters:
    S(j) = sum_{i=1}^n 1 / rank_{i}(j)

Because voters rank projects by perceived value-for-money u_i(j) / c_j, the
aggregate score S(j) already measures perceived utility per dollar. Therefore,
projects are directly ordered by their aggregate harmonic score S(j) descending,
breaking ties by ascending project ID. Projects are then greedily funded, skipping
non-fitting projects and continuing the scan until the budget is exhausted.

Reference: Benadè et al. (2021), "Preference Elicitation for Participatory Budgeting",
           Management Science 67(5):2813–2827.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pb_distortion.formats.vfm_ranking import VFMRankingBallot
from pb_distortion.models import Outcome, PBInstance


@dataclass(frozen=True, slots=True)
class HarmonicVFMRankingRule:
    """Aggregate value-for-money rankings by harmonic scoring, then greedily fund.

    A non-fitting project is skipped and later projects are still considered,
    matching the greedy scan behavior of GreedyApprovalRule.
    """

    name: str = "harmonic_vfm_ranking"

    def harmonic_scores(
        self,
        ballots: Sequence[VFMRankingBallot],
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
        ballots: Sequence[VFMRankingBallot],
        instance: PBInstance,
    ) -> Outcome:
        if not ballots or instance.budget <= 0:
            return ()

        scores = self.harmonic_scores(ballots, instance)
        costs = instance.cost_by_project()

        # Ballots are already ordered by value-per-cost; score directly indicates bang-per-buck.
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
