"""Strategic voting and manipulation analysis for participatory budgeting elicitation formats.

Connects preference elicitation in participatory budgeting to the Gibbard-Satterthwaite theorem.
A strategic voter who knows the aggregation rule misreports their ballot to promote their single
favorite project (j* = argmax v_i(j)). We evaluate and compare how much each format degrades
in utilitarian social welfare and distortion.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from pb_distortion.formats.knapsack import KnapsackBallot, KnapsackBallotFormat
from pb_distortion.formats.threshold_approval import (
    ThresholdApprovalBallot,
    ThresholdApprovalBallotFormat,
)
from pb_distortion.formats.value_ranking import ValueRankingBallot, ValueRankingFormat
from pb_distortion.formats.vfm_ranking import VFMRankingBallot, VFMRankingFormat
from pb_distortion.generation import generate_synthetic_election
from pb_distortion.metrics import distortion, social_welfare
from pb_distortion.models import Election, Outcome, PBInstance, UtilityProfile
from pb_distortion.protocols import AggregationRule, ElicitationFormat
from pb_distortion.simulation import SimulationSettings
from pb_distortion.solvers import bundle_cost, optimal_feasible_bundle


BallotT = TypeVar("BallotT")


def identify_favorite_project(
    utilities: NDArray[np.float64],
    instance: PBInstance,
) -> str:
    """Identify the voter's favorite project with deterministic tie-breaking.

    Picks the project with the maximum utility value. In case of ties, chooses
    the one with the lowest cost, then the lowest project ID.
    """
    values = np.asarray(utilities, dtype=np.float64)
    costs = instance.costs
    project_ids = instance.project_ids

    # Order by (-utility, cost, project_id)
    best_index = min(
        range(len(project_ids)),
        key=lambda idx: (-values[idx], costs[idx], project_ids[idx]),
    )
    return project_ids[best_index]


def generate_strategic_ballot(
    ballot_format: ElicitationFormat[BallotT],
    utilities: NDArray[np.float64],
    instance: PBInstance,
    favorite_id: str | None = None,
) -> BallotT:
    """Generate a strategic ballot tailored to the specific format to promote favorite_id.

    - Knapsack: Bullet vote for {j*} (if feasible; otherwise honest).
    - Threshold Approval: Bullet vote for {j*}.
    - Value Ranking: Place j* at rank 1, and bury rivals (lowest utility projects next).
    - VFM Ranking: Place j* at rank 1 (overriding v/c), and bury rivals.
    """
    if favorite_id is None:
        favorite_id = identify_favorite_project(utilities, instance)

    project_ids = instance.project_ids
    costs = instance.cost_by_project()
    values = np.asarray(utilities, dtype=np.float64)
    val_by_pid = dict(zip(project_ids, values, strict=True))

    format_name = ballot_format.name

    # 1. Knapsack Format
    if format_name == "knapsack" or isinstance(ballot_format, KnapsackBallotFormat):
        fav_cost = costs[favorite_id]
        if fav_cost <= instance.budget:
            return frozenset({favorite_id})  # type: ignore[return-value]
        # If favorite project alone exceeds the budget, fallback to honest ballot
        return ballot_format.elicit(utilities, instance)

    # 2. Threshold Approval Format
    if format_name == "threshold_approval" or isinstance(ballot_format, ThresholdApprovalBallotFormat):
        return frozenset({favorite_id})  # type: ignore[return-value]

    # 3. Value Ranking Format
    if format_name == "value_ranking" or isinstance(ballot_format, ValueRankingFormat):
        # Bury rivals: remaining projects sorted by ascending utility (least threatening first)
        other_projects = [pid for pid in project_ids if pid != favorite_id]
        buried_order = sorted(
            other_projects,
            key=lambda pid: (val_by_pid[pid], pid),
        )
        return (favorite_id, *buried_order)  # type: ignore[return-value]

    # 4. Value-For-Money Ranking Format
    if format_name == "vfm_ranking" or isinstance(ballot_format, VFMRankingFormat):
        # Place j* at rank 1 regardless of its cost, bury rivals by ascending v/c
        other_projects = [pid for pid in project_ids if pid != favorite_id]
        buried_order = sorted(
            other_projects,
            key=lambda pid: (val_by_pid[pid] / costs[pid], pid),
        )
        return (favorite_id, *buried_order)  # type: ignore[return-value]

    # Default fallback if unknown format
    return ballot_format.elicit(utilities, instance)


@dataclass(frozen=True, slots=True)
class StrategicTrialResult:
    """Trial-level result comparing honest vs strategic voting."""

    trial: int
    seed: int
    format_name: str
    rule_name: str
    project_count: int
    voter_count: int
    budget: int
    optimal_welfare: float
    honest_welfare: float
    strategic_welfare: float
    honest_distortion: float
    strategic_distortion: float
    distortion_degradation: float  # strategic_distortion - honest_distortion
    welfare_loss_percentage: float  # (honest_welfare - strategic_welfare) / honest_welfare * 100
    strategic_voter_honest_utility: float
    strategic_voter_strategic_utility: float
    strategic_utility_gain: float  # strategic_utility - honest_utility
    favorite_project_id: str
    favorite_funded_honest: bool
    favorite_funded_strategic: bool
    favorite_promoted: bool  # False in honest -> True in strategic
    manipulation_successful: bool  # strategic_utility > honest_utility + 1e-9
    honest_outcome: tuple[str, ...]
    strategic_outcome: tuple[str, ...]
    optimal_outcome: tuple[str, ...]


def evaluate_strategic_election(
    election: Election,
    ballot_format: ElicitationFormat[BallotT],
    aggregation_rule: AggregationRule[BallotT],
    *,
    strategic_voter_index: int = 0,
) -> StrategicTrialResult:
    """Run an election comparing truthful voting vs strategic manipulation by one voter."""
    perceived_utilities = election.perceived_utilities
    if perceived_utilities is None:
        perceived_utilities = election.utilities

    n_voters = election.utilities.voter_count
    instance = election.instance
    true_utilities = election.utilities

    # 1. Honest Ballots and Outcome
    honest_ballots: list[BallotT] = [
        ballot_format.elicit(perceived_utilities.values[i], instance)
        for i in range(n_voters)
    ]
    honest_outcome = aggregation_rule.aggregate(honest_ballots, instance)

    # 2. Strategic Voter Behavior
    strat_perceived = perceived_utilities.values[strategic_voter_index]
    strat_true = true_utilities.values[strategic_voter_index]
    val_by_pid = dict(zip(instance.project_ids, strat_true, strict=True))

    favorite_id = identify_favorite_project(strat_perceived, instance)
    strategic_ballot = generate_strategic_ballot(
        ballot_format,
        strat_perceived,
        instance,
        favorite_id=favorite_id,
    )

    strategic_ballots = list(honest_ballots)
    strategic_ballots[strategic_voter_index] = strategic_ballot
    strategic_outcome = aggregation_rule.aggregate(strategic_ballots, instance)

    # 3. Welfare and Distortion Benchmarks
    social_values = true_utilities.values.sum(axis=0)
    optimal_bundle = optimal_feasible_bundle(instance, social_values)
    opt_welfare = social_welfare(true_utilities, optimal_bundle)

    honest_welfare = social_welfare(true_utilities, honest_outcome)
    strategic_welfare = social_welfare(true_utilities, strategic_outcome)

    honest_dist = distortion(opt_welfare, honest_welfare)
    strategic_dist = distortion(opt_welfare, strategic_welfare)
    distortion_deg = strategic_dist - honest_dist

    welfare_loss_pct = (
        ((honest_welfare - strategic_welfare) / honest_welfare * 100.0)
        if honest_welfare > 0
        else 0.0
    )

    # 4. Strategic Voter Utility Metrics
    voter_honest_u = sum(val_by_pid[pid] for pid in honest_outcome)
    voter_strat_u = sum(val_by_pid[pid] for pid in strategic_outcome)
    utility_gain = voter_strat_u - voter_honest_u

    fav_in_honest = bool(favorite_id in honest_outcome)
    fav_in_strat = bool(favorite_id in strategic_outcome)
    fav_promoted = bool((not fav_in_honest) and fav_in_strat)
    manip_success = bool(utility_gain > 1e-9)

    return StrategicTrialResult(
        trial=0,
        seed=0,
        format_name=ballot_format.name,
        rule_name=aggregation_rule.name,
        project_count=instance.project_count,
        voter_count=n_voters,
        budget=instance.budget,
        optimal_welfare=opt_welfare,
        honest_welfare=honest_welfare,
        strategic_welfare=strategic_welfare,
        honest_distortion=honest_dist,
        strategic_distortion=strategic_dist,
        distortion_degradation=distortion_deg,
        welfare_loss_percentage=welfare_loss_pct,
        strategic_voter_honest_utility=voter_honest_u,
        strategic_voter_strategic_utility=voter_strat_u,
        strategic_utility_gain=utility_gain,
        favorite_project_id=favorite_id,
        favorite_funded_honest=fav_in_honest,
        favorite_funded_strategic=fav_in_strat,
        favorite_promoted=fav_promoted,
        manipulation_successful=manip_success,
        honest_outcome=honest_outcome,
        strategic_outcome=strategic_outcome,
        optimal_outcome=optimal_bundle,
    )


@dataclass(frozen=True, slots=True)
class StrategicFormatSummary:
    """Aggregated summary of strategic voting impact on a ballot format."""

    format_name: str
    rule_name: str
    trial_count: int
    voter_count: int
    project_count: int
    mean_honest_distortion: float
    mean_strategic_distortion: float
    mean_distortion_degradation: float
    mean_welfare_loss_percentage: float
    manipulation_success_rate: float
    favorite_promoted_rate: float
    mean_strategic_utility_gain: float
    distortion_ci_honest: tuple[float, float]
    distortion_ci_strategic: tuple[float, float]


def run_strategic_simulation(
    settings: SimulationSettings,
    ballot_format: ElicitationFormat[BallotT],
    aggregation_rule: AggregationRule[BallotT],
    *,
    strategic_voter_index: int = 0,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 2000,
) -> tuple[list[StrategicTrialResult], StrategicFormatSummary]:
    """Run repeated synthetic trials measuring strategic voting degradation."""
    seed_sequences = np.random.SeedSequence(settings.master_seed).spawn(settings.trials)
    results: list[StrategicTrialResult] = []

    for trial_idx, child_seq in enumerate(seed_sequences):
        trial_seed = int(child_seq.generate_state(1, dtype=np.uint64)[0])
        rng = np.random.default_rng(trial_seed)
        election = generate_synthetic_election(
            project_count=settings.project_count,
            voter_count=settings.voter_count,
            cost_min=settings.cost_min,
            cost_max=settings.cost_max,
            budget_fraction=settings.budget_fraction,
            rng=rng,
            utility_distribution=settings.utility_distribution,
            utility_minimum=settings.utility_minimum,
            utility_maximum=settings.utility_maximum,
            normal_mean=settings.normal_mean,
            normal_std=settings.normal_std,
            pareto_shape=settings.pareto_shape,
            normalize_rows=settings.normalize_rows,
            noise_sigma=settings.noise_sigma,
        )

        res = evaluate_strategic_election(
            election,
            ballot_format,
            aggregation_rule,
            strategic_voter_index=strategic_voter_index,
        )

        # Inject trial and seed
        trial_res = StrategicTrialResult(
            trial=trial_idx,
            seed=trial_seed,
            format_name=res.format_name,
            rule_name=res.rule_name,
            project_count=res.project_count,
            voter_count=res.voter_count,
            budget=res.budget,
            optimal_welfare=res.optimal_welfare,
            honest_welfare=res.honest_welfare,
            strategic_welfare=res.strategic_welfare,
            honest_distortion=res.honest_distortion,
            strategic_distortion=res.strategic_distortion,
            distortion_degradation=res.distortion_degradation,
            welfare_loss_percentage=res.welfare_loss_percentage,
            strategic_voter_honest_utility=res.strategic_voter_honest_utility,
            strategic_voter_strategic_utility=res.strategic_voter_strategic_utility,
            strategic_utility_gain=res.strategic_utility_gain,
            favorite_project_id=res.favorite_project_id,
            favorite_funded_honest=res.favorite_funded_honest,
            favorite_funded_strategic=res.favorite_funded_strategic,
            favorite_promoted=res.favorite_promoted,
            manipulation_successful=res.manipulation_successful,
            honest_outcome=res.honest_outcome,
            strategic_outcome=res.strategic_outcome,
            optimal_outcome=res.optimal_outcome,
        )
        results.append(trial_res)

    # Compute aggregate statistics
    h_dists = np.array([r.honest_distortion for r in results])
    s_dists = np.array([r.strategic_distortion for r in results])
    deg_dists = np.array([r.distortion_degradation for r in results])
    wl_pcts = np.array([r.welfare_loss_percentage for r in results])
    gains = np.array([r.strategic_utility_gain for r in results])

    success_rate = float(np.mean([1.0 if r.manipulation_successful else 0.0 for r in results]))
    promoted_rate = float(np.mean([1.0 if r.favorite_promoted else 0.0 for r in results]))

    # Bootstrap CI for distortions
    rng_boot = np.random.default_rng(np.random.SeedSequence([settings.master_seed, 8888]))
    idx_matrix = rng_boot.integers(0, len(results), size=(bootstrap_resamples, len(results)))
    alpha = (1.0 - confidence_level) / 2.0

    boot_h = h_dists[idx_matrix].mean(axis=1)
    ci_h = (float(np.quantile(boot_h, alpha)), float(np.quantile(boot_h, 1.0 - alpha)))

    boot_s = s_dists[idx_matrix].mean(axis=1)
    ci_s = (float(np.quantile(boot_s, alpha)), float(np.quantile(boot_s, 1.0 - alpha)))

    summary = StrategicFormatSummary(
        format_name=ballot_format.name,
        rule_name=aggregation_rule.name,
        trial_count=len(results),
        voter_count=settings.voter_count,
        project_count=settings.project_count,
        mean_honest_distortion=float(np.mean(h_dists)),
        mean_strategic_distortion=float(np.mean(s_dists)),
        mean_distortion_degradation=float(np.mean(deg_dists)),
        mean_welfare_loss_percentage=float(np.mean(wl_pcts)),
        manipulation_success_rate=success_rate,
        favorite_promoted_rate=promoted_rate,
        mean_strategic_utility_gain=float(np.mean(gains)),
        distortion_ci_honest=ci_h,
        distortion_ci_strategic=ci_s,
    )

    return results, summary
