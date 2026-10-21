"""Reproducible simulation and trial-level result export."""

from __future__ import annotations

import csv
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from pb_distortion.generation import generate_synthetic_election
from pb_distortion.metrics import distortion, social_welfare
from pb_distortion.models import Election, Outcome
from pb_distortion.protocols import AggregationRule, ElicitationFormat
from pb_distortion.solvers import optimal_feasible_bundle


BallotT = TypeVar("BallotT")


@dataclass(frozen=True, slots=True)
class SimulationSettings:
    trials: int
    master_seed: int
    project_count: int
    voter_count: int
    cost_min: int
    cost_max: int
    budget_fraction: float
    utility_minimum: float = 0.0
    utility_maximum: float = 1.0
    utility_distribution: str = "uniform"
    normal_mean: float = 0.5
    normal_std: float = 0.2
    pareto_shape: float = 1.5
    normalize_rows: bool = True
    noise_sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.trials <= 0:
            raise ValueError("trials must be positive.")
        if self.master_seed < 0:
            raise ValueError("master_seed must be non-negative.")
        if self.utility_distribution not in {
            "uniform",
            "normal",
            "pareto",
        }:
            raise ValueError("utility_distribution must be 'uniform', 'normal', or 'pareto'.")
        if (
            not math.isfinite(self.normal_mean)
            or not math.isfinite(self.normal_std)
            or self.normal_mean < 0
            or self.normal_std <= 0
        ):
            raise ValueError("Truncated-normal utilities require mean >= 0 and std > 0.")
        if not math.isfinite(self.pareto_shape) or self.pareto_shape <= 1:
            raise ValueError("pareto_shape must be greater than 1 for a finite-mean Pareto draw.")
        if not math.isfinite(self.noise_sigma) or self.noise_sigma < 0:
            raise ValueError("noise_sigma must be finite and nonnegative.")


@dataclass(frozen=True, slots=True)
class TrialResult:
    trial: int
    seed: int
    utility_distribution: str
    noise_sigma: float
    project_count: int
    voter_count: int
    budget: int
    total_project_cost: int
    optimal_welfare: float
    chosen_welfare: float
    distortion: float
    benchmark_exact: bool
    outcome_project_ids: tuple[str, ...]
    optimal_project_ids: tuple[str, ...]


def evaluate_election(
    election: Election,
    ballot_format: ElicitationFormat[BallotT],
    aggregation_rule: AggregationRule[BallotT],
) -> tuple[Outcome, Outcome, float, float, float]:
    """Run one format/rule pair and evaluate its outcome on latent utilities."""

    perceived_utilities = election.perceived_utilities
    if perceived_utilities is None:  # Election normally fills this with true utilities.
        perceived_utilities = election.utilities
    ballots = [
        ballot_format.elicit(perceived_utilities.values[voter_index], election.instance)
        for voter_index in range(election.utilities.voter_count)
    ]
    outcome = aggregation_rule.aggregate(ballots, election.instance)
    optimal_bundle = optimal_feasible_bundle(
        election.instance,
        election.utilities.values.sum(axis=0),
    )
    optimal_welfare = social_welfare(election.utilities, optimal_bundle)
    chosen_welfare = social_welfare(election.utilities, outcome)
    measured_distortion = distortion(optimal_welfare, chosen_welfare)
    return outcome, optimal_bundle, optimal_welfare, chosen_welfare, measured_distortion


def run_simulation(
    settings: SimulationSettings,
    ballot_format: ElicitationFormat[BallotT],
    aggregation_rule: AggregationRule[BallotT],
) -> list[TrialResult]:
    """Run independent elections with child seeds derived from a master seed."""

    seed_sequences = np.random.SeedSequence(settings.master_seed).spawn(settings.trials)
    results: list[TrialResult] = []
    for trial_index, child_sequence in enumerate(seed_sequences):
        trial_seed = int(child_sequence.generate_state(1, dtype=np.uint64)[0])
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
        outcome, optimum, optimum_welfare, chosen_welfare, measured_distortion = evaluate_election(
            election,
            ballot_format,
            aggregation_rule,
        )
        results.append(
            TrialResult(
                trial=trial_index,
                seed=trial_seed,
                utility_distribution=settings.utility_distribution,
                noise_sigma=settings.noise_sigma,
                project_count=settings.project_count,
                voter_count=settings.voter_count,
                budget=election.instance.budget,
                total_project_cost=election.instance.total_cost,
                optimal_welfare=optimum_welfare,
                chosen_welfare=chosen_welfare,
                distortion=measured_distortion,
                benchmark_exact=True,
                outcome_project_ids=outcome,
                optimal_project_ids=optimum,
            )
        )
    return results


def write_simulation_outputs(
    results: list[TrialResult],
    settings: SimulationSettings,
    *,
    ballot_format_name: str,
    aggregation_rule_name: str,
    output_directory: str | Path,
    source_config: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write CSV trial metrics and JSON settings/seeds for reproducibility."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "trials.csv"
    json_path = output_path / "run.json"

    fields = list(TrialResult.__dataclass_fields__)
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["outcome_project_ids"] = ";".join(result.outcome_project_ids)
            row["optimal_project_ids"] = ";".join(result.optimal_project_ids)
            writer.writerow(row)

    run_metadata = {
        "settings": asdict(settings),
        "ballot_format": ballot_format_name,
        "aggregation_rule": aggregation_rule_name,
        "master_seed": settings.master_seed,
        "trial_seeds": [result.seed for result in results],
        "trial_count": len(results),
        "benchmark": "exact integer-cost dynamic programming",
        "metric": "distortion (optimal welfare / chosen welfare; sampled instance)",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "source_config": source_config or {},
    }
    json_path.write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")
    return csv_path, json_path
