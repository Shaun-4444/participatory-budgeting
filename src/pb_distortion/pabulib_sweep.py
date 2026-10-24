"""Model-based project-count experiments on observed Pabulib knapsack ballots."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from pb_distortion.metrics import distortion, social_welfare
from pb_distortion.models import UtilityProfile
from pb_distortion.pabulib_data import PabulibElection, make_scaled_subinstance
from pb_distortion.protocols import AggregationRule, ElicitationFormat
from pb_distortion.solvers import bundle_cost, optimal_feasible_bundle
from pb_distortion.utility_inference import sample_knapsack_consistent_utilities


BallotT = TypeVar("BallotT")


@dataclass(frozen=True, slots=True)
class PabulibSweepSettings:
    """Reproducible settings for one real-data project-count sweep."""

    project_counts: tuple[int, ...] = (5, 10, 20, 30, 40, 50, 75, 85)
    subprofile_count: int = 20
    voters_per_subprofile: int = 5
    utility_profiles_per_subprofile: int = 5
    utility_witness_mix: float = 0.75
    master_seed: int = 42
    bootstrap_resamples: int = 2000
    confidence_level: float = 0.95
    max_utility_cuts: int = 500

    def __post_init__(self) -> None:
        project_counts = tuple(self.project_counts)
        object.__setattr__(self, "project_counts", project_counts)
        if not project_counts or any(count <= 0 for count in project_counts):
            raise ValueError("project_counts must contain positive integers.")
        if tuple(sorted(set(project_counts))) != project_counts:
            raise ValueError("project_counts must be strictly increasing.")
        if self.subprofile_count < 2:
            raise ValueError("At least two subprofiles are required for bootstrap intervals.")
        if self.voters_per_subprofile <= 0:
            raise ValueError("voters_per_subprofile must be positive.")
        if self.utility_profiles_per_subprofile <= 0:
            raise ValueError("utility_profiles_per_subprofile must be positive.")
        if not math.isfinite(self.utility_witness_mix) or not 0 <= self.utility_witness_mix <= 1:
            raise ValueError("utility_witness_mix must be finite and between zero and one.")
        if self.master_seed < 0:
            raise ValueError("master_seed must be non-negative.")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive.")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one.")
        if self.max_utility_cuts <= 0:
            raise ValueError("max_utility_cuts must be positive.")


@dataclass(frozen=True, slots=True)
class PabulibTrialResult:
    subprofile: int
    utility_draw: int
    subprofile_seed: int
    utility_seed: int
    project_count: int
    voter_count: int
    format_name: str
    aggregation_rule_name: str
    budget_source_units: int
    total_cost_source_units: int
    cost_divisor_source_units: int
    optimal_welfare: float
    chosen_welfare: float
    distortion: float
    benchmark_exact: bool
    lp_solves: int
    separation_cuts: int
    project_ids: tuple[str, ...]
    outcome_project_ids: tuple[str, ...]
    optimal_project_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PabulibProjectCountSummary:
    project_count: int
    format_name: str
    rule_name: str
    profile_count: int
    subprofile_count: int
    mean_distortion: float
    ci_lower: float
    ci_upper: float


def _format_label(
    ballot_format: ElicitationFormat[Any],
    aggregation_rule: AggregationRule[Any],
) -> str:
    return f"{ballot_format.name} + {aggregation_rule.name}"


def _bootstrap_cluster_mean(
    cluster_values: Sequence[float],
    *,
    rng: np.random.Generator,
    resamples: int,
    confidence_level: float,
) -> tuple[float, float, float]:
    values = np.asarray(cluster_values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Cluster means must contain at least two finite values.")
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be finite.")
    bootstrap_indices = rng.integers(0, values.size, size=(resamples, values.size))
    bootstrap_means = values[bootstrap_indices].mean(axis=1)
    alpha = (1 - confidence_level) / 2
    lower, upper = np.quantile(bootstrap_means, [alpha, 1 - alpha])
    return float(values.mean()), float(lower), float(upper)


def run_pabulib_project_count_sweep(
    election: PabulibElection,
    settings: PabulibSweepSettings,
    format_rules: Sequence[tuple[ElicitationFormat[Any], AggregationRule[Any]]],
) -> tuple[list[PabulibTrialResult], list[PabulibProjectCountSummary], dict[str, int]]:
    """Run a paired, model-based project-count sweep from a real knapsack profile.

    Each subprofile samples voters from the real election and draws normalized
    additive utility vectors consistent with their recorded full-instance
    knapsack ballots. The same sampled utilities and nested random project
    subsets are used at every project count. Reduced instances preserve the
    original budget-to-total-project-cost ratio. Ballots for each reduced
    instance are re-elicited from the sampled utilities, as in the paper's
    approach of inducing input profiles from utility profiles.
    """

    if not format_rules:
        raise ValueError("At least one format and aggregation rule pair is required.")
    if settings.voters_per_subprofile > election.voter_count:
        raise ValueError("voters_per_subprofile exceeds the number of Pabulib voters.")
    if max(settings.project_counts) > election.project_count:
        raise ValueError("A requested project count exceeds the source election.")
    labels = [_format_label(fmt, rule) for fmt, rule in format_rules]
    if len(labels) != len(set(labels)):
        raise ValueError("Format/rule labels must be unique.")

    observed_ballot_by_voter = dict(zip(election.voter_ids, election.ballots, strict=True))
    project_index = {project_id: index for index, project_id in enumerate(election.instance.project_ids)}
    root_sequence = np.random.SeedSequence(settings.master_seed)
    subprofile_sequences = root_sequence.spawn(settings.subprofile_count)
    results: list[PabulibTrialResult] = []
    total_lp_solves = 0
    total_separation_cuts = 0

    for subprofile_id, subprofile_sequence in enumerate(subprofile_sequences):
        subprofile_seed = int(subprofile_sequence.generate_state(1, dtype=np.uint64)[0])
        voter_sequence, project_sequence, *utility_sequences = subprofile_sequence.spawn(
            2 + settings.utility_profiles_per_subprofile
        )
        voter_rng = np.random.default_rng(voter_sequence)
        project_rng = np.random.default_rng(project_sequence)
        voter_indices = voter_rng.choice(
            election.voter_count,
            size=settings.voters_per_subprofile,
            replace=False,
        )
        project_order = tuple(
            int(index) for index in project_rng.permutation(election.project_count)
        )
        sampled_voter_ids = tuple(election.voter_ids[int(index)] for index in voter_indices)
        source_ballots = tuple(observed_ballot_by_voter[voter_id] for voter_id in sampled_voter_ids)

        for utility_draw_id, utility_sequence in enumerate(utility_sequences):
            utility_seed = int(utility_sequence.generate_state(1, dtype=np.uint64)[0])
            utility_rng = np.random.default_rng(utility_sequence)
            full_utilities: list[np.ndarray] = []
            lp_solves = 0
            separation_cuts = 0
            for ballot in source_ballots:
                utilities, diagnostics = sample_knapsack_consistent_utilities(
                    election.instance,
                    ballot,
                    utility_rng,
                    witness_mix=settings.utility_witness_mix,
                    max_cuts=settings.max_utility_cuts,
                )
                full_utilities.append(utilities)
                lp_solves += diagnostics.lp_solves
                separation_cuts += diagnostics.separation_cuts
            utility_matrix = np.vstack(full_utilities)
            total_lp_solves += lp_solves
            total_separation_cuts += separation_cuts

            for project_count in settings.project_counts:
                selected_indices = project_order[:project_count]
                subinstance, divisor, source_budget = make_scaled_subinstance(
                    election,
                    selected_indices,
                )
                subproject_ids = subinstance.project_ids
                utility_columns = [project_index[project_id] for project_id in subproject_ids]
                subprofile = UtilityProfile(
                    utility_matrix[:, utility_columns],
                    subproject_ids,
                )
                social_values = subprofile.values.sum(axis=0)
                optimum = optimal_feasible_bundle(subinstance, social_values)
                optimum_welfare = social_welfare(subprofile, optimum)
                total_cost_source_units = sum(
                    election.source_costs[index] for index in selected_indices
                )

                for (ballot_format, aggregation_rule), label in zip(
                    format_rules, labels, strict=True
                ):
                    ballots = [
                        ballot_format.elicit(subprofile.values[voter], subinstance)
                        for voter in range(subprofile.voter_count)
                    ]
                    outcome = aggregation_rule.aggregate(ballots, subinstance)
                    if bundle_cost(subinstance, outcome) > subinstance.budget:
                        raise RuntimeError(
                            f"{label} returned an infeasible outcome for {project_count} projects."
                        )
                    chosen_welfare = social_welfare(subprofile, outcome)
                    measured_distortion = distortion(optimum_welfare, chosen_welfare)
                    if not math.isfinite(measured_distortion):
                        raise ValueError(
                            "The selected outcome has zero welfare under a sampled utility profile; "
                            "the distortion is infinite."
                        )
                    results.append(
                        PabulibTrialResult(
                            subprofile=subprofile_id,
                            utility_draw=utility_draw_id,
                            subprofile_seed=subprofile_seed,
                            utility_seed=utility_seed,
                            project_count=project_count,
                            voter_count=settings.voters_per_subprofile,
                            format_name=ballot_format.name,
                            aggregation_rule_name=aggregation_rule.name,
                            budget_source_units=source_budget,
                            total_cost_source_units=total_cost_source_units,
                            cost_divisor_source_units=divisor,
                            optimal_welfare=optimum_welfare,
                            chosen_welfare=chosen_welfare,
                            distortion=measured_distortion,
                            benchmark_exact=True,
                            lp_solves=lp_solves,
                            separation_cuts=separation_cuts,
                            project_ids=subproject_ids,
                            outcome_project_ids=outcome,
                            optimal_project_ids=optimum,
                        )
                    )

    values_by_curve_cluster: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for result in results:
        curve = f"{result.format_name} + {result.aggregation_rule_name}"
        values_by_curve_cluster[(curve, result.project_count, result.subprofile)].append(
            result.distortion
        )

    bootstrap_rng = np.random.default_rng(
        np.random.SeedSequence([settings.master_seed, 0x504255])
    )
    summaries: list[PabulibProjectCountSummary] = []
    for label in labels:
        for project_count in settings.project_counts:
            cluster_means = [
                float(np.mean(values_by_curve_cluster[(label, project_count, cluster)]))
                for cluster in range(settings.subprofile_count)
            ]
            mean, lower, upper = _bootstrap_cluster_mean(
                cluster_means,
                rng=bootstrap_rng,
                resamples=settings.bootstrap_resamples,
                confidence_level=settings.confidence_level,
            )
            summaries.append(
                PabulibProjectCountSummary(
                    project_count=project_count,
                    format_name=label,
                    rule_name=next(rule.name for fmt, rule in format_rules if _format_label(fmt, rule) == label),
                    profile_count=settings.subprofile_count * settings.utility_profiles_per_subprofile,
                    subprofile_count=settings.subprofile_count,
                    mean_distortion=mean,
                    ci_lower=lower,
                    ci_upper=upper,
                )
            )

    diagnostics = {
        "utility_lp_solves": total_lp_solves,
        "utility_separation_cuts": total_separation_cuts,
        "sampled_voter_utility_vectors": (
            settings.subprofile_count
            * settings.utility_profiles_per_subprofile
            * settings.voters_per_subprofile
        ),
    }
    return results, summaries, diagnostics


def allocate_pabulib_output_directory(
    base_directory: str | Path,
    dataset_name: str,
    master_seed: int,
) -> Path:
    """Create a unique output folder without overwriting earlier runs."""

    base_path = Path(base_directory)
    base_path.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_name.lower().replace(' ', '_')}_seed_{master_seed}"
    candidate = base_path / stem
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = base_path / f"{stem}_{suffix}"
    candidate.mkdir()
    return candidate


def write_pabulib_sweep_outputs(
    election: PabulibElection,
    settings: PabulibSweepSettings,
    results: Sequence[PabulibTrialResult],
    summaries: Sequence[PabulibProjectCountSummary],
    diagnostics: dict[str, int],
    output_directory: str | Path,
) -> tuple[Path, Path, Path]:
    """Write trial data, summary intervals, and reproducibility metadata."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    trial_path = output_path / "trials.csv"
    summary_path = output_path / "summary.csv"
    metadata_path = output_path / "run.json"

    trial_fields = list(PabulibTrialResult.__dataclass_fields__)
    with trial_path.open("w", newline="", encoding="utf-8") as trial_file:
        writer = csv.DictWriter(trial_file, fieldnames=trial_fields)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            for field in ("project_ids", "outcome_project_ids", "optimal_project_ids"):
                row[field] = ";".join(row[field])
            writer.writerow(row)

    summary_fields = list(PabulibProjectCountSummary.__dataclass_fields__)
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(asdict(summary) for summary in summaries)

    digest = hashlib.sha256(election.source_path.read_bytes()).hexdigest()
    metadata: dict[str, Any] = {
        "experiment": "model-based Pabulib project-count sweep",
        "dataset_path": str(election.source_path),
        "dataset_sha256": digest,
        "dataset_metadata": election.metadata,
        "project_count": election.project_count,
        "voter_count": election.voter_count,
        "source_budget": election.metadata["budget"],
        "currency": election.metadata.get("currency", "unspecified"),
        "cost_precision_multiplier": election.decimal_multiplier,
        "cost_gcd": election.cost_gcd,
        "model_budget": election.instance.budget,
        "settings": asdict(settings),
        "diagnostics": diagnostics,
        "utility_sampling_method": (
            "Random linear objectives optimized over nonnegative normalized additive utilities "
            "consistent with each observed knapsack ballot; lazy violated-bundle cuts are added "
            "using an exact knapsack oracle. Each LP vertex is blended with a randomized Dirichlet "
            "utility vector supported on the submitted bundle; that witness is also consistent, so "
            "the convex mixture is consistent. This is a non-uniform model-based sampling rule, not "
            "uniform sampling from the utility polytope."
        ),
        "utility_witness_mix": settings.utility_witness_mix,
        "project_count_method": (
            "For each sampled utility profile, use nested random subsets of the real candidate set. "
            "Re-scale the budget for each subset to preserve the source election's budget / total "
            "candidate-cost ratio, then re-elicit knapsack ballots from the sampled utilities."
        ),
        "feasibility_model": (
            "Single total-budget constraint as in the paper's model. The source dataset also has "
            "neighborhood budget caps; these are not enforced in this comparison."
        ),
        "metric": (
            "sampled-instance distortion = exact optimal feasible welfare / chosen-outcome welfare, "
            "evaluated on inferred utilities consistent with sampled real ballots; not formal "
            "worst-case distortion and not measured true-utility welfare."
        ),
        "benchmark": "exact integer-cost dynamic programming after common-divisor scaling",
        "confidence_intervals": (
            "percentile bootstrap over voter subprofiles; all utility draws within a subprofile "
            "are kept together as one cluster"
        ),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    try:
        import scipy

        metadata["scipy_version"] = scipy.__version__
    except ImportError:  # pragma: no cover - SciPy is a declared dependency.
        metadata["scipy_version"] = "not installed"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return trial_path, summary_path, metadata_path
