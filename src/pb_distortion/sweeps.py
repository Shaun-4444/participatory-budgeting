"""Reproducible project-count sweeps with bootstrap uncertainty intervals."""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from pb_distortion.protocols import AggregationRule, ElicitationFormat
from pb_distortion.simulation import SimulationSettings, TrialResult, run_simulation


BallotT = TypeVar("BallotT")


@dataclass(frozen=True, slots=True)
class ProjectCountSummary:
    project_count: int
    trial_count: int
    mean_distortion: float
    ci_lower: float
    ci_upper: float


def run_project_count_sweep(
    settings: SimulationSettings,
    project_counts: tuple[int, ...] | list[int],
    ballot_format: ElicitationFormat[BallotT],
    aggregation_rule: AggregationRule[BallotT],
    *,
    bootstrap_resamples: int = 2000,
    confidence_level: float = 0.95,
) -> tuple[list[TrialResult], list[ProjectCountSummary]]:
    """Run each project count and estimate a bootstrap CI for mean distortion."""

    counts = tuple(project_counts)
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("project_counts must contain positive integers.")
    if len(set(counts)) != len(counts):
        raise ValueError("project_counts must not contain duplicates.")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    all_results: list[TrialResult] = []
    summaries: list[ProjectCountSummary] = []
    for project_count in counts:
        point_settings = replace(settings, project_count=project_count)
        results = run_simulation(point_settings, ballot_format, aggregation_rule)
        values = np.asarray([result.distortion for result in results], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Distortion must be finite for bootstrap intervals; project_count={project_count}."
            )

        mean_distortion = float(values.mean())
        bootstrap_rng = np.random.default_rng(
            np.random.SeedSequence([settings.master_seed, project_count])
        )
        sample_indices = bootstrap_rng.integers(
            0,
            len(values),
            size=(bootstrap_resamples, len(values)),
        )
        bootstrap_means = values[sample_indices].mean(axis=1)
        tail_probability = (1.0 - confidence_level) / 2.0
        ci_lower, ci_upper = np.quantile(
            bootstrap_means,
            [tail_probability, 1.0 - tail_probability],
        )

        all_results.extend(results)
        summaries.append(
            ProjectCountSummary(
                project_count=project_count,
                trial_count=len(results),
                mean_distortion=mean_distortion,
                ci_lower=float(ci_lower),
                ci_upper=float(ci_upper),
            )
        )
    return all_results, summaries


def run_noise_sweep(
    settings: SimulationSettings,
    noise_sigmas: tuple[float, ...] | list[float],
    project_counts: tuple[int, ...] | list[int],
    ballot_format: ElicitationFormat[BallotT],
    aggregation_rule: AggregationRule[BallotT],
    *,
    bootstrap_resamples: int = 2000,
    confidence_level: float = 0.95,
) -> dict[float, tuple[list[TrialResult], list[ProjectCountSummary]]]:
    """Run a separate constant-sigma project-count sweep for every noise level."""

    sigmas = tuple(float(sigma) for sigma in noise_sigmas)
    if not sigmas:
        raise ValueError("noise_sigmas must contain at least one value.")
    if len(set(sigmas)) != len(sigmas):
        raise ValueError("noise_sigmas must not contain duplicates.")

    output: dict[float, tuple[list[TrialResult], list[ProjectCountSummary]]] = {}
    for sigma in sigmas:
        noise_settings = replace(settings, noise_sigma=sigma)
        output[sigma] = run_project_count_sweep(
            noise_settings,
            project_counts,
            ballot_format,
            aggregation_rule,
            bootstrap_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
        )
    return output


def write_project_count_sweep_outputs(
    results: list[TrialResult],
    summaries: list[ProjectCountSummary],
    settings: SimulationSettings,
    *,
    project_counts: tuple[int, ...] | list[int],
    ballot_format_name: str,
    aggregation_rule_name: str,
    bootstrap_resamples: int,
    confidence_level: float,
    output_directory: str | Path,
    source_config: dict[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    """Write trial CSV, summary CSV, and run metadata for a sweep."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    trials_path = output_path / "trials.csv"
    summary_path = output_path / "summary.csv"
    metadata_path = output_path / "run.json"

    trial_fields = list(TrialResult.__dataclass_fields__)
    with trials_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=trial_fields)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["outcome_project_ids"] = ";".join(result.outcome_project_ids)
            row["optimal_project_ids"] = ";".join(result.optimal_project_ids)
            writer.writerow(row)

    summary_fields = list(ProjectCountSummary.__dataclass_fields__)
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(asdict(summary) for summary in summaries)

    trial_seeds_by_project_count = {
        str(project_count): [result.seed for result in results if result.project_count == project_count]
        for project_count in project_counts
    }
    metadata = {
        "settings": asdict(settings),
        "project_counts": list(project_counts),
        "ballot_format": ballot_format_name,
        "aggregation_rule": aggregation_rule_name,
        "master_seed": settings.master_seed,
        "trial_seeds_by_project_count": trial_seeds_by_project_count,
        "trial_count_per_project_count": settings.trials,
        "benchmark": "exact integer-cost dynamic programming",
        "metric": "distortion (optimal welfare / chosen welfare; sampled instance)",
        "uncertainty_interval": {
            "method": "percentile bootstrap of the mean across elections",
            "confidence_level": confidence_level,
            "resamples": bootstrap_resamples,
            "seed_derivation": "SeedSequence([master_seed, project_count])",
        },
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "source_config": source_config or {},
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return trials_path, summary_path, metadata_path
