"""Command-line entry point for the baseline knapsack experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tomllib
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from pb_distortion.formats.knapsack import KnapsackBallotFormat
from pb_distortion.formats.value_ranking import ValueRankingFormat
from pb_distortion.formats.vfm_ranking import VFMRankingFormat
from pb_distortion.plotting import plot_distortion_summary
from pb_distortion.rules.greedy_approval import GreedyApprovalRule
from pb_distortion.rules.value_ranking import HarmonicValueRankingRule
from pb_distortion.rules.vfm_ranking import HarmonicVFMRankingRule
from pb_distortion.simulation import SimulationSettings, run_simulation, write_simulation_outputs
from pb_distortion.sweeps import (
    run_noise_sweep,
    run_project_count_sweep,
    write_project_count_sweep_outputs,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as config_file:
        config = tomllib.load(config_file)
    required_sections = {"simulation", "instance", "utilities", "output"}
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    return config


def settings_from_config(
    config: dict[str, Any],
    *,
    project_count: int | None = None,
    utility_distribution: str | None = None,
    noise_sigma: float | None = None,
) -> SimulationSettings:
    utility_config = config["utilities"]
    noise_config = config.get("noise", {})
    instance_config = config["instance"]
    if project_count is None:
        configured_count = instance_config.get("project_count")
        if configured_count is None:
            project_counts = instance_config.get("project_counts", [])
            if not project_counts:
                raise ValueError("Set instance.project_count or instance.project_counts.")
            configured_count = project_counts[0]
        project_count = int(configured_count)
    return SimulationSettings(
        trials=int(config["simulation"]["trials"]),
        master_seed=int(config["simulation"]["master_seed"]),
        project_count=project_count,
        voter_count=int(instance_config["voter_count"]),
        cost_min=int(instance_config["cost_min"]),
        cost_max=int(instance_config["cost_max"]),
        budget_fraction=float(instance_config["budget_fraction"]),
        utility_minimum=float(utility_config.get("minimum", 0.0)),
        utility_maximum=float(utility_config.get("maximum", 1.0)),
        utility_distribution=(
            utility_distribution or str(utility_config.get("distribution", "uniform")).lower()
        ),
        normal_mean=float(utility_config.get("normal_mean", 0.5)),
        normal_std=float(utility_config.get("normal_std", 0.2)),
        pareto_shape=float(utility_config.get("pareto_shape", 1.5)),
        normalize_rows=bool(utility_config["normalize_rows"]),
        noise_sigma=float(
            noise_config.get("sigma", 0.0) if noise_sigma is None else noise_sigma
        ),
    )


def _allocate_unique_directory(base_directory: str | Path, run_name: str) -> Path:
    base_path = Path(base_directory)
    base_path.mkdir(parents=True, exist_ok=True)
    run_index = 1
    while True:
        suffix = "" if run_index == 1 else f"_run_{run_index}"
        candidate = base_path / f"{run_name}{suffix}"
        try:
            candidate.mkdir()
        except FileExistsError:
            run_index += 1
        else:
            return candidate


def _sigma_token(sigma: float) -> str:
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("noise sigma must be finite and nonnegative.")
    return format(sigma, "g").replace("-", "m").replace(".", "p")


def allocate_run_output_directory(
    base_directory: str | Path,
    utility_distribution: str,
    noise_sigma: float,
    master_seed: int,
) -> Path:
    """Create a fresh distribution-, noise-, and seed-specific output folder."""

    name = f"{utility_distribution}_sigma_{_sigma_token(noise_sigma)}_seed_{master_seed}"
    return _allocate_unique_directory(base_directory, name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PB distortion experiments with selectable ballot formats.")
    parser.add_argument(
        "--config",
        default="configs/baseline.toml",
        help="Path to a TOML run configuration (default: configs/baseline.toml).",
    )
    parser.add_argument(
        "--output",
        help="Override the output base directory; a unique run folder is created inside.",
    )
    parser.add_argument(
        "--distribution",
        choices=("uniform", "normal", "pareto"),
        help="Override the utility distribution selected in the config.",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        help="Use this constant lognormal noise level for one run.",
    )
    parser.add_argument(
        "--noise-sigmas",
        type=float,
        nargs="+",
        metavar="SIGMA",
        help="Run a separate project-count sweep at each sigma and plot them together.",
    )
    parser.add_argument(
        "--ballot",
        choices=("knapsack", "value_ranking", "vfm_ranking"),
        default="knapsack",
        help=(
            "Ballot format and aggregation rule to use. 'knapsack' uses the knapsack ballot "
            "with greedy approval (baseline). 'value_ranking' uses value-ranking ballots with "
            "harmonic scoring and cost bucketing. 'vfm_ranking' uses value-for-money ranking "
            "ballots with harmonic scoring and cost bucketing. Default: knapsack."
        ),
    )
    arguments = parser.parse_args()
    if arguments.noise_sigma is not None and arguments.noise_sigmas is not None:
        parser.error("Choose either --noise-sigma or --noise-sigmas, not both.")

    config_path = Path(arguments.config).resolve()
    config = load_config(config_path)
    settings = settings_from_config(
        config,
        utility_distribution=arguments.distribution,
        noise_sigma=arguments.noise_sigma,
    )
    output_value = arguments.output or config["output"]["directory"]
    output_base_directory = Path(output_value)
    if not output_base_directory.is_absolute():
        output_base_directory = config_path.parent.parent / output_base_directory

    _ballot_pairs = {
        "knapsack": (KnapsackBallotFormat(), GreedyApprovalRule()),
        "value_ranking": (ValueRankingFormat(), HarmonicValueRankingRule()),
        "vfm_ranking": (VFMRankingFormat(), HarmonicVFMRankingRule()),
    }
    ballot_format, aggregation_rule = _ballot_pairs[arguments.ballot]
    project_counts = config["instance"].get("project_counts")
    analysis_config = config.get("analysis", {})
    bootstrap_resamples = int(analysis_config.get("bootstrap_resamples", 2000))
    confidence_level = float(analysis_config.get("confidence_level", 0.95))

    if arguments.noise_sigmas is not None:
        if project_counts is None:
            parser.error("--noise-sigmas requires instance.project_counts in the config.")
        counts = tuple(int(count) for count in project_counts)
        sigmas = tuple(float(sigma) for sigma in arguments.noise_sigmas)
        if len(set(sigmas)) != len(sigmas):
            parser.error("--noise-sigmas values must be unique.")
        try:
            for sigma in sigmas:
                replace(settings, noise_sigma=sigma)
                _sigma_token(sigma)
        except ValueError as error:
            parser.error(str(error))

        batch_name = f"noise_sweep_{settings.utility_distribution}_seed_{settings.master_seed}"
        batch_directory = _allocate_unique_directory(output_base_directory, batch_name)
        print(f"Noise-sweep output directory: {batch_directory}")
        noise_results = run_noise_sweep(
            settings,
            sigmas,
            counts,
            ballot_format,
            aggregation_rule,
            bootstrap_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
        )

        comparison_means: dict[str, list[float]] = {}
        comparison_errors: dict[str, tuple[list[float], list[float]]] = {}
        summary_rows: list[dict[str, Any]] = []
        for sigma, (results, summaries) in noise_results.items():
            sigma_name = f"sigma_{_sigma_token(sigma)}"
            sigma_directory = batch_directory / sigma_name
            sigma_directory.mkdir()
            sigma_settings = replace(settings, noise_sigma=sigma)
            trials_path, summary_path, metadata_path = write_project_count_sweep_outputs(
                results,
                summaries,
                sigma_settings,
                project_counts=counts,
                ballot_format_name=ballot_format.name,
                aggregation_rule_name=aggregation_rule.name,
                bootstrap_resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                output_directory=sigma_directory,
                source_config=config,
            )
            series_name = f"σ={sigma:g}"
            values = [summary.mean_distortion for summary in summaries]
            comparison_means[series_name] = values
            comparison_errors[series_name] = (
                [summary.mean_distortion - summary.ci_lower for summary in summaries],
                [summary.ci_upper - summary.mean_distortion for summary in summaries],
            )
            summary_rows.extend(
                {
                    "noise_sigma": sigma,
                    "project_count": summary.project_count,
                    "trial_count": summary.trial_count,
                    "mean_distortion": summary.mean_distortion,
                    "ci_lower": summary.ci_lower,
                    "ci_upper": summary.ci_upper,
                }
                for summary in summaries
            )
            run_plot = plot_distortion_summary(
                counts,
                {ballot_format.name: values},
                {
                    ballot_format.name: comparison_errors[series_name],
                },
                sigma_directory / "distortion_by_project_count.svg",
                title=(
                    f"Mean distortion by project count ({series_name}; "
                    f"{settings.trials} elections; {confidence_level:.0%} bootstrap CI)"
                ),
            )
            print(f"sigma={sigma:g}: trials={trials_path}; summary={summary_path}; plot={run_plot}")
            print(f"sigma={sigma:g}: run settings and seeds={metadata_path}")
            for summary in summaries:
                print(
                    f"  m={summary.project_count}: mean distortion={summary.mean_distortion:.6f} "
                    f"({confidence_level:.0%} CI {summary.ci_lower:.6f} "
                    f"to {summary.ci_upper:.6f})"
                )

        combined_summary_path = batch_directory / "noise_sweep_summary.csv"
        with combined_summary_path.open("w", newline="", encoding="utf-8") as summary_file:
            fields = [
                "noise_sigma",
                "project_count",
                "trial_count",
                "mean_distortion",
                "ci_lower",
                "ci_upper",
            ]
            writer = csv.DictWriter(summary_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary_rows)

        combined_plot_path = plot_distortion_summary(
            counts,
            comparison_means,
            comparison_errors,
            batch_directory / "distortion_by_noise_sigma.svg",
            title=(
                f"Distortion by project count and noise level "
                f"({settings.utility_distribution}; {confidence_level:.0%} bootstrap CI)"
            ),
        )
        batch_settings = asdict(settings)
        batch_settings.pop("noise_sigma")
        batch_metadata = {
            "settings_shared_across_noise_levels": batch_settings,
            "noise_sigmas": list(sigmas),
            "project_counts": list(counts),
            "trial_count_per_project_count_and_sigma": settings.trials,
            "bootstrap_resamples": bootstrap_resamples,
            "confidence_level": confidence_level,
            "source_config": config,
        }
        (batch_directory / "noise_sweep_run.json").write_text(
            json.dumps(batch_metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Combined noise summary: {combined_summary_path}")
        print(f"Combined noise plot: {combined_plot_path}")
        return

    if project_counts is not None:
        counts = tuple(int(count) for count in project_counts)
        settings = settings_from_config(
            config,
            project_count=counts[0] if counts else None,
            utility_distribution=arguments.distribution,
            noise_sigma=arguments.noise_sigma,
        )

    output_directory = allocate_run_output_directory(
        output_base_directory,
        settings.utility_distribution,
        settings.noise_sigma,
        settings.master_seed,
    )
    print(f"Run output directory: {output_directory}")

    if project_counts is not None:
        results, summaries = run_project_count_sweep(
            settings,
            counts,
            ballot_format,
            aggregation_rule,
            bootstrap_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
        )
        trials_path, summary_path, json_path = write_project_count_sweep_outputs(
            results,
            summaries,
            settings,
            project_counts=counts,
            ballot_format_name=ballot_format.name,
            aggregation_rule_name=aggregation_rule.name,
            bootstrap_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            output_directory=output_directory,
            source_config=config,
        )
        means = {
            ballot_format.name: [summary.mean_distortion for summary in summaries]
        }
        errors = {
            ballot_format.name: (
                [summary.mean_distortion - summary.ci_lower for summary in summaries],
                [summary.ci_upper - summary.mean_distortion for summary in summaries],
            )
        }
        plot_path = plot_distortion_summary(
            counts,
            means,
            errors,
            output_directory / "distortion_by_project_count.svg",
            title=f"Mean distortion by project count ({settings.trials} elections; {confidence_level:.0%} bootstrap CI)",
        )
        print(f"Project counts: {', '.join(str(count) for count in counts)}")
        print(f"Elections per project count: {settings.trials}")
        for summary in summaries:
            print(
                f"m={summary.project_count}: mean distortion={summary.mean_distortion:.6f} "
                f"({confidence_level:.0%} CI {summary.ci_lower:.6f} to {summary.ci_upper:.6f})"
            )
        print(f"Trial results: {trials_path}")
        print(f"Sweep summary: {summary_path}")
        print(f"Run settings and seeds: {json_path}")
        print(f"Plot: {plot_path}")
        return

    results = run_simulation(settings, ballot_format, aggregation_rule)
    csv_path, json_path = write_simulation_outputs(
        results,
        settings,
        ballot_format_name=ballot_format.name,
        aggregation_rule_name=aggregation_rule.name,
        output_directory=output_directory,
        source_config=config,
    )
    mean_distortion = sum(result.distortion for result in results) / len(results)
    print(f"Trials: {len(results)}")
    print(f"Mean distortion: {mean_distortion:.6f}")
    print(f"Trial results: {csv_path}")
    print(f"Run settings and seeds: {json_path}")


if __name__ == "__main__":
    main()
