"""Command-line interface for model-based experiments on Pabulib files."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

from pb_distortion.formats.knapsack import KnapsackBallotFormat
from pb_distortion.pabulib_data import load_pabulib_knapsack
from pb_distortion.pabulib_sweep import (
    PabulibSweepSettings,
    allocate_pabulib_output_directory,
    run_pabulib_project_count_sweep,
    write_pabulib_sweep_outputs,
)
from pb_distortion.plotting import plot_distortion_summary
from pb_distortion.rules.greedy_approval import GreedyApprovalRule


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a model-based project-count sweep using utility profiles "
            "consistent with observed Pabulib knapsack ballots."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/pabulib_amsterdam.toml",
        help="Path to a TOML Pabulib experiment config.",
    )
    parser.add_argument("--dataset", help="Override the .pb path from the config.")
    parser.add_argument("--output", help="Override the output base directory.")
    parser.add_argument("--seed", type=int, help="Override the master seed.")
    arguments = parser.parse_args()

    config_path = Path(arguments.config).expanduser().resolve()
    config = _read_config(config_path)
    repository_root = config_path.parent.parent
    dataset_value = arguments.dataset or config["dataset"]["path"]
    dataset_path = Path(dataset_value).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = repository_root / dataset_path

    experiment = config["experiment"]
    analysis = config.get("analysis", {})
    settings = PabulibSweepSettings(
        project_counts=tuple(int(value) for value in experiment["project_counts"]),
        subprofile_count=int(experiment.get("subprofile_count", 20)),
        voters_per_subprofile=int(experiment.get("voters_per_subprofile", 5)),
        utility_profiles_per_subprofile=int(
            experiment.get("utility_profiles_per_subprofile", 5)
        ),
        utility_witness_mix=float(experiment.get("utility_witness_mix", 0.75)),
        master_seed=(
            int(arguments.seed)
            if arguments.seed is not None
            else int(experiment.get("master_seed", 42))
        ),
        bootstrap_resamples=int(analysis.get("bootstrap_resamples", 2000)),
        confidence_level=float(analysis.get("confidence_level", 0.95)),
        max_utility_cuts=int(experiment.get("max_utility_cuts", 500)),
    )
    output_value = arguments.output or config["output"]["directory"]
    output_base = Path(output_value).expanduser()
    if not output_base.is_absolute():
        output_base = repository_root / output_base

    election = load_pabulib_knapsack(dataset_path)
    ballot_format = KnapsackBallotFormat()
    aggregation_rule = GreedyApprovalRule()
    print(
        f"Loaded {election.metadata.get('description', election.source_path.name)}: "
        f"{election.voter_count:,} voters, {election.project_count} projects, "
        f"budget={election.metadata['budget']} {election.metadata.get('currency', '')}."
    )
    print(
        "Sampling utility profiles consistent with observed knapsack ballots; "
        "this can take a few minutes."
    )
    results, summaries, diagnostics = run_pabulib_project_count_sweep(
        election,
        settings,
        [(ballot_format, aggregation_rule)],
    )

    output_directory = allocate_pabulib_output_directory(
        output_base,
        f"{election.metadata.get('country', 'pabulib')}_"
        f"{election.metadata.get('unit', 'election')}_"
        f"{election.metadata.get('instance', 'unknown')}",
        settings.master_seed,
    )
    trial_path, summary_path, metadata_path = write_pabulib_sweep_outputs(
        election,
        settings,
        results,
        summaries,
        diagnostics,
        output_directory,
    )

    curve_name = f"{ballot_format.name} + {aggregation_rule.name}"
    summary_by_count = {
        summary.project_count: summary for summary in summaries
    }
    means = {curve_name: [summary_by_count[m].mean_distortion for m in settings.project_counts]}
    errors = {
        curve_name: (
            [
                summary_by_count[m].mean_distortion - summary_by_count[m].ci_lower
                for m in settings.project_counts
            ],
            [
                summary_by_count[m].ci_upper - summary_by_count[m].mean_distortion
                for m in settings.project_counts
            ],
        )
    }
    plot_path = plot_distortion_summary(
        settings.project_counts,
        means,
        errors,
        output_directory / "distortion_by_project_count.svg",
        title=(
            f"Amsterdam 2020 Centrum: model-based distortion by project count "
            f"({settings.subprofile_count} voter subprofiles; "
            f"{settings.confidence_level:.0%} cluster-bootstrap CI)"
        ),
    )

    print(f"Utility inference: {diagnostics}")
    for summary in summaries:
        print(
            f"m={summary.project_count}: mean distortion={summary.mean_distortion:.6f} "
            f"({settings.confidence_level:.0%} CI {summary.ci_lower:.6f} "
            f"to {summary.ci_upper:.6f})"
        )
    print(f"Trial results: {trial_path}")
    print(f"Summary: {summary_path}")
    print(f"Run metadata: {metadata_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
