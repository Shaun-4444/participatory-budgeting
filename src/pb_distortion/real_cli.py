"""Command-line interface for model-based experiments on Pabulib files."""

from __future__ import annotations

import argparse
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

from pb_distortion.formats.knapsack import KnapsackBallotFormat
from pb_distortion.formats.threshold_approval import ThresholdApprovalBallotFormat
from pb_distortion.formats.value_ranking import ValueRankingFormat
from pb_distortion.formats.vfm_ranking import VFMRankingFormat
from pb_distortion.pabulib_data import load_pabulib_knapsack
from pb_distortion.pabulib_sweep import (
    PabulibSweepSettings,
    allocate_pabulib_output_directory,
    run_pabulib_project_count_sweep,
    write_pabulib_sweep_outputs,
)
from pb_distortion.plotting import plot_distortion_summary
from pb_distortion.protocols import AggregationRule, ElicitationFormat
from pb_distortion.rules.greedy_approval import GreedyApprovalRule
from pb_distortion.rules.threshold_approval import ThresholdApprovalRule
from pb_distortion.rules.value_ranking import HarmonicValueRankingRule
from pb_distortion.rules.vfm_ranking import HarmonicVFMRankingRule


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
    parser.add_argument(
        "--format",
        "--ballot-format",
        "--ballot",
        dest="ballot_format",
        choices=("all", "knapsack", "value_ranking", "vfm_ranking", "threshold_approval"),
        default="all",
        help=(
            "Select ballot format(s) to evaluate: 'all', 'knapsack', 'value_ranking', "
            "'vfm_ranking', or 'threshold_approval' (default: all)."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Approval threshold for threshold_approval format (default: 1/m).",
    )
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

    format_rules: list[tuple[ElicitationFormat[Any], AggregationRule[Any]]] = []
    if arguments.ballot_format in ("all", "knapsack"):
        format_rules.append((KnapsackBallotFormat(), GreedyApprovalRule()))
    if arguments.ballot_format in ("all", "value_ranking"):
        format_rules.append((ValueRankingFormat(), HarmonicValueRankingRule()))
    if arguments.ballot_format in ("all", "vfm_ranking"):
        format_rules.append((VFMRankingFormat(), HarmonicVFMRankingRule()))
    if arguments.ballot_format in ("all", "threshold_approval"):
        format_rules.append(
            (
                ThresholdApprovalBallotFormat(
                    threshold=arguments.threshold,
                    relative=True,
                ),
                ThresholdApprovalRule(),
            )
        )

    print(
        f"Loaded {election.metadata.get('description', election.source_path.name)}: "
        f"{election.voter_count:,} voters, {election.project_count} projects, "
        f"budget={election.metadata['budget']} {election.metadata.get('currency', '')}."
    )
    print(
        f"Evaluating {[fmt.name for fmt, _ in format_rules]} on inferred utility profiles; "
        "this can take a few minutes."
    )
    results, summaries, diagnostics = run_pabulib_project_count_sweep(
        election,
        settings,
        format_rules,
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

    means: dict[str, list[float]] = {}
    errors: dict[str, tuple[list[float], list[float]]] = {}
    for fmt, rule in format_rules:
        curve_name = f"{fmt.name} + {rule.name}"
        summary_by_count = {
            summary.project_count: summary
            for summary in summaries
            if summary.format_name == curve_name
        }
        means[curve_name] = [
            summary_by_count[m].mean_distortion for m in settings.project_counts
        ]
        errors[curve_name] = (
            [
                summary_by_count[m].mean_distortion - summary_by_count[m].ci_lower
                for m in settings.project_counts
            ],
            [
                summary_by_count[m].ci_upper - summary_by_count[m].mean_distortion
                for m in settings.project_counts
            ],
        )

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

    print(f"Utility inference diagnostics: {diagnostics}")
    for fmt, rule in format_rules:
        curve_name = f"{fmt.name} + {rule.name}"
        summary_by_count = {
            summary.project_count: summary
            for summary in summaries
            if summary.format_name == curve_name
        }
        print(f"\n--- {curve_name} ---")
        for m in settings.project_counts:
            s = summary_by_count[m]
            print(
                f"  m={s.project_count}: mean distortion={s.mean_distortion:.6f} "
                f"({settings.confidence_level:.0%} CI {s.ci_lower:.6f} to {s.ci_upper:.6f})"
            )
    print(f"\nTrial results: {trial_path}")
    print(f"Summary: {summary_path}")
    print(f"Run metadata: {metadata_path}")
    print(f"Comparison plot: {plot_path}")


if __name__ == "__main__":
    main()
