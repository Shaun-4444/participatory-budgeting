from __future__ import annotations

from pathlib import Path

from pb_distortion.formats.knapsack import KnapsackBallotFormat
from pb_distortion.formats.threshold_approval import ThresholdApprovalBallotFormat
from pb_distortion.pabulib_data import load_pabulib_knapsack
from pb_distortion.pabulib_sweep import (
    PabulibSweepSettings,
    run_pabulib_project_count_sweep,
)
from pb_distortion.rules.greedy_approval import GreedyApprovalRule
from pb_distortion.rules.threshold_approval import ThresholdApprovalRule


def _tiny_election(path: Path):
    path.write_text(
        "META\nkey;value\ndescription;Tiny test\nnum_projects;3\nnum_votes;4\n"
        "budget;3\nmax_sum_cost;3\nvote_type;approval\n"
        "PROJECTS\nproject_id;cost\na;1\nb;2\nc;3\n"
        "VOTES\nvoter_id;vote\nv1;a,b\nv2;c\nv3;a,b\nv4;c\n",
        encoding="utf-8",
    )
    return load_pabulib_knapsack(path)


def test_pabulib_sweep_is_reproducible_and_distortion_is_at_least_one(tmp_path: Path) -> None:
    election = _tiny_election(tmp_path / "tiny.pb")
    settings = PabulibSweepSettings(
        project_counts=(1, 2, 3),
        subprofile_count=3,
        voters_per_subprofile=2,
        utility_profiles_per_subprofile=2,
        master_seed=42,
        bootstrap_resamples=100,
    )
    format_rules = [(KnapsackBallotFormat(), GreedyApprovalRule())]

    first_results, first_summaries, first_diagnostics = run_pabulib_project_count_sweep(
        election,
        settings,
        format_rules,
    )
    second_results, second_summaries, second_diagnostics = run_pabulib_project_count_sweep(
        election,
        settings,
        format_rules,
    )

    assert first_results == second_results
    assert first_summaries == second_summaries
    assert first_diagnostics == second_diagnostics
    assert len(first_results) == 3 * 3 * 2
    assert all(result.distortion >= 1 for result in first_results)
    assert all(summary.ci_lower >= 1 for summary in first_summaries)
    assert all(summary.ci_upper >= summary.ci_lower for summary in first_summaries)


def test_pabulib_sweep_with_threshold_approval_and_knapsack(tmp_path: Path) -> None:
    election = _tiny_election(tmp_path / "tiny.pb")
    settings = PabulibSweepSettings(
        project_counts=(2, 3),
        subprofile_count=2,
        voters_per_subprofile=2,
        utility_profiles_per_subprofile=2,
        master_seed=123,
        bootstrap_resamples=50,
    )
    format_rules = [
        (KnapsackBallotFormat(), GreedyApprovalRule()),
        (ThresholdApprovalBallotFormat(relative=True), ThresholdApprovalRule()),
    ]

    results, summaries, diagnostics = run_pabulib_project_count_sweep(
        election,
        settings,
        format_rules,
    )

    # 2 formats * 2 subprofiles * 2 utility profiles * 2 project counts = 16 trial results
    assert len(results) == 2 * 2 * 2 * 2
    assert len(summaries) == 2 * 2  # 2 formats * 2 project counts

    curve_labels = {summary.format_name for summary in summaries}
    assert curve_labels == {
        "knapsack + greedy_approval",
        "threshold_approval + threshold_approval",
    }
    assert all(r.distortion >= 1.0 for r in results)
    assert all(s.mean_distortion >= 1.0 for s in summaries)
