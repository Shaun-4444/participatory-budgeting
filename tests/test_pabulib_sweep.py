from __future__ import annotations

from pathlib import Path

from pb_distortion.formats.knapsack import KnapsackBallotFormat
from pb_distortion.formats.threshold_approval import ThresholdApprovalBallotFormat
from pb_distortion.formats.value_ranking import ValueRankingFormat
from pb_distortion.formats.vfm_ranking import VFMRankingFormat
from pb_distortion.pabulib_data import load_pabulib_knapsack
from pb_distortion.pabulib_sweep import (
    PabulibSweepSettings,
    run_pabulib_project_count_sweep,
)
from pb_distortion.rules.greedy_approval import GreedyApprovalRule
from pb_distortion.rules.threshold_approval import ThresholdApprovalRule
from pb_distortion.rules.value_ranking import HarmonicValueRankingRule
from pb_distortion.rules.vfm_ranking import HarmonicVFMRankingRule


def _tiny_election(path: Path):
    path.write_text(
        "META\nkey;value\ndescription;Tiny test\nnum_projects;3\nnum_votes;4\n"
        "budget;3\nmax_sum_cost;3\nvote_type;approval\n"
        "PROJECTS\nproject_id;cost\na;1\nb;2\nc;3\n"
        "VOTES\nvoter_id;vote\nv1;a,b\nv2;c\nv3;a,b\nv4;c\n",
        encoding="utf-8",
    )
    return load_pabulib_knapsack(path)


def _sparse_election(path: Path):
    path.write_text(
        "META\nkey;value\ndescription;Sparse test\nnum_projects;6\nnum_votes;4\n"
        "budget;6\nmax_sum_cost;6\nvote_type;approval\n"
        "PROJECTS\nproject_id;cost\na;2\nb;2\nc;3\nd;3\ne;4\nf;4\n"
        "VOTES\nvoter_id;vote\nv1;a,b\nv2;c\nv3;d\nv4;f\n",
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


def test_pabulib_sweep_runs_all_three_formats(tmp_path: Path) -> None:
    election = _tiny_election(tmp_path / "tiny_all.pb")
    settings = PabulibSweepSettings(
        project_counts=(1, 2, 3),
        subprofile_count=2,
        voters_per_subprofile=2,
        utility_profiles_per_subprofile=2,
        master_seed=123,
        bootstrap_resamples=50,
    )
    format_rules = [
        (KnapsackBallotFormat(), GreedyApprovalRule()),
        (ValueRankingFormat(), HarmonicValueRankingRule()),
        (VFMRankingFormat(), HarmonicVFMRankingRule()),
    ]

    results, summaries, _ = run_pabulib_project_count_sweep(
        election,
        settings,
        format_rules,
    )

    # 3 formats x 3 project counts x (2 subprofiles x 2 utility draws)
    assert len(results) == 3 * 3 * (2 * 2)
    assert len(summaries) == 3 * 3  # 3 formats x 3 counts
    format_names = {s.format_name for s in summaries}
    assert format_names == {
        "knapsack + greedy_approval",
        "value_ranking + harmonic_value_ranking",
        "vfm_ranking + harmonic_vfm_ranking",
    }
    assert all(r.distortion >= 1.0 for r in results)


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


def test_pabulib_sweep_with_sparse_subprofiles(tmp_path: Path) -> None:
    election = _sparse_election(tmp_path / "sparse.pb")
    settings = PabulibSweepSettings(
        project_counts=(2, 4),
        subprofile_count=2,
        voters_per_subprofile=3,
        utility_profiles_per_subprofile=2,
        master_seed=999,
        bootstrap_resamples=50,
    )
    format_rules = [
        (KnapsackBallotFormat(), GreedyApprovalRule()),
        (ValueRankingFormat(), HarmonicValueRankingRule()),
        (VFMRankingFormat(), HarmonicVFMRankingRule()),
    ]

    results, summaries, _ = run_pabulib_project_count_sweep(
        election,
        settings,
        format_rules,
    )

    assert len(results) == 3 * 2 * (2 * 2)
    assert len(summaries) == 3 * 2
    assert all(r.distortion >= 1.0 for r in results)
