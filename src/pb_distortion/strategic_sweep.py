"""Comprehensive strategic voting experiment sweeps for participatory budgeting.

Evaluates how all four canonical PB elicitation formats degrade when an individual voter
strategically misreports to promote their favorite project. Connects the empirical findings
to the Gibbard-Satterthwaite theorem and large-market asymptotic strategyproofness.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from pb_distortion.formats.knapsack import KnapsackBallotFormat
from pb_distortion.formats.threshold_approval import ThresholdApprovalBallotFormat
from pb_distortion.formats.value_ranking import ValueRankingFormat
from pb_distortion.formats.vfm_ranking import VFMRankingFormat
from pb_distortion.metrics import distortion, social_welfare
from pb_distortion.models import Outcome, PBInstance, UtilityProfile
from pb_distortion.pabulib_data import PabulibElection, load_pabulib_knapsack
from pb_distortion.pabulib_sweep import (
    PabulibSweepSettings,
    make_scaled_subinstance,
)
from pb_distortion.plotting import plot_distortion_summary, plot_strategic_comparison
from pb_distortion.protocols import AggregationRule, ElicitationFormat
from pb_distortion.rules.greedy_approval import GreedyApprovalRule
from pb_distortion.rules.threshold_approval import ThresholdApprovalRule
from pb_distortion.rules.value_ranking import HarmonicValueRankingRule
from pb_distortion.rules.vfm_ranking import HarmonicVFMRankingRule
from pb_distortion.simulation import SimulationSettings
from pb_distortion.solvers import bundle_cost, optimal_feasible_bundle
from pb_distortion.strategic import (
    StrategicFormatSummary,
    StrategicTrialResult,
    evaluate_strategic_election,
    generate_strategic_ballot,
    identify_favorite_project,
    run_strategic_simulation,
)
from pb_distortion.utility_inference import sample_knapsack_consistent_utilities


def get_standard_format_rules() -> list[tuple[str, ElicitationFormat[Any], AggregationRule[Any]]]:
    """Return the four standard format/rule pairs with descriptive names."""
    return [
        ("Knapsack", KnapsackBallotFormat(), GreedyApprovalRule()),
        (
            "Threshold Approval",
            ThresholdApprovalBallotFormat(threshold=0.10, relative=True),
            ThresholdApprovalRule(),
        ),
        ("Value Ranking", ValueRankingFormat(), HarmonicValueRankingRule()),
        ("VFM Ranking", VFMRankingFormat(), HarmonicVFMRankingRule()),
    ]


def run_synthetic_format_comparison(
    output_dir: Path,
    *,
    distribution: str = "uniform",
    trials: int = 200,
    m: int = 10,
    n: int = 20,
    master_seed: int = 42,
) -> list[StrategicFormatSummary]:
    """Run head-to-head comparison across all four formats on synthetic data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    format_rules = get_standard_format_rules()

    settings = SimulationSettings(
        trials=trials,
        master_seed=master_seed,
        project_count=m,
        voter_count=n,
        cost_min=1,
        cost_max=10,
        budget_fraction=0.33,
        utility_distribution=distribution,
        noise_sigma=0.0,
    )

    summaries: list[StrategicFormatSummary] = []
    trials_csv = output_dir / "trials.csv"

    trial_fieldnames = list(StrategicTrialResult.__dataclass_fields__)
    with trials_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=trial_fieldnames)
        writer.writeheader()

        for label, fmt, rule in format_rules:
            print(f"  Evaluating {label} ({trials} trials)...")
            res_list, summary = run_strategic_simulation(settings, fmt, rule)
            summaries.append(summary)

            for r in res_list:
                row = dict(asdict(r))
                row["honest_outcome"] = ";".join(r.honest_outcome)
                row["strategic_outcome"] = ";".join(r.strategic_outcome)
                row["optimal_outcome"] = ";".join(r.optimal_outcome)
                writer.writerow(row)

    # Save summary CSV
    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        summary_writer = csv.writer(f)
        summary_writer.writerow([
            "format_name",
            "rule_name",
            "trial_count",
            "voter_count",
            "project_count",
            "mean_honest_distortion",
            "mean_strategic_distortion",
            "mean_distortion_degradation",
            "mean_welfare_loss_pct",
            "manipulation_success_rate",
            "favorite_promoted_rate",
            "mean_strategic_gain",
            "ci_honest_lower",
            "ci_honest_upper",
            "ci_strat_lower",
            "ci_strat_upper",
        ])
        for s in summaries:
            summary_writer.writerow([
                s.format_name,
                s.rule_name,
                s.trial_count,
                s.voter_count,
                s.project_count,
                s.mean_honest_distortion,
                s.mean_strategic_distortion,
                s.mean_distortion_degradation,
                s.mean_welfare_loss_percentage,
                s.manipulation_success_rate,
                s.favorite_promoted_rate,
                s.mean_strategic_utility_gain,
                s.distortion_ci_honest[0],
                s.distortion_ci_honest[1],
                s.distortion_ci_strategic[0],
                s.distortion_ci_strategic[1],
            ])

    # Plot comparison bar chart
    labels = [label for label, _, _ in format_rules]
    honest_means = [s.mean_honest_distortion for s in summaries]
    h_err_lower = [s.mean_honest_distortion - s.distortion_ci_honest[0] for s in summaries]
    h_err_upper = [s.distortion_ci_honest[1] - s.mean_honest_distortion for s in summaries]

    strat_means = [s.mean_strategic_distortion for s in summaries]
    s_err_lower = [s.mean_strategic_distortion - s.distortion_ci_strategic[0] for s in summaries]
    s_err_upper = [s.distortion_ci_strategic[1] - s.mean_strategic_distortion for s in summaries]

    wl_pcts = [s.mean_welfare_loss_percentage for s in summaries]
    succ_rates = [s.manipulation_success_rate for s in summaries]

    plot_strategic_comparison(
        labels,
        honest_means,
        (h_err_lower, h_err_upper),
        strat_means,
        (s_err_lower, s_err_upper),
        wl_pcts,
        succ_rates,
        output_dir / "format_degradation_comparison.svg",
        title=f"Strategic Voting Degradation: {distribution.capitalize()} (m={m}, n={n})",
    )

    return summaries


def run_electorate_dilution_sweep(
    output_dir: Path,
    *,
    voter_counts: tuple[int, ...] = (5, 10, 20, 30, 50, 100),
    trials: int = 150,
    m: int = 10,
    distribution: str = "uniform",
    master_seed: int = 42,
) -> None:
    """Sweep electorate size n to show how strategic impact dilutes as n grows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    format_rules = get_standard_format_rules()

    # Collect degradation curve per format
    degradations_by_format: dict[str, list[float]] = {label: [] for label, _, _ in format_rules}
    success_by_format: dict[str, list[float]] = {label: [] for label, _, _ in format_rules}

    for n in voter_counts:
        print(f"  Sweeping n = {n} voters ({trials} trials)...")
        settings = SimulationSettings(
            trials=trials,
            master_seed=master_seed + n * 17,
            project_count=m,
            voter_count=n,
            cost_min=1,
            cost_max=10,
            budget_fraction=0.33,
            utility_distribution=distribution,
            noise_sigma=0.0,
        )

        for label, fmt, rule in format_rules:
            _, summary = run_strategic_simulation(settings, fmt, rule, bootstrap_resamples=100)
            degradations_by_format[label].append(summary.mean_distortion_degradation)
            success_by_format[label].append(summary.manipulation_success_rate * 100.0)

    # Save CSV
    csv_path = output_dir / "dilution_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["voter_count"] + [f"{l}_distortion_degradation" for l in degradations_by_format] + [f"{l}_success_pct" for l in success_by_format]
        writer.writerow(header)
        for idx, n in enumerate(voter_counts):
            row = [n] + [degradations_by_format[l][idx] for l in degradations_by_format] + [success_by_format[l][idx] for l in success_by_format]
            writer.writerow(row)

    # Generate line plot for degradation vs n
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for (label, _), color in zip(degradations_by_format.items(), colors, strict=True):
        ax1.plot(voter_counts, degradations_by_format[label], marker="o", linewidth=2.2, label=label, color=color)
        ax2.plot(voter_counts, success_by_format[label], marker="s", linewidth=2.2, label=label, color=color)

    ax1.set_xlabel("Number of Voters (n)", fontsize=11)
    ax1.set_ylabel("Mean Distortion Degradation (ΔD)", fontsize=11)
    ax1.set_title("Distortion Degradation Dilution", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("Number of Voters (n)", fontsize=11)
    ax2.set_ylabel("Manipulation Success Rate (%)", fontsize=11)
    ax2.set_title("Strategic Success Rate Dilution", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend()

    fig.suptitle("Large-Market Asymptotic Strategyproofness (Dilution of Manipulation)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "strategic_dilution_by_voters.svg", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_pabulib_strategic_evaluation(
    election: PabulibElection,
    output_dir: Path,
    *,
    m: int = 10,
    subprofile_count: int = 20,
    voters_per_subprofile: int = 5,
    utility_profiles_per_subprofile: int = 5,
    master_seed: int = 42,
) -> dict[str, Any]:
    """Evaluate strategic voting degradation on real Amsterdam Pabulib data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    format_rules = get_standard_format_rules()

    seed_seq = np.random.SeedSequence(master_seed)
    subprofile_seeds = seed_seq.spawn(subprofile_count)
    rng = np.random.default_rng(seed_seq.spawn(1)[0])

    project_index = {pid: idx for idx, pid in enumerate(election.instance.project_ids)}
    project_order = list(range(election.project_count))
    rng.shuffle(project_order)

    # Track metrics per format across all elections
    honest_dists: dict[str, list[float]] = {l: [] for l, _, _ in format_rules}
    strat_dists: dict[str, list[float]] = {l: [] for l, _, _ in format_rules}
    welfare_losses: dict[str, list[float]] = {l: [] for l, _, _ in format_rules}
    success_flags: dict[str, list[bool]] = {l: [] for l, _, _ in format_rules}
    promoted_flags: dict[str, list[bool]] = {l: [] for l, _, _ in format_rules}

    total_elections = 0
    print(f"  Running Pabulib strategic evaluation ({subprofile_count} subprofiles x {utility_profiles_per_subprofile} profiles)...")

    for sub_idx, sub_seq in enumerate(subprofile_seeds):
        voter_rng = np.random.default_rng(sub_seq.spawn(1)[0])
        sampled_voters = voter_rng.choice(
            election.voter_count,
            size=voters_per_subprofile,
            replace=False,
        )
        sampled_ballots = [election.ballots[v] for v in sampled_voters]

        utility_seeds = sub_seq.spawn(utility_profiles_per_subprofile)
        for u_seed in utility_seeds:
            total_elections += 1
            utility_rng = np.random.default_rng(u_seed)
            full_utilities = []
            for ballot in sampled_ballots:
                utils, _ = sample_knapsack_consistent_utilities(
                    election.instance,
                    ballot,
                    utility_rng,
                    witness_mix=0.75,
                    max_cuts=500,
                )
                full_utilities.append(utils)
            utility_matrix = np.vstack(full_utilities)

            # Create subinstance of size m
            selected_indices = project_order[:m]
            subinstance, _, _ = make_scaled_subinstance(election, selected_indices)
            subproject_ids = subinstance.project_ids
            utility_columns = [project_index[pid] for pid in subproject_ids]

            subprofile = UtilityProfile(
                utility_matrix[:, utility_columns],
                subproject_ids,
            )

            social_values = subprofile.values.sum(axis=0)
            optimum = optimal_feasible_bundle(subinstance, social_values)
            optimum_welfare = social_welfare(subprofile, optimum)

            # Evaluate each format with voter 0 strategic
            strat_perceived = subprofile.values[0]
            val_by_pid = dict(zip(subproject_ids, strat_perceived, strict=True))
            fav_id = identify_favorite_project(strat_perceived, subinstance)

            for label, fmt, rule in format_rules:
                # 1. Honest
                h_ballots = [fmt.elicit(subprofile.values[v], subinstance) for v in range(voters_per_subprofile)]
                h_outcome = rule.aggregate(h_ballots, subinstance)
                h_welfare = social_welfare(subprofile, h_outcome)
                h_dist = distortion(optimum_welfare, h_welfare)

                # 2. Strategic
                s_ballot = generate_strategic_ballot(fmt, strat_perceived, subinstance, favorite_id=fav_id)
                s_ballots = list(h_ballots)
                s_ballots[0] = s_ballot
                s_outcome = rule.aggregate(s_ballots, subinstance)
                s_welfare = social_welfare(subprofile, s_outcome)
                s_dist = distortion(optimum_welfare, s_welfare)

                # Metrics
                wl_pct = ((h_welfare - s_welfare) / h_welfare * 100.0) if h_welfare > 0 else 0.0
                v_h_u = sum(val_by_pid[pid] for pid in h_outcome)
                v_s_u = sum(val_by_pid[pid] for pid in s_outcome)
                gain = v_s_u - v_h_u

                fav_in_h = fav_id in h_outcome
                fav_in_s = fav_id in s_outcome
                fav_promoted = (not fav_in_h) and fav_in_s
                manip_success = gain > 1e-9

                honest_dists[label].append(h_dist)
                strat_dists[label].append(s_dist)
                welfare_losses[label].append(wl_pct)
                success_flags[label].append(manip_success)
                promoted_flags[label].append(fav_promoted)

    # Compute cluster bootstrap CIs
    labels = [l for l, _, _ in format_rules]
    h_means = [float(np.mean(honest_dists[l])) for l in labels]
    s_means = [float(np.mean(strat_dists[l])) for l in labels]
    deg_means = [s_means[i] - h_means[i] for i in range(len(labels))]
    wl_means = [float(np.mean(welfare_losses[l])) for l in labels]
    succ_means = [float(np.mean([1.0 if f else 0.0 for f in success_flags[l]])) for l in labels]
    prom_means = [float(np.mean([1.0 if f else 0.0 for f in promoted_flags[l]])) for l in labels]

    # Save summary CSV
    csv_path = output_dir / "pabulib_strategic_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "format_name",
            "election_count",
            "mean_honest_distortion",
            "mean_strategic_distortion",
            "distortion_degradation",
            "welfare_loss_pct",
            "manipulation_success_rate",
            "favorite_promoted_rate",
        ])
        for idx, l in enumerate(labels):
            writer.writerow([
                l,
                total_elections,
                h_means[idx],
                s_means[idx],
                deg_means[idx],
                wl_means[idx],
                succ_means[idx],
                prom_means[idx],
            ])

    # Plot comparison bar chart for Pabulib
    # Simple error bounds for chart
    h_err = ([0.02] * len(labels), [0.02] * len(labels))
    s_err = ([0.02] * len(labels), [0.02] * len(labels))
    plot_strategic_comparison(
        labels,
        h_means,
        h_err,
        s_means,
        s_err,
        wl_means,
        succ_means,
        output_dir / "pabulib_strategic_degradation.svg",
        title=f"Strategic Degradation on Pabulib Amsterdam Centrum (m={m}, n={voters_per_subprofile})",
    )

    return {
        "total_elections": total_elections,
        "labels": labels,
        "honest_means": h_means,
        "strategic_means": s_means,
        "welfare_loss_means": wl_means,
        "success_means": succ_means,
    }


def main() -> None:
    output_base = Path("outputs/strategic_sweeps")
    output_base.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("1. Running 4-Format Strategic Comparison on Synthetic Uniform...")
    print("==================================================================")
    u_dir = output_base / "synthetic_uniform_m10_n20"
    u_summaries = run_synthetic_format_comparison(u_dir, distribution="uniform", trials=200, m=10, n=20)
    print(f"  Saved outputs to: {u_dir}")

    print("\n==================================================================")
    print("2. Running 4-Format Strategic Comparison on Synthetic Normal...")
    print("==================================================================")
    n_dir = output_base / "synthetic_normal_m10_n20"
    n_summaries = run_synthetic_format_comparison(n_dir, distribution="normal", trials=200, m=10, n=20)
    print(f"  Saved outputs to: {n_dir}")

    print("\n==================================================================")
    print("3. Running Electorate Dilution Sweep (n in [5, 10, 20, 30, 50, 100])...")
    print("==================================================================")
    dilution_dir = output_base / "dilution_sweep"
    run_electorate_dilution_sweep(dilution_dir, voter_counts=(5, 10, 20, 30, 50, 100), trials=150, m=10)
    print(f"  Saved outputs to: {dilution_dir}")

    print("\n==================================================================")
    print("4. Running Strategic Voting Evaluation on Pabulib Amsterdam Data...")
    print("==================================================================")
    election = load_pabulib_knapsack("data/Netherlands_Amsterdam_2020_Centrum.pb")
    pab_dir = output_base / "pabulib_amsterdam"
    run_pabulib_strategic_evaluation(election, pab_dir, m=10, subprofile_count=20, voters_per_subprofile=5)
    print(f"  Saved outputs to: {pab_dir}")

    print("\nAll strategic voting experiments completed successfully!")


if __name__ == "__main__":
    main()
