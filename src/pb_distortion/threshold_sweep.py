"""Run extended threshold sweeps for synthetic and real Pabulib datasets.

Mirroring and extending Figure 4 of Benadè et al. (2021) to explore higher thresholds (up to 0.90)
and compare threshold approval directly against the knapsack benchmark.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from pb_distortion.formats.knapsack import KnapsackBallotFormat
from pb_distortion.formats.threshold_approval import ThresholdApprovalBallotFormat
from pb_distortion.pabulib_data import load_pabulib_knapsack
from pb_distortion.pabulib_sweep import (
    PabulibSweepSettings,
    run_pabulib_project_count_sweep,
)
from pb_distortion.plotting import plot_distortion_summary
from pb_distortion.rules.greedy_approval import GreedyApprovalRule
from pb_distortion.rules.threshold_approval import ThresholdApprovalRule
from pb_distortion.simulation import SimulationSettings, run_simulation


def run_synthetic_benchmark(
    m: int = 10,
    distribution: str = "uniform",
    trials: int = 200,
    master_seed: int = 42,
    bootstrap_resamples: int = 2000,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    settings = SimulationSettings(
        trials=trials,
        master_seed=master_seed,
        project_count=m,
        voter_count=50,
        cost_min=1,
        cost_max=10,
        budget_fraction=0.33,
        utility_distribution=distribution,
        noise_sigma=0.0,
    )
    results = run_simulation(settings, KnapsackBallotFormat(), GreedyApprovalRule())
    values = np.asarray([r.distortion for r in results], dtype=np.float64)
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, 9999]))
    indices = rng.integers(0, len(values), size=(bootstrap_resamples, len(values)))
    boot_means = values[indices].mean(axis=1)
    alpha = (1 - confidence_level) / 2
    lower, upper = np.quantile(boot_means, [alpha, 1 - alpha])
    return float(values.mean()), float(lower), float(upper)


def run_synthetic_threshold_sweep(
    thresholds: list[float],
    m: int = 10,
    distribution: str = "uniform",
    trials: int = 200,
    master_seed: int = 42,
    bootstrap_resamples: int = 2000,
    confidence_level: float = 0.95,
) -> tuple[list[float], list[float], list[float]]:
    means = []
    ci_lowers = []
    ci_uppers = []

    for t in thresholds:
        settings = SimulationSettings(
            trials=trials,
            master_seed=master_seed,
            project_count=m,
            voter_count=50,
            cost_min=1,
            cost_max=10,
            budget_fraction=0.33,
            utility_distribution=distribution,
            noise_sigma=0.0,
        )
        fmt = ThresholdApprovalBallotFormat(threshold=t)
        rule = ThresholdApprovalRule()
        results = run_simulation(settings, fmt, rule)
        values = np.asarray([r.distortion for r in results], dtype=np.float64)

        mean_val = float(values.mean())
        rng = np.random.default_rng(np.random.SeedSequence([master_seed, int(t * 1000)]))
        indices = rng.integers(0, len(values), size=(bootstrap_resamples, len(values)))
        boot_means = values[indices].mean(axis=1)
        alpha = (1 - confidence_level) / 2
        lower, upper = np.quantile(boot_means, [alpha, 1 - alpha])

        means.append(mean_val)
        ci_lowers.append(float(lower))
        ci_uppers.append(float(upper))

    return means, ci_lowers, ci_uppers


def main() -> None:
    output_base = Path("outputs/threshold_sweeps")
    output_base.mkdir(parents=True, exist_ok=True)

    # Extended threshold grid up to 0.90
    thresholds = [
        0.02, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20,
        0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60,
        0.70, 0.80, 0.90,
    ]
    x_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    m = 10  # Canonical project count from paper

    # 1. Synthetic Uniform
    print("==================================================================")
    print("1. Running Synthetic Uniform Sweep (m=10, 200 trials)...")
    print("==================================================================")
    synth_u_means, synth_u_lowers, synth_u_uppers = run_synthetic_threshold_sweep(
        thresholds, m=m, distribution="uniform"
    )
    kn_u_mean, kn_u_lower, kn_u_upper = run_synthetic_benchmark(m=m, distribution="uniform")
    print(f"Uniform Knapsack benchmark: {kn_u_mean:.4f} [{kn_u_lower:.4f}, {kn_u_upper:.4f}]")

    u_dir = output_base / "synthetic_uniform_m10"
    u_dir.mkdir(parents=True, exist_ok=True)
    with (u_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "mean_distortion", "ci_lower", "ci_upper"])
        for t, mean, low, up in zip(thresholds, synth_u_means, synth_u_lowers, synth_u_uppers, strict=True):
            writer.writerow([t, mean, low, up])

    plot_distortion_summary(
        thresholds,
        {
            "Threshold Approval (Uniform, m=10)": synth_u_means,
            "Knapsack Benchmark (Uniform, m=10)": [kn_u_mean] * len(thresholds),
        },
        {
            "Threshold Approval (Uniform, m=10)": (
                [m_val - l for m_val, l in zip(synth_u_means, synth_u_lowers, strict=True)],
                [u - m_val for m_val, u in zip(synth_u_means, synth_u_uppers, strict=True)],
            ),
            "Knapsack Benchmark (Uniform, m=10)": (
                [kn_u_mean - kn_u_lower] * len(thresholds),
                [kn_u_upper - kn_u_mean] * len(thresholds),
            ),
        },
        u_dir / "distortion_by_threshold.svg",
        title=f"Synthetic Uniform: Distortion vs. Threshold (m={m}; 95% bootstrap CI)",
        x_label="Threshold value (t)",
        x_ticks=x_ticks,
    )
    print(f"Saved: {u_dir / 'distortion_by_threshold.svg'}")

    # 2. Synthetic Normal
    print("\n==================================================================")
    print("2. Running Synthetic Normal Sweep (m=10, 200 trials)...")
    print("==================================================================")
    synth_n_means, synth_n_lowers, synth_n_uppers = run_synthetic_threshold_sweep(
        thresholds, m=m, distribution="normal"
    )
    kn_n_mean, kn_n_lower, kn_n_upper = run_synthetic_benchmark(m=m, distribution="normal")
    print(f"Normal Knapsack benchmark: {kn_n_mean:.4f} [{kn_n_lower:.4f}, {kn_n_upper:.4f}]")

    n_dir = output_base / "synthetic_normal_m10"
    n_dir.mkdir(parents=True, exist_ok=True)
    with (n_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "mean_distortion", "ci_lower", "ci_upper"])
        for t, mean, low, up in zip(thresholds, synth_n_means, synth_n_lowers, synth_n_uppers, strict=True):
            writer.writerow([t, mean, low, up])

    plot_distortion_summary(
        thresholds,
        {
            "Threshold Approval (Normal, m=10)": synth_n_means,
            "Knapsack Benchmark (Normal, m=10)": [kn_n_mean] * len(thresholds),
        },
        {
            "Threshold Approval (Normal, m=10)": (
                [m_val - l for m_val, l in zip(synth_n_means, synth_n_lowers, strict=True)],
                [u - m_val for m_val, u in zip(synth_n_means, synth_n_uppers, strict=True)],
            ),
            "Knapsack Benchmark (Normal, m=10)": (
                [kn_n_mean - kn_n_lower] * len(thresholds),
                [kn_n_upper - kn_n_mean] * len(thresholds),
            ),
        },
        n_dir / "distortion_by_threshold.svg",
        title=f"Synthetic Normal: Distortion vs. Threshold (m={m}; 95% bootstrap CI)",
        x_label="Threshold value (t)",
        x_ticks=x_ticks,
    )
    print(f"Saved: {n_dir / 'distortion_by_threshold.svg'}")

    # 3. Real Pabulib Amsterdam Centrum 2020
    print("\n==================================================================")
    print("3. Running Real Pabulib Amsterdam Extended Threshold Sweep (m=10)...")
    print("==================================================================")
    election = load_pabulib_knapsack("data/Netherlands_Amsterdam_2020_Centrum.pb")
    pabulib_settings = PabulibSweepSettings(
        project_counts=(m,),
        subprofile_count=20,
        voters_per_subprofile=5,
        utility_profiles_per_subprofile=5,
        utility_witness_mix=0.75,
        master_seed=42,
    )

    format_rules: list[tuple[Any, Any]] = [
        (KnapsackBallotFormat(), GreedyApprovalRule()),
    ] + [
        (
            ThresholdApprovalBallotFormat(
                name=f"t_{t:.3f}",
                threshold=t,
                relative=True,
            ),
            ThresholdApprovalRule(),
        )
        for t in thresholds
    ]

    p_results, p_summaries, p_diag = run_pabulib_project_count_sweep(
        election, pabulib_settings, format_rules
    )

    summary_by_label = {s.format_name: s for s in p_summaries}
    kn_amsterdam_sum = summary_by_label["knapsack + greedy_approval"]
    kn_pab_mean = kn_amsterdam_sum.mean_distortion
    kn_pab_lower = kn_amsterdam_sum.ci_lower
    kn_pab_upper = kn_amsterdam_sum.ci_upper
    print(f"Amsterdam Knapsack benchmark: {kn_pab_mean:.4f} [{kn_pab_lower:.4f}, {kn_pab_upper:.4f}]")

    pab_means = [summary_by_label[f"t_{t:.3f} + threshold_approval"].mean_distortion for t in thresholds]
    pab_lowers = [summary_by_label[f"t_{t:.3f} + threshold_approval"].ci_lower for t in thresholds]
    pab_uppers = [summary_by_label[f"t_{t:.3f} + threshold_approval"].ci_upper for t in thresholds]

    p_dir = output_base / "pabulib_amsterdam_m10"
    p_dir.mkdir(parents=True, exist_ok=True)
    with (p_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "mean_distortion", "ci_lower", "ci_upper"])
        for t, mean, low, up in zip(thresholds, pab_means, pab_lowers, pab_uppers, strict=True):
            writer.writerow([t, mean, low, up])

    plot_distortion_summary(
        thresholds,
        {
            "Threshold Approval (Amsterdam 2020, m=10)": pab_means,
            "Knapsack Benchmark (Amsterdam 2020, m=10)": [kn_pab_mean] * len(thresholds),
        },
        {
            "Threshold Approval (Amsterdam 2020, m=10)": (
                [m_val - l for m_val, l in zip(pab_means, pab_lowers, strict=True)],
                [u - m_val for m_val, u in zip(pab_means, pab_uppers, strict=True)],
            ),
            "Knapsack Benchmark (Amsterdam 2020, m=10)": (
                [kn_pab_mean - kn_pab_lower] * len(thresholds),
                [kn_pab_upper - kn_pab_mean] * len(thresholds),
            ),
        },
        p_dir / "distortion_by_threshold.svg",
        title=f"Amsterdam Centrum 2020: Distortion vs. Threshold (m={m}; 95% cluster CI)",
        x_label="Threshold value (t)",
        x_ticks=x_ticks,
    )
    print(f"Saved: {p_dir / 'distortion_by_threshold.svg'}")

    # 4. Multi-dataset comparison plot
    print("\n==================================================================")
    print("4. Generating Multi-Dataset Comparison Plot...")
    print("==================================================================")
    means_dict = {
        "Synthetic Uniform (m=10)": synth_u_means,
        "Synthetic Normal (m=10)": synth_n_means,
        "Amsterdam Centrum (m=10)": pab_means,
    }
    errors_dict = {
        "Synthetic Uniform (m=10)": (
            [m_val - l for m_val, l in zip(synth_u_means, synth_u_lowers, strict=True)],
            [u - m_val for m_val, u in zip(synth_u_means, synth_u_uppers, strict=True)],
        ),
        "Synthetic Normal (m=10)": (
            [m_val - l for m_val, l in zip(synth_n_means, synth_n_lowers, strict=True)],
            [u - m_val for m_val, u in zip(synth_n_means, synth_n_uppers, strict=True)],
        ),
        "Amsterdam Centrum (m=10)": (
            [m_val - l for m_val, l in zip(pab_means, pab_lowers, strict=True)],
            [u - m_val for m_val, u in zip(pab_means, pab_uppers, strict=True)],
        ),
    }

    comp_plot = output_base / "comparison_distortion_by_threshold.svg"
    plot_distortion_summary(
        thresholds,
        means_dict,
        errors_dict,
        comp_plot,
        title="Distortion vs. Approval Threshold t (Synthetic & Real Amsterdam, m=10)",
        x_label="Threshold value (t)",
        x_ticks=x_ticks,
    )
    print(f"Saved comparison plot: {comp_plot}")

    print("\nAll extended threshold experiments completed successfully!")


if __name__ == "__main__":
    main()
