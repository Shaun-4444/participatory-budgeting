"""Generate combined comparison SVG plots across ballot formats."""

from __future__ import annotations

import csv
from pathlib import Path
from pb_distortion.plotting import plot_distortion_summary


def load_summary(path: Path) -> dict[int, tuple[float, float, float]]:
    data: dict[int, tuple[float, float, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = int(row["project_count"])
            mean = float(row["mean_distortion"])
            lower = float(row["ci_lower"])
            upper = float(row["ci_upper"])
            data[m] = (mean, lower, upper)
    return data


def generate_comparison_plots(outputs_root: Path = Path("outputs")) -> list[Path]:
    created: list[Path] = []
    distributions = ["uniform", "normal", "pareto"]
    sigmas = [("0", "0.0"), ("0p5", "0.5")]

    formats = [
        ("Knapsack", "outputs"),
        ("Value Ranking", "outputs/value_ranking"),
        ("VFM Ranking", "outputs/vfm_ranking"),
    ]

    for dist in distributions:
        for sigma_tag, sigma_val in sigmas:
            means: dict[str, list[float]] = {}
            errors: dict[str, tuple[list[float], list[float]]] = {}
            project_counts: list[int] = []

            for fmt_label, base_dir in formats:
                pattern = f"{dist}_sigma_{sigma_tag}_seed_42"
                candidate_dirs = list(Path(base_dir).glob(f"{pattern}*"))
                if not candidate_dirs:
                    continue
                summary_file = candidate_dirs[0] / "summary.csv"
                if not summary_file.exists():
                    continue

                data = load_summary(summary_file)
                if not project_counts:
                    project_counts = sorted(data.keys())

                means[fmt_label] = [data[m][0] for m in project_counts]
                errors[fmt_label] = (
                    [data[m][0] - data[m][1] for m in project_counts],
                    [data[m][2] - data[m][0] for m in project_counts],
                )

            if len(means) >= 2:
                out_dir = outputs_root / "comparisons"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"compare_{dist}_sigma_{sigma_tag}.svg"
                plot_distortion_summary(
                    project_counts,
                    means,
                    errors,
                    out_path,
                    title=f"Distortion Comparison: {dist.capitalize()} Distribution (\u03c3 = {sigma_val})",
                )
                created.append(out_path)

    return created


if __name__ == "__main__":
    plots = generate_comparison_plots()
    for p in plots:
        print(f"Generated comparison plot: {p}")

