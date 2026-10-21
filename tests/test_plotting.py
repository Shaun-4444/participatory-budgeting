from __future__ import annotations

from pathlib import Path

import pytest

from pb_distortion.plotting import plot_distortion_summary


def test_svg_plot_starts_at_one_and_uses_precision_for_narrow_range(tmp_path: Path) -> None:
    output = tmp_path / "distortion.svg"

    plot_distortion_summary(
        [5, 10],
        {"knapsack": [1.002, 1.003]},
        {"knapsack": ([0.001, 0.001], [0.002, 0.002])},
        output,
    )

    svg = output.read_text(encoding="utf-8")
    assert ">1.0000</text>" in svg
    assert "Distortion (lower is better)" in svg


def test_plot_rejects_mean_below_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="means.*below 1"):
        plot_distortion_summary(
            [5],
            {"knapsack": [0.99]},
            {"knapsack": ([0.01], [0.01])},
            tmp_path / "distortion.svg",
        )


def test_plot_rejects_confidence_interval_below_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="confidence intervals.*below 1"):
        plot_distortion_summary(
            [5],
            {"knapsack": [1.01]},
            {"knapsack": ([0.02], [0.01])},
            tmp_path / "distortion.svg",
        )
