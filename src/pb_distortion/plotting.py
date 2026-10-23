"""Plot helpers for the later project-count and noise sweeps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path


def _write_svg_plot(
    project_counts: Sequence[int],
    means: Mapping[str, Sequence[float]],
    errors: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    destination: Path,
    title: str,
) -> None:
    """Write a self-contained vector plot without an optional plotting runtime."""

    width, height = 960, 600
    left, right, top, bottom = 92, 220, 82, 92
    plot_right = width - right
    plot_bottom = height - bottom
    plot_width = plot_right - left
    plot_height = plot_bottom - top

    lower_values = [
        mean - lower_error
        for name, values in means.items()
        for mean, lower_error in zip(values, errors[name][0], strict=True)
    ]
    upper_values = [
        mean + upper_error
        for name, values in means.items()
        for mean, upper_error in zip(values, errors[name][1], strict=True)
    ]
    data_min = min(lower_values)
    data_max = max(upper_values)
    y_min = 1.0 if data_min >= 1.0 else data_min - 0.05 * max(data_max - data_min, 0.1)
    y_max = data_max + 0.10 * max(data_max - y_min, 0.05)
    if y_max <= y_min:
        y_max = y_min + 0.1
    x_min = min(project_counts)
    x_max = max(project_counts)
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    x_position = lambda value: left + (value - x_min) / (x_max - x_min) * plot_width
    y_position = lambda value: plot_bottom - (value - y_min) / (y_max - y_min) * plot_height
    colors = ("#1769aa", "#d1495b", "#2a9d6f", "#8e63b0", "#e18f00")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202124}.grid{stroke:#d9dde3;stroke-width:1}.axis{stroke:#30343b;stroke-width:1.4}</style>',
        f'<text x="{width / 2}" y="38" text-anchor="middle" font-size="21" font-weight="600">{escape(title)}</text>',
    ]

    tick_count = 5
    for index in range(tick_count + 1):
        value = y_min + (y_max - y_min) * index / tick_count
        y = y_position(value)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}"/>')
        tick_label = f"{value:.4f}" if y_max - y_min < 0.1 else f"{value:.2f}"
        parts.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-size="13">{tick_label}</text>')

    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{plot_bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}"/>',
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 34}" text-anchor="middle" font-size="16">Number of projects</text>',
            f'<text x="25" y="{top + plot_height / 2:.2f}" text-anchor="middle" font-size="16" transform="rotate(-90 25 {top + plot_height / 2:.2f})">Distortion (lower is better)</text>',
        ]
    )

    for count in project_counts:
        x = x_position(count)
        parts.append(f'<line class="axis" x1="{x:.2f}" y1="{plot_bottom}" x2="{x:.2f}" y2="{plot_bottom + 5}"/>')
        parts.append(f'<text x="{x:.2f}" y="{plot_bottom + 24}" text-anchor="middle" font-size="12">{count}</text>')

    for format_index, (format_name, values) in enumerate(means.items()):
        color = colors[format_index % len(colors)]
        lower_errors, upper_errors = errors[format_name]
        points = [
            (x_position(count), y_position(value))
            for count, value in zip(project_counts, values, strict=True)
        ]
        point_string = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(f'<polyline points="{point_string}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for count, value, lower_error, upper_error, (x, y) in zip(
            project_counts, values, lower_errors, upper_errors, points, strict=True
        ):
            y_low = y_position(value - lower_error)
            y_high = y_position(value + upper_error)
            parts.append(f'<line x1="{x:.2f}" y1="{y_low:.2f}" x2="{x:.2f}" y2="{y_high:.2f}" stroke="{color}" stroke-width="1.6"/>')
            parts.append(f'<line x1="{x - 5:.2f}" y1="{y_low:.2f}" x2="{x + 5:.2f}" y2="{y_low:.2f}" stroke="{color}" stroke-width="1.6"/>')
            parts.append(f'<line x1="{x - 5:.2f}" y1="{y_high:.2f}" x2="{x + 5:.2f}" y2="{y_high:.2f}" stroke="{color}" stroke-width="1.6"/>')
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{color}"/>')

        legend_y = top + 22 * format_index
        legend_x = plot_right + 24
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 25}" y2="{legend_y}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<circle cx="{legend_x + 12.5}" cy="{legend_y}" r="3.5" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 34}" y="{legend_y + 5}" font-size="13">{escape(format_name)}</text>')

    parts.append("</svg>")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(parts), encoding="utf-8")


def plot_distortion_summary(
    project_counts: Sequence[int],
    means: Mapping[str, Sequence[float]],
    errors: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    output_path: str | Path,
    *,
    title: str = "Distortion by project count",
) -> Path:
    """Plot means with caller-provided error bars for each ballot format."""

    if set(means) != set(errors):
        raise ValueError("Means and error-bar mappings must have matching format names.")
    if any(len(values) != len(project_counts) for values in means.values()):
        raise ValueError("Each format needs one mean per project count.")
    if any(
        len(lower) != len(project_counts) or len(upper) != len(project_counts)
        for lower, upper in errors.values()
    ):
        raise ValueError("Each format needs lower and upper errors per project count.")
    for format_name, values in means.items():
        lower_errors, upper_errors = errors[format_name]
        if any(value < 1.0 - 1e-12 for value in values):
            raise ValueError(f"Distortion means for {format_name} cannot be below 1.")
        if any(
            value - lower_error < 1.0 - 1e-12
            for value, lower_error in zip(values, lower_errors, strict=True)
        ):
            raise ValueError(f"Distortion confidence intervals for {format_name} cannot be below 1.")
        if any(error < 0 for error in (*lower_errors, *upper_errors)):
            raise ValueError(f"Distortion error bars for {format_name} must be non-negative.")

    destination = Path(output_path)
    if destination.suffix.lower() == ".svg":
        _write_svg_plot(project_counts, means, errors, destination, title)
        return destination

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    for format_name, values in means.items():
        axis.errorbar(
            project_counts,
            values,
            yerr=errors[format_name],
            marker="o",
            capsize=3,
            label=format_name,
        )
    axis.set_xlabel("Number of projects")
    axis.set_ylabel("Distortion (lower is better)")
    axis.set_ylim(bottom=1.0)
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination
