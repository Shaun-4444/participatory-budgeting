"""Validation-based threshold selection for Threshold Approval preference elicitation.

Implements the holdout/validation set pipeline described in Benadè et al. (2021)
(Section 5 and Appendix C.2). Rather than fixing an arbitrary or hardcoded threshold,
the threshold is learned by evaluating candidate thresholds over a holdout validation
set and choosing the threshold that minimizes distortion on the validation data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pb_distortion.formats.threshold_approval import ThresholdApprovalBallotFormat
from pb_distortion.models import PBInstance
from pb_distortion.pabulib_data import PabulibElection
from pb_distortion.pabulib_sweep import (
    PabulibSweepSettings,
    run_pabulib_project_count_sweep,
)
from pb_distortion.rules.threshold_approval import ThresholdApprovalRule
from pb_distortion.simulation import SimulationSettings, run_simulation


DEFAULT_CANDIDATE_THRESHOLDS: tuple[float, ...] = (
    0.02,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
)


@dataclass(frozen=True, slots=True)
class ThresholdValidationResult:
    """Outcome of holdout validation search for optimal threshold."""

    selected_threshold: float
    best_validation_distortion: float
    validation_distortions: dict[float, float]
    candidate_thresholds: tuple[float, ...]
    validation_sample_size: int
    per_project_count_thresholds: dict[int, float] | None = None
    target_project_count: int | None = None

    def to_metadata_dict(self) -> dict[str, Any]:
        """Convert result to JSON-serializable dictionary for run metadata."""
        return {
            "selected_threshold": self.selected_threshold,
            "best_validation_distortion": self.best_validation_distortion,
            "candidate_thresholds": list(self.candidate_thresholds),
            "validation_distortions": {
                f"{t:.4f}": round(d, 6) for t, d in self.validation_distortions.items()
            },
            "validation_sample_size": self.validation_sample_size,
            "per_project_count_thresholds": (
                {str(k): v for k, v in self.per_project_count_thresholds.items()}
                if self.per_project_count_thresholds is not None
                else None
            ),
            "target_project_count": self.target_project_count,
        }


def learn_threshold_synthetic(
    settings: SimulationSettings,
    *,
    candidate_thresholds: Sequence[float] = DEFAULT_CANDIDATE_THRESHOLDS,
    validation_trials: int = 50,
) -> ThresholdValidationResult:
    """Learn the distortion-minimizing threshold on synthetic holdout trials.

    Parameters
    ----------
    settings : SimulationSettings
        The simulation settings of the target experiment.
    candidate_thresholds : Sequence[float]
        Grid of candidate threshold values to test.
    validation_trials : int
        Number of independent validation trials to run per candidate threshold.
    """
    if validation_trials < 2:
        raise ValueError("At least two validation trials are required.")
    candidates = tuple(candidate_thresholds)
    if not candidates:
        raise ValueError("At least one candidate threshold must be provided.")

    val_seed = settings.master_seed + 999_999
    val_settings = SimulationSettings(
        trials=validation_trials,
        master_seed=val_seed,
        project_count=settings.project_count,
        voter_count=settings.voter_count,
        cost_min=settings.cost_min,
        cost_max=settings.cost_max,
        budget_fraction=settings.budget_fraction,
        utility_minimum=settings.utility_minimum,
        utility_maximum=settings.utility_maximum,
        utility_distribution=settings.utility_distribution,
        normal_mean=settings.normal_mean,
        normal_std=settings.normal_std,
        pareto_shape=settings.pareto_shape,
        noise_sigma=settings.noise_sigma,
    )

    validation_distortions: dict[float, float] = {}
    for t in candidates:
        fmt = ThresholdApprovalBallotFormat(name=f"t_{t:.4f}", threshold=t)
        rule = ThresholdApprovalRule()
        results = run_simulation(val_settings, fmt, rule)
        distortions = [r.distortion for r in results]
        validation_distortions[t] = float(np.mean(distortions))

    best_threshold = min(validation_distortions, key=validation_distortions.get)  # type: ignore[arg-type]
    best_distortion = validation_distortions[best_threshold]

    return ThresholdValidationResult(
        selected_threshold=best_threshold,
        best_validation_distortion=best_distortion,
        validation_distortions=validation_distortions,
        candidate_thresholds=candidates,
        validation_sample_size=validation_trials,
        target_project_count=settings.project_count,
    )


def learn_threshold_pabulib(
    election: PabulibElection,
    settings: PabulibSweepSettings,
    *,
    candidate_thresholds: Sequence[float] = DEFAULT_CANDIDATE_THRESHOLDS,
    validation_subprofiles: int = 5,
    target_project_count: int | None = None,
    per_project_count: bool = False,
) -> ThresholdValidationResult:
    """Learn the optimal approval threshold on a Pabulib holdout/validation set.

    Parameters
    ----------
    election : PabulibElection
        The parsed Pabulib election.
    settings : PabulibSweepSettings
        The sweep settings for the real election experiment.
    candidate_thresholds : Sequence[float]
        Candidate thresholds to evaluate.
    validation_subprofiles : int
        Number of independent voter subprofiles sampled for validation.
    target_project_count : int | None
        If not per_project_count, which project count to use for validation.
        Defaults to 10 if present, or the closest project count in settings.
    per_project_count : bool
        If True, selects an optimal threshold for each project count in settings.
    """
    if validation_subprofiles < 2:
        raise ValueError("At least two validation subprofiles are required.")
    candidates = tuple(candidate_thresholds)
    if not candidates:
        raise ValueError("At least one candidate threshold must be provided.")

    if target_project_count is None and not per_project_count:
        # Default to 10 alternatives (as in Boston / Benadè et al.) or closest
        target_project_count = min(
            settings.project_counts,
            key=lambda c: abs(c - 10),
        )

    val_counts = (
        settings.project_counts if per_project_count else (target_project_count,)
    )

    val_seed = settings.master_seed + 999_999
    val_settings = PabulibSweepSettings(
        project_counts=val_counts,  # type: ignore[arg-type]
        subprofile_count=validation_subprofiles,
        voters_per_subprofile=settings.voters_per_subprofile,
        utility_profiles_per_subprofile=settings.utility_profiles_per_subprofile,
        utility_witness_mix=settings.utility_witness_mix,
        master_seed=val_seed,
        bootstrap_resamples=settings.bootstrap_resamples,
        confidence_level=settings.confidence_level,
        max_utility_cuts=settings.max_utility_cuts,
    )

    format_rules = [
        (
            ThresholdApprovalBallotFormat(
                name=f"t_{t:.4f}",
                threshold=t,
                relative=True,
            ),
            ThresholdApprovalRule(),
        )
        for t in candidates
    ]

    _, summaries, _ = run_pabulib_project_count_sweep(
        election,
        val_settings,
        format_rules,
    )

    validation_distortions: dict[float, float] = {}
    per_m_distortions: dict[int, dict[float, float]] = {m: {} for m in val_counts}

    for s in summaries:
        # Format name is e.g. "t_0.1000 + threshold_approval"
        raw_name = s.format_name.split(" + ")[0]
        thresh_val = float(raw_name.replace("t_", ""))
        per_m_distortions[s.project_count][thresh_val] = s.mean_distortion

    # Compute overall mean distortion across the evaluated project counts for each threshold
    for t in candidates:
        validation_distortions[t] = float(
            np.mean([per_m_distortions[m][t] for m in val_counts])
        )

    best_threshold = min(validation_distortions, key=validation_distortions.get)  # type: ignore[arg-type]
    best_distortion = validation_distortions[best_threshold]

    per_m_best: dict[int, float] | None = None
    if per_project_count:
        per_m_best = {
            m: min(dist_dict, key=dist_dict.get)  # type: ignore[arg-type]
            for m, dist_dict in per_m_distortions.items()
        }

    return ThresholdValidationResult(
        selected_threshold=best_threshold,
        best_validation_distortion=best_distortion,
        validation_distortions=validation_distortions,
        candidate_thresholds=candidates,
        validation_sample_size=validation_subprofiles,
        per_project_count_thresholds=per_m_best,
        target_project_count=target_project_count,
    )
