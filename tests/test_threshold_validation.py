"""Tests for validation-based threshold selection pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pb_distortion.formats.threshold_approval import ThresholdApprovalBallotFormat
from pb_distortion.models import PBInstance
from pb_distortion.pabulib_data import load_pabulib_knapsack
from pb_distortion.pabulib_sweep import PabulibSweepSettings
from pb_distortion.simulation import SimulationSettings
from pb_distortion.threshold_validation import (
    DEFAULT_CANDIDATE_THRESHOLDS,
    learn_threshold_pabulib,
    learn_threshold_synthetic,
)


def _tiny_election(path: Path):
    path.write_text(
        "META\nkey;value\ndescription;Tiny test\nnum_projects;4\nnum_votes;4\n"
        "budget;3\nmax_sum_cost;3\nvote_type;approval\n"
        "PROJECTS\nproject_id;cost\na;1\nb;1\nc;2\nd;2\n"
        "VOTES\nvoter_id;vote\nv1;a,b\nv2;c,a\nv3;a,b\nv4;d,b\n",
        encoding="utf-8",
    )
    return load_pabulib_knapsack(path)


def test_learn_threshold_synthetic():
    settings = SimulationSettings(
        trials=5,
        master_seed=42,
        project_count=10,
        voter_count=20,
        cost_min=1,
        cost_max=10,
        budget_fraction=0.33,
        utility_distribution="uniform",
    )
    candidates = (0.05, 0.10, 0.25)
    result = learn_threshold_synthetic(
        settings,
        candidate_thresholds=candidates,
        validation_trials=10,
    )

    assert result.selected_threshold in candidates
    assert result.validation_sample_size == 10
    assert set(result.validation_distortions.keys()) == set(candidates)
    assert result.best_validation_distortion == min(result.validation_distortions.values())
    assert result.best_validation_distortion >= 1.0

    meta = result.to_metadata_dict()
    assert "selected_threshold" in meta
    assert "validation_distortions" in meta
    assert len(meta["candidate_thresholds"]) == 3


def test_learn_threshold_synthetic_validations():
    settings = SimulationSettings(
        trials=5,
        master_seed=42,
        project_count=10,
        voter_count=20,
        cost_min=1,
        cost_max=10,
        budget_fraction=0.33,
        utility_distribution="uniform",
    )
    with pytest.raises(ValueError, match="At least two validation trials"):
        learn_threshold_synthetic(settings, validation_trials=1)
    with pytest.raises(ValueError, match="At least one candidate threshold"):
        learn_threshold_synthetic(settings, candidate_thresholds=[])


def test_learn_threshold_pabulib(tmp_path: Path):
    election = _tiny_election(tmp_path / "tiny.pb")
    settings = PabulibSweepSettings(
        project_counts=(2, 3),
        subprofile_count=2,
        voters_per_subprofile=2,
        utility_profiles_per_subprofile=2,
        master_seed=42,
        bootstrap_resamples=50,
    )
    candidates = (0.10, 0.25)
    result = learn_threshold_pabulib(
        election,
        settings,
        candidate_thresholds=candidates,
        validation_subprofiles=2,
        target_project_count=2,
    )

    assert result.selected_threshold in candidates
    assert result.best_validation_distortion >= 1.0
    assert set(result.validation_distortions.keys()) == set(candidates)
    assert result.validation_sample_size == 2


def test_learn_threshold_pabulib_per_project_count(tmp_path: Path):
    election = _tiny_election(tmp_path / "tiny.pb")
    settings = PabulibSweepSettings(
        project_counts=(2, 3),
        subprofile_count=2,
        voters_per_subprofile=2,
        utility_profiles_per_subprofile=2,
        master_seed=42,
        bootstrap_resamples=50,
    )
    candidates = (0.10, 0.30)
    result = learn_threshold_pabulib(
        election,
        settings,
        candidate_thresholds=candidates,
        validation_subprofiles=2,
        per_project_count=True,
    )

    assert result.per_project_count_thresholds is not None
    assert set(result.per_project_count_thresholds.keys()) == {2, 3}
    for m, thresh in result.per_project_count_thresholds.items():
        assert thresh in candidates


def test_threshold_approval_mapping_support():
    format_ = ThresholdApprovalBallotFormat(threshold={3: 0.20, 5: 0.15})
    inst3 = PBInstance(project_ids=("a", "b", "c"), costs=(1, 1, 1), budget=2)
    inst5 = PBInstance(project_ids=("a", "b", "c", "d", "e"), costs=(1, 1, 1, 1, 1), budget=3)
    inst4 = PBInstance(project_ids=("a", "b", "c", "d"), costs=(1, 1, 1, 1), budget=2)

    utils3 = np.array([0.19, 0.20, 0.21])
    assert format_.elicit(utils3, inst3) == frozenset({"b", "c"})

    utils5 = np.array([0.14, 0.15, 0.16, 0.10, 0.10])
    assert format_.elicit(utils5, inst5) == frozenset({"b", "c"})

    # Missing project count 4 falls back to 1/m = 1/4 = 0.25
    utils4 = np.array([0.24, 0.25, 0.26, 0.10])
    assert format_.elicit(utils4, inst4) == frozenset({"b", "c"})


def test_threshold_approval_invalid_mapping():
    with pytest.raises(ValueError, match="positive integer"):
        ThresholdApprovalBallotFormat(threshold={-1: 0.1})
    with pytest.raises(ValueError, match="finite non-negative"):
        ThresholdApprovalBallotFormat(threshold={5: -0.1})
    with pytest.raises(ValueError, match="finite non-negative"):
        ThresholdApprovalBallotFormat(threshold={5: float("nan")})


def test_learn_threshold_synthetic_per_project_count():
    settings = SimulationSettings(
        trials=5,
        master_seed=42,
        project_count=10,
        voter_count=20,
        cost_min=1,
        cost_max=10,
        budget_fraction=0.33,
        utility_distribution="uniform",
    )
    project_counts = (4, 8)
    multipliers = (0.5, 1.0, 1.5)
    result = learn_threshold_synthetic(
        settings,
        project_counts=project_counts,
        candidate_multipliers=multipliers,
        validation_trials=3,
    )

    assert result.per_project_count_thresholds is not None
    assert set(result.per_project_count_thresholds.keys()) == {4, 8}
    assert result.per_project_count_distortions is not None
    assert set(result.per_project_count_distortions.keys()) == {4, 8}

    cand_4 = {round(alpha / 4, 6) for alpha in multipliers}
    assert set(result.per_project_count_distortions[4].keys()) == cand_4
    assert result.per_project_count_thresholds[4] in cand_4

    cand_8 = {round(alpha / 8, 6) for alpha in multipliers}
    assert set(result.per_project_count_distortions[8].keys()) == cand_8
    assert result.per_project_count_thresholds[8] in cand_8

    meta = result.to_metadata_dict()
    assert "per_project_count_thresholds" in meta
    assert meta["per_project_count_thresholds"]["4"] == result.per_project_count_thresholds[4]

