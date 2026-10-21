from __future__ import annotations

import numpy as np
import pytest

from pb_distortion.cli import allocate_run_output_directory, load_config, settings_from_config
from pb_distortion.generation import generate_synthetic_election, perturb_utilities


@pytest.mark.parametrize("distribution", ["normal", "pareto"])
def test_nonuniform_distributions_generate_finite_nonnegative_normalized_values(distribution):
    election = generate_synthetic_election(
        project_count=40,
        voter_count=12,
        cost_min=1,
        cost_max=10,
        budget_fraction=0.33,
        rng=np.random.default_rng(42),
        utility_distribution=distribution,
        normal_mean=0.5,
        normal_std=0.2,
        pareto_shape=1.5,
    )

    values = election.utilities.values
    assert np.isfinite(values).all()
    assert (values >= 0).all()
    assert np.allclose(values.sum(axis=1), 1.0)


def test_one_config_can_select_each_distribution():
    config = load_config("configs/project_count_sweep.toml")

    for distribution in ("uniform", "normal", "pareto"):
        settings = settings_from_config(config, utility_distribution=distribution)
        assert settings.utility_distribution == distribution
    assert settings_from_config(config, utility_distribution="normal").normal_mean == 0.5
    assert settings_from_config(config, utility_distribution="normal").normal_std == 0.2
    assert settings_from_config(config, utility_distribution="pareto").pareto_shape == 1.5


def test_each_invocation_gets_a_distinct_output_directory(tmp_path):
    first = allocate_run_output_directory(tmp_path, "normal", 0.5, 42)
    second = allocate_run_output_directory(tmp_path, "normal", 0.5, 42)

    assert first.name == "normal_sigma_0p5_seed_42"
    assert second.name == "normal_sigma_0p5_seed_42_run_2"
    assert first != second


def test_zero_noise_returns_true_values_exactly():
    values = np.array([[0.2, 0.8], [0.6, 0.4]])
    perturbed = perturb_utilities(values, 0.0, np.random.default_rng(42))

    assert np.array_equal(perturbed, values)


def test_lognormal_perturbation_matches_mean_corrected_formula():
    values = np.array([[0.2, 0.8], [0.6, 0.4]])
    sigma = 0.7
    expected_rng = np.random.default_rng(42)
    epsilon = expected_rng.standard_normal(size=values.shape)
    expected = values * np.exp(sigma * epsilon - sigma**2 / 2.0)

    actual = perturb_utilities(values, sigma, np.random.default_rng(42))

    assert np.array_equal(actual, expected)
    assert (actual > 0).all()


def test_noise_changes_perceived_values_without_changing_true_values():
    kwargs = dict(
        project_count=12,
        voter_count=8,
        cost_min=1,
        cost_max=10,
        budget_fraction=0.33,
        utility_distribution="uniform",
    )
    baseline = generate_synthetic_election(rng=np.random.default_rng(123), **kwargs)
    noisy = generate_synthetic_election(
        rng=np.random.default_rng(123),
        noise_sigma=0.8,
        **kwargs,
    )

    assert np.array_equal(baseline.utilities.values, noisy.utilities.values)
    assert np.array_equal(noisy.perceived_utilities.values, noisy.utilities.values) is False
