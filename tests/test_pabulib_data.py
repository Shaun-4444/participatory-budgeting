from __future__ import annotations

from pathlib import Path

import pytest

from pb_distortion.pabulib_data import load_pabulib_knapsack, make_scaled_subinstance


def _write_pb(path: Path, *, costs: str = "1; 2\n2; 3\n3; 5", votes: str = "v1; 1,2\nv2; 3") -> Path:
    path.write_text(
        "META\n"
        "key;value\n"
        "description;Tiny test\n"
        "num_projects;3\n"
        "num_votes;2\n"
        "budget;5\n"
        "max_sum_cost;5\n"
        "vote_type;approval\n"
        "currency;EUR\n"
        "PROJECTS\n"
        "project_id;cost\n"
        f"{costs}\n"
        "VOTES\n"
        "voter_id;vote\n"
        f"{votes}\n",
        encoding="utf-8",
    )
    return path


def test_load_pabulib_knapsack_and_validate_real_ballots(tmp_path: Path) -> None:
    source = _write_pb(tmp_path / "tiny.pb")

    election = load_pabulib_knapsack(source)

    assert election.project_count == 3
    assert election.voter_count == 2
    assert election.instance.project_ids == ("1", "2", "3")
    assert election.instance.costs == (2, 3, 5)
    assert election.instance.budget == 5
    assert election.ballots == (frozenset({"1", "2"}), frozenset({"3"}))
    assert election.metadata["description"] == "Tiny test"


def test_load_pabulib_rejects_ballots_over_max_sum_cost(tmp_path: Path) -> None:
    source = _write_pb(tmp_path / "overbudget.pb", votes="v1; 1,2,3\nv2; 3")

    with pytest.raises(ValueError, match="exceeding max_sum_cost"):
        load_pabulib_knapsack(source)


def test_load_pabulib_exactly_scales_decimal_costs(tmp_path: Path) -> None:
    source = tmp_path / "decimal.pb"
    source.write_text(
        "META\nkey;value\nnum_projects;2\nnum_votes;1\nbudget;2.50\n"
        "max_sum_cost;2.50\nvote_type;approval\nPROJECTS\nproject_id;cost\n"
        "a;1.25\nb;2.50\nVOTES\nvoter_id;vote\nv1;a\n",
        encoding="utf-8",
    )

    election = load_pabulib_knapsack(source)

    assert election.decimal_multiplier == 100
    assert election.source_costs == (125, 250)
    assert election.source_budget == 250
    assert election.instance.costs == (1, 2)
    assert election.instance.budget == 2
    assert str(election.currency_units_per_cost_unit) == "1.25"


def test_make_scaled_subinstance_preserves_budget_share(tmp_path: Path) -> None:
    election = load_pabulib_knapsack(_write_pb(tmp_path / "tiny.pb"))

    instance, divisor, source_budget = make_scaled_subinstance(election, (0, 1))

    assert source_budget == 3
    assert divisor == 1
    assert instance.project_ids == ("1", "2")
    assert instance.costs == (2, 3)
    assert instance.budget == 3
