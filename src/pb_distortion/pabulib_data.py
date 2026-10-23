"""Load the approval-knapsack subset of the Pabulib ``.pb`` format.

This loader intentionally accepts only elections whose individual approval
ballots have a finite ``max_sum_cost`` equal to the election budget. That is
the semantics needed to interpret each recorded ballot as a knapsack choice.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pb_distortion.formats.knapsack import KnapsackBallot
from pb_distortion.models import PBInstance


@dataclass(frozen=True, slots=True)
class PabulibElection:
    """A validated Pabulib approval-knapsack election.

    ``costs`` and ``budget`` are integers in the smallest currency unit found
    in the source file, divided by ``cost_gcd`` for efficient exact dynamic
    programming. ``currency_units_per_cost_unit`` converts these model units
    back to the currency amounts recorded in the ``.pb`` file.
    """

    source_path: Path
    metadata: dict[str, str]
    project_rows: tuple[dict[str, str], ...]
    voter_ids: tuple[str, ...]
    instance: PBInstance
    ballots: tuple[KnapsackBallot, ...]
    source_costs: tuple[int, ...]
    source_budget: int
    decimal_multiplier: int
    cost_gcd: int

    @property
    def project_count(self) -> int:
        return self.instance.project_count

    @property
    def voter_count(self) -> int:
        return len(self.ballots)

    @property
    def total_source_cost(self) -> int:
        return sum(self.source_costs)

    @property
    def currency_units_per_cost_unit(self) -> Decimal:
        return Decimal(self.cost_gcd) / Decimal(self.decimal_multiplier)


def _decimal_integer_units(values: list[str]) -> tuple[list[int], int]:
    """Convert finite decimal strings to exact integers with shared precision."""

    decimals: list[Decimal] = []
    for value in values:
        try:
            parsed = Decimal(value.strip())
        except InvalidOperation as error:
            raise ValueError(f"Expected a numeric cost or budget, got {value!r}.") from error
        if not parsed.is_finite() or parsed < 0:
            raise ValueError(f"Costs and budgets must be finite and non-negative, got {value!r}.")
        decimals.append(parsed)

    decimal_places = max((max(0, -value.as_tuple().exponent) for value in decimals), default=0)
    multiplier = 10**decimal_places
    integers = [int(value * multiplier) for value in decimals]
    return integers, multiplier


def _read_pb_tables(path: Path) -> dict[str, tuple[list[str], list[list[str]]]]:
    tables: dict[str, tuple[list[str], list[list[str]]]] = {}
    active_section: str | None = None
    headers: list[str] | None = None

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for raw_row in csv.reader(source, delimiter=";"):
            row = [cell.strip() for cell in raw_row]
            if not row or not any(row):
                continue
            title = row[0].upper()
            if title in {"META", "PROJECTS", "VOTES"} and len(row) == 1:
                active_section = title
                headers = None
                continue
            if active_section is None:
                continue
            if headers is None:
                headers = row
                tables[active_section] = (headers, [])
                continue
            if len(row) != len(headers):
                raise ValueError(
                    f"Malformed {active_section} row in {path}: expected "
                    f"{len(headers)} fields, found {len(row)}."
                )
            tables[active_section][1].append(row)

    missing = {"META", "PROJECTS", "VOTES"} - tables.keys()
    if missing:
        raise ValueError(f"Pabulib file is missing sections: {sorted(missing)}.")
    return tables


def load_pabulib_knapsack(path: str | Path) -> PabulibElection:
    """Load and validate an approval election with budget-feasible ballots."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    tables = _read_pb_tables(source_path)

    meta_headers, meta_rows = tables["META"]
    if meta_headers[:2] != ["key", "value"]:
        raise ValueError("The META section must start with the 'key;value' header.")
    metadata: dict[str, str] = {}
    for row in meta_rows:
        key, value = row[0], row[1]
        if key in metadata:
            raise ValueError(f"Duplicate META key {key!r}.")
        metadata[key] = value

    if metadata.get("vote_type", "").lower() != "approval":
        raise ValueError(
            "This experiment requires Pabulib vote_type=approval with a finite "
            "max_sum_cost matching the election budget."
        )
    required_meta = {"budget", "max_sum_cost"}
    missing_meta = required_meta - metadata.keys()
    if missing_meta:
        raise ValueError(f"Missing required META fields: {sorted(missing_meta)}.")

    project_headers, project_rows_raw = tables["PROJECTS"]
    vote_headers, vote_rows_raw = tables["VOTES"]
    if "project_id" not in project_headers or "cost" not in project_headers:
        raise ValueError("PROJECTS must contain project_id and cost columns.")
    if "voter_id" not in vote_headers or "vote" not in vote_headers:
        raise ValueError("VOTES must contain voter_id and vote columns.")

    project_rows = tuple(dict(zip(project_headers, row, strict=True)) for row in project_rows_raw)
    voter_rows = tuple(dict(zip(vote_headers, row, strict=True)) for row in vote_rows_raw)
    project_ids = tuple(row["project_id"] for row in project_rows)
    voter_ids = tuple(row["voter_id"] for row in voter_rows)
    if not project_ids or not voter_ids:
        raise ValueError("The election must contain at least one project and one voter.")
    if len(set(project_ids)) != len(project_ids):
        raise ValueError("PROJECTS contains duplicate project IDs.")
    if len(set(voter_ids)) != len(voter_ids):
        raise ValueError("VOTES contains duplicate voter IDs.")

    project_cost_text = [row["cost"] for row in project_rows]
    numeric_text = project_cost_text + [metadata["budget"], metadata["max_sum_cost"]]
    numeric_units, decimal_multiplier = _decimal_integer_units(numeric_text)
    source_costs = tuple(numeric_units[: len(project_rows)])
    source_budget = numeric_units[-2]
    ballot_cost_cap = numeric_units[-1]
    if any(cost <= 0 for cost in source_costs):
        raise ValueError("Project costs must be positive.")
    if source_budget <= 0 or ballot_cost_cap <= 0:
        raise ValueError("The budget and max_sum_cost must be positive.")
    if ballot_cost_cap != source_budget:
        raise ValueError(
            "For this knapsack experiment, META.max_sum_cost must equal META.budget; "
            f"got max_sum_cost={metadata['max_sum_cost']} and budget={metadata['budget']}."
        )

    project_cost_by_id = dict(zip(project_ids, source_costs, strict=True))
    known_projects = set(project_ids)
    ballots: list[KnapsackBallot] = []
    for row in voter_rows:
        listed = [project_id.strip() for project_id in row["vote"].split(",") if project_id.strip()]
        if len(listed) != len(set(listed)):
            raise ValueError(f"Voter {row['voter_id']} lists a project more than once.")
        unknown = set(listed) - known_projects
        if unknown:
            raise ValueError(
                f"Voter {row['voter_id']} lists unknown projects: {sorted(unknown)}."
            )
        ballot = frozenset(listed)
        ballot_cost = sum(project_cost_by_id[project_id] for project_id in ballot)
        if ballot_cost > ballot_cost_cap:
            raise ValueError(
                f"Voter {row['voter_id']}'s ballot costs {ballot_cost}, exceeding "
                f"max_sum_cost={ballot_cost_cap}."
            )
        ballots.append(ballot)

    declared_projects = metadata.get("num_projects")
    if declared_projects is not None and int(declared_projects) != len(project_ids):
        raise ValueError("META.num_projects does not match the PROJECTS section.")
    declared_votes = metadata.get("num_votes")
    if declared_votes is not None and int(declared_votes) != len(voter_ids):
        raise ValueError("META.num_votes does not match the VOTES section.")

    cost_gcd = math.gcd(source_budget, *source_costs)
    reduced_costs = tuple(cost // cost_gcd for cost in source_costs)
    reduced_budget = source_budget // cost_gcd
    instance = PBInstance(project_ids, reduced_costs, reduced_budget)

    return PabulibElection(
        source_path=source_path,
        metadata=metadata,
        project_rows=project_rows,
        voter_ids=voter_ids,
        instance=instance,
        ballots=tuple(ballots),
        source_costs=source_costs,
        source_budget=source_budget,
        decimal_multiplier=decimal_multiplier,
        cost_gcd=cost_gcd,
    )


def make_scaled_subinstance(
    election: PabulibElection,
    project_indices: tuple[int, ...],
) -> tuple[PBInstance, int, int]:
    """Create a subinstance with a proportional budget and exact cost scaling.

    The budget share is held constant at the original election's budget / total
    candidate cost ratio. The returned integers are exact after division by the
    common divisor; the second return value is the divisor in source minor units
    and the third is the budget in source minor units.
    """

    if not project_indices:
        raise ValueError("A project subinstance must contain at least one project.")
    if len(set(project_indices)) != len(project_indices):
        raise ValueError("Project indices in a subinstance must be unique.")
    if any(index < 0 or index >= election.project_count for index in project_indices):
        raise ValueError("A project index is outside the election.")

    project_ids = tuple(election.instance.project_ids[index] for index in project_indices)
    source_costs = tuple(election.source_costs[index] for index in project_indices)
    total_source_cost = sum(source_costs)
    scaled_budget = max(
        min(source_costs),
        (2 * election.source_budget * total_source_cost + election.total_source_cost)
        // (2 * election.total_source_cost),
    )
    divisor = math.gcd(scaled_budget, *source_costs)
    instance = PBInstance(
        project_ids,
        tuple(cost // divisor for cost in source_costs),
        scaled_budget // divisor,
    )
    return instance, divisor, scaled_budget
