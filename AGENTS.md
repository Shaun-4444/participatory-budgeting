# Instructions for Coding Agents

## Project purpose

This repository is a game-theory course project about preference elicitation and distortion in participatory budgeting. The implemented baseline has synthetic voters with additive utilities, a knapsack ballot, greedy approval aggregation, exact integer-cost welfare optimization, selectable utility distributions, and lognormal perceived-value noise.

Read `README.md` and `docs/codebase_guide.md` for the complete experiment description and file map before making broad changes.

## Current experiment semantics

- `PBInstance` stores project IDs, integer costs, and a fixed budget.
- True voter utilities live in an `n × m` `UtilityProfile`. They are the evaluation ground truth.
- Perceived utilities are generated separately. Every ballot format receives perceived utilities; welfare and the social optimum always use true utilities.
- Noise uses `true * exp(sigma * epsilon - sigma**2 / 2)`. Sigma is constant within one run. Sigma zero must reproduce the baseline exactly.
- The reported distortion is optimal feasible welfare divided by chosen-outcome welfare for one sampled election. It is not the paper’s formal worst-case distortion.
- The welfare optimum and knapsack ballots use exact dynamic programming for integer costs. Do not replace the social optimum with a greedy approximation in the current experiment.
- The current aggregation rule counts ballot appearances, processes projects by descending approval count with project-ID tie-breaking, and skips non-fitting projects while continuing the scan.

Preserve these semantics unless the human explicitly asks to change them. In particular, an aggregation rule must never inspect hidden true utilities.

## Architecture and extension rules

- Keep ballot-specific behavior in `src/pb_distortion/formats/` and `src/pb_distortion/rules/`.
- A format implements the `ElicitationFormat` interface in `protocols.py`: `name` and `elicit(utilities, instance)`.
- Its aggregation rule implements `AggregationRule`: `name` and `aggregate(ballots, instance) -> Outcome`.
- Use the `PBInstance`, `UtilityProfile`, and `Outcome` models. Keep project ordering explicit and validate project IDs, dimensions, and budget feasibility.
- Keep `simulation.py` and `sweeps.py` format-neutral. Pass formats and rules into them; do not add `if ballot_format == ...` branches to the simulation logic.
- The CLI currently wires the knapsack implementation directly. When adding a format, add a user-facing CLI selection and wire its format/rule pair there.
- Write focused tests for ballot construction, rule behavior, budget feasibility, deterministic tie-breaking, and simulation reproducibility. Run `uv run pytest` after code changes.
- Add dependencies only when necessary. Runtime dependencies are NumPy and Matplotlib; pytest is a development dependency. If dependencies change, update `pyproject.toml` and `uv.lock` together.

## Reproducibility and output handling

- Use NumPy `Generator` instances passed into generation functions; do not use module-global random state.
- Preserve child-seed derivation and the paired-seed behavior across noise levels.
- Keep generated CSV, JSON, plots, and other experiment outputs under `outputs/`. `.gitignore` excludes them; do not force-add them unless the human requests it.
- New run folders must not overwrite previous runs. Reuse the CLI’s unique-directory allocation behavior.
- Record settings, selected format/rule, noise level, seeds, and metric definitions in run metadata.
- Check that all trial distortions and confidence intervals are at least 1 when using the exact optimum.

## Team changes

When asked to implement a new ballot:

1. Add its format module and its aggregation rule module.
2. Add tests that define the ballot semantics and verify rule constraints.
3. Add CLI selection for the new pair.
4. Update README experiment commands and the codebase guide.
5. Keep shared interfaces stable and compare formats using the same utility draws, noise, project counts, and aggregation-independent welfare benchmark.

Planned ballot formats are value ranking, value-for-money ranking, and threshold approval. Pabulib parsing and strategic voting are not currently implemented; do not claim real-data or strategic results exist unless code and outputs are added.
