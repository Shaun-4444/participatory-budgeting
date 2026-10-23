# Participatory Budgeting: Preference Elicitation and Distortion

This repository is a reproducible simulation project for studying how ballot formats affect social welfare in participatory budgeting. The implemented baseline is the **knapsack ballot** with greedy approval aggregation. It supports uniform, nonnegative-truncated normal, and Pareto utility draws, plus a lognormal noise model.

The reported quantity is the **sampled-instance distortion**

\[
\frac{\text{optimal feasible welfare under the generated true utilities}}
     {\text{welfare of the outcome selected from submitted ballots}}.
\]

This is an empirical ratio on a generated election. It is not the paper’s formal worst-case distortion over all utility profiles consistent with a ballot profile.

## Setup

Use Python 3.12 and `uv`. From the repository root:

```bash
uv sync --extra dev
```

Run the test suite with:

```bash
uv run pytest
```

## Experiments you can run

### Baseline election

```bash
uv run pb-distortion --config configs/baseline.toml
```

The baseline config uses 10 projects, 50 voters, integer project costs from 1 to 10, a budget equal to the ceiling of 30% of total project cost, 200 elections, Uniform[0, 1] utilities, and zero noise.

### Project-count curve

```bash
uv run pb-distortion --config configs/project_count_sweep.toml
```

The sweep config uses 200 elections per point, 50 voters, seed 42, costs from 1 to 10, budget fraction 0.33, and project counts 5, 10, 20, 30, 40, 50, 75, 100, 150, and 200. It writes trial results, a summary with 95% percentile-bootstrap confidence intervals, metadata, and an SVG plot.

### Choose the utility distribution

The same config can be reused for all supported distributions:

```bash
uv run pb-distortion --config configs/project_count_sweep.toml --distribution uniform
uv run pb-distortion --config configs/project_count_sweep.toml --distribution normal
uv run pb-distortion --config configs/project_count_sweep.toml --distribution pareto
```

Distribution parameters are in `[utilities]` in the TOML file:

- Uniform uses `minimum = 0.0` and `maximum = 1.0`.
- Normal uses `normal_mean = 0.5` and `normal_std = 0.2`; negative draws are resampled, so utilities follow a normal distribution conditioned on nonnegative values.
- Pareto uses `pareto_shape = 1.5`, which must be greater than 1. NumPy’s standard Pareto draw is shifted to give Pareto-I support starting at 1.
- `normalize_rows = true` rescales each voter’s utility vector to sum to one. Set it to `false` to preserve the raw draw scale.

### One noise level

Set one constant σ for the full run with `--noise-sigma`:

```bash
uv run pb-distortion --config configs/project_count_sweep.toml --distribution uniform --noise-sigma 0.5
```

You can also set the default in the config’s `[noise]` section with `sigma = 0.0`. For each voter-project pair, the perceived value is

```text
perceived = true * exp(sigma * epsilon - sigma**2 / 2)
```

where `epsilon` is a standard normal draw. The correction makes the multiplicative noise factor have mean one. Voters submit ballots based on perceived utilities; optimal and chosen welfare are evaluated using true utilities. At σ = 0, perceived utilities equal true utilities exactly.

### Compare noise levels

Use `--noise-sigmas` to run a distinct project-count sweep for each listed σ and write a combined plot:

```bash
uv run pb-distortion --config configs/project_count_sweep.toml --distribution uniform --noise-sigmas 0 0.5 1 2 4
```

Each σ is constant within its sweep. The σ runs use the same trial seeds, so they share costs and true utility draws while differing in the perturbation. The output includes one subdirectory per σ, each with its own trials, summary, metadata, and plot, plus a batch summary and a combined SVG. To reproduce the extended high-noise sweep currently in `outputs/`, use:

```bash
uv run pb-distortion --config configs/project_count_sweep.toml --distribution uniform --noise-sigmas 0 0.5 1 2 4 8 12 20 32
```

High σ values are stress tests; floating-point arithmetic can underflow at extreme values. The implementation raises an error if a perturbation produces non-finite values.

### Pabulib Amsterdam real-data-informed project-count sweep

The repository includes a Pabulib loader and a model-based experiment for the
Amsterdam Centrum 2020 dataset:

```bash
uv run pb-distortion-pabulib --config configs/pabulib_amsterdam.toml
```

The loader checks that the file contains approval ballots with a finite
`max_sum_cost` equal to the election budget, then verifies every recorded
ballot is cost-feasible. For this file, the source election has 85 projects,
8,520 voters, and a €500,000 budget. The configuration samples 20 voter
subprofiles of 5 voters, with 5 utility-profile draws per subprofile, using
seed 42.

Observed knapsack ballots do not reveal cardinal utilities. For each sampled
voter, the experiment finds a nonnegative additive utility vector that sums to
one and makes the recorded bundle utility-maximizing under the original
budget. It uses `scipy.optimize.linprog` and adds violated-bundle constraints
with an exact knapsack oracle. Each random LP vertex is blended with a
Dirichlet utility vector supported on the submitted bundle (75% witness by
default). This is a documented, non-uniform model of latent utilities, not a
uniform draw from the consistent-utility region.

For each sampled utility profile, the sweep uses nested random subsets of 5,
10, 20, 30, 40, 50, 75, and 85 projects. It preserves the source election's
budget-to-total-project-cost ratio for each subset, re-elicits knapsack
ballots from the sampled utilities, and applies the existing greedy approval
rule. The social optimum is computed exactly after dividing costs and budget
by their common divisor. Error bars are 95% percentile-bootstrap intervals
over voter subprofiles, keeping a subprofile's utility draws clustered.

This experiment produces a **model-based counterfactual** project-count curve
informed by the real election. The lower project-count points are not separate
recorded Pabulib elections: they use project subsets and re-elicited ballots.
The dataset also has neighborhood-specific budget caps and an official
`greedy-custom` outcome. This experiment follows the paper's single-budget
knapsack model, so it does not enforce those neighborhood caps or reproduce
the official outcome. Its distortion is evaluated on sampled utilities, not
unobserved true utilities. Only the knapsack format is currently implemented;
additional format modules can use the same sweep runner once added.

The run writes `trials.csv`, `summary.csv`, `run.json`, and
`distortion_by_project_count.svg` in a new folder under `outputs/pabulib/`.
The configured project counts and sampling settings can be changed in
`configs/pabulib_amsterdam.toml`.

### Output location and no-overwrite behavior

Each invocation creates a new folder under the configured output directory. The folder name includes the selected distribution, σ, and seed. Repeating the same run adds a numbered suffix rather than replacing prior results. Override the base output location with:

```bash
uv run pb-distortion --config configs/project_count_sweep.toml --output outputs/my_experiments
```

Generated output files are excluded from Git by `.gitignore`.

## How the baseline works

1. Generate integer project costs and set the budget to `ceil(budget_fraction * sum(costs))`.
2. Draw each voter’s additive project utilities and optionally normalize each voter’s row.
3. Apply the configured noise to produce perceived utilities.
4. For each voter, solve an exact 0/1 knapsack problem over perceived values and submit that affordable bundle.
5. Count each project’s appearances across ballots. Process projects by descending approval count, breaking ties by project ID. Fund each project that fits, skipping projects that do not fit and continuing down the list.
6. Solve the social-welfare-maximizing feasible bundle exactly using the sum of **true** utilities across voters.
7. Divide optimal welfare by the chosen outcome’s welfare. The sweep estimates the mean ratio and a bootstrap confidence interval over elections.

## Codebase map

See [the codebase guide](docs/codebase_guide.md) for a file-by-file explanation, data flow, algorithms, and teammate instructions.

```text
configs/                    Baseline and project-count TOML settings
src/pb_distortion/
  models.py                 Shared PB instance, utility profile, and election types
  generation.py             Utility draws, normalization, and noise
  protocols.py              Interfaces for ballots and aggregation rules
  formats/knapsack.py       Knapsack ballot elicitation
  rules/greedy_approval.py  Greedy approval-count aggregation
  solvers.py                Exact integer-cost knapsack dynamic program
  metrics.py                Welfare and sampled distortion
  simulation.py             Election runner and trial output
  sweeps.py                 Project-count/noise sweeps and bootstrap intervals
  plotting.py               SVG and Matplotlib plots
  cli.py                    Config, command-line options, and output orchestration
tests/                      Solver, format, rule, simulation, distribution, and plot tests
docs/codebase_guide.md      Shareable technical walkthrough for teammates
AGENTS.md                   Instructions for coding agents working in this repository
```

## Current scope and next stages

Implemented: synthetic data, the knapsack ballot, greedy approval, exact welfare benchmarks, distribution choice, noise stress tests, synthetic project-count sweeps, confidence intervals, Pabulib approval-knapsack loading, model-based utility sampling, and a real-data-informed project-count sweep.

Planned extensions: value-ranking, value-for-money ranking, threshold approval, alternative utility-inference assumptions for Pabulib, enforcement of dataset-specific constraints where appropriate, and optional strategic voting. The current repository has only one ballot/rule pair, so the Pabulib plot currently contains a single curve.

## Team workflow

Use `main` as the integration branch. Each teammate should work on a separate feature branch and submit a pull request. A new format should add its own ballot module, aggregation rule, tests, and CLI selection. Keep the shared protocols and simulation interface stable so formats can be compared under the same generated elections, noise, welfare metric, and benchmark.
