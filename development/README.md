# Development Workspace

This directory is the only executable workspace for the Socratic Tutor POC. The repository-level `docs/` and `proposal/` directories remain outside it so implementation artifacts do not obscure dissertation material.

## Prerequisites

- Pixi 0.63 or later.
- Git.
- D2 0.7.1 for local diagram compilation. CI installs the pinned version.

Python, Node.js, pnpm, and project dependencies are supplied by Pixi.

## First Run

Start in the repository root; subsequent commands run from `development/`. The lockfile supports macOS on Apple Silicon
and Linux x86-64. Initial installation requires network access to package registries.

```bash
cd development
pixi install --locked
pixi run install-web
pixi run check
```

Run the API and web client together:

```bash
pixi run dev
```

Then open `http://127.0.0.1:5173`. Vite proxies `/api` to FastAPI on port 8000.

The offline demo offers list aliasing and the distinction between `None` and falsy values. A submitted explanation passes
through a LangGraph turn, authored evidence rules, a simple tracker and a heuristic policy. The displayed estimate is a
demonstration state; it has not been validated as a measure of human mastery. Completed turns are written to
`.local/demo-events.jsonl` and recovered when the API restarts.

The Experiment view reads the bundled `data/demo/benchmark-replay-v1.json` and `experiment-summary-v1.json` records.
Practice and recorded replay require no provider credentials after installation.

The **Experiment** view also contains an optional live illustration. It uses one development case, up to three Cerebras requests,
and two network-blocked Modal executions to show that predictions are fixed before a later result is requested. It is disabled by
default and is not research evidence. To enable it, complete `pixi run modal setup`, then set the three commented live variables in
`.env.example` within the local `.env` file. The recorded replay remains the presentation fallback.

## Focused Commands

```bash
pixi run dev-api
pixi run dev-web
pixi run test
pixi run test-web
pixi run test-e2e
pixi run lint
pixi run typecheck
pixi run config-smoke
pixi run docs-check
pixi run build-web
```

`config-smoke` composes the default Hydra configuration without running a simulation.

## Analysis Entry Points

Inspect the command options before choosing inputs and an output directory:

```bash
pixi run python scripts/benchmark_cli.py --help
pixi run python scripts/acquisition_study_canonical.py --help
pixi run python scripts/acquisition_study_canonical_analysis.py --help
pixi run python scripts/csedm_study_reproduce.py --help
```

The benchmark commands use authored manifests and recorded responses. Reproducing the external study requires its retained
response files, ratings and execution records under `artifacts/`; a source checkout alone does not contain those inputs.
Commands that generate responses or execute code through Modal also require provider access and may incur charges.

The acquisition study uses controlled simulation. Its frozen run and analysis require the plans, calibration and seal records
specified by the command. Preserve those records when reproducing a result; creating another seal is a separate run.

The canonical acquisition analysis enforces the original lockfile hash recorded in
`configs/acquisition-study/v1-canonical-primary-analysis.yaml`. Later publication dependencies changed `pixi.lock`, so
the current checkout cannot automatically reproduce that sealed analysis. Its CLI accepts `--plan` only, and its output
directory is fixed by the validated plan. Reproduce it in a separate checkout with the matching historical lockfile and
retained inputs. Keep the original analysis outputs available for comparison; a different analysis revision can change
provenance fields and trigger an immutable-output conflict. Record the restored revision and any remaining dependency
differences before reporting successful reproduction.

The retained primary analysis records code revision `5d6e76b099f8904d5a70aca5909b49d6fa0d86af`. Its Git version of
`development/pixi.lock` has SHA-256 `7d68ee21e33528dc9f3423311f461736f0151ef71f9990607e641876526c7b83`,
matching the frozen plan. The reference analysis reads 98,000 public rows. Its report content hash is
`edec72e5a8764d4df5e52c5c09a1de213efeb3e22d4c5972708ec441e70a9d34`; the report file SHA-256 is
`86ecdd05d4a43111cd7c12f0032667fc898d3d62243804ff13fcc928377ca91e`. These identify the retained result;
on 11 September 2026, the historical revision was installed in a separate local clone with `pixi install --locked` and
the primary analysis completed successfully. All six output files matched the retained originals byte for byte. Installation
used available package caches. This check covers analysis of retained public observations; simulation regeneration and
installation on another machine remain unverified.

`pixi run csedm-reproduce` requires the authorised CSEDM archive specified by its plan. Dataset access and sharing conditions
must be confirmed before distributing inputs. See [the report build instructions](../dissertation/README.md) for building
the PDF from its bundled figures and tables.

Run the complete CSEDM analysis from `development/` with:

```bash
pixi run csedm-reproduce
```

This rebuilds the inventory, features, fitted predictions, analysis and publication outputs. It records the current code
revision and uses `artifacts/csedm-study/reproduction-<revision>/` by default. For a separate reproduction attempt, choose
an unused directory through the underlying CLI:

```bash
pixi run python scripts/csedm_study_reproduce.py --output-root artifacts/csedm-study/reproduction-review-001
```

Compare sample counts, predictions and reported statistics with the retained reference run. New revision metadata and
updated figure wording can change hashes even when scientific results agree. Save the terminal output and the generated
`pipeline_manifest.json` as the reproduction record.

## Common Failures

- **Port already in use:** stop the existing local demo before restarting `pixi run dev`. The default API uses port 8000.
- **Missing recorded response or report:** restore the required input from the retained experiment package. An empty file
  cannot replace a missing scientific record.
- **Immutable artifact conflict:** preserve the existing run. Use the command's output option to select a new directory
  when the inputs or code have changed; fixed-output Pixi shortcuts may need their underlying CLI command instead.
- **Live illustration unavailable:** check the local live settings and Modal authentication. Provider failures can occur
  after authentication succeeds; retain the returned error details for diagnosis.
- **Shell reports a filename as a command:** a multiline command was split incorrectly. Each continuation backslash must
  be the final character on its line. The short Pixi tasks avoid manual path entry where available.

## Data Policy

- `.local/` contains disposable SQLite and JSONL demo state.
- `artifacts/` contains generated experiment outputs.
- Neither directory is committed except for its ignore marker.
- W&B is optional and never the canonical data store.
- Keep historical learner archives and reviewer submissions local, with access limited by their applicable conditions.
- Demo conversations are application records and must not be represented as evidence from a student study.
