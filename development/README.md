# Development Workspace

This directory is the only executable workspace for the Socratic Tutor POC. The repository-level `docs/` and `proposal/` directories remain outside it so implementation artifacts do not obscure dissertation material.

## Prerequisites

- Pixi 0.63 or later.
- Git.
- D2 0.7.1 for local diagram compilation. CI installs the pinned version.

Python, Node.js, pnpm, and project dependencies are supplied by Pixi.

## First Run

```bash
pixi run install-web
pixi run check
```

Run the API and web client together:

```bash
pixi run dev
```

Then open `http://127.0.0.1:5173`. Vite proxies `/api` to FastAPI on port 8000.

## Focused Commands

```bash
pixi run dev-api
pixi run dev-web
pixi run test
pixi run test-web
pixi run lint
pixi run typecheck
pixi run config-smoke
pixi run docs-check
pixi run build-web
```

`config-smoke` composes the default Hydra configuration without running a simulation. Experiment behavior is added only after the browser vertical slice is complete.

## Data Policy

- `.local/` contains disposable SQLite and JSONL demo state.
- `artifacts/` contains generated experiment outputs.
- Neither directory is committed except for its ignore marker.
- W&B is optional and never the canonical data store.
- No human research data belongs in this workspace under the dissertation POC.
