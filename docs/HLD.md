# High-Level Design

## Master's POC: Socratic Tutoring and RL Evaluation

| Field | Value |
|---|---|
| Status | Approved POC architecture |
| Revision date | 2026-07-12 |
| Product requirements | `docs/PRD.md` |
| Engineering strategy | `docs/engineering-plan.md` |
| Deployment model | One FastAPI backend and one React client |
| Training model | Local CPU execution against Gymnasium |

> **Sequential-rescope status:** `docs/PRD.md`, `docs/engineering-plan.md`, and this HLD describe the current POC. `docs/LLD.md`, `docs/data-flow-and-schema.md`, and `docs/implementation-plan.md` remain on the previous enterprise design until their individual revision stages.

## 1. Purpose

This document defines the high-level architecture for a Master's-level proof of concept. The architecture must first support one complete tutoring turn through a browser and then provide a reproducible simulation path for evaluating CBFM, trackers, and tutor policies.

The first required flow is:

```text
One Python task
-> one student interaction
-> one LangGraph turn
-> simple tracker update
-> heuristic policy decision
-> next Socratic prompt
-> structured local log
-> minimal FastAPI + React demo
```

The design intentionally avoids distributed services. It preserves only the boundaries needed for scientific validity, testability, and a reliable demonstration.

## 2. Architecture Goals

- Deliver the browser vertical slice before advanced research components.
- Keep interactive LangGraph execution separate from RL rollouts.
- Run deterministic simulations without network credentials.
- Permit tracker and policy implementations to swap through small protocols.
- Prevent simulator truth from entering policy observations.
- Keep canonical trajectories in local Parquet files.
- Track experiment metadata and metrics in W&B without depending on it for reproduction.
- Render through OpenRouter only after deterministic templates work.
- Retain a functional demo when OpenRouter or the optional sandbox is unavailable.
- Defer human-study infrastructure and production scaling.

## 3. Architecture Decisions

| ID | Decision | Reason | Consequence |
|---|---|---|---|
| ADR-POC-001 | Use one monorepo and one demo deployment. | The project is implemented by one researcher on a fixed dissertation timeline. | React and FastAPI deploy together; no microservices. |
| ADR-POC-002 | Build the deterministic vertical slice first. | Integration risk is larger than algorithm implementation risk. | SMC, RL, LLM, and sandbox work cannot block the first gate. |
| ADR-POC-003 | Use LangGraph only for interactive tutoring. | It provides visible state transitions and resumable sessions but adds overhead to rollouts. | Training imports shared domain packages, not the graph runtime. |
| ADR-POC-004 | Use Gymnasium directly for training. | Pure `reset`/`step` loops are faster and more reproducible. | No LLM or web calls occur during policy learning. |
| ADR-POC-005 | Use SQLite and JSONL for the demo. | They provide sufficient local durability with negligible operations work. | PostgreSQL and Redis are deferred. |
| ADR-POC-006 | Use Parquet as canonical scientific storage. | It is typed, inspectable, efficient, and directly queryable through DuckDB. | W&B remains supplementary. |
| ADR-POC-007 | Use Hydra for resolved experiment configurations. | Policy/tracker/CBFM matrices require explicit, composable configurations. | Every run persists the fully resolved config and hash. |
| ADR-POC-008 | Use OpenRouter directly through a local protocol. | It provides one normalized model API without deploying another gateway. | LiteLLM proxy and multi-provider infrastructure are deferred. |
| ADR-POC-009 | Implement tabular policies directly in NumPy. | The algorithms and state coverage must remain inspectable. | RLlib, Ray, TorchRL, PPO, and DQN are not required. |
| ADR-POC-010 | Use one optional external sandbox. | Arbitrary code cannot run safely in the FastAPI host. | Text/probe interaction ships before sandbox integration. |
| ADR-POC-011 | Defer the human pilot. | Human research requires separate governance and study design. | The POC collects synthetic data only. |

## 4. System Context

```d2
direction: right

users: "Users" {
  researcher: "Researcher"
  evaluator: "Dissertation Evaluator"
  demo_user: "Demo Participant"
}

poc: "Socratic Tutoring POC" {
  web: "React Demo"
  api: "FastAPI Backend"
  interactive: "LangGraph Tutoring Flow"
  experiments: "Gymnasium Experiment Runner"
}

local_data: "Local Research Data" {
  sqlite: "SQLite Checkpoints"
  jsonl: "JSONL Turn Events"
  parquet: "Parquet Trajectories"
  artifacts: "Configs, Policies, and Reports"
}

external: "Optional External Services" {
  openrouter: "OpenRouter"
  wandb: "Weights & Biases"
  sandbox: "Code Sandbox"
}

future: "Deferred Future Pilot" {
  pilot: "Ethics-Approved Student Study"
}

users.researcher -> poc.web: "runs demo and inspects state"
users.researcher -> poc.experiments: "runs training and evaluation"
users.evaluator -> poc.web: "reviews tutoring behavior"
users.demo_user -> poc.web: "completes non-research demo task"

poc.web -> poc.api: "HTTP JSON"
poc.api -> poc.interactive: "start session and submit turn"
poc.interactive -> local_data.sqlite: "checkpoint state"
poc.interactive -> local_data.jsonl: "append committed event"
poc.experiments -> local_data.parquet: "write canonical trajectories"
poc.experiments -> local_data.artifacts: "write resolved configs and policies"

poc.interactive -> external.openrouter: "optional prompt rendering"
poc.interactive -> external.sandbox: "optional isolated code execution"
poc.experiments -> external.wandb: "metrics and artifact references"

future.pilot -> poc.web: "future frozen study build" {
  style.stroke-dash: 4
}
```

### 4.1 Context Rules

- The demo must run in deterministic template mode when all external services are unavailable.
- Demo interactions are not treated as human research data.
- W&B receives synthetic experiment metadata and approved artifacts only.
- OpenRouter receives the minimum prompt context needed to render a directive.
- The optional sandbox receives code, tests, limits, and an operation ID, not the full learner state.
- Future pilot functionality remains outside the current deployment and data lifecycle.

## 5. Logical Component Architecture

```d2
direction: right

client: "React/Vite Client" {
  task_view: "Task and Conversation"
  response_input: "Student Response Input"
  glass_box: "Evidence and State Inspector"
}

backend: "FastAPI POC" {
  routes: "Session and Turn Routes"
  session_service: "Tutor Session Service"
  graph: "LangGraph Turn Graph"
  event_writer: "JSONL Event Writer"
}

domain: "Shared Python Domain" {
  contracts: "Pydantic Contracts"
  evidence: "Deterministic Evidence Extractor"
  trackers: "Simple / BKT / Reduced SMC"
  policies: "Heuristic / Bandit / Q-Learning"
  templates: "Socratic Template Renderer"
  guardrail: "Rule-First Guardrail"
  cbfm: "Prompt Load and CBFM"
}

simulation: "Offline Research Path" {
  env: "SocraticTutor/POMDP-v0"
  runner: "Hydra Experiment Runner"
  trajectory_writer: "Parquet Writer"
  evaluator: "DuckDB and Statistical Analysis"
}

storage: "Local Storage" {
  sqlite: "SQLite Checkpointer"
  jsonl: "Append-Only JSONL"
  parquet: "Partitioned Parquet"
  artifact_dir: "Content-Addressed Artifacts"
}

adapters: "Optional Adapters" {
  model_gateway: "OpenRouter ModelGateway"
  sandbox_gateway: "External SandboxGateway"
  wandb_sink: "W&B Run Sink"
}

client.task_view -> backend.routes: "create/load session"
client.response_input -> backend.routes: "submit turn"
backend.routes -> backend.session_service: "validated command"
backend.session_service -> backend.graph: "invoke with thread id"
backend.graph -> domain.evidence: "classify response"
backend.graph -> domain.trackers: "update belief"
backend.graph -> domain.policies: "select directive"
backend.graph -> domain.templates: "render fallback prompt"
backend.graph -> domain.guardrail: "check final prompt"
backend.graph -> backend.event_writer: "committed turn"
backend.graph -> storage.sqlite: "checkpoint"
backend.event_writer -> storage.jsonl: "append"
backend.routes -> client.glass_box: "TurnResponse"

backend.graph -> adapters.model_gateway: "optional LLM render"
backend.graph -> adapters.sandbox_gateway: "optional code execution"

simulation.runner -> simulation.env: "construct and step directly"
simulation.env -> domain.cbfm: "true fast-state dynamics"
simulation.env -> domain.trackers: "subjective update"
simulation.env -> domain.policies: "policy-safe observation"
simulation.runner -> simulation.trajectory_writer: "turn records"
simulation.trajectory_writer -> storage.parquet: "canonical data"
simulation.runner -> storage.artifact_dir: "resolved config and policy"
simulation.runner -> adapters.wandb_sink: "metrics and references"
storage.parquet -> simulation.evaluator: "query"
storage.artifact_dir -> simulation.evaluator: "frozen provenance"
```

### 5.1 Component Boundaries

#### React Client

Provides one focused workflow: read task, inspect snippet, submit explanation, receive next prompt, and inspect the committed state transition. It does not contain tracking, policy, or scoring logic.

#### FastAPI Backend

Validates API payloads, resolves sessions, invokes LangGraph, and returns typed responses. It is the only browser-facing process.

#### LangGraph Turn Graph

Runs the interactive sequence and checkpoints policy-safe state. It coordinates components but does not implement mathematical updates itself.

#### Shared Domain

Contains pure, testable contracts and algorithms. Tracker, policy, evidence, CBFM, template, and guardrail code is usable without FastAPI or LangGraph.

#### Gymnasium Environment

Owns synthetic true state and mathematical transitions. It directly invokes shared trackers and policies while ensuring policy observations contain no true state.

#### Experiment Runner

Loads a fully resolved Hydra configuration, derives seeds, executes local rollouts, writes Parquet/artifacts, and reports metrics to W&B.

#### Local Storage

Keeps the POC reproducible without running infrastructure services. SQLite is operational state; JSONL is demo evidence; Parquet and content-addressed artifacts are scientific state.

## 6. Two Execution Paths

### 6.1 Interactive Path

```text
React request
-> FastAPI validation
-> LangGraph invocation
-> deterministic evidence
-> tracker update
-> heuristic/frozen policy
-> template or OpenRouter rendering
-> rule guardrail
-> JSONL append
-> SQLite checkpoint
-> React response
```

The vertical slice initially uses the simple tracker, heuristic policy, and templates. Later components replace implementations behind the same protocols.

### 6.2 Training Path

```text
Hydra resolved config
-> Gymnasium reset(seed)
-> policy action
-> mathematical environment step
-> policy update
-> Parquet trajectory
-> local policy artifact
-> W&B metrics/reference
```

The training path has no dependency on:

- FastAPI.
- React.
- LangGraph.
- SQLite checkpoints.
- OpenRouter.
- The sandbox.

### 6.3 Evaluation Path

```text
Frozen held-out config
-> matched seed/task/profile rollouts
-> validated Parquet partitions
-> DuckDB queries
-> SciPy/statsmodels intervals
-> figures, tables, and gate decisions
-> W&B summary/reference
```

Final reports remain reconstructable from local files.

## 7. State Ownership and Visibility

| State | Owner | Interactive visibility | Training visibility |
|---|---|---|---|
| Synthetic true mastery/misconceptions | Gymnasium environment | Glass Box synthetic mode only | Environment and controlled evaluator |
| True `B_t` and `F_t` | Gymnasium environment | Labeled synthetic view only | Environment and controlled evaluator |
| Tracker internals/particles | Active tracker | Not exposed | Tracker diagnostics only |
| Tracker summaries | Active tracker | Policy, tutor, Glass Box | Policy and evaluation |
| Policy observation | Observation projector | Active policy | Active policy and public trajectory |
| Policy decision | Active policy | Tutor renderer and Glass Box | Environment and trajectory |
| Graph session state | LangGraph | API projection only | Not used |
| Demo event | Event writer | Researcher/evaluator | Not used for training by default |
| Experiment trajectory | Parquet writer | Selected examples | Trainer/evaluator according to schema |

Enforcement:

- Simulator truth resides in a private package.
- Policy protocols accept `PolicyObservation`, never environment state.
- Pydantic contracts reject extra fields.
- Training readers enumerate permitted Parquet columns.
- Glass Box truth routes are enabled only for synthetic mode.

## 8. Vertical-Slice Architecture

### 8.1 Initial Domain Fixture

The first task concerns Python mutable-list aliasing:

```python
items = [1, 2]
alias = items
alias.append(3)
```

The fixture contains the task text, target concept, prerequisite, misconception, expected evidence, heuristic rules, and Socratic templates. It is versioned in the repository.

### 8.2 Initial Components

- **Evidence extractor:** deterministic lexical/rubric rules producing correct, misconception, uncertain, or empty evidence.
- **Tracker:** bounded scalar mastery and misconception probability update with hand-calculated fixtures.
- **Policy:** ordered heuristic decision table.
- **Renderer:** authored templates for every legal directive.
- **Guardrail:** rule checks against complete answers and solution fragments.
- **Persistence:** one append-only JSONL turn event; in-memory checkpointing is acceptable before SQLite integration.

### 8.3 Vertical-Slice Gate

The slice is complete when a browser test proves:

1. Session creation returns the authored prompt.
2. Student submission reaches the correct session/turn.
3. Evidence output matches the fixture.
4. Tracker before/after values are correct.
5. Heuristic action matches the rule table.
6. The next prompt passes leakage checks.
7. A complete event is appended before success returns.
8. The workflow runs without network credentials.

## 9. Experiment Architecture

```d2
direction: right

configuration: "Experiment Definition" {
  hydra: "Hydra Config Groups"
  resolved: "Resolved Config + Hash"
  split: "Frozen Split Manifest"
}

execution: "Local CPU Execution" {
  matrix: "Matched Condition/Seed Matrix"
  envs: "Gymnasium Environments"
  policy: "Heuristic / Bandit / Q-Learning"
}

outputs: "Canonical Local Outputs" {
  trajectory: "Parquet Trajectories"
  policy_artifact: "Policy Table Artifact"
  run_manifest: "Run Manifest"
}

analysis: "Evaluation" {
  duckdb: "DuckDB Queries"
  statistics: "Bootstrap / Calibration / Rates"
  report: "Figures and Gate Decisions"
}

tracking: "Supplementary Tracking" {
  wandb: "W&B Run"
}

configuration.hydra -> configuration.resolved: "compose"
configuration.split -> configuration.resolved: "reference and hash"
configuration.resolved -> execution.matrix: "freeze before run"
execution.matrix -> execution.envs: "reset with derived seeds"
execution.envs <-> execution.policy: "observation, action, reward"
execution.envs -> outputs.trajectory: "validated turns"
execution.policy -> outputs.policy_artifact: "frozen learned table"
configuration.resolved -> outputs.run_manifest: "provenance"

outputs.trajectory -> analysis.duckdb: "query"
outputs.policy_artifact -> analysis.statistics: "coverage and visitation"
analysis.duckdb -> analysis.statistics: "matched samples"
analysis.statistics -> analysis.report: "estimates and intervals"

outputs.run_manifest -> tracking.wandb: "config and hashes"
outputs.trajectory -> tracking.wandb: "artifact reference"
outputs.policy_artifact -> tracking.wandb: "artifact reference"
analysis.report -> tracking.wandb: "summary metrics and figures"
```

### 9.1 Configuration Strategy

Hydra composes environment, tracker, policy, CBFM, task/profile, split, seed, and evaluation settings. The runner saves and hashes the fully resolved configuration before execution.

### 9.2 Seed Strategy

A root seed deterministically derives:

- Environment initialization seed.
- Transition-noise seed.
- Observation-noise seed.
- Policy-exploration seed.
- Optional generation seed.

Matched conditions reuse the relevant environment/task/profile seeds while retaining independent policy exploration streams according to the experiment manifest.

### 9.3 W&B Boundary

W&B receives configuration, metrics, summaries, selected tables, and approved artifact references. It is not queried to reconstruct canonical trajectories during final analysis.

### 9.4 Scale Strategy

Run sequentially first. If profiling demonstrates a bottleneck, use Gymnasium `SyncVectorEnv` or local multiprocessing. Distributed compute remains out of scope.

## 10. LLM Rendering Architecture

### 10.1 Gateway

`ModelGateway` accepts a typed request and returns content, provider/model metadata, usage, latency, finish reason, and request/generation identifiers.

OpenRouter is the sole initial provider path. The application pins:

- Model ID.
- Provider order where available.
- Fallback behavior.
- Structured-output requirement.
- Sampling parameters.
- Prompt version/hash.

### 10.2 Experimental Use

The language layer is evaluated after policy training. A policy action is rendered through:

- An authored template.
- A pinned OpenRouter configuration.

This permits attribution between policy selection and language-generation quality.

### 10.3 Failure Policy

- Timeout or provider failure: use deterministic template.
- Invalid structured output: reject and use template.
- Guardrail rejection: one bounded rewrite, then template.
- Budget exhaustion: stop LLM calls and remain in template mode.

No failure discards an already accepted student turn.

## 11. Code Execution Boundary

Arbitrary code execution is not required for the first vertical slice.

When enabled:

```text
FastAPI/LangGraph
-> SandboxGateway
-> one external isolated sandbox
-> normalized execution evidence
-> tracker
```

The sandbox request contains source, test bundle, runtime, operation ID, and resource limits. It does not contain unrelated conversation or simulator truth.

There is no host-process fallback for untrusted code. If the sandbox is unavailable, the UI permits text responses or disables code submission with a clear message.

## 12. Local Data Architecture

### 12.1 SQLite

Stores LangGraph checkpoints and minimal session metadata after the vertical slice. It is not the experiment trajectory database.

### 12.2 JSONL

Stores one complete event per committed demo turn. A final partial line after process failure is ignored during recovery.

### 12.3 Parquet

Stores experiment trajectories partitioned by experiment, split, condition, and seed. Public policy features and privileged simulator truth use separate schemas/directories.

### 12.4 Local Artifacts

Content-addressed files store:

- Resolved Hydra configurations.
- Split manifests.
- Calibration parameters.
- BKT/SMC configuration.
- Bandit/Q tables.
- Prompt/template versions.
- Metric summaries.
- Figures and reports.

## 13. Deployment Architecture

```d2
direction: down

developer: "Researcher Laptop / Demo Host" {
  pixi: "Pixi Workspace"
  pnpm: "pnpm Frontend Workspace"

  app: "Single POC Deployment" {
    fastapi: "FastAPI Process"
    react: "Built React Assets"
    graph: "LangGraph Runtime"
  }

  research: "Offline Commands" {
    train: "Gymnasium Training"
    evaluate: "DuckDB Evaluation"
  }

  files: "Persistent Local Directory" {
    sqlite: "demo.sqlite"
    jsonl: "events.jsonl"
    parquet: "trajectories/*.parquet"
    artifacts: "artifacts/*"
  }
}

external: "Optional Network Dependencies" {
  openrouter: "OpenRouter API"
  wandb: "W&B API"
  sandbox: "External Sandbox API"
}

browser: "Browser"

browser -> developer.app.fastapi: "HTTP"
developer.app.fastapi -> developer.app.react: "serve static assets"
developer.app.fastapi -> developer.app.graph: "invoke tutoring turn"
developer.app.graph -> developer.files.sqlite: "checkpoint"
developer.app.graph -> developer.files.jsonl: "turn event"

developer.research.train -> developer.files.parquet: "write trajectories"
developer.research.train -> developer.files.artifacts: "write policies/configs"
developer.files.parquet -> developer.research.evaluate: "read"
developer.files.artifacts -> developer.research.evaluate: "read provenance"

developer.app.graph -> external.openrouter: "optional render"
developer.app.graph -> external.sandbox: "optional execution"
developer.research.train -> external.wandb: "optional metrics sync"

external.openrouter -> developer.app.graph: "failure falls back locally" {
  style.stroke-dash: 4
}
external.wandb -> developer.research.train: "unavailable does not block run" {
  style.stroke-dash: 4
}
```

### 13.1 Deployment Modes

#### Local Development

- Vite development server proxies `/api` to FastAPI.
- In-memory LangGraph checkpointer may be used in unit/integration tests.
- SQLite and artifacts live under a gitignored workspace directory.
- W&B defaults to offline/disabled for tests.

#### Demonstration Build

- Build React static assets.
- Serve assets and API from FastAPI or one simple PaaS application.
- Persist SQLite and artifact directory if the host supports it.
- Keep template mode preconfigured as the emergency fallback.

#### Future Pilot

Not specified as a current deployment. It requires separate identity, consent, human-data storage, retention, monitoring, and governance design after approval.

## 14. Reliability and Failure Handling

| Failure | Required behavior | Integrity rule |
|---|---|---|
| Invalid student request | Return typed 4xx response | Do not append event or advance turn |
| Duplicate turn submission | Return existing result or conflict | Never commit two logical events for one turn ID |
| Evidence extraction failure | Return uncertain evidence or typed failure | Never infer mastery from missing output silently |
| Tracker numerical failure | Fall back to last valid/simple tracker state and mark degraded | Never use simulator truth as recovery |
| Unknown policy state | Use documented heuristic fallback | Log fallback and coverage miss |
| OpenRouter timeout/failure | Render deterministic template | Student turn remains committed once valid |
| Guardrail rejection | One rewrite then template | Never expose rejected candidate |
| Sandbox failure | Return structured unavailable/timeout evidence | Never execute on host |
| JSONL append failure | Fail turn before success response | No acknowledged unlogged turn |
| SQLite failure | Use explicit non-resumable error or stop session | Do not claim durable checkpoint |
| W&B unavailable | Continue local experiment and sync later | Local files remain canonical |
| Final partial Parquet output | Mark run incomplete and exclude partition | Evaluation reads validated partitions only |

## 15. Security and Privacy

- Secrets load from local environment or approved host secret settings and never enter W&B configs.
- Pydantic models forbid extra fields at policy and provider boundaries.
- Simulator-private types cannot be imported by policy modules.
- Public and privileged trajectory writers use distinct schemas and paths.
- React never receives simulator truth unless synthetic Glass Box mode is explicitly enabled.
- OpenRouter prompts omit secrets and unnecessary private context.
- No raw hidden reasoning is stored, logged, traced, or displayed.
- Untrusted code never runs in the FastAPI process.
- No human research data is collected in the dissertation POC.

## 16. Observability for the POC

Use lightweight observability:

- Structured application logs with session/turn IDs.
- JSONL tutoring events.
- W&B experiment runs.
- Local error output and test reports.
- Cost/token/latency fields for OpenRouter calls.

Prometheus, Grafana, OpenTelemetry infrastructure, Sentry, and LangSmith are optional debugging additions, not delivery requirements.

## 17. Requirement Traceability

| Architecture area | Primary PRD requirements |
|---|---|
| Vertical slice | FR-VS-001 through FR-VS-010 |
| Experiment configuration | FR-EXP-001 through FR-EXP-006, NFR-REP-001 through NFR-REP-005 |
| Simulator | FR-SIM-001 through FR-SIM-008 |
| CBFM | FR-CBFM-001 through FR-CBFM-009 |
| Tracking | FR-TRK-001 through FR-TRK-009 |
| Policies and RL | FR-POL-001 through FR-POL-009 |
| Interactive graph | FR-GRF-001 through FR-GRF-005 |
| LLM and guardrail | FR-GEN-001 through FR-GRD-003 |
| Trajectories and W&B | FR-TRJ-001 through FR-TRJ-006 |
| Demo | FR-UI-001 through FR-UI-006 |
| Optional code execution | SEC-SBX-001 through SEC-SBX-004 |
| Security and future pilot | NFR-SEC-001 through NFR-PRV-002 |

## 18. HLD Acceptance Criteria

This architecture is ready for low-level revision when:

- The vertical slice is the first dependency gate.
- React and FastAPI form one POC application boundary.
- LangGraph is used only for interactive tutoring.
- Gymnasium training is independent of LangGraph, React, FastAPI, OpenRouter, and W&B availability.
- SQLite, JSONL, Parquet, local artifacts, and W&B have distinct responsibilities.
- Simulator truth cannot enter policy or future-human interfaces.
- OpenRouter and sandbox failures have deterministic local behavior.
- Production/distributed infrastructure is explicitly deferred.
- Future pilot infrastructure and data collection remain outside current scope.
- All D2 diagrams compile successfully.
