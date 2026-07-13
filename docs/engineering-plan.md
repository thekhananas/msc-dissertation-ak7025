# Engineering Plan

## Master's POC: Socratic Tutoring and RL Evaluation

| Field | Value |
|---|---|
| Status | Approved POC implementation strategy |
| Revision date | 2026-07-12 |
| Product requirements | `docs/PRD.md` |
| Delivery target | Working demo and reproducible dissertation evidence by 2026-09-15 |
| Deployment model | One FastAPI backend and one React client |
| Training model | Local CPU Gymnasium rollouts without LangGraph or LLM calls |

> **Sequential-rescope status:** This document and `docs/PRD.md` now define the current POC. `docs/HLD.md`, `docs/LLD.md`, `docs/data-flow-and-schema.md`, and `docs/implementation-plan.md` remain on the previous enterprise scope until their individual revision stages.

## 1. Engineering Objective

Build the smallest credible system that demonstrates and evaluates the dissertation's central loop:

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

The vertical slice must work before adding CBFM, BKT, SMC, contextual bandits, Q-learning, OpenRouter rendering, code sandboxing, or extensive visualization.

The project is a research POC, not an enterprise platform. Engineering quality is concentrated on mathematical correctness, reproducibility, state separation, experiment validity, demo reliability, and interpretable failure behavior.

## 2. Architecture Decisions

### 2.1 One Repository, One POC Deployment

Use a monorepo containing:

- Python domain and experiment packages.
- FastAPI application.
- React/Vite client.
- Hydra configurations.
- Versioned task/profile/split fixtures.
- Tests and analysis scripts.

For the demo, FastAPI serves API routes and the built React assets from one process or one deployment unit. There is no separate API gateway, worker cluster, queue, registry service, or observability stack.

### 2.2 Separate Interactive and Training Execution

Interactive tutoring:

```text
React
-> FastAPI
-> LangGraph
-> tracker
-> policy
-> template/OpenRouter renderer
-> guardrail
-> JSONL event + SQLite checkpoint
-> React
```

RL training:

```text
Resolved Hydra config
-> Gymnasium environment
-> heuristic/bandit/Q-learning policy
-> local rollout loop
-> Parquet trajectories
-> W&B metrics/artifact references
-> DuckDB/statistical evaluation
```

LangGraph is not imported by the training loop. OpenRouter is not called by the training loop. This prevents LLM cost, latency, and nondeterminism from contaminating policy learning.

### 2.3 State Ownership

Maintain four distinct state classes:

1. **Simulator truth:** private to the Gymnasium environment.
2. **Tracker belief:** owned by the active tracker implementation.
3. **Policy observation:** a validated projection of tracker-visible data.
4. **Interactive graph state:** session data needed to resume the demo.

Policy code may depend on policy-safe contracts only. It may not import simulator-private state.

### 2.4 Canonical Data Ownership

- JSONL is the append-only development/demo event log.
- Parquet is the canonical experiment trajectory format.
- SQLite stores resumable demo checkpoints and minimal session metadata.
- Local content-addressed files store policy tables, resolved configurations, calibration parameters, and reports.
- W&B stores run metadata, metrics, selected tables, and references/copies of approved artifacts.

The dissertation must remain reproducible if W&B is unavailable.

## 3. Approved Technology Stack

### 3.1 Workspace and Dependencies

Use Pixi from the repository's `development/` workspace as the task runner:

- Commit `pixi.toml` and `pixi.lock`.
- Pin Python 3.12, Node.js, pnpm, and native scientific dependencies.
- Use conda packages where native resolution helps and PyPI packages where appropriate.
- Commit `pnpm-lock.yaml` for frontend dependencies.

Pixi's lockfile captures exact environment resolution and its tasks provide repeatable cross-platform commands. See [Pixi workspaces](https://pixi.prefix.dev/latest/first_workspace/).

Required tasks:

```text
cd development
pixi run install-web
pixi run dev
pixi run test
pixi run lint
pixi run typecheck
pixi run config-smoke
pixi run docs-check
pixi run check

# Added by later milestones
pixi run check-env
pixi run train
pixi run evaluate
pixi run experiment
pixi run build-demo
```

### 3.2 Application

| Concern | Tool |
|---|---|
| Web client | React, TypeScript, Vite |
| Backend API | FastAPI, Pydantic v2 |
| Interactive orchestration | LangGraph |
| Demo checkpoints | LangGraph SQLite checkpointer |
| Development session metadata | SQLite |
| LLM provider | OpenRouter through local typed adapter |
| Optional code execution | One external sandbox provider |

React/Vite is retained because it provides the desired polished demo and later pilot path. The initial UI remains one task screen rather than a general learning platform.

### 3.3 Research and RL

| Concern | Tool |
|---|---|
| RL environment | Gymnasium |
| Policies | NumPy/SciPy implementations |
| Experiment configuration | Hydra |
| Experiment tracking | Weights & Biases |
| Canonical trajectories | PyArrow/Parquet |
| Query and analysis | DuckDB |
| Statistics | SciPy, statsmodels |
| Figures | Matplotlib, seaborn |
| Tests | pytest, Hypothesis, Gymnasium `check_env` |
| Quality | Ruff, Pyright |

Do not add RLlib, Ray, TorchRL, Stable-Baselines3, or CleanRL for the required tabular experiments. Reconsider a neural framework only after tabular coverage demonstrates a real limitation and the dissertation's critical path is complete.

### 3.4 Explicitly Deferred Infrastructure

The dissertation implementation does not require:

- PostgreSQL.
- Redis Streams or distributed queues.
- MLflow.
- Modal or Ray clusters.
- Terraform or Kubernetes.
- A LiteLLM proxy.
- Prometheus, Grafana, OpenTelemetry collectors, or Sentry.
- A policy registry or automated promotion service.
- Institutional SSO.
- Multi-region storage or production high availability.

These remain possible future-pilot or production improvements, not prerequisites.

## 4. Repository Layout

```text
research-brain-v2/
|-- development/                     # Pixi workspace and executable project
|   |-- apps/
|   |   |-- api/                     # FastAPI routes and app composition
|   |   `-- web/                     # React/Vite/TypeScript client
|   |-- src/
|   |   `-- socratic_tutor/
|   |       |-- contracts/           # Policy-safe Pydantic types
|   |       |-- graph/               # Interactive LangGraph workflow
|   |       |-- tracking/            # Simple, BKT, reduced SMC
|   |       |-- policies/            # Heuristic, bandit, Q-learning
|   |       |-- simulator/           # Gymnasium env and private truth
|   |       |-- cognitive/           # Prompt load and CBFM
|   |       |-- generation/          # Templates and OpenRouter adapter
|   |       |-- guardrails/          # Rule-first leakage checks
|   |       |-- trajectories/        # JSONL and Parquet I/O
|   |       `-- evaluation/          # Metrics, bootstrap, reports
|   |-- configs/                     # Hydra config groups
|   |-- data/                        # Versioned tasks, profiles, splits, schemas
|   |-- experiments/                 # Thin executable experiment entry points
|   |-- analysis/                    # Reproducible report commands
|   |-- tests/                       # Unit, property, integration, env, browser
|   |-- artifacts/                   # Generated and gitignored
|   |-- pixi.toml
|   |-- pixi.lock
|   |-- pyproject.toml
|   `-- pnpm-lock.yaml
|-- docs/
|-- proposal/
`-- README.md
```

All implementation commands run from `development/`. Production logic must not live only in notebooks. Notebooks may inspect results but final experiments and figures run through commands.

## 5. Core Interfaces

### 5.1 Interactive API

The first public endpoints are:

```text
POST /api/sessions
GET  /api/sessions/{session_id}
POST /api/sessions/{session_id}/turns
GET  /api/health
```

The vertical slice does not require WebSockets. Standard request/response behavior is easier to test and sufficiently responsive for one turn at a time. Streaming may be added after the complete demo is stable.

### 5.2 Policy-Safe Contracts

```python
class TrackerState(BaseModel):
    mastery: float
    misconception_probability: float
    estimated_bandwidth: float | None = None
    estimated_friction: float | None = None
    uncertainty: float

class PolicyObservation(BaseModel):
    target_concept: str
    tracker: TrackerState
    repeated_failures: int
    turn_index: int

class PolicyDecision(BaseModel):
    directive: PedagogicalDirective
    target_concept: str
    policy_version: str
    propensity: float | None = None
```

All Pydantic contracts use `extra="forbid"` and reject NaN/infinite values.

### 5.3 Extensible Protocols

```python
class BeliefTracker(Protocol):
    def update(
        self,
        previous: TrackerState,
        evidence: ObservationEvidence,
    ) -> TrackerState: ...

class TutorPolicy(Protocol):
    def act(
        self,
        observation: PolicyObservation,
        rng: numpy.random.Generator,
    ) -> PolicyDecision: ...

class ModelGateway(Protocol):
    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult: ...
```

The simple tracker, BKT, and reduced SMC share `BeliefTracker`. Heuristic, bandit, and Q-learning share `TutorPolicy`. Template and OpenRouter renderers share the generation boundary.

## 6. Vertical-Slice Implementation

### 6.1 Initial Task

Use one authored Python task about mutable-list aliasing:

```python
items = [1, 2]
alias = items
alias.append(3)
```

The learner explains the final values and why both names observe the mutation. The task fixture defines:

- Target concept.
- Prerequisite concept.
- Common copy-versus-alias misconception.
- Expected evidence phrases/concepts.
- Correctness rubric.
- Heuristic policy rules.
- Safe Socratic templates.

### 6.2 Turn Flow

1. React requests a new session.
2. FastAPI creates a session and invokes LangGraph.
3. LangGraph returns the authored initial prompt.
4. The learner submits text.
5. A deterministic evidence extractor produces structured evidence.
6. The simple tracker updates scalar mastery and misconception probability.
7. The heuristic policy chooses a directive.
8. The template renderer creates the next Socratic prompt.
9. The rule guardrail verifies no complete answer leakage.
10. One JSONL event is appended.
11. FastAPI returns the new state and prompt to React.

### 6.3 First Acceptance Gate

The vertical slice passes only when:

- A browser user completes the turn.
- Correct, misconception, uncertain, and empty responses work.
- Tracker updates match hand-calculated fixtures.
- Heuristic decisions match the authored rule table.
- The next prompt is safe and deterministic.
- The JSONL event contains complete versioned provenance.
- The whole test runs without network credentials.

No advanced component may delay this gate.

## 7. Gymnasium Simulator

Register:

```text
SocraticTutor/POMDP-v0
```

The environment must:

- Call `super().reset(seed=seed)`.
- Define explicit observation and action spaces.
- Keep true state in `simulator.private_state`.
- Return policy-visible observations only.
- Separate `terminated` and `truncated`.
- Expose decomposed reward in `info` for analysis, not policy input.
- Pass Gymnasium `check_env` in CI.
- Replay deterministic template episodes from seeds.

Gymnasium supports versioned custom environments, deterministic reset seeding, environment checking, and local vectorization. See [Gymnasium custom environments](https://gymnasium.farama.org/main/introduction/create_custom_env/).

Start with sequential environments. Add `SyncVectorEnv` only if profiling shows rollout throughput is inadequate.

## 8. Cognitive and Tracking Sequence

### 8.1 CBFM

Implement after the deterministic simulator:

1. Extract prompt token, code-span, AST-depth, concept-novelty, and entropy features.
2. Fit normalizers on calibration data only.
3. Compute overload friction from positive ZPD mismatch.
4. Compute bounded bandwidth drain and recovery.
5. Implement complete CBFM disablement.
6. Run monotonicity, recovery, sensitivity, and shuffled-feature controls.

Do not expose CBFM to policies until its basic invariants pass.

### 8.2 Trackers

Implement in this order:

1. Simple deterministic bounded tracker.
2. BKT tracker.
3. Reduced SMC tracker.

Reduced SMC tracks only:

- Active target-concept mastery.
- A small active misconception set.
- Bandwidth.
- Friction.

It must not initially particle-filter the full curriculum state. If SMC is unstable or offers no measurable calibration/decision benefit over BKT, retain BKT and report the negative result.

## 9. Policy Training

### 9.1 Policy Order

Implement:

1. Heuristic policy.
2. Contextual bandit.
3. Tabular Q-learning.
4. Optional static reference.

Use direct NumPy implementations so update equations, tables, and coverage remain inspectable.

### 9.2 Training Loop

```python
for episode_seed in seeds:
    observation, info = env.reset(seed=episode_seed)
    done = False

    while not done:
        decision = policy.act(observation, rng)
        next_observation, reward, terminated, truncated, info = env.step(
            decision.action_index
        )
        policy.observe(observation, decision, reward, next_observation, terminated, truncated)
        trajectory_writer.append(...)
        observation = next_observation
        done = terminated or truncated
```

For Q-learning, terminal transitions do not bootstrap. Truncation behavior is explicit in configuration. Persist the action-index mapping with the Q-table.

### 9.3 Neural Escalation Rule

Do not add a neural policy because tabular results are disappointing. Add one only if:

- The compact representation has demonstrably sparse coverage.
- Simplifying bins would remove decision-relevant information.
- The heuristic/bandit/Q-learning experiment is already complete.
- Time remains for a fair neural baseline and evaluation.

CleanRL is the preferred later reference due to its readable single-file algorithms and reproducibility features. See [CleanRL](https://docs.cleanrl.dev/).

## 10. Experiment Configuration

Use Hydra to compose environment, tracker, policy, CBFM, model, and experiment conditions:

```text
configs/
|-- config.yaml
|-- env/
|   |-- deterministic.yaml
|   `-- stochastic.yaml
|-- tracker/
|   |-- simple.yaml
|   |-- bkt.yaml
|   `-- smc.yaml
|-- policy/
|   |-- heuristic.yaml
|   |-- bandit.yaml
|   `-- q_learning.yaml
|-- cbfm/
|   |-- enabled.yaml
|   `-- disabled.yaml
`-- experiment/
    |-- vertical_slice.yaml
    |-- policy_comparison.yaml
    |-- cbfm_ablation.yaml
    `-- final_held_out.yaml
```

Hydra supports configuration composition and local multirun matrices. See [Hydra experiment configuration](https://hydra.cc/docs/patterns/configuring_experiments/).

Before each run:

- Resolve the complete configuration.
- Validate it with Pydantic.
- Save it locally.
- Hash it.
- Send the resolved values and hash to W&B.

Do not rely on source fragments alone because Hydra composes lazily at launch.

## 11. W&B Tracking

Use one W&B project with run groups for:

- Vertical-slice smoke tests.
- CBFM calibration.
- Tracker comparison.
- Policy training.
- Held-out evaluation.
- Ablations.

Every run logs:

- Git revision and dirty status.
- Pixi lock hash.
- Resolved Hydra configuration and hash.
- Root and component seeds.
- Task/profile/split versions.
- Environment, tracker, policy, CBFM, action-space, and prompt versions.
- OpenRouter model/provider route where applicable.
- Returns and decomposed reward.
- Diagnostic gain.
- Leakage and overload rates.
- Calibration metrics.
- Q-table coverage and visitation.
- Latency, tokens, and cost.
- Canonical artifact paths and checksums.

W&B configurations and artifacts support run comparison and lineage. See [W&B configurations](https://docs.wandb.ai/models/track/config) and [W&B Artifacts](https://docs.wandb.ai/models/artifacts).

Use W&B Sweeps only for policy-selection/validation hyperparameters. Never optimize against held-out data.

Do not upload future participant text, code, identity, or cognitive-load responses without approved governance.

## 12. Trajectory and Analysis Pipeline

### 12.1 Development JSONL

Each complete line contains one versioned `TrajectoryEvent`. Use write-then-flush behavior and recover by ignoring a final incomplete line.

### 12.2 Canonical Parquet

Write partitioned data by:

```text
artifacts/trajectories/
`-- experiment={experiment_id}/
    `-- split={split}/
        `-- condition={condition}/
            `-- seed={seed}/part.parquet
```

Keep public policy data and privileged simulator truth in separate schemas/directories. Training readers enumerate allowed columns rather than removing forbidden columns.

### 12.3 DuckDB Analysis

Query Parquet directly with DuckDB. It supports filter and projection pushdown and avoids running a database service. See [DuckDB Parquet support](https://duckdb.org/docs/stable/data/parquet/overview).

Final analysis scripts produce:

- Per-condition summaries.
- Matched-seed differences.
- Bootstrap intervals.
- Tracker calibration tables.
- CBFM controls.
- Q-table coverage.
- Action-distribution plots.
- Cost/latency tables.
- Gate decisions.

## 13. OpenRouter Integration

Use an OpenAI-compatible client behind `ModelGateway`. Do not deploy a LiteLLM proxy for the POC.

For primary experiments:

- Pin exact model ID.
- Pin provider order where supported.
- Disable provider fallback where reproducibility matters.
- Require structured-output support for structured calls.
- Pin temperature, top-p, maximum tokens, and other sampling parameters.
- Record generation ID, returned model, usage, finish reason, latency, and cost.
- Hash request prompts and final responses.

OpenRouter normalizes model APIs and supports structured output, usage, provider routing, and cost lookup. See [OpenRouter API](https://openrouter.ai/docs/api/reference/overview) and [provider routing](https://openrouter.ai/docs/guides/routing/provider-selection).

Never assume an LLM is deterministic even at temperature zero. Deterministic templates remain the replay and demo-fallback reference.

## 14. Guardrail and Sandbox

### 14.1 Guardrail

Start with deterministic checks for:

- Complete expected answer.
- Full solution code.
- Forbidden output fragments.
- Excessive prompt length.

Each rejected candidate receives one bounded rewrite attempt; exhaustion selects an authored template. A model-based guardrail is optional and may not replace deterministic leakage fixtures.

### 14.2 Sandbox

The first vertical slice uses text reasoning and does not require arbitrary execution.

If code submission is enabled:

- Integrate one external sandbox provider.
- Deny outbound network.
- Limit CPU, memory, wall time, process creation, filesystem, and output.
- Normalize tests, stdout, stderr, exit code, and timeout into structured evidence.
- Never fall back to executing untrusted code in the FastAPI host.

## 15. Demo Engineering

### 15.1 Required Screen

One React screen provides:

- Task and code snippet.
- Conversation history.
- Text response input.
- Submit/continue command.
- Current tracker estimate.
- Selected directive.
- Next Socratic prompt.
- Expandable structured evidence/log view.

Glass Box synthetic mode may additionally display true state and CBFM values in clearly labeled sections.

### 15.2 Reliability

- Use request/response REST initially.
- Disable duplicate submit while a turn is running.
- Assign a client turn ID for idempotency.
- Persist the completed event before returning success.
- Return typed, user-safe error states.
- Fall back to templates on OpenRouter failure.
- Provide a reset-session control.

### 15.3 Deployment

For evaluation and demonstration:

- Build React static assets.
- Serve them from FastAPI or one simple PaaS deployment.
- Mount or persist SQLite and artifact directories where required.
- Keep deterministic mode available if provider credentials or network fail.

No autoscaling, queue, infrastructure-as-code, or production SLO is required.

## 16. Testing Strategy

### 16.1 Vertical Slice

- Contract validation.
- Evidence fixtures.
- Hand-calculated tracker updates.
- Heuristic decision table.
- Template leakage fixtures.
- JSONL round trip.
- FastAPI endpoint flow.
- React browser E2E turn.
- Offline execution.

### 16.2 Mathematical Components

- CBFM boundedness, monotonicity, recovery, ablation, and seeded noise.
- BKT slip/guess/learn fixtures.
- SMC likelihood, active masks, finite weights, ESS, and resampling.
- Reward decomposition.
- Terminal/truncation updates.
- Discretizer boundaries and unseen-state fallback.

### 16.3 Experiment Integrity

- `check_env` in CI.
- Deterministic golden trajectories.
- Split overlap rejection.
- Matched-seed completeness.
- Policy feature parity.
- Privileged-state leakage scan.
- Resolved-configuration capture.
- Parquet schema and checksum validation.
- W&B-disabled reproduction.
- Held-out access control.

### 16.4 Demo Reliability

- OpenRouter timeout and malformed response.
- Template fallback.
- Duplicate turn submission.
- Backend restart and checkpoint recovery.
- Empty/long/adversarial input.
- Mobile and desktop layout.
- Keyboard and focus behavior.

## 17. Implementation Phases

### Phase 0: Workspace

Create Pixi/Python/Node environments, repository skeleton, lint/type/test tasks, and React/FastAPI health smoke tests.

### Phase 1: Vertical Slice

Implement one task, deterministic evidence, simple tracker, heuristic policy, template renderer, LangGraph turn, JSONL log, API endpoints, and one React workflow.

**Exit:** browser E2E works offline.

### Phase 2: Simulator and CBFM

Implement Gymnasium environment, private truth, deterministic profiles/transitions/reward, CBFM, replay, and Parquet output.

**Exit:** `check_env`, replay, mathematical invariants, and CBFM controls pass.

### Phase 3: LLM Demo Layer

Add typed OpenRouter adapter, prompt versioning, structured output where needed, rule guardrail, template fallback, and optional one-provider sandbox.

**Exit:** provider failures cannot break the demo.

### Phase 4: Trackers

Implement BKT, evaluate calibration, then implement reduced SMC and compare against BKT.

**Exit:** one tracker is selected from evidence; instability is reported, not hidden.

### Phase 5: Learned Policies

Implement contextual bandit and tabular Q-learning, coverage/visitation diagnostics, local training, and frozen artifacts.

**Exit:** all policies complete matched deterministic conditions through one interface.

### Phase 6: Experiments

Freeze configurations/splits, run matched seeds, execute CBFM/tracker/policy/guardrail controls, calculate intervals, and generate reproducible reports.

**Exit:** retain/reject/inconclusive decisions are locked before dissertation writing.

### Phase 7: Demo and Dissertation

Harden the UI, rehearse deterministic and LLM scenarios, archive artifacts, reproduce on a clean environment, and complete figures/results/limitations.

**Exit:** demo and dissertation package are submitted.

## 18. Scope Reduction Rules

If the project misses a phase exit date:

1. Preserve the vertical slice and deterministic simulator.
2. Preserve heuristic, BKT, and one learned-policy comparison.
3. Preserve canonical trajectories and valid evaluation.
4. Drop model-based guardrails before deterministic guardrails.
5. Drop arbitrary code execution before text/probe interaction.
6. Drop SMC before BKT if SMC blocks integration.
7. Drop Q-learning before fabricating inadequate coverage.
8. Drop UI embellishment before scientific analysis.
9. Never remove split integrity, matched seeds, provenance, or uncertainty reporting to save time.

## 19. POC Definition of Done

- Pixi and pnpm reproduce backend/frontend environments.
- The one-task browser vertical slice works offline.
- LangGraph orchestrates interactive turns but is absent from RL rollouts.
- `SocraticTutor/POMDP-v0` passes `check_env` and deterministic replay.
- Simple/BKT tracking provides a reliable baseline.
- CBFM and reduced SMC receive retain/reject evidence rather than assumed acceptance.
- Heuristic, bandit, and Q-learning share observations/actions and matched seeds where implemented.
- Hydra saves resolved configurations before execution.
- W&B runs resolve to local Parquet, policy, calibration, and report artifacts.
- DuckDB/statistical scripts reproduce final tables and figures.
- OpenRouter calls are pinned, recorded, bounded, and replaceable by templates.
- The demo remains usable when OpenRouter is unavailable.
- No production infrastructure or human-study data collection is required.
