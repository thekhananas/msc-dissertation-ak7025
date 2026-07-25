# Engineering Plan

## Master's POC: Socratic Tutoring and RL Evaluation

| Field | Value |
|---|---|
| Status | Proposed benchmark-first amendment for review |
| Revision date | 2026-07-14 |
| Product requirements | `docs/PRD.md` |
| Delivery target | Working demo and reproducible dissertation evidence by 2026-09-15 |
| Deployment model | One FastAPI backend and one React client |
| Training model | Local CPU Gymnasium rollouts without LangGraph or LLM calls |

> **Sequential-amendment status:** This document implements the measurement requirements in the revised PRD. The HLD, LLD, data design, implementation plan, and dissertation proposal do not yet contain this benchmark-first amendment and must be revised one at a time.

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

After the vertical slice, the next deliverable is not the simulator or RL. It is a frozen measurement instrument:

```text
Public interaction
-> dialogue-only prediction/action
-> isolated evidence probe
-> probe-informed prediction/action
-> commit both decisions
-> reveal a different held-out criterion probe
-> score paired calibration and false-mastery outcomes
```

The deterministic benchmark fixtures validate this pipeline. Pinned LLM-student runs then stress-test whether the effect occurs in selected student simulators. The criterion probe is never visible to trackers or policies before their outputs are committed.

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

Benchmark evaluation:

```text
Frozen benchmark manifest
-> independent condition runner
-> public-response evidence
-> optional evidence-probe result
-> committed tracker prediction and policy action
-> criterion-probe reveal
-> execution-backed label
-> paired case-level metrics and controls
```

The benchmark runner is not a Gymnasium rollout and does not optimize a policy. It evaluates evidence validity. Deterministic fixtures run offline; LLM-student stress tests call pinned models through `ModelGateway` but never use an LLM judge for correctness.

### 2.3 State Ownership

Maintain four distinct state classes:

1. **Simulator truth:** private to the Gymnasium environment.
2. **Tracker belief:** owned by the active tracker implementation.
3. **Policy observation:** a validated projection of tracker-visible data.
4. **Interactive graph state:** session data needed to resume the demo.

Benchmark evaluation adds two non-policy payloads:

5. **Evidence-probe payload:** available only in the probe-informed and control conditions.
6. **Criterion payload:** evaluator-owned and inaccessible until all compared predictions/actions are committed.

Policy code may depend on policy-safe contracts only. It may not import simulator-private state.

### 2.4 Canonical Data Ownership

- JSONL is the append-only development/demo event log.
- Checked-in JSON manifests and trusted Python fixtures define benchmark cases, probes, executable tests, reviews, and split membership.
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
pixi run benchmark-validate
pixi run benchmark-smoke
pixi run benchmark-generate
pixi run benchmark-evaluate
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
| Benchmark contracts | Pydantic v2 and generated JSON Schema |
| Benchmark labels | Held-out executable criterion probes and reviewed rubrics |
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
|   |       |-- benchmark/           # Cases, isolation, runners, metrics
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
|   |-- data/
|   |   |-- benchmark/v1/            # Cases, probes, reviews, split manifest
|   |   |-- tasks/                   # Interactive/simulator task fixtures
|   |   |-- profiles/                # Synthetic learner profiles
|   |   |-- splits/                  # Non-benchmark experiment splits
|   |   `-- schemas/                 # Generated schema snapshots
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

### 5.4 Benchmark Contracts and Capabilities

Do not place criterion fields in policy-safe contracts. Use separate evaluator-owned models:

```python
class BenchmarkConditionInput(BaseModel):
    case_id: str
    condition: Literal["dialogue_only", "probe_informed", "unrelated_probe", "corrupted_probe"]
    public_evidence: ObservationEvidence
    evidence_probe: ObservationEvidence | None

class CommittedPrediction(BaseModel):
    case_id: str
    condition: str
    mastery_probability: float
    policy_decision: PolicyDecision
    tracker_version: str

class CriterionRecord(BaseModel):
    case_id: str
    criterion_probe_id: str
    demonstrated_mastery: bool
    execution_record_hash: str
```

The condition runner receives `BenchmarkConditionInput` and returns `CommittedPrediction`. Only the scorer can load `CriterionRecord`, and only after every required condition for that case is committed. Runtime spies fail a run if criterion fields are accessed earlier.

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

## 7. Benchmark and Measurement Engineering

### 7.1 Artifact Layout

Keep authored inputs, decision-time payloads, and criterion data physically distinguishable:

```text
development/data/benchmark/v1/
|-- manifest.json
|-- schema/
|   |-- case.schema.json
|   `-- review.schema.json
|-- cases/
|   `-- {case_id}/
|       |-- metadata.json
|       |-- public_interaction.json
|       |-- evidence_probe.json
|       |-- criterion/
|       |   |-- probe.json
|       |   `-- expected_tests.json
|       `-- review.json
|-- splits/
|   |-- development.json
|   |-- calibration.json
|   |-- policy_selection.json
|   `-- held_out.json
`-- LIMITATIONS.md
```

Generated student responses, execution results, condition predictions, and metrics belong under `artifacts/benchmark/`, not the checked-in case definitions. The frozen manifest hashes every case file, executable test, review, split, prompt template, and schema.

### 7.2 Case Construction

Construct at least 24 cases from four Python concepts, two misconception states per concept, and three transfer cases per concept/misconception pair. Each concept must include supported mastery, false public mastery, honest non-mastery, underconfidence, and ambiguous evidence patterns.

Every case contains:

- A public tutoring interaction or deterministic public-response fixture.
- An evidence probe assessing the target concept without tutor dialogue.
- A structurally different criterion probe assessing transfer of the same concept.
- Executable tests or a deterministic rubric for both probes.
- A criterion demonstrated-mastery label and rationale.
- An author review and an independent review/adjudication record.

Evidence and criterion probes may share the target concept but not the solution, surface template, answer-bearing wording, or exact execution trace. Automated lexical/AST similarity checks are warnings, not substitutes for review.

### 7.3 Authoring and Freeze Workflow

```text
draft case
-> schema validation
-> trusted executable-test validation
-> isolation and similarity checks
-> independent review
-> task-family split assignment
-> manifest hashing
-> benchmark freeze
```

After freeze, a semantic correction increments the benchmark version. Never patch a held-out case in place. Formatting-only changes still update hashes and are recorded in a change log.

### 7.4 Paired Condition Runner

For each case:

1. Load one immutable initial tracker state and public evidence.
2. Clone the state into isolated condition instances.
3. Run dialogue-only, probe-informed, unrelated-probe, and corrupted-probe conditions.
4. Commit each mastery probability and policy action to an append-only prediction record.
5. Verify that every pair shares the same case, initial state, tracker, policy, and pre-update history.
6. Load the criterion probe/result only after commitments exist.
7. Score every condition against the same criterion demonstrated-mastery label.

No condition may update shared tracker state. The runner rejects missing pairs rather than silently comparing unmatched cases.

### 7.5 Deterministic and LLM-Student Layers

The deterministic layer uses authored public, evidence-probe, and criterion-probe responses with known outcomes. It proves that schemas, isolation, tracker updates, controls, and metric directions are correct; it is not evidence that real LLM student simulators behave sycophantically.

The LLM-student layer uses a pinned `ModelGateway` and three separate generation contexts:

- Public dialogue receives the tutor history and declared synthetic profile.
- Evidence probe receives the profile and probe only, without tutor history.
- Criterion probe receives the same profile and its different probe only, in another fresh context.

At least one pinned model/provider condition is required for the primary stress test. A second model family is a replication condition when budget and schedule permit. Store public outputs and final code, never hidden reasoning. Replays consume recorded responses and execution records rather than calling the provider again.

### 7.6 Execution Boundary

Checked-in authored fixtures may execute locally only through a hash allowlist in test/evaluation commands. Any LLM-generated or participant-supplied code is untrusted and must use the external sandbox boundary. A sandbox outage yields an explicit missing criterion/evidence status; it does not convert failure to incorrect mastery.

Normalize execution into test counts, exit status, stdout/stderr hashes, timeout/resource flags, and rubric outcomes. Criterion records remain evaluator-only until condition commitment.

### 7.7 Metrics and Statistical Unit

The primary metric is the per-case paired difference:

```text
Brier(dialogue_only, criterion_label)
- Brier(probe_informed, criterion_label)
```

Positive values favor probe-informed estimation. Report the mean paired difference, a case-level paired bootstrap interval, and a paired randomization sensitivity test. Freeze the exact interval and exclusion procedure before held-out execution.

Repeated LLM samples are nested observations, not independent benchmark cases. Aggregate within case/model before the primary case-level comparison or use a declared clustered analysis. Report false-mastery acceptance, detection precision/recall, unsafe advancement, confidence, policy disagreement, invalid executions, missing pairs, and case-level effects.

Probe-informed improvement is non-specific if unrelated or corrupted probe controls improve similarly. Policy rank reversal is exploratory until there are enough independent policies and cases for a stable comparison.

### 7.8 Benchmark Commands and Gate

Add deterministic commands:

```text
pixi run benchmark-validate   # schemas, hashes, balance, splits, isolation, tests
pixi run benchmark-smoke      # authored fixtures and all four conditions
pixi run benchmark-generate   # pinned LLM-student responses, when enabled
pixi run benchmark-evaluate   # frozen paired metrics and report artifacts
```

The benchmark gate passes only when all 16 `FR-BMK` requirements pass, an independent reviewer has signed off the case set, and the deterministic smoke report recovers expected metric directions. CBFM fitting, tracker comparison, learned-policy training, and held-out evaluation remain blocked until then.

## 8. Gymnasium Simulator

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

## 9. Cognitive and Tracking Sequence

### 9.1 CBFM

Implement after the deterministic simulator:

1. Extract prompt token, code-span, AST-depth, concept-novelty, and entropy features.
2. Fit normalizers on calibration data only.
3. Compute overload friction from positive ZPD mismatch.
4. Compute bounded bandwidth drain and recovery.
5. Implement complete CBFM disablement.
6. Run monotonicity, recovery, sensitivity, and shuffled-feature controls.

Do not expose CBFM to policies until its basic invariants pass.

### 9.2 Trackers

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

## 10. Policy Training

### 10.1 Policy Order

Implement:

1. Heuristic policy.
2. Contextual bandit.
3. Tabular Q-learning.
4. Optional static reference.

Use direct NumPy implementations so update equations, tables, and coverage remain inspectable.

### 10.2 Training Loop

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

### 10.3 Neural Escalation Rule

Do not add a neural policy because tabular results are disappointing. Add one only if:

- The compact representation has demonstrably sparse coverage.
- Simplifying bins would remove decision-relevant information.
- The heuristic/bandit/Q-learning experiment is already complete.
- Time remains for a fair neural baseline and evaluation.

CleanRL is the preferred later reference due to its readable single-file algorithms and reproducibility features. See [CleanRL](https://docs.cleanrl.dev/).

## 11. Experiment Configuration

Use Hydra to compose benchmark, student generator, environment, tracker, policy, CBFM, model, and experiment conditions:

```text
configs/
|-- config.yaml
|-- benchmark/
|   `-- v1.yaml
|-- student/
|   |-- authored_fixture.yaml
|   `-- pinned_llm.yaml
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
    |-- benchmark_smoke.yaml
    |-- benchmark_primary.yaml
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

The paired benchmark conditions execute inside one benchmark run from one frozen manifest; do not launch dialogue-only and probe-informed conditions as unrelated sweep jobs. This preserves complete pairs, identical initial state, and criterion timing.

Do not rely on source fragments alone because Hydra composes lazily at launch.

## 12. W&B Tracking

Use one W&B project with run groups for:

- Vertical-slice smoke tests.
- Deterministic benchmark validation.
- LLM-student benchmark stress tests.
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
- Benchmark version, manifest hash, case IDs, condition-pair completeness, and criterion reveal status.
- Student-generator model, provider, prompt/profile version, and recorded-response artifact for LLM stress tests.
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

Do not tune benchmark cases, evidence probes, criterion probes, labels, exclusions, or primary metrics through W&B Sweeps.

Do not upload future participant text, code, identity, or cognitive-load responses without approved governance.

## 13. Trajectory and Analysis Pipeline

### 13.1 Development JSONL

Each complete line contains one versioned `TrajectoryEvent`. Use write-then-flush behavior and recover by ignoring a final incomplete line.

### 13.2 Canonical Parquet

Write partitioned data by:

```text
artifacts/trajectories/
`-- experiment={experiment_id}/
    `-- split={split}/
        `-- condition={condition}/
            `-- seed={seed}/part.parquet
```

Keep public policy data and privileged simulator truth in separate schemas/directories. Training readers enumerate allowed columns rather than removing forbidden columns.

Benchmark outputs use a separate layout:

```text
artifacts/benchmark/
`-- version={benchmark_version}/
    `-- run={run_id}/
        |-- generation_records.parquet
        |-- condition_predictions.parquet
        |-- criterion_records.parquet
        |-- paired_metrics.parquet
        `-- manifest.json
```

`condition_predictions.parquet` is written before criterion records are loaded. The final manifest records both commit timestamps/hashes so timing can be audited. RL training readers cannot open benchmark criterion records.

### 13.3 DuckDB Analysis

Query Parquet directly with DuckDB. It supports filter and projection pushdown and avoids running a database service. See [DuckDB Parquet support](https://duckdb.org/docs/stable/data/parquet/overview).

Final analysis scripts produce:

- Dialogue-only versus probe-informed paired Brier differences.
- False-mastery, unsafe-advancement, specificity-control, and missing-pair tables.
- Case-level and model-stratified benchmark effects.
- Per-condition summaries.
- Matched-seed differences.
- Bootstrap intervals.
- Tracker calibration tables.
- CBFM controls.
- Q-table coverage.
- Action-distribution plots.
- Cost/latency tables.
- Gate decisions.

## 14. OpenRouter Integration

Use an OpenAI-compatible client behind `ModelGateway`. Do not deploy a LiteLLM proxy for the POC.

For primary experiments:

- Pin exact model ID.
- Pin provider order where supported.
- Disable provider fallback where reproducibility matters.
- Require structured-output support for structured calls.
- Pin temperature, top-p, maximum tokens, and other sampling parameters.
- Record generation ID, returned model, usage, finish reason, latency, and cost.
- Hash request prompts and final responses.

For LLM-student benchmark generation, use separate request templates and fresh provider contexts for public dialogue, evidence probe, and criterion probe. The criterion request must not include public dialogue or evidence-probe output. Store final public responses/code and request metadata, not hidden reasoning.

OpenRouter normalizes model APIs and supports structured output, usage, provider routing, and cost lookup. See [OpenRouter API](https://openrouter.ai/docs/api/reference/overview) and [provider routing](https://openrouter.ai/docs/guides/routing/provider-selection).

Never assume an LLM is deterministic even at temperature zero. Deterministic templates remain the replay and demo-fallback reference.

## 15. Guardrail and Sandbox

### 15.1 Guardrail

Start with deterministic checks for:

- Complete expected answer.
- Full solution code.
- Forbidden output fragments.
- Excessive prompt length.

Each rejected candidate receives one bounded rewrite attempt; exhaustion selects an authored template. A model-based guardrail is optional and may not replace deterministic leakage fixtures.

### 15.2 Sandbox

The first vertical slice uses text reasoning and does not require arbitrary execution.

If code submission is enabled:

- Integrate one external sandbox provider.
- Deny outbound network.
- Limit CPU, memory, wall time, process creation, filesystem, and output.
- Normalize tests, stdout, stderr, exit code, and timeout into structured evidence.
- Never fall back to executing untrusted code in the FastAPI host.

The same boundary applies to LLM-generated benchmark probe responses. Only repository-authored fixtures whose hashes appear in the frozen benchmark manifest may use the local trusted test executor.

## 16. Demo Engineering

### 16.1 Required Screen

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

For the benchmark demonstration, reveal the criterion result only after the UI has displayed committed dialogue-only and probe-informed predictions/actions. This is an explanatory replay view, not a live source of policy evidence.

### 16.2 Reliability

- Use request/response REST initially.
- Disable duplicate submit while a turn is running.
- Assign a client turn ID for idempotency.
- Persist the completed event before returning success.
- Return typed, user-safe error states.
- Fall back to templates on OpenRouter failure.
- Provide a reset-session control.

### 16.3 Deployment

For evaluation and demonstration:

- Build React static assets.
- Serve them from FastAPI or one simple PaaS deployment.
- Mount or persist SQLite and artifact directories where required.
- Keep deterministic mode available if provider credentials or network fail.

No autoscaling, queue, infrastructure-as-code, or production SLO is required.

## 17. Testing Strategy

### 17.1 Vertical Slice

- Contract validation.
- Evidence fixtures.
- Hand-calculated tracker updates.
- Heuristic decision table.
- Template leakage fixtures.
- JSONL round trip.
- FastAPI endpoint flow.
- React browser E2E turn.
- Offline execution.

### 17.2 Benchmark Measurement

- Benchmark-case and review schema validation.
- Minimum matrix and evidence-pattern coverage.
- Evidence/criterion context allowlists.
- Evidence/criterion structural-similarity warnings and review status.
- Trusted-fixture hash allowlist.
- Task-family split and template-overlap rejection.
- Criterion access forbidden before prediction/action commitment.
- Independent condition-state cloning and complete-pair enforcement.
- Hand-calculated Brier and false-mastery metric fixtures.
- Unrelated, corrupted, and shuffled-probe negative controls.
- Repeated-sample nesting and case-level aggregation.
- Benchmark freeze, hash, and version-bump behavior.

### 17.3 Mathematical Components

- CBFM boundedness, monotonicity, recovery, ablation, and seeded noise.
- BKT slip/guess/learn fixtures.
- SMC likelihood, active masks, finite weights, ESS, and resampling.
- Reward decomposition.
- Terminal/truncation updates.
- Discretizer boundaries and unseen-state fallback.

### 17.4 Experiment Integrity

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

### 17.5 Demo Reliability

- OpenRouter timeout and malformed response.
- Template fallback.
- Duplicate turn submission.
- Backend restart and checkpoint recovery.
- Empty/long/adversarial input.
- Mobile and desktop layout.
- Keyboard and focus behavior.

## 18. Implementation Phases

### Phase 0: Workspace

Create Pixi/Python/Node environments, repository skeleton, lint/type/test tasks, and React/FastAPI health smoke tests.

### Phase 1: Vertical Slice

Implement one task, deterministic evidence, simple tracker, heuristic policy, template renderer, LangGraph turn, JSONL log, API endpoints, and one React workflow.

**Exit:** browser E2E works offline.

### Phase 2: Benchmark and Measurement Freeze

Implement benchmark contracts, the 24-case minimum matrix, separate evidence/criterion probes, trusted executable tests, independent review records, task-family splits, condition-state cloning, timing guards, primary/secondary metrics, and negative controls.

Run deterministic authored fixtures first. Freeze the benchmark manifest and measurement protocol before advanced component work.

**Exit:** all `FR-BMK` requirements pass, deterministic controls recover expected directions, criterion leakage tests fail closed, and the benchmark has independent review approval.

### Phase 3: Primary LLM-Student Stress Test

Implement the minimum `ModelGateway`, separate public/evidence/criterion generation contexts, one external sandbox path for generated code, normalized execution records, and recorded-response replay. Run at least one pinned model/provider over the frozen benchmark without changing cases, labels, prompts, exclusions, or metrics in response to outcomes.

**Exit:** one complete paired report exists over the required case set, including missingness and negative controls. A positive, negative, or inconclusive result is acceptable; an invalid measurement pipeline is not.

### Phase 4: Deterministic Simulator

Implement Gymnasium environment, private truth, deterministic profiles/transitions/reward, replay, and Parquet output without importing benchmark criterion labels.

**Exit:** `check_env`, replay, timing, state-separation, and trajectory validation pass.

### Phase 5: CBFM and Tracker Baselines

Implement CBFM, BKT, and their controls. Evaluate CBFM against prompt-feature-only prediction and BKT against the simple tracker. Add reduced SMC only after a documented go/no-go decision.

**Exit:** CBFM and tracker components have retain/reject/inconclusive evidence without changing benchmark labels or primary metrics.

### Phase 6: Learned Policies

Implement contextual bandit and tabular Q-learning, coverage/visitation diagnostics, local training, and frozen artifacts.

**Exit:** retained policies complete matched deterministic conditions through one interface and cannot access benchmark criterion or simulator truth.

### Phase 7: Tutor Generation and Demo Layer

Extend the existing gateway for tutor rendering, add prompt versioning, the rule guardrail, template fallback, and selected generation-layer evaluations. Add a second LLM-student model family only if it provides replication value within the remaining budget.

**Exit:** provider failures cannot break the deterministic demo, and retained generation experiments have frozen prompts and recorded outputs.

### Phase 8: Secondary Experiments and Results Freeze

Reproduce the frozen benchmark primary study, then execute matched CBFM/tracker/policy/guardrail controls. Calculate case-level and matched-seed intervals and generate reproducible reports.

**Exit:** retain/reject/inconclusive decisions are locked before dissertation writing.

### Phase 9: Demo and Dissertation

Harden the UI, rehearse deterministic and LLM scenarios, archive artifacts, reproduce on a clean environment, and complete figures/results/limitations.

**Exit:** demo and dissertation package are submitted.

## 19. Scope Reduction Rules

If the project misses a phase exit date:

1. Preserve the vertical slice and benchmark measurement gate.
2. Preserve separate evidence/criterion probes, task-family splits, negative controls, and paired case-level analysis.
3. Preserve the deterministic simulator only after the primary benchmark path is safe.
4. Preserve heuristic and simple/BKT controls before optional tracker or policy complexity.
5. Drop model-based guardrails before deterministic guardrails.
6. Drop peer/multi-agent breadth and additional model families before the required pinned LLM-student condition.
7. Drop SMC before BKT if SMC blocks integration.
8. Drop CBFM as a retained mechanism if it misses its construct gate.
9. Drop Q-learning only with a supervisor-approved RQ amendment; never fabricate adequate coverage.
10. Drop UI embellishment before scientific analysis.
11. Never remove criterion isolation, split integrity, complete pairs, provenance, or uncertainty reporting to save time.

## 20. POC Definition of Done

- Pixi and pnpm reproduce backend/frontend environments.
- The one-task browser vertical slice works offline.
- A frozen, reviewed benchmark contains at least 24 cases with distinct evidence and criterion probes.
- Dialogue-only and probe-informed conditions commit predictions/actions before criterion reveal.
- Deterministic fixtures, unrelated probes, and corrupted-probe controls validate measurement specificity.
- At least one pinned LLM-student condition produces a reproducible primary benchmark result, with recorded-response replay.
- Paired case-level Brier, false-mastery, unsafe-advancement, missingness, and policy-disagreement reports are reproducible.
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
