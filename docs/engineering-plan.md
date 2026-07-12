<!-- docs/engineering-plan.md -->

# Production Engineering Plan: Socratic POMDP Tutoring System

## 1. Architecture Decisions

### 1.1 Recommended System Shape

Use a **Python-first monorepo with strict package boundaries**, deployed as independently scalable processes.

Do not begin with separate repositories or fully independent microservices. LangGraph orchestration, the Gymnasium environment, tracker, policy implementations, and evaluation code share critical state definitions and must evolve atomically during the dissertation.

Deployment boundaries should still be explicit:

| Runtime | Responsibility | Scaling profile |
|---|---|---|
| API gateway | Authentication, REST/WebSocket sessions, rate limits | Low-latency horizontal scaling |
| LangGraph workers | Live and synthetic graph execution | I/O-heavy, concurrency-limited |
| Rollout workers | Seeded simulator episodes and trajectory generation | CPU/API-call parallelism |
| RL trainer | Bandit and Q-learning updates | Batch CPU; GPU unnecessary initially |
| Evaluation workers | Ablations, judges, statistics, reports | Batch and API-call parallelism |
| Sandbox service | Untrusted code execution | Isolated, short-lived microVMs |

This gives the project monorepo-level consistency without coupling production serving to heavy training jobs.

### 1.2 Critical Ownership Boundaries

The implementation must preserve four separate forms of state:

1. **Simulator truth:** `TrueStudentState`, owned exclusively by the environment.
2. **Tracker belief:** particles and posterior summaries, owned exclusively by the epistemic tracker.
3. **Graph workflow state:** serializable orchestration state, owned by LangGraph.
4. **Policy observation:** a deliberately lossy projection of tracker-visible information.

Never place `TrueStudentState` in the same policy-facing object as tracker estimates. Keep it in a simulator-private structure and expose it only through privileged evaluation records.

The tutor policy chooses a directive and target concept:

```text
PolicyObservation -> PolicyDecision(a_t, q_t)
```

The graph subsequently realizes the complete causal intervention:

```text
PolicyDecision -> TutorControl(a_t, q_t, T_t, load_features, difficulty)
```

This distinction prevents policies from controlling arbitrary text while still allowing the environment to model prompt complexity.

### 1.3 Execution Architecture

```text
React Glass Box
       |
FastAPI REST/WebSocket Gateway
       |
LangGraph Runtime -------- Postgres Checkpointer
       |                           |
       |                           +-- resumable session state
       |
       +-- Policy Adapter -------- Policy Registry
       +-- Tutor Generator ------- LLM Gateway
       +-- Guardrail
       +-- Student Simulator ----- Sandbox Adapter
       +-- Epistemic Tracker
       |
Immutable Trajectory Sink ------- Object Storage / Parquet
                                       |
                              Rollout and Training Jobs
                                       |
                                    MLflow
```

Live serving and RL training communicate through immutable artifacts, not synchronous trainer RPC calls.

The serving application loads a frozen, versioned policy at session initialization. Training produces a new policy artifact, evaluates it, and promotes it through a registry. There is no online policy mutation during a student session.

### 1.4 LangGraph Design

Implement these nodes:

- `initialize_session`
- `update_tracker`
- `select_policy_action`
- `generate_tutor_candidate`
- `check_guardrail`
- `rewrite_or_fallback`
- `compute_prompt_intervention`
- `simulate_student`
- `collect_evidence`
- `persist_turn`
- `terminate_episode`
- `await_human_input`, disabled during synthetic experiments

Use dynamic `interrupt()` for future human input rather than relying on static `interrupt_before`. Current LangGraph documentation recommends durable checkpointers and requires pre-interrupt side effects to be idempotent because interrupted nodes restart on resume. See [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) and [persistence](https://docs.langchain.com/oss/python/langgraph/persistence).

Each node must:

- Return a state delta rather than mutate shared objects.
- Be deterministic when given recorded external responses.
- Assign idempotency keys to LLM, sandbox, and persistence operations.
- Avoid storing particle arrays or large transcripts directly in checkpoints.
- Store references to large artifacts in object storage.
- Record graph version, prompt version, model version, policy version, and random seed.

Use SQLite checkpointing locally and asynchronous PostgreSQL checkpointing in deployed environments.

## 2. Repository and Interface Design

### 2.1 Repository Layout

```text
research-brain-v2/
|-- apps/
|   |-- api/                     # FastAPI REST/WebSocket gateway
|   |-- graph_worker/            # LangGraph serving process
|   |-- training_worker/         # Training and rollout entry points
|   `-- web/                     # React Glass Box application
|-- packages/
|   |-- contracts/               # Shared Pydantic models and schema versions
|   |-- tutor_graph/             # Graph construction, nodes, routing
|   |-- simulator/               # Gymnasium environment and transition kernels
|   |-- epistemic_tracker/       # SMC, BKT, calibration and diagnostics
|   |-- cognitive_model/         # Prompt load, B_t/F_t dynamics
|   |-- policies/                # Static, heuristic, bandit, Q-learning
|   |-- generation/              # Tutor/student generators and prompt registry
|   |-- guardrails/              # Leakage classifiers and fallback templates
|   |-- sandbox/                 # E2B/Modal/local mock adapters
|   |-- trajectories/            # Episode schemas, writers, readers
|   `-- evaluation/              # Metrics, ablations, judges, statistics
|-- experiments/
|   |-- configs/                 # Frozen experiment configurations
|   |-- manifests/               # Dataset/task/profile split manifests
|   `-- notebooks/               # Exploratory analysis only
|-- tests/
|   |-- unit/
|   |-- contract/
|   |-- integration/
|   |-- simulation/
|   `-- regression/
|-- infra/
|   |-- docker/
|   |-- terraform/
|   |-- modal/
|   `-- observability/
|-- data/
|   |-- README.md
|   `-- schemas/
|-- proposal/
|-- pyproject.toml
|-- uv.lock
`-- Makefile
```

Keep production logic out of notebooks. Every reported experiment must be executable from a versioned configuration and CLI entry point.

### 2.2 Core Public Contracts

Create versioned Pydantic v2 contracts for:

```python
class PolicyObservation(BaseModel):
    schema_version: Literal["1"]
    target_concept: str
    target_mastery: float
    prerequisite_mastery_min: float
    misconception_probability: float
    estimated_bandwidth: float
    estimated_friction: float
    mastery_uncertainty: float
    bandwidth_variance: float
    repeated_failures: int
    hints_used: int
    turn_index: int
    recent_guardrail_events: int
    execution_summary: ExecutionSummary | None

class PolicyDecision(BaseModel):
    directive: PedagogicalDirective
    target_concept: str
    policy_id: str
    policy_version: str
    propensity: float | None

class TutorControl(BaseModel):
    decision: PolicyDecision
    rendered_prompt: str
    prompt_features: PromptLoadFeatures
    estimated_difficulty: float
    prompt_template_version: str
    generator_model: str
```

The contextual bandit must log action propensity. This is required for future inverse-propensity or doubly robust analysis.

Define a stable Gymnasium environment:

```python
class SocraticTutorEnv(gymnasium.Env):
    def reset(
        self,
        *,
        seed: int | None = None,
        options: EpisodeOptions | None = None,
    ) -> tuple[PolicyObservation, EpisodeInfo]: ...

    def step(
        self,
        action: PolicyDecision | int,
    ) -> tuple[
        PolicyObservation,
        float,
        bool,
        bool,
        TurnInfo,
    ]: ...
```

Register it as `SocraticTutor/POMDP-v0` and run Gymnasium's `check_env` in CI. Gymnasium recommends registered, versioned environments, deterministic seeding, explicit truncation, and contract checking; see its [custom environment guidance](https://gymnasium.farama.org/introduction/create_custom_env/).

### 2.3 Trajectory Schema

Every turn record must contain:

- Episode, turn, profile, task, cohort, and seed identifiers.
- Train/calibration/validation/test partition.
- Policy observation and decision.
- Realized `TutorControl`.
- Tracker posterior summaries and uncertainty.
- Evidence masks and observable evidence.
- Reward components separately, not only aggregate reward.
- `terminated` and `truncated` reasons.
- LLM, prompt, graph, simulator, tracker, and policy versions.
- Latency, token usage, sandbox cost, and provider retries.
- Privileged simulator state in a separately access-controlled evaluation payload.

Store trajectories as partitioned Parquet. Use JSONL only for debugging and interchange.

## 3. Tooling and Infrastructure

### 3.1 Application Stack

| Concern | Recommended tool |
|---|---|
| Language | Python 3.12 |
| Dependency management | `uv` with locked dependencies |
| API | FastAPI, Pydantic v2 |
| Graph orchestration | LangGraph |
| Database | PostgreSQL with Alembic migrations |
| Local services | Docker Compose |
| Web client | React, TypeScript, Vite, Monaco |
| Queue | Redis Streams initially |
| Object storage | S3-compatible bucket |
| Tabular data | PyArrow, Parquet, DuckDB |
| Statistics | SciPy, statsmodels, scikit-learn |
| Quality | Ruff, Pyright, pytest, Hypothesis |
| CI/CD | GitHub Actions |
| Infrastructure | Terraform |

Redis Pub/Sub alone is insufficient for durable training jobs because messages can be lost. Use Redis Streams for the initial job queue, then move to a managed queue only if operational load justifies it.

### 3.2 LLM and Sandbox Integration

Place every model provider behind a typed `ModelGateway`:

```python
class ModelGateway(Protocol):
    async def generate(
        self,
        request: GenerationRequest,
        *,
        idempotency_key: str,
    ) -> GenerationResult: ...
```

Support OpenRouter and Cerebras through provider adapters. Add:

- Per-provider concurrency limits.
- Exponential backoff with jitter.
- Hard token and cost budgets per episode.
- Prompt and response hashing.
- Cache only for deterministic synthetic experiments.
- Provider/model allowlists in experiment manifests.
- Template fallback after repeated generation failures.

Use E2B or Modal microVM execution for untrusted submissions. The local executor must be a mock for trusted fixtures, not a production security fallback. AST filtering and `multiprocessing` do not constitute a secure boundary against hostile Python.

### 3.3 Observability and Experiment Tracking

Use separate tools for separate questions:

- **LangSmith:** graph traces, node behavior, prompt regressions, evaluator datasets, and sampled LLM traces.
- **OpenTelemetry:** cross-service trace IDs, queue latency, database operations, provider calls.
- **Prometheus/Grafana:** rates, latency, failures, queue depth, cost, sandbox timeouts.
- **Sentry:** API and worker exceptions.
- **MLflow:** experiment parameters, datasets, aggregate metrics, policy artifacts, and promotion status.

LangSmith supports offline datasets, code/LLM/human evaluators, and production trace evaluation; see [LangSmith evaluation](https://docs.langchain.com/langsmith/evaluation). MLflow supports dataset-linked runs, PostgreSQL metadata, object-store artifacts, and model registration; see [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking).

Do not use LangSmith as the canonical scientific data store. Export immutable trajectory data and metric tables to object storage.

## 4. RL and Evaluation Pipeline

### 4.1 Framework Strategy

Implement the dissertation baselines directly before adopting a large RL framework:

1. Static policy.
2. Expert heuristic.
3. BKT plus heuristic.
4. Contextual bandit.
5. Tabular Q-learning.

NumPy and scikit-learn are sufficient for these policies and make the algorithms auditable. Add RLlib only for distributed rollout management or later neural-policy experiments. RLlib's current architecture scales Gymnasium environments through `EnvRunner` actors, but external-environment support on its newer stack remains specialized; use local environment construction inside workers instead of making rollout clients call a remote trainer. See [RLlib environments](https://docs.ray.io/en/latest/rllib/rllib-env.html).

Do not introduce PPO, DQN, or recurrent policies unless tabular coverage demonstrates that the compact state representation is inadequate. A neural result would otherwise weaken interpretability and expand the experimental burden without answering the thesis question.

### 4.2 Training Data Flow

```text
Experiment Manifest
        |
Rollout Coordinator
        |
Seeded Rollout Workers
  |     |     |
Local Gymnasium environments
  |     |     |
Immutable trajectories
        |
Validation and leakage checks
        |
Training job
        |
Candidate policy artifact
        |
Held-out evaluation gates
        |
Policy Registry
        |
Explicit promotion
```

Rollout workers should construct the graph and simulator in-process. Network calls remain limited to external LLM and sandbox providers.

The trainer receives normalized trajectories or environment factories. It must never access live LangGraph checkpoints, human session databases, production WebSocket connections, or unredacted privileged simulator state as policy features.

### 4.3 Training Stages

#### Stage A: Deterministic Surrogate Validation

Run transition kernels with template-based student and tutor generators. Validate reward, CBFM, belief updates, termination, and reproducibility without LLM variance.

#### Stage B: Policy Baseline Training

Train contextual bandit and tabular Q-learning over matched episode manifests. Use the same observation representation, action set, reward weights, and seeds.

Record Q-table coverage, state and state-action visitation, return, diagnostic gain, policy entropy, per-action frequency, leakage, latency, cost, and performance by profile and task subgroup.

#### Stage C: LLM-Backed Rollout

Replace templates with pinned generation models. Keep transition parameters and splits frozen. Separate model stochasticity from environment randomness using distinct seeds.

#### Stage D: Domain Randomization

Randomize only parameters declared in the manifest:

- Initial mastery and misconception distributions.
- CBFM coefficients within calibrated ranges.
- Observation/classifier noise.
- Learning and forgetting rates.
- Student response generator.
- Task and profile assignment.

Do not randomize held-out tasks into training indirectly through prompt examples or diagnostic probes.

#### Stage E: Candidate Promotion

A candidate policy is promotable only when it:

- Passes schema and deterministic replay checks.
- Uses no privileged simulator features.
- Beats or matches the incumbent on preregistered gates.
- Does not materially regress leakage, cost, or latency.
- Has a signed manifest containing code commit, data version, and configuration hash.

### 4.4 CBFM Calibration Pipeline

Treat CBFM calibration as a separate estimation problem:

1. Fit prompt-feature normalizers on calibration data only.
2. Calibrate CBFM parameters against external load proxies, not aggregate reward.
3. Freeze parameters before policy training.
4. Evaluate `M0`, `M1`, and `M2` on held-out profiles and tasks.
5. Report Brier score, negative log-likelihood, calibration error, AUROC, bootstrap intervals, and subgroup results.
6. Run monotonicity, recovery, parameter sensitivity, and negative-control tests.
7. Reject CBFM as policy input if it does not improve over prompt-feature-only `M1`.

Persist raw features, fitted normalizers, coefficient bounds, calibration split hashes, and optimization diagnostics as an MLflow artifact.

### 4.5 Compute Deployment

#### Local Development

- Docker Compose with PostgreSQL, Redis, MLflow, API, and one worker.
- DuckDB over local Parquet.
- Template or mocked model providers.
- Single-process seeded rollouts.

#### Dissertation Batch Experiments

- Modal batch jobs for independent rollout shards.
- Managed PostgreSQL for metadata.
- S3-compatible object storage for trajectories.
- CPU-only training for bandit and tabular Q-learning.
- API-bound concurrency tuned to provider rate limits.

Modal supports detached, horizontally scaled batch jobs that persist outputs externally; see [Modal batch processing](https://modal.com/docs/guide/batch-processing).

#### Scale-Up Path

Use a Ray cluster only when rollout scheduling, vectorization, or neural policies exceed Modal's simple batch model. GPUs are justified only for self-hosted LLM inference or neural policies, not tabular RL.

## 5. Implementation Roadmap and Acceptance Criteria

### Phase 0: Contracts and Reproducibility

Deliver the monorepo skeleton, locked environment, contracts, experiment manifest format, task/profile split manifests, and deterministic seed hierarchy.

Acceptance criteria:

- Contracts generate JSON Schema.
- Simulator truth cannot be imported by policy packages.
- Every run has a reproducible manifest and configuration hash.
- CI runs linting, typing, unit tests, schema compatibility, and Gymnasium checks.

### Phase 1: Mathematical Simulator

Implement prompt-load extraction, CBFM dynamics, learning/misconception kernels, reward decomposition, termination, domain randomization, and template generators.

Acceptance criteria:

- `B_t` and `F_t` remain bounded.
- Higher load and positive ZPD mismatch produce expected monotonic behavior.
- Low-load sequences recover bandwidth in expectation.
- Identical seeds reproduce complete template trajectories.
- Ablating CBFM produces exactly `B_t=1`, `F_t=0`.

### Phase 2: Tracker and Baselines

Implement SMC, active evidence masks, log-space weighting, ESS-triggered resampling, uncertainty summaries, BKT, static, and heuristic policies.

Acceptance criteria:

- No evidence updates inactive dimensions.
- Particle weights remain finite and normalized.
- Synthetic known-state tests demonstrate posterior contraction.
- Tracker parameters differ from simulator truth in robustness tests.
- BKT and heuristic baselines operate through the same policy interface.

### Phase 3: LangGraph Integration

Implement graph nodes, guardrail rewrite routing, provider adapters, checkpointing, trajectory emission, and deterministic replay with recorded external responses.

Acceptance criteria:

- Tracker is the only writer of tutor-facing belief estimates.
- Graph resumes correctly after worker failure.
- Replayed external responses reproduce state transitions.
- Guardrail retries are bounded and end in a safe template.
- Large particle sets remain outside checkpoints.

### Phase 4: Training System

Implement rollout manifests, contextual bandit and Q-learning trainers, policy artifacts, MLflow registration, and explicit promotion.

Acceptance criteria:

- Policies cannot read privileged state.
- Matched-seed baseline comparisons are automatic.
- Q-table coverage and unseen-state fallback behavior are reported.
- Interrupted jobs resume at shard boundaries without duplicate episodes.
- Frozen policy versions can be loaded by serving workers.

### Phase 5: Evaluation Gates

Implement guardrail, scratchpad, CBFM, policy-complexity, and generation-layer experiments with bootstrap and Wilson intervals.

Acceptance criteria:

- Every proposal claim maps to a machine-readable gate result.
- Calibration, policy selection, and held-out evaluation remain disjoint.
- Negative results are emitted without changing thresholds retrospectively.
- Reports include latency, token cost, leakage, and subgroup performance.

### Phase 6: Glass Box and Deployment

Implement FastAPI, WebSockets, React/Monaco UI, state visualization, authentication hooks, sandbox integration, and production telemetry.

Acceptance criteria:

- Synthetic and human modes are separate deployment configurations.
- Human mode is disabled by default and requires an explicit ethics feature flag.
- The UI never exposes raw hidden reasoning.
- True simulator state appears only in synthetic administrator views.
- Code execution has no outbound network and enforces CPU, memory, and output limits.
- WebSocket reconnection resumes by session and checkpoint ID.

### Phase 7: Operational Hardening

Add migrations, backups, retention policies, secrets management, rate limits, load tests, disaster recovery, and cost alarms.

Production readiness requires:

- Deployment rollback by graph and policy version.
- Database restore rehearsal.
- Provider-outage fallback tests.
- Queue backpressure and per-episode budget enforcement.
- PII redaction before external tracing.
- Data deletion and retention procedures before any human pilot.

## 6. Assumptions and Defaults

- The first release is a synthetic research platform, not a validated human tutoring product.
- Source remains in one monorepo through the dissertation.
- Runtime processes may scale independently but share versioned contracts.
- Python 3.12, Pydantic v2, FastAPI, LangGraph, Gymnasium, PostgreSQL, Redis Streams, Parquet, MLflow, and LangSmith are the default stack.
- Policies are frozen per episode; production performs no online learning.
- Tabular Q-learning is the most complex required policy unless state coverage falsifies its feasibility.
- Modal is the initial batch-compute platform; Ray is an optional scale-up mechanism.
- MLflow is the scientific system of record; LangSmith is the graph and LLM observability layer.
- External microVMs are mandatory for untrusted code.
- Human deployment, human data collection, and claims about actual mental fatigue remain outside the implementation until ethics and data-protection approval.
