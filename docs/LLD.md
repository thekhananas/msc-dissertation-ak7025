# Low-Level Design

## Master's POC: Socratic Tutoring and RL Evaluation

| Field | Value |
|---|---|
| Status | Approved POC component design |
| Revision date | 2026-07-12 |
| Product requirements | `docs/PRD.md` |
| High-level architecture | `docs/HLD.md` |
| Engineering strategy | `docs/engineering-plan.md` |
| Python package | `src/socratic_tutor` |
| Gymnasium environment | `SocraticTutor/POMDP-v0` |
| Public schema version | `1` |

> **Sequential-rescope status:** this LLD is aligned with the revised PRD, engineering plan, and HLD. `docs/data-flow-and-schema.md` and `docs/implementation-plan.md` remain on the earlier enterprise design until their separate revision stages.

## 1. Purpose and Scope

This document turns the POC architecture into implementable Python and TypeScript boundaries. It specifies the critical-path classes, data contracts, graph nodes, environment timing, mathematical updates, APIs, local persistence behavior, experiment loops, and tests.

The first implementation target is deliberately narrow:

```text
one authored Python task
-> one submitted explanation
-> one LangGraph turn
-> deterministic evidence
-> simple tracker update
-> heuristic directive
-> safe template prompt
-> one JSONL event
-> one React state update
```

The following do not belong on that path: SMC, learned policies, OpenRouter, W&B, arbitrary code execution, streaming, distributed workers, or human-study data.

Database entity details and physical Parquet layouts are finalized in `docs/data-flow-and-schema.md`. This document defines only the contracts those stores must preserve.

## 2. Implementation Principles

1. **Pure domain first.** Evidence, tracker, policy, CBFM, and simulator functions work without FastAPI or LangGraph.
2. **One owner per state field.** Graph nodes return validated deltas and cannot mutate another component's fields.
3. **No privileged policy path.** Simulator truth has a private type and is projected into an explicit policy-safe observation.
4. **Offline is the reference mode.** Templates and deterministic fixtures are the test oracle.
5. **Exactly one logical turn event.** API retries cannot create two committed events for the same turn.
6. **Local files are canonical.** W&B reports runs but does not own the only reproducible copy.
7. **Complexity is gated.** BKT precedes reduced SMC; heuristics precede bandit and Q-learning.

## 3. Source Layout and Dependency Rules

### 3.1 POC Source Layout

All paths in this section are relative to the repository's `development/` Pixi workspace.

```text
apps/
|-- api/
|   |-- main.py
|   |-- dependencies.py
|   `-- routes/
|       |-- health.py
|       `-- sessions.py
`-- web/
    `-- src/
        |-- api/client.ts
        |-- components/
        |-- pages/TutorPage.tsx
        `-- types.ts
src/socratic_tutor/
|-- contracts/
|   |-- common.py
|   |-- evidence.py
|   |-- policy.py
|   |-- session.py
|   `-- trajectory.py
|-- tasks/
|   |-- loader.py
|   `-- schema.py
|-- graph/
|   |-- state.py
|   |-- nodes.py
|   |-- builder.py
|   `-- service.py
|-- evidence/
|   `-- deterministic.py
|-- tracking/
|   |-- protocol.py
|   |-- simple.py
|   |-- bkt.py
|   `-- smc.py
|-- policies/
|   |-- protocol.py
|   |-- heuristic.py
|   |-- bandit.py
|   `-- q_learning.py
|-- generation/
|   |-- protocol.py
|   |-- templates.py
|   `-- openrouter.py
|-- guardrails/
|   `-- leakage.py
|-- cognitive/
|   |-- features.py
|   `-- dynamics.py
|-- simulator/
|   |-- private_state.py
|   |-- env.py
|   |-- transitions.py
|   `-- registration.py
|-- trajectories/
|   |-- jsonl.py
|   |-- parquet.py
|   `-- validation.py
|-- experiments/
|   |-- config.py
|   |-- rollout.py
|   |-- train.py
|   `-- evaluate.py
`-- sandbox/
    |-- protocol.py
    `-- external.py
```

The root `experiments/*.py` commands may be thin entry points. Reusable logic stays under `src/socratic_tutor` and is tested there.

### 3.2 Dependency Diagram

```d2
direction: right

interface: "POC Interfaces" {
  web: "React Client"
  api: "FastAPI Routes"
  graph: "LangGraph Turn Service"
}

domain: "Policy-Safe Domain" {
  contracts: "contracts"
  tasks: "tasks"
  evidence: "evidence"
  tracking: "tracking"
  policies: "policies"
  generation: "generation"
  guardrails: "guardrails"
  cognitive: "cognitive"
}

research: "Offline Research" {
  simulator: "simulator"
  private_state: "simulator.private_state"
  experiments: "experiments"
  trajectories: "trajectories"
}

optional: "Optional Adapters" {
  openrouter: "OpenRouter Adapter"
  sandbox: "External Sandbox Adapter"
  wandb: "W&B Sink"
}

interface.web -> interface.api: "typed HTTP"
interface.api -> interface.graph: "session commands"
interface.graph -> domain.contracts: "validated state"
interface.graph -> domain.evidence: "extract"
interface.graph -> domain.tracking: "update"
interface.graph -> domain.policies: "act"
interface.graph -> domain.generation: "render"
interface.graph -> domain.guardrails: "check"

domain.contracts -> domain.evidence
domain.contracts -> domain.tracking
domain.contracts -> domain.policies
domain.tasks -> domain.evidence
domain.tasks -> domain.generation
domain.cognitive -> research.simulator
research.private_state -> research.simulator: "private ownership"
research.simulator -> research.experiments: "Gymnasium factory"
domain.tracking -> research.experiments: "tracker factory"
domain.policies -> research.experiments: "policy factory"
research.experiments -> research.trajectories: "validated records"

domain.generation -> optional.openrouter: "optional call"
interface.graph -> optional.sandbox: "optional probe"
research.experiments -> optional.wandb: "metrics only"

research.private_state -> domain.policies: "FORBIDDEN" {
  style.stroke: "#c62828"
  style.stroke-dash: 4
}
research.private_state -> interface.graph: "FORBIDDEN" {
  style.stroke: "#c62828"
  style.stroke-dash: 4
}
```

CI shall reject imports from `socratic_tutor.simulator.private_state` in `policies`, `graph`, `generation`, `apps/api`, or frontend code. The simulator may depend on policy-safe contracts; policy code may not depend on the simulator package.

## 4. Shared Contracts

### 4.1 Contract Base Types

All externally persisted or transmitted models extend a strict base:

```python
from pydantic import BaseModel, ConfigDict

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
```

Bounded numeric aliases must reject NaN and infinity. IDs are opaque UUID strings at external boundaries. Timestamps are timezone-aware UTC values. Enum member names, never ordinal positions, are persisted.

### 4.2 Task and Evidence

```python
class TaskFixture(StrictModel):
    schema_version: Literal[1]
    task_id: str
    version: str
    title: str
    source_code: str
    target_concept: str
    prerequisite_concepts: tuple[str, ...]
    misconception_id: str
    initial_prompt: str
    evidence_rules: tuple[EvidenceRule, ...]
    templates: dict[PedagogicalDirective, str]

class ObservationEvidence(StrictModel):
    schema_version: Literal[1]
    correctness: float          # finite [0, 1]
    confidence: float           # finite [0, 1]
    misconception_present: bool
    uncertainty_reason: str | None
    active_concepts: tuple[str, ...]
    extractor_version: str
```

The deterministic extractor normalizes case and whitespace, applies task-owned positive and misconception rules, and emits a structured result. It must not infer incorrectness for concepts absent from `active_concepts`.

### 4.3 Tracker and Policy

```python
class TrackerState(StrictModel):
    schema_version: Literal[1]
    mastery: float
    misconception_probability: float
    uncertainty: float
    estimated_bandwidth: float | None = None
    estimated_friction: float | None = None
    tracker_version: str

class PolicyObservation(StrictModel):
    schema_version: Literal[1]
    target_concept: str
    tracker: TrackerState
    repeated_failures: int
    turn_index: int

class PolicyDecision(StrictModel):
    schema_version: Literal[1]
    directive: PedagogicalDirective
    target_concept: str
    policy_version: str
    propensity: float | None = None
    fallback_used: bool = False
```

`PolicyObservation` is an allowlist. Adding a field requires a schema-version decision and a feature-parity audit across every compared policy.

### 4.4 Generation and Guardrail

```python
class GenerationRequest(StrictModel):
    directive: PedagogicalDirective
    task_id: str
    task_version: str
    target_concept: str
    observable_evidence: ObservationEvidence
    template_fallback: str
    prompt_version: str

class GenerationResult(StrictModel):
    text: str
    renderer: Literal["template", "openrouter"]
    model: str | None
    provider: str | None
    prompt_hash: str
    latency_ms: int
    token_usage: TokenUsage | None

class GuardrailResult(StrictModel):
    accepted: bool
    reason_codes: tuple[str, ...]
    guardrail_version: str
```

No generation request contains simulator truth, particles, hidden model reasoning, credentials, or full prior transcripts unless a later experiment explicitly versions that context choice.

### 4.5 Core Class Diagram

```d2
direction: right

contracts: "Contracts" {
  task: "TaskFixture" {
    shape: class
    task_id: "str"
    target_concept: "str"
    evidence_rules: "tuple[EvidenceRule]"
    templates: "dict[Directive, str]"
  }
  evidence: "ObservationEvidence" {
    shape: class
    correctness: "UnitFloat"
    confidence: "UnitFloat"
    misconception_present: "bool"
    active_concepts: "tuple[str]"
  }
  tracker_state: "TrackerState" {
    shape: class
    mastery: "UnitFloat"
    misconception_probability: "UnitFloat"
    uncertainty: "UnitFloat"
    estimated_bandwidth: "UnitFloat?"
    estimated_friction: "UnitFloat?"
  }
  observation: "PolicyObservation" {
    shape: class
    target_concept: "str"
    tracker: "TrackerState"
    repeated_failures: "int"
    turn_index: "int"
  }
  decision: "PolicyDecision" {
    shape: class
    directive: "PedagogicalDirective"
    policy_version: "str"
    propensity: "float?"
  }
}

ports: "Protocols" {
  extractor: "EvidenceExtractor" {
    shape: class
    extract(response, task): "ObservationEvidence"
  }
  tracker: "BeliefTracker" {
    shape: class
    initial(task): "TrackerState"
    update(previous, evidence): "TrackerState"
  }
  policy: "TutorPolicy" {
    shape: class
    act(observation, rng): "PolicyDecision"
  }
  renderer: "PromptRenderer" {
    shape: class
    render(request): "GenerationResult"
  }
  writer: "TurnEventWriter" {
    shape: class
    append_once(event): "WriteResult"
  }
}

implementations: "Initial Implementations" {
  rules: "DeterministicEvidenceExtractor" { shape: class }
  simple: "SimpleBeliefTracker" { shape: class }
  heuristic: "HeuristicTutorPolicy" { shape: class }
  templates: "TemplatePromptRenderer" { shape: class }
  jsonl: "JsonlTurnEventWriter" { shape: class }
}

service: "TutorSessionService" {
  shape: class
  create_session(task_id): "SessionView"
  submit_turn(command): "TurnResponse"
}

contracts.task -> ports.extractor
ports.extractor -> contracts.evidence
contracts.evidence -> ports.tracker
ports.tracker -> contracts.tracker_state
contracts.tracker_state -> contracts.observation
contracts.observation -> ports.policy
ports.policy -> contracts.decision

implementations.rules -> ports.extractor: "implements"
implementations.simple -> ports.tracker: "implements"
implementations.heuristic -> ports.policy: "implements"
implementations.templates -> ports.renderer: "implements"
implementations.jsonl -> ports.writer: "implements"

service -> ports.extractor
service -> ports.tracker
service -> ports.policy
service -> ports.renderer
service -> ports.writer
```

## 5. Initial Task Fixture

The first repository fixture is `data/tasks/python_mutable_aliasing_v1.yaml`:

```python
items = [1, 2]
alias = items
alias.append(3)
```

The learner explains the final values and why mutation is visible through both names. The fixture contains:

- Target concept `python.reference_aliasing`.
- Prerequisite `python.mutable_list`.
- Misconception `assignment_copies_list`.
- Positive evidence rules for shared identity and in-place mutation.
- Misconception evidence rules for independent copies.
- Safe templates for every directive.
- A versioned rubric and authored test responses.

The loader validates fixtures at startup and rejects duplicate `(task_id, version)` pairs, unknown directives, missing templates, or unversioned evidence rules.

## 6. Vertical-Slice Algorithms

### 6.1 Deterministic Evidence Extraction

The initial extractor uses authored rules, not an LLM classifier:

1. Normalize Unicode, case, and repeated whitespace without altering the stored raw response.
2. If the response is empty, emit correctness `0.0`, confidence `1.0`, and reason `empty_response`.
3. Evaluate positive and misconception rules from the task fixture.
4. Map an unambiguous positive match to correctness `1.0`.
5. Map a misconception match without positive evidence to correctness `0.0` and set `misconception_present`.
6. Map conflicting or absent evidence to correctness `0.5` with reduced confidence.
7. Activate only the target concept and any explicitly observed prerequisite.

Rule results and versions are logged. The extractor is a deterministic fixture-based measurement instrument, not a general natural-language understanding claim.

### 6.2 Simple Tracker

The first tracker is a hand-checkable exponential update. For prior mastery `m_t`, observed correctness `e_t`, confidence `q_t`, mastery rate `eta_m`, prior misconception probability `u_t`, and misconception indicator `z_t`:

```text
m_(t+1) = clip(m_t + eta_m * q_t * (e_t - m_t), 0, 1)
u_(t+1) = clip(u_t + eta_u * q_t * (z_t - u_t), 0, 1)
c_t     = q_t * abs(2 * e_t - 1)
h_(t+1) = clip(1 - c_t, 0, 1)
```

where `h_(t+1)` is uncertainty. Defaults are fixture/config values, not learned parameters. Empty evidence may update uncertainty but must not silently update inactive concepts. Tests use decimal fixtures with expected values stated explicitly.

This tracker is a POC baseline, not Bayesian Knowledge Tracing. BKT is introduced only after this component and the complete vertical slice pass their gates.

### 6.3 Heuristic Policy

The versioned action set is:

```python
class PedagogicalDirective(StrEnum):
    ASK_PREDICTION = "ask_prediction"
    REQUEST_TRACE = "request_trace"
    PROBE_ALIASING = "probe_aliasing"
    ASK_COUNTERFACTUAL = "ask_counterfactual"
    MINIMAL_HINT = "minimal_hint"
```

Rules are evaluated top to bottom:

| Condition | Directive |
|---|---|
| `estimated_bandwidth < low_threshold` when available | `MINIMAL_HINT` |
| misconception probability at or above threshold | `PROBE_ALIASING` |
| high uncertainty | `REQUEST_TRACE` |
| repeated failures at or above limit | `MINIMAL_HINT` |
| mastery below target | `ASK_PREDICTION` |
| otherwise | `ASK_COUNTERFACTUAL` |

The heuristic stores a semantic `policy_version` and always reports propensity `1.0`. Unknown or invalid observations fail closed to the versioned `MINIMAL_HINT` fallback and log the reason.

### 6.4 Template Rendering

`TemplatePromptRenderer` retrieves the template keyed by `(task version, directive)` and substitutes only an allowlisted context. Missing templates are startup errors. Templates ask one question, avoid full code solutions, and have golden snapshots.

### 6.5 Rule-First Guardrail

The initial guardrail rejects:

- A complete final output plus explanation.
- A corrected full program.
- Direct answer markers defined by the fixture.
- Prompts exceeding the configured size.

Template prompts are prevalidated. An OpenRouter candidate receives at most one rewrite attempt; a second rejection returns the deterministic template. Rejected text is available only in restricted debugging output and never sent to the browser.

## 7. Interactive LangGraph Design

### 7.1 Graph State

```python
class TutorGraphState(TypedDict):
    schema_version: Literal[1]
    session_id: str
    task: TaskFixture
    turn_index: int
    pending_turn_id: str | None
    student_response: str | None
    evidence: ObservationEvidence | None
    tracker: TrackerState
    pending_tracker_before: TrackerState | None
    pending_tracker_after: TrackerState | None
    policy_observation: PolicyObservation | None
    decision: PolicyDecision | None
    generation: GenerationResult | None
    guardrail: GuardrailResult | None
    visible_prompt: str
    repeated_failures: int
    committed_turn_ids: tuple[str, ...]
    degraded_reasons: tuple[str, ...]
```

Simulator truth and SMC particles are prohibited. The graph checkpointer contains only interactive state. `tracker` is the last committed estimate. `validate_turn` snapshots it into `pending_tracker_before`, `update_tracker` computes `pending_tracker_after`, and `commit_turn` promotes the latter to `tracker`. Pending snapshots may remain until their owning nodes overwrite them on the next valid turn; they are ignored unless `pending_turn_id` identifies an uncommitted turn.

### 7.2 Node Order and Ownership

| Node | Reads | Sole writes |
|---|---|---|
| `validate_turn` | command, session state | `pending_turn_id`, `student_response`, `pending_tracker_before` |
| `extract_evidence` | response, task | `evidence` |
| `update_tracker` | pending tracker before, evidence | `pending_tracker_after` |
| `project_observation` | pending tracker after, turn counters | `policy_observation` |
| `select_directive` | policy observation | `decision` |
| `render_prompt` | task, evidence, decision | `generation` |
| `check_prompt` | generation | `guardrail`, safe `visible_prompt` |
| `commit_turn` | complete pending state | `tracker`, event append, counters, committed ID |

Every node returns a Pydantic-validated delta. `commit_turn` refuses incomplete state. The graph has no autonomous agent loop; one API request causes at most one policy decision and one committed turn.

### 7.3 Interactive Turn Sequence

```d2
shape: sequence_diagram

browser: "React Client"
api: "FastAPI"
service: "TutorSessionService"
graph: "LangGraph"
extractor: "Evidence Extractor"
tracker: "Simple Tracker"
policy: "Heuristic Policy"
renderer: "Template / OpenRouter"
guardrail: "Rule Guardrail"
log: "JSONL Writer"
checkpoint: "SQLite Checkpointer"

browser -> api: "POST /sessions/{id}/turns"
api -> service: "SubmitTurnCommand"
service -> graph: "invoke(thread_id=session_id)"
graph -> extractor: "extract(response, task)"
extractor -> graph: "ObservationEvidence"
graph -> tracker: "update(previous, evidence)"
tracker -> graph: "TrackerState"
graph -> policy: "act(PolicyObservation, rng)"
policy -> graph: "PolicyDecision"
graph -> renderer: "render(GenerationRequest)"
renderer -> graph: "GenerationResult or failure"
graph -> guardrail: "check(candidate)"
guardrail -> graph: "accepted or template fallback"
graph -> log: "append_once(TurnEvent)"
log -> graph: "event hash"
graph -> checkpoint: "commit graph state"
checkpoint -> graph: "checkpoint id"
graph -> service: "committed projection"
service -> api: "TurnResponse"
api -> browser: "200 JSON"
```

### 7.4 Commit and Idempotency

The client creates a UUID `turn_id` and reuses it on retry. The server enforces:

- A turn ID can belong to only one session and turn index.
- If `turn_id` is already committed, return the prior response without another tracker update or log append.
- If a different turn ID targets an already committed index, return `409 turn_conflict`.
- A turn is acknowledged only after the JSONL event is durably appended and graph state is checkpointed.

The local POC cannot atomically commit JSONL and SQLite in one transaction. Recovery therefore treats the JSONL event key `(session_id, turn_id)` as the idempotency record and rebuilds/repairs the checkpoint from that event when necessary. This limitation must be exercised by a crash-between-writes integration test.

## 8. FastAPI and React Contracts

### 8.1 Endpoints

```text
GET  /api/health
POST /api/sessions
GET  /api/sessions/{session_id}
POST /api/sessions/{session_id}/turns
```

`POST /api/sessions` accepts a task ID and mode, creates the tracker and frozen policy selection, and returns the authored initial prompt. The initial version supports `mode="demo"` only.

`POST /turns` is synchronous. WebSockets and token streaming are deferred because they add state and retry complexity without improving the one-turn research demonstration.

### 8.2 API Models

```python
class CreateSessionRequest(StrictModel):
    task_id: str
    mode: Literal["demo"] = "demo"

class SubmitTurnRequest(StrictModel):
    turn_id: UUID
    response_text: str = Field(max_length=4000)
    expected_turn_index: NonNegativeInt

class TurnResponse(StrictModel):
    session_id: UUID
    turn_id: UUID
    turn_index: NonNegativeInt
    next_prompt: str
    evidence: ObservationEvidence
    tracker: TrackerState
    decision: PolicyDecision
    session_status: Literal["active", "complete", "degraded"]
    degraded_reasons: tuple[str, ...]
```

Errors use `{code, message, request_id}` with HTTP `400`, `404`, `409`, `422`, or `503`. Internal traces and rejected model text are not returned.

### 8.3 React State

The page owns only presentation state:

```typescript
type TutorPageState = {
  session: SessionView | null;
  responseText: string;
  pendingTurnId: string | null;
  submitting: boolean;
  error: ApiError | null;
};
```

The browser preserves `pendingTurnId` across a retry, disables duplicate submission while a request is active, and renders:

- Task title and source code.
- Conversation prompts and submitted responses.
- Text response input and submit command.
- Session status.
- Expandable evidence, tracker estimate, and directive inspection.

The UI never computes mastery or policy decisions. Simulator truth is not part of this demo endpoint.

## 9. Persistence Contracts

### 9.1 JSONL Turn Event

One successfully committed turn produces one immutable `TurnEventV1` containing:

- Schema, session, turn, task, graph, tracker, policy, prompt, and guardrail versions.
- UTC event time and deterministic/non-deterministic mode.
- Raw student response for the non-research local demo only.
- Structured evidence.
- Tracker before and after.
- Policy observation and decision.
- Realized prompt plus renderer/model metadata.
- Degraded/fallback reasons.
- Latency, token, and cost fields when applicable.
- Previous event hash and current canonical event hash.

`JsonlTurnEventWriter.append_once` serializes one compact UTF-8 JSON object, writes one newline, flushes, and calls `fsync` in demo mode. It scans or indexes existing event keys at startup. A partial final line is quarantined during recovery; earlier lines remain valid.

### 9.2 SQLite Checkpoint

Use the supported LangGraph SQLite checkpointer. Application code stores only thread/checkpoint IDs and must not couple to provider-managed table internals. Schema migrations for those internal tables are outside application code.

### 9.3 Canonical Research Files

Offline experiments write:

```text
artifacts/runs/{run_id}/resolved_config.yaml
artifacts/runs/{run_id}/manifest.json
artifacts/runs/{run_id}/public/trajectories.parquet
artifacts/runs/{run_id}/privileged/truth.parquet
artifacts/runs/{run_id}/policies/{policy_hash}.npz
artifacts/runs/{run_id}/metrics.json
```

Public and privileged rows share opaque episode/turn keys but are written through separate schemas and readers. Training code receives only the public reader unless the algorithm is part of the simulator itself.

## 10. Gymnasium Environment

### 10.1 Registration and Spaces

Register `SocraticTutor/POMDP-v0` through the package entry point. `reset(seed=seed)` calls `super().reset(seed=seed)` and derives named random generators from the episode seed.

The action space is `Discrete(5)` with an artifact-persisted directive mapping. The observation space is a `Dict` containing only finite numeric encodings of:

- Tracker mastery, misconception probability, and uncertainty.
- Estimated bandwidth/friction or explicit missing-value masks.
- Repeated failures and normalized turn index.
- Target concept/task feature IDs from a frozen encoder.

Text is not part of the learning observation. Human-readable values remain in `info` only when they are policy-safe.

### 10.2 Private True State

```python
@dataclass(frozen=True, slots=True)
class SimulatorState:
    mastery: np.ndarray
    misconceptions: np.ndarray
    bandwidth: float
    friction: float
    turn_index: int
```

This type lives in `simulator/private_state.py`. The environment owns it. It is never accepted by `TutorPolicy.act` or serialized into public trajectories.

### 10.3 Step Timing

For state `S_t` and directive `a_t`, `step(a_t)` executes exactly:

1. Validate `a_t` and resolve its prompt/intervention features.
2. Compute normalized prompt load and ZPD diagnostics.
3. Update true fast state `B_(t+1)` and `F_(t+1)`.
4. Generate observable response evidence from pre-learning mastery, misconceptions, and updated fast state.
5. Update the selected tracker from observable evidence only.
6. Apply the slow mastery/misconception learning transition to produce the rest of `S_(t+1)`.
7. Compute each reward component and aggregate reward.
8. Evaluate `terminated` and `truncated` independently.
9. Project the next policy-safe observation.
10. Return `(observation, reward, terminated, truncated, info)`.

This order avoids using learning caused by the current prompt to generate the response that supposedly preceded that learning. Order-spy tests assert every transition boundary.

### 10.4 Environment Step Sequence

```d2
shape: sequence_diagram

runner: "Rollout Runner"
policy: "Tutor Policy"
env: "Gymnasium Env"
cbfm: "CBFM Dynamics"
student: "Student Kernel"
tracker: "Belief Tracker"
learning: "Learning Kernel"
reward: "Reward Model"
projector: "Observation Projector"
writer: "Parquet Writer"

runner -> policy: "act(policy-safe observation)"
policy -> runner: "directive and propensity"
runner -> env: "step(action)"
env -> cbfm: "prompt features and private state"
cbfm -> env: "next bandwidth and friction"
env -> student: "sample evidence before learning"
student -> env: "observable evidence"
env -> tracker: "update from evidence"
tracker -> env: "belief summary"
env -> learning: "apply slow transition"
learning -> env: "next mastery and misconceptions"
env -> reward: "compute named components"
reward -> env: "reward breakdown"
env -> projector: "allowlisted tracker features"
projector -> env: "next observation"
env -> runner: "obs, reward, terminated, truncated, info"
runner -> writer: "public and privileged rows"
```

### 10.5 Termination

`terminated=True` represents a task-domain terminal state such as mastery target or task completion. `truncated=True` represents an external bound such as maximum turns or invalid experiment budget. `info` contains one typed reason. The environment never returns both without an explicit precedence rule and test.

The environment must pass Gymnasium `check_env`, deterministic reset/step replay, finite-value checks, and action/observation containment in CI.

## 11. Cognitive Bandwidth and Friction

### 11.1 Prompt Features

`PromptLoadExtractor` computes declared, versioned features such as normalized token count, code density, concept count, nesting, novelty, and directive complexity. Normalizers are fit on calibration data only and then frozen.

Let normalized aggregate load be `L_t in [0,1]`, task difficulty `d_t in [0,1]`, mastery `m_t in [0,1]`, and scaffold credit `s(a_t) in [0,1]`:

```text
z_t = max(0, d_t - m_t - s(a_t))       # overload mismatch
k_t = max(0, m_t - d_t)                # underchallenge diagnostic
```

Underchallenge `k_t` is logged separately and is not relabeled as cognitive friction.

### 11.2 Bounded Dynamics

For non-negative configured coefficients:

```text
F_(t+1) = clip(
    (1 - rho_F) * F_t
    + alpha_F * z_t
    + beta_F * max(0, L_t - B_t),
    0, 1
)

B_(t+1) = clip(
    B_t
    + rho_B * (1 - B_t) * (1 - L_t)
    - delta_B * L_t
    - gamma_B * F_(t+1),
    0, 1
)
```

Constraints are validated at configuration load:

```text
0 <= rho_F, rho_B <= 1
alpha_F, beta_F, delta_B, gamma_B >= 0
```

Clipping guarantees range, not scientific validity. Property tests cover boundedness, positive-overload monotonicity, low-load recovery in expectation, deterministic replay, and sensitivity. The complete ablation bypasses the recurrence and fixes `B_t=1`, `F_t=0`, removes estimates from policy features, and emits an ablation flag.

### 11.3 Construct Evaluation Boundary

CBFM is calibrated against external synthetic load proxies or later approved instruments, never the same aggregate reward it helps define. Evaluation compares:

- `M0`: mastery/misconception features.
- `M1`: `M0` plus observable prompt-load features.
- `M2`: `M1` plus CBFM state/history.

If `M2` does not improve held-out Brier score or negative log-likelihood over `M1`, CBFM is rejected as a useful policy input. It remains a simulator construct and is never described as a direct measurement of human fatigue.

## 12. Tracker Implementations

### 12.1 Protocol

```python
class BeliefTracker(Protocol):
    version: str

    def initial(self, task: TaskFixture) -> TrackerState: ...

    def update(
        self,
        previous: TrackerState,
        evidence: ObservationEvidence,
        *,
        rng: np.random.Generator,
    ) -> TrackerState: ...
```

All trackers consume identical observable evidence and expose summaries only.

### 12.2 Delivery Order

1. `SimpleBeliefTracker`: vertical-slice baseline defined in Section 6.2.
2. `BKTBeliefTracker`: active-concept mastery using configured learn, guess, slip, and optional forget probabilities.
3. `ReducedSMCTracker`: optional, active concept plus one or two misconception flags and estimated `B_t`, `F_t`.

SMC uses log weights, log-sum-exp normalization, finite checks, effective sample size, and seeded systematic resampling. Inactive evidence masks contribute neutral likelihood. Particle arrays remain internal and never enter graph checkpoints, API responses, policy observations, or public trajectories.

SMC is retained only if it is numerically stable and improves held-out calibration or decision utility over BKT. Otherwise, BKT remains the final advanced tracker.

## 13. Policies and Training

### 13.1 Policy Protocol

```python
class TutorPolicy(Protocol):
    version: str

    def act(
        self,
        observation: PolicyObservation,
        *,
        rng: np.random.Generator,
        explore: bool,
    ) -> PolicyDecision: ...
```

Compared policies receive the same `PolicyObservation` schema and action mapping. Demo sessions set `explore=False` and freeze one policy artifact for the episode.

### 13.2 Baselines

- **Heuristic:** versioned rule table and deterministic fallback.
- **Contextual bandit:** NumPy linear value estimates with epsilon-greedy action selection; logs exact chosen-action propensity.
- **Tabular Q-learning:** compact documented bins, epsilon-greedy training, unseen-state heuristic fallback, and explicit state-action visitation counts.

There is no PPO, DQN, recurrent policy, neural network, Ray, or RLlib requirement.

### 13.3 Local Training Loop

```python
for episode_spec in manifest.episodes:
    observation, info = env.reset(seed=episode_spec.environment_seed)
    policy_rng = np.random.default_rng(episode_spec.policy_seed)
    done = False
    while not done:
        decision = policy.act(observation, rng=policy_rng, explore=True)
        next_observation, reward, terminated, truncated, info = env.step(
            action_map.to_index(decision.directive)
        )
        trainer.observe(observation, decision, reward, next_observation, terminated)
        writer.append(build_rows(...))
        observation = next_observation
        done = terminated or truncated
    trainer.end_episode()
```

The environment is constructed in process. The loop imports no FastAPI, React, LangGraph, OpenRouter, sandbox, or W&B requirement. W&B calls occur through an optional sink after local artifacts are written.

### 13.4 Training and Evaluation Sequence

```d2
shape: sequence_diagram

command: "Pixi Command"
hydra: "Hydra Config"
runner: "Local Runner"
env: "Gymnasium Env"
policy: "Trainable Policy"
parquet: "Parquet Writer"
artifact: "Local Artifact Store"
wandb: "Optional W&B Sink"
evaluator: "DuckDB Evaluator"

command -> hydra: "compose experiment"
hydra -> runner: "resolved config and hash"
runner -> artifact: "save config and manifest before rollout"
runner -> env: "construct locally"
runner -> env: "reset(matched seed)"
runner -> policy: "act(observation, explore=true)"
runner -> env: "step(action)"
env -> runner: "transition and reward components"
runner -> policy: "observe transition"
runner -> parquet: "append validated rows"
runner -> artifact: "save candidate policy and checksums"
runner -> wandb: "metrics and local artifact references"
parquet -> evaluator: "query frozen partitions"
artifact -> evaluator: "read frozen config and policy hashes"
evaluator -> artifact: "write tables, figures, gate result"
```

## 14. Experiment Configuration and Seeds

Hydra config groups are `env`, `tracker`, `policy`, `cbfm`, `generation`, and `experiment`. Before the first episode, the runner writes the fully resolved YAML plus a canonical SHA-256 hash.

One root seed derives stable named streams with `numpy.random.SeedSequence`:

```text
root
|-- episode assignment
|-- environment initialization
|-- transition sampling
|-- tracker sampling
|-- policy exploration
`-- generation sampling
```

Episode seeds derive from stable episode keys, not worker order. Reordering or resuming jobs therefore leaves completed episode trajectories unchanged. LLM generation seeds are recorded separately and never substitute for environment seeds.

The manifest includes Git commit, dirty flag, Pixi lock hash, resolved config hash, task/profile/split hashes, schema versions, action map, and output paths. Split overlap or a mutable held-out configuration aborts before rollout.

## 15. Trajectory and W&B Boundaries

### 15.1 Parquet Rows

Each turn emits:

- A public policy row with identifiers, split, seeds, policy observation, decision, propensity, observable evidence, tracker summaries, reward components, termination, versions, and costs.
- A privileged truth row with the matching key and simulator state before/after.

Writers reject non-finite values, duplicate `(run_id, episode_id, turn_index)` keys, unknown schema versions, missing reward components, or invalid termination combinations. Completed output is written to a temporary file and renamed only after schema and row-count validation.

### 15.2 W&B Sink

The sink records resolved configuration, aggregate metrics, selected example tables, and references or approved copies of local artifacts. Tests and reference experiments support `WANDB_MODE=offline` or a null sink. A W&B outage never invalidates locally completed trajectories.

W&B Sweeps may operate on policy-selection or validation data only. Held-out metrics are unavailable to sweep controllers.

## 16. OpenRouter and Sandbox Adapters

### 16.1 OpenRouter

`OpenRouterRenderer` implements `PromptRenderer` through a typed local gateway. Requests set:

- Pinned model and provider routing for experiments.
- Explicit temperature, maximum tokens, timeout, and retry count.
- Structured output when supported.
- A request/prompt hash and operation ID.

Returned model/provider metadata, usage, latency, and cost are normalized. The interactive graph permits one bounded retry/rewrite and then uses `TemplatePromptRenderer`. Training never calls this adapter.

### 16.2 Optional Code Sandbox

The vertical slice uses text and authored probes, so no arbitrary execution is required. If enabled later, `SandboxGateway` sends only code, tests, limits, and an operation ID to one external isolation provider. It returns normalized pass/fail/timeout/resource-limit evidence.

There is no host-process fallback. A provider outage yields typed unavailable evidence and leaves the text interaction usable.

## 17. Failure Semantics

| Failure | Required behavior | Test oracle |
|---|---|---|
| Invalid or oversized response | Typed `422`; no state advance | Event count unchanged |
| Duplicate committed turn ID | Return prior result | Same event hash and turn index |
| Conflicting turn index | Typed `409` | Tracker unchanged |
| Evidence rule failure | Typed uncertain evidence or `500` in strict tests | Never silently mark incorrect |
| Tracker non-finite output | Reject delta and use last valid state in degraded mode | No NaN in API/log |
| Unknown policy state/bin | Versioned heuristic fallback | Coverage miss logged |
| OpenRouter timeout/malformed output | Deterministic template | Turn remains usable |
| Guardrail rejects candidate twice | Deterministic template | Rejected candidate absent from API |
| JSONL append failure | Do not acknowledge turn | Retry does not double update |
| SQLite checkpoint failure | Return explicit non-resumable failure | JSONL recovery path tested |
| W&B unavailable | Continue locally | Artifact hashes still complete |
| Invalid Parquet partition | Mark run incomplete and exclude | Evaluator rejects partition |
| Sandbox unavailable | Typed unavailable evidence | No host execution |

All loops are bounded: one turn per request, one OpenRouter retry, one guardrail rewrite, configured episode length, and configured API/provider timeouts.

## 18. Test Design

### 18.1 Gate 1: Vertical Slice

- Task fixture schema and version tests.
- Correct, misconception, uncertain, and empty evidence fixtures.
- Hand-calculated simple-tracker fixtures.
- Heuristic rule-table branch coverage.
- Template snapshot and leakage tests.
- LangGraph node-order and single-writer tests.
- JSONL append, duplicate, partial-line, and crash-recovery tests.
- FastAPI contract and idempotency tests.
- Playwright browser test completing one turn offline.

### 18.2 Simulator and Mathematics

- Gymnasium `check_env`.
- Reset/step deterministic replay.
- Observation/action space containment.
- Timing order spies.
- Bounded and finite transition property tests.
- CBFM monotonicity, recovery, sensitivity, and complete-ablation tests.
- Reward decomposition and termination/truncation tests.
- Recursive policy-feature leakage audit.

### 18.3 Trackers and Policies

- Shared tracker and policy protocol suites.
- BKT hand calculations and calibration fixtures.
- Reduced-SMC log-weight, ESS, resampling, misspecification, and neutral-mask tests.
- Matched-observation parity across policies.
- Contextual-bandit propensity checks.
- Q-learning bin boundaries, update equation, coverage, and unseen-state fallback.
- Frozen-policy episode tests.

### 18.4 Experiment Integrity

- Hydra resolved-config snapshot and hash.
- Split overlap rejection.
- Stable seed derivation under job reordering.
- Public/privileged schema separation.
- Parquet uniqueness, finite values, completion marker, and checksums.
- W&B-disabled reproduction.
- Held-out access prohibition during tuning.
- DuckDB report reproduction from local artifacts.

### 18.5 Demo Reliability

- OpenRouter success, timeout, malformed output, and provider mismatch.
- Guardrail rejection and template fallback.
- Backend restart and session recovery.
- Duplicate browser submission.
- No-credential startup.
- No-hidden-reasoning and no-simulator-truth API scans.

## 19. Implementation Order and Stop Rules

| Order | Component | Entry condition | Exit gate |
|---:|---|---|---|
| 1 | Workspace, contracts, task fixture | Revised docs approved | Pixi tasks and schema tests pass |
| 2 | Evidence, simple tracker, heuristic, templates | Contracts frozen | Hand fixtures pass |
| 3 | LangGraph, JSONL, FastAPI, React | Pure components pass | Offline browser vertical slice passes |
| 4 | Gymnasium simulator and Parquet | Gate 1 passed | `check_env` and replay pass |
| 5 | CBFM | Simulator timing frozen | Construct tests and ablation pass |
| 6 | OpenRouter and guardrail evaluation | Offline demo stable | Failure fallback and leakage tests pass |
| 7 | BKT | Evidence schema stable | Calibration baseline passes |
| 8 | Reduced SMC | BKT baseline complete | Retain/reject gate recorded |
| 9 | Contextual bandit | Heuristic trajectories reproducible | Matched evaluation passes |
| 10 | Tabular Q-learning | Bandit baseline complete | Coverage and retain/reject gate recorded |
| 11 | Final experiments and demo hardening | Config/splits frozen | Reports and rehearsal pass |

Stop rules:

- If the vertical slice is not complete, do not begin Gymnasium/RL work.
- If BKT is sufficient, reduced SMC may be rejected without threatening the dissertation.
- If heuristic or bandit performance is sufficient, Q-learning may be a negative result.
- If OpenRouter is unreliable, demonstrate templates and report the renderer comparison separately.
- If sandbox work threatens the schedule, omit arbitrary code execution.
- Human pilot engineering remains deferred until separate approval.

## 20. Requirement Traceability

| LLD area | Primary PRD requirements |
|---|---|
| Contracts and thin slice | FR-VS-001 through FR-VS-010, NFR-MNT-001 through NFR-MNT-003 |
| LangGraph and API | FR-GRF-001 through FR-GRF-005, FR-UI-001 through FR-UI-006 |
| Simulator | FR-SIM-001 through FR-SIM-008, NFR-SEC-002 |
| CBFM | FR-CBFM-001 through FR-CBFM-009 |
| Trackers | FR-TRK-001 through FR-TRK-009 |
| Policies and local RL | FR-POL-001 through FR-POL-009 |
| OpenRouter and guardrail | FR-GEN-001 through FR-GRD-003 |
| Persistence and experiments | FR-EXP-001 through FR-EXP-006, FR-TRJ-001 through FR-TRJ-006 |
| Optional sandbox | SEC-SBX-001 through SEC-SBX-004 |
| Reproducibility and privacy | NFR-REP-001 through NFR-REP-005, NFR-SEC-001 through NFR-PRV-002 |

## 21. LLD Acceptance Criteria

This design is implementation-ready when:

- The first task, evidence rules, tracker update, heuristic rules, and templates are deterministic and hand-checkable.
- Graph state contains no simulator truth and each mutable field has one node owner.
- One request causes at most one policy action and one committed event.
- API retry/idempotency and JSONL/SQLite crash behavior are explicit.
- React contains presentation logic only and supports the complete offline turn.
- Gymnasium timing distinguishes prompt effects, response evidence, and slow learning.
- CBFM recurrence, constraints, ablation, and rejection criterion are specified.
- BKT and reduced SMC are sequenced as gated additions rather than critical-path dependencies.
- Bandit and Q-learning use direct local Gymnasium loops and the same policy-safe interface.
- Hydra, Parquet, local artifacts, DuckDB, and W&B have non-overlapping responsibilities.
- OpenRouter and sandbox adapters have deterministic failure behavior and cannot block the demo.
- Human data collection remains outside the POC.
- Every D2 diagram compiles and all referenced PRD requirement IDs resolve.
