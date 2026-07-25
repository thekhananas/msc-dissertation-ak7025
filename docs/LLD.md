# Low-Level Design

## Master's POC: Socratic Tutoring and Evidence-Validity Evaluation

| Field | Value |
|---|---|
| Status | Proposed benchmark-first amendment for review |
| Revision date | 2026-07-14 |
| Product requirements | `docs/PRD.md` |
| High-level architecture | `docs/HLD.md` |
| Engineering strategy | `docs/engineering-plan.md` |
| Python package | `src/socratic_tutor` |
| Gymnasium environment | `SocraticTutor/POMDP-v0` |
| Public schema version | `1` |

> **Sequential-amendment status:** the PRD, engineering plan, HLD, and this LLD now describe the benchmark-first POC. `docs/data-flow-and-schema.md` and `docs/implementation-plan.md` still describe the earlier POC and must be revised one at a time. The dissertation proposal has not yet been amended; the primary-study change requires supervisor approval.

## 1. Purpose and Scope

This document turns the benchmark-first POC architecture into implementable Python and TypeScript boundaries. It specifies the critical-path classes, capability boundaries, graph nodes, benchmark protocol, environment timing, mathematical updates, APIs, persistence behavior, experiment loops, and tests.

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

After that path works, the first scientific implementation target is:

```text
frozen reviewed benchmark case
-> public interaction and isolated evidence probe
-> cloned dialogue-only, probe-informed, and negative-control conditions
-> tracker prediction and policy action per condition
-> atomic complete-set commitment
-> evaluator-only reveal of a different criterion probe
-> execution-backed criterion label
-> paired case-level Brier analysis
```

Deterministic authored fixtures validate this machinery. At least one pinned LLM-student condition is required to test the actual empirical claim. Gymnasium, CBFM, BKT/SMC comparisons, and learned policies are downstream secondary studies.

Database entity details and physical Parquet layouts are finalized in `docs/data-flow-and-schema.md`. This document defines only the contracts those stores must preserve.

## 2. Implementation Principles

1. **Pure domain first.** Evidence, tracker, policy, CBFM, and simulator functions work without FastAPI or LangGraph.
2. **One owner per state field.** Graph nodes return validated deltas and cannot mutate another component's fields.
3. **No privileged policy path.** Criterion records and simulator truth have private types and never enter `PolicyObservation`.
4. **Commit before reveal.** All condition predictions and actions are durably sealed before criterion access is granted.
5. **Measurement before modelling.** Benchmark cases, splits, labels, metrics, controls, and exclusions freeze before advanced component work.
6. **Offline replay is the reference mode.** Authored fixtures and recorded provider responses reproduce the pipeline without network access.
7. **Execution outranks assertion.** The primary benchmark label comes from a different held-out executable probe and reviewed rubric, not self-report or an LLM judge.
8. **Exactly one logical turn event.** API retries cannot create two committed events for the same turn.
9. **Local files are canonical.** W&B reports runs but does not own the only reproducible copy.
10. **Complexity is gated.** BKT precedes reduced SMC; heuristics precede bandit and Q-learning.

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
|   |-- trajectory.py
|   `-- benchmark_public.py
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
|-- benchmark/
|   |-- manifest.py
|   |-- validation.py
|   |-- generation.py
|   |-- conditions.py
|   |-- commitments.py
|   |-- controls.py
|   |-- replay.py
|   |-- writer.py
|   `-- evaluator/
|       |-- private_models.py
|       |-- criterion.py
|       |-- scoring.py
|       `-- analysis.py
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

Benchmark definitions are separate from generated artifacts:

```text
data/benchmark/v1/
|-- manifest.yaml
|-- cases/
|-- public/
|-- evidence/
|-- criterion/
|-- tests/
|-- reviews/
|-- splits/
`-- limitations.md

artifacts/benchmark/version=<version>/run=<run_id>/
|-- generated_responses.parquet
|-- evidence_executions.parquet
|-- condition_predictions.parquet
|-- commitment_manifest.json
|-- criterion_records.parquet
|-- paired_case_metrics.parquet
`-- report/
```

The root command modules may be thin entry points. Reusable logic stays under `src/socratic_tutor` and is tested there. The `benchmark.evaluator` package is evaluator-owned: policies, trackers, graph code, and simulation training readers may not import it.

### 3.2 Dependency Diagram

```d2
direction: right

interface: "Interactive Interface" {
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

benchmark: "Benchmark Plane" {
  public: "benchmark public runner"
  commitments: "commitment barrier"
  replay: "recorded-response replay"
  evaluator: "benchmark.evaluator"
  criterion_private: "evaluator.private_models"
}

secondary: "Secondary Research" {
  simulator: "simulator"
  simulator_private: "simulator.private_state"
  experiments: "experiments"
  trajectories: "trajectories"
}

adapters: "External Adapters" {
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

benchmark.public -> domain.tracking: "condition evidence"
benchmark.public -> domain.policies: "policy-safe observation"
benchmark.public -> benchmark.commitments: "predictions and actions"
benchmark.commitments -> benchmark.evaluator: "sealed complete-set token"
benchmark.criterion_private -> benchmark.evaluator: "criterion-only records"
benchmark.replay -> benchmark.public: "recorded allowed channels"
benchmark.public -> adapters.openrouter: "pinned generation"
benchmark.evaluator -> adapters.sandbox: "criterion execution"

domain.cognitive -> secondary.simulator
secondary.simulator_private -> secondary.simulator: "private ownership"
secondary.simulator -> secondary.experiments: "Gymnasium factory"
domain.tracking -> secondary.experiments: "tracker factory"
domain.policies -> secondary.experiments: "policy factory"
secondary.experiments -> secondary.trajectories: "validated rows"
secondary.experiments -> adapters.wandb: "summary metrics"

benchmark.criterion_private -> domain.tracking: "FORBIDDEN" {
  style.stroke: "#c62828"
  style.stroke-dash: 4
}
benchmark.criterion_private -> domain.policies: "FORBIDDEN" {
  style.stroke: "#c62828"
  style.stroke-dash: 4
}
benchmark.criterion_private -> benchmark.public: "FORBIDDEN" {
  style.stroke: "#c62828"
  style.stroke-dash: 4
}
secondary.simulator_private -> domain.policies: "FORBIDDEN" {
  style.stroke: "#c62828"
  style.stroke-dash: 4
}
secondary.simulator_private -> interface.graph: "FORBIDDEN" {
  style.stroke: "#c62828"
  style.stroke-dash: 4
}
```

CI rejects:

- Imports from `socratic_tutor.benchmark.evaluator` in `tracking`, `policies`, `graph`, `generation`, `simulator`, or API code.
- Imports from `socratic_tutor.simulator.private_state` in `policies`, `graph`, `generation`, API code, or benchmark public runners.
- Benchmark criterion column names in policy-safe Pydantic schemas or training feature lists.

The evaluator may consume sealed condition records. It may not call tracker updates or policy actions.

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

### 4.5 Benchmark Public Contracts

The public benchmark contract deliberately has no criterion identifier, path, response, result, or label:

```python
class BenchmarkCondition(StrEnum):
    DIALOGUE_ONLY = "dialogue_only"
    PROBE_INFORMED = "probe_informed"
    UNRELATED_PROBE = "unrelated_probe"
    CORRUPTED_PROBE = "corrupted_probe"

class BenchmarkCaseView(StrictModel):
    schema_version: Literal[1]
    benchmark_version: str
    case_id: str
    task_family: str
    split: BenchmarkSplit
    target_concept: str
    misconception_id: str
    public_fixture_id: str
    evidence_probe_id: str
    initial_tracker_state: TrackerState
    case_content_hash: str

class ChannelRecord(StrictModel):
    schema_version: Literal[1]
    case_id: str
    channel: Literal["public", "evidence", "unrelated", "corrupted"]
    source: Literal["authored", "openrouter", "recorded_replay"]
    final_response: str
    execution_ref: str | None
    request_hash: str
    response_hash: str
    model_route: ModelRoute | None

class BenchmarkConditionInput(StrictModel):
    case: BenchmarkCaseView
    condition: BenchmarkCondition
    public_record: ChannelRecord
    additional_evidence: ObservationEvidence | None
    initial_state_hash: str

class CommittedPrediction(StrictModel):
    schema_version: Literal[1]
    run_id: str
    case_id: str
    sample_id: str
    condition: BenchmarkCondition
    mastery_probability: float
    uncertainty: float
    decision: PolicyDecision
    tracker_version: str
    policy_version: str
    initial_state_hash: str
    input_hash: str
    committed_at: datetime
    record_hash: str

class ConditionCommitSet(StrictModel):
    schema_version: Literal[1]
    run_id: str
    case_id: str
    sample_id: str
    expected_conditions: tuple[BenchmarkCondition, ...]
    prediction_hashes: tuple[str, ...]
    sealed_at: datetime
    seal_hash: str
```

The runner creates every `BenchmarkConditionInput` from the same serialized initial tracker state and verifies the resulting `initial_state_hash`. Trackers and policies receive ordinary `ObservationEvidence`, `TrackerState`, and `PolicyObservation`; they do not receive benchmark evaluator models.

### 4.6 Evaluator-Private Contracts

These types live only in `benchmark/evaluator/private_models.py`:

```python
class CriterionExecution(StrictModel):
    operation_id: str
    test_bundle_hash: str
    passed: int
    failed: int
    exit_code: int | None
    timed_out: bool
    resource_limited: bool
    stdout_hash: str
    stderr_hash: str

class CriterionRecord(StrictModel):
    schema_version: Literal[1]
    benchmark_version: str
    run_id: str
    case_id: str
    sample_id: str
    criterion_probe_id: str
    response_hash: str
    execution: CriterionExecution | None
    demonstrated_mastery: bool | None
    missing_reason: str | None
    label_rationale_hash: str
    reviewer_record_hash: str
    revealed_at: datetime

@dataclass(frozen=True, slots=True)
class CriterionCapability:
    run_id: str
    case_id: str
    sample_id: str
    condition_seal_hash: str
    nonce: bytes
```

`CriterionCapability` is an in-memory evaluator capability, not a serializable API model. `CommitmentStore.seal_complete_set` is the only constructor. It returns the capability only after validating all expected conditions exactly once, matching initial-state hashes, record hashes, and a durable commit-set manifest.

```python
class CommitmentStore(Protocol):
    def append_once(self, value: CommittedPrediction) -> str: ...
    def seal_complete_set(
        self,
        key: BenchmarkSampleKey,
        expected: tuple[BenchmarkCondition, ...],
    ) -> tuple[ConditionCommitSet, CriterionCapability]: ...

class CriterionRepository(Protocol):
    def reveal_spec(
        self,
        capability: CriterionCapability,
        commit_set: ConditionCommitSet,
    ) -> PrivateCriterionSpec: ...

class BenchmarkScorer(Protocol):
    def score(
        self,
        commit_set: ConditionCommitSet,
        predictions: tuple[CommittedPrediction, ...],
        criterion: CriterionRecord,
    ) -> tuple[CaseConditionScore, ...]: ...
```

The repository verifies the capability nonce in process and the persisted seal hash for audit. This is not a hostile-process security boundary; package import rules, capability construction, schema allowlists, and runtime spies jointly enforce the research boundary.

### 4.7 Core Class Diagram

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

### 4.8 Benchmark Class Diagram

```d2
direction: right

public_models: "Policy-Safe Models" {
  case: "BenchmarkCaseView" {
    shape: class
    case_id: "str"
    task_family: "str"
    public_fixture_id: "str"
    evidence_probe_id: "str"
    initial_tracker_state: "TrackerState"
  }
  input: "BenchmarkConditionInput" {
    shape: class
    condition: "BenchmarkCondition"
    public_record: "ChannelRecord"
    additional_evidence: "ObservationEvidence?"
    initial_state_hash: "str"
  }
  prediction: "CommittedPrediction" {
    shape: class
    mastery_probability: "UnitFloat"
    decision: "PolicyDecision"
    record_hash: "str"
  }
  commit_set: "ConditionCommitSet" {
    shape: class
    prediction_hashes: "tuple[str]"
    sealed_at: "datetime"
    seal_hash: "str"
  }
}

services: "Benchmark Services" {
  runner: "ConditionRunner" {
    shape: class
    run_case(case, records): "tuple[CommittedPrediction]"
  }
  store: "CommitmentStore" {
    shape: class
    append_once(prediction): "str"
    seal_complete_set(key, expected): "CommitSet + Capability"
  }
  controls: "ControlFactory" {
    shape: class
    unrelated(case, pool): "ObservationEvidence"
    corrupted(case, evidence, seed): "ObservationEvidence"
  }
}

private: "Evaluator-Only Models and Services" {
  capability: "CriterionCapability" {
    shape: class
    condition_seal_hash: "str"
    nonce: "bytes"
  }
  record: "CriterionRecord" {
    shape: class
    criterion_probe_id: "str"
    demonstrated_mastery: "bool?"
    execution: "CriterionExecution?"
  }
  repository: "CriterionRepository" {
    shape: class
    reveal_spec(capability, commit_set): "PrivateCriterionSpec"
  }
  scorer: "BenchmarkScorer" {
    shape: class
    score(commit_set, predictions, criterion): "CaseConditionScore"
  }
}

public_models.case -> public_models.input
services.controls -> public_models.input: "constructs allowed evidence"
public_models.input -> services.runner
services.runner -> public_models.prediction
public_models.prediction -> services.store
services.store -> public_models.commit_set
services.store -> private.capability: "mints only after complete seal"
private.capability -> private.repository
public_models.commit_set -> private.repository
private.repository -> private.record
public_models.prediction -> private.scorer: "after seal"
public_models.commit_set -> private.scorer
private.record -> private.scorer

private.record -> services.runner: "FORBIDDEN" {
  style.stroke: "#c62828"
  style.stroke-dash: 4
}
private.capability -> services.runner: "FORBIDDEN" {
  style.stroke: "#c62828"
  style.stroke-dash: 4
}
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

The benchmark and secondary experiments write separate trees:

```text
artifacts/benchmark/version={version}/run={run_id}/...
artifacts/simulation/run={run_id}/public/trajectories.parquet
artifacts/simulation/run={run_id}/privileged/truth.parquet
artifacts/simulation/run={run_id}/policies/{policy_hash}.npz
artifacts/simulation/run={run_id}/metrics.json
```

Benchmark condition predictions and criterion records have separate schemas and writers. The complete condition commit manifest must exist before the criterion writer can be constructed. Simulation public and privileged rows share opaque episode/turn keys but use separate schemas and readers. Training code receives only the simulation public reader; it cannot open benchmark criterion records.

## 10. Benchmark Measurement Design

### 10.1 Manifest and Case Validation

`FrozenBenchmarkManifestV1` contains:

- Benchmark version, schema version, status, freeze time, and parent version.
- At least 24 case IDs with concept, misconception, transfer-case, evidence-pattern, and task-family strata.
- Development, calibration, policy-selection, and held-out task-family assignments.
- Expected conditions and repeated-sample plan.
- Hashes for public fixtures, evidence probes, criterion probes, tests, rubrics, independent reviews, prompts, controls, metrics, and exclusions.
- Primary endpoint direction, aggregation unit, confidence-interval method, and missing-pair treatment.
- Required model/provider route and generation policy for the primary LLM-student run.

The validator rejects:

- Fewer than four concepts, two misconception states per concept, or three transfer cases per pair.
- Missing coverage of supported mastery, false public mastery, honest non-mastery, underconfidence, or ambiguous evidence.
- Shared task/probe templates across forbidden task-family splits.
- Missing independent review or unresolved adjudication.
- An expected-condition set other than dialogue-only, probe-informed, unrelated-probe, and corrupted-probe.
- Identical evidence and criterion source, test bundle, normalized AST, or answer-bearing wording.
- A changed file hash under a frozen version.
- A criterion label derived from simulator truth, aggregate reward, tracker output, or an LLM judge.

Lexical and AST similarity thresholds produce review warnings; they do not replace human review.

### 10.2 Condition Runner

For each case/sample, the runner:

1. Loads `BenchmarkCaseView`, which cannot represent criterion data.
2. Generates or replays the public response in the public channel.
3. Extracts public evidence and updates the tracker once to obtain `post_public_state`.
4. Serializes and hashes `post_public_state`; every condition starts from a fresh validated copy.
5. Generates or replays the isolated evidence-probe response.
6. Constructs the valid, unrelated, and corrupted evidence records using frozen mappings/transforms.
7. For each condition, applies either no additional evidence or exactly one additional evidence record.
8. Projects `PolicyObservation`, calls the same frozen policy, and builds `CommittedPrediction`.
9. Appends each prediction idempotently and seals the complete expected set.
10. Exits the decision phase before criterion generation begins.

```python
post_public = tracker.update(initial, public_evidence, rng=public_rng)
initial_state_bytes = canonical_json(post_public)
initial_state_hash = sha256(initial_state_bytes).hexdigest()

for condition in manifest.expected_conditions:
    state = TrackerState.model_validate_json(initial_state_bytes)
    extra = control_factory.evidence_for(condition, case, evidence_record)
    final_state = state if extra is None else tracker.update(
        state, extra, rng=condition_rng(condition)
    )
    observation = projector.from_tracker(case, final_state)
    decision = policy.act(observation, rng=policy_rng(condition), explore=False)
    commitments.append_once(build_prediction(
        condition=condition,
        tracker=final_state,
        decision=decision,
        initial_state_hash=initial_state_hash,
    ))

commit_set, capability = commitments.seal_complete_set(
    case.sample_key, manifest.expected_conditions
)
```

The dialogue-only condition uses `extra=None`; this is a no-op, not fabricated negative evidence. The same tracker, policy, versions, state encoder, and action mapping are used for all conditions. Condition order is deterministically randomized for execution but cannot affect state or RNG streams.

### 10.3 Decision and Criterion Phases

The primary command has two explicit phases:

- **Decision phase:** creates public/evidence records, predictions, actions, and one atomic commitment manifest for every required case/sample. Criterion repositories are not constructed.
- **Criterion phase:** starts only after a complete-matrix audit passes, verifies commit hashes, constructs evaluator capabilities, generates/replays criterion responses in fresh contexts, executes them, and writes criterion records.

The implementation may use two subprocesses or two separately invoked pure functions, but the criterion loader constructor requires a verified commitment manifest. Tests inject a spy loader that raises if instantiated during the decision phase.

### 10.4 Context-Isolated Generation

```python
class GenerationChannel(StrEnum):
    PUBLIC = "public"
    EVIDENCE = "evidence"
    CRITERION = "criterion"

class StudentGenerationRequest(StrictModel):
    run_id: str
    case_id: str
    sample_id: str
    channel: GenerationChannel
    system_prompt_version: str
    task_payload: dict[str, JsonValue]
    model_route: ModelRoute
    sampling: SamplingConfig
    request_hash: str
```

`ChannelContextBuilder` maintains a field allowlist per channel:

| Channel | Permitted content | Prohibited content |
|---|---|---|
| Public | Public task/dialogue fixture and public persona | Evidence probe, criterion probe, labels, tracker/policy output |
| Evidence | Evidence probe setup and minimal persona fields | Public dialogue/output, criterion content, labels, tracker/policy output |
| Criterion | Criterion probe setup and minimal persona fields | Public dialogue/output, evidence content/output, labels, tracker/policy output |

Each call uses a fresh provider request with no conversation identifier or prior messages. Cache keys include channel and request hash. A response from one channel cannot satisfy another channel's cache lookup.

Store final text/code and provider metadata, not hidden reasoning. `RecordedResponseGateway` validates request hashes and reconstructs runs without network calls.

### 10.5 Negative Controls

`ControlFactory` is configured entirely by the frozen manifest:

- **Unrelated probe:** selects a reviewed probe from a different target concept with a similar response/execution shape.
- **Corrupted probe:** applies a declared deterministic corruption to the valid evidence result, or uses a frozen within-design result permutation that never reads criterion labels.

Control assignments and transformation seeds are generated before held-out results exist and are hashed into the manifest. A control may not be selected or modified using criterion performance.

### 10.6 Criterion Execution and Scoring

The criterion is structurally different from the evidence probe and assesses transfer of the same target concept. Authored criterion fixtures use manifest-hashed trusted tests. LLM-generated criterion code uses `SandboxGateway`; there is no local fallback. The resulting label means demonstrated performance on that criterion probe, not latent human mastery.

```python
def brier(probability: float, label: bool) -> float:
    return (probability - float(label)) ** 2

delta_brier = (
    brier(dialogue_prediction.mastery_probability, criterion_label)
    - brier(probe_prediction.mastery_probability, criterion_label)
)
```

Positive `delta_brier` favors the probe-informed tracker. `BenchmarkScorer` verifies that:

- Every score uses one sealed prediction and the same criterion record for that case/sample.
- Prediction timestamps precede criterion reveal time.
- Prediction and commit hashes match the sealed manifest.
- The criterion label is execution/rubric derived.
- Missing or invalid criterion execution yields no primary score and an explicit reason.

Secondary outputs are false-mastery acceptance, false-mastery detection precision/recall, unsafe advancement, confidence, policy-action disagreement, controls, invalid execution, missingness, and exploratory policy rank reversal.

### 10.7 Statistical Aggregation

The benchmark case is the primary unit. Repeated model generations are nested within case/model:

- Aggregate repeats within case/model before the prespecified case-level comparison, or use the frozen clustered method.
- Never report repeated generations as independent case count.
- Compute paired intervals from complete case pairs under the frozen method.
- Report every denominator, missing pair, exclusion, and case-level effect.
- Stratified concept/misconception analyses are secondary unless prespecified.

A positive, negative, or inconclusive primary result is valid. Changing labels, exclusions, controls, or metrics after viewing held-out outcomes is not.

### 10.8 Benchmark Sequence

```d2
shape: sequence_diagram

command: "Benchmark Command"
manifest: "Manifest Validator"
generator: "Channel Generator / Replay"
runner: "Condition Runner"
tracker: "Frozen Tracker"
policy: "Frozen Policy"
commitments: "Commitment Store"
gate: "Criterion Gate"
sandbox: "External Sandbox"
scorer: "Benchmark Scorer"
analyzer: "Paired Analyzer"

command -> manifest: "validate version, hashes, reviews, splits"
manifest -> command: "public case views and expected matrix"
command -> generator: "public and evidence fresh-context requests"
generator -> command: "recorded responses and executions"
command -> runner: "case view plus allowed channel records"
runner -> tracker: "common public update"
tracker -> runner: "hashed post-public state"
runner -> tracker: "cloned state plus condition evidence"
tracker -> runner: "condition mastery prediction"
runner -> policy: "policy-safe observation"
policy -> runner: "condition action"
runner -> commitments: "append predictions and actions"
commitments -> commitments: "seal complete matrix atomically"
commitments -> gate: "verified seal and capability"
gate -> generator: "fresh criterion-only requests"
generator -> sandbox: "execute generated criterion code"
sandbox -> gate: "normalized execution records"
gate -> scorer: "criterion records after reveal"
commitments -> scorer: "sealed predictions"
scorer -> analyzer: "same-label condition scores"
analyzer -> command: "paired Brier, controls, intervals, missingness"
```

### 10.9 Benchmark Commands

| Command | Purpose | Network |
|---|---|---|
| `pixi run benchmark-validate` | Validate schemas, hashes, balance, reviews, splits, probe separation, and trusted tests | Never |
| `pixi run benchmark-smoke` | Run authored fixtures through every condition and recover expected metric directions | Never |
| `pixi run benchmark-generate` | Run frozen decision and criterion phases for a pinned LLM-student condition | OpenRouter and sandbox |
| `pixi run benchmark-evaluate` | Rebuild paired metrics and reports from recorded canonical artifacts | Never |

CBFM fitting, advanced tracker comparison, simulator-policy training, and held-out secondary evaluation remain blocked until benchmark validation, independent review, deterministic smoke tests, and the measurement freeze pass.

## 11. Gymnasium Environment

### 11.1 Registration and Spaces

Register `SocraticTutor/POMDP-v0` through the package entry point. `reset(seed=seed)` calls `super().reset(seed=seed)` and derives named random generators from the episode seed.

The action space is `Discrete(5)` with an artifact-persisted directive mapping. The observation space is a `Dict` containing only finite numeric encodings of:

- Tracker mastery, misconception probability, and uncertainty.
- Estimated bandwidth/friction or explicit missing-value masks.
- Repeated failures and normalized turn index.
- Target concept/task feature IDs from a frozen encoder.

Text is not part of the learning observation. Human-readable values remain in `info` only when they are policy-safe.

### 11.2 Private True State

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

### 11.3 Step Timing

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

### 11.4 Environment Step Sequence

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

### 11.5 Termination

`terminated=True` represents a task-domain terminal state such as mastery target or task completion. `truncated=True` represents an external bound such as maximum turns or invalid experiment budget. `info` contains one typed reason. The environment never returns both without an explicit precedence rule and test.

The environment must pass Gymnasium `check_env`, deterministic reset/step replay, finite-value checks, and action/observation containment in CI.

## 12. Cognitive Bandwidth and Friction

### 12.1 Prompt Features

`PromptLoadExtractor` computes declared, versioned features such as normalized token count, code density, concept count, nesting, novelty, and directive complexity. Normalizers are fit on calibration data only and then frozen.

Let normalized aggregate load be `L_t in [0,1]`, task difficulty `d_t in [0,1]`, mastery `m_t in [0,1]`, and scaffold credit `s(a_t) in [0,1]`:

```text
z_t = max(0, d_t - m_t - s(a_t))       # overload mismatch
k_t = max(0, m_t - d_t)                # underchallenge diagnostic
```

Underchallenge `k_t` is logged separately and is not relabeled as cognitive friction.

### 12.2 Bounded Dynamics

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

### 12.3 Construct Evaluation Boundary

CBFM is calibrated against external synthetic load proxies or later approved instruments, never the same aggregate reward it helps define. Its parameters are not fitted to held-out benchmark criterion labels. Evaluation compares:

- `M0`: mastery/misconception features.
- `M1`: `M0` plus observable prompt-load features.
- `M2`: `M1` plus CBFM state/history.

If `M2` does not improve held-out Brier score or negative log-likelihood over `M1`, CBFM is rejected as a useful policy input. This is a secondary construct-validity result, distinct from the primary dialogue-versus-probe benchmark contrast. CBFM remains a simulator construct and is never described as a direct measurement of human fatigue.

## 13. Tracker Implementations

### 13.1 Protocol

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

### 13.2 Delivery Order

1. `SimpleBeliefTracker`: vertical-slice baseline defined in Section 6.2.
2. `BKTBeliefTracker`: active-concept mastery using configured learn, guess, slip, and optional forget probabilities.
3. `ReducedSMCTracker`: optional, active concept plus one or two misconception flags and estimated `B_t`, `F_t`.

The frozen primary benchmark uses the simple tracker and heuristic policy unless the preregistered protocol says otherwise before freeze. BKT and SMC comparisons begin only after the primary measurement path has produced a valid report; they cannot redefine benchmark cases, criterion labels, controls, or the primary endpoint.

SMC uses log weights, log-sum-exp normalization, finite checks, effective sample size, and seeded systematic resampling. Inactive evidence masks contribute neutral likelihood. Particle arrays remain internal and never enter graph checkpoints, API responses, policy observations, or public trajectories.

SMC is retained only if it is numerically stable and improves held-out calibration or decision utility over BKT. Otherwise, BKT remains the final advanced tracker.

## 14. Policies and Training

### 14.1 Policy Protocol

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

### 14.2 Baselines

- **Heuristic:** versioned rule table and deterministic fallback.
- **Contextual bandit:** NumPy linear value estimates with epsilon-greedy action selection; logs exact chosen-action propensity.
- **Tabular Q-learning:** compact documented bins, epsilon-greedy training, unseen-state heuristic fallback, and explicit state-action visitation counts.

There is no PPO, DQN, recurrent policy, neural network, Ray, or RLlib requirement.

### 14.3 Local Training Loop

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

### 14.4 Training and Evaluation Sequence

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

## 15. Experiment Configuration and Seeds

Hydra config groups are `benchmark`, `student_generator`, `controls`, `env`, `tracker`, `policy`, `cbfm`, `generation`, and `experiment`. Before any generated response or episode, the runner writes the fully resolved YAML plus a canonical SHA-256 hash.

One root seed derives stable named streams with `numpy.random.SeedSequence`:

```text
root
|-- benchmark case/sample assignment
|-- benchmark channel generation
|-- benchmark control transformation
|-- episode assignment
|-- environment initialization
|-- transition sampling
|-- tracker sampling
|-- policy exploration
`-- generation sampling
```

Benchmark seeds derive from stable `(benchmark_version, case_id, sample_id, channel)` keys. Simulation seeds derive from stable episode keys, not worker order. Reordering or resuming jobs therefore leaves completed records unchanged. Provider seed requests are recorded but never treated as proof of deterministic LLM output; recorded responses provide replay.

The run manifest includes Git commit, dirty flag, Pixi lock hash, resolved config hash, benchmark/task/profile/split hashes, schema versions, model/provider route, action map, and output paths. Split overlap, an unfrozen benchmark, a changed held-out hash, or a model-route mismatch aborts before execution.

## 16. Trajectory and W&B Boundaries

### 16.1 Benchmark Rows

The benchmark writes separate immutable tables for:

- Generated public/evidence responses and provider metadata.
- Evidence execution records.
- Condition predictions/actions.
- Commit sets and global complete-matrix status.
- Evaluator-only criterion responses, executions, labels, and reveal metadata.
- Condition scores and paired case metrics.

Keys include `(benchmark_version, run_id, case_id, sample_id, condition)` where applicable. Criterion records omit condition because every condition for one case/sample shares the same criterion. Writers reject criterion timestamps that do not follow the global decision-phase seal.

### 16.2 Simulation Rows

Each turn emits:

- A public policy row with identifiers, split, seeds, policy observation, decision, propensity, observable evidence, tracker summaries, reward components, termination, versions, and costs.
- A privileged truth row with the matching key and simulator state before/after.

Writers reject non-finite values, duplicate `(run_id, episode_id, turn_index)` keys, unknown schema versions, missing reward components, or invalid termination combinations. Benchmark and simulation outputs are written to temporary files and renamed only after schema, key, row-count, hash, and completeness validation.

### 16.3 W&B Sink

The sink records resolved configuration, aggregate metrics, selected example tables, and references or approved copies of local artifacts. Tests and reference experiments support `WANDB_MODE=offline` or a null sink. A W&B outage never invalidates locally completed trajectories.

W&B Sweeps may operate on policy-selection or validation data only. Frozen benchmark cases, criterion labels, controls, exclusions, and held-out metrics are unavailable to sweep controllers.

## 17. OpenRouter and Sandbox Adapters

### 17.1 OpenRouter Model Gateway

`OpenRouterModelGateway` supports both benchmark student generation and optional tutor rendering. `OpenRouterRenderer` is a thin `PromptRenderer` adapter over that gateway. Requests set:

- Pinned model and provider routing for experiments.
- Explicit temperature, maximum tokens, timeout, and retry count.
- Structured output when supported.
- A request/prompt hash and operation ID.

Returned model/provider metadata, usage, latency, cost, finish reason, request ID, and generation ID are normalized. The gateway enforces per-run call/token/cost ceilings.

Benchmark requests additionally require a `GenerationChannel` and are built by channel-specific allowlists. Public, evidence, and criterion calls never reuse message history. The primary run pins at least one exact model/provider route. A route mismatch is an invalid condition, not an automatic fallback.

The interactive graph permits one bounded retry/rewrite and then uses `TemplatePromptRenderer`. Gymnasium training never calls this adapter.

### 17.2 Code Sandbox

The vertical slice uses text, so no arbitrary execution is required. The pinned LLM-student benchmark requires `SandboxGateway` whenever generated evidence or criterion code is executed. The gateway sends only code, manifest-hashed tests, limits, runtime, and an operation ID to one external isolation provider. It returns normalized test counts, exit state, timeout/resource-limit flags, and stdout/stderr hashes.

Only checked-in fixtures whose hashes match the frozen benchmark manifest may use the local trusted executor. There is no host-process fallback for generated or participant code. A sandbox outage yields typed unavailable evidence, marks the relevant case/sample missing, and never converts missing execution into failed mastery.

## 18. Failure Semantics

| Failure | Required behavior | Test oracle |
|---|---|---|
| Invalid or oversized response | Typed `422`; no state advance | Event count unchanged |
| Duplicate committed turn ID | Return prior result | Same event hash and turn index |
| Conflicting turn index | Typed `409` | Tracker unchanged |
| Evidence rule failure | Typed uncertain evidence or `500` in strict tests | Never silently mark incorrect |
| Tracker non-finite output | Reject delta and use last valid state in degraded mode | No NaN in API/log |
| Unknown policy state/bin | Versioned heuristic fallback | Coverage miss logged |
| Benchmark case/hash/review invalid | Abort before generation | No run artifact marked valid |
| Condition state hashes differ | Abort case and invalidate pair | No criterion capability minted |
| Duplicate or missing condition | Refuse complete-set seal | Criterion loader remains unavailable |
| Criterion loader constructed in decision phase | Raise integrity error and invalidate run | Spy observes zero early constructions |
| Prediction time follows criterion reveal | Reject score and invalidate run | No primary metric emitted |
| Benchmark OpenRouter timeout | Retry under frozen policy, then explicit missingness | Never substitute authored/different-model output |
| Benchmark provider-route mismatch | Reject response | Condition identity remains fixed |
| Criterion execution unavailable | Record missing criterion | Never score as non-mastery |
| Negative-control mapping missing/changed | Abort before held-out run | Frozen hash mismatch |
| OpenRouter tutor timeout/malformed output | Deterministic template | Interactive turn remains usable |
| Guardrail rejects candidate twice | Deterministic template | Rejected candidate absent from API |
| JSONL append failure | Do not acknowledge turn | Retry does not double update |
| SQLite checkpoint failure | Return explicit non-resumable failure | JSONL recovery path tested |
| W&B unavailable | Continue locally | Artifact hashes still complete |
| Invalid Parquet partition | Mark run incomplete and exclude | Evaluator rejects partition |
| Sandbox unavailable | Typed unavailable execution and missing case/sample | No host execution and no false failure label |

All loops are bounded: one turn per request, frozen benchmark retry counts, one tutor-rendering retry, one guardrail rewrite, configured episode length, and configured API/provider timeouts.

## 19. Test Design

### 19.1 Gate 1: Vertical Slice

- Task fixture schema and version tests.
- Correct, misconception, uncertain, and empty evidence fixtures.
- Hand-calculated simple-tracker fixtures.
- Heuristic rule-table branch coverage.
- Template snapshot and leakage tests.
- LangGraph node-order and single-writer tests.
- JSONL append, duplicate, partial-line, and crash-recovery tests.
- FastAPI contract and idempotency tests.
- Playwright browser test completing one turn offline.

### 19.2 Benchmark Measurement

- Schema and manifest-hash validation.
- Minimum case-matrix, evidence-pattern, and task-family balance tests.
- Independent-review and adjudication-status checks.
- Evidence/criterion source, test, AST, wording, and context-isolation checks.
- Public case-view schema scans proving criterion fields cannot deserialize.
- Import-linter rules for evaluator-private modules.
- Runtime spy proving criterion loaders are not constructed during decision phase.
- Identical post-public state hashes across all conditions.
- Exactly-once condition writes and atomic complete-set sealing.
- Missing/duplicate condition refusal and criterion-capability denial.
- Public, evidence, and criterion request-field allowlists.
- Fresh-context and channel-specific cache-key tests.
- Recorded-response request-hash and replay equivalence tests.
- Trusted-fixture allowlist and generated-code external-sandbox tests.
- Unrelated and corrupted-control frozen-assignment tests.
- Hand-calculated Brier, false-mastery, unsafe-advancement, and disagreement fixtures.
- Deterministic false-mastery, supported-mastery, honest-nonmastery, underconfidence, and ambiguous smoke cases.
- Criterion-after-commit timestamp and common-label scoring checks.
- Missing execution, incomplete-pair, denominator, and exclusion reporting.
- Nested-repeat aggregation test that prevents inflated independent case counts.
- Clean offline reproduction of the frozen benchmark summary.

### 19.3 Simulator and Mathematics

- Gymnasium `check_env`.
- Reset/step deterministic replay.
- Observation/action space containment.
- Timing-order spies.
- Bounded and finite transition property tests.
- CBFM monotonicity, recovery, sensitivity, and complete-ablation tests.
- Reward decomposition and termination/truncation tests.
- Recursive policy-feature leakage audit.
- Test proving simulator truth cannot define or overwrite benchmark criterion labels.

### 19.4 Trackers and Policies

- Shared tracker and policy protocol suites.
- BKT hand calculations and calibration fixtures.
- Reduced-SMC log-weight, ESS, resampling, misspecification, and neutral-mask tests.
- Matched-observation parity across policies.
- Contextual-bandit propensity checks.
- Q-learning bin boundaries, update equation, coverage, and unseen-state fallback.
- Frozen-policy episode tests.
- Feature-parity tests across benchmark conditions and policy baselines.

### 19.5 Experiment Integrity

- Hydra resolved-config snapshot and hash.
- Benchmark and simulation split-overlap rejection.
- Stable seed derivation under job reordering.
- Condition/criterion and simulation public/privileged schema separation.
- Parquet uniqueness, finite values, completion marker, and checksums.
- W&B-disabled reproduction.
- Held-out access prohibition during tuning.
- DuckDB report reproduction from local artifacts.
- Positive, negative, and inconclusive gate-result snapshots.

### 19.6 Demo Reliability

- OpenRouter success, timeout, malformed output, and provider mismatch.
- Guardrail rejection and template fallback.
- Backend restart and session recovery.
- Duplicate browser submission.
- No-credential startup.
- No-hidden-reasoning and no-simulator-truth API scans.
- Benchmark replay view hides criterion until committed predictions/actions are displayed.

## 20. Implementation Order and Stop Rules

| Order | Component | Entry condition | Exit gate |
|---:|---|---|---|
| 1 | Workspace, contracts, task fixture | Revised docs approved | Pixi tasks and schema tests pass |
| 2 | Evidence, simple tracker, heuristic, templates | Contracts frozen | Hand fixtures pass |
| 3 | LangGraph, JSONL, FastAPI, React | Pure components pass | Offline browser vertical slice passes |
| 4 | Benchmark contracts and authoring tools | Vertical slice passes | Public/private import and schema gates pass |
| 5 | Minimum 24-case benchmark and measurement freeze | Authoring tools pass | Independent review, splits, metrics, controls, and hashes freeze |
| 6 | Deterministic benchmark smoke run | Benchmark frozen | Expected labels, metric directions, controls, and replay pass |
| 7 | Pinned LLM-student primary run | Smoke run passes; external sandbox available | Complete paired report or explicit missingness exists |
| 8 | Gymnasium simulator and secondary Parquet | Primary measurement path valid | `check_env`, truth isolation, and replay pass |
| 9 | CBFM, BKT, then reduced SMC | Simulator timing frozen | Separate retain/reject/inconclusive gates recorded |
| 10 | Contextual bandit, then tabular Q-learning | Heuristic secondary trajectories reproduce | Coverage and matched evaluation gates recorded |
| 11 | Tutor rendering and demo hardening | Core evidence result frozen | Fallback, leakage, replay view, and rehearsal pass |
| 12 | Secondary experiments and dissertation outputs | Configurations and splits frozen | Reports reproduce from local artifacts |

Stop rules:

- If the vertical slice is not complete, do not begin Gymnasium/RL work.
- If the benchmark is not reviewed and frozen, do not fit CBFM, compare advanced trackers, train policies, or run held-out conditions.
- If deterministic benchmark controls fail, repair the measurement pipeline before any LLM claim.
- If the required pinned LLM-student run cannot execute safely, report the deterministic pipeline only and do not claim observed LLM sycophancy.
- A negative or inconclusive primary result is acceptable and must not trigger benchmark relabelling.
- If BKT is sufficient, reduced SMC may be rejected without threatening the dissertation.
- If heuristic or bandit performance is sufficient, Q-learning may be a negative result.
- Drop a second model family, tutor-generation comparison, SMC, and Q-learning before weakening the required first benchmark condition.
- Interactive code execution remains optional; generated benchmark code never receives a host-process fallback.
- Human pilot engineering remains deferred until separate approval.

## 21. Requirement Traceability

| LLD area | Primary PRD requirements |
|---|---|
| Contracts and thin slice | FR-VS-001 through FR-VS-010, NFR-MNT-001 through NFR-MNT-003 |
| LangGraph and API | FR-GRF-001 through FR-GRF-005, FR-UI-001 through FR-UI-006 |
| Benchmark contracts, isolation, and cases | FR-BMK-001 through FR-BMK-008, FR-BMK-013 through FR-BMK-016 |
| Benchmark metrics, controls, and replay | FR-BMK-009 through FR-BMK-012, FR-EXP-001 through FR-EXP-006 |
| Simulator | FR-SIM-001 through FR-SIM-008, NFR-SEC-002 |
| CBFM | FR-CBFM-001 through FR-CBFM-009 |
| Trackers | FR-TRK-001 through FR-TRK-009 |
| Policies and local RL | FR-POL-001 through FR-POL-009 |
| OpenRouter and guardrail | FR-GEN-001 through FR-GRD-003, FR-BMK-012 |
| Persistence and experiments | FR-EXP-001 through FR-EXP-006, FR-TRJ-001 through FR-TRJ-006 |
| Generated-code sandbox | SEC-SBX-001 through SEC-SBX-004, FR-BMK-003, FR-BMK-008 |
| Reproducibility and privacy | NFR-REP-001 through NFR-REP-005, NFR-SEC-001 through NFR-PRV-002 |

## 22. LLD Acceptance Criteria

This design is implementation-ready when:

- The first task, evidence rules, tracker update, heuristic rules, and templates are deterministic and hand-checkable.
- Graph state contains no simulator truth and each mutable field has one node owner.
- One request causes at most one policy action and one committed event.
- API retry/idempotency and JSONL/SQLite crash behavior are explicit.
- React contains presentation logic only and supports the complete offline turn.
- Public benchmark contracts cannot represent criterion content, result, or label.
- Every benchmark condition starts from the same hashed post-public tracker state.
- The commitment store refuses missing/duplicate conditions and is the only criterion-capability issuer.
- The decision phase completes and seals before any criterion loader or context is constructed.
- Public, evidence, and criterion generations use fresh, field-allowlisted contexts.
- Dialogue-only, probe-informed, unrelated, and corrupted conditions use the same tracker/policy versions and common criterion label.
- Primary Brier scoring is hand-checkable, paired by case, and never treats repeated generations as independent cases.
- Deterministic fixtures validate the pipeline without being presented as evidence of LLM sycophancy.
- At least one pinned LLM-student route and generated-code sandbox path are specified for the primary empirical test.
- Missing execution, incomplete pairs, exclusions, and negative controls are first-class report outputs.
- Gymnasium timing distinguishes prompt effects, response evidence, and slow learning.
- CBFM recurrence, constraints, ablation, and rejection criterion are specified.
- BKT and reduced SMC are sequenced as gated additions rather than critical-path dependencies.
- Bandit and Q-learning use direct local Gymnasium loops and the same policy-safe interface.
- Hydra, Parquet, local artifacts, DuckDB, and W&B have non-overlapping responsibilities.
- OpenRouter and sandbox adapters have explicit failure behavior; they cannot block the offline demo but are hard dependencies of the pinned generated-code condition.
- Human data collection remains outside the POC.
- Every D2 diagram compiles and all referenced PRD requirement IDs resolve.
