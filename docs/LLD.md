<!-- docs/LLD.md -->

# Low-Level Design

## Socratic Multi-Agent Tutoring and Evaluation Platform

| Field | Value |
|---|---|
| Status | Draft for engineering review |
| Design level | Components, interfaces, algorithms, and runtime sequences |
| Product requirements | `docs/PRD.md` |
| High-level architecture | `docs/HLD.md` |
| Language baseline | Python 3.12, Pydantic v2 |
| Public schema version | `1` |
| Gymnasium environment ID | `SocraticTutor/POMDP-v0` |

## 1. Purpose

This document specifies the component-level design required to implement the platform described by the PRD and HLD. It defines package dependencies, classes and protocols, state schemas, LangGraph nodes, Gymnasium behavior, mathematical execution order, API operations, idempotency, failure handling, and test seams.

Database entities and physical data layouts are introduced only where needed to define component behavior. Their complete schemas, relationships, indexes, and retention rules are deferred to `docs/data-flow-and-schema.md`.

## 2. Package and Dependency Design

### 2.1 Python Package Layout

```text
packages/
|-- contracts/
|   |-- identifiers.py
|   |-- enums.py
|   |-- policy.py
|   |-- evidence.py
|   |-- graph.py
|   |-- trajectory.py
|   `-- errors.py
|-- cognitive_model/
|   |-- features.py
|   |-- dynamics.py
|   |-- calibration.py
|   `-- diagnostics.py
|-- simulator/
|   |-- private_state.py
|   |-- env.py
|   |-- profiles.py
|   |-- transitions.py
|   |-- rewards.py
|   `-- registration.py
|-- epistemic_tracker/
|   |-- particles.py
|   |-- likelihood.py
|   |-- smc.py
|   |-- summaries.py
|   `-- baselines.py
|-- policies/
|   |-- protocol.py
|   |-- static.py
|   |-- heuristic.py
|   |-- bandit.py
|   |-- q_learning.py
|   |-- discretization.py
|   `-- artifact.py
|-- generation/
|   |-- gateway.py
|   |-- providers/
|   |-- tutor.py
|   |-- student.py
|   `-- prompts.py
|-- guardrails/
|   |-- protocol.py
|   |-- leakage.py
|   |-- rewrite.py
|   `-- fallback.py
|-- sandbox/
|   |-- gateway.py
|   |-- e2b.py
|   |-- modal.py
|   `-- trusted_fake.py
|-- tutor_graph/
|   |-- state.py
|   |-- ownership.py
|   |-- nodes/
|   |-- routing.py
|   |-- builder.py
|   `-- runtime.py
|-- trajectories/
|   |-- builder.py
|   |-- validator.py
|   |-- writer.py
|   `-- reader.py
|-- training/
|   |-- manifests.py
|   |-- coordinator.py
|   |-- rollout.py
|   |-- trainers.py
|   `-- registry.py
`-- evaluation/
    |-- gates.py
    |-- metrics.py
    |-- statistics.py
    |-- judges.py
    `-- reports.py
```

### 2.2 Dependency Rules

```d2
direction: right

safe: "Policy-Safe Packages" {
  contracts: "contracts"
  cognitive: "cognitive_model"
  tracker: "epistemic_tracker"
  policies: "policies"
  generation: "generation"
  guardrails: "guardrails"
  sandbox: "sandbox"
  trajectories: "trajectories"
}

private: "Privileged Simulator Boundary" {
  simulator: "simulator"
  true_state: "simulator.private_state"
}

orchestration: "Orchestration" {
  graph: "tutor_graph"
}

research: "Research Jobs" {
  training: "training"
  evaluation: "evaluation"
}

contracts -> cognitive: "typed inputs and outputs"
contracts -> tracker: "evidence and summaries"
contracts -> policies: "PolicyObservation / PolicyDecision"
contracts -> generation: "generation requests"
contracts -> guardrails: "guardrail results"
contracts -> sandbox: "execution contracts"
contracts -> trajectories: "trajectory schemas"

cognitive -> simulator: "true fast-state dynamics"
cognitive -> tracker: "subjective fast-state dynamics"
true_state -> simulator: "private state only"
simulator -> graph: "environment adapter"

tracker -> graph: "belief updates"
policies -> graph: "policy adapter"
generation -> graph: "tutor and student generation"
guardrails -> graph: "prompt decision"
sandbox -> graph: "execution evidence"
trajectories -> graph: "turn commit"

simulator -> training: "local environment factory"
policies -> training: "trainable implementations"
trajectories -> training: "public trajectory reader"
trajectories -> evaluation: "controlled readers"
policies -> evaluation: "candidate evaluation"

true_state -> policies: "FORBIDDEN" {
  style.stroke: "#c62828"
  style.stroke-dash: 4
}
true_state -> generation: "FORBIDDEN" {
  style.stroke: "#c62828"
  style.stroke-dash: 4
}
```

The two red dashed edges document prohibited dependencies. CI shall enforce them with an import-boundary test. `policies`, `generation`, public API code, and human-mode graph nodes may not import `simulator.private_state`.

### 2.3 Dependency Injection

Construct runtimes through factories rather than module-level singletons:

```python
@dataclass(frozen=True)
class RuntimeDependencies:
    policy_resolver: PolicyResolver
    model_gateway: ModelGateway
    guardrail: Guardrail
    sandbox_gateway: SandboxGateway
    tracker_factory: TrackerFactory
    trajectory_writer: TrajectoryWriter
    artifact_store: ArtifactStore
    clock: Clock
```

Tests inject deterministic fakes. Production factories resolve providers from validated settings and fail startup when required dependencies are unavailable or unsafe for the selected mode.

## 3. Core Types and Invariants

### 3.1 Identifier Types

Identifiers are opaque UUIDv7 strings serialized as text:

- `ExperimentId`
- `ShardId`
- `SessionId`
- `ThreadId`
- `EpisodeId`
- `TurnId`
- `TaskId`
- `ProfileId`
- `PolicyId`
- `ArtifactId`

Identifiers are generated once at command acceptance. Retries reuse the same identifier.

### 3.2 Enums

```python
class RunMode(StrEnum):
    DETERMINISTIC = "deterministic"
    LLM_SIMULATION = "llm_simulation"
    GLASS_BOX = "glass_box"
    HUMAN = "human"

class Split(StrEnum):
    CALIBRATION = "calibration"
    POLICY_SELECTION = "policy_selection"
    VALIDATION = "validation"
    HELD_OUT_TEST = "held_out_test"

class PedagogicalDirective(StrEnum):
    DECONSTRUCT_CODE = "deconstruct_code"
    PRESENT_ANALOGY = "present_analogy"
    PROMPT_PREDICTION = "prompt_prediction"
    ASK_COUNTERFACTUAL = "ask_counterfactual"
    EXPLAIN_CONCEPT = "explain_concept"
    MINIMAL_HINT = "minimal_hint"

class TerminationReason(StrEnum):
    MASTERY_COMPLETE = "mastery_complete"
    TASK_COMPLETE = "task_complete"
    MAX_TURNS = "max_turns"
    BUDGET_EXCEEDED = "budget_exceeded"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    SAFETY_ABORT = "safety_abort"
    CANCELLED = "cancelled"
```

Directive changes require an action-space version increment. Enum order is never used as a stored action index; artifacts persist an explicit directive-to-index mapping.

### 3.3 Unit-Interval Values

All mastery, probability, bandwidth, and friction fields validate as finite values in `[0,1]`. NaN and infinity are rejected before graph commit or trajectory write.

### 3.4 Core Class Diagram

```d2
direction: right

contracts: "Policy-Safe Contracts" {
  policy_observation: "PolicyObservation" {
    shape: class
    schema_version: "Literal[1]"
    target_concept: "str"
    target_mastery: "UnitFloat"
    prerequisite_mastery_min: "UnitFloat"
    misconception_probability: "UnitFloat"
    estimated_bandwidth: "UnitFloat"
    estimated_friction: "UnitFloat"
    posterior_uncertainty: "PosteriorUncertainty"
    counters: "TurnCounters"
    execution_summary: "ExecutionSummary | None"
  }

  policy_decision: "PolicyDecision" {
    shape: class
    directive: "PedagogicalDirective"
    target_concept: "str"
    policy_id: "PolicyId"
    policy_version: "str"
    propensity: "UnitFloat | None"
  }

  tutor_control: "TutorControl" {
    shape: class
    decision: "PolicyDecision"
    rendered_prompt: "str"
    prompt_features: "PromptLoadFeatures"
    estimated_difficulty: "UnitFloat"
    prompt_template_version: "str"
    generator_model: "str"
  }

  evidence: "ObservationEvidence" {
    shape: class
    student_text: "str"
    execution: "ExecutionSummary | None"
    probe: "ProbeResult | None"
    guardrail: "GuardrailEvent"
    load_proxy: "LoadProxySignals"
    active_masks: "EvidenceMasks"
  }

  estimates: "TrackerEstimates" {
    shape: class
    mastery: "dict[str, UnitFloat]"
    misconception_marginals: "dict[str, UnitFloat]"
    cognitive: "CognitiveEstimate"
    effective_sample_size: "float"
    posterior_uncertainty: "PosteriorUncertainty"
  }
}

private: "Simulator-Private Types" {
  true_state: "TrueStudentState" {
    shape: class
    mastery: "dict[str, UnitFloat]"
    misconceptions: "dict[str, bool]"
    bandwidth: "UnitFloat"
    overload_friction: "UnitFloat"
    underchallenge: "UnitFloat"
  }

  particle: "BeliefParticle" {
    shape: class
    state: "LatentStateHypothesis"
    log_weight: "float"
  }
}

services: "Domain Services" {
  env: "SocraticTutorEnv" {
    shape: class
    +reset: "(seed, options) -> (PolicyObservation, EpisodeInfo)"
    +step: "(action) -> StepResult"
    -true_state: "TrueStudentState"
    -tracker: "EpistemicTracker"
  }

  tracker: "EpistemicTracker" {
    shape: class
    +initialize: "(prior, seed) -> TrackerEstimates"
    +update: "(control, evidence) -> TrackerEstimates"
    +snapshot_ref: "() -> ArtifactRef | None"
  }

  policy: "TutorPolicy" {
    shape: class
    +act: "(PolicyObservation, PolicyContext) -> PolicyDecision"
    +metadata: "() -> PolicyMetadata"
  }

  graph: "TutorGraphRuntime" {
    shape: class
    +start: "(SessionCommand) -> GraphResult"
    +resume: "(ThreadId, ResumeCommand) -> GraphResult"
    +stream: "(ThreadId) -> AsyncIterator[GraphEvent]"
  }

  writer: "TrajectoryWriter" {
    shape: class
    +append_turn: "(TurnEnvelope) -> CommitReceipt"
    +finalize_episode: "(EpisodeSummary) -> ArtifactRef"
  }
}

contracts.policy_decision -> contracts.tutor_control: "embedded in"
contracts.tutor_control -> contracts.evidence: "causes observable"
contracts.evidence -> services.tracker: "corrects belief"
services.tracker -> contracts.estimates: "produces"
contracts.estimates -> contracts.policy_observation: "projected to"
contracts.policy_observation -> services.policy: "input"
services.policy -> contracts.policy_decision: "output"

private.true_state -> services.env: "owned by"
private.particle -> services.tracker: "owned by"
services.env -> services.tracker: "invokes subjective update"
services.env -> services.writer: "commits trajectory"
services.graph -> services.env: "drives in synthetic mode"
services.graph -> services.writer: "commits live turns"
```

## 4. Public Contract Specifications

### 4.1 `PolicyObservation`

The observation is the only input accepted by a policy:

```python
class PolicyObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    target_concept: str
    target_mastery: UnitFloat
    prerequisite_mastery_min: UnitFloat
    misconception_probability: UnitFloat
    estimated_bandwidth: UnitFloat
    estimated_friction: UnitFloat
    mastery_entropy: NonNegativeFloat
    bandwidth_variance: UnitFloat
    repeated_failures: NonNegativeInt
    hints_used: NonNegativeInt
    turn_index: NonNegativeInt
    recent_guardrail_events: NonNegativeInt
    execution_summary: ExecutionSummary | None = None
```

The policy adapter constructs this object from `TrackerEstimates`, task context, and counters. `extra="forbid"` prevents accidental privileged fields from being ignored silently.

### 4.2 `PolicyDecision`

```python
class PolicyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    directive: PedagogicalDirective
    target_concept: str
    policy_id: PolicyId
    policy_version: str
    propensity: UnitFloat | None
    action_space_version: Literal["1"] = "1"
```

Deterministic policies set `propensity=None` and `is_deterministic=true` in policy metadata. Stochastic policies must provide the probability assigned to the chosen action.

### 4.3 `PromptLoadFeatures`

```python
class PromptLoadFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    token_count_normalized: UnitFloat
    code_span_normalized: UnitFloat
    ast_depth_normalized: UnitFloat
    concept_novelty_normalized: UnitFloat
    entropy_normalized: UnitFloat
    concept_count: NonNegativeInt
    aggregate_load: UnitFloat
    normalizer_version: str
```

Features are computed from the final guardrail-approved prompt, not the rejected candidate.

### 4.4 `ObservationEvidence`

```python
class ObservationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    student_text: str
    execution: ExecutionSummary | None
    probe: ProbeResult | None
    guardrail_event: GuardrailEvent
    load_proxy: LoadProxySignals
    concept_probabilities: dict[str, UnitFloat]
    misconception_probabilities: dict[str, UnitFloat]
    active_concepts: frozenset[str]
    active_misconceptions: frozenset[str]
```

Empty active sets yield a neutral likelihood update. They do not imply failure or non-mastery.

### 4.5 `StepResult`

The Gymnasium tuple is assembled from:

```python
class StepResult(BaseModel):
    observation: PolicyObservation
    reward: FiniteFloat
    terminated: bool
    truncated: bool
    info: TurnInfo
```

`TurnInfo` is not passed to the policy. Privileged fields are nested under a separate evaluation-only structure and are removed by the policy wrapper.

## 5. LangGraph State and Nodes

### 5.1 Graph State

```python
class TutorGraphState(TypedDict, total=False):
    schema_version: Literal["1"]
    session: SessionContext
    task: TaskContext
    tracker_estimates: TrackerEstimates
    latest_policy_observation: PolicyObservation
    latest_policy_decision: PolicyDecision
    latest_tutor_control: TutorControl
    latest_evidence: ObservationEvidence
    latest_guardrail_result: GuardrailResult
    counters: TurnCounters
    budget: BudgetState
    artifact_refs: ArtifactRefs
    termination: TerminationState
    pending_human_prompt: HumanPrompt | None
```

Synthetic `TrueStudentState` is held by the environment instance referenced through runtime context, not serialized into the policy-safe graph-state contract. An optional privileged snapshot reference may be attached to `artifact_refs` for authorized evaluation.

### 5.2 Field Ownership

| Field group | Single writer | Readers |
|---|---|---|
| `session`, `task` | `initialize_session` | All nodes |
| `tracker_estimates` | `update_tracker` | Policy, tutor, UI projection |
| `latest_policy_observation` | `select_policy_action` adapter | Policy and trajectory writer |
| `latest_policy_decision` | `select_policy_action` | Tutor, environment, trajectory writer |
| `latest_tutor_control` | `compute_prompt_intervention` | Student/environment, tracker, writer |
| `latest_guardrail_result` | `check_guardrail` | Rewrite router and writer |
| `latest_evidence` | `collect_evidence` | Tracker and writer |
| `counters`, `budget` | `persist_turn` | Routing and API projection |
| `termination` | `terminate_episode` | Runtime and client projection |
| `pending_human_prompt` | `await_human_input` | Runtime and WebSocket adapter |

The ownership validator receives the current node name and state delta. It rejects keys outside the node's allowlist before checkpoint persistence.

### 5.3 Node Contracts

| Node | Required inputs | Writes | Side effects |
|---|---|---|---|
| `initialize_session` | Validated start command | Session, task, initial tracker state, counters, budget | Resolve frozen policy; create session metadata |
| `update_tracker` | Previous control and evidence, or prior | Tracker estimates | Optional particle artifact snapshot |
| `select_policy_action` | Tracker estimates, task, counters | Policy observation and decision | Load frozen policy artifact if not cached |
| `generate_tutor_candidate` | Decision, task, safe context | Candidate held in node-local result | Model call or template lookup |
| `check_guardrail` | Candidate and task leakage fixture | Guardrail result | Guardrail classifier call if configured |
| `rewrite_or_fallback` | Rejected result and bounded retry count | New candidate or fallback marker | Bounded model call or template lookup |
| `compute_prompt_intervention` | Accepted prompt and decision | Tutor control | Feature extraction only |
| `simulate_student` | Tutor control and environment handle | Raw student/probe results in runtime context | Model and sandbox calls |
| `collect_evidence` | Raw results and guardrail event | Observation evidence | Structured classifier call if configured |
| `persist_turn` | Complete turn envelope | Counters, budget, artifact refs | Atomic trajectory append and metadata update |
| `terminate_episode` | State and environment terminal status | Termination state | Finalize trajectory and summary |
| `await_human_input` | Approved tutor control | Pending prompt then resumed raw input | Dynamic interrupt; no pre-interrupt non-idempotent write |

### 5.4 Routing Rules

```text
initialize_session
  -> update_tracker
  -> select_policy_action
  -> generate_tutor_candidate
  -> check_guardrail

check_guardrail.accepted
  -> compute_prompt_intervention

check_guardrail.rejected AND rewrites_remaining
  -> rewrite_or_fallback
  -> check_guardrail

check_guardrail.rejected AND no_rewrites_remaining
  -> rewrite_or_fallback.safe_template
  -> compute_prompt_intervention

compute_prompt_intervention
  -> simulate_student       [synthetic]
  -> await_human_input      [human]

simulate_student OR await_human_input.resume
  -> collect_evidence
  -> persist_turn
  -> terminate_episode      [terminal/truncated]
  -> update_tracker         [continue]
```

All loops are bounded by manifest values. Routing evaluates budget exhaustion before issuing another external call.

## 6. Synthetic Episode Sequence

```d2
synthetic_episode: "Synthetic Episode Turn" {
  shape: sequence_diagram

  client: "Client / Rollout Worker"
  graph: "TutorGraphRuntime"
  tracker: "EpistemicTracker"
  policy: "TutorPolicy"
  tutor: "TutorGenerator"
  guardrail: "Guardrail"
  env: "SocraticTutorEnv"
  student: "StudentGenerator"
  sandbox: "SandboxGateway"
  writer: "TrajectoryWriter"
  checkpoint: "Postgres Checkpointer"

  client -> graph: "start or continue episode"
  graph -> tracker: "initialize prior or update(previous control, evidence)"
  tracker -> graph: "TrackerEstimates"
  graph -> policy: "act(PolicyObservation)"
  policy -> graph: "PolicyDecision"
  graph -> tutor: "render(decision, safe context)"
  tutor -> graph: "candidate prompt"
  graph -> guardrail: "check(candidate, task policy)"
  guardrail -> graph: "accepted, rejected, or fallback instruction"
  graph -> env: "realize TutorControl from accepted prompt"
  env -> env: "compute prompt load, F_t, and B_t"
  env -> student: "generate bounded-context response and probe attempt"
  student -> sandbox: "execute diagnostic code with idempotency key"
  sandbox -> student: "normalized ExecutionSummary"
  student -> env: "public response and private probe evidence"
  env -> env: "apply slow mastery and misconception transition"
  env -> graph: "ObservationEvidence, reward parts, terminal status"
  graph -> tracker: "stage evidence for next belief update"
  graph -> writer: "append immutable TurnEnvelope"
  writer -> graph: "CommitReceipt"
  graph -> checkpoint: "commit graph state and artifact reference"
  checkpoint -> graph: "checkpoint id"
  graph -> client: "committed turn event"
}
```

The trajectory commit occurs before the turn is announced as committed. A client disconnect after commit is recovered through checkpoint history rather than by rerunning the turn.

## 7. Gymnasium Environment Design

### 7.1 Construction

```python
class SocraticTutorEnv(gymnasium.Env[PolicyObservation, int]):
    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(self, config: EnvironmentConfig) -> None: ...

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[PolicyObservation, EpisodeInfo]: ...

    def step(
        self,
        action: int,
    ) -> tuple[PolicyObservation, float, bool, bool, TurnInfo]: ...
```

The policy-facing registered environment uses a discrete action index mapped through the versioned action-space manifest. Internal graph code may use `PolicyDecision` directly through an adapter.

### 7.2 `reset` Algorithm

1. Call `super().reset(seed=seed)`.
2. Validate `EpisodeOptions` and split authorization.
3. Derive environment, transition, observation, policy-exploration, and generation seeds.
4. Load or sample the profile, task, prerequisite DAG, initial misconceptions, and private state.
5. Initialize CBFM state with configured `B_0` and `F_0`.
6. Initialize the tracker from its subjective prior; do not copy private truth unless the manifest explicitly defines a controlled oracle baseline.
7. Resolve the initial target concept.
8. Project `TrackerEstimates` into `PolicyObservation`.
9. Return observation and non-policy `EpisodeInfo`.

### 7.3 `step` Transaction

1. Reject calls after termination until `reset`.
2. Resolve the action index to a directive and validate target selection.
3. Render and guardrail the final tutor prompt.
4. Compute `PromptLoadFeatures` from the final prompt.
5. Compute fast true-state projection:
   - ZPD mismatch and overload friction.
   - Underchallenge diagnostic.
   - Bandwidth drain, recovery, and optional seeded noise.
6. Emit public and private student evidence from the post-prompt state.
7. Apply the slow private mastery/misconception transition.
8. Compute decomposed simulator reward.
9. Update the subjective tracker using control and evidence.
10. Build the next policy observation from tracker summaries only.
11. Evaluate termination and truncation independently.
12. Build `TurnInfo`, keeping privileged data outside the policy wrapper.
13. Return the Gymnasium tuple.

An exception before the slow transition leaves the in-memory environment turn uncommitted. Batch orchestration retries the whole deterministic turn with the same seed and idempotency keys.

### 7.4 Reward Decomposition

```python
class RewardComponents(BaseModel):
    mastery_gain: float
    overload_penalty: float
    bandwidth_penalty: float
    leakage_penalty: float
    optional_efficiency_penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.mastery_gain
            - self.overload_penalty
            - self.bandwidth_penalty
            - self.leakage_penalty
            - self.optional_efficiency_penalty
        )
```

Reward weights are immutable within an experiment. Every component is stored separately. Held-out diagnostic outcomes remain distinct from simulator reward.

## 8. Cognitive Model Design

### 8.1 Prompt Feature Pipeline

`PromptLoadExtractor.extract(prompt, task, demonstrated_concepts)` performs:

1. Tokenization with a pinned tokenizer version.
2. Code-span detection and normalized code volume.
3. AST parsing of displayed Python snippets; parse failure is recorded and uses a configured conservative feature value.
4. Concept extraction against the versioned domain ontology.
5. Novelty calculation relative to demonstrated tracker-visible concepts for the subjective model and true demonstrated concepts for the simulator model.
6. Lexical/syntactic entropy calculation.
7. Calibration-normalizer application.
8. Weighted aggregate computation with non-negative weights summing to one.

Raw and normalized features are both retained in the scientific trajectory, while only normalized features enter `TutorControl`.

### 8.2 CBFM Interface

```python
class CognitiveDynamics(Protocol):
    def project_fast(
        self,
        prior: CognitiveState,
        prompt: PromptLoadFeatures,
        tutor_difficulty: float,
        student_difficulty: float,
        rng: numpy.random.Generator,
    ) -> CognitiveProjection: ...
```

`CognitiveProjection` contains `bandwidth`, `overload_friction`, `underchallenge`, `overload_probability`, and decomposed drain/recovery terms.

The implementation clips only the final bandwidth result. Diagnostics record pre-projection values so excessive clipping is detectable during calibration.

### 8.3 Calibration Artifacts

The calibrator produces an immutable artifact containing:

- Feature-normalizer version and fitted statistics.
- CBFM coefficients and constraints.
- Calibration split hash.
- Objective and optimizer configuration.
- Fit diagnostics and convergence state.
- Monotonicity, recovery, sensitivity, and negative-control outcomes.

Policy-selection and held-out runs accept only a frozen calibration artifact ID.

## 9. Epistemic Tracker Design

### 9.1 Tracker Protocol

```python
class EpistemicTracker(Protocol):
    def initialize(
        self,
        prior: TrackerPrior,
        rng: numpy.random.Generator,
    ) -> TrackerEstimates: ...

    def update(
        self,
        control: TutorControl,
        evidence: ObservationEvidence,
        rng: numpy.random.Generator,
    ) -> TrackerEstimates: ...
```

### 9.2 SMC Update

For each particle, `SMCTracker.update` performs:

1. **Fast projection:** Apply subjective `CognitiveDynamics`; mastery and misconception remain unchanged.
2. **Active extraction:** Validate active masks against known concept and misconception IDs.
3. **Likelihood:** Compute masked mastery and misconception likelihood terms in log space.
4. **Smoothing:** Apply the configured likelihood floor using stable log-add-exp operations.
5. **Normalization:** Subtract `logsumexp`; reject an all-non-finite posterior.
6. **Diagnostics:** Compute ESS, weight entropy, marginal entropy, and bandwidth variance before resampling.
7. **Resampling:** Apply systematic resampling when ESS is below the threshold and reset weights uniformly.
8. **Slow propagation:** Apply subjective learning and misconception transitions after correction.
9. **Summary:** Produce posterior means and marginals for the next turn.

### 9.3 Numerical Failure Policy

- A single invalid likelihood term fails the evidence update with a typed `TrackerNumericalError`.
- The tracker never silently replaces a failed posterior with simulator truth.
- Configurable recovery may retain the prior with an explicit degraded-status flag, but primary experiments treat such events as run failures unless preregistered otherwise.
- Particle snapshots are stored only for selected diagnostics because they are large and privileged.

### 9.4 Baseline Trackers

- `BKTTracker` maintains per-concept learned/unlearned probabilities with slip, guess, learn, and optional forget parameters.
- `LastObservationTracker` updates only active dimensions from the latest evidence.
- `OracleTracker` is permitted only as a clearly labeled simulator upper-bound ablation and cannot be selected in production or primary policy comparisons.

All trackers return `TrackerEstimates` so policies remain unchanged.

## 10. Policy Design

### 10.1 Policy Protocol

```python
class TutorPolicy(Protocol):
    def act(
        self,
        observation: PolicyObservation,
        context: PolicyContext,
        rng: numpy.random.Generator,
    ) -> PolicyDecision: ...

    def metadata(self) -> PolicyMetadata: ...
```

### 10.2 Implementations

| Policy | State use | Update behavior | Unseen-state behavior |
|---|---|---|---|
| Static | Target concept only | None | Fixed directive |
| Expert heuristic | Observation fields and authored rules | None | Ordered safe rule fallback |
| BKT plus heuristic | BKT-derived observation and rules | None | Same heuristic fallback |
| Contextual bandit | Current discretized context | Offline fit or simulator training | Heuristic fallback or optimistic initialized values, fixed by manifest |
| Tabular Q-learning | Compact discretized state | Offline simulator episodes | Heuristic fallback, never arbitrary table default |

### 10.3 Discretization

`StateDiscretizerV1` maps:

- Target mastery: low, medium, high.
- Minimum prerequisite mastery: low, medium, high, or no-prerequisite sentinel.
- Active misconception: below/above threshold.
- Estimated bandwidth: low, medium, high.
- Estimated friction: low, medium, high.
- Repeated failures: `0`, `1`, `2+`.
- Hints used: `0`, `1`, `2+`.
- Turn index: early, middle, late.

Bin boundaries are artifact metadata. Values at boundaries use left-closed/right-open bins except the final closed bin. The discretizer is shared by bandit and Q-learning.

### 10.4 Q-Learning Update

Terminal updates omit the bootstrapped value:

```python
target = reward if terminated else reward + gamma * max(Q[next_state])
Q[state, action] += alpha * (target - Q[state, action])
```

Truncation bootstrapping behavior is explicit in the training manifest. Budget and maximum-turn truncations default to bootstrapping from the final observation; safety aborts do not.

## 11. Generation, Guardrail, and Sandbox Interfaces

### 11.1 Model Gateway

```python
class ModelGateway(Protocol):
    async def generate(
        self,
        request: GenerationRequest,
        *,
        idempotency_key: str,
        budget: CallBudget,
    ) -> GenerationResult: ...
```

The gateway enforces timeout, retry, concurrency, allowed model, token budget, response-size limit, structured-output validation, and usage capture. Provider adapters do not decide pedagogical actions.

### 11.2 Guardrail

```python
class Guardrail(Protocol):
    async def check(
        self,
        candidate: TutorCandidate,
        policy: LeakagePolicy,
    ) -> GuardrailResult: ...
```

`GuardrailResult` contains `accepted`, reason codes, confidence where calibrated, matched rule IDs, a rewrite brief, and evaluator metadata. A schema failure is a rejection. The graph owns retry counting and fallback selection.

### 11.3 Sandbox

```python
class SandboxGateway(Protocol):
    async def execute(
        self,
        request: ExecutionRequest,
        *,
        idempotency_key: str,
    ) -> ExecutionResult: ...
```

`ExecutionRequest` includes source code, test-bundle artifact reference, language/runtime version, CPU, memory, wall-time, process, filesystem, network, and output limits. Production adapters reject any request that enables outbound network or exceeds the service policy.

## 12. Trajectory Commit Design

### 12.1 Turn Envelope

`TurnEnvelope` groups:

- Identity and provenance.
- Public policy observation and decision.
- Realized tutor control.
- Observation evidence.
- Tracker summaries before and after the turn.
- Reward components.
- Termination/truncation state.
- Provider, sandbox, latency, token, and cost metadata.
- Optional privileged-state artifact reference.

### 12.2 Atomicity

The writer follows a prepare/publish protocol:

1. Validate public and privileged schemas independently.
2. Compute canonical content and checksum.
3. Write to a temporary content-addressed object key.
4. Verify size and checksum.
5. Atomically publish or write a completion marker.
6. Upsert the turn index in PostgreSQL using `(episode_id, turn_index)` as the idempotency constraint.
7. Return `CommitReceipt` containing object reference and checksum.
8. Allow LangGraph to checkpoint the receipt.

Duplicate content is idempotent. Conflicting content for an existing episode turn raises `TrajectoryConflictError` and stops the episode.

## 13. Batch Training Sequence

```d2
batch_training: "Batch Rollout, Training, and Promotion" {
  shape: sequence_diagram

  researcher: "Researcher CLI"
  api: "Experiment API"
  coordinator: "ExperimentCoordinator"
  queue: "Redis Streams"
  rollout: "RolloutWorker"
  object_store: "Object Storage"
  trainer: "PolicyTrainer"
  registry: "MLflow Policy Registry"
  evaluator: "EvaluationWorker"
  operator: "Authorized Operator"

  researcher -> api: "submit signed ExperimentManifest"
  api -> coordinator: "validate manifest, splits, budgets, and versions"
  coordinator -> queue: "publish deterministic shard commands"
  queue -> rollout: "claim shard through consumer group"
  rollout -> rollout: "construct local Gymnasium environments"
  rollout -> object_store: "publish validated trajectory partition"
  rollout -> queue: "acknowledge shard after artifact commit"
  coordinator -> trainer: "start training with approved partition refs"
  object_store -> trainer: "read policy-safe trajectory columns"
  trainer -> registry: "register candidate artifact and diagnostics"
  registry -> evaluator: "candidate policy and lineage"
  object_store -> evaluator: "held-out partitions and gate fixtures"
  evaluator -> registry: "attach feature audit, replay, safety, and quality gates"
  registry -> operator: "present promotion decision record"
  operator -> registry: "explicit promote, reject, or roll back"
  registry -> api: "publish immutable serving alias change event"
  api -> researcher: "final experiment and policy status"
}
```

The trainer cannot write the serving alias. The evaluator cannot retune the candidate. The operator cannot promote a candidate with missing mandatory gate records.

## 14. API and Event Design

### 14.1 Command Endpoints

| Method and path | Request | Response | Idempotency |
|---|---|---|---|
| `POST /v1/sessions` | `CreateSessionRequest` | `202 SessionAccepted` | Required header; reuses session ID |
| `GET /v1/sessions/{id}` | None | `SessionView` | Naturally idempotent |
| `POST /v1/sessions/{id}/resume` | `ResumeSessionRequest` | `202 ResumeAccepted` | Required per human turn |
| `POST /v1/sessions/{id}/cancel` | `CancelSessionRequest` | `202 CancelAccepted` | Repeated cancel returns current terminal state |
| `POST /v1/experiments/validate` | `ExperimentManifest` | `ManifestValidationResult` | Content hash |
| `POST /v1/experiments` | Validated manifest reference | `202 ExperimentAccepted` | Manifest hash plus caller scope |
| `GET /v1/experiments/{id}` | None | `ExperimentView` | Naturally idempotent |
| `POST /v1/experiments/{id}/cancel` | Reason | `202 CancelAccepted` | Repeated cancel is safe |
| `GET /v1/policies/{id}` | None | `PolicyView` | Naturally idempotent |
| `POST /v1/policies/{id}/promote` | `PromotionCommand` | `PromotionReceipt` | Candidate plus target alias |
| `POST /v1/policies/{id}/reject` | `RejectionCommand` | `PolicyView` | Candidate plus decision version |

All error responses use `ProblemDetails` with stable code, correlation ID, retryability, and safe detail. Validation errors never echo secrets, hidden answers, or privileged state.

### 14.2 WebSocket Events

Server events share an envelope:

```python
class GraphEvent(BaseModel):
    event_id: UUID
    session_id: SessionId
    checkpoint_id: str | None
    sequence: NonNegativeInt
    event_type: GraphEventType
    occurred_at: datetime
    payload: GraphEventPayload
```

Event types include `session_started`, `tutor_prompt_ready`, `turn_committed`, `execution_result`, `session_interrupted`, `session_completed`, `session_truncated`, and `session_failed`. Reconnecting clients provide the last acknowledged sequence; the API reconstructs durable state and may replay safe events.

Token-level model streaming is advisory and never considered a committed tutor prompt. Only a guardrail-approved `tutor_prompt_ready` event may be acted upon.

## 15. Future Human Interrupt Sequence

```d2
human_turn: "Future Human Turn with Durable Interrupt" {
  shape: sequence_diagram

  student: "Student Browser"
  websocket: "WebSocket Hub"
  api: "FastAPI Gateway"
  graph: "TutorGraphRuntime"
  checkpoint: "Postgres Checkpointer"
  sandbox: "SandboxGateway"
  tracker: "EpistemicTracker"
  writer: "TrajectoryWriter"

  graph -> graph: "generate and guardrail TutorControl"
  graph -> checkpoint: "persist pre-interrupt graph state"
  graph -> graph: "interrupt(HumanPrompt)"
  graph -> websocket: "publish tutor_prompt_ready with checkpoint cursor"
  websocket -> student: "deliver approved prompt"
  student -> websocket: "submit text/code and turn idempotency key"
  websocket -> api: "forward authenticated ResumeSessionRequest"
  api -> graph: "resume same thread with validated input"
  graph -> sandbox: "execute code when present"
  sandbox -> graph: "normalized ExecutionResult"
  graph -> tracker: "update from observable evidence"
  tracker -> graph: "TrackerEstimates"
  graph -> writer: "append human turn without simulator truth"
  writer -> graph: "CommitReceipt"
  graph -> checkpoint: "persist committed turn state"
  graph -> websocket: "publish turn_committed or next prompt"
  websocket -> student: "update session view"
}
```

The interrupt node performs no non-idempotent operation before calling `interrupt()`. On resume, the node restarts and obtains the persisted resume value through LangGraph's command mechanism.

## 16. Error Model

### 16.1 Error Categories

| Category | Examples | Retry policy |
|---|---|---|
| Validation | Invalid schema, unknown directive, split overlap | Never retry without changed input |
| Authorization | Privileged artifact access, human mode disabled | Never retry automatically |
| Budget | Token, cost, time, turn limit | Truncate with typed reason |
| Transient provider | Timeout, 429, temporary 5xx | Bounded exponential backoff with jitter |
| Permanent provider | Unsupported model, invalid credentials | Fail dependency or use preregistered fallback |
| Guardrail | Rejected prompt, malformed classifier output | Bounded rewrite then safe fallback |
| Sandbox | Timeout, memory limit, policy rejection | Return structured evidence; no host fallback |
| Tracker numerical | Non-finite likelihood, invalid posterior | Fail run or explicit degraded mode per manifest |
| Artifact conflict | Checksum or turn-content mismatch | Stop; requires investigation |
| Infrastructure | Database or object-store outage | Stop acceptance or retry from durable boundary |

### 16.2 Retry Identity

Idempotency keys use:

```text
{scope_id}:{episode_id}:{turn_index}:{node}:{operation}:{logical_attempt}
```

Provider transport retries reuse the same logical attempt key. A pedagogically requested rewrite increments `logical_attempt` because it is a new candidate, not a transport retry.

## 17. Configuration Design

Settings are split into:

- **Deployment settings:** database, queue, object store, telemetry, provider endpoints, secrets references.
- **Experiment manifest:** policies, prompts, models, profiles, tasks, splits, seeds, CBFM/tracker parameters, reward, budgets, repetitions, and gates.
- **Immutable artifacts:** calibration, action space, discretizer, policy, test bundles, and data splits.

Precedence is `built-in safe defaults < deployment configuration < signed manifest`, except deployment security policy may only tighten manifest limits. Environment variables cannot silently alter scientific parameters after manifest signing.

## 18. Test Design

### 18.1 Contract Tests

- JSON Schema snapshots for all public version-1 contracts.
- `extra="forbid"`, finite-number, unit-interval, and enum validation.
- Forward/backward compatibility checks for stored artifacts.
- Policy feature audit proving true-state fields are impossible to deserialize.

### 18.2 Mathematical Tests

- CBFM boundedness, monotonicity, low-load recovery, clipping-rate, ablation, and seeded-noise tests.
- Prompt-feature parser failure and normalizer-version tests.
- SMC log-weight stability, active-mask neutrality, ESS threshold, systematic resampling, posterior contraction, and parameter-misspecification tests.
- Reward decomposition and terminal/truncation bootstrap tests.

### 18.3 Graph Tests

- Golden synthetic turn ordering.
- Node field-ownership violations.
- Guardrail accept, bounded rewrite, malformed output, and safe fallback paths.
- Checkpoint recovery after every node boundary.
- Duplicate external-call and trajectory-commit idempotency.
- Terminal and all truncation routes.
- Human interrupt/resume with node restart.

### 18.4 Environment and Policy Tests

- Gymnasium `check_env` in CI.
- Repeatable reset and full template trajectory by seed.
- Policy parity over the same observation/action spaces.
- Discretizer boundary fixtures and Q-table unseen-state fallback.
- Q-table coverage accounting and action-propensity logging.
- Import-boundary test preventing simulator-private dependencies.

### 18.5 Integration Tests

- API to graph to checkpoint to trajectory commit.
- Rollout shard claim, artifact publish, acknowledgement, and reclaim.
- Candidate training, gate attachment, explicit promotion, and rollback.
- Model-provider timeout/rate-limit fallback with cost accounting.
- MicroVM timeout, memory, output, network, and normalization behavior.
- WebSocket disconnect and cursor-based recovery.

## 19. Requirement Traceability

| LLD area | Primary PRD requirements |
|---|---|
| Package boundaries and contracts | FR-SIM-002, FR-SIM-006, FR-POL-002, FR-TRJ-002, NFR-MNT-001 |
| Graph state and nodes | FR-GRF-001 through FR-GRF-007 |
| Gymnasium environment | FR-SIM-001 through FR-SIM-008 |
| CBFM | FR-CBFM-001 through FR-CBFM-009 |
| SMC and tracker baselines | FR-TRK-001 through FR-TRK-009 |
| Policies and Q-learning | FR-POL-001 through FR-POL-009 |
| Generation and guardrail | FR-GEN-001 through FR-GRD-006 |
| Student and sandbox | FR-STU-001 through FR-STU-007, SEC-SBX-001 through SEC-SBX-005 |
| Trajectories and training | FR-TRJ-001 through FR-RL-005 |
| API, events, and human interrupt | FR-UI-001, FR-UI-005, FR-GRF-006, NFR-REL-001 |
| Tests and reproducibility | FR-EXP-001 through FR-EXP-006, NFR-REP-001 through NFR-REP-005 |

## 20. LLD Acceptance Criteria

The implementation design is complete when:

- Public and simulator-private types are separated by enforceable package boundaries.
- Every graph-state field has one declared writer.
- Every node has typed inputs, writes, side effects, retry behavior, and routing.
- The Gymnasium reset/step transaction preserves the proposal's fast-evidence-slow timing.
- CBFM and SMC algorithms define numerical, calibration, and failure behavior.
- All policies implement one observation/decision protocol and a documented unseen-state fallback.
- External model, guardrail, sandbox, artifact, and registry interfaces are provider-neutral.
- Trajectory commit and all external side effects have deterministic idempotency behavior.
- Training and promotion cannot mutate serving state without completed gates and operator action.
- Human interrupts are resumable and contain no simulator truth.
- Contract, mathematical, graph, policy, integration, recovery, and security tests are specified.
- Every D2 class, dependency, and sequence diagram compiles successfully.
