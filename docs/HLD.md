# High-Level Design

## Master's POC: Socratic Tutoring and Evidence-Validity Evaluation

| Field | Value |
|---|---|
| Status | Proposed benchmark-first amendment for review |
| Revision date | 2026-07-14 |
| Product requirements | docs/PRD.md |
| Engineering strategy | docs/engineering-plan.md |
| Deployment model | One FastAPI backend and one React client |
| Primary research path | Frozen benchmark plus pinned LLM-student stress test |
| Secondary research path | Local CPU Gymnasium simulation and interpretable policy training |

> **Sequential-amendment status:** The PRD, engineering plan, and this HLD now describe the benchmark-first POC. The LLD, data-flow and schema design, and implementation plan still describe the earlier POC architecture and must be revised one at a time. The dissertation proposal has not yet been amended; changing the primary study requires supervisor approval.

## 1. Purpose

This document defines the high-level architecture for a Master's proof of concept. It deliberately separates three concerns:

1. A working browser tutoring turn that proves the product path.
2. A benchmark path that tests whether context-isolated executable evidence improves prediction of a different held-out criterion.
3. A later Gymnasium path for CBFM, tracker, and policy experiments.

The critical product path remains:

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

The critical scientific path is now:

```text
Freeze reviewed benchmark cases and metrics
-> run dialogue-only and probe-informed conditions from identical case state
-> commit mastery predictions and policy actions
-> reveal a different criterion probe
-> score both conditions against the same execution-backed criterion label
-> report paired case-level effects and negative controls
```

The architecture is local-first and intentionally avoids distributed services. Scientific isolation, replay, provenance, and a reliable demonstration take priority over production scale.

## 2. Architecture Goals

- Deliver the browser vertical slice before advanced research components.
- Freeze measurement design before CBFM fitting, SMC comparison, or learned-policy training.
- Keep benchmark evaluation independent of LangGraph tutoring and Gymnasium simulation.
- Prevent criterion records and simulator truth from entering tracker or policy inputs.
- Compare conditions on cloned pre-update case state and enforce a commitment barrier before criterion reveal.
- Use execution-backed outcomes and reviewed rubrics rather than an LLM judge for the primary score.
- Validate the pipeline with deterministic fixtures, then test the claim with at least one pinned LLM-student model/provider condition.
- Use fresh, isolated model contexts for public interaction, evidence probe, and criterion probe.
- Treat the benchmark case as the primary statistical unit; repeated model samples are nested observations.
- Run deterministic simulations without network credentials.
- Permit tracker and policy implementations to swap through small protocols.
- Keep canonical records in local Parquet files and use W&B as supplementary tracking.
- Retain deterministic template mode for the interactive demonstration.
- Defer human-study infrastructure, neural RL, and production scaling.

## 3. Architecture Decisions

| ID | Decision | Reason | Consequence |
|---|---|---|---|
| ADR-POC-001 | Use one monorepo and one demo deployment. | One researcher owns a fixed dissertation schedule. | React and FastAPI deploy together; no microservices. |
| ADR-POC-002 | Build the deterministic vertical slice first. | End-to-end integration is the first delivery risk. | SMC, RL, LLM tutoring, and sandbox work cannot block the first demo gate. |
| ADR-POC-003 | Use LangGraph only for interactive tutoring. | It makes turn state visible and resumable but adds unnecessary rollout overhead. | Benchmark and training runners import shared domain packages, not the graph runtime. |
| ADR-POC-004 | Create a dedicated benchmark execution plane. | Evidence validity cannot be established by a simulator built from the same assumptions. | The primary benchmark is not a Gymnasium rollout and does not optimize a policy. |
| ADR-POC-005 | Separate evidence and criterion capabilities. | Reusing one probe for update and scoring makes improvement self-fulfilling. | Trackers and policies cannot import or deserialize criterion-owned records. |
| ADR-POC-006 | Commit every condition before criterion reveal. | Temporal isolation must be auditable, not merely documented. | The criterion gate opens only after a complete prediction/action commit set exists. |
| ADR-POC-007 | Use executable outcomes and reviewed rubrics as benchmark labels. | Model self-report and LLM judging are weak or circular targets. | Missing sandbox output is explicit missingness, not automatic failure. |
| ADR-POC-008 | Use deterministic fixtures only for pipeline validation. | Authored examples cannot demonstrate actual LLM sycophancy. | At least one pinned LLM-student stress test is required for the primary empirical claim. |
| ADR-POC-009 | Split benchmark data by task family. | Random dialogue-row splits leak surface templates and near-duplicate probes. | Development, calibration, selection, and held-out families have disjoint manifests. |
| ADR-POC-010 | Use Gymnasium directly for secondary simulation and training. | Pure reset/step loops are fast, transparent, and reproducible. | No LangGraph, web, or LLM dependency occurs during policy learning. |
| ADR-POC-011 | Use SQLite and JSONL for the demo. | They provide enough local durability without operational overhead. | PostgreSQL and Redis are deferred. |
| ADR-POC-012 | Use Parquet as canonical scientific storage. | It is typed, inspectable, efficient, and queryable with DuckDB. | W&B is never the sole source required to reproduce a result. |
| ADR-POC-013 | Use Hydra for resolved experiment configurations. | Benchmark and secondary experiment matrices need explicit composition. | Every run persists the resolved config and content hash. |
| ADR-POC-014 | Use OpenRouter through a typed local gateway. | A normalized API is sufficient without operating a separate proxy. | Model, provider route, prompts, and sampling settings are pinned and recorded. |
| ADR-POC-015 | Implement interpretable policies directly in NumPy. | State coverage and policy behavior must remain inspectable. | RLlib, Ray, PPO, DQN, and GPU training are not required. |
| ADR-POC-016 | Require an external sandbox for generated code. | LLM- or participant-generated Python is untrusted. | Only checked-in, manifest-hashed fixtures may use the local trusted executor. |
| ADR-POC-017 | Defer the human pilot. | Human research requires ethics, consent, privacy, and operational design. | The POC and primary benchmark use synthetic or generated student behavior only. |

## 4. System Context

```d2
direction: right

users: "Users" {
  researcher: "Researcher"
  evaluator: "Dissertation Evaluator"
  demo_user: "Demo Participant"
  reviewer: "Independent Benchmark Reviewer"
}

poc: "Socratic Tutoring POC" {
  web: "React Demo"
  api: "FastAPI Backend"
  interactive: "LangGraph Tutoring Flow"
  benchmark: "Benchmark Runner"
  experiments: "Gymnasium Experiment Runner"
  analysis: "Evaluation and Report Builder"
}

local_data: "Local Research Data" {
  benchmark_defs: "Frozen Benchmark Definitions"
  sqlite: "SQLite Checkpoints"
  jsonl: "JSONL Turn Events"
  parquet: "Parquet Research Records"
  artifacts: "Configs, Policies, and Reports"
}

external: "Bounded External Services" {
  openrouter: "OpenRouter"
  wandb: "Weights & Biases"
  sandbox: "External Code Sandbox"
}

future: "Deferred Future Pilot" {
  pilot: "Ethics-Approved Student Study"
}

users.researcher -> poc.web: "runs demo and inspects state"
users.researcher -> poc.benchmark: "validates and runs primary study"
users.researcher -> poc.experiments: "runs secondary studies"
users.evaluator -> poc.web: "reviews tutoring behavior"
users.evaluator -> poc.analysis: "reviews reproducible results"
users.demo_user -> poc.web: "completes non-research demo task"
users.reviewer -> local_data.benchmark_defs: "reviews cases, labels, and splits"

poc.web -> poc.api: "HTTP JSON"
poc.api -> poc.interactive: "start session and submit turn"
poc.interactive -> local_data.sqlite: "checkpoint state"
poc.interactive -> local_data.jsonl: "append committed event"

local_data.benchmark_defs -> poc.benchmark: "frozen case manifest"
poc.benchmark -> local_data.parquet: "condition, criterion, and metric records"
poc.benchmark -> external.openrouter: "pinned student generation"
poc.benchmark -> external.sandbox: "execute generated probe code"

poc.experiments -> local_data.parquet: "secondary trajectories"
poc.experiments -> local_data.artifacts: "resolved configs and policies"
local_data.parquet -> poc.analysis: "canonical inputs"
local_data.artifacts -> poc.analysis: "provenance"
poc.analysis -> external.wandb: "optional metrics and references"

poc.interactive -> external.openrouter: "optional tutor rendering"
poc.interactive -> external.sandbox: "optional student code execution"

future.pilot -> poc.web: "future frozen study build" {
  style.stroke-dash: 4
}
```

### 4.1 Context Rules

- The demo runs in deterministic template mode without external services.
- Demo interactions are not human research data and are excluded from the primary benchmark by default.
- Benchmark definitions are checked-in, versioned, reviewed artifacts; generated outputs are immutable run artifacts.
- OpenRouter is required only for the pinned LLM-student stress test and optional tutor rendering.
- Public, evidence-probe, and criterion-probe generations use separate fresh contexts.
- The sandbox receives code, tests, limits, and an operation ID, never unrelated conversation, tracker state, or simulator truth.
- W&B receives synthetic metadata and approved artifact references only.
- Future pilot functionality remains outside the current deployment and data lifecycle.

## 5. Logical Component Architecture

```d2
direction: right

client: "React/Vite Client" {
  task_view: "Task and Conversation"
  response_input: "Student Response Input"
  glass_box: "Evidence and State Inspector"
}

backend: "Interactive POC" {
  routes: "FastAPI Session and Turn Routes"
  session_service: "Tutor Session Service"
  graph: "LangGraph Turn Graph"
  event_writer: "JSONL Event Writer"
}

domain: "Policy-Safe Shared Domain" {
  contracts: "Pydantic Contracts"
  evidence: "Evidence Extractors"
  trackers: "Simple / BKT / Reduced SMC"
  policies: "Heuristic / Bandit / Q-Learning"
  templates: "Socratic Template Renderer"
  guardrail: "Rule-First Guardrail"
  cbfm: "Prompt Load and CBFM"
}

benchmark: "Benchmark Plane" {
  manifest_loader: "Frozen Manifest Loader"
  generator: "Fixture / Pinned LLM Generator"
  condition_runner: "Paired Condition Runner"
  commitment_store: "Commitment Barrier"
  criterion_gate: "Evaluator-Only Criterion Gate"
  scorer: "Execution-Backed Scorer"
  analyzer: "Paired Case Analyzer"
  replay: "Recorded-Response Replay"
}

simulation: "Secondary Simulation Plane" {
  env: "SocraticTutor/POMDP-v0"
  runner: "Hydra Experiment Runner"
  trajectory_writer: "Simulation Parquet Writer"
}

storage: "Local Storage" {
  benchmark_defs: "Cases, Probes, Tests, Reviews, Splits"
  generated: "Generated Response Records"
  commitments: "Committed Condition Predictions"
  criterion: "Evaluator-Only Criterion Records"
  parquet: "Partitioned Research Parquet"
  sqlite: "SQLite Checkpointer"
  jsonl: "Append-Only Demo JSONL"
  artifacts: "Content-Addressed Artifacts"
}

adapters: "External Adapters" {
  model_gateway: "OpenRouter ModelGateway"
  sandbox_gateway: "External SandboxGateway"
  wandb_sink: "W&B Run Sink"
}

analysis: "Local Analysis" {
  duckdb: "DuckDB"
  statistics: "Paired Statistics"
  reports: "Figures and Gate Decisions"
}

client.task_view -> backend.routes: "create or load session"
client.response_input -> backend.routes: "submit turn"
backend.routes -> backend.session_service: "validated command"
backend.session_service -> backend.graph: "invoke with thread ID"
backend.graph -> domain.evidence: "classify response"
backend.graph -> domain.trackers: "update belief"
backend.graph -> domain.policies: "select directive"
backend.graph -> domain.templates: "render fallback"
backend.graph -> domain.guardrail: "check prompt"
backend.graph -> backend.event_writer: "committed turn"
backend.graph -> storage.sqlite: "checkpoint"
backend.event_writer -> storage.jsonl: "append"
backend.routes -> client.glass_box: "policy-safe response"

storage.benchmark_defs -> benchmark.manifest_loader: "validated frozen inputs"
benchmark.manifest_loader -> benchmark.condition_runner: "case without criterion capability"
benchmark.generator -> benchmark.condition_runner: "public and allowed evidence records"
benchmark.condition_runner -> domain.trackers: "condition-specific evidence"
benchmark.condition_runner -> domain.policies: "policy-safe observation"
benchmark.condition_runner -> benchmark.commitment_store: "predictions and actions"
benchmark.commitment_store -> storage.commitments: "atomic complete set"
storage.commitments -> benchmark.criterion_gate: "commit hash proves completeness"
storage.benchmark_defs -> benchmark.criterion_gate: "criterion capability after barrier"
benchmark.criterion_gate -> adapters.sandbox_gateway: "execute generated criterion code"
benchmark.criterion_gate -> storage.criterion: "sealed criterion record"
storage.commitments -> benchmark.scorer: "committed predictions"
storage.criterion -> benchmark.scorer: "common held-out label"
benchmark.scorer -> benchmark.analyzer: "case scores and exclusions"
benchmark.analyzer -> storage.parquet: "paired metrics"
benchmark.generator -> adapters.model_gateway: "fresh-context generation"
adapters.model_gateway -> storage.generated: "recorded public and probe outputs"
storage.generated -> benchmark.replay: "network-free replay"
benchmark.replay -> benchmark.condition_runner: "reconstructed allowed inputs"

simulation.runner -> simulation.env: "construct and step"
simulation.env -> domain.cbfm: "private fast-state dynamics"
simulation.env -> domain.trackers: "subjective update"
simulation.env -> domain.policies: "policy-safe observation"
simulation.runner -> simulation.trajectory_writer: "turn records"
simulation.trajectory_writer -> storage.parquet: "secondary trajectories"
simulation.runner -> storage.artifacts: "resolved config and policy"

storage.parquet -> analysis.duckdb: "query canonical records"
analysis.duckdb -> analysis.statistics: "case and seed samples"
analysis.statistics -> analysis.reports: "estimates and intervals"
analysis.reports -> adapters.wandb_sink: "optional summaries"
```

### 5.1 Component Boundaries

#### React Client

Provides one focused tutoring workflow and a read-only research replay view. It contains no tracker, scoring, criterion-gating, or policy logic.

#### FastAPI Backend

Validates browser payloads, resolves sessions, invokes LangGraph, and returns typed projections. It is the only browser-facing process and is not required for benchmark or training runs.

#### LangGraph Turn Graph

Coordinates one interactive turn and checkpoints policy-safe session state. It delegates all mathematical updates and cannot load benchmark criterion records or synthetic simulator truth.

#### Policy-Safe Shared Domain

Contains small, testable contracts and algorithms shared by the interactive, benchmark, and simulation paths. Criterion-owned and simulator-private models live outside this package.

#### Benchmark Plane

Evaluates evidence quality rather than training a policy. It clones condition state, supplies only condition-permitted evidence, records tracker predictions and policy actions, enforces the all-condition commitment barrier, then reveals and scores the common held-out criterion.

#### Criterion Gate and Scorer

Own the only capability that can deserialize criterion probes, execution results, and labels. The gate fails closed unless the expected condition set and commit hash are complete. The scorer cannot call trackers or policies.

#### Gymnasium Environment

Owns private synthetic state and secondary mathematical transitions. It cannot import benchmark criterion labels, and its latent truth is never the primary benchmark label.

#### Local Analysis

Builds final tables from immutable local records. The primary analysis is a paired case-level comparison. Repeated LLM samples are aggregated within case/model or handled by a declared clustered analysis.

## 6. Three Execution Paths

### 6.1 Interactive Tutoring Path

```text
React request
-> FastAPI validation
-> LangGraph invocation
-> deterministic evidence
-> simple tracker update
-> heuristic or frozen policy
-> template or optional OpenRouter tutor rendering
-> rule guardrail
-> JSONL append
-> SQLite checkpoint
-> React response
```

This path proves that the tutoring product works end to end. It does not establish evidence validity or train an RL policy.

### 6.2 Benchmark Measurement Path

```text
Frozen benchmark manifest
-> load public interaction without criterion capability
-> clone identical pre-update state for every condition
-> supply dialogue-only, probe-informed, unrelated, or corrupted evidence
-> tracker prediction and policy action per condition
-> atomically commit the complete condition set
-> open evaluator-only criterion capability
-> run structurally different criterion probe
-> score all conditions against the same demonstrated-performance label
-> paired case-level analysis
```

The benchmark runner does not depend on FastAPI, React, LangGraph, or Gymnasium. The deterministic fixture mode is network-free. The primary stress test adds a pinned OpenRouter model/provider and external sandbox while preserving the same frozen cases, prompts, metrics, exclusions, and commitment protocol.

### 6.3 Secondary Simulation and Training Path

```text
Hydra resolved config
-> Gymnasium reset(seed)
-> policy-safe observation
-> policy action
-> mathematical environment step
-> optional policy update
-> Parquet trajectory
-> local policy artifact
-> optional W&B summary
```

This path has no dependency on React, FastAPI, LangGraph, OpenRouter, benchmark criterion records, or W&B availability.

### 6.4 Reporting Path

```text
Frozen benchmark and experiment manifests
-> validated Parquet partitions
-> completeness and leakage checks
-> DuckDB queries
-> paired or matched-seed intervals
-> figures, tables, limitations, and gate decisions
-> optional W&B summary
```

Final reports remain reconstructable from local files and may report positive, negative, or inconclusive results.

## 7. State Ownership and Visibility

| State or artifact | Owner | Decision-time visibility | Evaluation visibility |
|---|---|---|---|
| Benchmark case metadata and public interaction | Benchmark manifest | All benchmark conditions | Evaluator |
| Context-isolated evidence probe/result | Evidence capability | Probe-informed condition only; transformed control variants for control conditions | Evaluator after run |
| Condition pre-update state | Condition runner | Isolated clone for each condition | Pairing audit |
| Mastery prediction and policy action | Active tracker/policy | Current condition only until commit | Scorer after complete-set commit |
| Commit set and hash | Commitment barrier | Write only from condition runner | Criterion gate and audit |
| Criterion probe, response, execution, and label | Criterion capability | Never available before commitment | Criterion gate, scorer, evaluator |
| Recorded provider response | Model gateway | Channel-specific consumer only | Replay and provenance |
| Synthetic true mastery/misconceptions | Gymnasium environment | Never visible to benchmark or policy | Separate secondary diagnostic only |
| True B_t and F_t | Gymnasium environment | Never visible to policy | Controlled synthetic analysis |
| Tracker internals or particles | Active tracker | Tracker only | Diagnostics |
| Tracker summary | Active tracker | Active policy and permitted renderer | Benchmark/simulation analysis |
| Policy observation | Observation projector | Active policy | Public trajectory |
| Graph session state | LangGraph | API projection only | Demo replay |
| Demo event | Event writer | Researcher/evaluator | Excluded from training by default |
| Benchmark and simulation records | Dedicated Parquet writers | Schema-specific readers only | Approved analysis commands |

Enforcement:

- Policy-safe contracts contain no criterion or simulator-private fields.
- Criterion models are evaluator-owned and loaded only through the criterion capability.
- The condition runner receives a case view that omits criterion paths and labels.
- The commit barrier validates expected condition IDs, case ID, benchmark version, timestamps, and content hashes.
- Simulator truth resides in a private package and cannot be imported by policies.
- Pydantic contracts reject unknown fields.
- Analysis readers enumerate allowed Parquet columns.
- Glass Box truth routes exist only for explicitly labelled synthetic views.

## 8. Vertical-Slice Architecture

### 8.1 Initial Domain Fixture

The first task concerns Python mutable-list aliasing:

```python
items = [1, 2]
alias = items
alias.append(3)
```

The fixture contains task text, target concept, prerequisite, misconception, expected evidence, heuristic rules, and Socratic templates. It is versioned in the repository.

### 8.2 Initial Components

- Evidence extractor: deterministic rubric rules producing correct, misconception, uncertain, or empty evidence.
- Tracker: bounded scalar mastery and misconception-probability update with hand-calculated fixtures.
- Policy: ordered heuristic decision table.
- Renderer: authored templates for every legal directive.
- Guardrail: rule checks against complete answers and solution fragments.
- Persistence: one append-only JSONL turn event; in-memory checkpointing is acceptable before SQLite integration.

### 8.3 Vertical-Slice Gate

The slice is complete when a browser test proves:

1. Session creation returns the authored prompt.
2. Student submission reaches the correct session and turn.
3. Evidence output matches the fixture.
4. Tracker before/after values match hand calculations.
5. Heuristic action matches the rule table.
6. The next prompt passes leakage checks.
7. A complete event is appended before success returns.
8. The workflow runs without network credentials.

Passing this gate proves integration, not tutoring efficacy or measurement validity.

## 9. Benchmark Architecture

### 9.1 Commit-Then-Reveal Protocol

```d2
shape: sequence_diagram

researcher: "Researcher"
runner: "Benchmark Runner"
manifest: "Frozen Manifest"
public_gen: "Public Context"
evidence_gen: "Evidence Context"
conditions: "Condition Runner"
tracker: "Tracker"
policy: "Policy"
commit: "Commitment Barrier"
criterion_gen: "Fresh Criterion Context"
sandbox: "External Sandbox"
scorer: "Evaluator Scorer"
analysis: "Paired Analyzer"

researcher -> runner: "start versioned run"
runner -> manifest: "validate hashes, reviews, and split"
manifest -> runner: "case view without criterion fields"
runner -> public_gen: "generate or replay public interaction"
public_gen -> runner: "public response record"
runner -> evidence_gen: "generate or replay isolated evidence probe"
evidence_gen -> runner: "evidence execution record"
runner -> conditions: "clone identical pre-update state into four conditions"
conditions -> tracker: "dialogue-only / valid / unrelated / corrupted evidence"
tracker -> conditions: "mastery probability and uncertainty"
conditions -> policy: "policy-safe tracker summary"
policy -> conditions: "pedagogical action"
conditions -> commit: "commit all predictions and actions"
commit -> commit: "verify complete set and seal hash"
commit -> runner: "criterion capability token"
runner -> criterion_gen: "reveal different held-out criterion probe"
criterion_gen -> sandbox: "execute generated criterion response"
sandbox -> criterion_gen: "normalized execution result"
criterion_gen -> scorer: "criterion record and demonstrated-performance label"
commit -> scorer: "sealed condition records"
scorer -> analysis: "same criterion score for every condition"
analysis -> researcher: "paired Brier effect, controls, missingness, intervals"
```

The criterion context never receives public dialogue or evidence-probe output. The condition runner never receives the criterion capability token. An architecture test must prove that moving criterion access before the commit step fails.

### 9.2 Benchmark Contents

Version 1 contains at least 24 reviewed cases:

- Four Python concepts.
- Two misconception states per concept.
- Three transfer cases per concept/misconception pair.
- Coverage of supported mastery, false public mastery, honest non-mastery, underconfidence, and ambiguous evidence.
- A public interaction, an evidence probe, a structurally different criterion probe, executable tests or deterministic rubrics, label rationale, reviewer record, and split assignment for every case.

Evidence and criterion probes may test the same concept but cannot share the exact solution, surface template, answer-bearing wording, or execution trace.

### 9.3 Conditions

Every case uses the same initial state and common held-out criterion:

| Condition | Decision-time evidence |
|---|---|
| Dialogue only | Public interaction only |
| Probe informed | Public interaction plus valid isolated evidence-probe result |
| Unrelated probe control | Public interaction plus an unrelated probe result |
| Corrupted probe control | Public interaction plus corrupted or within-design shuffled evidence |

The primary contrast is dialogue only versus probe informed. Controls test whether any extra structured record creates an apparent improvement.

### 9.4 Primary and Secondary Outputs

For case i with criterion label y_i and predicted mastery probabilities p_i,D and p_i,P:

```text
Delta Brier_i
= Brier(p_i,D, y_i) - Brier(p_i,P, y_i)
```

Positive values favor the probe-informed condition. The report includes:

- Mean or prespecified robust summary of paired case-level Brier differences with uncertainty.
- False-mastery acceptance and detection precision/recall.
- Unsafe advancement.
- Tracker confidence and policy-action disagreement.
- Negative-control effects.
- Invalid execution, missing criterion, incomplete pair, and exclusion counts.
- Exploratory policy rank reversal, clearly separated from confirmatory endpoints.

Repeated generations are nested within case/model and never counted as independent benchmark cases.

### 9.5 Deterministic and LLM Layers

Deterministic authored fixtures prove schema, isolation, execution, scoring, control, and expected-direction behavior. They do not prove that an LLM student is sycophantic.

The primary stress test uses at least one pinned model/provider route. Public, evidence, and criterion calls use separate request templates and fresh contexts. Responses, final code, usage, provider route, timing, and execution records are persisted. Hidden reasoning is neither requested nor stored. Recorded-response replay must reproduce the analysis without network access.

A second model family is optional replication and is dropped before compromising the required first condition.

### 9.6 Freeze and Split Rules

- Case schema, prompts, rubrics, executable tests, labels, exclusions, metrics, and split manifest freeze before held-out generation.
- Splits separate task families rather than random response rows.
- Independent review signs off label rationale, misconception mapping, and probe separation.
- Semantic corrections create a new benchmark version and rerun every affected condition.
- Formatting changes update hashes and the change log.
- Benchmark cases are not tuned from W&B sweeps or held-out outcomes.

### 9.7 Benchmark Failure Semantics

- Missing one required condition blocks the paired case from primary aggregation and is reported.
- Sandbox unavailability yields missing execution, not incorrect mastery.
- Criterion access before complete commitment invalidates the run.
- Provider failure may be retried under the frozen policy; unresolved failure remains explicit missingness.
- A negative or inconclusive effect is a valid result.
- A leakage, pairing, or scoring-integrity failure is not a valid result.

## 10. Secondary Simulator and RL Architecture

```d2
direction: right

configuration: "Secondary Experiment Definition" {
  hydra: "Hydra Config Groups"
  resolved: "Resolved Config and Hash"
  split: "Frozen Non-Benchmark Split"
}

execution: "Local CPU Execution" {
  matrix: "Matched Condition and Seed Matrix"
  envs: "Gymnasium Environments"
  policy: "Heuristic / Bandit / Q-Learning"
}

outputs: "Canonical Outputs" {
  trajectory: "Simulation Parquet"
  policy_artifact: "Policy Table Artifact"
  run_manifest: "Run Manifest"
}

analysis: "Secondary Evaluation" {
  duckdb: "DuckDB Queries"
  statistics: "Matched-Seed Statistics"
  report: "Figures and Retain/Reject Decisions"
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
analysis.report -> tracking.wandb: "summary only"
```

### 10.1 Configuration and Seeds

Hydra composes benchmark, generation, environment, tracker, policy, CBFM, task/profile, split, seed, and evaluation settings. Each command saves and hashes the fully resolved configuration before execution.

A root seed derives separate environment, transition-noise, observation-noise, policy-exploration, and generation seeds. Matched secondary conditions reuse environment/task/profile seeds. Provider nondeterminism is handled through recorded outputs and repeated samples, not by claiming perfect seed determinism.

### 10.2 Policy Order

Implement and compare:

1. Static baseline.
2. Expert heuristic.
3. BKT plus heuristic.
4. Contextual bandit.
5. Tabular Q-learning.

Neural RL is considered only if compact-state coverage is empirically inadequate and all required benchmark/demo work is complete.

### 10.3 W&B Boundary

W&B receives resolved configuration, hashes, metrics, selected tables, and approved artifact references. It is not queried to reconstruct canonical records during final analysis and does not define benchmark splits, labels, or exclusions.

### 10.4 Scale Strategy

Run sequentially first. If profiling shows a bottleneck, use Gymnasium SyncVectorEnv or local multiprocessing. CPU execution is sufficient for tabular policies; GPUs and distributed clusters remain out of scope.

## 11. LLM Architecture

### 11.1 Model Gateway

ModelGateway accepts a typed request and returns final content, provider/model metadata, usage, latency, finish reason, request identifier, generation identifier, and prompt/config hashes.

OpenRouter is the initial provider path. Every experimental request pins:

- Model ID and provider route where supported.
- Prompt template and version.
- Sampling parameters and output schema.
- Retry and timeout policy.
- Per-run token and cost budget.

### 11.2 LLM-Student Benchmark Generation

The benchmark uses three isolated channels:

1. Public interaction context.
2. Evidence-probe context with only the evidence task and allowed setup.
3. Criterion context with only the held-out criterion task and allowed setup.

The criterion request excludes public dialogue, evidence-probe content/output, tracker state, and policy action. The generated final answer or code is stored; hidden chain-of-thought is not requested or retained.

### 11.3 Tutor Rendering

Tutor rendering is a later presentation comparison. A committed policy action may be rendered by an authored template or a pinned model configuration. It cannot alter the already committed tracker update or policy action.

### 11.4 Failure and Replay Policy

- Timeout/provider failure: retry under the frozen policy, then record missingness.
- Interactive tutor failure: use deterministic template.
- Invalid structured tutor output: reject and use template.
- Tutor guardrail rejection: one bounded rewrite, then template.
- Benchmark generation failure: never substitute an authored response into an LLM condition without changing the condition label.
- Replay: consume recorded final responses and execution records; do not call the provider again.

## 12. Code Execution Boundary

Arbitrary code execution is not required for the first vertical slice but is required for generated executable benchmark probes.

```text
Benchmark or interactive caller
-> typed SandboxGateway request
-> external isolated sandbox
-> normalized execution record
-> evidence capability or evaluator-only criterion capability
```

The request contains source, test bundle/hash, runtime, operation ID, and resource limits. It excludes unrelated conversation, policy output, criterion data not required for that execution, and simulator truth.

Security rules:

- LLM-generated and participant-supplied code never executes on the FastAPI or researcher host.
- Only checked-in fixtures whose hashes match the frozen manifest may use the local trusted executor.
- The external sandbox has no outbound network and enforces CPU, memory, wall-time, output, and filesystem limits.
- A sandbox outage produces a typed unavailable result and explicit missingness.

## 13. Local Data Architecture

### 13.1 SQLite and JSONL

SQLite stores LangGraph checkpoints and minimal demo session metadata. JSONL stores one complete event per committed interactive turn. Neither is canonical benchmark or training storage.

### 13.2 Frozen Benchmark Definitions

Checked-in benchmark inputs are immutable within a version:

```text
development/data/benchmark/v1/
|-- manifest
|-- cases
|-- public
|-- evidence
|-- criterion
|-- tests
|-- reviews
|-- splits
|-- limitations
```

The manifest hashes case files, prompts, tests, reviews, split membership, schemas, and metric version.

### 13.3 Generated Benchmark Artifacts

Generated run data remains outside checked-in definitions:

```text
artifacts/benchmark/version=<benchmark_version>/run=<run_id>/
|-- generated_responses.parquet
|-- evidence_executions.parquet
|-- condition_predictions.parquet
|-- commitment_manifest.json
|-- criterion_records.parquet
|-- paired_case_metrics.parquet
|-- report
```

condition_predictions and the commitment manifest are durably written before criterion records can be loaded. Timestamps and hashes make the order auditable. Criterion records are not valid tracker or RL training inputs.

### 13.4 Simulation Parquet

Secondary trajectories are partitioned separately by experiment, split, condition, and seed. Policy-safe observations and privileged simulator diagnostics use separate schemas and directories.

### 13.5 Content-Addressed Artifacts

Artifacts include resolved Hydra configurations, manifests, calibration parameters, BKT/SMC settings, policy tables, prompt/template versions, metric summaries, figures, and reports.

## 14. Deployment Architecture

```d2
direction: down

host: "Researcher Laptop / Demo Host" {
  workspace: "Pixi and pnpm Workspace"

  app: "Single Demo Deployment" {
    fastapi: "FastAPI Process"
    react: "Built React Assets"
    graph: "LangGraph Runtime"
  }

  benchmark: "Benchmark Commands" {
    validate: "benchmark-validate"
    smoke: "benchmark-smoke"
    generate: "benchmark-generate"
    evaluate: "benchmark-evaluate"
  }

  secondary: "Secondary Commands" {
    simulate: "Gymnasium Simulation and Training"
    analyze: "DuckDB Analysis"
  }

  files: "Persistent Local Directory" {
    defs: "Frozen Benchmark Definitions"
    demo: "SQLite and JSONL"
    parquet: "Benchmark and Simulation Parquet"
    artifacts: "Configs, Policies, and Reports"
  }
}

external: "Bounded Network Dependencies" {
  openrouter: "OpenRouter API"
  sandbox: "External Sandbox API"
  wandb: "W&B API"
}

browser: "Browser"

browser -> host.app.fastapi: "HTTP"
host.app.fastapi -> host.app.react: "serve static assets"
host.app.fastapi -> host.app.graph: "invoke tutoring turn"
host.app.graph -> host.files.demo: "checkpoint and append"

host.files.defs -> host.benchmark.validate: "schemas, hashes, splits, reviews"
host.benchmark.validate -> host.benchmark.smoke: "validated deterministic inputs"
host.benchmark.smoke -> host.files.parquet: "pipeline validation records"
host.files.defs -> host.benchmark.generate: "frozen manifest"
host.benchmark.generate -> external.openrouter: "isolated pinned generation"
host.benchmark.generate -> external.sandbox: "execute generated code"
host.benchmark.generate -> host.files.parquet: "responses, commits, and criterion records"
host.files.parquet -> host.benchmark.evaluate: "paired benchmark inputs"
host.benchmark.evaluate -> host.files.artifacts: "primary report"

host.secondary.simulate -> host.files.parquet: "secondary trajectories"
host.secondary.simulate -> host.files.artifacts: "policies and configs"
host.files.parquet -> host.secondary.analyze: "validated secondary inputs"

host.app.graph -> external.openrouter: "optional tutor rendering"
host.app.graph -> external.sandbox: "optional code execution"
host.benchmark.evaluate -> external.wandb: "optional summary"
host.secondary.analyze -> external.wandb: "optional summary"

external.openrouter -> host.benchmark.generate: "failure is recorded, never silently replaced" {
  style.stroke-dash: 4
}
external.wandb -> host.benchmark.evaluate: "unavailable does not block local report" {
  style.stroke-dash: 4
}
```

### 14.1 Local Development

- Vite proxies API calls to FastAPI.
- Tests may use an in-memory LangGraph checkpointer.
- SQLite, generated records, and artifacts live under gitignored workspace paths.
- W&B defaults to offline or disabled.
- Benchmark validation and deterministic smoke tests require no network.
- Recorded LLM responses support network-free regression tests.

### 14.2 Demonstration Build

- React assets and FastAPI run as one application.
- Template rendering is the emergency fallback for tutoring turns.
- A benchmark replay view may show predictions/actions first and reveal criterion results only after displaying their committed status.
- The demonstration must not imply that benchmark performance is a validated measure of human mental fatigue or educational efficacy.

### 14.3 Primary LLM Run

The required pinned benchmark run may execute from the researcher host but sends generated code only to the external sandbox. Provider and sandbox credentials remain local secrets. Artifacts are downloaded to canonical local storage before analysis.

### 14.4 Future Pilot

No current pilot deployment is specified. A real-student build requires approved ethics protocol, consent, identity and pseudonymization, retention/deletion controls, incident procedures, accessibility testing, study registration, and a frozen intervention.

## 15. Reliability and Failure Handling

| Failure | Required behavior | Integrity rule |
|---|---|---|
| Invalid student request | Return typed 4xx | Do not append or advance |
| Duplicate interactive turn | Return prior result or conflict | Never commit two logical events for one turn ID |
| Evidence extraction failure | Emit uncertain evidence or typed failure | Never infer mastery from missing output silently |
| Tracker numerical failure | Use last valid/simple state and mark degraded | Never recover from simulator truth or criterion |
| Unknown policy state | Use documented heuristic fallback | Log fallback and coverage miss |
| Criterion requested before commit | Fail closed and invalidate attempted run | No tracker or policy can observe criterion data |
| Incomplete benchmark condition set | Exclude from primary pair and report count | Never impute a complete pair silently |
| Benchmark provider failure | Retry under frozen policy, then mark missing | Never replace with a different condition |
| Sandbox failure | Emit unavailable/timeout execution | Missing is not incorrect |
| Corrupt benchmark hash or split leak | Stop run | Never evaluate an unfrozen or leaked manifest |
| OpenRouter tutor failure | Render deterministic template | Preserve accepted interactive turn |
| Guardrail rejection | One rewrite then template | Never expose rejected tutor candidate |
| JSONL append failure | Fail turn before success | No acknowledged unlogged turn |
| SQLite failure | Stop or return explicit non-resumable error | Do not claim durable checkpoint |
| W&B unavailable | Continue local run and sync later | Local records remain canonical |
| Partial Parquet output | Mark run incomplete and exclude partition | Analysis reads validated partitions only |

## 16. Security and Privacy

- Secrets load from environment or approved host settings and never enter W&B configs or committed manifests.
- Pydantic models forbid extra fields at policy, model, and sandbox boundaries.
- Criterion-owned and simulator-private types cannot be imported by policy modules.
- Public condition, criterion, and privileged simulation writers use distinct schemas and paths.
- React receives simulator truth only in explicitly labelled synthetic Glass Box mode.
- Benchmark replay exposes criterion only after committed condition outputs.
- OpenRouter requests omit secrets and all context not required by the channel.
- Raw hidden model reasoning is not requested, logged, traced, or displayed.
- Generated or participant code never runs in the FastAPI process or local trusted executor.
- No human research data is collected by the dissertation POC.

## 17. Observability and Provenance

Required structured fields include:

- Benchmark version, manifest hash, case ID, task family, split, and condition.
- Tracker, policy, prompt, model, provider-route, code, and metric versions.
- Prediction/action commit timestamp, complete-set hash, and criterion-reveal timestamp.
- Public, evidence, and criterion channel request hashes.
- Sandbox operation ID, execution status, test counts, timeout/resource flags, and output hashes.
- Pair completeness, exclusion reason, invalid/missing status, and negative-control assignment.
- Session/turn IDs for interactive logs.
- Seed hierarchy and fully resolved Hydra configuration.
- OpenRouter latency, token use, retries, and bounded cost.

JSONL and Parquet are canonical. W&B provides experiment comparison and artifact references. Prometheus, Grafana, OpenTelemetry, Sentry, and LangSmith remain optional debugging aids rather than graduation dependencies.

## 18. Requirement Traceability

| Architecture area | Primary PRD requirements |
|---|---|
| Vertical slice | FR-VS-001 through FR-VS-010 |
| Benchmark definitions and isolation | FR-BMK-001 through FR-BMK-008, FR-BMK-013 through FR-BMK-016 |
| Benchmark metrics and controls | FR-BMK-009 through FR-BMK-012 |
| Experiment configuration | FR-EXP-001 through FR-EXP-006, NFR-REP-001 through NFR-REP-005 |
| Secondary simulator | FR-SIM-001 through FR-SIM-008 |
| CBFM | FR-CBFM-001 through FR-CBFM-009 |
| Tracking | FR-TRK-001 through FR-TRK-009 |
| Policies and RL | FR-POL-001 through FR-POL-009 |
| Interactive graph | FR-GRF-001 through FR-GRF-005 |
| LLM and guardrail | FR-GEN-001 through FR-GRD-003 |
| Trajectories and W&B | FR-TRJ-001 through FR-TRJ-006 |
| Demo | FR-UI-001 through FR-UI-006 |
| Code execution | SEC-SBX-001 through SEC-SBX-004 |
| Security and future pilot | NFR-SEC-001 through NFR-PRV-002 |

## 19. HLD Acceptance Criteria

This architecture is ready for low-level revision when:

- The deterministic browser vertical slice remains the first integration gate.
- Benchmark measurement is a distinct path from LangGraph and Gymnasium.
- At least 24 reviewed cases separate public, evidence, and criterion channels.
- Dialogue-only, valid-probe, unrelated-probe, and corrupted-probe conditions start from identical state.
- Every tracker prediction and policy action is committed before criterion reveal.
- Criterion records and simulator truth are unavailable through policy-safe contracts.
- Primary scoring uses the same held-out demonstrated-performance label for paired conditions.
- Deterministic fixtures are described only as pipeline validation.
- At least one pinned LLM-student condition is required before the primary claim can be evaluated.
- Repeated LLM samples are treated as nested observations rather than extra benchmark cases.
- Generated code requires the external sandbox; missing execution remains explicit missingness.
- Gymnasium, CBFM, SMC, and learned policies remain secondary to the measurement gate.
- Local records reproduce reports without W&B or network access.
- The demo remains usable in deterministic template mode.
- Production/distributed infrastructure and the human pilot remain explicitly deferred.
- All D2 diagrams compile successfully.
