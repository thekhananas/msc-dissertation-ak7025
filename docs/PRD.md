<!-- docs/PRD.md -->

# Product Requirements Document

## Socratic Multi-Agent Tutoring and Evaluation Platform

| Field | Value |
|---|---|
| Status | Draft for engineering review |
| Product phase | Simulation-first research platform |
| Primary source | `proposal/dissertation_proposal.tex` |
| Initial delivery horizon | MSc dissertation, June-September 2026 |
| Product owner | Dissertation researcher |
| Primary users | ML researcher, evaluator, system operator |
| Future users | Student and educator, after ethics approval |

## 1. Purpose

This document defines the product requirements for a Socratic multi-agent tutoring platform that evaluates adaptive pedagogical policies under partial observability. The platform orchestrates a tutor, epistemic tracker, pedagogical guardrail, and simulated student through LangGraph. It represents student knowledge and misconceptions as hidden state, approximates tutor-facing beliefs using Sequential Monte Carlo (SMC), and models bounded simulator constructs called Cognitive Bandwidth (`B_t`) and Cognitive Friction (`F_t`).

The initial product is an experimental instrument. It must support reproducible comparison of tutoring architectures and policy classes without claiming that synthetic student outcomes establish efficacy for human learners. A future human-facing mode is included as an architectural extension, but it remains disabled until ethics, privacy, security, and institutional approvals are complete.

## 2. Product Vision

Provide a transparent and falsifiable environment in which researchers can answer:

- Whether explicit epistemic tracking improves tutoring decisions over dialogue-only prompting.
- Whether private diagnostic probes detect unsupported public mastery claims.
- Whether guardrail filtering reduces direct-answer and solution-code leakage.
- Whether `B_t` and `F_t` add predictive or policy value beyond observable prompt features.
- Whether delayed-return Q-learning provides value over strong heuristics and contextual bandits.

The product succeeds when it makes those decisions credible, reproducible, and inspectable, including when the correct result is to reject an architectural component.

## 3. Product Principles

1. **Simulation truth is privileged.** Policies and tutor-generation components must never receive simulator-owned latent state.
2. **Claims follow evidence.** Internal state traces are not validation; held-out behavioral evidence and external ratings are required.
3. **Simpler policies are first-class.** Heuristics, BKT, and contextual bandits receive the same interface, data budget, and evaluation conditions as Q-learning.
4. **Reproducibility is a feature.** Every reported result must be recoverable from versioned code, manifests, seeds, prompts, model identifiers, and artifacts.
5. **Safety is enforced structurally.** Guardrails, bounded retries, safe fallbacks, isolated execution, and access controls are product requirements.
6. **Synthetic and human modes are different products.** Human deployment does not inherit validity merely because simulation is successful.
7. **No raw hidden reasoning.** The system may retain structured evidence and execution telemetry, but it must not expose or depend on hidden chain-of-thought.

## 4. Users and Stakeholders

### 4.1 Primary Personas

#### Researcher

Configures simulator cohorts, policies, ablations, model providers, and evaluation runs. Needs deterministic execution, complete provenance, cost controls, and statistically valid outputs.

#### Evaluator or Dissertation Reviewer

Inspects trajectories, aggregate results, policy comparisons, leakage examples, belief calibration, and CBFM diagnostics. Needs a clear distinction between observed evidence, tracker estimates, and privileged simulator truth.

#### System Operator

Deploys workers, monitors queues and provider usage, manages secrets, responds to failures, and controls policy promotion. Needs health signals, audit records, budgets, rollback, and runbooks.

### 4.2 Future Personas

#### Student

Receives Socratic prompts, writes code, submits explanations, and sees learning-oriented feedback. The student must not see private simulator variables or raw hidden reasoning.

#### Educator or Study Administrator

Selects tasks, reviews aggregate learning evidence, supervises a human pilot, and handles escalation. Access must be role-based and compliant with approved study procedures.

## 5. Product Modes

### 5.1 Deterministic Surrogate Mode

Uses authored templates and mathematical transition kernels without LLM generation. This is the reference mode for debugging, invariant testing, replay, and low-cost policy development.

### 5.2 LLM-Backed Simulation Mode

Runs tutor and simulated-student generation through pinned model providers while retaining simulator-owned latent state and controlled evidence channels. This mode is used for primary dissertation experiments.

### 5.3 Glass Box Demonstration Mode

Shows a synthetic episode with separate views of true simulator state, tracker estimates, policy actions, guardrail decisions, and structured scratchpad evidence. Privileged information must be visibly labeled as synthetic and unavailable in human mode.

### 5.4 Human Tutoring Mode

Replaces the simulated-student node with a durable LangGraph interrupt and browser interaction. This mode is future scope, disabled by default, and cannot be enabled without documented ethics and data-protection approval.

## 6. Goals and Non-Goals

### 6.1 Goals

- Build a reproducible Gymnasium-compatible tutoring environment.
- Implement a strict separation between latent state, belief state, workflow state, and policy observation.
- Support static, heuristic, BKT-driven, contextual-bandit, and tabular Q-learning policies.
- Produce auditable trajectories with decomposed rewards and complete provenance.
- Validate or reject CBFM using calibration, ablation, sensitivity, and held-out construct-utility tests.
- Measure diagnostic gain, leakage, sycophancy, calibration, latency, and cost.
- Provide a production-shaped architecture that can later support a governed human pilot.

### 6.2 Non-Goals

- Claiming that `B_t` directly measures human working memory or mental fatigue.
- Establishing clinical, psychological, or educational efficacy for human students from synthetic experiments.
- Online reinforcement learning against live students.
- Autonomous promotion of newly trained policies into serving.
- Training a foundation model from scratch.
- Supporting arbitrary programming languages in the initial release.
- Exposing private chain-of-thought or simulator-only variables to students.
- Treating LLM-as-judge scores as the sole source of educational validity.

## 7. Core Domain Model

The product must maintain these conceptual boundaries:

| Domain object | Owner | Visibility |
|---|---|---|
| `TrueStudentState` | Simulator environment | Privileged evaluation only |
| SMC particle set | Epistemic tracker | Tracker internals only |
| `TrackerEstimates` | Epistemic tracker | Tutor policy and research UI |
| `PolicyObservation` | Policy adapter | Selected policy only |
| `PolicyDecision` | Policy | Graph and trajectory record |
| `TutorControl` | Graph/environment boundary | Simulator and trajectory record |
| `ObservationEvidence` | Evidence pipeline | Tracker and evaluation |
| LangGraph checkpoint | LangGraph runtime | Runtime and operators |
| Trajectory | Trajectory service | Training and evaluation by access level |

The turn order is normative:

1. The policy selects a directive and target concept from tracker-visible information.
2. The tutor renders a candidate prompt.
3. The guardrail accepts, rewrites, or replaces the prompt.
4. The environment computes the realized prompt load and difficulty.
5. Fast cognitive dynamics produce post-prompt `B_t` and `F_t`.
6. The student produces public response and private diagnostic evidence.
7. The tracker corrects its belief using active evidence only.
8. Slow mastery and misconception transitions produce the next simulator state.

## 8. Functional Requirements

### 8.1 Session and Experiment Management

| ID | Requirement | Acceptance criteria | Priority |
|---|---|---|---|
| FR-EXP-001 | The system shall create an experiment from a versioned manifest. | A run is rejected if policy, simulator, model, prompt, split, seed, or reward configuration is missing. | Must |
| FR-EXP-002 | The system shall maintain disjoint calibration, policy-selection, validation, and held-out test partitions. | Automated validation detects overlapping profile IDs, task IDs, or diagnostic items and blocks execution. | Must |
| FR-EXP-003 | The system shall derive component-specific seeds from an episode root seed. | Template-mode replay produces byte-equivalent trajectory records excluding timestamps. | Must |
| FR-EXP-004 | The system shall support matched-seed comparisons across experimental conditions. | Every primary comparison pairs conditions by profile, task, initial state, and environment seed. | Must |
| FR-EXP-005 | The system shall allow bounded cancellation and resumption of batch runs. | A resumed run continues from completed shard boundaries without duplicate episode IDs. | Should |
| FR-EXP-006 | The system shall enforce per-run token, request, sandbox, time, and monetary budgets. | Exceeding a hard budget ends or truncates affected episodes with a typed reason. | Must |

### 8.2 Simulator Environment

| ID | Requirement | Acceptance criteria | Priority |
|---|---|---|---|
| FR-SIM-001 | The system shall expose `SocraticTutor/POMDP-v0` through the Gymnasium API. | `reset` and `step` pass Gymnasium environment checks and return typed, space-conformant values. | Must |
| FR-SIM-002 | The simulator shall exclusively own and mutate `TrueStudentState`. | Static dependency checks and tests prevent policy modules from importing or receiving the true-state type. | Must |
| FR-SIM-003 | The simulator shall implement separate fast cognitive and slow learning transitions. | Turn traces demonstrate that evidence is emitted after fast projection and before slow propagation. | Must |
| FR-SIM-004 | The simulator shall support configurable student profiles and misconception sets. | A manifest can load fixed profiles or sample from versioned profile distributions. | Must |
| FR-SIM-005 | The simulator shall enforce prerequisite-DAG constraints. | Invalid cyclic graphs are rejected; learning updates cannot silently bypass configured prerequisites. | Must |
| FR-SIM-006 | The simulator shall expose privileged state only through access-controlled evaluation records. | Policy observations and normal API responses contain no true mastery, misconception, `B_t`, or `F_t`. | Must |
| FR-SIM-007 | The simulator shall distinguish natural termination from truncation. | Completion, maximum-turn, budget, provider, and safety outcomes use distinct typed reasons. | Must |
| FR-SIM-008 | The simulator shall support deterministic template generators. | All core transitions and policies can be tested without external model calls. | Must |

### 8.3 Cognitive Bandwidth and Friction Model

| ID | Requirement | Acceptance criteria | Priority |
|---|---|---|---|
| FR-CBFM-001 | The system shall compute prompt load from normalized token, code-span, AST-depth, concept-novelty, and entropy features. | Feature values, normalization version, missing-value behavior, and aggregate load are recorded per turn. | Must |
| FR-CBFM-002 | The system shall compute overload friction from positive prompt-student difficulty mismatch and configured ZPD tolerance. | `F_t` remains in `[0,1]` and is weakly non-decreasing with positive mismatch when other inputs are fixed. | Must |
| FR-CBFM-003 | The system shall update bandwidth with bounded drain, recovery, friction, and optional process noise. | Property tests prove `B_t` remains in `[0,1]`; deterministic mode disables process noise. | Must |
| FR-CBFM-004 | The system shall retain underchallenge as a separate diagnostic. | Underchallenge is logged separately and is not substituted for overload friction in the primary reward. | Should |
| FR-CBFM-005 | The system shall support a complete CBFM ablation. | Ablation fixes `B_t=1` and `F_t=0` and removes CBFM features from policy observations. | Must |
| FR-CBFM-006 | The system shall calibrate CBFM only on the calibration partition and external load proxies. | Calibration code cannot use held-out outcomes or aggregate policy reward as its target. | Must |
| FR-CBFM-007 | The system shall compare mastery-only, prompt-feature-only, and CBFM-augmented load models. | Held-out Brier score and negative log-likelihood are reported with bootstrap intervals. | Must |
| FR-CBFM-008 | The system shall run monotonicity, recovery, sensitivity, and negative-control tests. | Violations are surfaced as failed construct-validity gates, not hidden in aggregate reward. | Must |
| FR-CBFM-009 | Product copy shall describe `B_t` and `F_t` as simulator constructs. | No user-facing or generated report labels them validated measures of human mental fatigue. | Must |

### 8.4 Epistemic Tracking

| ID | Requirement | Acceptance criteria | Priority |
|---|---|---|---|
| FR-TRK-001 | The tracker shall maintain an SMC approximation of the hidden-state belief. | Particle state and weights initialize from a versioned prior and produce posterior summaries each turn. | Must |
| FR-TRK-002 | The tracker shall apply active evidence masks. | Unmentioned and unprobed concepts or misconceptions contribute no observation likelihood term. | Must |
| FR-TRK-003 | The tracker shall correct particle weights in log space. | Extreme likelihood tests produce finite normalized weights without underflow. | Must |
| FR-TRK-004 | The tracker shall monitor effective sample size and resample below a configured threshold. | Resampling tests verify particle selection and uniform post-resampling weights. | Must |
| FR-TRK-005 | The tracker shall expose posterior means, misconception marginals, and uncertainty summaries. | The tutor receives summaries but never particles or simulator truth. | Must |
| FR-TRK-006 | The tracker shall distinguish particle-weight degeneracy from epistemic uncertainty. | ESS/weight entropy and posterior marginal uncertainty are recorded separately. | Must |
| FR-TRK-007 | The tracker shall support subjective parameters distinct from simulator parameters. | Robustness experiments can deliberately mis-specify tracker transition and noise parameters. | Must |
| FR-TRK-008 | The system shall provide BKT and last-observation tracker baselines. | Baselines implement the same tracker-summary contract used by policy adapters. | Must |
| FR-TRK-009 | The tracker shall be the single writer of tutor-facing belief estimates. | Graph-state validation rejects belief updates from all other nodes. | Must |

### 8.5 Tutor Policy and Prompt Generation

| ID | Requirement | Acceptance criteria | Priority |
|---|---|---|---|
| FR-POL-001 | All policies shall act over one finite, versioned directive space. | Unsupported directives fail validation before prompt generation. | Must |
| FR-POL-002 | Policies shall receive the same observation budget within a comparison. | Automated feature audits show identical allowed fields across static, heuristic, bandit, and Q-learning conditions. | Must |
| FR-POL-003 | The system shall implement static, expert heuristic, BKT-plus-heuristic, contextual-bandit, and tabular Q-learning policies. | Each policy completes the shared evaluation suite without policy-specific environment changes. | Must |
| FR-POL-004 | The policy shall select a directive and target concept, not arbitrary tutor text. | The generator receives a typed `PolicyDecision`; rendered text is stored in `TutorControl`. | Must |
| FR-POL-005 | The contextual bandit shall record action propensities. | Every training and evaluation decision includes propensity or an explicit deterministic-policy marker. | Must |
| FR-POL-006 | Q-learning shall use the proposal's compact discretized belief summary. | State-bin definitions and unseen-state fallback behavior are versioned and reported. | Must |
| FR-POL-007 | Training shall report Q-table coverage and visitation. | Reports include visited state-action fraction and per-state action support. | Must |
| FR-POL-008 | Serving shall load a frozen policy version for each episode. | A running episode cannot change policy artifact after initialization. | Must |
| FR-POL-009 | Policy promotion shall require explicit approval after held-out evaluation. | Training jobs cannot directly change the serving alias. | Must |
| FR-GEN-001 | Tutor generation shall be abstracted behind a typed model gateway. | Provider adapters return validated content, usage, latency, model ID, and retry metadata. | Must |
| FR-GEN-002 | Prompts and templates shall be versioned and hashed. | Every generated turn identifies its system prompt and template versions. | Must |
| FR-GEN-003 | The system shall support template-only, smaller-model, and stronger-model generation variants. | Generation-layer comparisons hold policy and episode manifests constant. | Must |

### 8.6 Guardrail and Pedagogical Safety

| ID | Requirement | Acceptance criteria | Priority |
|---|---|---|---|
| FR-GRD-001 | The guardrail shall detect direct answers, complete solution code, and prohibited leakage patterns. | A versioned adversarial fixture set reports recall, precision, and false-positive rate. | Must |
| FR-GRD-002 | Guardrail outputs shall use a validated structured schema. | Invalid or incomplete model output is treated as a failed check and cannot pass a prompt. | Must |
| FR-GRD-003 | Rewrites shall be bounded. | After the configured retry count, the graph selects a safe authored Socratic template. | Must |
| FR-GRD-004 | Guardrail events shall be observable but not reveal hidden answers to students. | Operator traces include reasons; student-facing payloads contain only safe feedback. | Must |
| FR-GRD-005 | The system shall evaluate leakage per turn and per dialogue. | Reports include Wilson intervals and false-positive effects on valid scaffolding. | Must |
| FR-GRD-006 | The system shall support guardrail removal as an ablation. | Guardrail-disabled runs preserve all other policy, model, seed, and task settings. | Must |

### 8.7 Student Simulation and Diagnostic Scratchpad

| ID | Requirement | Acceptance criteria | Priority |
|---|---|---|---|
| FR-STU-001 | The simulated student shall receive bounded context derived from its profile and state, not unrestricted expert history. | Context inspection tests show no unauthorized tutor solution or privileged true-state serialization. | Must |
| FR-STU-002 | Student behavior shall support mastery, misconception, and overload response policies. | Controlled fixtures activate each branch under known state and probe conditions. | Must |
| FR-STU-003 | The system shall generate private diagnostic probes targeted to active concepts or misconceptions. | Each probe records its target, version, expected outcome, and leakage relationship to the main task. | Must |
| FR-STU-004 | Diagnostic answers shall produce structured pass/fail and execution telemetry. | Evidence includes tests, exit status, bounded stdout/stderr summaries, and probe result. | Must |
| FR-STU-005 | Public mastery claims shall be distinguishable from demonstrated private performance. | Sycophancy metrics flag public mastery claims without supporting probe evidence. | Must |
| FR-STU-006 | The system shall support scratchpad removal as an ablation. | Dialogue-only inference runs retain matched seeds and all unrelated components. | Must |
| FR-STU-007 | Raw private chain-of-thought shall not be persisted or displayed. | Stored scratchpad artifacts contain code submissions, outputs, tests, and structured summaries only. | Must |

### 8.8 Code Execution

| ID | Requirement | Acceptance criteria | Priority |
|---|---|---|---|
| SEC-SBX-001 | Untrusted code shall execute only inside an isolated on-demand microVM or equivalent kernel boundary. | Production configuration rejects host-process execution. | Must |
| SEC-SBX-002 | Sandboxes shall have no outbound network and enforce CPU, memory, wall-time, process, filesystem, and output limits. | Integration tests verify every configured limit and typed failure result. | Must |
| SEC-SBX-003 | Sandbox requests shall be idempotent and auditable. | Duplicate idempotency keys do not create inconsistent evidence records. | Must |
| SEC-SBX-004 | Local execution shall be limited to trusted fixtures and explicit development mode. | Runtime configuration prevents local fallback in staging and production. | Must |
| SEC-SBX-005 | Execution telemetry shall be normalized before tracker ingestion. | Provider-specific responses map to a stable, versioned evidence contract. | Must |

### 8.9 LangGraph Orchestration

| ID | Requirement | Acceptance criteria | Priority |
|---|---|---|---|
| FR-GRF-001 | The system shall implement the proposal's cyclic tracker-policy-tutor-guardrail-student-evidence flow. | Integration tests verify the expected node order and conditional rewrite routing. | Must |
| FR-GRF-002 | Nodes shall return validated state deltas and respect declared field ownership. | Unauthorized writes fail before checkpoint commit. | Must |
| FR-GRF-003 | External side effects shall use stable idempotency keys. | Node replay does not duplicate LLM charges, sandbox records, or trajectory turns where provider support permits deduplication. | Must |
| FR-GRF-004 | Graph execution shall use durable checkpointing. | A worker restart resumes from the latest committed checkpoint without losing accepted evidence. | Must |
| FR-GRF-005 | Large particle and trajectory payloads shall remain outside graph checkpoints. | Checkpoints contain summaries and artifact references; size alarms detect regressions. | Must |
| FR-GRF-006 | Human mode shall pause with a dynamic interrupt and resume using the same thread identity. | End-to-end test pauses, persists, reconnects, validates input, and resumes once. | Future |
| FR-GRF-007 | The graph shall enforce terminal conditions and bounded loops. | Maximum turn, rewrite, retry, and provider budgets prevent unbounded execution. | Must |

### 8.10 Trajectories, Training, and Policy Registry

| ID | Requirement | Acceptance criteria | Priority |
|---|---|---|---|
| FR-TRJ-001 | Every turn shall emit an immutable, versioned trajectory record. | Records include observation, decision, control, evidence, decomposed reward, termination, versions, cost, latency, and seeds. | Must |
| FR-TRJ-002 | Privileged simulator fields shall be stored separately from policy-consumable trajectory fields. | Dataset validation proves policy loaders cannot select privileged columns. | Must |
| FR-TRJ-003 | Canonical trajectories shall be stored as partitioned Parquet. | Partitions support experiment, split, policy, profile, task, and date filtering. | Must |
| FR-RL-001 | Rollout workers shall construct environments locally rather than call a live tutoring API. | Training operates when the API and WebSocket services are unavailable. | Must |
| FR-RL-002 | The trainer shall consume environment factories or immutable trajectories. | Trainer dependencies exclude serving checkpoints and live session tables. | Must |
| FR-RL-003 | Training shall produce a versioned policy artifact and signed manifest. | Artifact records code commit, data version, hyperparameters, schema, metrics, and parent policy. | Must |
| FR-RL-004 | Policy candidates shall pass offline gates before registration or promotion. | Failed safety, leakage, reproducibility, or feature-audit checks block promotion. | Must |
| FR-RL-005 | Interrupted training shall resume at defined durable boundaries. | Recovery tests produce no duplicate episode or conflicting artifact IDs. | Should |

### 8.11 Evaluation and Reporting

| ID | Requirement | Acceptance criteria | Priority |
|---|---|---|---|
| FR-EVL-001 | The system shall execute the guardrail leakage gate. | Prompt-only and guarded conditions are compared under normal, adversarial, and persistent-confusion scenarios. | Must |
| FR-EVL-002 | The system shall execute the scratchpad falsification gate. | Dialogue-only and probe-supported mastery inference are compared on unsupported mastery detection. | Must |
| FR-EVL-003 | The system shall execute the CBFM construct-utility gate. | `M0`, `M1`, and `M2` are evaluated on held-out data with prespecified metrics. | Must |
| FR-EVL-004 | The system shall execute the policy-complexity gate. | Heuristic, bandit, and Q-learning policies are compared on diagnostic gain, leakage, latency, cost, and expert quality. | Must |
| FR-EVL-005 | The system shall support all proposal component ablations. | CBFM, SMC, guardrail, scratchpad, and LLM-generation removals run through a common manifest system. | Must |
| FR-EVL-006 | Primary comparisons shall report uncertainty and effect sizes. | Reports contain bootstrap confidence intervals, Wilson intervals for rates, and paired analysis where applicable. | Must |
| FR-EVL-007 | Evaluation shall preserve negative conclusions. | Gate outputs support `accepted`, `rejected`, and `inconclusive` without changing thresholds after test inspection. | Must |
| FR-EVL-008 | LLM judge outputs shall be blinded and schema-validated. | Condition names and privileged state are absent from judge inputs; malformed grades are rejected. | Must |
| FR-EVL-009 | Reports shall distinguish simulator reward from external outcomes. | Aggregate reward cannot be the only headline metric for any component claim. | Must |

### 8.12 Glass Box Interface

| ID | Requirement | Acceptance criteria | Priority |
|---|---|---|---|
| FR-UI-001 | The interface shall display the active dialogue, task, code editor, and structured execution results. | A synthetic episode can be started, observed, paused, and inspected from the browser. | Should |
| FR-UI-002 | The interface shall show true and estimated state as separate, labeled synthetic views. | Visual checks prevent true state from being presented as a tracker estimate. | Must |
| FR-UI-003 | The interface shall visualize `B_t`, `F_t`, uncertainty, guardrail events, and prerequisite mastery. | Values update per committed turn and identify their source and timestamp. | Should |
| FR-UI-004 | The interface shall display structured scratchpad evidence without hidden reasoning. | Only code, test outcomes, stdout/stderr summaries, and probe summaries appear. | Must |
| FR-UI-005 | The interface shall support reconnecting to resumable sessions. | Reconnection restores the latest committed turn without duplicating user input. | Should |
| FR-UI-006 | Privileged synthetic controls shall be unavailable in human mode. | Role and mode tests prevent students from requesting true-state endpoints. | Must |

## 9. Non-Functional Requirements

### 9.1 Reproducibility and Scientific Integrity

| ID | Requirement | Target |
|---|---|---|
| NFR-REP-001 | Every reported run must identify code, configuration, data, prompt, model, policy, and schema versions. | 100% provenance completeness |
| NFR-REP-002 | Deterministic template runs must be replayable. | Byte-equivalent content excluding timestamps and generated IDs |
| NFR-REP-003 | Primary train/test split manifests must be immutable after preregistration. | Hash mismatch blocks execution |
| NFR-REP-004 | Derived reports must reference source trajectory partitions and analysis code. | 100% report-to-artifact traceability |
| NFR-REP-005 | Randomness from environment, policy exploration, and model generation must use separate seed namespaces. | Verified in manifest and replay tests |

### 9.2 Reliability and Performance

| ID | Requirement | Target |
|---|---|---|
| NFR-REL-001 | A graph worker restart must not lose committed turn state. | Recovery from latest checkpoint |
| NFR-REL-002 | All loops and external calls must have explicit limits. | No unbounded retries or episode loops |
| NFR-REL-003 | Batch jobs must isolate failed episodes. | One failed episode does not fail an entire completed shard |
| NFR-PERF-001 | Non-model orchestration overhead per turn must remain bounded. | p95 below 500 ms in local integration tests |
| NFR-PERF-002 | Live mode must stream or acknowledge accepted user input promptly. | p95 acknowledgement below 500 ms, excluding network conditions |
| NFR-PERF-003 | Experiment concurrency must respect provider rate and budget limits. | Zero uncontrolled rate-limit retry storms |

### 9.3 Security and Privacy

| ID | Requirement | Target |
|---|---|---|
| NFR-SEC-001 | Secrets must be loaded from an approved secret store or local untracked environment file. | No secrets in repository or trajectory artifacts |
| NFR-SEC-002 | Access to privileged simulator state must be role- and mode-restricted. | Deny by default |
| NFR-SEC-003 | External traces must be redacted before transmission. | No credentials, hidden answers, or future human PII |
| NFR-SEC-004 | Administrative actions and policy promotions must be auditable. | Actor, timestamp, artifact, and reason recorded |
| NFR-PRV-001 | Synthetic experiments must not require personally identifiable information. | No PII fields in simulation schemas |
| NFR-PRV-002 | Human-mode retention and deletion rules must be configured before activation. | Activation blocked without approved policy |

### 9.4 Observability and Cost

| ID | Requirement | Target |
|---|---|---|
| NFR-OBS-001 | Every episode and external call must share correlated trace identifiers. | 100% correlation for accepted turns |
| NFR-OBS-002 | Operators must observe queue depth, failures, latency, token usage, sandbox usage, and cost. | Dashboards and alerts available before batch scale-up |
| NFR-OBS-003 | Scientific metrics must be stored outside transient tracing systems. | Canonical artifacts in object storage and MLflow |
| NFR-COST-001 | Every experiment must declare hard budgets. | Execution blocked without budget configuration |
| NFR-COST-002 | Cost must be attributable by experiment, condition, model, and policy. | Aggregate discrepancy below 2% of provider-recorded usage where available |

### 9.5 Maintainability and Compatibility

| ID | Requirement | Target |
|---|---|---|
| NFR-MNT-001 | Shared contracts must be versioned and compatibility-tested. | Breaking changes require a schema-version increment |
| NFR-MNT-002 | Core mathematical behavior must be independent of external model providers. | Full unit suite runs offline |
| NFR-MNT-003 | Production logic must not depend on notebooks. | All runs available through package APIs or CLI |
| NFR-MNT-004 | Static analysis and tests must run in CI. | Ruff, Pyright, pytest, contract checks, and Gymnasium checks pass |

### 9.6 Accessibility and Human Factors

| ID | Requirement | Target |
|---|---|---|
| NFR-ACC-001 | The Glass Box must support keyboard navigation and visible focus states. | WCAG 2.2 AA-oriented audit before human demo |
| NFR-ACC-002 | State visualizations must not rely on color alone. | Labels, values, or patterns accompany color encoding |
| NFR-ACC-003 | Cognitive-state labels must communicate uncertainty and simulation status. | No deterministic or clinical wording |

## 10. User Journeys

### 10.1 Run a Reproducible Policy Comparison

1. The researcher selects a signed experiment manifest.
2. The system validates splits, schemas, budgets, providers, and policy features.
3. Rollout workers execute matched episodes locally through Gymnasium environments.
4. Immutable trajectories are validated and written to object storage.
5. Evaluation workers compute prespecified metrics and intervals.
6. MLflow records artifacts, parameters, and gate outcomes.
7. The researcher reviews differences without seeing held-out results during tuning.

Success condition: another operator can reproduce the report from the same manifest and code revision.

### 10.2 Inspect an Episode in the Glass Box

1. The evaluator opens a completed synthetic episode.
2. The interface displays the current tutor prompt, student response, directive, and evidence.
3. True state and tracker belief are shown side by side with explicit synthetic labels.
4. The evaluator navigates turn history and inspects guardrail or probe events.
5. The interface shows provenance and links to the experiment result.

Success condition: the evaluator can identify why a policy acted without being shown raw hidden reasoning.

### 10.3 Train and Promote a Candidate Policy

1. The researcher submits a training manifest referencing approved trajectory partitions.
2. The trainer performs feature leakage checks and trains the candidate.
3. The candidate is registered with its complete provenance.
4. Held-out evaluation executes the required safety and complexity gates.
5. An authorized operator reviews the report and explicitly promotes or rejects the candidate.

Success condition: no training process can mutate the serving policy alias directly.

### 10.4 Future Human Interaction

1. An authenticated student starts an approved learning session.
2. LangGraph loads prior tracker estimates without simulator truth.
3. The graph generates and guardrails a Socratic prompt, then interrupts.
4. The browser displays the prompt and accepts text or code.
5. Code is evaluated in a microVM and normalized evidence resumes the graph.
6. The tracker updates beliefs and the next turn begins.

Success condition: the workflow is resumable, privacy-preserving, and cannot expose synthetic privileged state.

## 11. Success Metrics and Decision Gates

### 11.1 Scientific Metrics

- Held-out normalized diagnostic gain and bootstrap confidence interval.
- Probe and code-execution performance on unseen diagnostic items.
- Tracker calibration, Brier score, negative log-likelihood, and posterior dispersion.
- False mastery detection rate and public-claim/probe disagreement.
- Leakage rate per turn and per dialogue with Wilson intervals.
- CBFM held-out construct utility relative to prompt-feature controls.
- Q-table coverage, policy return, action support, and performance by subgroup.
- Expert-rated Socratic quality and prompt-load appropriateness.

### 11.2 Operational Metrics

- End-to-end and per-node latency.
- Provider failures, retries, and fallback rate.
- Token and monetary cost per episode and condition.
- Sandbox timeout and rejection rate.
- Checkpoint recovery success.
- Trajectory validation failure rate.
- Queue depth and worker utilization.

### 11.3 Decision Gates

| Gate | Survives when | Rejected when |
|---|---|---|
| Guardrail | Leakage falls materially with tolerable false positives and latency | Blocking or cost outweighs safety gain |
| Scratchpad | Unsupported mastery is detected beyond dialogue-only inference | Public dialogue is sufficient in tested conditions |
| CBFM | `B_t` and `F_t` improve held-out load prediction beyond prompt features and show robust qualitative behavior | Internal traces add no predictive value or are parameter-fragile |
| Policy complexity | Q-learning improves held-out gain or expert quality over heuristic and bandit without safety/cost regression | Simpler adaptive policy matches or outperforms it |
| Generation model | Stronger model improves expert quality or out-of-template robustness at acceptable cost | Template or smaller model is sufficient |

Each gate returns `accepted`, `rejected`, or `inconclusive`. No component is mandatory merely because it appears in the proposed architecture.

## 12. Analytics and Reporting Requirements

- All primary metrics must be computable from immutable trajectory and evaluation artifacts.
- Aggregate views must support filtering by experiment, condition, policy, model, profile, task, and split.
- Reports must include sample counts, exclusions, missingness, and truncation reasons.
- Paired analyses must preserve matching identifiers.
- Results must identify whether metrics use privileged simulator state, observable evidence, tracker belief, or external ratings.
- LLM judge metrics must report model, prompt, temperature, repetitions, disagreement, and schema failure rate.
- Generated figures and tables must be traceable to analysis code and source artifact hashes.

## 13. Release Strategy

### Release 0: Contracts and Deterministic Core

Deliver schemas, manifests, Gymnasium environment, mathematical transitions, tracker core, template generators, and offline tests.

Exit criteria: deterministic replay, state-ownership tests, CBFM invariant tests, and tracker numerical tests pass.

### Release 1: LLM-Backed Simulation

Deliver model gateway, tutor/student generation, guardrail, scratchpad, sandbox adapter, LangGraph cycle, and trajectory pipeline.

Exit criteria: bounded end-to-end episodes complete with full provenance and no privileged-feature leakage.

### Release 2: Policy Training and Evaluation

Deliver all baselines, rollout jobs, training, policy registry, ablations, statistical analysis, and decision-gate reports.

Exit criteria: the preregistered experiment matrix can run from manifests and produce reproducible reports.

### Release 3: Glass Box Demonstrator

Deliver API, WebSocket flow, React/Monaco interface, synthetic state visualization, observability, and deployment automation.

Exit criteria: reviewers can inspect synthetic episodes and state provenance without accessing raw hidden reasoning.

### Release 4: Human-Pilot Readiness

Deliver only after separate approval: authentication integration, consent workflow, privacy controls, retention/deletion, incident response, accessibility review, and ethics-approved study configuration.

Exit criteria: institutional approvals and an independent readiness review are recorded. This release is not required for the simulation-phase dissertation claims.

## 14. Dependencies

- LangGraph and a durable PostgreSQL checkpointer.
- Pydantic v2 and Gymnasium.
- FastAPI, PostgreSQL, Redis Streams, and S3-compatible storage.
- OpenRouter/Cerebras or equivalent pinned model endpoints.
- E2B, Modal, or an equivalent isolated sandbox provider.
- MLflow for scientific artifacts and policy registration.
- LangSmith for graph and LLM tracing/evaluation.
- APPS-style programming tasks and versioned misconception/task fixtures with compatible licenses.
- Independent experts or approved evaluators for transcript grading.

## 15. Risks and Mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Simulator-policy overfitting | Apparent gains do not transfer beyond authored dynamics | Domain randomization, held-out profiles/tasks, subjective tracker mismatch, external diagnostics |
| CBFM circular validation | The model predicts labels derived from itself | External proxies, frozen calibration split, prompt-feature control, negative controls |
| Privileged-state leakage | Invalid policy evaluation | Separate schemas/storage, dependency rules, feature audits, adversarial tests |
| Sparse Q-table coverage | Unreliable RL conclusions | Compact state, coverage reporting, bandit/heuristic controls, reject RL if unsupported |
| LLM nondeterminism | Poor reproducibility | Pinned versions, recorded responses, separated seeds, repeated runs, template reference mode |
| Judge bias | Misleading quality conclusions | Blinding, multiple evaluators, structured rubrics, disagreement reporting, expert checks |
| Prompt or answer leakage | Inflated learning and unsafe tutoring | Versioned guardrail tests, bounded rewrites, held-out probes, task/probe separation |
| Sandbox escape | Host or data compromise | External microVM boundary, no network, strict limits, no production local fallback |
| Provider outage or cost spike | Incomplete experiments | Budgets, concurrency control, retries, fallback templates, resumable shards |
| Human-mode overreach | Ethical or privacy violation | Disabled-by-default feature flag and approval gate |
| Trace data exposure | Leakage of answers or future PII | Redaction, sampling, access control, retention policies |

## 16. Open Product Questions

These questions do not block the simulation-first implementation but must be resolved before the relevant release:

- Which exact programming concepts and prerequisite DAG constitute the first benchmark domain?
- Which model versions remain available and cost-effective at experiment freeze time?
- What practical-effect threshold will be preregistered for each decision gate?
- Which expert-rating protocol and inter-rater agreement threshold will be used?
- Which institutional identity provider, retention policy, and consent workflow would govern a future human pilot?

Answers must be recorded in versioned experiment or governance documents rather than silently embedded in code.

## 17. Requirement Traceability

| Research objective | Primary requirements | Decision evidence |
|---|---|---|
| Reliable belief tracking | FR-TRK-001 through FR-TRK-009, FR-SIM-002 | Calibration, posterior contraction, tracker ablation |
| Guardrail effectiveness | FR-GRD-001 through FR-GRD-006 | Leakage and false-positive gate |
| Scratchpad falsification | FR-STU-003 through FR-STU-007 | Unsupported mastery detection gate |
| CBFM utility | FR-CBFM-001 through FR-CBFM-009 | `M0`/`M1`/`M2`, ablation, sensitivity |
| RL marginal value | FR-POL-001 through FR-POL-009, FR-RL-001 through FR-RL-005 | Heuristic/bandit/Q-learning comparison |
| Reproducible research | FR-EXP-001 through FR-EXP-006, NFR-REP-001 through NFR-REP-005 | Replay and artifact audit |
| Safe production pathway | SEC-SBX-001 through SEC-SBX-005, NFR-SEC-001 through NFR-PRV-002 | Security tests and governance approval |

## 18. Definition of Done for the Product Phase

The simulation-first product phase is complete when:

- All Must requirements applicable to Releases 0-3 have automated or documented acceptance evidence.
- The deterministic and LLM-backed environments run from versioned manifests.
- All policy baselines use the same observation and action contracts.
- Privileged-state leakage checks pass.
- The four primary decision gates execute on held-out partitions.
- Results include uncertainty, cost, latency, and negative conclusions where supported.
- A clean environment can reproduce the primary report from archived artifacts and instructions.
- The Glass Box accurately distinguishes simulator truth, tracker belief, and observable evidence.
- Security controls prevent host execution of untrusted code.
- Documentation explicitly limits claims to simulation and does not present CBFM as validated human fatigue measurement.

Human tutoring mode has a separate definition of done and remains incomplete until ethics, privacy, accessibility, security, and study-governance requirements are approved and verified.
