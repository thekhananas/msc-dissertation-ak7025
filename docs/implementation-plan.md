<!-- docs/implementation-plan.md -->

# SMART Implementation Plan

## Socratic Multi-Agent Tutoring and Evaluation Platform

| Field | Value |
|---|---|
| Status | Active implementation checklist |
| Baseline date | 2026-07-11 |
| Dissertation delivery target | 2026-09-15 |
| Source requirements | `docs/PRD.md` |
| Architecture | `docs/HLD.md`, `docs/LLD.md`, `docs/data-flow-and-schema.md` |
| Planning approach | Simulation-first, evidence-gated, critical-path delivery |

## 1. How to Use This Checklist

Task status uses standard Markdown checkboxes:

- `[ ]` means the task is not yet complete. This includes blocked and in-progress work.
- `[x]` means the stated completion condition has been met and its evidence exists.

Rules:

1. Do not mark a task complete because code was written; mark it complete only after the listed evidence passes review.
2. Do not mark a milestone complete until its milestone gate is checked.
3. Record evidence as a repository path, CI run, MLflow run, report artifact, or signed decision record.
4. Keep task IDs stable. Add follow-up tasks rather than renumbering completed work.
5. A negative experiment result can complete a task when the test was correctly executed and reported.
6. Human-subject work remains unchecked and blocked until the required approvals exist.
7. Dates are rebaselined from 2026-07-11. Any missed critical-path date requires a scope and schedule review within one working day.

Every task below is SMART:

- **Specific:** one named deliverable or behavior.
- **Measurable:** a concrete done condition and evidence.
- **Achievable:** scoped to one implementation or verification outcome.
- **Relevant:** tied to requirements or a milestone gate.
- **Time-bound:** assigned a target date.

## 2. Milestone Overview

- [ ] **M0 - Engineering foundation** - Target: 2026-07-14
- [ ] **M1 - Contracts and reproducibility** - Target: 2026-07-18
- [ ] **M2 - Deterministic POMDP simulator** - Target: 2026-07-24
- [ ] **M3 - Cognitive Bandwidth and Friction Model** - Target: 2026-07-28
- [ ] **M4 - Epistemic tracker and baselines** - Target: 2026-07-31
- [ ] **M5 - Tutor generation and guardrails** - Target: 2026-08-03
- [ ] **M6 - Student simulator and diagnostic scratchpad** - Target: 2026-08-06
- [ ] **M7 - LangGraph orchestration** - Target: 2026-08-09
- [ ] **M8 - Policy baselines** - Target: 2026-08-12
- [ ] **M9 - RL rollout and training infrastructure** - Target: 2026-08-17
- [ ] **M10 - Evaluation and statistical pipeline** - Target: 2026-08-22
- [ ] **M11 - Glass Box application** - Target: 2026-08-26
- [ ] **M12 - Security and operational hardening** - Target: 2026-08-28
- [ ] **M13 - Deployment and release automation** - Target: 2026-08-30
- [ ] **M14 - Dissertation experiment execution** - Target: 2026-09-07
- [ ] **M15 - Future human-pilot readiness decision** - Target: 2026-09-09
- [ ] **M16 - Final documentation, dissertation, and handover** - Target: 2026-09-15

## 3. Critical Path and Parallel Work

The critical path is:

```text
M0 -> M1 -> M2 -> M3/M4 -> M7 -> M8 -> M9 -> M10 -> M14 -> M16
```

Parallel tracks:

- M5 and M6 can begin once M1 contracts are stable and use deterministic fakes until providers are integrated.
- M11 can begin against mocked API contracts after M1 and run in parallel with M8-M10.
- M12 and M13 run continuously but have final gates after the integrated system exists.
- M15 is a governance-readiness track. It cannot authorize live human data collection.

If schedule pressure threatens M14, preserve the deterministic simulator, tracker, core baselines, four decision gates, and reproducibility package. Defer optional neural RL, multiple sandbox providers, advanced UI polish, and human-mode implementation first.

---

## M0 - Engineering Foundation

**Objective:** Establish a reproducible monorepo, quality gates, local services, and approved documentation baseline.

**Target:** 2026-07-14  
**Dependencies:** None  
**Primary requirements:** NFR-MNT-001 through NFR-MNT-004, NFR-REP-001

- [x] **M0.1 Publish the production engineering plan.** Target: 2026-07-11. Done when `docs/engineering-plan.md` exists and passes whitespace/fence checks. Evidence: `docs/engineering-plan.md`.
- [x] **M0.2 Publish the PRD with stable requirement IDs.** Target: 2026-07-11. Done when functional/non-functional requirements, decision gates, and scope limits are documented with unique IDs. Evidence: `docs/PRD.md`.
- [x] **M0.3 Publish the HLD with compiled architecture diagrams.** Target: 2026-07-11. Done when system context, logical architecture, and deployment D2 diagrams compile. Evidence: `docs/HLD.md` and D2 validation output.
- [x] **M0.4 Publish the LLD with compiled class and sequence diagrams.** Target: 2026-07-11. Done when interfaces, algorithms, graph nodes, and five D2 diagrams are documented and compile. Evidence: `docs/LLD.md`.
- [x] **M0.5 Publish the data-flow and schema design.** Target: 2026-07-11. Done when PostgreSQL, Parquet, lineage, retention, and six D2 diagrams are documented and compile. Evidence: `docs/data-flow-and-schema.md`.
- [x] **M0.6 Publish this SMART implementation checklist.** Target: 2026-07-11. Done when all implementation stages through final handover have dated tasks and milestone gates. Evidence: `docs/implementation-plan.md`.
- [ ] **M0.7 Create the monorepo package and application directories.** Target: 2026-07-12. Done when the HLD layout exists with importable placeholder packages and no generated cache files tracked. Evidence: repository tree and import smoke test.
- [ ] **M0.8 Configure Python 3.12 and locked dependencies with `uv`.** Target: 2026-07-12. Done when `pyproject.toml` and `uv.lock` install successfully in a clean environment. Evidence: clean-install CI job.
- [ ] **M0.9 Configure Ruff, Pyright, pytest, Hypothesis, and pre-commit.** Target: 2026-07-13. Done when all tools run from documented commands and fail on an intentional fixture violation. Evidence: CI quality-gate run.
- [ ] **M0.10 Create the initial GitHub Actions pipeline.** Target: 2026-07-13. Done when lint, type, unit, contract, D2, and secret-scan jobs run on pull requests. Evidence: passing workflow run.
- [ ] **M0.11 Create local Docker Compose services.** Target: 2026-07-14. Done when PostgreSQL, Redis, MLflow, API placeholder, and worker placeholder become healthy from one command. Evidence: compose health-check output.
- [ ] **M0.12 Implement typed deployment settings and secret loading.** Target: 2026-07-14. Done when local safe defaults validate, secrets stay untracked, and unsafe production fallbacks fail startup. Evidence: settings tests.
- [ ] **M0.13 Record initial architecture decisions.** Target: 2026-07-14. Done when monorepo, storage, queue, sandbox, training separation, and human-mode decisions have ADRs. Evidence: `docs/adr/`.
- [ ] **M0.GATE Complete the engineering-foundation gate.** Target: 2026-07-14. Done when a clean checkout installs, starts local dependencies, and passes the initial CI pipeline. Evidence: tagged foundation CI run.

---

## M1 - Contracts and Reproducibility

**Objective:** Freeze version-1 domain contracts and deterministic experiment identity before implementing behavior.

**Target:** 2026-07-18  
**Dependencies:** M0  
**Primary requirements:** FR-EXP-001 through FR-EXP-006, FR-POL-001, FR-TRJ-001, NFR-REP-001 through NFR-REP-005

- [ ] **M1.1 Implement opaque UUIDv7 identifier types.** Target: 2026-07-15. Done when all experiment, session, episode, turn, policy, task, profile, shard, and artifact IDs validate and serialize consistently. Evidence: identifier contract tests.
- [ ] **M1.2 Implement shared enums and versioned action-space mapping.** Target: 2026-07-15. Done when directives, modes, splits, statuses, and termination reasons have stable serialization and explicit indices. Evidence: JSON Schema and snapshot tests.
- [ ] **M1.3 Implement policy-safe Pydantic contracts.** Target: 2026-07-16. Done when `PolicyObservation`, `PolicyDecision`, `TutorControl`, and supporting types use `extra="forbid"`, finite values, and schema version `1`. Evidence: contract tests and generated JSON Schema.
- [ ] **M1.4 Implement evidence and tracker-summary contracts.** Target: 2026-07-16. Done when active masks, execution/probe summaries, load proxies, posterior summaries, and uncertainty validate. Evidence: positive and adversarial schema fixtures.
- [ ] **M1.5 Implement experiment and episode manifests.** Target: 2026-07-16. Done when manifests require code, data, prompts, models, policy, reward, budgets, seeds, splits, and gate versions. Evidence: manifest schema and invalid-manifest tests.
- [ ] **M1.6 Implement canonical hashing and signing hooks.** Target: 2026-07-17. Done when semantically identical manifests hash identically and changed scientific parameters alter the hash. Evidence: canonicalization tests.
- [ ] **M1.7 Implement the deterministic seed hierarchy.** Target: 2026-07-17. Done when root seeds derive independent environment, transition, observation, policy, and generation seeds without order dependence. Evidence: seed vector fixtures.
- [ ] **M1.8 Implement split-manifest validation.** Target: 2026-07-17. Done when overlapping profile, task, or diagnostic members block execution. Evidence: overlap and clean-split tests.
- [ ] **M1.9 Enforce package import boundaries.** Target: 2026-07-18. Done when policies, generation, API, and human graph modules cannot import `simulator.private_state`. Evidence: import-linter CI test.
- [ ] **M1.10 Generate and snapshot public JSON Schemas.** Target: 2026-07-18. Done when schema changes are reviewed and breaking changes require a version increment. Evidence: `data/schemas/` snapshots.
- [ ] **M1.11 Create representative version-1 contract fixtures.** Target: 2026-07-18. Done when fixtures cover normal, sparse-evidence, guardrail-rewrite, sandbox-failure, terminal, and truncation turns. Evidence: `tests/fixtures/contracts/`.
- [ ] **M1.GATE Freeze contracts version 1.** Target: 2026-07-18. Done when JSON Schemas, seed vectors, split checks, and import-boundary tests pass in CI. Evidence: contract-freeze tag and CI run.

---

## M2 - Deterministic POMDP Simulator

**Objective:** Deliver an offline, reproducible Gymnasium environment implementing the proposal's turn timing and private-state ownership.

**Target:** 2026-07-24  
**Dependencies:** M1  
**Primary requirements:** FR-SIM-001 through FR-SIM-008

- [ ] **M2.1 Implement simulator-private state types.** Target: 2026-07-19. Done when true mastery, misconceptions, bandwidth, friction, and underchallenge exist only under the private simulator package. Evidence: type tests and import-boundary test.
- [ ] **M2.2 Implement concept, prerequisite-DAG, task, and profile loaders.** Target: 2026-07-19. Done when versioned fixtures load and cyclic prerequisite graphs fail validation. Evidence: catalog loader tests.
- [ ] **M2.3 Implement deterministic profile sampling and initialization.** Target: 2026-07-20. Done when a seed reproduces task, profile, initial mastery, misconceptions, and parameter draws. Evidence: initialization golden vectors.
- [ ] **M2.4 Implement true slow mastery and misconception transitions.** Target: 2026-07-21. Done when learning, forgetting, misconception activation/resolution, and prerequisite constraints pass property tests. Evidence: transition test suite.
- [ ] **M2.5 Implement decomposed reward calculation.** Target: 2026-07-21. Done when mastery gain, overload, bandwidth, leakage, and optional efficiency terms sum exactly within tolerance. Evidence: reward fixtures.
- [ ] **M2.6 Implement termination and truncation rules.** Target: 2026-07-21. Done when mastery/task completion, max turns, budget, provider, safety, and cancellation produce typed outcomes. Evidence: terminal-route tests.
- [ ] **M2.7 Implement deterministic tutor and student template engines.** Target: 2026-07-22. Done when every directive and student branch runs without external providers. Evidence: template coverage tests.
- [ ] **M2.8 Implement `SocraticTutorEnv.reset`.** Target: 2026-07-22. Done when seeding, private-state initialization, tracker prior, target selection, and policy-safe observation conform to Gymnasium. Evidence: reset tests.
- [ ] **M2.9 Implement `SocraticTutorEnv.step` with fast-evidence-slow ordering.** Target: 2026-07-23. Done when a turn realizes control, projects cognitive state, emits evidence, transitions learning, updates tracker, and returns a valid tuple. Evidence: order-spy integration test.
- [ ] **M2.10 Register `SocraticTutor/POMDP-v0`.** Target: 2026-07-23. Done when `gymnasium.make` and `check_env` pass in CI. Evidence: environment contract job.
- [ ] **M2.11 Implement privileged/public `info` separation.** Target: 2026-07-23. Done when policy wrappers cannot access private state and evaluation adapters can access labeled references only. Evidence: leakage tests.
- [ ] **M2.12 Create golden deterministic episode replays.** Target: 2026-07-24. Done when at least five profiles and five tasks reproduce byte-equivalent content excluding timestamps/IDs. Evidence: golden trajectory fixtures.
- [ ] **M2.GATE Complete the deterministic simulator gate.** Target: 2026-07-24. Done when Gymnasium, timing, state-ownership, terminal, reward, and replay tests pass offline. Evidence: simulator CI report.

---

## M3 - Cognitive Bandwidth and Friction Model

**Objective:** Implement bounded, inspectable CBFM dynamics and a calibration workflow that can reject the construct.

**Target:** 2026-07-28  
**Dependencies:** M1, M2  
**Primary requirements:** FR-CBFM-001 through FR-CBFM-009

- [ ] **M3.1 Implement prompt token and code-span features.** Target: 2026-07-22. Done when pinned tokenization and fenced/inline code extraction produce bounded raw and normalized features. Evidence: prompt fixture tests.
- [ ] **M3.2 Implement AST-depth extraction with parse-failure policy.** Target: 2026-07-23. Done when valid Python yields stable depth and invalid snippets use the configured conservative value with diagnostics. Evidence: AST fixture suite.
- [ ] **M3.3 Implement concept novelty and entropy features.** Target: 2026-07-23. Done when ontology matching, demonstrated-concept context, and entropy calculations are deterministic. Evidence: feature-vector snapshots.
- [ ] **M3.4 Implement versioned feature normalizers and weighted load.** Target: 2026-07-24. Done when weights are non-negative, sum to one, and produce `L_t` in `[0,1]`. Evidence: property tests and normalizer artifact fixture.
- [ ] **M3.5 Implement overload friction and underchallenge.** Target: 2026-07-24. Done when positive mismatch increases `F_t`, tolerated ZPD produces zero overload, and underchallenge is separate. Evidence: monotonicity tests.
- [ ] **M3.6 Implement bandwidth drain, recovery, and seeded noise.** Target: 2026-07-25. Done when `B_t` remains bounded, deterministic mode disables noise, and low-load sequences recover in expectation. Evidence: Hypothesis property tests.
- [ ] **M3.7 Implement soft overload probability.** Target: 2026-07-25. Done when lower bandwidth weakly increases overload probability and extreme parameters remain numerically stable. Evidence: numerical tests.
- [ ] **M3.8 Implement complete CBFM ablation.** Target: 2026-07-25. Done when ablation fixes true and estimated inputs as specified and removes CBFM policy features. Evidence: ablation contract test.
- [ ] **M3.9 Implement calibration artifact fitting.** Target: 2026-07-26. Done when calibration uses external proxies only and stores coefficients, normalizers, split hash, objective, convergence, and diagnostics. Evidence: calibration smoke run.
- [ ] **M3.10 Implement mastery-only, prompt-only, and CBFM-augmented predictors.** Target: 2026-07-27. Done when `M0`, `M1`, and `M2` share splits and report held-out Brier/NLL interfaces. Evidence: synthetic calibration test.
- [ ] **M3.11 Implement parameter sensitivity and negative controls.** Target: 2026-07-27. Done when single-parameter 0.5x/1x/1.5x sweeps and shuffled-load controls run from manifests. Evidence: sensitivity report fixture.
- [ ] **M3.12 Add construct-language guard tests.** Target: 2026-07-28. Done when reports/UI labels identify CBFM as a simulator construct and prohibited human-fatigue claims fail content checks. Evidence: copy regression tests.
- [ ] **M3.GATE Complete the CBFM implementation gate.** Target: 2026-07-28. Done when boundedness, monotonicity, recovery, ablation, calibration, sensitivity, and negative-control tests pass. Evidence: CBFM validation report.

---

## M4 - Epistemic Tracker and Baselines

**Objective:** Deliver numerically stable SMC tracking plus simpler comparison trackers through one summary interface.

**Target:** 2026-07-31  
**Dependencies:** M1, M2; M3 fast-state interface  
**Primary requirements:** FR-TRK-001 through FR-TRK-009

- [ ] **M4.1 Implement tracker priors and particle initialization.** Target: 2026-07-23. Done when seeded priors produce valid weighted particles without copying simulator truth. Evidence: prior fixtures.
- [ ] **M4.2 Implement active evidence-mask validation.** Target: 2026-07-24. Done when unknown IDs fail and empty/unmentioned dimensions produce neutral likelihood. Evidence: sparse-evidence tests.
- [ ] **M4.3 Implement masked mastery and misconception likelihoods.** Target: 2026-07-25. Done when classifier noise parameters are bounded and controlled cases rank particles correctly. Evidence: likelihood fixtures.
- [ ] **M4.4 Implement log-space correction and smoothing.** Target: 2026-07-26. Done when extreme evidence produces finite normalized weights without underflow. Evidence: numerical stress tests.
- [ ] **M4.5 Implement ESS, weight entropy, and posterior uncertainty.** Target: 2026-07-26. Done when degeneracy and epistemic uncertainty are computed and stored separately. Evidence: diagnostic tests.
- [ ] **M4.6 Implement systematic resampling.** Target: 2026-07-27. Done when threshold crossing resamples reproducibly and resets weights uniformly. Evidence: seeded resampling tests.
- [ ] **M4.7 Implement subjective slow propagation.** Target: 2026-07-27. Done when tracker parameters can differ from simulator parameters and propagation occurs after evidence correction. Evidence: ordering and misspecification tests.
- [ ] **M4.8 Implement posterior summary projection.** Target: 2026-07-28. Done when mastery means, misconception marginals, estimated cognitive state, and uncertainty satisfy the public contract. Evidence: summary schema tests.
- [ ] **M4.9 Implement numerical failure policy and diagnostics.** Target: 2026-07-28. Done when non-finite posteriors raise typed failures and never substitute simulator truth. Evidence: injected-failure tests.
- [ ] **M4.10 Implement BKT and last-observation baselines.** Target: 2026-07-29. Done when both return `TrackerEstimates` and run against shared fixtures. Evidence: baseline comparison tests.
- [ ] **M4.11 Implement labeled oracle upper-bound tracker.** Target: 2026-07-29. Done when it is available only in synthetic upper-bound manifests and blocked from primary/production modes. Evidence: authorization tests.
- [ ] **M4.12 Validate calibration and posterior contraction.** Target: 2026-07-30. Done when known synthetic sequences report Brier/NLL and contract toward truth under informative evidence. Evidence: tracker validation report.
- [ ] **M4.13 Implement optional particle snapshot artifacts.** Target: 2026-07-30. Done when selected snapshots are restricted, checksummed, and absent from graph checkpoints. Evidence: artifact tests.
- [ ] **M4.GATE Complete the tracker gate.** Target: 2026-07-31. Done when numerical, masking, resampling, ordering, baseline, misspecification, and calibration tests pass. Evidence: tracker CI and validation artifact.

---

## M5 - Tutor Generation and Guardrails

**Objective:** Produce controlled Socratic prompts through provider-neutral generation and bounded leakage defenses.

**Target:** 2026-08-03  
**Dependencies:** M1; integrates with M2 and M7  
**Primary requirements:** FR-GEN-001 through FR-GEN-003, FR-GRD-001 through FR-GRD-006

- [ ] **M5.1 Implement versioned prompt and template registry.** Target: 2026-07-25. Done when each role prompt, directive template, hash, model allowance, and output schema resolves by version. Evidence: prompt registry tests.
- [ ] **M5.2 Implement `ModelGateway` and deterministic fake.** Target: 2026-07-26. Done when typed requests return content, usage, latency, model identity, attempts, and validation status. Evidence: gateway contract tests.
- [ ] **M5.3 Implement primary LLM provider adapter.** Target: 2026-07-27. Done when one approved OpenRouter or Cerebras model passes structured-output, timeout, usage, and retry tests. Evidence: recorded integration run.
- [ ] **M5.4 Implement concurrency, timeout, retry, and budget middleware.** Target: 2026-07-28. Done when 429/5xx/timeout simulations obey bounded backoff and hard budgets. Evidence: fault-injection tests.
- [ ] **M5.5 Implement experiment-only deterministic response caching.** Target: 2026-07-28. Done when cache keys include model, prompt, parameters, and schema and caching is disabled for inappropriate live calls. Evidence: cache-key tests.
- [ ] **M5.6 Implement tutor candidate generation.** Target: 2026-07-29. Done when `PolicyDecision` and safe context produce a schema-valid candidate without policy re-selection. Evidence: directive coverage tests.
- [ ] **M5.7 Implement rule-based leakage detection.** Target: 2026-07-29. Done when direct answers, complete solution code, and known forbidden patterns are detected on a versioned fixture set. Evidence: precision/recall report.
- [ ] **M5.8 Implement structured classifier guardrail.** Target: 2026-07-30. Done when accepted/rejected results, reason codes, confidence, and rewrite briefs validate; malformed output rejects safely. Evidence: classifier contract tests.
- [ ] **M5.9 Implement bounded rewrite and authored fallback.** Target: 2026-07-31. Done when retries stop at the manifest limit and always yield a safe template or safety abort. Evidence: routing tests.
- [ ] **M5.10 Create adversarial guardrail fixtures.** Target: 2026-08-01. Done when normal, adversarial, persistent-confusion, indirect-leak, and false-positive cases are versioned. Evidence: fixture catalog.
- [ ] **M5.11 Implement template, small-model, and stronger-model variants.** Target: 2026-08-02. Done when all generators share the same policy decision and trajectory contract. Evidence: variant parity test.
- [ ] **M5.GATE Complete the generation/guardrail gate.** Target: 2026-08-03. Done when provider faults, schema failures, bounded rewrites, fallback, fixture metrics, and cost capture pass. Evidence: guardrail readiness report.

---

## M6 - Student Simulator and Diagnostic Scratchpad

**Objective:** Produce state-consistent simulated behavior and structured private performance evidence without storing hidden reasoning.

**Target:** 2026-08-06  
**Dependencies:** M1, M2, M3; M5 gateway  
**Primary requirements:** FR-STU-001 through FR-STU-007, SEC-SBX-001 through SEC-SBX-005

- [ ] **M6.1 Implement bounded student context builder.** Target: 2026-07-28. Done when prompts include only allowed task, profile, misconception, and evidence context and exclude tutor solutions/private serialization. Evidence: context inspection tests.
- [ ] **M6.2 Implement mastery, misconception, and overload response policies.** Target: 2026-07-29. Done when controlled fixtures deterministically activate each branch. Evidence: branch coverage tests.
- [ ] **M6.3 Build versioned diagnostic probe catalog.** Target: 2026-07-30. Done when each probe has target concepts, misconceptions, expected result, test bundle, difficulty, and leakage metadata. Evidence: probe catalog validation.
- [ ] **M6.4 Implement probe selection and generation.** Target: 2026-07-31. Done when active misconceptions/targets select a held-out compatible probe without leaking the main answer. Evidence: selection tests.
- [ ] **M6.5 Implement `SandboxGateway` and trusted fake.** Target: 2026-07-31. Done when requests include runtime and resource limits and responses normalize execution state. Evidence: gateway contract tests.
- [ ] **M6.6 Integrate one external microVM sandbox.** Target: 2026-08-01. Done when E2B or Modal executes Python with no network, bounded CPU/memory/time/output, and typed failures. Evidence: sandbox integration report.
- [ ] **M6.7 Implement execution telemetry normalization.** Target: 2026-08-01. Done when tests, exit status, bounded stdout/stderr, timeouts, and policy violations map to one schema. Evidence: provider fixture tests.
- [ ] **M6.8 Implement public/probe response selection.** Target: 2026-08-02. Done when overload takes precedence and probe pass/fail controls the public response branch as specified. Evidence: sequence tests.
- [ ] **M6.9 Implement load-proxy extraction.** Target: 2026-08-02. Done when clarification, confusion, probe failure, edit distance, latency, and optional expert labels remain observable and versioned. Evidence: proxy tests.
- [ ] **M6.10 Implement structured evidence builder and active masks.** Target: 2026-08-03. Done when text, execution, probe, guardrail, load, probabilities, and masks validate. Evidence: evidence contract tests.
- [ ] **M6.11 Implement sycophancy/disagreement metric.** Target: 2026-08-04. Done when public mastery without supporting probe performance is identified with a versioned classifier/rule. Evidence: metric fixtures.
- [ ] **M6.12 Implement scratchpad ablation.** Target: 2026-08-04. Done when dialogue-only inference preserves all unrelated manifest settings and matched seeds. Evidence: ablation test.
- [ ] **M6.13 Enforce no-hidden-reasoning persistence.** Target: 2026-08-05. Done when schemas, logs, traces, and UI projections reject raw reasoning fields and retain only code/evidence summaries. Evidence: content and schema scans.
- [ ] **M6.GATE Complete the student/scratchpad gate.** Target: 2026-08-06. Done when state branches, probe selection, sandbox limits, evidence, ablation, and no-hidden-reasoning tests pass. Evidence: student simulator readiness report.

---

## M7 - LangGraph Orchestration

**Objective:** Integrate all domain components into a resumable, single-writer, bounded graph runtime.

**Target:** 2026-08-09  
**Dependencies:** M1-M6  
**Primary requirements:** FR-GRF-001 through FR-GRF-007

- [ ] **M7.1 Implement `TutorGraphState` and ownership allowlists.** Target: 2026-07-31. Done when every field has one writer and unauthorized deltas fail before persistence. Evidence: ownership tests.
- [ ] **M7.2 Implement session initialization and tracker-update nodes.** Target: 2026-08-01. Done when frozen policy/task/profile, prior, budgets, and next-turn belief update are correct. Evidence: node tests.
- [ ] **M7.3 Implement policy-selection and tutor-generation nodes.** Target: 2026-08-02. Done when the policy receives only `PolicyObservation` and the tutor receives a validated decision. Evidence: graph spy tests.
- [ ] **M7.4 Implement guardrail, rewrite, and fallback routing.** Target: 2026-08-03. Done when accepted, bounded rewrite, malformed, exhausted, and safety-abort paths are deterministic. Evidence: route coverage.
- [ ] **M7.5 Implement intervention, student, and evidence nodes.** Target: 2026-08-04. Done when prompt features derive from the accepted prompt and evidence follows fast projection. Evidence: turn-order integration test.
- [ ] **M7.6 Implement persist-turn and termination nodes.** Target: 2026-08-04. Done when trajectory receipt precedes committed event and all terminal/truncation routes finalize once. Evidence: commit tests.
- [ ] **M7.7 Implement graph builder and mode-specific routing.** Target: 2026-08-05. Done when deterministic, LLM simulation, Glass Box, and disabled human configurations compile with only allowed nodes. Evidence: graph topology snapshots.
- [ ] **M7.8 Integrate SQLite and PostgreSQL checkpointers.** Target: 2026-08-05. Done when local and deployed checkpointers persist/reload policy-safe state without pickle fallback. Evidence: checkpointer integration tests.
- [ ] **M7.9 Implement stable side-effect idempotency keys.** Target: 2026-08-06. Done when transport retries reuse keys and pedagogical rewrites create new logical attempts. Evidence: duplicate-call tests.
- [ ] **M7.10 Implement dynamic human interrupt behind a disabled feature gate.** Target: 2026-08-06. Done when a synthetic test pauses/resumes the same thread and no non-idempotent operation precedes interrupt. Evidence: interrupt integration test.
- [ ] **M7.11 Implement checkpoint recovery at every node boundary.** Target: 2026-08-07. Done when injected worker termination resumes without duplicate turn commits. Evidence: recovery matrix.
- [ ] **M7.12 Implement graph event streaming and committed cursors.** Target: 2026-08-07. Done when advisory generation and committed events are distinct and reconnect can recover from checkpoint state. Evidence: event tests.
- [ ] **M7.13 Create end-to-end golden graph trajectories.** Target: 2026-08-08. Done when deterministic normal, rewrite, sandbox-failure, overload, and terminal episodes match fixtures. Evidence: graph golden tests.
- [ ] **M7.GATE Complete the LangGraph gate.** Target: 2026-08-09. Done when ownership, routing, bounds, checkpoint, idempotency, recovery, and golden episode tests pass. Evidence: graph readiness CI report.

---

## M8 - Policy Baselines

**Objective:** Implement fair, interpretable policy classes over one observation/action interface.

**Target:** 2026-08-12  
**Dependencies:** M1, M2, M4, M7  
**Primary requirements:** FR-POL-001 through FR-POL-009

- [ ] **M8.1 Implement `TutorPolicy` and policy metadata protocols.** Target: 2026-08-03. Done when all policies return validated decisions and declare determinism, versions, features, and action mappings. Evidence: protocol contract tests.
- [ ] **M8.2 Implement static Socratic policy.** Target: 2026-08-04. Done when each target concept maps to a fixed versioned directive/template choice. Evidence: static policy fixtures.
- [ ] **M8.3 Implement strong expert heuristic policy.** Target: 2026-08-05. Done when repeated failures, execution errors, misconceptions, bandwidth, friction, and guardrail warnings drive documented ordered rules. Evidence: decision-table tests.
- [ ] **M8.4 Implement BKT-plus-heuristic condition.** Target: 2026-08-05. Done when BKT summaries feed the same heuristic policy without graph/environment changes. Evidence: condition parity test.
- [ ] **M8.5 Implement `StateDiscretizerV1`.** Target: 2026-08-06. Done when all bins, boundaries, sentinels, and metadata match the LLD and are shared by bandit/Q-learning. Evidence: boundary snapshots.
- [ ] **M8.6 Implement contextual bandit.** Target: 2026-08-07. Done when immediate reward updates, exploration, chosen-action propensities, and unseen-context fallback are reproducible. Evidence: bandit convergence toy test.
- [ ] **M8.7 Implement tabular Q-learning.** Target: 2026-08-08. Done when terminal/truncation targets, epsilon schedule, updates, and unseen-state fallback match the manifest. Evidence: Q-update and convergence tests.
- [ ] **M8.8 Implement policy artifact serialization.** Target: 2026-08-09. Done when tables, discretizer, action map, hyperparameters, metadata, and checksum round-trip safely without arbitrary code execution. Evidence: artifact round-trip tests.
- [ ] **M8.9 Implement Q-table coverage and visitation reporting.** Target: 2026-08-09. Done when visited states/actions, support, unseen rate, and action frequencies are emitted per run. Evidence: coverage test report.
- [ ] **M8.10 Implement policy feature parity audit.** Target: 2026-08-10. Done when comparison policies receive identical allowed observation fields and any privileged field blocks execution. Evidence: parity audit.
- [ ] **M8.11 Run matched deterministic baseline smoke study.** Target: 2026-08-11. Done when all five policy conditions complete matched tasks/profiles/seeds and produce comparable trajectories. Evidence: smoke-study artifact.
- [ ] **M8.GATE Complete the policy-baseline gate.** Target: 2026-08-12. Done when all policies pass protocol, parity, artifact, update, fallback, propensity, and coverage tests. Evidence: baseline readiness report.

---

## M9 - RL Rollout and Training Infrastructure

**Objective:** Deliver resumable batch rollouts, immutable trajectories, policy training, registration, and explicit promotion.

**Target:** 2026-08-17  
**Dependencies:** M7, M8; data design from approved documentation  
**Primary requirements:** FR-TRJ-001 through FR-TRJ-003, FR-RL-001 through FR-RL-005

- [ ] **M9.1 Implement PostgreSQL application/catalog migrations.** Target: 2026-08-05. Done when clean upgrade creates required schemas/tables/indexes without modifying LangGraph/MLflow internals. Evidence: migration CI test.
- [ ] **M9.2 Implement artifact-store prepare/publish protocol.** Target: 2026-08-06. Done when temporary upload, checksum, final publish, metadata commit, and orphan cleanup are idempotent. Evidence: fault-injection tests.
- [ ] **M9.3 Implement public and privileged Arrow/Parquet schemas.** Target: 2026-08-07. Done when schemas match the data design and reject non-finite, missing, duplicate, or forbidden fields. Evidence: schema registry tests.
- [ ] **M9.4 Implement trajectory builder and writer.** Target: 2026-08-08. Done when shared turn IDs produce separate public/privileged records and conflicting duplicates fail. Evidence: commit and leakage tests.
- [ ] **M9.5 Implement experiment coordinator and shard identities.** Target: 2026-08-09. Done when signed manifests expand deterministically into matched condition/profile/task/seed shards. Evidence: shard snapshot tests.
- [ ] **M9.6 Implement transactional outbox and Redis Streams dispatch.** Target: 2026-08-10. Done when duplicate delivery, consumer failure, reclaim, and acknowledgement preserve one logical shard. Evidence: queue recovery tests.
- [ ] **M9.7 Implement local rollout worker.** Target: 2026-08-10. Done when it constructs environments in process, respects budgets, writes partitions, and runs with the API offline. Evidence: batch integration test.
- [ ] **M9.8 Implement policy-safe trajectory reader.** Target: 2026-08-11. Done when the reader enumerates allowed features and cannot mount/read privileged prefixes. Evidence: IAM/local policy and recursive schema tests.
- [ ] **M9.9 Integrate MLflow run and dataset lineage.** Target: 2026-08-11. Done when runs record manifests, code, input partitions, parameters, metrics, and artifact checksums. Evidence: MLflow smoke run.
- [ ] **M9.10 Implement bandit and Q-learning training entry points.** Target: 2026-08-12. Done when both train from a manifest, resume at durable boundaries, and emit candidate artifacts. Evidence: trainer integration tests.
- [ ] **M9.11 Implement application policy registry and promotion records.** Target: 2026-08-13. Done when candidates, gates, aliases, promotions, rejections, and rollback are immutable/audited. Evidence: registry transaction tests.
- [ ] **M9.12 Implement promotion authorization and mandatory-gate checks.** Target: 2026-08-13. Done when trainers cannot promote and incomplete/failed mandatory gates block operators. Evidence: authorization tests.
- [ ] **M9.13 Implement Modal rollout/evaluation jobs.** Target: 2026-08-14. Done when one multi-shard experiment scales to zero and stores all outputs externally. Evidence: cloud smoke experiment.
- [ ] **M9.14 Implement cost and quota accounting.** Target: 2026-08-15. Done when tokens, requests, sandbox calls, latency, and money aggregate by experiment/condition/model/policy and hard limits stop dispatch. Evidence: budget test report.
- [ ] **M9.15 Run interruption and resume drills.** Target: 2026-08-16. Done when worker, queue, object upload, and trainer interruptions recover without duplicate logical episodes/artifacts. Evidence: recovery report.
- [ ] **M9.GATE Complete the training-infrastructure gate.** Target: 2026-08-17. Done when a clean manifest produces trajectories, a candidate policy, gate placeholders, and a reversible non-production promotion. Evidence: end-to-end training run.

---

## M10 - Evaluation and Statistical Pipeline

**Objective:** Automate preregistered decision gates, ablations, uncertainty estimates, blinded grading, and reproducible reports.

**Target:** 2026-08-22  
**Dependencies:** M3-M9  
**Primary requirements:** FR-EVL-001 through FR-EVL-009

- [ ] **M10.1 Freeze metric registry and source classifications.** Target: 2026-08-10. Done when each metric names formula, unit, direction, public/privileged source, exclusions, and uncertainty method. Evidence: versioned metric registry.
- [ ] **M10.2 Preregister gate thresholds and practical-effect rules.** Target: 2026-08-11. Done when guardrail, scratchpad, CBFM, and policy criteria are signed before held-out access. Evidence: immutable gate-definition artifact.
- [ ] **M10.3 Implement diagnostic gain and efficiency metrics.** Target: 2026-08-12. Done when pre/post held-out probes and code outcomes produce normalized gain without reading reward-state truth. Evidence: metric fixtures.
- [ ] **M10.4 Implement leakage and sycophancy rate estimators.** Target: 2026-08-12. Done when per-turn/dialogue rates and Wilson intervals handle zero counts and clustering. Evidence: statistical tests.
- [ ] **M10.5 Implement tracker calibration metrics.** Target: 2026-08-13. Done when Brier, NLL, calibration curves, uncertainty, and posterior-contraction summaries run on declared sources. Evidence: known-distribution tests.
- [ ] **M10.6 Implement CBFM construct-utility evaluation.** Target: 2026-08-14. Done when `M0`/`M1`/`M2`, bootstrap intervals, monotonicity, recovery, sensitivity, and negative controls yield a gate input. Evidence: CBFM evaluation smoke report.
- [ ] **M10.7 Implement guardrail leakage gate.** Target: 2026-08-14. Done when prompt-only/guarded normal, adversarial, and persistent-confusion conditions report leakage, false positives, latency, and cost. Evidence: gate test.
- [ ] **M10.8 Implement scratchpad falsification gate.** Target: 2026-08-15. Done when dialogue-only/probe-supported conditions report unsupported mastery detection and side effects. Evidence: gate test.
- [ ] **M10.9 Implement policy-complexity gate.** Target: 2026-08-15. Done when heuristic, bandit, and Q-learning are compared on matched held-out gain, quality, leakage, latency, cost, and coverage. Evidence: gate test.
- [ ] **M10.10 Implement full component-ablation matrix.** Target: 2026-08-16. Done when no-CBFM, no-SMC, no-guardrail, no-scratchpad, and no-LLM conditions preserve unrelated settings. Evidence: manifest-diff audit.
- [ ] **M10.11 Implement paired/bootstrap/Wilson statistical utilities.** Target: 2026-08-16. Done when coverage is validated on simulated known distributions and paired IDs are preserved. Evidence: statistical validation suite.
- [ ] **M10.12 Implement blinded evaluator pipeline.** Target: 2026-08-17. Done when condition labels, policy names, and privileged state are removed and structured ratings validate. Evidence: blinding inspection and schema tests.
- [ ] **M10.13 Implement judge disagreement and expert agreement reporting.** Target: 2026-08-18. Done when repeated ratings, missingness, disagreement, and inter-rater statistics are included. Evidence: rating report fixture.
- [ ] **M10.14 Implement accepted/rejected/inconclusive gate engine.** Target: 2026-08-19. Done when missing or conflicting evidence cannot default to acceptance. Evidence: decision-table tests.
- [ ] **M10.15 Implement reproducible report builder.** Target: 2026-08-20. Done when figures/tables reference source hashes, analysis revision, exclusions, sample counts, and gate decisions. Evidence: sample report artifact.
- [ ] **M10.16 Run end-to-end evaluation dry run.** Target: 2026-08-21. Done when a reduced experiment matrix produces all four gates and a versioned report without manual data edits. Evidence: dry-run report.
- [ ] **M10.GATE Complete the evaluation gate.** Target: 2026-08-22. Done when metrics, uncertainty, blinding, ablations, gate decisions, and report lineage pass review. Evidence: evaluation-readiness signoff.

---

## M11 - Glass Box Application

**Objective:** Deliver a usable synthetic research interface that clearly separates true state, belief, action, and evidence.

**Target:** 2026-08-26  
**Dependencies:** M1 API contracts; M7 integrated graph  
**Primary requirements:** FR-UI-001 through FR-UI-006, NFR-ACC-001 through NFR-ACC-003

- [ ] **M11.1 Scaffold React, TypeScript, Vite, and test tooling.** Target: 2026-08-08. Done when the app builds, tests, and serves a functional shell from CI. Evidence: web build job.
- [ ] **M11.2 Implement FastAPI session/experiment/policy endpoints.** Target: 2026-08-10. Done when LLD endpoint families validate commands, return problem details, and expose OpenAPI. Evidence: API contract tests.
- [ ] **M11.3 Implement WebSocket event envelope and reconnect cursor.** Target: 2026-08-11. Done when disconnect/reconnect restores the latest committed turn without duplicate input. Evidence: socket integration test.
- [ ] **M11.4 Implement interaction workspace.** Target: 2026-08-13. Done when chat, task, run controls, and episode status support deterministic and LLM simulation modes. Evidence: component/e2e tests.
- [ ] **M11.5 Integrate Monaco and structured execution results.** Target: 2026-08-15. Done when code submission, test outcomes, bounded output, timeout, and error states render accessibly. Evidence: browser e2e test.
- [ ] **M11.6 Implement labeled true-versus-estimated mastery view.** Target: 2026-08-17. Done when synthetic truth and tracker estimates are visually and textually distinct and truth endpoints require privileged mode. Evidence: screenshot/access tests.
- [ ] **M11.7 Implement `B_t`, `F_t`, and uncertainty views.** Target: 2026-08-18. Done when estimates identify source/uncertainty and true synthetic values are separately labeled. Evidence: visual regression fixtures.
- [ ] **M11.8 Implement prerequisite DAG visualization.** Target: 2026-08-19. Done when concepts, prerequisites, mastery estimates, evidence activity, and non-color encodings render responsively. Evidence: component tests/screenshots.
- [ ] **M11.9 Implement guardrail and scratchpad evidence views.** Target: 2026-08-20. Done when rewrite events, code, tests, and structured probes display without hidden reasoning or hidden answers. Evidence: content security tests.
- [ ] **M11.10 Implement experiment and policy comparison views.** Target: 2026-08-21. Done when users can inspect status, costs, metrics, gates, and artifacts without editing held-out results. Evidence: e2e tests.
- [ ] **M11.11 Implement responsive and off-screen rendering controls.** Target: 2026-08-22. Done when desktop/mobile layouts have no overlap and expensive views pause when hidden. Evidence: Playwright screenshots and performance trace.
- [ ] **M11.12 Complete keyboard/accessibility audit.** Target: 2026-08-23. Done when key workflows are keyboard-usable, focus-visible, labeled, and not color-only. Evidence: automated and manual WCAG-oriented checklist.
- [ ] **M11.13 Run user-facing demo rehearsal.** Target: 2026-08-25. Done when a fresh operator can launch, run, inspect, reconnect, and explain one episode using the runbook. Evidence: rehearsal record.
- [ ] **M11.GATE Complete the Glass Box gate.** Target: 2026-08-26. Done when browser e2e, visual, accessibility, authorization, reconnect, and content-safety checks pass. Evidence: UI readiness report.

---

## M12 - Security and Operational Hardening

**Objective:** Enforce trust boundaries, safe execution, observability, recovery, and cost containment before scaled experiments.

**Target:** 2026-08-28  
**Dependencies:** M7, M9, M11  
**Primary requirements:** SEC-SBX-001 through SEC-SBX-005, NFR-SEC-001 through NFR-COST-002

- [ ] **M12.1 Complete a system threat model.** Target: 2026-08-12. Done when assets, actors, trust boundaries, abuse cases, controls, residual risks, and owners are reviewed. Evidence: `docs/threat-model.md`.
- [ ] **M12.2 Implement mode and role authorization.** Target: 2026-08-14. Done when privileged synthetic, policy promotion, artifact, and human-mode routes deny by default. Evidence: authorization matrix tests.
- [ ] **M12.3 Implement workload-specific secrets and credentials.** Target: 2026-08-15. Done when API, graph, rollout, trainer, and evaluator use least-privilege credentials and no shared owner account. Evidence: deployment/IAM review.
- [ ] **M12.4 Enforce production sandbox policy.** Target: 2026-08-16. Done when host execution is impossible, network is disabled, all resource limits apply, and local fake is development-only. Evidence: sandbox security test report.
- [ ] **M12.5 Implement trace/log redaction.** Target: 2026-08-17. Done when credentials, hidden answers, prohibited fields, and future human identifiers are removed before external export. Evidence: redaction adversarial tests.
- [ ] **M12.6 Implement API quotas, request limits, and budget guards.** Target: 2026-08-18. Done when oversized, excessive, or over-budget operations fail with typed responses before unbounded work. Evidence: rate/budget tests.
- [ ] **M12.7 Add OpenTelemetry correlation.** Target: 2026-08-19. Done when API, graph, provider, sandbox, shard, artifact, and report operations share trace/experiment/episode/turn IDs. Evidence: trace walkthrough.
- [ ] **M12.8 Add Prometheus/Grafana operational dashboards.** Target: 2026-08-20. Done when queue, latency, error, token, cost, sandbox, checkpoint, and artifact metrics are visible. Evidence: dashboard export/screenshots.
- [ ] **M12.9 Integrate LangSmith with sampling and redaction.** Target: 2026-08-20. Done when graph traces support debugging without becoming the scientific system of record. Evidence: sanitized trace review.
- [ ] **M12.10 Integrate Sentry and actionable alerts.** Target: 2026-08-21. Done when representative failures include correlation IDs and alert routing without sensitive payloads. Evidence: test incident.
- [ ] **M12.11 Implement database/object backup and restore.** Target: 2026-08-23. Done when a documented restore recovers sessions, metadata, checksums, and canonical artifacts within target RTO/RPO. Evidence: restore rehearsal.
- [ ] **M12.12 Run service and queue load tests.** Target: 2026-08-24. Done when target concurrency respects p95 orchestration latency, backpressure, provider limits, and bounded memory. Evidence: load report.
- [ ] **M12.13 Run provider/sandbox/database/object-store failure drills.** Target: 2026-08-25. Done when each failure follows the LLD recovery rule and preserves data integrity. Evidence: resilience matrix.
- [ ] **M12.14 Publish incident, rollback, and cost-spike runbooks.** Target: 2026-08-26. Done when an uninvolved operator can follow each procedure in staging. Evidence: runbook rehearsal.
- [ ] **M12.15 Run dependency, secret, and container scans.** Target: 2026-08-27. Done when no unresolved critical finding remains or an explicit risk acceptance exists. Evidence: scan reports.
- [ ] **M12.GATE Complete the security/operations gate.** Target: 2026-08-28. Done when threat-model controls, isolation, redaction, authorization, observability, backups, load, and resilience tests pass. Evidence: readiness signoff.

---

## M13 - Deployment and Release Automation

**Objective:** Deliver repeatable local, staging, batch, and production-like releases with migration and rollback safety.

**Target:** 2026-08-30  
**Dependencies:** M9, M12  
**Primary requirements:** NFR-REL-001 through NFR-MNT-004

- [ ] **M13.1 Build minimal pinned container images.** Target: 2026-08-17. Done when API, graph, worker, and web images run as non-root and pass vulnerability/size checks. Evidence: image build report.
- [ ] **M13.2 Finalize reproducible local Compose environment.** Target: 2026-08-18. Done when a fresh machine starts dependencies, applies migrations, seeds fixtures, and runs a smoke episode. Evidence: clean-machine rehearsal.
- [ ] **M13.3 Implement Terraform for staging data/services.** Target: 2026-08-20. Done when plan creates isolated network, PostgreSQL, Redis, object storage, IAM, secrets references, and observability without manual drift. Evidence: reviewed Terraform plan.
- [ ] **M13.4 Configure Modal batch deployment.** Target: 2026-08-21. Done when images, secrets, concurrency, timeouts, and object-store access are reproducible from versioned code. Evidence: Modal deployment smoke test.
- [ ] **M13.5 Implement migration job and readiness gates.** Target: 2026-08-22. Done when app waits for compatible app/catalog/LangGraph/MLflow schema states and fails safely on mismatch. Evidence: upgrade/rollback tests.
- [ ] **M13.6 Implement CI release pipeline.** Target: 2026-08-23. Done when tagged commits build, scan, test, publish, migrate staging, deploy, and run smoke tests. Evidence: staging release run.
- [ ] **M13.7 Implement configuration promotion and scientific freeze.** Target: 2026-08-24. Done when deployment settings can change independently but signed scientific manifests cannot mutate. Evidence: configuration audit.
- [ ] **M13.8 Implement policy and application rollback.** Target: 2026-08-25. Done when staging restores the previous app image and policy alias without rewriting history. Evidence: rollback drill.
- [ ] **M13.9 Add environment cost alarms and scale-to-zero policy.** Target: 2026-08-26. Done when idle batch compute reaches zero and forecast/actual spend alerts trigger before budget exhaustion. Evidence: billing policy test.
- [ ] **M13.10 Execute staging acceptance suite.** Target: 2026-08-28. Done when one interactive simulation, one batch rollout, one training run, and one evaluation report complete from deployed services. Evidence: staging acceptance report.
- [ ] **M13.GATE Complete the deployment gate.** Target: 2026-08-30. Done when clean deployment, migration, smoke, cost, and rollback rehearsals pass. Evidence: release candidate record.

---

## M14 - Dissertation Experiment Execution

**Objective:** Freeze and execute the complete decision-critical experiment matrix without contaminating held-out evaluation.

**Target:** 2026-09-07  
**Dependencies:** M3, M4, M8-M10, M12-M13  
**Primary requirements:** FR-EVL-001 through FR-EVL-009, PRD decision gates

- [ ] **M14.1 Freeze code, dependencies, prompts, models, catalog, and split manifests.** Target: 2026-08-22. Done when content hashes are signed and held-out membership is access-controlled. Evidence: experiment-freeze manifest.
- [ ] **M14.2 Run reduced production smoke matrix.** Target: 2026-08-23. Done when every condition/gate executes on non-held-out fixtures within projected budget and time. Evidence: smoke report.
- [ ] **M14.3 Fit and freeze CBFM calibration artifacts.** Target: 2026-08-24. Done when calibration uses only calibration episodes and passes required diagnostics or records rejection. Evidence: MLflow calibration run.
- [ ] **M14.4 Tune policies on policy-selection/validation partitions.** Target: 2026-08-26. Done when heuristic constants, bandit, and Q-learning hyperparameters are selected without held-out access. Evidence: tuning report and selected manifests.
- [ ] **M14.5 Execute matched held-out policy episodes.** Target: 2026-08-29. Done when static, heuristic, BKT, bandit, and Q-learning conditions complete required profiles/tasks/repetitions or document exclusions. Evidence: validated trajectory partitions.
- [ ] **M14.6 Execute guardrail gate matrix.** Target: 2026-08-30. Done when prompt-only/guarded normal, adversarial, and persistent-confusion conditions complete. Evidence: gate dataset and result.
- [ ] **M14.7 Execute scratchpad falsification matrix.** Target: 2026-08-31. Done when dialogue-only/probe-supported conditions complete on matched episodes. Evidence: gate dataset and result.
- [ ] **M14.8 Execute CBFM construct-utility and sensitivity matrix.** Target: 2026-09-01. Done when `M0`/`M1`/`M2`, monotonicity, recovery, sensitivity, and negative controls complete. Evidence: gate dataset and result.
- [ ] **M14.9 Execute component and generation ablations.** Target: 2026-09-02. Done when no-SMC, no-CBFM, no-guardrail, no-scratchpad, template, small-model, and stronger-model conditions complete within approved scope. Evidence: ablation partitions.
- [ ] **M14.10 Collect blinded expert/LLM ratings.** Target: 2026-09-03. Done when the target sample has schema-valid ratings, blinding audit, disagreement, and missingness records. Evidence: rating artifact.
- [ ] **M14.11 Run prespecified statistical analysis.** Target: 2026-09-04. Done when primary estimates, effect sizes, intervals, exclusions, subgroup checks, and multiplicity policy are applied without manual spreadsheet edits. Evidence: analysis run and tables.
- [ ] **M14.12 Reproduce a stratified sample independently.** Target: 2026-09-05. Done when reruns from archived manifests match expected deterministic results and stochastic intervals within stated tolerance. Evidence: reproducibility audit.
- [ ] **M14.13 Lock gate decisions and result artifacts.** Target: 2026-09-06. Done when all four gates are accepted/rejected/inconclusive with immutable evidence and no threshold changes. Evidence: signed gate report.
- [ ] **M14.GATE Complete the dissertation experiment gate.** Target: 2026-09-07. Done when primary/ablation runs, statistics, costs, exclusions, lineage, and gate decisions pass final data audit. Evidence: locked experiment release.

---

## M15 - Future Human-Pilot Readiness Decision

**Objective:** Produce a responsible go/defer decision and backlog for future human use without collecting human data in the dissertation simulation phase.

**Target:** 2026-09-09  
**Dependencies:** PRD/HLD/LLD, M11-M14  
**Primary requirements:** FR-GRF-006, NFR-PRV-001, NFR-PRV-002, PRD Release 4

- [ ] **M15.1 Confirm simulation-phase ethics classification with supervisor/department.** Target: 2026-08-20. Done when written confirmation and scope boundaries are archived. Evidence: governance decision record.
- [ ] **M15.2 Draft human-pilot ethics and data-protection requirements.** Target: 2026-08-25. Done when consent, lawful basis, withdrawal, retention, deletion, safeguarding, and evaluator access are documented. Evidence: draft governance package.
- [ ] **M15.3 Produce a human-data flow and DPIA-ready inventory.** Target: 2026-08-27. Done when identity, submissions, tracker estimates, traces, exports, processors, regions, and deletion paths are enumerated. Evidence: data inventory.
- [ ] **M15.4 Define institutional identity and role integration.** Target: 2026-08-29. Done when OIDC claims, roles, session binding, revocation, and least-privilege access are specified without storing credentials. Evidence: identity design addendum.
- [ ] **M15.5 Define human-mode product safeguards.** Target: 2026-08-31. Done when escalation, direct-answer policy, accessibility, transparency, uncertainty language, and educator controls are specified. Evidence: human-mode requirements addendum.
- [ ] **M15.6 Verify human mode remains disabled in dissertation deployments.** Target: 2026-09-01. Done when build/runtime tests prove no ordinary configuration can activate human endpoints. Evidence: feature-gate tests.
- [ ] **M15.7 Define pilot protocol and stopping rules.** Target: 2026-09-03. Done when recruitment, control, pre/post measures, adverse events, sample rationale, and stopping criteria are drafted for review. Evidence: protocol draft.
- [ ] **M15.8 Conduct security/privacy/accessibility readiness review.** Target: 2026-09-05. Done when unresolved blockers and owners are recorded; this review does not authorize deployment. Evidence: readiness findings.
- [ ] **M15.9 Record go/defer decision.** Target: 2026-09-08. Done when the project explicitly defers human collection or records all required approvals and remaining implementation gates. Evidence: signed decision record.
- [ ] **M15.GATE Complete the human-readiness decision gate.** Target: 2026-09-09. Done when governance documents, disabled-mode evidence, risks, and next actions are archived. A defer decision satisfies this gate; live collection does not begin automatically. Evidence: readiness package.

---

## M16 - Final Documentation, Dissertation, and Handover

**Objective:** Deliver a reproducible dissertation artifact, operational handover, and honest statement of results and limitations.

**Target:** 2026-09-15  
**Dependencies:** M14; M11-M13 for demo/handover  
**Primary requirements:** PRD Definition of Done, NFR-REP-001 through NFR-MNT-004

- [ ] **M16.1 Update architecture documents to match the implemented system.** Target: 2026-09-08. Done when PRD/HLD/LLD/data schema record all accepted deviations and current interfaces. Evidence: documentation review diff.
- [ ] **M16.2 Publish API, CLI, configuration, and experiment-reference documentation.** Target: 2026-09-09. Done when a new engineer can start a session, run a batch, train a policy, and build a report. Evidence: clean-reader walkthrough.
- [ ] **M16.3 Publish runbooks and operational limits.** Target: 2026-09-09. Done when deploy, migrate, rollback, restore, provider outage, sandbox incident, cost spike, and data-integrity procedures are complete. Evidence: reviewed runbooks.
- [ ] **M16.4 Publish model, policy, simulator, and data cards.** Target: 2026-09-10. Done when intended use, provenance, metrics, limitations, bias, costs, security, and sim-to-real caveats are documented. Evidence: card documents.
- [ ] **M16.5 Generate dissertation figures and tables from locked artifacts.** Target: 2026-09-10. Done when every figure/table has an analysis command and source hashes. Evidence: generated report directory and lineage manifest.
- [ ] **M16.6 Complete methodology and implementation chapters.** Target: 2026-09-11. Done when text matches implemented timing, state ownership, baselines, gates, and limitations. Evidence: compiled dissertation sections.
- [ ] **M16.7 Complete results and discussion chapters.** Target: 2026-09-12. Done when all gate outcomes, negative results, uncertainty, cost, sensitivity, and sim-to-real limitations are reported. Evidence: compiled dissertation sections.
- [ ] **M16.8 Run clean-environment reproducibility test.** Target: 2026-09-12. Done when a fresh checkout recreates a representative trajectory, policy, metric report, and documentation build from archived inputs. Evidence: reproducibility CI/rehearsal log.
- [ ] **M16.9 Run final test and security suite.** Target: 2026-09-13. Done when contract, mathematical, graph, policy, integration, UI, recovery, D2, secret, dependency, and container checks pass or have documented residual risks. Evidence: release-candidate test report.
- [ ] **M16.10 Rehearse and record the Glass Box demonstration.** Target: 2026-09-13. Done when deterministic and LLM-backed scenarios run within budget and clearly distinguish truth, belief, and evidence. Evidence: demo checklist/recording reference.
- [ ] **M16.11 Archive reproducibility artifacts.** Target: 2026-09-14. Done when manifests, locks, schemas, trajectories, policies, metrics, reports, checksums, licenses, and retention metadata are stored durably. Evidence: archive manifest.
- [ ] **M16.12 Tag and package the final release.** Target: 2026-09-14. Done when the release tag resolves to tested images, migrations, documentation, artifacts, and known issues. Evidence: release record.
- [ ] **M16.13 Complete supervisor/engineering handover.** Target: 2026-09-15. Done when architecture, operation, results, unresolved risks, future-human backlog, and recovery procedures are reviewed. Evidence: signed handover checklist.
- [ ] **M16.14 Submit the dissertation and final project package.** Target: 2026-09-15. Done when submission receipts exist and the archived package checksum matches the submitted release. Evidence: submission receipt and checksum.
- [ ] **M16.GATE Close the project.** Target: 2026-09-15. Done when dissertation, code, documentation, reproducibility archive, demo, handover, and explicit future-work status are complete. Evidence: final closure record.

## 4. Weekly Control Cadence

- [ ] **CTRL.1 Run a Monday critical-path review.** Target: every Monday through 2026-09-14. Done when milestone status, blockers, spend, and seven-day deliverables are recorded. Evidence: weekly status note.
- [ ] **CTRL.2 Run a Wednesday scientific-validity review.** Target: every Wednesday through 2026-09-09. Done when leakage, split integrity, calibration, baselines, and claim language are checked. Evidence: validity checklist.
- [ ] **CTRL.3 Run a Friday integration and budget review.** Target: every Friday through 2026-09-11. Done when main is green, staging smoke passes, spend is reconciled, and recovery status is reviewed. Evidence: CI/staging/cost report.
- [ ] **CTRL.4 Reforecast after any critical-path slip over one day.** Target: within one working day of detection. Done when scope, owner, dates, and dissertation impact are updated without silently reducing evaluation validity. Evidence: rebaseline note.

## 5. Definition of Project Completion

The project is complete only when all of the following are checked:

- [ ] **DONE.1 The simulation system runs reproducibly from signed manifests.** Target: 2026-09-15. Evidence: M14 locked experiment release.
- [ ] **DONE.2 Simulator truth is absent from policy and human-facing inputs.** Target: 2026-09-15. Evidence: import, schema, IAM, and feature-leakage audits.
- [ ] **DONE.3 The four decision-critical gates have immutable outcomes.** Target: 2026-09-15. Evidence: guardrail, scratchpad, CBFM, and policy gate records.
- [ ] **DONE.4 Negative or inconclusive findings are reported without threshold changes.** Target: 2026-09-15. Evidence: preregistration and final report comparison.
- [ ] **DONE.5 Untrusted code executes only in an approved isolated sandbox.** Target: 2026-09-15. Evidence: M12 security report.
- [ ] **DONE.6 The Glass Box accurately distinguishes truth, belief, action, and evidence.** Target: 2026-09-15. Evidence: M11 readiness report.
- [ ] **DONE.7 A clean environment reproduces representative outputs.** Target: 2026-09-15. Evidence: M16 reproducibility test.
- [ ] **DONE.8 The dissertation and archive are submitted.** Target: 2026-09-15. Evidence: M16 submission receipt.
- [ ] **DONE.9 Human deployment is either formally deferred or separately approved and gated.** Target: 2026-09-15. Evidence: M15 decision record.
- [ ] **DONE.10 Final code, documents, artifacts, runbooks, and known risks are handed over.** Target: 2026-09-15. Evidence: M16 closure record.
