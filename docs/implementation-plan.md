# SMART Implementation Plan

## Master's POC: Socratic Tutoring and Evidence-Validity Evaluation

| Field | Value |
|---|---|
| Status | Proposed benchmark-first amendment for supervisor review |
| Revision date | 2026-07-14 |
| Planned submission | 2026-09-15 |
| Primary research path | Context-isolated evidence benchmark with one pinned LLM-student condition |
| Secondary research path | Deterministic Gymnasium simulation, CBFM, tracker baselines, and minimal interpretable RL |
| Required demonstration | Offline FastAPI, React, and LangGraph tutoring turn plus benchmark replay |
| Human pilot | Post-dissertation and ethics-gated |

> The PRD, engineering plan, HLD, LLD, data design, and this checklist now describe the benchmark-first POC. The dissertation proposal has not yet been amended. Supervisor approval of the research-question change is an early governance gate, not an assumption.

## 1. How to Use This Plan

1. Every implementation item starts unchecked and is marked [x] only when its stated evidence exists.
2. CORE means required for the minimum dissertation.
3. RESEARCH means required to support an empirical claim but may yield a negative or inconclusive result.
4. CONDITIONAL means implement only after its written go/no-go gate.
5. PILOT means post-dissertation work and must not begin without approval.
6. A milestone closes only after all CORE and required RESEARCH tasks in that milestone pass.
7. A task is not complete because code exists. Tests, immutable artifacts, review evidence, or a reproducible command must also exist.
8. Dates are latest acceptable dates. Pull work earlier when dependencies permit.
9. Friday reviews remove scope before moving dates.
10. New feature ideas enter the backlog and do not interrupt the critical path.

Each task is SMART:

- Specific: one component, decision, artifact, or validation outcome.
- Measurable: completion and evidence are explicit.
- Achievable: normally one half-day to two days.
- Relevant: traces to the primary claim, required demo, proposal, or reproducibility.
- Time-bound: every task has a date.

## 2. Minimum Successful Dissertation

The dissertation is viable when all of the following exist:

- One offline browser turn executes Python task -> response -> LangGraph -> evidence -> simple tracker -> heuristic policy -> safe Socratic prompt -> JSONL event.
- A frozen, independently reviewed benchmark contains 24 primary evaluation cases across four Python concepts, two misconception states per concept, and three transfer cases per pair.
- Public interaction, evidence probe, and a structurally different held-out criterion probe are separated by schema, context, process phase, and artifact root.
- Dialogue-only, probe-informed, unrelated-probe, and corrupted-probe predictions/actions commit before criterion reveal.
- Deterministic authored fixtures validate measurement direction, controls, missingness, and replay.
- One pinned LLM/provider condition runs three samples for each of the 24 primary cases, subject to a pre-frozen missingness policy.
- The primary paired case-level Brier result is reported as positive, negative, or inconclusive without changing the frozen benchmark.
- A deterministic Gymnasium environment and heuristic baseline reproduce from local artifacts.
- CBFM and BKT receive explicit retain, reject, or inconclusive results; reduced SMC may be omitted.
- One minimal learned policy, preferably a contextual bandit, is compared with the heuristic on matched simulation conditions; Q-learning may be omitted.
- Final statistics reproduce from local Parquet, manifests, recorded responses, and DuckDB without OpenRouter, sandbox, or W&B.
- The live demo and a recorded backup show both the tutoring turn and commit-before-criterion benchmark replay.
- The dissertation states that criterion performance is not latent human mastery, CBFM is not a validated fatigue measure, and synthetic results are not evidence of human learning efficacy.

The project does not need microservices, Kubernetes, distributed RL, neural policies, self-hosted LLMs, online learning, a production database, or a student pilot to succeed.

## 3. Milestones and Hard Dates

- [x] **M0 - Workspace and benchmark-first documentation ready.** Target: 2026-07-15.
- [ ] **M1 - Pure vertical-slice domain complete.** Target: 2026-07-18.
- [ ] **M2 - Offline browser vertical slice complete.** Target: 2026-07-24. **Hard Gate 1.**
- [ ] **M3 - Benchmark infrastructure complete.** Target: 2026-07-29.
- [ ] **M4 - Benchmark authored, reviewed, and frozen.** Target: 2026-08-05. **Hard Gate 2.**
- [ ] **M5 - Deterministic benchmark validation complete.** Target: 2026-08-08.
- [ ] **M6 - Pinned LLM-student primary study complete.** Target: 2026-08-14. **Primary Result Gate.**
- [ ] **M7 - Deterministic simulator and scientific trajectory path complete.** Target: 2026-08-21.
- [ ] **M8 - CBFM and tracker baseline decisions complete.** Target: 2026-08-28.
- [ ] **M9 - Minimal learned-policy comparison complete.** Target: 2026-09-01.
- [ ] **M10 - Tutor rendering and demo hardening complete.** Target: 2026-09-04. **Demo Feature Freeze.**
- [ ] **M11 - Secondary experiments and result freeze complete.** Target: 2026-09-07. **Results Freeze.**
- [ ] **M12 - Dissertation and examiner package complete.** Target: 2026-09-11.
- [ ] **M13 - Final QA, archive, and submission complete.** Target: 2026-09-15.

Critical path:

M0 -> M1 -> M2 -> M3 -> M4 -> M5 -> M6 -> M7 -> M8 -> M9 -> M11 -> M12 -> M13

M10 begins once M6 is stable and can run in parallel with secondary research. Continuous writing and supervisor review run throughout.

## 4. Scope Kill Dates

| Date | Required evidence | If absent |
|---|---|---|
| 2026-07-18 | Pure turn snapshot | Stop benchmark/simulator code and finish domain contracts |
| 2026-07-24 | Offline browser E2E | Freeze all advanced work until the vertical slice passes |
| 2026-07-25 | Named independent benchmark reviewer | Ask supervisor to appoint a reviewer; do not self-approve cases |
| 2026-07-29 | Criterion-isolation and commitment tests | Stop case expansion and repair measurement infrastructure |
| 2026-08-05 | Frozen reviewed 24-case primary matrix | Do not fit CBFM, compare advanced trackers, or train policies |
| 2026-08-08 | Deterministic smoke and negative controls | Do not spend LLM budget; repair the instrument |
| 2026-08-14 | One valid pinned LLM-student primary report | Drop second model, SMC, Q-learning, and tutor LLM comparison |
| 2026-08-21 | Reproducible Gymnasium/Parquet path | Limit secondary claims to benchmark and implementation evidence |
| 2026-08-28 | CBFM/BKT decisions | Freeze available results; SMC is automatically omitted |
| 2026-09-01 | Heuristic versus one learned baseline | Report heuristic-only limitation if bandit remains invalid |
| 2026-09-04 | Stable offline demo and replay | Disable optional network modes in the final demo |
| 2026-09-07 | Frozen metrics and figures | No new experiments; write from validated evidence |
| 2026-09-11 | Complete dissertation draft and demo package | Only critical correctness and submission fixes |
| 2026-09-13 | Submission-ready PDF and archive | Enforce two-day buffer; no feature work |

## 5. Evidence Conventions

Every completed milestone must record:

- Git commit and dirty state.
- Pixi and pnpm lock hashes where applicable.
- Exact command and resolved Hydra configuration.
- Benchmark/task/split/schema/model/provider versions and hashes.
- Test report or immutable run/report manifest.
- Known failures, missingness, exclusions, and deviations.
- Positive, negative, or inconclusive gate decision.
- Reviewer or supervisor note where required.

Generated data belongs under development/artifacts and remains gitignored unless a deliberately small fixture is approved for source control. Authored benchmark definitions, schemas, tests, rubrics, reviews, and split manifests belong under development/data/benchmark.

## M0 - Workspace and Benchmark-First Documentation

**Objective:** Establish one reproducible local workspace and freeze the engineering direction without building enterprise infrastructure.

**Target:** 2026-07-15
**Depends on:** None
**PRD:** NFR-REP-001 through NFR-MNT-003

- [x] **M0.1 [CORE] Revise the PRD around evidence validity.** Target: 2026-07-14. **Done:** the primary question, 16 benchmark requirements, controls, success criteria, and human-claim limits are explicit. **Evidence:** docs/PRD.md.
- [x] **M0.2 [CORE] Revise the engineering plan around benchmark-first delivery.** Target: 2026-07-14. **Done:** tooling, package boundaries, primary data flow, phases, and scope rules are defined. **Evidence:** docs/engineering-plan.md.
- [x] **M0.3 [CORE] Revise and compile the HLD.** Target: 2026-07-14. **Done:** five HLD diagrams define interactive, benchmark, simulator, deployment, and commit/reveal boundaries. **Evidence:** docs/HLD.md and docs-check output.
- [x] **M0.4 [CORE] Revise and compile the LLD.** Target: 2026-07-14. **Done:** public/evaluator contracts, capability gates, class diagrams, and execution sequences are implementable. **Evidence:** docs/LLD.md and docs-check output.
- [x] **M0.5 [CORE] Revise and compile the data design.** Target: 2026-07-14. **Done:** authored, decision, criterion, paired-metric, simulation, and operational schemas are separated. **Evidence:** docs/data-flow-and-schema.md and docs-check output.
- [x] **M0.6 [CORE] Publish this benchmark-first SMART plan.** Target: 2026-07-14. **Done:** dated tasks cover the vertical slice through submission and post-dissertation pilot readiness. **Evidence:** docs/implementation-plan.md.
- [x] **M0.7 [CORE] Create the source/application skeleton.** Target: 2026-07-13. **Done:** package, API, web, config, data, experiment, analysis, and test directories import. **Evidence:** repository tree and import tests.
- [x] **M0.8 [CORE] Configure Pixi and Python 3.12.** Target: 2026-07-13. **Done:** locked environment installs and imports the package. **Evidence:** development/pixi.toml, pixi.lock, and import command.
- [x] **M0.9 [CORE] Configure Python quality tasks.** Target: 2026-07-14. **Done:** Ruff, Pyright, pytest, and Hypothesis run under a failure-propagating check task. **Evidence:** passing check task.
- [x] **M0.10 [CORE] Bootstrap React/Vite/TypeScript with pnpm.** Target: 2026-07-14. **Done:** lockfile, typecheck, tests, and production build exist. **Evidence:** frontend task output.
- [x] **M0.11 [CORE] Bootstrap FastAPI/frontend health integration.** Target: 2026-07-14. **Done:** typed health endpoint and Vite proxy work. **Evidence:** backend/frontend integration tests.
- [x] **M0.12 [CORE] Add CI and documentation diagram checks.** Target: 2026-07-15. **Done:** clean install, lint, type, test, frontend, D2, and secret checks are configured. **Evidence:** workflow and passing CI.
- [x] **M0.13 [CORE] Configure safe local settings.** Target: 2026-07-15. **Done:** template mode and disabled W&B are defaults; secrets are absent. **Evidence:** settings tests and example environment.
- [x] **M0.14 [CORE] Confirm hardware assumptions.** Target: 2026-07-15. **Done:** CPU/local execution is the default; GPU and distributed compute are not dependencies. **Evidence:** engineering plan and configuration defaults.
- [x] **M0.GATE Close workspace gate.** Target: 2026-07-15. **Done:** a clean checkout installs and passes backend/frontend/documentation smoke checks. **Evidence:** CI or clean-environment transcript.

## M1 - Pure Vertical-Slice Domain

**Objective:** Produce one hand-checkable tutoring turn without frameworks or network calls.

**Target:** 2026-07-18
**Depends on:** M0
**PRD:** FR-VS-001 through FR-VS-007

- [x] **M1.1 [CORE] Implement strict shared contracts.** Target: 2026-07-16. **Done:** task, evidence, tracker, policy, guardrail, session, and event models reject undeclared fields and constrain bounded values. **Evidence:** contract and turn tests.
- [x] **M1.2 [CORE] Author the mutable-list-aliasing task fixture.** Target: 2026-07-16. **Done:** the versioned YAML task contains the concept, instructions, starter code, initial prompt, and reviewed evidence signals. **Evidence:** fixture and loader tests.
- [x] **M1.3 [CORE] Implement fixture loading and validation.** Target: 2026-07-16. **Done:** the bundled YAML is validated into a typed task and unknown task IDs fail explicitly. **Evidence:** loader use in the complete test suite.
- [x] **M1.4 [CORE] Implement deterministic evidence extraction.** Target: 2026-07-17. **Done:** correct, misconception, uncertain, conflicting, and empty responses produce authored expected evidence. **Evidence:** five-path fixture suite.
- [x] **M1.5 [CORE] Implement the simple tracker.** Target: 2026-07-17. **Done:** hand-calculated updates match the expected values and repeated updates remain bounded. **Evidence:** tracker tests.
- [x] **M1.6 [CORE] Implement the heuristic policy.** Target: 2026-07-17. **Done:** every evidence category returns the expected transparent tutor action. **Evidence:** full five-path branch table.
- [x] **M1.7 [CORE] Implement deterministic template rendering.** Target: 2026-07-18. **Done:** every tutor action produces a reviewed Socratic prompt. **Evidence:** full turn tests.
- [x] **M1.8 [CORE] Implement the rule-first leakage guardrail.** Target: 2026-07-18. **Done:** direct-answer markers trigger a safe fallback and every standard template passes. **Evidence:** guardrail and full turn tests.
- [x] **M1.9 [CORE] Compose one pure run_turn function.** Target: 2026-07-18. **Done:** extractor, tracker, policy, renderer, and guardrail produce one reproducible result without FastAPI or LangGraph. **Evidence:** deterministic integration tests.
- [x] **M1.10 [CORE] Add policy/private-state import guards.** Target: 2026-07-18. **Done:** policy-safe packages cannot import simulator-private modules. **Evidence:** AST-based CI import test.
- [x] **M1.GATE Close pure-domain gate.** Target: 2026-07-18. **Done:** five response categories produce explainable evidence, state, action, and prompts. **Evidence:** passing focused test report.

## M2 - Offline Browser Vertical Slice

**Objective:** Demonstrate the exact tutoring path end to end with no external credentials.

**Target:** 2026-07-24
**Depends on:** M1
**PRD:** FR-VS-001 through FR-VS-010, FR-GRF-001 through FR-GRF-005, FR-UI-001 through FR-UI-006

- [x] **M2.1 [CORE] Implement canonical JSONL events.** Target: 2026-07-19. **Done:** complete events append with idempotency keys, fsync, and SHA-256 hash chaining. **Evidence:** event-store and recovery tests.
- [x] **M2.2 [CORE] Implement duplicate and partial-line recovery.** Target: 2026-07-19. **Done:** retries return prior state and incomplete trailing writes are removed without losing valid events. **Evidence:** idempotency and partial-write tests.
- [x] **M2.3 [CORE] Define LangGraph state and node deltas.** Target: 2026-07-20. **Done:** typed graph state separates task input, evidence, tracker state, policy decision, candidate prompt, and guardrail result. **Evidence:** strict typecheck and graph tests.
- [x] **M2.4 [CORE] Build the bounded one-turn graph.** Target: 2026-07-20. **Done:** evidence, tracker, policy, rendering, and guardrail nodes execute once on an acyclic path. **Evidence:** graph equivalence and browser tests.
- [x] **M2.5 [CORE] Implement in-memory session service.** Target: 2026-07-20. **Done:** create/get/submit handles sessions and idempotent retry. **Evidence:** service tests.
- [x] **M2.6 [CORE] Implement FastAPI session endpoints.** Target: 2026-07-21. **Done:** typed success, 404, 409, and 422 responses validate. **Evidence:** integration tests.
- [x] **M2.7 [CORE] Implement the single-page React tutor.** Target: 2026-07-22. **Done:** task, code, conversation, response, next prompt, evidence, tracker, and action render accessibly. **Evidence:** component tests and desktop/mobile screenshots.
- [x] **M2.8 [CORE] Preserve client turn IDs across retries.** Target: 2026-07-22. **Done:** retry reuses its key and duplicate submission creates one event and one tracker update. **Evidence:** component, API, and service tests.
- [x] **M2.9 [CORE] Add Playwright offline E2E tests.** Target: 2026-07-23. **Done:** correct, misconception, uncertain, and empty cases complete without credentials on desktop and mobile. **Evidence:** Playwright report.
- [x] **M2.10 [CORE] Recover sessions from JSONL after restart.** Target: 2026-07-23. **Done:** rebuilding the service restores the latest committed session without requiring SQLite. **Evidence:** restart test.
- [x] **M2.11 [CORE] Verify event-log integrity and crash-tail recovery.** Target: 2026-07-23. **Done:** sequence and hash checks reject corruption while incomplete final writes are safely discarded. **Evidence:** event-store tests.
- [x] **M2.12 [CORE] Write a two-minute demo script.** Target: 2026-07-24. **Done:** a reviewer can start, complete, inspect, reset, and explain the turn. **Evidence:** docs/runbooks/demo.md.
- [ ] **M2.13 [CORE] Run one non-research usability rehearsal.** Target: 2026-07-24. **Done:** an observer completes the task without developer tools and critical issues are fixed. **Evidence:** local checklist with no retained participant data.
- [x] **M2.GATE Close technical vertical-slice gate.** Target: 2026-07-24. **Done:** browser E2E completes offline, writes valid events, recovers after restart, and is explainable. **Evidence:** full check output, screenshots, and demo script.

## M3 - Benchmark Infrastructure

**Objective:** Implement the measurement system before authoring or evaluating the full benchmark.

**Target:** 2026-07-29
**Depends on:** M2; C1 supervisor direction before freeze
**PRD:** FR-BMK-001, FR-BMK-003 through FR-BMK-006, FR-BMK-009 through FR-BMK-014

- [ ] **M3.1 [CORE] Add benchmark public and evaluator package roots.** Target: 2026-07-25. **Done:** evaluator-private imports fail in tracker, policy, graph, simulator, API, and decision-runner packages. **Evidence:** import-linter tests.
- [ ] **M3.2 [CORE] Implement authored manifest, case, criterion, review, and split schemas.** Target: 2026-07-25. **Done:** JSON schemas match the data design and reject criterion fields in public case views. **Evidence:** schema snapshots and adversarial fixtures.
- [ ] **M3.3 [CORE] Implement public/evaluator manifest projection.** Target: 2026-07-25. **Done:** the public projection contains no criterion path, ID, content, result, rubric, or label. **Evidence:** recursive scan test.
- [ ] **M3.4 [CORE] Implement canonical hashing and frozen-file inventory.** Target: 2026-07-26. **Done:** changed authored files invalidate manifest/review hashes. **Evidence:** mutation tests.
- [ ] **M3.5 [CORE] Implement benchmark matrix and task-family split validation.** Target: 2026-07-26. **Done:** concept/misconception/transfer counts and forbidden family overlap are mechanically checked. **Evidence:** validator fixtures.
- [ ] **M3.6 [CORE] Implement channel-specific request builders.** Target: 2026-07-26. **Done:** public, evidence, and criterion contexts have distinct field allowlists, cache namespaces, and request hashes. **Evidence:** leakage tests.
- [ ] **M3.7 [CORE] Implement recorded-response gateway.** Target: 2026-07-26. **Done:** request-hash mismatch fails and valid records replay without network. **Evidence:** replay tests.
- [ ] **M3.8 [CORE] Implement frozen negative-control construction.** Target: 2026-07-27. **Done:** unrelated and corrupted evidence never read criterion labels and are stable by manifest/seed. **Evidence:** control fixtures.
- [ ] **M3.9 [CORE] Implement the paired condition runner.** Target: 2026-07-27. **Done:** all four conditions clone one post-public tracker state and use the same tracker/policy versions. **Evidence:** state-hash and parity tests.
- [ ] **M3.10 [CORE] Implement decision-safe prediction records.** Target: 2026-07-27. **Done:** probabilities, uncertainty, action, versions, input/state hashes, and commit time validate without criterion fields. **Evidence:** Arrow/Pydantic tests.
- [ ] **M3.11 [CORE] Implement exactly-once condition commitments.** Target: 2026-07-28. **Done:** duplicate/missing conditions cannot seal and a complete set has exactly four prediction hashes. **Evidence:** commitment-store tests.
- [ ] **M3.12 [CORE] Implement global decision-phase sealing.** Target: 2026-07-28. **Done:** every planned sample is complete, pre-criterion missing, or invalid before atomic commitment_manifest.json publication. **Evidence:** crash/accounting tests.
- [ ] **M3.13 [CORE] Implement evaluator criterion capability.** Target: 2026-07-28. **Done:** criterion reader construction before a valid seal raises and complete sets receive one capability. **Evidence:** runtime-spy tests.
- [ ] **M3.14 [CORE] Implement benchmark scoring.** Target: 2026-07-28. **Done:** hand-calculated Brier, false-mastery, unsafe-advancement, disagreement, and control metrics match. **Evidence:** metric fixtures.
- [ ] **M3.15 [CORE] Implement decision, criterion, score, and paired Parquet writers.** Target: 2026-07-29. **Done:** schemas, roots, timestamps, hashes, and atomic publication match the data design. **Evidence:** dataset tests.
- [ ] **M3.16 [CORE] Add benchmark-validate, benchmark-smoke, benchmark-generate, and benchmark-evaluate commands.** Target: 2026-07-29. **Done:** commands expose clear phase boundaries and nonzero failure exits. **Evidence:** CLI integration tests.
- [ ] **M3.17 [CORE] Recruit the independent benchmark reviewer.** Target: 2026-07-25. **Done:** a named reviewer accepts the case/rubric/probe-separation review role and deadline. **Evidence:** meeting note or written confirmation.
- [ ] **M3.GATE Close benchmark-infrastructure gate.** Target: 2026-07-29. **Done:** a miniature two-case fixture proves public/criterion isolation, commit-before-reveal, controls, scoring, and replay. **Evidence:** complete smoke artifact.

## M4 - Benchmark Authoring, Review, and Freeze

**Objective:** Create the actual measurement instrument before advanced modelling or held-out generation.

**Target:** 2026-08-05
**Depends on:** M3 and supervisor approval of the amended primary question
**PRD:** FR-BMK-002 through FR-BMK-004, FR-BMK-007, FR-BMK-008, FR-BMK-013 through FR-BMK-016

- [ ] **M4.1 [RESEARCH] Freeze four Python concepts and two misconception states per concept.** Target: 2026-07-30. **Done:** concepts are distinct, teachable, executable, and approved by supervisor. **Evidence:** benchmark blueprint.
- [ ] **M4.2 [RESEARCH] Freeze the case allocation and statistical unit.** Target: 2026-07-30. **Done:** 24 held-out primary cases implement 4 concepts x 2 misconceptions x 3 transfer cases, plus at least one distinct task-family fixture for development, calibration, and policy selection; auxiliary fixtures do not count toward primary N. **Evidence:** case matrix.
- [ ] **M4.3 [RESEARCH] Define five evidence-pattern strata.** Target: 2026-07-30. **Done:** supported mastery, false public mastery, honest non-mastery, underconfidence, and ambiguous evidence are represented and reportable. **Evidence:** coverage table.
- [ ] **M4.4 [CORE] Author public interaction fixtures.** Target: 2026-07-31. **Done:** every case has a versioned public task/response fixture with no criterion content. **Evidence:** schema-valid files.
- [ ] **M4.5 [CORE] Author context-isolated evidence probes.** Target: 2026-08-01. **Done:** every probe excludes public dialogue, solution-bearing feedback, criterion content, and hidden state. **Evidence:** context-allowlist validation.
- [ ] **M4.6 [CORE] Author evidence executable tests/rubrics.** Target: 2026-08-01. **Done:** each probe maps output to structured evidence and trusted test hashes validate. **Evidence:** local oracle tests.
- [ ] **M4.7 [CORE] Author structurally different criterion probes.** Target: 2026-08-02. **Done:** each assesses transfer of the same concept without reusing solution, wording, AST, or exact execution trace. **Evidence:** criterion files and difference records.
- [ ] **M4.8 [CORE] Author criterion executable tests and label rationale.** Target: 2026-08-02. **Done:** each demonstrated-performance label follows a deterministic rubric independent of tracker/policy/simulator/reward. **Evidence:** oracle and rationale records.
- [ ] **M4.9 [CORE] Freeze unrelated-probe assignments.** Target: 2026-08-02. **Done:** every case maps to a reviewed different-concept probe with comparable response shape. **Evidence:** control manifest.
- [ ] **M4.10 [CORE] Freeze corruption transformations.** Target: 2026-08-02. **Done:** deterministic corruptions/permutations use no criterion labels and have stable hashes. **Evidence:** control tests.
- [ ] **M4.11 [CORE] Build task-family splits.** Target: 2026-08-03. **Done:** development, calibration, selection, and held-out families share no forbidden template/variant. **Evidence:** overlap report.
- [ ] **M4.12 [RESEARCH] Freeze primary/secondary metric definitions.** Target: 2026-08-03. **Done:** direction, threshold, case aggregation, repeat handling, interval, exclusions, and missingness are fixed before held-out generation. **Evidence:** metric specification.
- [ ] **M4.13 [RESEARCH] Freeze LLM condition and budget.** Target: 2026-08-03. **Done:** one exact model/provider route, three samples per primary case, sampling, retry, token/cost ceilings, and failure policy are recorded. **Evidence:** generation specification.
- [ ] **M4.14 [CORE] Run automated structural-difference warnings.** Target: 2026-08-03. **Done:** lexical/AST/test overlap reports exist for every evidence/criterion pair. **Evidence:** similarity report.
- [ ] **M4.15 [RESEARCH] Complete independent review and adjudication.** Target: 2026-08-04. **Done:** every primary case, tests, labels, misconception mapping, control, and split has signed-off hashes or resolved concerns. **Evidence:** review records.
- [ ] **M4.16 [CORE] Write the benchmark limitations card.** Target: 2026-08-04. **Done:** scope, synthetic nature, concept coverage, label meaning, model dependence, and non-human limitations are explicit. **Evidence:** limitations.md.
- [ ] **M4.17 [CORE] Freeze benchmark v1.** Target: 2026-08-05. **Done:** all authored files, reviews, prompts, controls, splits, schemas, and metrics are hashed; mutation fails validation. **Evidence:** frozen manifest.
- [ ] **M4.GATE Close measurement-freeze gate.** Target: 2026-08-05. **Done:** all 16 FR-BMK requirements pass and independent review is complete. **Evidence:** benchmark-validate report and supervisor sign-off.

## M5 - Deterministic Benchmark Validation

**Objective:** Prove the instrument and data pipeline behave correctly before spending LLM budget.

**Target:** 2026-08-08
**Depends on:** M4
**PRD:** FR-BMK-005, FR-BMK-006, FR-BMK-009 through FR-BMK-012, FR-BMK-014

- [ ] **M5.1 [CORE] Author deterministic student outputs for all channels.** Target: 2026-08-06. **Done:** public, evidence, and criterion outputs cover every evidence pattern and expected label. **Evidence:** versioned fixture records.
- [ ] **M5.2 [CORE] Enforce trusted local execution allowlist.** Target: 2026-08-06. **Done:** only frozen manifest hashes execute locally; modified/generated code is rejected. **Evidence:** allowlist tests.
- [ ] **M5.3 [CORE] Run all four conditions for every primary fixture.** Target: 2026-08-06. **Done:** each eligible sample has one shared state hash and four sealed predictions/actions. **Evidence:** decision Parquet and commit manifest.
- [ ] **M5.4 [CORE] Prove criterion timing.** Target: 2026-08-06. **Done:** criterion loader spy observes zero decision-phase access and every reveal follows both seals. **Evidence:** timing audit.
- [ ] **M5.5 [CORE] Recover expected Brier directions.** Target: 2026-08-07. **Done:** hand-designed false-mastery and supported-mastery fixtures match prespecified condition effects. **Evidence:** case-level metric report.
- [ ] **M5.6 [CORE] Validate unrelated/corrupted negative controls.** Target: 2026-08-07. **Done:** control fixtures recover expected non-specific or degraded effects. **Evidence:** control report.
- [ ] **M5.7 [CORE] Validate missingness semantics.** Target: 2026-08-07. **Done:** missing evidence prevents capability; missing criterion yields a null label; neither becomes incorrect mastery. **Evidence:** failure-injection report.
- [ ] **M5.8 [CORE] Validate common-label scoring.** Target: 2026-08-07. **Done:** all four condition rows join one criterion record and Brier recomputes exactly. **Evidence:** DuckDB parity test.
- [ ] **M5.9 [CORE] Validate nested-repeat aggregation.** Target: 2026-08-07. **Done:** repeated samples do not inflate independent case N. **Evidence:** aggregation fixture.
- [ ] **M5.10 [CORE] Validate recorded-response replay.** Target: 2026-08-08. **Done:** network-disabled rerun reproduces prediction, criterion, and paired-metric hashes. **Evidence:** replay transcript.
- [ ] **M5.11 [CORE] Run corruption, interruption, and resume tests.** Target: 2026-08-08. **Done:** partial files never publish and sealed decisions never rewrite. **Evidence:** fault-injection tests.
- [ ] **M5.12 [RESEARCH] Review the smoke report without changing frozen semantics.** Target: 2026-08-08. **Done:** defects create a new version; disappointing outcomes do not. **Evidence:** dated review note.
- [ ] **M5.GATE Close deterministic-measurement gate.** Target: 2026-08-08. **Done:** benchmark-smoke and offline benchmark-evaluate pass with controls, missingness, and replay. **Evidence:** complete deterministic report.

## M6 - Pinned LLM-Student Primary Study

**Objective:** Test whether the measurement effect exists in an actual pinned LLM-student condition.

**Target:** 2026-08-14
**Depends on:** M5
**PRD:** FR-BMK-005 through FR-BMK-015, FR-EXP-001 through FR-EXP-006, SEC-SBX-001 through SEC-SBX-004

- [ ] **M6.1 [CORE] Implement the typed OpenRouter ModelGateway.** Target: 2026-08-09. **Done:** request/response metadata, exact route, sampling, timeout, retries, usage, cost, and hashes normalise. **Evidence:** fake-transport tests.
- [ ] **M6.2 [CORE] Enforce channel context isolation.** Target: 2026-08-09. **Done:** public, evidence, and criterion requests use fresh messages and fail on prohibited fields. **Evidence:** payload-capture tests.
- [ ] **M6.3 [CORE] Select and configure one external sandbox.** Target: 2026-08-09. **Done:** generated Python runs without outbound network under CPU/memory/time/output limits. **Evidence:** provider smoke and adversarial tests.
- [ ] **M6.4 [CORE] Implement normalised sandbox execution records.** Target: 2026-08-09. **Done:** test counts, exit state, timeout/resource flags, and output hashes validate. **Evidence:** adapter tests.
- [ ] **M6.5 [CORE] Prove no host fallback for generated code.** Target: 2026-08-09. **Done:** sandbox outage records missingness and no local executor is called. **Evidence:** runtime spy.
- [ ] **M6.6 [CORE] Add per-run provider budgets.** Target: 2026-08-10. **Done:** call, token, cost, concurrency, and retry ceilings stop safely and persist status. **Evidence:** budget tests.
- [ ] **M6.7 [RESEARCH] Freeze exact model/provider/prompt configuration.** Target: 2026-08-10. **Done:** route resolution succeeds and hashes match M4 specification. **Evidence:** run manifest.
- [ ] **M6.8 [RESEARCH] Run a non-held-out generation rehearsal.** Target: 2026-08-10. **Done:** fresh contexts, code execution, records, retry policy, and replay work without modifying benchmark semantics. **Evidence:** rehearsal artifact.
- [ ] **M6.9 [RESEARCH] Execute the primary decision phase.** Target: 2026-08-11. **Done:** 24 primary cases x 3 samples are complete or explicitly pre-criterion missing; eligible samples have four sealed condition records. **Evidence:** commitment manifest and accounting report.
- [ ] **M6.10 [RESEARCH] Audit the global decision seal before criterion generation.** Target: 2026-08-11. **Done:** case/sample/model counts, hashes, timestamps, versions, and missingness pass. **Evidence:** signed validation output.
- [ ] **M6.11 [RESEARCH] Execute the criterion phase once.** Target: 2026-08-12. **Done:** only eligible samples receive fresh-context criterion requests and external execution records. **Evidence:** criterion dataset.
- [ ] **M6.12 [CORE] Validate provider route and execution integrity.** Target: 2026-08-12. **Done:** route mismatch, missing labels, test hashes, and reveal timing are reported and invalid rows excluded by frozen rules. **Evidence:** criterion audit.
- [ ] **M6.13 [RESEARCH] Compute the primary paired Brier result.** Target: 2026-08-13. **Done:** repeats reduce within case/model, 24 case units or declared missing cases are reported, and uncertainty follows the frozen method. **Evidence:** primary metric artifact.
- [ ] **M6.14 [RESEARCH] Compute negative controls and secondary outcomes.** Target: 2026-08-13. **Done:** unrelated/corrupted effects, false mastery, precision/recall, unsafe advancement, confidence, disagreement, and denominators appear. **Evidence:** complete report.
- [ ] **M6.15 [CORE] Reproduce the report from recorded responses.** Target: 2026-08-13. **Done:** network-disabled evaluation recreates canonical row/metric hashes. **Evidence:** replay transcript.
- [ ] **M6.16 [RESEARCH] Record positive, negative, or inconclusive interpretation.** Target: 2026-08-14. **Done:** no cases, labels, exclusions, controls, or metrics change in response to the result. **Evidence:** frozen gate report.
- [ ] **M6.17 [CONDITIONAL] Decide on a second model family.** Target: 2026-08-14. **Done:** replication proceeds only if primary report is valid, budget remains, and M7/M12 are on schedule. **Evidence:** decision note; default is defer.
- [ ] **M6.GATE Close primary-result gate.** Target: 2026-08-14. **Done:** one pinned LLM-student report is complete, replayable, honest about missingness, and scoped to demonstrated criterion performance. **Evidence:** frozen primary report manifest.

## M7 - Deterministic Simulator and Scientific Trajectories

**Objective:** Implement the proposal's POMDP-style simulator as a secondary experimental system without contaminating benchmark labels.

**Target:** 2026-08-21
**Depends on:** M6
**PRD:** FR-SIM-001 through FR-SIM-008, FR-TRJ-002 through FR-TRJ-004

- [ ] **M7.1 [CORE] Freeze the action map and policy observation contract.** Target: 2026-08-15. **Done:** five directives, numeric observation fields, and missing-CBFM masks are versioned. **Evidence:** schema snapshot.
- [ ] **M7.2 [CORE] Implement simulator-private state.** Target: 2026-08-15. **Done:** true mastery, misconceptions, bandwidth, and friction remain private. **Evidence:** import/type tests.
- [ ] **M7.3 [CORE] Author minimum task/profile fixtures distinct from benchmark labels.** Target: 2026-08-16. **Done:** at least two tasks and three profiles load without importing benchmark criterion records. **Evidence:** fixtures and dependency test.
- [ ] **M7.4 [CORE] Implement seeded reset and named RNG streams.** Target: 2026-08-16. **Done:** initial state reproduces independently of job order. **Evidence:** golden reset vectors.
- [ ] **M7.5 [CORE] Implement response and slow-learning kernels.** Target: 2026-08-17. **Done:** configured probabilities are bounded and seed-reproducible. **Evidence:** unit/property tests.
- [ ] **M7.6 [CORE] Implement decomposed reward.** Target: 2026-08-17. **Done:** named components sum to total and are not benchmark labels. **Evidence:** hand fixtures.
- [ ] **M7.7 [CORE] Implement terminated/truncated rules.** Target: 2026-08-17. **Done:** domain terminal state and external budget are distinct. **Evidence:** boundary tests.
- [ ] **M7.8 [CORE] Implement reset/step timing.** Target: 2026-08-18. **Done:** prompt effects, response, tracker update, slow learning, reward, and next projection follow the LLD. **Evidence:** order-spy test.
- [ ] **M7.9 [CORE] Register SocraticTutor/POMDP-v0.** Target: 2026-08-18. **Done:** gymnasium.make, check_env, containment, and deterministic replay pass. **Evidence:** CI test.
- [ ] **M7.10 [CORE] Implement Hydra run configuration and stable seed plan.** Target: 2026-08-19. **Done:** resolved config/hash and order-independent episode seeds persist before reset. **Evidence:** config tests.
- [ ] **M7.11 [CORE] Validate simulation task-family splits.** Target: 2026-08-19. **Done:** calibration, selection, validation, and held-out overlap aborts. **Evidence:** split tests.
- [ ] **M7.12 [CORE] Implement separate public/privileged Parquet builders.** Target: 2026-08-20. **Done:** roots, schemas, key parity, and forbidden-prefix checks pass. **Evidence:** Arrow snapshots.
- [ ] **M7.13 [CORE] Implement atomic dataset publication.** Target: 2026-08-20. **Done:** temporary parts, lifecycle, counts, schema fingerprints, and hashes gate completion. **Evidence:** publication tests.
- [ ] **M7.14 [CORE] Produce heuristic smoke trajectories.** Target: 2026-08-21. **Done:** at least 10 matched seeds per minimum condition replay and query through DuckDB. **Evidence:** complete run manifest.
- [ ] **M7.15 [CORE] Prove benchmark/simulator separation.** Target: 2026-08-21. **Done:** simulator truth cannot enter benchmark criterion or policy features. **Evidence:** recursive leakage audit.
- [ ] **M7.GATE Close simulator gate.** Target: 2026-08-21. **Done:** environment, replay, Parquet, DuckDB, timing, and leakage checks pass offline. **Evidence:** frozen smoke artifact.

## M8 - CBFM and Tracker Baselines

**Objective:** Evaluate cognitive-state and tracking complexity only after the primary measurement path is valid.

**Target:** 2026-08-28
**Depends on:** M7
**PRD:** FR-CBFM-001 through FR-CBFM-009, FR-TRK-001 through FR-TRK-009

- [ ] **M8.1 [CORE] Implement versioned prompt-load features.** Target: 2026-08-22. **Done:** token, code, concept, novelty, and directive features are finite and reproducible. **Evidence:** snapshots.
- [ ] **M8.2 [CORE] Fit normalisers on calibration data only.** Target: 2026-08-22. **Done:** held-out and benchmark criterion data are inaccessible. **Evidence:** split/access test.
- [ ] **M8.3 [CORE] Implement ZPD overload and separate underchallenge.** Target: 2026-08-23. **Done:** monotonic tests preserve distinct meanings. **Evidence:** property tests.
- [ ] **M8.4 [CORE] Implement bounded F_t recurrence.** Target: 2026-08-23. **Done:** hand fixtures and finite range tests pass. **Evidence:** unit/property tests.
- [ ] **M8.5 [CORE] Implement bounded B_t drain/recovery recurrence.** Target: 2026-08-23. **Done:** high load drains, low load recovers under declared conditions, and outputs remain bounded. **Evidence:** property tests.
- [ ] **M8.6 [CORE] Implement complete CBFM ablation.** Target: 2026-08-23. **Done:** B=1, F=0, policy estimates removed, and ablation flag emitted. **Evidence:** exact test.
- [ ] **M8.7 [RESEARCH] Build M0/M1/M2 construct datasets.** Target: 2026-08-24. **Done:** mastery-only, prompt-feature, and CBFM-augmented models share authorised splits. **Evidence:** dataset manifest.
- [ ] **M8.8 [RESEARCH] Run CBFM calibration, sensitivity, and negative controls.** Target: 2026-08-25. **Done:** Brier/NLL, calibration, perturbation, shuffled features, and subgroup counts report. **Evidence:** CBFM report.
- [ ] **M8.9 [CORE] Implement BKT through BeliefTracker.** Target: 2026-08-24. **Done:** learn/guess/slip/forget hand calculations and active masks pass. **Evidence:** fixture suite.
- [ ] **M8.10 [RESEARCH] Compare simple tracker and BKT on authorised data.** Target: 2026-08-26. **Done:** calibration, false mastery, uncertainty, and action effects report without retuning benchmark labels. **Evidence:** tracker report.
- [ ] **M8.11 [CORE] Add tracker misspecification controls.** Target: 2026-08-26. **Done:** tracker parameters differ from simulator truth and remain numerically valid. **Evidence:** robustness run.
- [ ] **M8.12 [RESEARCH] Record CBFM retain/reject/inconclusive result.** Target: 2026-08-27. **Done:** M2 is judged against M1 under predeclared criteria. **Evidence:** gate report.
- [ ] **M8.13 [RESEARCH] Record BKT retain/reject/inconclusive result.** Target: 2026-08-27. **Done:** complexity is justified only by calibration or decision benefit. **Evidence:** gate report.
- [ ] **M8.14 [CONDITIONAL] Make reduced-SMC go/no-go decision.** Target: 2026-08-27. **Done:** decision considers BKT gap, state size, numerical risk, thesis value, and remaining time. **Evidence:** docs/decisions/reduced-smc.md.
- [ ] **M8.15 [CONDITIONAL] Implement reduced SMC or record omission.** Target: 2026-08-30. **Done:** retained implementation passes log-weight, ESS, resampling, mask, and summary tests; otherwise omission is explicit. **Evidence:** tests/report or decision.
- [ ] **M8.GATE Close cognitive/tracker gate.** Target: 2026-08-28. **Done:** CBFM and simple/BKT results are frozen; SMC has a documented decision. **Evidence:** combined validation report.

## M9 - Minimal Learned-Policy Comparison

**Objective:** Demonstrate a small auditable RL pipeline without allowing it to dominate the dissertation.

**Target:** 2026-09-01
**Depends on:** M7; M8 tracker choice where used
**PRD:** FR-POL-001 through FR-POL-009

- [ ] **M9.1 [CORE] Freeze policy feature roles.** Target: 2026-08-26. **Done:** all policies receive the same observation/action schema and no criterion/truth/reward target at action time. **Evidence:** parity audit.
- [ ] **M9.2 [CORE] Implement a static reference policy.** Target: 2026-08-26. **Done:** it uses the common interface and logs valid propensities. **Evidence:** contract tests.
- [ ] **M9.3 [CORE] Implement the contextual bandit.** Target: 2026-08-27. **Done:** NumPy value updates, epsilon-greedy action, exact propensity, seeds, and fallback pass. **Evidence:** equation tests.
- [ ] **M9.4 [CORE] Implement local training commands.** Target: 2026-08-27. **Done:** Gymnasium constructs in process with no FastAPI, LangGraph, OpenRouter, criterion, sandbox, or W&B dependency. **Evidence:** import/network-disabled test.
- [ ] **M9.5 [CORE] Implement policy artifacts.** Target: 2026-08-28. **Done:** metadata, feature roles, action map, training hashes, NumPy weights, and fallback reload exactly. **Evidence:** checksum test.
- [ ] **M9.6 [CORE] Implement policy diagnostics.** Target: 2026-08-28. **Done:** action frequency, propensity, return, visitation, coverage, and fallback rate emit locally. **Evidence:** metrics fixtures.
- [ ] **M9.7 [CORE] Add optional W&B sink.** Target: 2026-08-28. **Done:** null/offline/online modes allowlist aggregates and never block local training. **Evidence:** sink tests.
- [ ] **M9.8 [RESEARCH] Run heuristic/static/bandit matched comparison.** Target: 2026-08-30. **Done:** matched tasks/profiles/seeds and paired intervals validate. **Evidence:** comparison report.
- [ ] **M9.9 [RESEARCH] Record bandit retain/reject/inconclusive result.** Target: 2026-08-31. **Done:** performance, safety/load, coverage, and complexity are interpreted together. **Evidence:** gate report.
- [ ] **M9.10 [CONDITIONAL] Make Q-learning go/no-go decision.** Target: 2026-08-31. **Done:** state coverage, bandit gap, implementation cost, and schedule decide. **Evidence:** docs/decisions/q-learning.md.
- [ ] **M9.11 [CONDITIONAL] Implement tabular Q-learning or record omission.** Target: 2026-09-02. **Done:** retained version passes bin/update/coverage/fallback/artifact tests; otherwise omission is complete. **Evidence:** tests/report or decision.
- [ ] **M9.12 [CORE] Freeze one policy artifact per demo session.** Target: 2026-09-01. **Done:** no production/demo online learning occurs. **Evidence:** episode-freeze test.
- [ ] **M9.GATE Close minimal-RL gate.** Target: 2026-09-01. **Done:** heuristic and bandit have one reproducible matched comparison; Q-learning is explicitly decided. **Evidence:** frozen policy report.

## M10 - Tutor Rendering and Demo Hardening

**Objective:** Convert the research system into a reliable examiner-facing demonstration without changing committed decisions.

**Target:** 2026-09-04
**Depends on:** M2 and M6; can overlap M8-M9
**PRD:** FR-GEN-001 through FR-GRD-003, FR-UI-001 through FR-UI-006

- [ ] **M10.1 [CONDITIONAL] Add tutor rendering over the existing ModelGateway.** Target: 2026-08-29. **Done:** a committed directive renders through one pinned route; otherwise templates remain final. **Evidence:** adapter test or omission decision.
- [ ] **M10.2 [CORE] Preserve deterministic template fallback.** Target: 2026-08-29. **Done:** missing credentials, timeout, malformed output, and route mismatch leave the turn usable. **Evidence:** graph failure tests.
- [ ] **M10.3 [CONDITIONAL] Add one bounded guardrail rewrite.** Target: 2026-08-30. **Done:** rejected text never reaches browser and second failure uses template. **Evidence:** routing tests.
- [ ] **M10.4 [RESEARCH] Evaluate template versus LLM rendering if retained.** Target: 2026-08-31. **Done:** leakage, prompt load, latency, tokens, cost, and failures report separately from policy quality. **Evidence:** renderer report.
- [ ] **M10.5 [CORE] Label Glass Box values by source.** Target: 2026-08-31. **Done:** observable evidence, tracker estimate, policy action, criterion performance, and synthetic truth are visually distinct. **Evidence:** UI tests/screenshots.
- [ ] **M10.6 [CORE] Implement benchmark replay view.** Target: 2026-09-01. **Done:** predictions/actions display as committed before criterion reveal and no live policy call occurs after reveal. **Evidence:** browser sequence test.
- [ ] **M10.7 [CORE] Add robust UI retry/reset/error states.** Target: 2026-09-01. **Done:** failures remain understandable and duplicate turns are prevented. **Evidence:** browser tests.
- [ ] **M10.8 [CORE] Test desktop/mobile layout and accessibility.** Target: 2026-09-02. **Done:** controls/text do not overlap and keyboard/focus checks pass. **Evidence:** Playwright screenshots/audit.
- [ ] **M10.9 [CORE] Add a deterministic examiner dataset.** Target: 2026-09-02. **Done:** demo replay requires no provider/sandbox and matches frozen report records. **Evidence:** bundled fixture hashes.
- [ ] **M10.10 [CORE] Rehearse provider and sandbox outage modes.** Target: 2026-09-03. **Done:** live tutoring falls back locally and benchmark replay remains available. **Evidence:** failure checklist.
- [ ] **M10.11 [CORE] Create a short recorded backup demonstration.** Target: 2026-09-03. **Done:** recording shows tutoring turn, committed conditions, criterion reveal, and primary result. **Evidence:** approved archive path.
- [ ] **M10.12 [CORE] Freeze demo features.** Target: 2026-09-04. **Done:** only critical fixes may enter afterward with rerun evidence. **Evidence:** tag and artifact hashes.
- [ ] **M10.GATE Close demo-feature gate.** Target: 2026-09-04. **Done:** offline live demo and replay work three consecutive times. **Evidence:** rehearsal report.

## M11 - Secondary Experiments and Results Freeze

**Objective:** Freeze all defensible primary and secondary results without outcome-driven redesign.

**Target:** 2026-09-07
**Depends on:** M6-M10 as applicable
**PRD:** FR-EXP-001 through FR-EXP-006 and all retained component requirements

- [ ] **M11.1 [CORE] Freeze the final claim/experiment matrix.** Target: 2026-09-02. **Done:** primary benchmark and retained CBFM/tracker/policy/generation conditions are explicit without factorial explosion. **Evidence:** matrix manifest.
- [ ] **M11.2 [CORE] Freeze code, schemas, configs, policies, prompts, and splits.** Target: 2026-09-02. **Done:** Git and artifact hashes precede final runs. **Evidence:** freeze manifest.
- [ ] **M11.3 [CORE] Run one complete dry run.** Target: 2026-09-03. **Done:** every retained command publishes valid local artifacts and reports. **Evidence:** dry-run manifests.
- [ ] **M11.4 [RESEARCH] Execute retained secondary matched runs.** Target: 2026-09-04. **Done:** frozen seed plan completes without held-out tuning. **Evidence:** dataset manifests.
- [ ] **M11.5 [CORE] Validate all final datasets.** Target: 2026-09-05. **Done:** schemas, keys, ranges, phases, splits, leakage, counts, hashes, and denominators pass. **Evidence:** validation report.
- [ ] **M11.6 [RESEARCH] Compute frozen uncertainty and subgroup tables.** Target: 2026-09-05. **Done:** case-level or matched-seed intervals, counts, exclusions, and missingness reproduce. **Evidence:** metric artifacts.
- [ ] **M11.7 [RESEARCH] Finalise CBFM/tracker/policy decisions.** Target: 2026-09-05. **Done:** each is retain, reject, or inconclusive under prespecified criteria. **Evidence:** gate reports.
- [ ] **M11.8 [CORE] Run negative-result audit.** Target: 2026-09-06. **Done:** thresholds, labels, cases, and exclusions did not change to improve outcomes. **Evidence:** manifest diff and decision log.
- [ ] **M11.9 [CORE] Generate final tables/figures from commands.** Target: 2026-09-06. **Done:** outputs contain units, uncertainty, case/seed counts, and source hashes. **Evidence:** report build.
- [ ] **M11.10 [CORE] Freeze claim-to-evidence matrix.** Target: 2026-09-07. **Done:** every claim maps to a result hash or explicit limitation. **Evidence:** traceability table.
- [ ] **M11.11 [CORE] Freeze result artifacts.** Target: 2026-09-07. **Done:** report manifest links benchmark, decisions, criteria, trajectories, policies, configs, and analysis hashes. **Evidence:** final report manifest.
- [ ] **M11.GATE Close results-freeze gate.** Target: 2026-09-07. **Done:** no new experiment or tuning work remains. **Evidence:** committed gate summary.

## M12 - Dissertation and Examiner Package

**Objective:** Produce the written argument, reproducibility package, and rehearsed demonstration.

**Target:** 2026-09-11
**Depends on:** M11; writing tasks begin earlier under continuous controls
**PRD:** Definition of Done

- [ ] **M12.1 [CORE] Reproduce the project in a clean Pixi environment.** Target: 2026-09-08. **Done:** one documented workflow runs tests, demo fixture, benchmark replay, simulator smoke, and representative report. **Evidence:** clean transcript.
- [ ] **M12.2 [CORE] Build the final offline demo bundle.** Target: 2026-09-08. **Done:** FastAPI and React run together with templates, local state, reset, and replay. **Evidence:** versioned artifact.
- [ ] **M12.3 [CORE] Finalise methods and implementation chapters.** Target: 2026-09-08. **Done:** contracts, benchmark timing, equations, splits, model route, simulator, and policy methods match code. **Evidence:** source review.
- [ ] **M12.4 [CORE] Finalise results chapter.** Target: 2026-09-09. **Done:** primary result leads; controls, secondary experiments, uncertainty, counts, and missingness are complete. **Evidence:** claim links.
- [ ] **M12.5 [CORE] Finalise limitations and ethics boundary.** Target: 2026-09-09. **Done:** construct validity, LLM simulation, benchmark coverage, criterion meaning, CBFM limits, and no-human-pilot scope are explicit. **Evidence:** review checklist.
- [ ] **M12.6 [CORE] Complete reproducibility documentation.** Target: 2026-09-09. **Done:** install, demo, benchmark, simulation, training, analysis, data layout, and known failures work from a clean checkout. **Evidence:** README/runbooks.
- [ ] **M12.7 [CORE] Run an examiner-style demo review.** Target: 2026-09-10. **Done:** reviewer can ask why the criterion is hidden, inspect a case, reproduce a result, and observe fallback within allotted time. **Evidence:** notes and fixes.
- [ ] **M12.8 [CORE] Check proposal-to-dissertation deviations.** Target: 2026-09-10. **Done:** benchmark-first amendment, dropped components, and rationale are transparent. **Evidence:** deviation table.
- [ ] **M12.9 [CORE] Audit citations, tables, figures, and terminology.** Target: 2026-09-10. **Done:** no result is described as human mastery/fatigue efficacy and all sources resolve. **Evidence:** proof checklist.
- [ ] **M12.10 [CORE] Freeze examiner package.** Target: 2026-09-11. **Done:** live instructions, backup video, primary figure, architecture figure, limitations, and reproducibility links are bundled. **Evidence:** package manifest.
- [ ] **M12.GATE Close dissertation/evidence gate.** Target: 2026-09-11. **Done:** complete draft, demo, report, and archive are reviewer-ready. **Evidence:** signed checklist.

## M13 - Final QA, Archive, and Submission

**Objective:** Submit a reproducible, honest project with a protected time buffer.

**Target:** 2026-09-15
**Depends on:** M12

- [ ] **M13.1 [CORE] Run the full test suite on the frozen tag.** Target: 2026-09-12. **Done:** backend, frontend, benchmark, sandbox adapter, simulator, leakage, data, browser, and D2 checks pass or have approved noncritical exceptions. **Evidence:** final test report.
- [ ] **M13.2 [CORE] Audit requirement and claim traceability.** Target: 2026-09-12. **Done:** every Must requirement and claim maps to code/test/result or explicit deviation. **Evidence:** final matrix.
- [ ] **M13.3 [CORE] Audit mathematical notation against implementation.** Target: 2026-09-12. **Done:** Brier, tracker, CBFM, reward, timing, and RL notation are consistent. **Evidence:** peer/supervisor notes.
- [ ] **M13.4 [CORE] Audit privacy, secrets, and prohibited data.** Target: 2026-09-12. **Done:** secret scan passes; demo text, hidden reasoning, criterion leakage, simulator leakage, and accidental human data are absent. **Evidence:** data audit.
- [ ] **M13.5 [CORE] Archive frozen scientific artifacts.** Target: 2026-09-12. **Done:** two verified copies contain authored benchmark, recorded responses, manifests, selected data, policies, metrics, figures, and checksums. **Evidence:** archive verification.
- [ ] **M13.6 [CORE] Purge local demo responses from distributable files.** Target: 2026-09-12. **Done:** SQLite/JSONL contain no observer submissions. **Evidence:** archive scan.
- [ ] **M13.7 [CORE] Complete proofreading and reference checks.** Target: 2026-09-13. **Done:** PDF builds, figures/tables resolve, and no planned feature is written as complete. **Evidence:** clean build.
- [ ] **M13.8 [CORE] Perform final clean-machine rehearsal.** Target: 2026-09-13. **Done:** offline demo/replay and backup recording work with network disabled. **Evidence:** rehearsal checklist.
- [ ] **M13.9 [CORE] Preserve the submission buffer.** Target: 2026-09-13. **Done:** only formatting or critical reproducibility fixes enter after this point. **Evidence:** change log.
- [ ] **M13.10 [CORE] Submit dissertation and required artifacts.** Target: 2026-09-15. **Done:** receipts and final hashes are retained. **Evidence:** submission confirmation.
- [ ] **M13.GATE Close the dissertation project.** Target: 2026-09-15. **Done:** submission, demo, archive, and reproduction package are complete; remaining work is post-dissertation. **Evidence:** closure checklist.

## 6. Continuous Research, Writing, and Supervisor Controls

- [ ] **C1 [CORE] Obtain supervisor decision on the benchmark-first amendment.** Target: 2026-07-17. **Done:** primary question, criterion design, minimum case count, and proposal deviation are accepted or revised. **Evidence:** meeting note.
- [ ] **C2 [CORE] Amend the proposal only after C1.** Target: 2026-07-19. **Done:** research questions, method order, primary metric, controls, and timeline match the accepted plan. **Evidence:** proposal diff and supervisor acknowledgement.
- [ ] **C3 [CORE] Freeze the minimum claims.** Target: 2026-07-19. **Done:** supervisor agrees what survives if SMC, Q-learning, second model, or tutor LLM are dropped. **Evidence:** claim hierarchy.
- [ ] **C4 [CORE] Draft vertical-slice methods after M2.** Target: 2026-07-26. **Done:** dissertation describes implemented LangGraph/API/UI behavior. **Evidence:** methods source.
- [ ] **C5 [CORE] Draft benchmark methods before M4 freeze.** Target: 2026-08-03. **Done:** cases, separation, labels, controls, units, metrics, missingness, and model condition are prespecified. **Evidence:** methods/specification source.
- [ ] **C6 [CORE] Draft primary results immediately after M6.** Target: 2026-08-16. **Done:** result and limitations are written before secondary complexity changes interpretation. **Evidence:** results draft.
- [ ] **C7 [CORE] Draft simulator/CBFM methods from tested equations.** Target: 2026-08-25. **Done:** notation, order, constraints, calibration, and falsification match code. **Evidence:** equation-to-test map.
- [ ] **C8 [CORE] Draft policy/evaluation methods before final secondary runs.** Target: 2026-09-02. **Done:** baselines, seeds, splits, outcomes, coverage, and exclusions are frozen. **Evidence:** methods source.
- [ ] **C9 [CORE] Hold a scope review every Friday.** Target: weekly through 2026-09-07. **Done:** variance, next gate, risks, and removed scope are recorded. **Evidence:** dated progress notes.
- [ ] **C10 [CORE] Maintain conditional-component decisions.** Target: continuous. **Done:** second model, SMC, Q-learning, tutor LLM, and extra tasks each have a dated decision. **Evidence:** docs/decisions.
- [ ] **C11 [CORE] Maintain claim-to-evidence traceability.** Target: weekly from 2026-08-14. **Done:** no claim lacks a frozen hash and no synthetic result becomes a human claim. **Evidence:** traceability table.
- [ ] **C12 [CORE] Maintain an experiment cost ledger.** Target: after every network run. **Done:** calls, tokens, sandbox time, failures, and cost reconcile with manifests. **Evidence:** cost report.

## 7. Frozen Experiment Matrices

### 7.1 Required Primary Matrix

| Dimension | Required condition |
|---|---|
| Primary cases | 24 held-out cases: 4 concepts x 2 misconceptions x 3 transfer cases |
| Student generator | One pinned model/provider route |
| Repeats | Three per primary case, nested under case/model |
| Tracker | Simple tracker, frozen before generation |
| Policy | Heuristic policy, frozen before generation |
| Evidence | Dialogue-only, valid isolated probe, unrelated probe, corrupted probe |
| Criterion | One different held-out executable probe per case/sample after commitment |
| Primary endpoint | Paired case-level Brier difference: dialogue minus probe |
| Controls | Unrelated and corrupted effects |
| Secondary outcomes | False mastery, precision/recall, unsafe advancement, confidence, action disagreement |
| Result types | Positive, negative, or inconclusive |

Do not add a second model family until this matrix has a valid report and recorded-response replay.

### 7.2 Minimum Secondary Matrix

| Component | Minimum comparison |
|---|---|
| Simulator | Deterministic template student profiles with matched seeds |
| Tracker | Simple versus BKT |
| CBFM | M0 mastery-only, M1 prompt features, M2 CBFM augmentation; complete ablation |
| Policy | Heuristic versus contextual bandit |
| Generation | Templates are sufficient for final demo |
| Statistics | Matched-seed intervals, coverage, missingness, and subgroup counts |

### 7.3 Expanded Matrix

Add in this order only when earlier gates and writing remain on schedule:

1. Second LLM model family.
2. Reduced SMC.
3. Tabular Q-learning.
4. Tutor template-versus-LLM renderer comparison.
5. Additional benchmark concepts/cases.
6. Additional simulator profiles/tasks.

No expanded component may modify benchmark v1 labels, controls, splits, exclusions, or primary metrics.

## 8. Scope Reduction Order

When behind schedule, remove work in this order:

1. Additional visual polish beyond a readable demo.
2. Second LLM model family.
3. Tutor LLM rendering comparison.
4. Additional simulator tasks/profiles beyond minimum.
5. Reduced SMC.
6. Tabular Q-learning.
7. W&B online synchronisation; retain local artifacts.
8. Extra secondary ablations.
9. Contextual bandit only if implementation remains invalid by its kill date.

Never remove:

- Offline vertical slice.
- Independent benchmark review.
- Public/evidence/criterion isolation.
- Commit-before-reveal.
- The 24-case primary matrix.
- Negative controls.
- One pinned LLM-student condition.
- Case-level paired analysis and honest missingness.
- Recorded-response replay.
- External sandboxing for generated code.
- Local provenance and reproducibility.
- Human-claim limitations.
- Submission buffer.

## 9. Risk Register and Triggered Actions

| Risk | Earliest signal | Immediate action | Escalation |
|---|---|---|---|
| Vertical slice consumes research time | Pure turn not passing by 2026-07-18 | Stop frameworks; fix contracts and one task only | Supervisor scope review |
| Supervisor rejects primary amendment | No agreement by 2026-07-17 | Keep vertical slice; revise benchmark question before authoring | Do not edit proposal silently |
| Case authoring is too slow | Fewer than 12 reviewed primary cases by 2026-08-02 | Stop secondary coding; template authoring/review workflow | Reduce auxiliary fixtures, not primary integrity |
| Independent review unavailable | No reviewer by 2026-07-25 | Ask supervisor for named substitute | Benchmark cannot self-freeze |
| Evidence/criterion leakage | Early loader/schema test succeeds unexpectedly | Invalidate run and repair boundary | New benchmark version if frozen |
| Negative controls improve equally | Deterministic smoke or primary controls mimic valid probe | Treat effect as non-specific; inspect measurement | Do not relabel cases post hoc |
| Held-out primary N falls below plan | Missing samples exceed frozen tolerance | Report denominator and widen uncertainty | Do not count repeats as cases |
| OpenRouter route changes | Returned route differs from manifest | Reject response and stop condition | Pin another route only as new condition/version |
| Sandbox unavailable/unsafe | Generated-code smoke fails | Do not execute locally; pause generated run | Primary run blocked until safe |
| LLM result is null/negative | Delta Brier not positive | Freeze and report honestly | Focus contribution on benchmark/falsification |
| Simulator validates itself | Metrics use transition truth as primary label | Separate as synthetic diagnostic | Benchmark remains primary |
| CBFM adds no value over prompt features | M2 fails M1 gate | Reject as policy input | Preserve negative result |
| SMC numerical/schedule risk | BKT sufficient or tests unstable | Omit SMC | Explain complexity gate |
| RL coverage is poor | Bandit/Q states sparse | Keep heuristic and report limitation | Do not add neural RL |
| W&B/network failure | Sync unavailable | Continue local run | Sync later or omit |
| Demo network failure | Provider unavailable | Use templates and recorded replay | Use backup video |
| Writing falls behind | Methods/results not drafted by dates | Stop optional implementation | Supervisor review |

## 10. Compute and Hardware Plan

- [ ] **H1 [CORE] Measure local benchmark throughput.** Target: 2026-08-10. **Done:** calls/hour, sandbox seconds/case, and estimated full-run duration/cost are recorded. **Evidence:** rehearsal metrics.
- [ ] **H2 [CORE] Measure local simulation throughput.** Target: 2026-08-20. **Done:** episodes/second and memory use justify final seed budget. **Evidence:** benchmark output.
- [ ] **H3 [CONDITIONAL] Use bounded CPU multiprocessing only after profiling.** Target: before final secondary runs. **Done:** deterministic seed/order tests still pass. **Evidence:** parity report.
- [ ] **H4 [CONDITIONAL] Request GPU compute only for a separately approved self-hosted model or neural policy.** Target: before any request. **Done:** CPU/API path is demonstrably insufficient and optional work cannot threaten critical path. **Evidence:** decision note.

Expected hardware:

- Current laptop CPU and memory are sufficient for rule systems, BKT, reduced SMC, contextual bandit, tabular Q-learning, DuckDB, and 24-case analysis.
- LLM inference is API-bound through OpenRouter; local GPU training is not required.
- Generated code executes in an external sandbox, not on local hardware.
- W&B is optional and does not provide compute.
- Do not spend time acquiring a GPU for the required dissertation path.

## 11. Post-Dissertation Human Pilot Backlog

T0 means formal ethics and data-protection approval, not dissertation submission.

- [ ] **P1 [PILOT] Define one narrow human-study question.** Target: before application. **Done:** intervention, comparator, outcome, population, and analysis are explicit. **Evidence:** protocol synopsis.
- [ ] **P2 [PILOT] Confirm institutional ethics/data-protection route.** Target: before application. **Done:** sponsor, controller/processors, review body, and required documents are confirmed. **Evidence:** institutional correspondence.
- [ ] **P3 [PILOT] Select independent learning and workload measures.** Target: before application. **Done:** pre/post assessment and validated instrument are chosen without treating CBFM as truth. **Evidence:** rationale/permissions.
- [ ] **P4 [PILOT] Perform sample-size and recruitment feasibility analysis.** Target: before application. **Done:** pilot purpose, uncertainty, attrition, and stopping rules are justified. **Evidence:** analysis note.
- [ ] **P5 [PILOT] Write recruitment, consent, withdrawal, and adverse-event materials.** Target: before application. **Done:** complete supervisor-reviewed pack exists. **Evidence:** versioned study documents.
- [ ] **P6 [PILOT] Design a separate human-data schema.** Target: before application. **Done:** pseudonyms, identity separation, consent versions, deletion, withdrawal, access, processors, and retention are defined. **Evidence:** data management plan.
- [ ] **P7 [PILOT] Submit and wait for approval.** Target: before participant contact. **Done:** no recruitment or student data occurs while pending. **Evidence:** submission/approval.
- [ ] **P8 [PILOT] Freeze the intervention and comparator.** Target: after approval conditions. **Done:** policy, tasks, prompts, model routes, outcomes, and analysis are preregistered. **Evidence:** study manifest.
- [ ] **P9 [PILOT] Build a separate pilot mode.** Target: after approval. **Done:** no simulator truth, online learning, hidden reasoning, or unapproved processors exist. **Evidence:** pilot security/data audit.
- [ ] **P10 [PILOT] Rehearse consent, accessibility, deletion, failure, and support.** Target: before recruitment. **Done:** dry run uses synthetic identities only. **Evidence:** readiness checklist.
- [ ] **P11 [PILOT] Conduct formal go/no-go review.** Target: before recruitment. **Done:** supervisor confirms approvals, build, instruments, data plan, and support procedures. **Evidence:** signed decision.
- [ ] **P12 [PILOT] Recruit and collect only under the approved protocol.** Target: after P1-P11. **Done:** deviations and withdrawals are handled as approved. **Evidence:** study audit.

## 12. Final Project Completion Checklist

- [ ] All CORE milestone gates M0-M13 are closed or explicitly approved as scoped deviations.
- [ ] Supervisor-approved research question matches the final dissertation.
- [ ] Offline tutoring turn works from a clean checkout.
- [ ] Frozen benchmark has 24 primary cases and independent review.
- [ ] Criterion access occurs only after complete decision accounting.
- [ ] Deterministic controls and replay pass.
- [ ] One pinned LLM-student primary report exists.
- [ ] Primary case-level result includes controls, uncertainty, denominators, exclusions, and missingness.
- [ ] Gymnasium environment and heuristic smoke run reproduce.
- [ ] CBFM and BKT have retain/reject/inconclusive decisions.
- [ ] Minimal learned-policy comparison is complete or transparently scoped out after failure evidence.
- [ ] Final demo works offline and has a backup recording.
- [ ] Final tables/figures rebuild from frozen local artifacts.
- [ ] No W&B export is required for final statistics.
- [ ] No generated code executed on the host.
- [ ] No criterion or simulator truth entered policy inputs.
- [ ] No hidden reasoning or secrets are stored.
- [ ] No human research data was collected under the dissertation POC.
- [ ] Dissertation limitations distinguish probe performance, synthetic state, and human learning.
- [ ] Two verified artifact archives exist.
- [ ] Submission buffer was preserved.
- [ ] Dissertation and required artifacts were submitted with retained receipts.
