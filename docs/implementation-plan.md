# SMART Implementation Plan

## Master's POC: Socratic Tutoring and RL Evaluation

| Field | Value |
|---|---|
| Status | Active critical-path checklist |
| Baseline date | 2026-07-12 |
| Dissertation target | 2026-09-15 |
| Product requirements | `docs/PRD.md` |
| Architecture | `docs/HLD.md`, `docs/LLD.md`, `docs/data-flow-and-schema.md` |
| Delivery principle | Working offline demo first; research complexity only after integration |

## 1. How to Use This Plan

Task status uses standard Markdown checkboxes:

- `[ ]` means incomplete, including blocked or in progress.
- `[x]` means the stated completion condition has been met and its evidence exists.

Rules:

1. Mark a task complete only when its **Done** condition and **Evidence** both exist.
2. Keep task IDs stable. Add a suffixed follow-up rather than renumbering completed work.
3. A milestone gate can close only after all tasks labelled **CORE** in that milestone pass.
4. A **CONDITIONAL** task completes either through implementation evidence or a dated retain/reject decision that removes it from scope.
5. Negative or inconclusive experimental findings count as completion when the experiment was prespecified, executed correctly, and reported honestly.
6. Human-subject recruitment and data collection are outside this dissertation plan and remain blocked pending prior approval.
7. Rebaseline within one working day whenever a critical-path target slips by more than two days.
8. Never trade away split integrity, seed control, data lineage, leakage checks, uncertainty reporting, or the offline fallback to recover schedule.

Priority labels:

- **CORE:** required to graduate with a working, defensible POC.
- **RESEARCH:** needed for the central dissertation comparison after the vertical slice works.
- **CONDITIONAL:** useful only if its entry gate passes and schedule remains healthy.

Every task is SMART: it names one deliverable, a measurable completion condition, evidence, and a date.

## 2. Definition of the Minimum Successful Dissertation

The project is successful without enterprise deployment or a student pilot when all of the following are true:

- A reviewer completes the mutable-list-aliasing task in a browser.
- One submitted explanation passes through FastAPI, LangGraph, deterministic evidence, a simple tracker, a heuristic policy, a safe renderer, and JSONL logging.
- The entire browser path works without OpenRouter, W&B, or sandbox credentials.
- `SocraticTutor/POMDP-v0` passes Gymnasium contract and replay tests.
- CBFM is tested against a prompt-feature-only control and receives a retain/reject result.
- BKT provides a credible tracker baseline; SMC is optional.
- A heuristic and at least one simple learned policy are compared on matched synthetic episodes; Q-learning is conditional on adequate coverage.
- Final tables and figures reproduce from frozen Hydra configs, local Parquet, policy artifacts, and analysis commands.
- The dissertation states that synthetic success is not evidence of human learning efficacy or validated mental-fatigue measurement.

## 3. Milestones and Hard Dates

- [ ] **M0 - POC workspace ready.** Target: 2026-07-15.
- [ ] **M1 - Pure vertical-slice domain complete.** Target: 2026-07-18.
- [ ] **M2 - Offline browser vertical slice complete.** Target: 2026-07-24. **Hard Gate 1.**
- [ ] **M3 - Deterministic simulator and scientific data path complete.** Target: 2026-07-31.
- [ ] **M4 - CBFM and tracker baselines complete.** Target: 2026-08-07.
- [ ] **M5 - Minimal learned-policy comparison complete.** Target: 2026-08-14.
- [ ] **M6 - Optional LLM layer and demo hardening complete.** Target: 2026-08-19.
- [ ] **M7 - Experiment freeze and final runs complete.** Target: 2026-08-28. **Results Freeze.**
- [ ] **M8 - Demo and dissertation evidence package complete.** Target: 2026-09-07. **Demo Freeze.**
- [ ] **M9 - Final QA, archive, and submission complete.** Target: 2026-09-15.

Critical path:

```text
M0 -> M1 -> M2 -> M3 -> M4 -> M5 -> M7 -> M8 -> M9
```

`M6` runs only after the deterministic demo is safe. Reduced SMC, Q-learning, arbitrary code execution, and advanced UI work are conditional branches, never critical-path dependencies.

## 4. Scope Kill Dates

| Date | Trigger | Mandatory response |
|---|---|---|
| 2026-07-20 | Pure components are not integrated through FastAPI | Stop all simulator, CBFM, RL, OpenRouter, and UI-polish work until the slice runs |
| 2026-07-24 | Browser slice is not complete | Reduce UI to one page and use in-memory checkpointing; request supervisor scope review |
| 2026-08-03 | Simulator replay/data publication is unstable | Reduce tasks/profiles and reward terms; preserve deterministic validity |
| 2026-08-08 | BKT baseline is incomplete | Drop SMC from implementation scope |
| 2026-08-15 | Contextual bandit comparison is incomplete | Drop Q-learning and keep heuristic versus bandit/static controls |
| 2026-08-20 | OpenRouter path is unstable | Freeze template-only demo; keep LLM results supplemental |
| 2026-08-20 | Core code/config/splits are not frozen | Cancel optional features and run only minimum experiment matrix |
| 2026-08-28 | Final results are not frozen | Stop feature work and analyze the valid completed subset |
| 2026-09-05 | Demo is unreliable | Use deterministic local build plus recorded backup walkthrough |

## 5. Evidence Conventions

Use these evidence locations unless implementation reveals a better repository-consistent path:

```text
development/tests/                  automated evidence
development/data/                   authored/versioned inputs and schemas
development/artifacts/runs/{run_id}/       experiment evidence
development/artifacts/reports/{report_id}/ tables, figures, and report manifests
docs/decisions/                     retain/reject and scope decisions
docs/runbooks/                      demo and reproduction procedures
```

W&B URLs may supplement evidence but never replace local paths and hashes.

---

## M0 - POC Workspace Ready

**Objective:** Make one clean local workspace capable of running Python, frontend, quality checks, and placeholder API/browser smoke tests.

**Target:** 2026-07-15
**Dependencies:** None
**Requirements:** NFR-REP-001, NFR-MNT-001 through NFR-MNT-004

- [x] **M0.1 [CORE] Revise the PRD for the Master's POC.** Target: 2026-07-12. **Done:** requirements put the one-task offline slice first and defer production/human-study scope. **Evidence:** `docs/PRD.md`.
- [x] **M0.2 [CORE] Revise the engineering plan for the approved POC stack.** Target: 2026-07-12. **Done:** Pixi, React, FastAPI, LangGraph, OpenRouter, Gymnasium, Hydra, W&B, Parquet, and DuckDB responsibilities are documented. **Evidence:** `docs/engineering-plan.md`.
- [x] **M0.3 [CORE] Revise and compile the HLD diagrams.** Target: 2026-07-12. **Done:** four POC D2 diagrams compile and requirement references resolve. **Evidence:** `docs/HLD.md`.
- [x] **M0.4 [CORE] Revise and compile the LLD diagrams.** Target: 2026-07-12. **Done:** five POC D2 diagrams compile and concrete interfaces are defined. **Evidence:** `docs/LLD.md`.
- [x] **M0.5 [CORE] Revise and compile the data-flow/schema design.** Target: 2026-07-12. **Done:** six POC D2 diagrams compile and local schemas are defined. **Evidence:** `docs/data-flow-and-schema.md`.
- [x] **M0.6 [CORE] Publish this revised SMART implementation plan.** Target: 2026-07-12. **Done:** milestones, dates, evidence, kill rules, and pilot boundary cover delivery through submission. **Evidence:** `docs/implementation-plan.md`.
- [x] **M0.7 [CORE] Create the source and application directory skeleton.** Target: 2026-07-13. **Done:** `development/src/socratic_tutor`, `apps/api`, `apps/web`, `configs`, `data`, `experiments`, `analysis`, and test directories exist with import smoke tests. **Evidence:** repository tree and `development/tests/unit/test_imports.py`.
- [x] **M0.8 [CORE] Configure the Pixi workspace and Python 3.12 environment.** Target: 2026-07-13. **Done:** committed `development/pixi.toml` and `development/pixi.lock` create the Python 3.12 environment and import the package. **Evidence:** `cd development && pixi run python -c "import socratic_tutor"`.
- [x] **M0.9 [CORE] Configure Python quality tasks.** Target: 2026-07-14. **Done:** Pixi tasks run Ruff, Pyright, pytest, and Hypothesis under one failure-propagating `check` task. **Evidence:** passing `pixi run check` with 16 Python tests.
- [x] **M0.10 [CORE] Bootstrap React/Vite/TypeScript with pnpm.** Target: 2026-07-14. **Done:** committed `development/pnpm-lock.yaml`; typecheck, two unit tests, and production build pass through Pixi. **Evidence:** frontend task output.
- [x] **M0.11 [CORE] Bootstrap FastAPI and frontend health integration.** Target: 2026-07-14. **Done:** `/api/health` returns a typed response, the React page displays backend status, and the live Vite proxy reaches FastAPI. **Evidence:** `development/tests/integration/test_health.py`, React tests, and localhost smoke response.
- [ ] **M0.12 [CORE] Add the initial CI workflow.** Target: 2026-07-15. **Done:** clean install, lint, type, unit, frontend, D2, and secret-scan jobs run on pushes/PRs. **Evidence:** passing CI run.
- [x] **M0.13 [CORE] Configure safe local settings.** Target: 2026-07-15. **Done:** `development/.env.example` contains no secret; typed settings default to template rendering and disabled W&B, and unsafe model access without a key fails. **Evidence:** `development/tests/unit/test_settings.py` and secret-safe configuration.
- [ ] **M0.GATE Close workspace gate.** Target: 2026-07-15. **Done:** a clean checkout installs and passes backend/frontend smoke tests using one documented Pixi command. **Evidence:** clean-environment transcript or CI artifact.

---

## M1 - Pure Vertical-Slice Domain

**Objective:** Implement and hand-verify the logic for one tutoring turn before introducing orchestration or web integration.

**Target:** 2026-07-18
**Dependencies:** M0
**Requirements:** FR-VS-001, FR-VS-004 through FR-VS-007, FR-TRK-001, FR-POL-001 through FR-POL-004

- [ ] **M1.1 [CORE] Implement strict shared contracts.** Target: 2026-07-16. **Done:** task, evidence, tracker, policy observation/decision, generation, guardrail, session, and event models forbid extra fields and non-finite values. **Evidence:** positive/adversarial contract tests and schema snapshots.
- [ ] **M1.2 [CORE] Author the mutable-list-aliasing task fixture.** Target: 2026-07-16. **Done:** fixture includes version, concept, misconception, rubric, evidence rules, initial prompt, and every directive template. **Evidence:** `data/tasks/python_mutable_aliasing_v1.*` and loader test.
- [ ] **M1.3 [CORE] Implement fixture loading and startup validation.** Target: 2026-07-16. **Done:** duplicate versions, missing templates, and unknown directives fail with typed errors. **Evidence:** fixture validation tests.
- [ ] **M1.4 [CORE] Implement deterministic evidence extraction.** Target: 2026-07-17. **Done:** correct, copy-misconception, uncertain, conflicting, and empty responses produce authored expected evidence. **Evidence:** table-driven fixtures.
- [ ] **M1.5 [CORE] Implement the simple tracker equation.** Target: 2026-07-17. **Done:** hand-calculated mastery, misconception, and uncertainty cases match exactly within declared tolerance and stay bounded. **Evidence:** `tests/unit/tracking/test_simple.py`.
- [ ] **M1.6 [CORE] Implement the heuristic policy.** Target: 2026-07-17. **Done:** low bandwidth, misconception, uncertainty, repeated failure, low mastery, and default branches select the expected versioned directive. **Evidence:** complete rule-table test.
- [ ] **M1.7 [CORE] Implement deterministic template rendering.** Target: 2026-07-18. **Done:** all directives render one safe Socratic question with stable snapshots. **Evidence:** template snapshot suite.
- [ ] **M1.8 [CORE] Implement the rule-first leakage guardrail.** Target: 2026-07-18. **Done:** authored direct-answer/corrected-code fixtures are rejected and safe templates pass. **Evidence:** versioned guardrail fixture metrics.
- [ ] **M1.9 [CORE] Build one pure `run_turn` integration fixture.** Target: 2026-07-18. **Done:** one function composes extractor, tracker, heuristic, renderer, and guardrail without FastAPI/LangGraph/network access. **Evidence:** hand-inspectable integration snapshot.
- [ ] **M1.10 [CORE] Enforce the private-state import boundary.** Target: 2026-07-18. **Done:** policies, graph, generation, API, and tests cannot import simulator-private state. **Evidence:** CI import-boundary test.
- [ ] **M1.GATE Close pure-domain gate.** Target: 2026-07-18. **Done:** all five response categories produce reproducible evidence, state, action, and prompt snapshots that can be explained line by line. **Evidence:** focused test report.

---

## M2 - Offline Browser Vertical Slice

**Objective:** Complete the critical path from one React submission to one durable, inspectable tutoring turn.

**Target:** 2026-07-24
**Dependencies:** M1
**Requirements:** FR-VS-002, FR-VS-003, FR-VS-008 through FR-VS-010, FR-GRF-001 through FR-GRF-005, FR-UI-001, FR-UI-004, FR-UI-005, NFR-REL-001

- [ ] **M2.1 [CORE] Implement `TurnEventV1` canonicalization and JSONL append.** Target: 2026-07-19. **Done:** complete lines append with fsync, unique turn keys, and hash chaining. **Evidence:** event writer tests.
- [ ] **M2.2 [CORE] Implement JSONL duplicate and partial-line recovery.** Target: 2026-07-19. **Done:** duplicate IDs return prior data and a partial final line is quarantined without losing earlier events. **Evidence:** crash/recovery tests.
- [ ] **M2.3 [CORE] Define LangGraph state and eight node deltas.** Target: 2026-07-20. **Done:** nodes validate writes and single-writer tests reject unauthorized fields. **Evidence:** graph state/node tests.
- [ ] **M2.4 [CORE] Build the bounded one-turn LangGraph.** Target: 2026-07-20. **Done:** validation, evidence, tracker, projection, policy, render, guardrail, and commit execute once in the specified order. **Evidence:** graph order-spy integration test.
- [ ] **M2.5 [CORE] Implement in-memory session service first.** Target: 2026-07-20. **Done:** create/get/submit supports one session and idempotent retries without SQLite. **Evidence:** service contract tests.
- [ ] **M2.6 [CORE] Implement FastAPI session endpoints.** Target: 2026-07-21. **Done:** health, create, get, and submit endpoints validate typed success, `404`, `409`, and `422` cases. **Evidence:** FastAPI integration suite.
- [ ] **M2.7 [CORE] Implement the single-page React tutor.** Target: 2026-07-22. **Done:** task, source, conversation, response field, submit state, next prompt, status, evidence, tracker, and directive render accessibly. **Evidence:** component tests and screenshot.
- [ ] **M2.8 [CORE] Preserve client turn IDs across retries.** Target: 2026-07-22. **Done:** double-click/network retry creates one server event and one tracker update. **Evidence:** API/browser retry test.
- [ ] **M2.9 [CORE] Add Playwright offline E2E coverage.** Target: 2026-07-23. **Done:** correct, misconception, uncertain, and empty scenarios complete without external credentials. **Evidence:** browser test report and screenshots.
- [ ] **M2.10 [CORE] Add SQLite checkpointing after in-memory E2E passes.** Target: 2026-07-23. **Done:** backend restart restores the latest committed session and graph state. **Evidence:** restart integration test.
- [ ] **M2.11 [CORE] Test JSONL/SQLite split-commit recovery.** Target: 2026-07-23. **Done:** injected crashes before/after event append recover without duplicate logical turns. **Evidence:** fault-injection suite.
- [ ] **M2.12 [CORE] Write a two-minute deterministic demo script.** Target: 2026-07-24. **Done:** a reviewer can start, complete, inspect, reset, and explain the turn using documented commands. **Evidence:** `docs/runbooks/demo.md`.
- [ ] **M2.13 [CORE] Run an external-observer usability check.** Target: 2026-07-24. **Done:** one non-participant reviewer completes the demo without developer tools; issues are recorded and critical ones fixed. **Evidence:** local demo checklist with no research data retained.
- [ ] **M2.GATE Close vertical-slice gate.** Target: 2026-07-24. **Done:** browser E2E completes the exact critical path offline, writes one valid event, restarts, and is explainable. **Evidence:** tagged test run plus demo screenshot/video.

**Hard rule:** Do not start M3 implementation before `M2.GATE` passes.

---

## M3 - Deterministic Simulator and Scientific Data Path

**Objective:** Build a small, auditable Gymnasium environment and publish reproducible synthetic trajectories locally.

**Target:** 2026-07-31
**Dependencies:** M2
**Requirements:** FR-EXP-001 through FR-EXP-005, FR-SIM-001 through FR-SIM-008, FR-TRJ-002 through FR-TRJ-004

- [ ] **M3.1 [CORE] Freeze the five-action mapping and observation contract.** Target: 2026-07-25. **Done:** semantic directive map, compact numeric observation, missing-CBFM masks, and schema snapshot are versioned. **Evidence:** contract snapshot.
- [ ] **M3.2 [CORE] Implement simulator-private state.** Target: 2026-07-25. **Done:** true mastery, misconception, bandwidth, and friction live only in `simulator/private_state.py`. **Evidence:** type and import-boundary tests.
- [ ] **M3.3 [CORE] Author the minimum task/profile set.** Target: 2026-07-26. **Done:** at least two tasks and three synthetic profiles run from versioned fixtures; expansion to four tasks is conditional on schedule. **Evidence:** fixture manifests and loader tests.
- [ ] **M3.4 [CORE] Implement seeded reset and named random streams.** Target: 2026-07-26. **Done:** identical seeds reproduce initial state independent of job order. **Evidence:** golden reset vectors.
- [ ] **M3.5 [CORE] Implement response and slow-learning kernels.** Target: 2026-07-27. **Done:** probabilities are bounded, configured, deterministic by seed, and separately testable. **Evidence:** kernel unit/property tests.
- [ ] **M3.6 [CORE] Implement decomposed reward and episode bounds.** Target: 2026-07-27. **Done:** named components sum to total and termination differs from truncation. **Evidence:** hand-calculated fixtures.
- [ ] **M3.7 [CORE] Implement `SocraticTutorEnv.reset/step` timing.** Target: 2026-07-28. **Done:** prompt effects, evidence, tracker update, slow learning, reward, and next projection occur in LLD order. **Evidence:** order-spy test.
- [ ] **M3.8 [CORE] Register and validate `SocraticTutor/POMDP-v0`.** Target: 2026-07-28. **Done:** `gymnasium.make`, `check_env`, space containment, and deterministic replay pass. **Evidence:** environment CI job.
- [ ] **M3.9 [CORE] Implement Hydra config and stable seed derivation.** Target: 2026-07-29. **Done:** every run saves resolved config and a reorder-stable episode seed plan before reset. **Evidence:** config/seed tests.
- [ ] **M3.10 [CORE] Implement split manifest validation.** Target: 2026-07-29. **Done:** calibration, selection, validation, and held-out overlap aborts the run. **Evidence:** overlap tests.
- [ ] **M3.11 [CORE] Implement separate public and privileged Parquet builders.** Target: 2026-07-30. **Done:** schemas, roots, readers, forbidden-prefix checks, and one-to-one turn keys match the data design. **Evidence:** Arrow schema snapshots and leakage tests.
- [ ] **M3.12 [CORE] Implement dataset publication and manifest validation.** Target: 2026-07-30. **Done:** temporary parts, unique keys, finite ranges, episode lifecycle, row counts, and hashes gate `complete` status. **Evidence:** publication/failure tests.
- [ ] **M3.13 [CORE] Produce matched heuristic smoke trajectories.** Target: 2026-07-31. **Done:** at least 10 matched seeds per minimum condition replay and query through DuckDB. **Evidence:** complete smoke-run artifact and query output.
- [ ] **M3.14 [CORE] Prove policy-feature separation.** Target: 2026-07-31. **Done:** runtime spy and recursive schema audit show no simulator truth in any action-time observation. **Evidence:** leakage audit report.
- [ ] **M3.GATE Close simulator/data gate.** Target: 2026-07-31. **Done:** environment, replay, timing, splits, Parquet publication, DuckDB query, and leakage tests pass offline. **Evidence:** frozen smoke-run manifest.

---

## M4 - CBFM and Tracker Baselines

**Objective:** Implement the central cognitive construct so it can fail honestly, and establish simple/BKT tracking before considering SMC.

**Target:** 2026-08-07
**Dependencies:** M3
**Requirements:** FR-CBFM-001 through FR-CBFM-009, FR-TRK-001 through FR-TRK-009

- [ ] **M4.1 [CORE] Implement versioned prompt-load features.** Target: 2026-08-01. **Done:** token count, code density, concept count, directive complexity, and declared normalizers produce finite bounded outputs. **Evidence:** prompt fixture snapshots.
- [ ] **M4.2 [CORE] Implement ZPD overload and separate underchallenge.** Target: 2026-08-01. **Done:** higher positive mismatch increases overload while underchallenge remains a separately logged diagnostic. **Evidence:** monotonicity tests.
- [ ] **M4.3 [CORE] Implement bounded `F_t` recurrence.** Target: 2026-08-02. **Done:** configured non-negative coefficients, decay, overload, and load/bandwidth mismatch match hand fixtures and remain in `[0,1]`. **Evidence:** unit/property tests.
- [ ] **M4.4 [CORE] Implement bounded `B_t` drain/recovery recurrence.** Target: 2026-08-02. **Done:** high load drains and low-load sequences recover under declared conditions; values remain finite/bounded. **Evidence:** Hypothesis tests.
- [ ] **M4.5 [CORE] Implement complete CBFM ablation.** Target: 2026-08-02. **Done:** ablation fixes `B=1`, `F=0`, removes policy estimates, and marks the condition. **Evidence:** exact ablation test.
- [ ] **M4.6 [CORE] Implement M0/M1/M2 construct-evaluation datasets.** Target: 2026-08-03. **Done:** mastery-only, prompt-feature-only, and CBFM-augmented predictors share calibration/held-out splits. **Evidence:** dataset/config tests.
- [ ] **M4.7 [CORE] Implement CBFM calibration and sensitivity commands.** Target: 2026-08-04. **Done:** calibration excludes aggregate reward/held-out data and emits Brier, NLL, perturbation, and shuffled-feature controls. **Evidence:** calibration smoke report.
- [ ] **M4.8 [CORE] Implement BKT through `BeliefTracker`.** Target: 2026-08-04. **Done:** learn/guess/slip/forget updates match hand calculations and use active evidence only. **Evidence:** BKT fixture suite.
- [ ] **M4.9 [CORE] Compare simple tracker and BKT.** Target: 2026-08-05. **Done:** matched synthetic sequences report calibration, uncertainty, and decision-relevant error. **Evidence:** tracker baseline report.
- [ ] **M4.10 [CORE] Add tracker misspecification controls.** Target: 2026-08-05. **Done:** tracker parameters differ from simulator truth through config and still produce valid outputs. **Evidence:** misspecification run.
- [ ] **M4.11 [CONDITIONAL] Make the reduced-SMC go/no-go decision.** Target: 2026-08-06. **Done:** decision records expected thesis value, implementation estimate, state dimensionality, BKT gap, and schedule health. **Evidence:** `docs/decisions/reduced-smc.md`.
- [ ] **M4.12 [CONDITIONAL] Implement reduced SMC or record rejection.** Target: 2026-08-09. **Done:** if retained, log-space weights, ESS, seeded resampling, neutral masks, and summary-only outputs pass; otherwise the dated rejection explains why BKT is sufficient. **Evidence:** SMC tests/report or rejection decision.
- [ ] **M4.13 [RESEARCH] Record CBFM interim retain/reject criteria before final runs.** Target: 2026-08-06. **Done:** thresholds and interpretation against M1 prompt features are written before held-out evaluation. **Evidence:** evaluation specification.
- [ ] **M4.GATE Close cognitive/tracker gate.** Target: 2026-08-07. **Done:** CBFM mathematical controls and simple/BKT baselines pass; SMC has a documented conditional decision. **Evidence:** validation report and decision records.

---

## M5 - Minimal Learned-Policy Comparison

**Objective:** Compare the heuristic against one auditable learned policy using direct local Gymnasium rollouts.

**Target:** 2026-08-14
**Dependencies:** M4
**Requirements:** FR-POL-001 through FR-POL-009, FR-EXP-003 through FR-EXP-006, FR-TRJ-005, FR-TRJ-006

- [ ] **M5.1 [CORE] Freeze the shared policy feature-role manifest.** Target: 2026-08-08. **Done:** every policy receives identical observation/action fields and reward/next-state/truth columns are excluded at action time. **Evidence:** parity audit.
- [ ] **M5.2 [CORE] Implement a static reference policy.** Target: 2026-08-08. **Done:** fixed directive distribution runs through the same interface and logs valid propensities. **Evidence:** policy contract test.
- [ ] **M5.3 [CORE] Implement the contextual bandit.** Target: 2026-08-09. **Done:** NumPy linear estimates, epsilon-greedy action selection, exact propensities, seeded replay, and heuristic fallback pass tests. **Evidence:** bandit equation/propensity fixtures.
- [ ] **M5.4 [CORE] Implement direct local training commands.** Target: 2026-08-09. **Done:** Pixi/Hydra command constructs Gymnasium in process and imports no FastAPI, LangGraph, OpenRouter, or sandbox runtime. **Evidence:** import/network-disabled integration test.
- [ ] **M5.5 [CORE] Implement versioned policy artifacts.** Target: 2026-08-10. **Done:** metadata, feature roles, action map, training hashes, NumPy weights, and fallback policy reload exactly. **Evidence:** save/load checksum test.
- [ ] **M5.6 [CORE] Implement policy diagnostics.** Target: 2026-08-10. **Done:** action frequency, propensity, return, state/action visitation, and fallback rate appear in local metrics. **Evidence:** metrics fixture.
- [ ] **M5.7 [CORE] Add an optional W&B sink.** Target: 2026-08-10. **Done:** online/offline/null modes record allowlisted aggregate metrics and local run hashes; W&B outage does not fail training. **Evidence:** null/offline integration tests.
- [ ] **M5.8 [CORE] Run heuristic/static/bandit smoke comparison.** Target: 2026-08-11. **Done:** matched tasks/profiles/seeds complete and paired rows validate. **Evidence:** comparison smoke report.
- [ ] **M5.9 [CONDITIONAL] Make the Q-learning go/no-go decision.** Target: 2026-08-11. **Done:** state-size estimate, bandit gap, coverage projection, implementation cost, and schedule determine retention. **Evidence:** `docs/decisions/q-learning.md`.
- [ ] **M5.10 [CONDITIONAL] Implement tabular Q-learning or record rejection.** Target: 2026-08-13. **Done:** if retained, documented bins, epsilon-greedy updates, visitation, unseen-state fallback, and artifact reload pass; otherwise rejection is evidence-complete. **Evidence:** Q tests/report or rejection decision.
- [ ] **M5.11 [CORE] Freeze demo policy loading.** Target: 2026-08-13. **Done:** a session pins one versioned heuristic or selected learned artifact and never updates online. **Evidence:** episode-freeze test.
- [ ] **M5.12 [RESEARCH] Prespecify policy retain/reject analysis.** Target: 2026-08-14. **Done:** primary outcome, safety/load checks, paired intervals, exclusions, and coverage interpretation are frozen before held-out runs. **Evidence:** evaluation specification.
- [ ] **M5.GATE Close minimal-RL gate.** Target: 2026-08-14. **Done:** heuristic and contextual bandit complete matched reproducible comparisons; Q-learning has a documented decision. **Evidence:** complete smoke dataset and policy artifacts.

---

## M6 - Optional LLM Layer and Demo Hardening

**Objective:** Improve the demonstration without making an external model or code sandbox necessary for success.

**Target:** 2026-08-19
**Dependencies:** M2; may run after M5 core work
**Requirements:** FR-GEN-001 through FR-GRD-003, FR-UI-002 through FR-UI-006, SEC-SBX-001 through SEC-SBX-004

- [ ] **M6.1 [RESEARCH] Implement typed OpenRouter rendering.** Target: 2026-08-15. **Done:** local protocol pins model/provider route, sampling, timeout, token bound, prompt hash, and returned metadata. **Evidence:** adapter contract tests with fake transport.
- [ ] **M6.2 [CORE] Preserve deterministic template fallback.** Target: 2026-08-15. **Done:** timeout, malformed response, route mismatch, or missing credential returns a safe template without losing the turn. **Evidence:** graph failure tests.
- [ ] **M6.3 [RESEARCH] Implement bounded LLM guardrail routing.** Target: 2026-08-16. **Done:** at most one rewrite occurs; second rejection uses a template and rejected content never reaches the browser. **Evidence:** route/fixture tests.
- [ ] **M6.4 [RESEARCH] Build a versioned leakage evaluation set.** Target: 2026-08-16. **Done:** safe questions, subtle hints, direct answers, and corrected code have labels and report precision/recall/false positives. **Evidence:** guardrail report.
- [ ] **M6.5 [RESEARCH] Run template-versus-pinned-renderer comparison.** Target: 2026-08-17. **Done:** selected synthetic cases report prompt features, leakage, latency, token use, cost, and provider metadata. **Evidence:** renderer comparison artifact.
- [ ] **M6.6 [CORE] Label Glass Box values by source.** Target: 2026-08-17. **Done:** observable evidence, tracker estimates, policy action, and synthetic truth are visually distinct; normal demo API has no truth. **Evidence:** UI tests and screenshots.
- [ ] **M6.7 [CORE] Add robust UI error/retry/reset states.** Target: 2026-08-18. **Done:** network failure, conflict, long/empty input, restart, and reset remain understandable and do not duplicate turns. **Evidence:** browser tests.
- [ ] **M6.8 [CORE] Test desktop and mobile demo layouts.** Target: 2026-08-18. **Done:** required controls/text do not overlap at agreed viewports and keyboard/focus behavior works. **Evidence:** Playwright screenshots and accessibility scan.
- [ ] **M6.9 [CONDITIONAL] Decide whether arbitrary code execution adds thesis value.** Target: 2026-08-18. **Done:** decision compares research need, provider setup, safety burden, and schedule. Default decision is omission. **Evidence:** `docs/decisions/sandbox.md`.
- [ ] **M6.10 [CONDITIONAL] Integrate one external sandbox or record omission.** Target: 2026-08-19. **Done:** if retained, no host fallback exists and timeout/resource/network limits return typed evidence; otherwise text/probe demo remains complete. **Evidence:** sandbox tests or omission record.
- [ ] **M6.GATE Close demo-layer gate.** Target: 2026-08-19. **Done:** deterministic demo remains primary and reliable; LLM/sandbox branches have bounded behavior or explicit omission decisions. **Evidence:** no-credential and provider-failure demo runs.

---

## M7 - Experiment Freeze and Final Runs

**Objective:** Freeze the scientific protocol, execute only the defensible matrix, and produce validated result artifacts.

**Target:** 2026-08-28
**Dependencies:** M3, M4, M5; M6 only for renderer/guardrail experiments
**Requirements:** FR-EXP-001 through FR-EXP-006, FR-CBFM-006 through FR-CBFM-008, FR-TRJ-002 through FR-TRJ-006, NFR-REP-002 through NFR-REP-005

- [ ] **M7.1 [CORE] Write the final experiment specification.** Target: 2026-08-18. **Done:** research questions, hypotheses, primary/secondary metrics, conditions, exclusions, intervals, and retain/reject logic are fixed. **Evidence:** versioned experiment specification.
- [ ] **M7.2 [CORE] Freeze task/profile/split manifests.** Target: 2026-08-19. **Done:** disjoint calibration/selection/validation/held-out files have hashes and cannot be mutated by run commands. **Evidence:** frozen split manifest and overlap report.
- [ ] **M7.3 [CORE] Determine the seed budget empirically.** Target: 2026-08-19. **Done:** pilot variance and laptop runtime justify a feasible target, normally 20-30 matched seeds per condition. **Evidence:** seed-budget note.
- [ ] **M7.4 [CORE] Freeze core code, schemas, configs, and policy artifacts.** Target: 2026-08-20. **Done:** Git commit and all hashes are recorded before held-out execution. **Evidence:** freeze manifest/tag.
- [ ] **M7.5 [CORE] Lock the minimum condition matrix.** Target: 2026-08-20. **Done:** at minimum heuristic versus bandit, simple/BKT tracker control, and CBFM off/on are represented without an unmanageable factorial explosion. **Evidence:** matrix file and episode count.
- [ ] **M7.6 [CONDITIONAL] Add SMC and/or Q-learning conditions only after their gates.** Target: 2026-08-20. **Done:** retained components have frozen artifacts and sufficient budget; rejected components are absent with reasons. **Evidence:** matrix/decision consistency check.
- [ ] **M7.7 [CORE] Run a full-matrix dry run.** Target: 2026-08-21. **Done:** one seed per condition publishes complete public/privileged data, metrics, and W&B references if enabled. **Evidence:** dry-run dataset manifest.
- [ ] **M7.8 [CORE] Execute calibration and policy-selection runs.** Target: 2026-08-22. **Done:** selected parameters/artifacts derive only from authorized splits and are frozen afterward. **Evidence:** calibration/selection manifests.
- [ ] **M7.9 [CORE] Execute validation runs and resolve only predefined failures.** Target: 2026-08-23. **Done:** fixes address implementation defects, not outcome-driven threshold changes; any re-freeze is documented. **Evidence:** validation report/change log.
- [ ] **M7.10 [CORE] Execute held-out matched runs once.** Target: 2026-08-25. **Done:** all core conditions complete the frozen seed plan without tuning from held-out outcomes. **Evidence:** complete held-out dataset manifests.
- [ ] **M7.11 [CORE] Validate all final datasets.** Target: 2026-08-25. **Done:** schemas, keys, ranges, lifecycle, public/truth parity, feature leakage, counts, and hashes pass. **Evidence:** dataset validation report.
- [ ] **M7.12 [CORE] Compute paired/stratified uncertainty.** Target: 2026-08-26. **Done:** point estimates, paired bootstrap intervals, exclusions, subgroup counts, and missing pairs are emitted deterministically. **Evidence:** metric artifact.
- [ ] **M7.13 [CORE] Evaluate CBFM construct utility.** Target: 2026-08-26. **Done:** M0/M1/M2 held-out Brier/NLL, calibration, sensitivity, and negative controls produce retain/reject/inconclusive status. **Evidence:** CBFM gate report.
- [ ] **M7.14 [CORE] Evaluate tracker and policy utility.** Target: 2026-08-27. **Done:** calibration, return, diagnostic gain, safety/load, action frequency, coverage, and fallback rates produce explicit decisions. **Evidence:** tracker/policy gate report.
- [ ] **M7.15 [RESEARCH] Evaluate generation and guardrails where retained.** Target: 2026-08-27. **Done:** leakage, false positives, prompt load, latency, cost, and failure behavior are reported separately from RL learning. **Evidence:** generation report.
- [ ] **M7.16 [CORE] Freeze final metrics and figures source data.** Target: 2026-08-28. **Done:** report manifest links every result to run, dataset, policy, config, and analysis hashes. **Evidence:** frozen report manifest.
- [ ] **M7.GATE Close results-freeze gate.** Target: 2026-08-28. **Done:** no feature/tuning work remains; valid negative and inconclusive results are frozen for writing. **Evidence:** signed/committed gate summary.

---

## M8 - Demo and Dissertation Evidence Package

**Objective:** Turn the working system and frozen results into an examiner-ready demonstration and reproducible academic argument.

**Target:** 2026-09-07
**Dependencies:** M7; writing tasks start earlier
**Requirements:** FR-UI-001 through FR-UI-006, NFR-REP-003 through NFR-REP-005, NFR-PERF-001, NFR-SEC-001 through NFR-SEC-003

- [ ] **M8.1 [CORE] Reproduce the project in a clean Pixi environment.** Target: 2026-08-29. **Done:** one documented command runs tests, one deterministic episode, and one representative report from a clean directory. **Evidence:** clean reproduction transcript.
- [ ] **M8.2 [CORE] Build the final deterministic demo bundle.** Target: 2026-08-30. **Done:** React assets and FastAPI run together with templates, local storage, reset, and no credentials. **Evidence:** versioned demo artifact.
- [ ] **M8.3 [RESEARCH] Add the pinned LLM demo mode if stable.** Target: 2026-08-30. **Done:** mode records provider/model and falls back visibly but smoothly; otherwise it is disabled in the final build. **Evidence:** retained mode test or disable decision.
- [ ] **M8.4 [CORE] Finalize the Glass Box explanation view.** Target: 2026-08-31. **Done:** examiner can distinguish input/evidence, tracker estimate, action, prompt, CBFM estimate, and synthetic truth without developer tools. **Evidence:** annotated screenshots.
- [ ] **M8.5 [CORE] Run demo reliability rehearsal.** Target: 2026-09-01. **Done:** three consecutive local runs cover normal, provider-failure, restart, and reset scenarios without manual repair. **Evidence:** rehearsal checklist.
- [ ] **M8.6 [CORE] Create a recorded backup walkthrough.** Target: 2026-09-01. **Done:** short recording shows the critical path and one experiment result in case live infrastructure fails. **Evidence:** local/approved archive path.
- [ ] **M8.7 [CORE] Generate final tables and figures from commands.** Target: 2026-09-02. **Done:** outputs match frozen metric artifacts and contain uncertainty/sample sizes. **Evidence:** report build output and hashes.
- [ ] **M8.8 [CORE] Complete methods and implementation chapters.** Target: 2026-09-03. **Done:** text matches actual final architecture, equations, timing, schemas, splits, and versions. **Evidence:** dissertation source review.
- [ ] **M8.9 [CORE] Complete results and limitations chapters.** Target: 2026-09-04. **Done:** all claims trace to frozen results and explicitly cover construct validity, simulator circularity, sim-to-real limits, and negative findings. **Evidence:** claim-to-evidence table.
- [ ] **M8.10 [CORE] Complete reproducibility documentation.** Target: 2026-09-05. **Done:** install, demo, experiment, analysis, data-layout, and known-limitation instructions work from a clean checkout. **Evidence:** `README.md` and runbooks.
- [ ] **M8.11 [CORE] Run an examiner-style demo review.** Target: 2026-09-06. **Done:** reviewer asks for state/action rationale, failure fallback, and experimental result; all can be shown within the allotted time. **Evidence:** review notes and fixed critical issues.
- [ ] **M8.12 [CORE] Freeze the demo.** Target: 2026-09-07. **Done:** code/version/config is tagged; only critical bug fixes may enter afterward with rerun evidence. **Evidence:** demo tag and artifact hashes.
- [ ] **M8.GATE Close demo/evidence gate.** Target: 2026-09-07. **Done:** live and recorded demonstrations, clean reproduction, final figures, and claim traceability are ready. **Evidence:** examiner package checklist.

---

## M9 - Final QA, Archive, and Submission

**Objective:** Protect the completed work from last-week regressions and submit a coherent, reproducible dissertation package.

**Target:** 2026-09-15
**Dependencies:** M8

- [ ] **M9.1 [CORE] Run the complete automated test suite on the frozen tag.** Target: 2026-09-08. **Done:** backend, frontend, property, environment, leakage, data, browser, and D2 checks pass or documented noncritical exceptions are approved. **Evidence:** final test report.
- [ ] **M9.2 [CORE] Audit requirement and claim traceability.** Target: 2026-09-09. **Done:** each implemented Must requirement and dissertation claim maps to code/test/result or explicit scope deviation. **Evidence:** final traceability matrix.
- [ ] **M9.3 [CORE] Audit mathematical notation against implementation.** Target: 2026-09-09. **Done:** CBFM, tracker, reward, timing, and RL update equations use consistent indices, variables, and constraints. **Evidence:** supervisor/peer review notes.
- [ ] **M9.4 [CORE] Audit privacy, secrets, and prohibited data.** Target: 2026-09-10. **Done:** secret scan passes; demo content, hidden reasoning, simulator truth leakage, and accidental human data are absent from submission artifacts. **Evidence:** final data audit.
- [ ] **M9.5 [CORE] Archive frozen scientific artifacts.** Target: 2026-09-10. **Done:** two verified copies contain configs, manifests, selected trajectories, policies, metrics, tables, figures, and checksums. **Evidence:** archive verification output.
- [ ] **M9.6 [CORE] Purge local demonstration responses from distributable artifacts.** Target: 2026-09-10. **Done:** SQLite/JSONL demo data is absent or reset to authored synthetic fixtures. **Evidence:** archive content scan.
- [ ] **M9.7 [CORE] Complete dissertation proofreading and reference checks.** Target: 2026-09-11. **Done:** figures/tables resolve, citations compile, limitations are explicit, and no planned-but-unimplemented feature is written as complete. **Evidence:** clean PDF build and checklist.
- [ ] **M9.8 [CORE] Perform final clean-machine demo rehearsal.** Target: 2026-09-12. **Done:** deterministic demo and backup recording are available; network/provider failure has a rehearsed fallback. **Evidence:** signed rehearsal checklist.
- [ ] **M9.9 [CORE] Preserve a two-day submission buffer.** Target: 2026-09-13. **Done:** no new feature work occurs; only formatting or critical reproducibility fixes are accepted. **Evidence:** final change log.
- [ ] **M9.10 [CORE] Submit the dissertation and required project artifacts.** Target: 2026-09-15. **Done:** institutional submission receipts and final artifact hashes are retained. **Evidence:** submission confirmation.
- [ ] **M9.GATE Close the dissertation project.** Target: 2026-09-15. **Done:** submission, demo, archive, and reproducibility package are complete; remaining work is explicitly post-dissertation. **Evidence:** closure checklist.

---

## 6. Continuous Writing and Supervisor Controls

These tasks run alongside implementation so writing is not deferred until results freeze.

- [ ] **C1 [CORE] Freeze the revised research questions and minimum claims.** Target: 2026-07-17. **Done:** supervisor agrees which claims survive if SMC/Q-learning are dropped. **Evidence:** meeting note and proposal amendment list.
- [ ] **C2 [CORE] Update methods architecture after Gate 1.** Target: 2026-07-26. **Done:** dissertation describes the implemented vertical slice rather than the former enterprise plan. **Evidence:** committed methods draft.
- [ ] **C3 [CORE] Draft simulator/CBFM methods from tested equations.** Target: 2026-08-05. **Done:** notation, ordering, parameters, and falsification criteria match code. **Evidence:** methods draft and equation-to-test map.
- [ ] **C4 [CORE] Draft RL/evaluation methods before held-out runs.** Target: 2026-08-18. **Done:** baselines, splits, seeds, metrics, exclusions, and intervals are prespecified. **Evidence:** committed evaluation chapter/specification.
- [ ] **C5 [CORE] Hold a scope review after each Friday milestone.** Target: weekly through 2026-08-28. **Done:** each review records schedule variance, next gate, risks, and dropped scope. **Evidence:** dated notes in `docs/progress/`.
- [ ] **C6 [CORE] Maintain a decision log for every conditional component.** Target: continuous through 2026-08-20. **Done:** SMC, Q-learning, OpenRouter, sandbox, and task/profile expansion each have a dated decision. **Evidence:** `docs/decisions/`.
- [ ] **C7 [CORE] Maintain a claim-to-evidence table.** Target: weekly from 2026-08-14. **Done:** no result claim lacks a run/metric hash and no synthetic result is presented as human efficacy. **Evidence:** traceability table.

## 7. Minimum and Expanded Experiment Matrices

### 7.1 Minimum Matrix

Required if schedule is constrained:

- Policies: heuristic and contextual bandit; static reference where cheap.
- Trackers: simple and BKT.
- CBFM: disabled and enabled.
- Tasks/profiles: minimum validated set from M3.
- Seeds: empirically justified matched budget, targeting 20-30 per condition.
- Rendering: deterministic templates for all core policy comparisons.

This is already a meaningful controlled study. It tests whether CBFM and a learned policy add value over simple observable/tracker baselines without depending on SMC or LLM nondeterminism.

### 7.2 Expanded Matrix

Add only after entry gates:

- Reduced SMC if it improves calibration/decision utility over BKT and is stable.
- Tabular Q-learning if discretized state coverage is adequate.
- Pinned OpenRouter rendering on selected evaluation episodes.
- Guardrail enabled/disabled over the fixed leakage dataset.
- More tasks/profiles only if fixtures are validated before freeze.

Do not create a full Cartesian product automatically. Each added interaction must answer a stated question and fit the run/analysis budget.

## 8. Scope Reduction Order

When schedule or integration risk rises, remove work in this order:

1. Arbitrary code execution and sandbox integration.
2. UI animation, streaming, Monaco, and nonessential dashboard views.
3. Reduced SMC, retaining BKT.
4. Tabular Q-learning, retaining contextual bandit and heuristic.
5. LLM rendering in the live demo, retaining templates and a supplemental offline renderer comparison.
6. Additional tasks/profiles beyond the minimum validated set.
7. W&B online synchronization, retaining local manifests and metrics.

Never remove the browser vertical slice, deterministic simulator, simple/BKT baseline, one learned-policy comparison, CBFM control, split/seed discipline, public/truth separation, uncertainty, or reproducible analysis without an explicit supervisor-approved change to the dissertation question.

## 9. Risk Register and Triggered Actions

| Risk | Early signal | Immediate action | Escalation threshold |
|---|---|---|---|
| Integration consumes research time | No offline browser turn by 2026-07-20 | Stop all advanced work; pair-debug only the critical path | Gate 1 misses 2026-07-24 |
| Simulator encodes the desired conclusion | CBFM wins only on reward it defines | Prioritize M1 prompt-feature control and negative controls | No external/load-proxy comparison by 2026-08-07 |
| SMC instability | Non-finite weights or no BKT gain | Reject SMC and document limitation | Two focused days without stable baseline |
| RL exploits sparse authored states | High return with poor coverage/generalization | Report visitation/fallback; simplify claims | Q unseen-state rate exceeds prespecified limit |
| LLM route variability | Returned provider/model differs or prompts drift | Pin route; move comparison offline; retain templates | Any unreproducible primary run |
| Data leakage | Policy observation contains truth-derived field | Stop runs, invalidate affected artifacts, fix audit | Any held-out result affected |
| Experiment matrix explosion | Estimated runtime/analysis exceeds four days | Use minimum matrix and conditional decisions | Freeze date threatened |
| Dissertation writing lags | Methods not current by 2026-08-18 | Stop optional coding for two writing days | Results start with missing methods |
| Demo host/network failure | Provider or deployment fails rehearsal | Use local deterministic build and recording | Failure in two consecutive rehearsals |
| Pilot starts informally | Request to test students before approval | Do not collect data; contact supervisor/ethics route | Any identifiable/student data proposed |

## 10. Post-Dissertation Pilot Readiness Backlog

This section is not part of the September dissertation Definition of Done. It does not authorize recruitment or data collection. Dates are relative to a future supervisor-approved pilot start decision `T0`.

- [ ] **P1 Define one human-study question and primary outcome.** Target: `T0 + 1 week`. **Done:** intervention, comparator, outcome, population, and analysis are narrow enough for a small pilot. **Evidence:** protocol synopsis.
- [ ] **P2 Confirm the institutional ethics and data-protection route.** Target: `T0 + 1 week`. **Done:** named review route, sponsor/supervisor, controller/processors, and required submissions are confirmed in writing. **Evidence:** institutional correspondence.
- [ ] **P3 Select independent human measures.** Target: `T0 + 2 weeks`. **Done:** pre/post learning test and validated cognitive-load instrument are chosen without treating CBFM as ground truth. **Evidence:** instrument rationale and permissions.
- [ ] **P4 Write protocol, recruitment, consent, withdrawal, and adverse-event materials.** Target: `T0 + 3 weeks`. **Done:** complete application package is supervisor-reviewed. **Evidence:** versioned study documents.
- [ ] **P5 Perform sample-size and feasibility analysis.** Target: `T0 + 3 weeks`. **Done:** pilot purpose, recruitment ceiling, uncertainty goals, attrition, and stopping rules are justified. **Evidence:** analysis note.
- [ ] **P6 Design a separate human-data schema and retention plan.** Target: `T0 + 3 weeks`. **Done:** pseudonyms, identity separation, consent versions, deletion/withdrawal, access, processors, and retention are specified. **Evidence:** approved data management plan.
- [ ] **P7 Submit for approval and freeze data collection.** Target: `T0 + 4 weeks`. **Done:** application is submitted; no participant interaction occurs while approval is pending. **Evidence:** submission receipt.
- [ ] **P8 Build the frozen pilot application only after approval conditions are known.** Target: within 3 weeks after approval. **Done:** no simulator truth, online learning, hidden reasoning, or unapproved external processing exists; policy/tasks/prompts are frozen. **Evidence:** pilot build audit.
- [ ] **P9 Rehearse safety, accessibility, consent, deletion, and failure procedures.** Target: within 1 week before recruitment. **Done:** dry run passes with synthetic/test identities and no real participant data. **Evidence:** readiness checklist.
- [ ] **P10 Begin recruitment only after a formal go/no-go review.** Target: after P1-P9 and approval. **Done:** supervisor confirms approval conditions, frozen build, instruments, data plan, and support procedures are satisfied. **Evidence:** signed go decision.

## 11. Final Project Completion Checklist

- [ ] The one-task browser path works offline from submission to logged next prompt.
- [ ] LangGraph is confined to interactive turns; RL runs directly through Gymnasium.
- [ ] The simulator passes `check_env`, deterministic replay, timing, and leakage tests.
- [ ] Public and privileged trajectories validate and remain separately readable.
- [ ] CBFM has a prompt-feature control, negative controls, and a retain/reject/inconclusive result.
- [ ] Simple and BKT trackers provide tested baselines; SMC has an evidence-based decision.
- [ ] Heuristic and contextual bandit policies complete matched comparisons; Q-learning has an evidence-based decision.
- [ ] OpenRouter and sandbox dependencies are optional and cannot break the deterministic demo.
- [ ] Final analysis reproduces from frozen local manifests, Parquet, artifacts, and commands.
- [ ] Every reported result includes uncertainty, sample counts, exclusions, and lineage.
- [ ] The demo is rehearsed locally and has a recorded fallback.
- [ ] The dissertation matches implemented behavior and states construct/sim-to-real limitations.
- [ ] No human research data was collected under the dissertation POC.
- [ ] Submission and archive receipts are retained.
