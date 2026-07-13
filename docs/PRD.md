# Product Requirements Document

## Socratic Tutoring Research POC

| Field | Value |
|---|---|
| Status | Approved rescope for implementation |
| Product phase | Master's dissertation proof of concept |
| Revision date | 2026-07-12 |
| Primary source | `proposal/dissertation_proposal.tex` |
| Delivery target | Working demo and reproducible simulation study by 2026-09-15 |
| Initial domain | Python mutable-list aliasing and mutation |
| Initial users | Researcher, dissertation evaluator, demo participant |
| Future users | Pilot-study students after separate ethics approval |

> **Downstream-document status:** This PRD is the current source of truth. `docs/engineering-plan.md`, `docs/HLD.md`, `docs/LLD.md`, `docs/data-flow-and-schema.md`, and `docs/implementation-plan.md` still describe the previous enterprise scope and must be revised sequentially before implementation begins beyond the vertical slice.

## 1. Purpose

This document defines a deliberately narrow proof of concept for evaluating adaptive Socratic tutoring. The system uses LangGraph to orchestrate a tutoring turn, maintains an estimated learner state, selects an interpretable pedagogical directive, and renders the next Socratic prompt through a deterministic template or a pinned LLM.

The dissertation is not building a production tutoring platform. Its required outcome is a working, inspectable demo and a reproducible simulation experiment that can determine whether Cognitive Bandwidth and Friction modelling, probabilistic tracking, and delayed-return policy learning add value over simpler alternatives.

The first mandatory product is one complete vertical slice:

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

BKT, Cognitive Bandwidth and Friction Modelling (CBFM), reduced Sequential Monte Carlo (SMC), contextual bandits, and tabular Q-learning are introduced only after this path works end to end.

## 2. Product Vision

Provide a small research instrument that makes the tutoring decision loop visible and falsifiable. A reviewer must be able to inspect:

- What the student submitted.
- What evidence was extracted.
- How the tracker changed its estimate.
- Which pedagogical action was selected.
- Which prompt was shown next.
- How simulated reward and external evaluation metrics were calculated.

The project succeeds even if SMC, CBFM, or Q-learning is rejected, provided the comparison is rigorous and the simpler winning approach is reported honestly.

## 3. Product Principles

1. **A working vertical slice comes first.** No RL, SMC, distributed infrastructure, or advanced UI work precedes it.
2. **The simplest credible baseline is mandatory.** Every proposed component must outperform or explain its value over a simpler alternative.
3. **Simulation truth is privileged.** Tutor policies and future human interfaces never receive simulator-owned latent state.
4. **LangGraph is orchestration, not the RL environment.** Interactive turns use LangGraph; training rollouts call Gymnasium directly.
5. **LLMs do not participate in RL training.** Mathematical/template simulation generates training trajectories; LLMs render frozen actions for the demo and language-layer evaluation.
6. **Canonical data remains local and inspectable.** Parquet is the scientific source of truth; W&B tracks configurations, metrics, and artifact references.
7. **CBFM is a simulator construct.** `B_t` and `F_t` are not validated measurements of human fatigue or working memory.
8. **No raw hidden reasoning is stored or displayed.** The product retains public text, code, tests, structured evidence, and final outputs only.
9. **Human research is separate.** The dissertation demo does not authorize student recruitment or data collection.

## 4. Users and Stakeholders

### 4.1 Researcher

Builds the system, configures experiments, inspects trajectories, trains policies, executes ablations, and generates dissertation evidence.

### 4.2 Dissertation Evaluator

Runs the demo, inspects one complete tutoring turn, compares true versus estimated synthetic state, and reviews reproducible experiment reports.

### 4.3 Demo Participant

Interacts with the POC without their interaction being treated as research data. The participant receives a Python task, submits an explanation or code, and sees the next Socratic prompt.

### 4.4 Future Pilot Student

Participates only after a separate ethics, consent, privacy, recruitment, and study-protocol process. Future pilot functionality is a deferred product boundary, not part of the dissertation's required data collection.

## 5. Product Modes

### 5.1 Vertical-Slice Mode

Uses one authored task, deterministic evidence rules, a simple mastery tracker, a heuristic policy, template prompts, local session state, and append-only JSONL logging. This mode must work without network access.

### 5.2 Deterministic Simulation Mode

Runs `SocraticTutor/POMDP-v0` through Gymnasium using mathematical student transitions, template prompts, frozen seeds, and no LLM calls. This is the policy-training and reproducibility reference.

### 5.3 LLM-Rendered Demo Mode

Uses the same tracker and policy decisions as deterministic mode but allows a pinned OpenRouter model/provider route to render the selected directive. A safe deterministic template remains the fallback.

### 5.4 Glass Box Mode

Displays synthetic true state, tracker estimates, selected actions, prompt-load features, CBFM state, evidence, and rewards in clearly separated views. Simulator truth is labeled and unavailable outside synthetic sessions.

### 5.5 Future Pilot Mode

Deferred until after dissertation delivery and prior institutional approval. It will contain no simulator truth and will use frozen policies, approved instruments, consent, and a separate human-data lifecycle.

## 6. Goals and Non-Goals

### 6.1 Required Goals

- Complete the thin vertical slice through a browser.
- Build a deterministic Gymnasium environment with reproducible trajectories.
- Implement a simple tracker before BKT and reduced SMC.
- Implement a heuristic policy before contextual bandit and Q-learning.
- Implement bounded CBFM dynamics and test whether they add predictive value.
- Compare policies over identical observations, actions, splits, and matched seeds.
- Produce canonical Parquet trajectories and reproducible statistical reports.
- Deliver a stable FastAPI + React demonstration.

### 6.2 Non-Goals

- Production availability, horizontal scaling, or enterprise service decomposition.
- Redis queues, PostgreSQL clusters, Kubernetes, Terraform, Ray, or RLlib.
- A self-hosted LiteLLM proxy or organizational LLM gateway.
- Online RL against human students.
- Neural PPO, DQN, recurrent policies, or foundation-model training.
- Multiple programming languages or a broad curriculum.
- Establishing educational efficacy from synthetic students.
- Running a real-student pilot before dissertation submission.
- Claiming that `B_t` or `F_t` directly measures a human mental state.

## 7. Approved Technical Constraints

| Concern | Required POC choice |
|---|---|
| Workspace isolation | Pixi and committed `pixi.lock` |
| Python runtime | Python 3.12 |
| Frontend runtime | Node.js and pnpm with committed `pnpm-lock.yaml` |
| Frontend | React, TypeScript, Vite |
| API | FastAPI and Pydantic v2 |
| Orchestration | LangGraph with in-memory tests and SQLite demo checkpointing |
| LLM access | OpenRouter through a typed local `ModelGateway` |
| RL environment | Gymnasium |
| RL implementations | NumPy/SciPy heuristic, contextual bandit, and tabular Q-learning |
| Experiment configuration | Hydra with saved fully resolved configurations |
| Experiment tracking | Weights & Biases |
| Development logging | Append-only JSONL |
| Canonical trajectories | Versioned Parquet |
| Analysis | DuckDB, SciPy, statsmodels |
| Testing | pytest, Hypothesis, Gymnasium `check_env` |

W&B is not the canonical data store. Complete trajectories, policy tables, calibration artifacts, and final reports must remain reconstructable from repository-controlled configuration and local/versioned artifacts.

## 8. Functional Requirements

### 8.1 Thin Vertical Slice

| ID | Requirement | Acceptance criterion | Priority |
|---|---|---|---|
| FR-VS-001 | The system shall provide one versioned Python task about mutable-list aliasing and mutation. | The task, concept, misconception, rubric, expected evidence, and template prompts load from repository data. | Must |
| FR-VS-002 | A user shall create a session and receive the initial Socratic prompt in the React client. | Browser-to-FastAPI integration test returns a valid session and prompt. | Must |
| FR-VS-003 | A user shall submit one text response through the browser. | The turn endpoint validates and records the response under the correct session and turn. | Must |
| FR-VS-004 | A deterministic evidence extractor shall classify the response. | Known correct, misconception, uncertain, and empty fixtures produce expected structured evidence. | Must |
| FR-VS-005 | A simple tracker shall update a bounded mastery estimate from evidence. | Hand-calculated update fixtures match implementation output and remain in `[0,1]`. | Must |
| FR-VS-006 | A heuristic policy shall select the next pedagogical directive. | Rule-table fixtures cover correct, incorrect, uncertain, repeated-failure, and low-bandwidth states. | Must |
| FR-VS-007 | A template renderer shall produce the next Socratic prompt. | Every legal directive has a safe authored template and no complete solution leakage. | Must |
| FR-VS-008 | Every turn shall append one structured local event. | Restart-safe JSONL contains session, turn, input, evidence, tracker before/after, action, prompt, versions, and timestamp. | Must |
| FR-VS-009 | The browser shall display the interaction and inspectable state transition. | A reviewer can complete and explain one full turn without using developer tools. | Must |
| FR-VS-010 | The vertical slice shall run without LLM, W&B, or sandbox credentials. | Offline end-to-end test passes using templates and local files. | Must |

### 8.2 Experiment Configuration and Reproducibility

| ID | Requirement | Acceptance criterion | Priority |
|---|---|---|---|
| FR-EXP-001 | Every experiment shall start from a versioned Hydra configuration. | A run saves the fully resolved configuration before its first episode. | Must |
| FR-EXP-002 | Calibration, policy-selection, validation, and held-out splits shall be disjoint. | Overlap validation blocks execution before rollout. | Must |
| FR-EXP-003 | A root seed shall derive environment, transition, policy, and generation seeds. | Reordering jobs does not change per-episode deterministic results. | Must |
| FR-EXP-004 | Policy comparisons shall use matched tasks, profiles, and seeds. | Comparison reports identify complete pairs and exclusions. | Must |
| FR-EXP-005 | Every run shall record code revision and dependency/configuration hashes. | W&B and local manifest contain Git revision, dirty state, Pixi lock hash, resolved config hash, split hash, and artifact versions. | Must |
| FR-EXP-006 | Held-out evaluation shall execute only after configuration and policy freeze. | Evaluation command rejects mutable or unselected candidate configurations. | Must |

### 8.3 Gymnasium Simulator

| ID | Requirement | Acceptance criterion | Priority |
|---|---|---|---|
| FR-SIM-001 | The system shall register `SocraticTutor/POMDP-v0`. | `gymnasium.make` and `check_env` pass in CI. | Must |
| FR-SIM-002 | The environment shall exclusively own simulator true state. | Policies cannot import, deserialize, or receive the true-state type. | Must |
| FR-SIM-003 | The environment shall preserve fast-prompt, evidence, and slow-learning timing. | Order-spy tests demonstrate the specified transition order. | Must |
| FR-SIM-004 | The simulator shall support a small versioned profile/task set. | At least 4 tasks across 3 concepts can run from frozen fixtures before final experiments. | Should |
| FR-SIM-005 | `reset(seed=...)` shall reproduce initial state and random streams. | Golden reset fixtures match exactly. | Must |
| FR-SIM-006 | Policy observations shall contain tracker-visible features only. | Recursive feature audit finds no simulator truth. | Must |
| FR-SIM-007 | Natural termination and administrative truncation shall be separate. | Completion, maximum-turn, budget, and failure fixtures return correct flags/reasons. | Must |
| FR-SIM-008 | RL rollouts shall run without LangGraph or LLM calls. | Training test succeeds with network disabled. | Must |

### 8.4 Cognitive Bandwidth and Friction

| ID | Requirement | Acceptance criterion | Priority |
|---|---|---|---|
| FR-CBFM-001 | Prompt load shall use normalized token, code-span, AST-depth, novelty, and entropy features. | Feature fixtures are deterministic and versioned. | Must |
| FR-CBFM-002 | Overload friction shall increase with positive ZPD mismatch, holding other inputs fixed. | Monotonicity property tests pass. | Must |
| FR-CBFM-003 | Bandwidth updates shall include bounded drain and recovery. | Hypothesis tests prove finite values in `[0,1]` and expected low-load recovery. | Must |
| FR-CBFM-004 | Underchallenge shall remain a separate diagnostic. | It is logged separately and does not replace overload friction. | Should |
| FR-CBFM-005 | A full CBFM ablation shall be available. | Ablation fixes `B_t=1`, `F_t=0`, and removes CBFM policy features. | Must |
| FR-CBFM-006 | Calibration shall use calibration data and external load proxies only. | Aggregate policy reward and held-out outcomes are inaccessible to the fitter. | Must |
| FR-CBFM-007 | Evaluation shall compare mastery-only, prompt-feature-only, and CBFM-augmented models. | Held-out Brier score and negative log-likelihood are reported with intervals. | Must |
| FR-CBFM-008 | Sensitivity and negative controls shall be reported. | Parameter perturbation and shuffled-feature tests produce a robustness result. | Must |
| FR-CBFM-009 | Product language shall identify CBFM as simulated or estimated. | Copy tests reject unsupported human-fatigue claims. | Must |

### 8.5 Tracking

| ID | Requirement | Acceptance criterion | Priority |
|---|---|---|---|
| FR-TRK-001 | All trackers shall implement one `BeliefTracker` interface. | Simple, BKT, and reduced-SMC implementations pass shared contract tests. | Must |
| FR-TRK-002 | The first tracker shall use deterministic structured evidence. | The vertical slice works without an LLM evidence classifier. | Must |
| FR-TRK-003 | A BKT tracker shall be implemented before SMC. | BKT completes deterministic simulator episodes and calibration tests. | Must |
| FR-TRK-004 | Reduced SMC shall track only the active concept, limited misconception flags, `B_t`, and `F_t`. | State dimensionality is declared and particle updates remain numerically stable. | Should |
| FR-TRK-005 | Trackers shall expose estimates and uncertainty, not particles, to policies. | `PolicyObservation` contains only approved summaries. | Must |
| FR-TRK-006 | Sparse evidence shall not update inactive concepts as failures. | Empty active masks yield neutral likelihood/update behavior. | Must |
| FR-TRK-007 | SMC shall report ESS and resample below a configured threshold. | Seeded resampling and degeneracy tests pass. | Should |
| FR-TRK-008 | Tracker parameters may differ from simulator truth. | Misspecification experiments run from configuration without code changes. | Should |
| FR-TRK-009 | The tracker shall be the only writer of tutor-facing estimates. | LangGraph node/state tests reject unauthorized writes. | Must |

### 8.6 Tutor Policies and RL

| ID | Requirement | Acceptance criterion | Priority |
|---|---|---|---|
| FR-POL-001 | Policies shall select from one finite versioned directive set. | Unknown directives fail validation. | Must |
| FR-POL-002 | Compared policies shall receive identical observation fields. | Automated parity audit passes before training/evaluation. | Must |
| FR-POL-003 | The heuristic policy shall work before learned policies are implemented. | The vertical slice and deterministic simulator complete using only heuristics. | Must |
| FR-POL-004 | The policy shall choose a directive and target, not generate arbitrary text. | All decisions validate against `PolicyDecision`. | Must |
| FR-POL-005 | The contextual bandit shall record chosen-action propensity. | Every stochastic decision logs a valid propensity. | Should |
| FR-POL-006 | Tabular Q-learning shall use a compact documented discretization. | Boundary fixtures and unseen-state fallback tests pass. | Should |
| FR-POL-007 | Q-table coverage and state/action visitation shall be reported. | Final reports include coverage, unseen-state rate, and action frequencies. | Should |
| FR-POL-008 | RL training shall use local CPU rollouts and no LLM calls. | Network-disabled training produces a policy artifact. | Must |
| FR-POL-009 | Demo and evaluation sessions shall use frozen policy artifacts. | Policy version cannot change within an episode. | Must |

### 8.7 LangGraph, Generation, and Guardrails

| ID | Requirement | Acceptance criterion | Priority |
|---|---|---|---|
| FR-GRF-001 | LangGraph shall orchestrate tracker, policy, rendering, guardrail, and persistence for interactive turns. | End-to-end graph topology and turn-order tests pass. | Must |
| FR-GRF-002 | Graph nodes shall return validated state deltas with declared ownership. | Unauthorized or extra fields fail before checkpointing. | Must |
| FR-GRF-003 | Demo sessions shall use SQLite checkpointing after the vertical slice. | Restart test restores the latest committed turn. | Should |
| FR-GRF-004 | External calls shall be isolated and replaceable by deterministic fakes. | Graph integration suite runs offline. | Must |
| FR-GRF-005 | Turn loops and guardrail rewrites shall be bounded. | Maximum turns/retries terminate with typed reasons. | Must |
| FR-GEN-001 | OpenRouter access shall be hidden behind a typed local `ModelGateway`. | Tutor code depends on the protocol rather than provider SDK details. | Must |
| FR-GEN-002 | Primary experiments shall pin model, provider routing, sampling parameters, and prompt hash. | Run manifest records request and returned provider/model metadata. | Must |
| FR-GEN-003 | A deterministic template renderer shall always be available. | LLM timeout/failure returns an authored Socratic prompt. | Must |
| FR-GRD-001 | A rule-first guardrail shall reject complete answers and solution code. | Versioned leakage fixtures report precision, recall, and false positives. | Must |
| FR-GRD-002 | Guardrail output shall be structured and validated. | Malformed classifier output, if later added, fails closed. | Should |
| FR-GRD-003 | Rewrite attempts shall be bounded and end in a safe template. | Exhaustion route is deterministic and tested. | Must |

### 8.8 Logging, Trajectories, and W&B

| ID | Requirement | Acceptance criterion | Priority |
|---|---|---|---|
| FR-TRJ-001 | Development turns shall append versioned JSONL events. | Interrupted writes cannot corrupt earlier complete lines. | Must |
| FR-TRJ-002 | Experiment trajectories shall use versioned Parquet schemas. | Schema, finite-value, unique-key, and row-count validation pass. | Must |
| FR-TRJ-003 | Public policy data and privileged simulator truth shall be separated. | Policy readers enumerate public columns and cannot read privileged files. | Must |
| FR-TRJ-004 | DuckDB shall query canonical Parquet partitions for analysis. | Analysis runs without importing data from W&B. | Must |
| FR-TRJ-005 | W&B shall track resolved configuration, metrics, summaries, and artifact references. | Every final run resolves to local canonical artifacts and hashes. | Must |
| FR-TRJ-006 | W&B Tables shall contain selected examples only. | Complete raw trajectories remain reproducible when W&B is unavailable. | Must |

### 8.9 Demo Interface

| ID | Requirement | Acceptance criterion | Priority |
|---|---|---|---|
| FR-UI-001 | React shall display the task, conversation, response input, next prompt, and session status. | Browser E2E test completes one vertical-slice turn. | Must |
| FR-UI-002 | Glass Box views shall distinguish simulator truth, tracker estimate, and observable evidence. | Labels and access tests prevent ambiguous presentation. | Must |
| FR-UI-003 | The UI shall show selected directive, mastery estimate, and CBFM values when available. | Values update only after a committed turn and identify their source. | Should |
| FR-UI-004 | The UI shall not display raw hidden reasoning. | Content tests permit only public text, code, tests, and structured evidence. | Must |
| FR-UI-005 | The demo shall handle LLM failure gracefully. | Simulated timeout produces a deterministic template without losing the turn. | Must |
| FR-UI-006 | Synthetic privileged controls shall be clearly labeled and absent from future pilot builds. | Build/configuration test enforces the mode boundary. | Must |

### 8.10 Code Execution

| ID | Requirement | Acceptance criterion | Priority |
|---|---|---|---|
| SEC-SBX-001 | Arbitrary submitted code shall never execute in the FastAPI host process. | Deployed configuration has no host executor path. | Must when code execution is enabled |
| SEC-SBX-002 | One external sandbox provider shall enforce network, CPU, memory, time, process, filesystem, and output limits. | Integration tests produce typed outcomes for every limit. | Should |
| SEC-SBX-003 | Sandbox telemetry shall be normalized before tracker ingestion. | Provider-specific responses map to one evidence schema. | Should |
| SEC-SBX-004 | The vertical slice may use authored text/probe fixtures without arbitrary code execution. | Initial demo remains usable before sandbox integration. | Must |

## 9. Non-Functional Requirements

### 9.1 Reproducibility

| ID | Requirement | Target |
|---|---|---|
| NFR-REP-001 | Pixi and pnpm lockfiles shall be committed. | Clean environment installs the pinned Python and frontend dependencies. |
| NFR-REP-002 | Every run shall save its fully resolved Hydra configuration. | 100% configuration capture before episode execution. |
| NFR-REP-003 | Deterministic template episodes shall replay from seeds. | Content-equivalent trajectories excluding timestamps/generated IDs. |
| NFR-REP-004 | Final reports shall reference trajectory, configuration, code, and policy hashes. | 100% report-to-source traceability. |
| NFR-REP-005 | Held-out data shall not influence calibration or tuning. | Split-access tests and run manifests show no leakage. |

### 9.2 Reliability and Performance

| ID | Requirement | Target |
|---|---|---|
| NFR-REL-001 | The deterministic vertical slice shall work without external services. | Complete browser turn using local templates and storage. |
| NFR-REL-002 | Every loop, request, and external call shall have a bound. | No unbounded episode, retry, or generation path. |
| NFR-PERF-001 | Non-LLM turn processing shall be responsive on a laptop. | p95 backend processing below 500 ms in local tests. |
| NFR-PERF-002 | Mathematical simulation shall remain cheap enough for matched multi-seed evaluation. | Target 20–30 seeds per condition without distributed compute. |

### 9.3 Security and Privacy

| ID | Requirement | Target |
|---|---|---|
| NFR-SEC-001 | Secrets shall remain outside source, logs, trajectories, and W&B configuration. | Secret scan passes. |
| NFR-SEC-002 | Simulator truth shall be inaccessible to policies. | Import, schema, and trajectory feature audits pass. |
| NFR-SEC-003 | Demo logs shall contain no credentials or raw hidden reasoning. | Adversarial content scan passes. |
| NFR-PRV-001 | Dissertation experiments shall use synthetic data only. | No participant identifiers or human-study records exist. |
| NFR-PRV-002 | Future pilot data shall not be sent to W&B or OpenRouter without approved governance. | Pilot activation remains blocked pending ethics/data review. |

### 9.4 Maintainability

| ID | Requirement | Target |
|---|---|---|
| NFR-MNT-001 | Tracker and policy implementations shall use small shared protocols. | Baselines swap without changing API or environment behavior. |
| NFR-MNT-002 | Mathematical components shall be testable without LangGraph. | Unit tests import no web or LLM runtime. |
| NFR-MNT-003 | Training and evaluation shall run from commands, not notebooks. | Pixi tasks reproduce all final runs. |
| NFR-MNT-004 | CI shall run lint, type, unit, property, environment, frontend, and D2 checks. | Required jobs pass before final experiments. |

## 10. Release and Dependency Gates

### Gate 1: Vertical Slice

Pass when one browser interaction completes the full critical path, writes a valid JSONL event, works offline, and is explainable by a reviewer.

No SMC, RL, LLM integration, sandbox, W&B, or advanced visualization work may block this gate.

### Gate 2: Deterministic Simulator

Pass when `SocraticTutor/POMDP-v0` passes `check_env`, deterministic replay, state-ownership, timing, reward, and termination tests.

### Gate 3: CBFM

Pass when boundedness, monotonicity, recovery, complete ablation, calibration separation, and sensitivity tests pass. Failure means CBFM remains an analyzed but rejected component.

### Gate 4: Tracking

Pass BKT before reduced SMC begins. Retain SMC only if it remains stable and improves calibration or policy-relevant uncertainty over the simpler tracker.

### Gate 5: RL

Begin bandit and Q-learning only after heuristic trajectories are reproducible. Retain Q-learning only if it improves over heuristic and bandit controls without safety or cost regression.

### Gate 6: Experiments

Pass when configurations, splits, seeds, policy artifacts, prompt/model routes, and analysis code are frozen before held-out execution.

### Gate 7: Demo

Pass when FastAPI + React reliably completes deterministic and LLM-rendered scenarios, clearly distinguishes evidence/estimates/truth, and survives provider failure through templates.

### Gate 8: Dissertation

Pass when final results, uncertainty, negative findings, costs, limitations, canonical artifacts, and reproduction instructions are complete.

## 11. Evaluation Requirements

The final simulation study shall compare:

- Heuristic policy.
- Contextual bandit.
- Tabular Q-learning.
- Optional static Socratic reference.

Required controls:

- Simple/BKT tracker versus reduced SMC.
- CBFM disabled versus enabled.
- Template rendering versus pinned LLM rendering for selected evaluation episodes.
- Guardrail disabled versus enabled on leakage fixtures.

Required reporting:

- Matched task/profile/seed results.
- Episode return and decomposed reward.
- Held-out diagnostic gain.
- Leakage and false-positive rates.
- Tracker calibration and uncertainty.
- CBFM load-proxy prediction.
- Q-table coverage and unseen-state rate.
- State/action visitation and action frequency.
- Latency, token use, and cost for LLM-rendered episodes.
- Paired or stratified bootstrap intervals.
- Negative and inconclusive decisions.

W&B Sweeps may tune policy hyperparameters on policy-selection/validation data only. Held-out data shall never drive a sweep.

## 12. Success Metrics

| Outcome | Success criterion |
|---|---|
| Working POC | A reviewer completes one browser turn and inspects the resulting state/action/log transition. |
| Reproducibility | A clean environment reproduces a deterministic episode and a representative report. |
| Simulator validity | Environment invariants, timing, seeds, and feature-separation tests pass. |
| CBFM utility | CBFM improves held-out load-proxy prediction beyond prompt features or is explicitly rejected. |
| Tracker utility | Reduced SMC improves calibration/decision utility beyond BKT or is explicitly rejected. |
| RL utility | Q-learning outperforms heuristic and bandit controls on prespecified outcomes or is explicitly rejected. |
| Demo resilience | Provider failure falls back to safe deterministic behavior. |
| Dissertation quality | Claims match evidence and clearly limit sim-to-real interpretation. |

## 13. Risks and Mitigations

| Risk | Consequence | Required mitigation |
|---|---|---|
| Vertical slice delayed by architecture work | No working demo | Freeze advanced work until Gate 1 passes. |
| CBFM validates its own simulator assumptions | Circular result | Use external proxies, prompt-only control, negative controls, and held-out calibration. |
| SMC becomes numerically or dimensionally unstable | Tracker blocks integration | Reduce state, retain BKT fallback, and require stability gate. |
| Q-learning exploits authored transitions | Simulator-only policy gains | Use held-out tasks/profiles, parameter sweeps, strong heuristic/bandit baselines, and cautious claims. |
| OpenRouter routes across providers | Hidden experimental variability | Pin model/provider settings, disable fallback for primary experiments, and record returned metadata. |
| LLM output changes directive difficulty | Action is not a stable intervention | Store realized prompt features and evaluate multiple/frozen renderings. |
| W&B becomes the only copy of results | Reproducibility and access risk | Keep canonical Parquet/configuration/policy/report artifacts locally. |
| React work consumes research time | Evaluation remains incomplete | Keep one task workflow and postpone nonessential dashboard features. |
| Sandbox integration delays POC | No demo | Use text/probe interaction first; enable arbitrary code only after one provider works safely. |
| Future pilot is mistaken for approved research | Unusable or unethical data | Keep pilot disabled and require separate prior approval. |

## 14. Future Pilot Boundary

The dissertation shall produce a working demo but shall not recruit students or treat demo interactions as research data.

A future pilot requires, before any collection:

- Confirmed ethics-review route and approval.
- Supervisor/institutional sponsorship as required.
- Participant information and consent.
- Defined research question and primary outcome.
- Recruitment and sample-size rationale.
- Frozen two-condition study design.
- Pre/post assessment and independent cognitive-load instrument.
- Human-data retention, withdrawal, deletion, and processor review.
- Security, accessibility, and failure rehearsal.
- Frozen application, policy, prompt, and task versions.

Synthetic success does not establish human educational efficacy.

## 15. Requirement Traceability

| Research objective | Primary requirements | Evidence gate |
|---|---|---|
| Working tutoring loop | FR-VS-001 through FR-VS-010 | Vertical-slice gate |
| Reproducible simulator | FR-EXP-001 through FR-SIM-008 | Simulator gate |
| CBFM construct utility | FR-CBFM-001 through FR-CBFM-009 | CBFM gate |
| Belief tracking | FR-TRK-001 through FR-TRK-009 | Tracking gate |
| Policy-learning value | FR-POL-001 through FR-POL-009 | RL gate |
| Safe orchestration and rendering | FR-GRF-001 through FR-GRD-003 | Demo and leakage tests |
| Scientific lineage | FR-TRJ-001 through FR-TRJ-006, NFR-REP-001 through NFR-REP-005 | Experiment gate |
| Usable demonstration | FR-UI-001 through FR-UI-006 | Demo gate |
| Secure optional code execution | SEC-SBX-001 through SEC-SBX-004 | Sandbox gate when enabled |
| Future human governance | NFR-PRV-001, NFR-PRV-002 | Separate future pilot approval |

## 16. Definition of Done

The Master's POC is complete when:

- The browser vertical slice passes from task display through structured log.
- The deterministic Gymnasium environment passes contract and replay tests.
- The heuristic policy and simple/BKT tracker provide reliable baselines.
- CBFM, reduced SMC, contextual bandit, and Q-learning each receive a clear retain/reject result where implemented.
- Final comparisons use frozen configurations, disjoint splits, and matched seeds.
- Canonical Parquet trajectories and analysis code reproduce the reported results.
- W&B runs resolve to local artifacts and complete provenance.
- The Glass Box clearly separates observable evidence, tracker estimates, policy actions, and simulator truth.
- The LLM renderer has a deterministic fallback and cannot block the demonstration.
- The dissertation reports uncertainty, cost, limitations, negative results, and the sim-to-real gap.
- No human research data has been collected under the dissertation POC.
