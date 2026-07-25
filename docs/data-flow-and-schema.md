# Data Flow and Storage Schema Design

## Master's POC: Socratic Tutoring and Evidence-Validity Evaluation

| Field | Value |
|---|---|
| Status | Proposed benchmark-first amendment for review |
| Revision date | 2026-07-14 |
| Product requirements | `docs/PRD.md` |
| High-level architecture | `docs/HLD.md` |
| Low-level design | `docs/LLD.md` |
| Interactive checkpoint store | Local SQLite through LangGraph's supported checkpointer |
| Demo event store | Append-only JSONL |
| Canonical research store | Versioned local Parquet datasets and content-addressed files |
| Analysis engine | DuckDB over Parquet |
| Experiment dashboard | Optional Weights & Biases; never canonical |

> **Sequential-amendment status:** the PRD, engineering plan, HLD, LLD, and this data design now describe the benchmark-first POC. `docs/implementation-plan.md` still describes the earlier POC and is the next sequential revision. The dissertation proposal has not yet been amended; the primary-study change requires supervisor approval.

## 1. Purpose and Scope

This document defines how data enters, moves through, persists in, and leaves the Master's POC. It covers:

- Interactive browser-turn data.
- LangGraph checkpoints and JSONL commit evidence.
- Authored benchmark cases, reviews, splits, probes, tests, and freeze manifests.
- Public/evidence generation, condition decisions, atomic commitments, evaluator-only criterion records, and paired case metrics.
- Deterministic simulation, training, and secondary evaluation data.
- Public policy-safe and privileged simulator schemas.
- Local artifact lineage and optional W&B synchronization.
- Validation, publication, retention, deletion, and schema evolution.

This is not a production database design. There is no PostgreSQL, Redis, object-store service, MLflow registry, distributed queue, or human-study database. The design preserves only the boundaries needed for the tutoring demonstration, benchmark evidence-validity study, and optional secondary simulation experiments.

## 2. Data Principles

1. **Local canonical evidence.** Final claims must reproduce from repository-controlled configs plus local artifacts without W&B or network access.
2. **Separate decision evidence from criterion.** Public/evidence records and evaluator-only criterion records use different schemas, roots, readers, and process phases.
3. **Commit before reveal.** Every planned sample is accounted for and every eligible condition set is sealed before criterion writers or readers are constructed.
4. **Separate observation from simulator truth.** Policy-safe turns and privileged simulator truth use different schemas, directories, readers, and validation commands.
5. **Explicit column roles.** Observation features, training targets, metadata, benchmark criterion, and simulator truth are classified separately.
6. **Append rather than mutate.** Demo events and published research datasets are immutable.
7. **Checkpoint is not research data.** SQLite supports session recovery; it is not the source for dissertation analysis.
8. **One event per logical turn.** `(session_id, turn_id)` is the interactive idempotency key.
9. **Hash every scientific artifact.** Reports resolve to benchmark, configs, data, model route, policy, code, and analysis versions.
10. **No hidden reasoning.** Raw chain-of-thought and provider-internal reasoning are never stored.
11. **Case is the primary unit.** Repeated LLM samples are nested records, never silently promoted to independent benchmark cases.
12. **Synthetic-only dissertation data.** A future human pilot requires a new approved data design.
13. **Honest trust boundary.** Separate local directories and typed readers prevent accidental scientific leakage; they are not adversarial isolation.

## 3. Data Classification

| Class | Examples | Allowed locations | Prohibited destinations |
|---|---|---|---|
| Repository-controlled public | Task fixtures, benchmark public case views, evidence probes, split manifests, schemas, prompt templates, limitations card | Git repository | Secret stores and human-data stores |
| Repository-controlled evaluator | Criterion probes, oracle tests, rubrics, label rationale, reviewer/adjudication records | Git benchmark criterion/review roots; evaluator reader | Policies, trackers, graph state, public case views, simulation training |
| Benchmark decision-safe synthetic | Public/evidence responses, execution summaries, tracker predictions, policy actions, commit hashes | Benchmark decision Parquet and manifests | Criterion fields, simulation truth, interactive checkpoints |
| Benchmark evaluator-only synthetic | Criterion responses/code, execution, demonstrated-performance labels, reveal metadata | Criterion Parquet and evaluator commands | Trackers, policies, graph/API, RL training, raw W&B tables |
| Policy-safe synthetic | Observations, evidence, tracker summaries, actions, rewards, terminations | Public Parquet, selected W&B summaries | Interactive prompt payloads unless needed |
| Privileged synthetic | True mastery, misconceptions, true `B_t`/`F_t`, transition draws, simulator parameters | Privileged Parquet and controlled evaluator | Policies, React demo API, OpenRouter, W&B Tables |
| Local demo content | Submitted explanation, generated prompt, evidence, tracker transition | Local JSONL and SQLite checkpoint | W&B and OpenRouter except minimum prompt context |
| Operational metadata | Latency, token counts, cost, provider/model IDs, errors | JSONL, manifests, metrics files | Source control when generated |
| Secret | OpenRouter and W&B keys | Environment/host secret mechanism | Logs, configs, JSONL, Parquet, browser, Git |
| Prohibited | Hidden model reasoning, credentials in prompts, unrestricted sandbox contents | None | Every store |
| Future human restricted | Consent, identity, study responses, withdrawal records | Not defined in this POC | Current SQLite, JSONL, Parquet, W&B by default |

`criterion_demonstrated_mastery` means performance on one held-out executable probe; it is not latent human mastery. `true_bandwidth` and `true_friction` are privileged simulator state. `estimated_bandwidth` and `estimated_friction` are policy-safe tracker summaries only when named as estimates and produced without reading simulator truth.

A reviewer using the demo could type personal information. The interface shall warn against this, logs stay local, and demo events are excluded from dissertation datasets unless a separately approved human-study protocol says otherwise.

## 4. Storage Map

The paths below are relative to the repository's `development/` workspace. Root-level `docs/` and `proposal/` contain no runtime or experiment data.

```text
.local/                              # gitignored runtime state
|-- demo/
|   |-- demo.sqlite                 # LangGraph provider-managed checkpoints
|   |-- events.jsonl                # committed demo turns
|   |-- events.index.json           # rebuildable idempotency index
|   `-- quarantine/                 # partial/corrupt final lines
`-- cache/                          # disposable provider/cache files

artifacts/                           # generated research artifacts
|-- benchmark/version={version}/run={run_id}/
|   |-- resolved_config.yaml
|   |-- run_manifest.json
|   |-- decision/
|   |   |-- generated_public.parquet
|   |   |-- generated_evidence.parquet
|   |   |-- evidence_executions.parquet
|   |   |-- sample_status.parquet
|   |   |-- condition_predictions.parquet
|   |   |-- condition_commit_sets.parquet
|   |   `-- commitment_manifest.json
|   |-- criterion/
|   |   |-- generated_criterion.parquet
|   |   |-- criterion_executions.parquet
|   |   `-- criterion_records.parquet
|   |-- analysis/
|   |   |-- condition_scores.parquet
|   |   |-- paired_case_metrics.parquet
|   |   |-- aggregate_metrics.json
|   |   `-- exclusions.parquet
|   |-- dataset_manifest.json
|   `-- run_status.json
|-- simulation/run={run_id}/
|   |-- resolved_config.yaml
|   |-- manifest.json
|   |-- trajectories/
|   |   |-- public/
|   |   |   `-- split={split}/condition={condition}/seed={seed}/part-000.parquet
|   |   `-- privileged/
|   |       `-- split={split}/condition={condition}/seed={seed}/part-000.parquet
|   |-- dataset_manifest.json
|   |-- policies/{policy_sha256}.npz
|   |-- calibration/{artifact_sha256}.json
|   |-- metrics.json
|   `-- run_status.json
`-- reports/{report_id}/
    |-- report_manifest.json
    |-- tables/
    `-- figures/

data/                                # versioned authored input data
|-- tasks/
|-- profiles/
|-- splits/
|-- benchmark/v1/
|   |-- manifest.yaml
|   |-- cases/
|   |-- public/
|   |-- evidence/
|   |-- criterion/
|   |-- tests/
|   |-- reviews/
|   |-- splits/
|   `-- limitations.md
`-- schemas/
```

Decision and criterion roots never share a row schema. `commitment_manifest.json` implements `benchmark.decision_phase_manifest.v1` and is atomically published before any criterion file is created. Small datasets may use one Parquet file; larger repeated-sample runs may partition by model route or case family without changing logical keys.

## 5. Level-1 Data Flow

```d2
direction: right

people: "People" {
  demo_user: "Demo User"
  researcher: "Researcher"
  evaluator: "Dissertation Evaluator"
  reviewer: "Independent Benchmark Reviewer"
}

interactive: "Interactive POC" {
  react: "React Client"
  api: "FastAPI"
  graph: "LangGraph Turn"
  domain: "Evidence / Tracker / Policy / Renderer"
}

benchmark: "Primary Benchmark" {
  authoring: "Case and Review Tools"
  decision: "Decision Phase"
  gate: "Commitment Gate"
  criterion: "Evaluator Criterion Phase"
  paired: "Paired Case Analysis"
}

experiments: "Secondary Experiments" {
  hydra: "Hydra Configuration"
  runner: "Gymnasium Runner"
  trainer: "Bandit / Q Trainer"
  analysis: "Secondary Analysis"
}

local: "Canonical Local Data" {
  fixtures: "Versioned Tasks and Profiles"
  benchmark_defs: "Benchmark Cases, Probes, Tests, Reviews, Splits"
  sqlite: "SQLite Checkpoints"
  jsonl: "Demo Turn JSONL"
  decisions: "Decision-Safe Benchmark Records"
  criterion: "Evaluator-Only Criterion Records"
  paired: "Paired Benchmark Metrics"
  public: "Policy-Safe Parquet"
  truth: "Privileged Truth Parquet"
  artifacts: "Configs, Policies, Metrics, Reports"
}

optional: "Optional External Services" {
  openrouter: "OpenRouter"
  wandb: "Weights & Biases"
  sandbox: "External Sandbox"
}

people.demo_user -> interactive.react: "text explanation"
interactive.react -> interactive.api: "typed HTTP"
interactive.api -> interactive.graph: "session command"
local.fixtures -> interactive.graph: "task and templates"
interactive.graph -> interactive.domain: "one bounded turn"
interactive.graph -> local.sqlite: "checkpoint state"
interactive.graph -> local.jsonl: "committed event"
interactive.api -> interactive.react: "prompt and inspectable estimates"

people.researcher -> benchmark.authoring: "author and validate cases"
people.reviewer -> local.benchmark_defs: "review labels and probe separation"
local.benchmark_defs -> benchmark.authoring: "versioned definitions"
benchmark.authoring -> benchmark.decision: "frozen public case views"
benchmark.decision -> local.decisions: "responses, predictions, actions"
local.decisions -> benchmark.gate: "complete-set hashes"
benchmark.gate -> benchmark.criterion: "capability after global seal"
local.benchmark_defs -> benchmark.criterion: "private criterion specs"
benchmark.criterion -> local.criterion: "responses, executions, labels"
local.decisions -> benchmark.paired: "sealed predictions"
local.criterion -> benchmark.paired: "common criterion per case/sample"
benchmark.paired -> local.paired: "case scores and intervals"
people.evaluator -> local.paired: "primary result and provenance"

people.researcher -> experiments.hydra: "select versioned run config"
local.fixtures -> experiments.runner: "task/profile/split inputs"
experiments.hydra -> experiments.runner: "resolved config and seeds"
experiments.runner -> experiments.trainer: "policy-safe transitions"
experiments.runner -> local.public: "public turn rows"
experiments.runner -> local.truth: "separate simulator truth"
experiments.trainer -> local.artifacts: "policy artifact"
local.public -> experiments.analysis: "standard metrics"
local.truth -> experiments.analysis: "declared truth-based metrics"
local.artifacts -> experiments.analysis: "frozen provenance"
experiments.analysis -> local.artifacts: "metrics, tables, and figures"
people.evaluator -> local.artifacts: "reproduction package"

interactive.graph -> optional.openrouter: "minimum optional render context"
interactive.graph -> optional.sandbox: "optional code and limits only"
benchmark.decision -> optional.openrouter: "public and evidence channels"
benchmark.criterion -> optional.openrouter: "fresh criterion channel"
benchmark.decision -> optional.sandbox: "generated evidence code"
benchmark.criterion -> optional.sandbox: "generated criterion code"
experiments.runner -> optional.wandb: "metrics and approved references"

local.criterion -> benchmark.decision: "FORBIDDEN" {
  style.stroke: "#c62828"
  style.stroke-dash: 4
}
local.criterion -> experiments.trainer: "FORBIDDEN" {
  style.stroke: "#c62828"
  style.stroke-dash: 4
}
local.truth -> interactive.domain: "FORBIDDEN" {
  style.stroke: "#c62828"
  style.stroke-dash: 4
}
local.truth -> optional.wandb: "FORBIDDEN raw rows" {
  style.stroke: "#c62828"
  style.stroke-dash: 4
}
```

### 5.1 Boundary Rules

- Interactive demo events are not training trajectories.
- Benchmark decision readers cannot open criterion roots or deserialize criterion columns.
- Every expected case/sample is marked complete, pre-criterion missing, or invalid before the decision phase is sealed.
- Criterion capabilities are minted only for samples with complete condition commit sets.
- Criterion records reference the decision-phase seal and have a later reveal timestamp.
- The same criterion record scores every condition for one case/sample.
- Repeated LLM samples remain nested under one case and model route.
- Gymnasium constructs public and privileged rows from one transition but writes them through separate typed builders.
- Policies receive only the `PolicyObservation` object, not a raw row or `info` dictionary.
- Trainers may read public observation, action, reward-target, and next-observation columns; they cannot mount or open the privileged root.
- Evaluators must declare whether a metric uses public or privileged inputs.
- W&B receives aggregate synthetic metrics, resolved config, approved policy artifacts, and selected policy-safe examples only.
- OpenRouter receives a field-allowlisted fresh context for the public, evidence, criterion, or tutor-rendering channel.

## 6. Interactive Data Flow

### 6.1 Session Creation

1. FastAPI validates `task_id` and loads a versioned task fixture.
2. The session service generates `session_id`; the LangGraph `thread_id` is the same opaque value.
3. The configured tracker produces its initial state and the policy version is frozen for the session.
4. LangGraph checkpoints the initial state to SQLite.
5. FastAPI returns the task, initial prompt, session ID, and turn index `0`.

No JSONL turn event is written until a learner response produces a committed turn. Optional session lifecycle logs are ordinary application logs, not canonical tutoring events.

### 6.2 Turn Commit Flow

```d2
shape: sequence_diagram

client: "React Client"
api: "FastAPI"
reader: "Event Index / Reader"
graph: "LangGraph Turn"
checkpoint: "SQLite Checkpointer"
writer: "JSONL Writer"

client -> api: "turn_id, expected index, response"
api -> reader: "lookup(session_id, turn_id)"
reader -> api: "existing event or miss"
api -> graph: "invoke one turn on miss"
graph -> checkpoint: "intermediate node checkpoints"
graph -> writer: "append_once complete TurnEventV1"
writer -> writer: "flush, fsync, update hash chain"
writer -> graph: "event hash"
graph -> checkpoint: "final committed state"
checkpoint -> graph: "checkpoint id"
graph -> api: "committed response projection"
api -> client: "200 response"
```

### 6.3 Commit States and Recovery

| SQLite state | JSONL state | Interpretation | Recovery |
|---|---|---|---|
| No pending turn | No event | Turn not accepted | Client may submit |
| Intermediate checkpoint | No event | Uncommitted computation | Resume or restart same turn ID |
| Intermediate checkpoint | Complete event | Event committed; final checkpoint lost | Rebuild/promote state from event, do not recompute |
| Final checkpoint | Complete event | Fully committed | Return stored projection on retry |
| Final checkpoint | No event | Integrity violation | Mark session degraded; do not acknowledge until repaired |
| Any | Partial final line | Process interrupted during append | Quarantine partial bytes and retain prior complete lines |

The JSONL key `(session_id, turn_id)` is the logical commit/idempotency record. `events.index.json` is only a rebuildable performance cache and is never authoritative.

## 7. JSONL Event Schema

### 7.1 `TurnEventV1`

Each line is one compact UTF-8 JSON object:

| Field | Type | Required | Notes |
|---|---|---:|---|
| `schema_name` | string | Yes | `demo.turn_event` |
| `schema_version` | integer | Yes | `1` |
| `event_id` | UUID string | Yes | Unique event identity |
| `session_id` | UUID string | Yes | Also LangGraph thread ID |
| `turn_id` | UUID string | Yes | Idempotency key within session |
| `turn_index` | integer | Yes | Non-negative, contiguous per committed session |
| `event_time_utc` | RFC 3339 string | Yes | Presentation/audit only; not deterministic replay input |
| `task` | object | Yes | ID, version, target concept |
| `input` | object | Yes | Raw local response and normalized response hash |
| `evidence` | object | Yes | `ObservationEvidence` including extractor version |
| `tracker_before` | object | Yes | Policy-safe `TrackerState` |
| `tracker_after` | object | Yes | Policy-safe `TrackerState` |
| `policy_observation` | object | Yes | Exact allowlisted object supplied to policy |
| `policy_decision` | object | Yes | Directive, target, version, propensity, fallback flag |
| `generation` | object | Yes | Visible prompt, renderer, hashes, model metadata, usage |
| `guardrail` | object | Yes | Accepted result, reason codes, version |
| `degraded_reasons` | array[string] | Yes | Empty when normal |
| `versions` | object | Yes | Graph, task, prompt, tracker, policy, schema versions |
| `timing_ms` | object | Yes | Named local component durations |
| `previous_event_hash` | SHA-256 string/null | Yes | Null for first committed turn |
| `event_hash` | SHA-256 string | Yes | Hash of canonical payload excluding this field |

No raw hidden reasoning, credentials, simulator truth, particles, or rejected model candidate is allowed.

### 7.2 JSON Validation Rules

- All bounded values are finite and in `[0,1]`.
- `(session_id, turn_id)` and `(session_id, turn_index)` are unique.
- `tracker_before` equals the previous committed `tracker_after` for the session.
- `policy_observation.tracker` equals `tracker_after` for the current evidence update.
- Directive and propensity obey the policy contract.
- A template fallback records the failure reason but not secret provider payloads.
- The visible prompt equals the accepted/fallback generation output.
- Hash-chain verification succeeds from the session's first event.

Hashing uses a versioned canonicalizer: UTF-8 JSON, sorted object keys, no insignificant whitespace, and `event_hash` omitted. Changing canonicalization increments `hash_format_version` even if the event schema is unchanged.

### 7.3 Demo Retention

Demo JSONL can contain freely typed text and is therefore treated cautiously even though it is not research data:

- Keep it local and gitignored.
- Do not upload it to W&B.
- Provide a command to delete one session or reset all demo data.
- Purge demonstration events before distributing a reproduction archive.
- Do not reuse these events as pilot observations.

## 8. SQLite Checkpoint Schema

SQLite is used only through LangGraph's supported checkpointer. Table names and columns are provider-managed and may vary by installed version. The following is a conceptual ownership diagram, not an application migration contract.

```d2
direction: right

application: "Application-Owned Identifiers" {
  session: "Session Reference" {
    shape: sql_table
    session_id: "TEXT logical PK"
    thread_id: "same opaque value"
    task_version: "in graph state"
    policy_version: "in graph state"
  }
}

sqlite: "demo.sqlite - LangGraph Managed" {
  checkpoints: "Checkpoint Records (conceptual)" {
    shape: sql_table
    thread_id: "TEXT key"
    checkpoint_namespace: "TEXT key"
    checkpoint_id: "TEXT key"
    parent_checkpoint_id: "TEXT nullable"
    checkpoint_payload: "BLOB / JSON"
    metadata_payload: "BLOB / JSON"
  }

  pending_writes: "Pending Channel Writes (conceptual)" {
    shape: sql_table
    thread_id: "TEXT key"
    checkpoint_id: "TEXT key"
    task_id: "TEXT key"
    channel: "TEXT"
    value: "BLOB"
  }

  provider_meta: "Provider Migrations (conceptual)" {
    shape: sql_table
    version: "INTEGER"
  }
}

external: "External Commit Evidence" {
  events: "events.jsonl" {
    shape: sql_table
    session_id: "UUID key"
    turn_id: "UUID key"
    turn_index: "INTEGER"
    event_hash: "SHA-256"
  }
}

application.session -> sqlite.checkpoints: "thread_id"
sqlite.checkpoints -> sqlite.pending_writes: "thread and checkpoint"
sqlite.provider_meta -> sqlite.checkpoints: "provider-managed schema"
application.session -> external.events: "session_id"
sqlite.checkpoints -> external.events: "committed turn IDs in graph state"
```

### 8.1 SQLite Rules

- Application code does not create foreign keys into provider tables.
- Application migrations do not alter provider tables.
- The exact installed LangGraph/checkpointer version is locked by Pixi.
- Tests may use an in-memory checkpointer; the demo uses a file-backed SQLite checkpointer.
- Graph state stores task IDs/versions, tracker summaries, policy version, committed turn IDs, and visible prompts.
- Graph state excludes secrets, simulator truth, particle arrays, raw provider reasoning, and full experiment trajectories.
- Copying `demo.sqlite` while the process writes is not a backup protocol; stop the process or use SQLite's supported backup operation.

No additional application database is required for the first POC. Session lookup uses `session_id == thread_id`, and turn idempotency uses JSONL. Add an application table only if measured lookup/recovery problems justify it.

## 9. Benchmark Data Flow

### 9.1 Decision-Before-Criterion Publication

```d2
shape: sequence_diagram

command: "Benchmark Command"
validator: "Benchmark Validator"
definitions: "Frozen Definition Store"
decision: "Decision Runner"
decision_writer: "Decision-Safe Writer"
commitment: "Commitment Store"
phase_seal: "Decision-Phase Seal"
criterion_reader: "Evaluator Criterion Reader"
criterion_writer: "Criterion Writer"
model: "Pinned OpenRouter Route"
sandbox: "External Sandbox"
scorer: "Paired Scorer"
analysis: "DuckDB Analysis"

command -> validator: "version, hashes, reviews, splits, metric plan"
validator -> definitions: "validate public and evaluator inventories"
definitions -> validator: "frozen projected manifests"
validator -> decision: "public case views only"
decision -> model: "fresh public and evidence requests"
model -> decision: "recorded final responses"
decision -> sandbox: "execute generated evidence code"
sandbox -> decision: "normalized evidence execution"
decision -> decision_writer: "public/evidence records and executions"
decision -> commitment: "condition predictions and actions"
commitment -> commitment: "seal complete condition sets"
decision -> decision_writer: "terminal pre-criterion missing records"
decision_writer -> phase_seal: "all planned samples accounted for"
commitment -> phase_seal: "eligible commit-set hashes"
phase_seal -> phase_seal: "atomically publish global seal"
phase_seal -> criterion_reader: "construct evaluator capability"
definitions -> criterion_reader: "private criterion specs"
criterion_reader -> model: "fresh criterion-only request"
model -> criterion_reader: "recorded final response"
criterion_reader -> sandbox: "execute generated criterion code"
sandbox -> criterion_reader: "normalized execution"
criterion_reader -> criterion_writer: "criterion response, execution, label"
commitment -> scorer: "sealed condition predictions"
criterion_writer -> scorer: "one criterion per eligible sample"
scorer -> analysis: "condition and paired case records"
analysis -> command: "primary effect, controls, intervals, missingness"
```

The decision phase is complete only when every planned `(case_id, sample_id, model_route_id)` is represented by either:

- One sealed set containing exactly the four expected condition predictions; or
- One terminal `precriterion_missing` record with a frozen reason and no criterion capability.

Only the first state is criterion-eligible. This permits honest missingness reporting without revealing a criterion for an incomplete pair.

### 9.2 Benchmark Run State Machine

| State | Meaning | Writable roots | May be scored? |
|---|---|---|---:|
| `created` | Config and run manifest saved | Manifest only | No |
| `decision_running` | Public/evidence generation and condition decisions in progress | Decision temporary files | No |
| `decision_validating` | Planned-sample accounting, condition, hash, and timestamp checks | Decision temporary files | No |
| `decision_sealed` | Global decision manifest atomically published | Criterion temporary files only | No |
| `criterion_running` | Eligible criterion responses executing after seal | Criterion temporary files | No |
| `evaluating` | Criterion and decision datasets immutable; scores being built | Analysis temporary files | No final claim |
| `complete` | Schemas, hashes, timing, denominators, and reports validate | No source-row mutation | Yes |
| `failed` | Integrity or execution failure not represented as declared missingness | Diagnostics only | No |
| `superseded` | Replaced before final freeze | None | No final claim |
| `frozen` | Immutable run selected for dissertation reporting | None | Yes |

Publishing `commitment_manifest.json` is irreversible for that run ID. A retry after sealing creates missing criterion records or resumes criterion execution; it cannot rewrite predictions. Any correction to a frozen benchmark definition creates a new benchmark version and run ID.

## 10. Authored Benchmark Schemas

### 10.1 Frozen Manifest

`benchmark.manifest.v1` contains:

| Group | Required fields |
|---|---|
| Identity | `benchmark_version`, `status=frozen`, `frozen_at_utc`, optional `parent_version` |
| Matrix | Case IDs, four concepts, two misconception states per concept, three transfer cases per pair, evidence-pattern strata |
| Conditions | Exact ordered semantic set: dialogue-only, probe-informed, unrelated-probe, corrupted-probe |
| Splits | Task-family membership for development, calibration, policy-selection, and held-out roles |
| Files | Relative path, artifact class, schema, SHA-256, and byte size for every case, probe, test, rubric, review, prompt, control, and limitation file |
| Primary metric | Paired Brier direction, case aggregation, interval method, exclusion/missingness rules |
| Secondary metrics | False-mastery, precision/recall, unsafe advancement, confidence, disagreement, controls |
| Generation | Required pinned model/provider route, sampling config, repeats, retry policy, budgets |
| Review | Author, independent reviewer, adjudication status, review-record hashes |

The checked-in root manifest is validated by an evaluator command. It produces two content-addressed projections:

- `benchmark.public_manifest.v1`: case metadata, public/evidence references, control mappings, split, and hashes; no criterion path, probe ID, response, result, rubric, or label.
- `benchmark.evaluator_manifest.v1`: criterion inventory and review provenance, available only to evaluator commands after the decision seal.

### 10.2 Public Case Schema

`benchmark.case_public.v1`:

| Column | Type | Constraint |
|---|---|---|
| `benchmark_version`, `case_id` | string | Composite unique identity |
| `task_family`, `split` | dictionary string | Split is task-family derived |
| `target_concept`, `misconception_id` | string | Declared stratum |
| `transfer_case_id`, `evidence_pattern` | string | Pattern covers required matrix |
| `public_fixture_ref`, `evidence_probe_ref` | content-addressed string | Must resolve in public manifest |
| `initial_tracker_ref`, `initial_tracker_hash` | string | Frozen policy-safe state |
| `unrelated_control_ref`, `corruption_spec_ref` | string | Frozen before held-out run |
| `case_content_hash` | SHA-256 string | Canonical case projection |

Forbidden fields include every `criterion_*`, `true_*`, reward, simulator parameter, held-out outcome, or model-generated label column.

### 10.3 Evaluator Criterion Specification

`benchmark.criterion_spec.v1` is a checked-in evaluator artifact:

| Field | Type | Notes |
|---|---|---|
| `benchmark_version`, `case_id` | string | Joins only inside evaluator commands |
| `criterion_probe_id` | string | Never present in the public case schema |
| `criterion_prompt_ref` | content-addressed string | Structurally different transfer probe |
| `test_bundle_ref`, `test_bundle_sha256` | string | Trusted authored oracle |
| `rubric_ref`, `rubric_sha256` | string | Deterministic interpretation |
| `label_rationale_ref` | string | Explains demonstrated-performance mapping |
| `review_record_ref` | string | Independent sign-off |
| `structural_difference_record_ref` | string | Evidence/criterion comparison |

Criterion specifications contain no tracker-specific target, policy-specific reward, CBFM value, or simulator latent state.

### 10.4 Review and Split Records

`benchmark.review.v1` records author, independent reviewer, timestamps, decision, concerns, adjudication, and hashes of exactly reviewed files. Review records are append-only; a changed reviewed file invalidates sign-off.

`benchmark.split.v1` maps task-family IDs, never response rows, to roles. The validator rejects shared task templates, probe templates, or declared surface variants across forbidden partitions.

### 10.5 Benchmark Run Manifest

`benchmark.run_manifest.v1` links one execution to:

- Run ID, purpose, creation time, Git commit/dirty flag, Pixi lock hash, and resolved Hydra config hash.
- Frozen benchmark version/root hash plus public/evaluator manifest hashes.
- Planned case/sample/model-route key hash and exact condition set.
- Tracker, policy, evidence extractor, action mapping, metric, control, prompt, and schema versions.
- Pinned model/provider route, sampling values, retry policy, token/cost budget, and channel-specific seed plan.
- Decision, criterion, analysis, status, and report roots.

The decision runner receives a projected run configuration that includes the evaluator-manifest hash but no criterion path or content. The evaluator receives its path only after verifying the decision-phase seal. Secrets are absent from both persisted projections.

## 11. Decision-Phase Schemas

### 11.1 Channel Records

`benchmark.channel_record.v1` stores only public and evidence-side generated/replayed outputs:

| Column | Type | Constraint |
|---|---|---|
| `benchmark_version`, `run_id`, `case_id`, `sample_id` | string | Logical sample identity |
| `model_route_id` | string | Pinned model/provider condition |
| `channel` | dictionary string | `public`, `evidence`, `unrelated`, or `corrupted`; never criterion |
| `source` | dictionary string | Authored, OpenRouter, or recorded replay |
| `request_hash`, `response_hash` | SHA-256 string | Channel-specific canonical hashes |
| `final_response` | string | Final text/code only; no hidden reasoning |
| `provider_request_id`, `generation_id` | string nullable | Operational provenance |
| `model_id`, `provider_id`, `sampling_hash` | string nullable | Must match run condition |
| `input_tokens`, `output_tokens`, `cost_usd_micros`, `latency_ms` | integer nullable | Operational values |
| `status`, `missing_reason` | string | Explicit success/failure |

A channel-specific cache key is `(channel, request_hash, model_route_id)`; a record cannot be reused across channels.

### 11.2 Evidence Execution

`benchmark.evidence_execution.v1`:

| Column | Type | Notes |
|---|---|---|
| Sample identity plus `channel` | strings | Must reference a non-criterion channel record |
| `operation_id`, `sandbox_provider` | string | External execution identity |
| `test_bundle_sha256` | SHA-256 string | Evidence oracle only |
| `passed`, `failed`, `exit_code` | integer nullable | Normalized result |
| `timed_out`, `resource_limited` | bool | Failure classification |
| `stdout_hash`, `stderr_hash` | SHA-256 string | Raw output is not required |
| `status`, `missing_reason` | string | Missing is not incorrect |

Only manifest-hashed authored fixtures may have `sandbox_provider=trusted_local`. Generated code requires an external provider.

### 11.3 Sample Accounting

`benchmark.decision_sample_status.v1` has one row per planned sample:

| Column | Type | Constraint |
|---|---|---|
| Benchmark/run/case/sample/model identity | strings | Unique planned sample |
| `status` | dictionary string | `complete`, `precriterion_missing`, or `invalid` |
| `reason_code` | string nullable | Required unless complete |
| `expected_condition_count`, `committed_condition_count` | int8 | Expected value is four |
| `initial_state_hash` | SHA-256 nullable | Required when complete |
| `recorded_at_utc` | timestamp UTC | Must precede phase seal |

`invalid` is an integrity failure and blocks publication. `precriterion_missing` is a declared provider/sandbox failure and contributes to denominators but receives no criterion.

### 11.4 Condition Prediction

`benchmark.condition_prediction.v1`:

| Column | Type | Role |
|---|---|---|
| Benchmark/run/case/sample/model identity | strings | Composite key prefix |
| `condition` | dictionary string | Fourth key; exactly one of four frozen conditions |
| `initial_state_hash` | SHA-256 | Equal across all four conditions |
| `public_record_hash`, `additional_evidence_hash` | SHA-256 nullable | Decision provenance |
| `tracker_mastery_probability` | float64 | Primary prediction in `[0,1]` |
| `tracker_uncertainty`, `tracker_misconception_probability` | float64 | Secondary predictions in `[0,1]` |
| `directive`, `policy_target_concept` | string | Committed policy action |
| `policy_propensity` | float64 nullable | Required for stochastic policies |
| `tracker_version`, `policy_version`, `observation_schema` | string | Component provenance |
| `input_hash`, `record_hash` | SHA-256 | Canonical integrity |
| `committed_at_utc` | timestamp UTC | Must precede criterion reveal |

This schema has no criterion, reward, or simulator-truth columns.

### 11.5 Commit Sets and Decision Seal

`benchmark.condition_commit_set.v1` has one row per complete sample:

- Planned sample key.
- Exact sorted condition names and prediction-record hashes.
- Shared initial-state hash.
- Tracker/policy versions.
- Sealed timestamp and `condition_seal_hash`.

`benchmark.decision_phase_manifest.v1` contains:

- Run and benchmark identity.
- Expected planned-sample key hash.
- Sorted complete commit-set hashes.
- Sorted pre-criterion-missing sample hashes.
- Counts by status, task family, condition, model route, and evidence pattern.
- Dataset hashes for channel, execution, status, prediction, and commit-set files.
- `decision_phase_sealed_at_utc`, schema fingerprints, and global `decision_phase_seal_hash`.

The global seal is written with temporary-file plus atomic rename. Criterion writers require its hash at construction and cannot write into the decision root.

## 12. Criterion and Score Schemas

### 12.1 Criterion Channel and Execution

`benchmark.criterion_channel_record.v1` mirrors provider metadata from the decision channel schema but:

- Uses only `channel=criterion`.
- Lives under the evaluator-only criterion root.
- References an eligible `condition_seal_hash` and the global decision-phase seal.
- Uses a fresh request hash and no public/evidence message history.

`benchmark.criterion_execution.v1` records test counts, exit status, timeout/resource flags, output hashes, external sandbox operation, test-bundle hash, and explicit missingness.

### 12.2 Criterion Record

`benchmark.criterion_record.v1` has one row per eligible case/sample/model route:

| Column | Type | Constraint |
|---|---|---|
| Benchmark/run/case/sample/model identity | strings | Unique; no condition column |
| `criterion_probe_id` | string | Evaluator-only provenance |
| `condition_seal_hash`, `decision_phase_seal_hash` | SHA-256 | Must resolve to prior sealed records |
| `criterion_request_hash`, `criterion_response_hash` | SHA-256 nullable | Generation provenance |
| `criterion_execution_hash` | SHA-256 nullable | Execution provenance |
| `criterion_demonstrated_mastery` | bool nullable | Held-out demonstrated performance |
| `criterion_status`, `missing_reason` | string | Null label requires reason |
| `label_rationale_hash`, `review_record_hash` | SHA-256 | Frozen interpretation provenance |
| `revealed_at_utc` | timestamp UTC | Strictly later than both seals |
| `criterion_record_hash` | SHA-256 | Canonical row hash |

The label is neither a simulator latent state nor a claim about general human mastery.

Allowed criterion statuses are `valid`, `missing`, and `invalid`. `valid` requires a non-null label and validated execution/rubric mapping. `missing` requires a null label and declared provider/sandbox reason. `invalid` denotes an integrity failure and blocks run publication.

### 12.3 Condition Scores

`benchmark.condition_score.v1` has one row per condition with a valid criterion:

| Column | Type | Notes |
|---|---|---|
| Sample identity plus `condition` | strings | Joins one prediction to common criterion |
| `prediction_record_hash`, `criterion_record_hash` | SHA-256 | Input lineage |
| `brier_score` | float64 | `(p - y)^2` |
| `predicted_mastery_at_threshold` | bool | Threshold frozen in metric spec |
| `false_mastery_acceptance` | bool | Predicted mastery with failed criterion |
| `unsafe_advancement` | bool | Advancement action with failed criterion |
| `policy_action_disagrees_with_dialogue` | bool | Secondary outcome |
| `metric_version`, `scored_at_utc` | string/timestamp | Provenance |

### 12.4 Paired Case Metrics

`benchmark.paired_case_metric.v1` has one row per case/sample/model route:

| Column | Type | Notes |
|---|---|---|
| Case/sample/model identity | strings | Nested sample key |
| `dialogue_brier`, `probe_brier` | float64 | Primary pair |
| `delta_brier` | float64 | Dialogue minus probe; positive favors probe |
| `unrelated_delta_brier`, `corrupted_delta_brier` | float64 | Specificity controls |
| `policy_action_disagreement` | bool | Dialogue versus probe action |
| `pair_complete` | bool | False rows remain in exclusions/denominators |
| `exclusion_reason` | string nullable | Frozen reason vocabulary |
| `paired_metric_hash` | SHA-256 | Canonical result row |

The confirmatory aggregate first reduces repeated samples within `(case_id, model_route_id)` under the frozen method, then compares cases. Raw repeat count is never reported as the independent case count.

## 13. Benchmark Relationship Diagram

```d2
direction: right

authored: "Authored and Reviewed" {
  manifest: "benchmark.manifest.v1" {
    shape: sql_table
    benchmark_version: "STRING PK"
    status: "frozen"
    primary_metric_hash: "SHA256"
  }
  public_case: "benchmark.case_public.v1" {
    shape: sql_table
    benchmark_version: "STRING PK/FK"
    case_id: "STRING PK"
    task_family: "STRING"
    evidence_probe_ref: "STRING"
  }
  criterion_spec: "benchmark.criterion_spec.v1" {
    shape: sql_table
    benchmark_version: "STRING PK/FK"
    case_id: "STRING PK"
    criterion_probe_id: "STRING"
    test_bundle_sha256: "SHA256"
  }
  review: "benchmark.review.v1" {
    shape: sql_table
    benchmark_version: "STRING PK/FK"
    case_id: "STRING PK"
    review_record_hash: "SHA256"
  }
}

decision: "Decision Phase" {
  sample: "decision_sample_status.v1" {
    shape: sql_table
    run_id: "STRING PK"
    case_id: "STRING PK"
    sample_id: "STRING PK"
    model_route_id: "STRING PK"
    status: "STRING"
  }
  prediction: "condition_prediction.v1" {
    shape: sql_table
    run_id: "STRING PK/FK"
    case_id: "STRING PK/FK"
    sample_id: "STRING PK/FK"
    model_route_id: "STRING PK/FK"
    condition: "STRING PK"
    mastery_probability: "FLOAT"
    committed_at: "TIMESTAMP"
  }
  commit_set: "condition_commit_set.v1" {
    shape: sql_table
    run_id: "STRING PK/FK"
    case_id: "STRING PK/FK"
    sample_id: "STRING PK/FK"
    model_route_id: "STRING PK/FK"
    condition_seal_hash: "SHA256"
  }
  phase: "decision_phase_manifest.v1" {
    shape: sql_table
    run_id: "STRING PK"
    phase_seal_hash: "SHA256"
    sealed_at: "TIMESTAMP"
  }
}

evaluator: "Evaluator Phase" {
  criterion: "criterion_record.v1" {
    shape: sql_table
    run_id: "STRING PK/FK"
    case_id: "STRING PK/FK"
    sample_id: "STRING PK/FK"
    model_route_id: "STRING PK/FK"
    demonstrated_mastery: "BOOLEAN?"
    revealed_at: "TIMESTAMP"
  }
  score: "condition_score.v1" {
    shape: sql_table
    run_id: "STRING PK/FK"
    case_id: "STRING PK/FK"
    sample_id: "STRING PK/FK"
    model_route_id: "STRING PK/FK"
    condition: "STRING PK/FK"
    brier_score: "FLOAT"
  }
  paired: "paired_case_metric.v1" {
    shape: sql_table
    run_id: "STRING PK/FK"
    case_id: "STRING PK/FK"
    sample_id: "STRING PK/FK"
    model_route_id: "STRING PK/FK"
    delta_brier: "FLOAT"
    pair_complete: "BOOLEAN"
  }
}

authored.manifest -> authored.public_case: "benchmark version"
authored.manifest -> authored.criterion_spec: "evaluator inventory"
authored.public_case -> authored.review: "reviewed case"
authored.criterion_spec -> authored.review: "reviewed separation and label"
authored.public_case -> decision.sample: "planned case"
decision.sample -> decision.prediction: "four rows when complete"
decision.prediction -> decision.commit_set: "sealed record hashes"
decision.commit_set -> decision.phase: "global accounting"
decision.phase -> evaluator.criterion: "must preexist"
authored.criterion_spec -> evaluator.criterion: "revealed after seal"
decision.prediction -> evaluator.score: "committed prediction"
evaluator.criterion -> evaluator.score: "common held-out label"
evaluator.score -> evaluator.paired: "paired conditions"

evaluator.criterion -> decision.prediction: "FORBIDDEN decision-time path" {
  style.stroke: "#c62828"
  style.stroke-dash: 4
}
```

These are file schemas rather than tables in a running database. DuckDB creates validated views over immutable Parquet and JSON manifests.

## 14. Secondary Simulation Data Flow

### 14.1 Run Publication Flow

```d2
shape: sequence_diagram

command: "Pixi Experiment Command"
hydra: "Hydra"
validator: "Manifest Validator"
runner: "Gymnasium Runner"
public_writer: "Public Parquet Writer"
truth_writer: "Privileged Parquet Writer"
artifact_store: "Local Artifact Directory"
wandb: "Optional W&B"
analysis: "DuckDB Analysis"

command -> hydra: "compose requested condition"
hydra -> validator: "fully resolved config"
validator -> artifact_store: "write config and run manifest"
validator -> runner: "approved splits, hashes, and seed plan"
runner -> public_writer: "policy-safe turn rows"
runner -> truth_writer: "simulator-truth rows"
public_writer -> artifact_store: "temporary public parts"
truth_writer -> artifact_store: "temporary privileged parts"
runner -> validator: "episode counts and termination results"
validator -> artifact_store: "checksums and dataset manifest"
artifact_store -> wandb: "aggregate metrics and approved references"
artifact_store -> analysis: "frozen complete dataset"
analysis -> artifact_store: "metrics, tables, figures, gate result"
```

### 14.2 Run State Machine

| State | Meaning | May be analyzed? |
|---|---|---:|
| `created` | Resolved config and manifest saved | No |
| `running` | At least one episode is in progress | No final analysis |
| `validating` | Rollout ended; schema/checksum checks running | No |
| `complete` | Dataset manifest and expected rows/episodes validate | Yes |
| `failed` | Execution or validation failed | Diagnostics only |
| `superseded` | Retained but replaced before freeze | No final claims |
| `frozen` | Selected immutable run used in final report | Yes |

`run_status.json` is updated atomically through write-to-temporary plus rename. Parquet part files are immutable after `complete`.

## 15. Secondary Run Manifest Schema

`manifest.json` uses schema `experiment.run_manifest.v1`:

| Group | Required fields |
|---|---|
| Identity | `run_id`, `created_at_utc`, `purpose`, `condition_id` |
| Source | Git commit, dirty flag, Pixi lock SHA-256, schema versions |
| Configuration | Resolved-config path and SHA-256, Hydra overrides |
| Inputs | Task, profile, split manifest paths/versions/hashes |
| Seeds | Root seed, derivation version, episode seed-plan hash |
| Components | Environment, transition, CBFM, tracker, policy, action-map versions |
| Generation | Template/prompt-feature version; Gymnasium training makes no model-provider calls |
| Outputs | Public root, privileged root, policy/calibration paths, expected episodes |
| Governance | Split role and `held_out_frozen` flag |

Validation occurs before the first reset:

- Referenced files exist and hashes match.
- Calibration, policy-selection, validation, and held-out IDs are disjoint.
- Held-out runs use a previously frozen policy/configuration.
- Action mapping and observation schema match the policy artifact.
- Required seeds are present and stable.
- Secrets are absent from the resolved configuration.

## 16. Public Simulation Trajectory Schema

The logical dataset is `trajectory.public.v1`. It contains policy-safe metadata, observations, actions, observable evidence, learning targets, and outcomes. A column's presence does not automatically make it a policy feature.

### 16.1 Public Column Groups

#### Identity and Provenance

| Column | Type | Role |
|---|---|---|
| `schema_version` | int16 | Metadata; constant `1` |
| `run_id` | string | Metadata |
| `condition_id` | string | Metadata |
| `episode_id` | string | Metadata |
| `turn_index` | int32 | Metadata/observation |
| `split` | dictionary string | Metadata |
| `task_id`, `task_version` | string | Metadata |
| `profile_id`, `profile_version` | string | Synthetic metadata |
| `episode_seed`, `environment_seed`, `policy_seed` | uint64 | Replay metadata |

#### Observation Before Action

| Column | Type | Role |
|---|---|---|
| `target_concept_id` | string | Observation feature |
| `tracker_mastery_before` | float64 | Observation feature `[0,1]` |
| `tracker_misconception_before` | float64 | Observation feature `[0,1]` |
| `tracker_uncertainty_before` | float64 | Observation feature `[0,1]` |
| `estimated_bandwidth_before` | float64 nullable | Optional observation feature |
| `estimated_friction_before` | float64 nullable | Optional observation feature |
| `has_cbfm_estimate` | bool | Observation mask |
| `repeated_failures_before` | int16 | Observation feature |

#### Decision and Realized Intervention

| Column | Type | Role |
|---|---|---|
| `action_index` | int8 | Action |
| `directive` | dictionary string | Action, semantic persisted value |
| `policy_target_concept` | string | Action |
| `policy_version` | string | Metadata |
| `policy_propensity` | float64 nullable | Behavior-policy metadata `(0,1]` |
| `policy_fallback_used` | bool | Diagnostic |
| `prompt_version`, `prompt_hash` | string | Intervention provenance |
| `renderer` | dictionary string | Template/OpenRouter diagnostic |
| `prompt_token_count`, `prompt_code_density`, `prompt_concept_count` | numeric | Observable intervention features |
| `normalized_prompt_load` | float64 | Observable load feature `[0,1]` |

#### Evidence and Tracker After Action

| Column | Type | Role |
|---|---|---|
| `evidence_correctness` | float64 | Observable outcome `[0,1]` |
| `evidence_confidence` | float64 | Observable outcome `[0,1]` |
| `evidence_misconception_present` | bool | Observable outcome |
| `evidence_active_concepts` | list[string] | Observable mask |
| `extractor_version` | string | Provenance |
| `tracker_mastery_after` | float64 | Next observation `[0,1]` |
| `tracker_misconception_after` | float64 | Next observation `[0,1]` |
| `tracker_uncertainty_after` | float64 | Next observation `[0,1]` |
| `estimated_bandwidth_after` | float64 nullable | Next optional observation |
| `estimated_friction_after` | float64 nullable | Next optional observation |
| `tracker_version` | string | Provenance |

#### Reward, Termination, and Cost

| Column | Type | Role |
|---|---|---|
| `reward_total` | float64 | Training target; never action-time feature |
| `reward_diagnostic`, `reward_learning`, `reward_safety`, `reward_load` | float64 | Named training/analysis targets |
| `terminated`, `truncated` | bool | Transition targets |
| `termination_reason` | dictionary string nullable | Outcome metadata |
| `latency_ms` | int32 | Operational metric |
| `input_tokens`, `output_tokens` | int32 nullable | Optional generation cost |
| `cost_usd_micros` | int64 nullable | Optional generation cost |

### 16.2 Feature Role Manifest

Each policy artifact contains exact lists:

```json
{
  "observation_columns": ["tracker_mastery_before", "..."],
  "action_column": "action_index",
  "reward_column": "reward_total",
  "next_observation_columns": ["tracker_mastery_after", "..."],
  "metadata_columns": ["episode_id", "turn_index", "..."],
  "forbidden_prefixes": ["criterion_", "true_", "simulator_", "transition_draw_"]
}
```

The dataset reader projects these lists explicitly. It never uses `SELECT *` or infers model features from numeric dtype. Reward and next-state columns are training targets, not action-time features.

## 17. Privileged Simulation Truth Schema

The logical dataset is `trajectory.privileged.v1`. It is keyed to the public dataset but never read by policy code.

| Column | Type | Notes |
|---|---|---|
| `schema_version` | int16 | Constant `1` |
| `run_id`, `episode_id` | string | Join identity |
| `turn_index` | int32 | Join identity |
| `task_id`, `profile_id`, `condition_id`, `split` | string | Controlled evaluation metadata |
| `true_mastery_before`, `true_mastery_after` | float64 | Active-concept truth `[0,1]` |
| `true_misconception_before`, `true_misconception_after` | bool | Active misconception truth |
| `true_bandwidth_before`, `true_bandwidth_after` | float64 | Simulator truth `[0,1]` |
| `true_friction_before`, `true_friction_after` | float64 | Simulator truth `[0,1]` |
| `zpd_overload`, `underchallenge` | float64 | Separate diagnostics `[0,1]` |
| `student_correct_probability` | float64 | Student-kernel probability `[0,1]` |
| `learning_probability` | float64 | Transition-kernel probability `[0,1]` |
| `transition_draw_id` | string | Replay/debug reference, not raw RNG state |
| `simulator_parameter_hash` | SHA-256 string | Frozen parameter set |
| `simulator_version` | string | Provenance |

If the final simulator tracks multiple concepts, mastery and misconception truth move to a separate nested/list schema version rather than silently changing these scalar meanings.

## 18. Simulation Dataset Relationship Diagram

```d2
direction: right

inputs: "Versioned Inputs" {
  run_manifest: "experiment.run_manifest.v1" {
    shape: sql_table
    run_id: "STRING PK"
    config_sha256: "STRING"
    split_sha256: "STRING"
    seed_plan_sha256: "STRING"
    expected_episodes: "INTEGER"
  }
}

public: "trajectory.public.v1" {
  shape: sql_table
  run_id: "STRING PK/FK"
  episode_id: "STRING PK"
  turn_index: "INTEGER PK"
  tracker_before: "allowlisted columns"
  action: "directive and propensity"
  evidence: "observable columns"
  tracker_after: "allowlisted columns"
  reward: "training target"
  done: "terminated / truncated"
}

truth: "trajectory.privileged.v1" {
  shape: sql_table
  run_id: "STRING PK/FK"
  episode_id: "STRING PK"
  turn_index: "INTEGER PK"
  true_mastery: "before / after"
  true_cbfm: "bandwidth / friction"
  transition_data: "evaluation only"
}

dataset_manifest: "experiment.dataset_manifest.v1" {
  shape: sql_table
  run_id: "STRING PK/FK"
  public_rows: "INTEGER"
  truth_rows: "INTEGER"
  public_dataset_sha256: "STRING"
  truth_dataset_sha256: "STRING"
  validation_status: "STRING"
}

policy: "policy.artifact.v1" {
  shape: sql_table
  policy_sha256: "STRING PK"
  training_run_id: "STRING FK"
  observation_schema: "STRING"
  action_map_sha256: "STRING"
  feature_roles_sha256: "STRING"
}

metrics: "evaluation.metrics.v1" {
  shape: sql_table
  evaluation_id: "STRING PK"
  run_id: "STRING FK"
  policy_sha256: "STRING FK"
  metric_source: "public / privileged"
  analysis_sha256: "STRING"
}

inputs.run_manifest -> public: "run_id"
inputs.run_manifest -> truth: "run_id"
inputs.run_manifest -> dataset_manifest: "run_id"
public -> truth: "one-to-one turn key"
public -> dataset_manifest: "validated row set"
truth -> dataset_manifest: "validated row set"
public -> policy: "training source when applicable"
policy -> metrics: "evaluated candidate"
dataset_manifest -> metrics: "frozen input"
```

The diagram is relational for clarity; these are file schemas, not tables in a running SQL database. DuckDB creates views over them at analysis time.

## 19. Dataset Publication and Validation

### 19.1 Common Write Protocol

1. Create a new `run_id` directory and save the resolved configuration and run manifest before execution.
2. Write each Parquet part with a temporary suffix in its final filesystem.
3. Validate Arrow schema, keys, finite/range constraints, manifest membership, and forbidden columns.
4. Close and atomically rename the part into an immutable final name.
5. Validate all parts as one logical dataset.
6. Compute sorted per-file hashes and one dataset hash over relative path, file hash, row count, and schema fingerprint.
7. Atomically publish the relevant phase/dataset manifest.
8. Change run status to `complete` only after all required phases and reports validate.

An interrupted run remains in its last non-complete state. Evaluators read only manifests explicitly marked `complete` or `frozen`.

### 19.2 Benchmark Checks

- Benchmark version is frozen and every authored file hash matches.
- At least 24 cases and all required strata, reviews, controls, and task-family splits validate.
- Planned sample keys are unique and each is exactly one of complete, pre-criterion missing, or invalid.
- Complete samples have exactly four unique conditions and one shared initial-state hash.
- Prediction/action records are finite, version-compatible, and timestamped before both seals.
- Invalid samples block publication; pre-criterion-missing samples have no criterion records.
- The global decision seal accounts for every planned sample and every decision dataset hash.
- No criterion file predates or omits the decision seal reference.
- Every criterion row maps to exactly one eligible commit set and has no condition column.
- Every condition score joins one committed prediction to the same criterion row.
- Brier values recompute exactly within numeric tolerance.
- Paired rows contain all four condition scores or an explicit exclusion.
- Repeat-level, case-level, and model-level denominators are reported separately.
- Public/evidence channel tables contain no criterion fields; criterion tables never enter policy/training readers.
- Recorded-response replay reproduces row hashes and aggregate metrics without network access.

### 19.3 Simulation Checks

- Unique `(run_id, episode_id, turn_index)` in each dataset.
- Exact public-to-privileged key equality for synthetic runs.
- Contiguous turn indices per episode beginning at zero.
- Exactly one terminal/truncation row per completed episode and no row afterward.
- All probabilities are finite and in `[0,1]`.
- Action index agrees with the persisted directive mapping.
- Propensity is present and valid for stochastic behavior policies.
- Reward total agrees with named components within tolerance.
- Public schema contains no criterion or privileged columns.
- Split/task/profile membership agrees with the frozen manifest.
- Expected episode and row counts agree with the seed plan.
- Simulator truth is absent from benchmark criterion and policy-feature builders.

### 19.4 Content Hashes

File-level SHA-256 detects byte changes. Dataset hashes combine sorted part paths, hashes, row counts, and schema fingerprints. Semantic replay equivalence is tested separately because Parquet metadata or writer versions can change bytes without changing rows.

## 20. DuckDB Analysis Views

Analysis commands create temporary views, not a persistent analytical database. The benchmark evaluator command is the only command that attaches both decision and criterion roots:

```sql
CREATE VIEW benchmark_predictions AS
SELECT
    benchmark_version, run_id, case_id, sample_id, model_route_id,
    condition, tracker_mastery_probability, directive,
    record_hash, committed_at_utc
FROM read_parquet(
    'artifacts/benchmark/version=*/run=*/decision/condition_predictions.parquet',
    hive_partitioning = true
);

CREATE VIEW benchmark_criteria AS
SELECT
    benchmark_version, run_id, case_id, sample_id, model_route_id,
    criterion_demonstrated_mastery, criterion_status,
    criterion_record_hash, revealed_at_utc
FROM read_parquet(
    'artifacts/benchmark/version=*/run=*/criterion/criterion_records.parquet',
    hive_partitioning = true
);
```

The primary pair can be independently recomputed:

```sql
WITH scored AS (
    SELECT
        p.run_id,
        p.case_id,
        p.sample_id,
        p.model_route_id,
        p.condition,
        POWER(
            p.tracker_mastery_probability
            - CAST(c.criterion_demonstrated_mastery AS DOUBLE),
            2
        ) AS brier
    FROM benchmark_predictions AS p
    JOIN benchmark_criteria AS c
      USING (benchmark_version, run_id, case_id, sample_id, model_route_id)
    WHERE c.criterion_status = 'valid'
),
paired AS (
    SELECT
        run_id,
        case_id,
        sample_id,
        model_route_id,
        MAX(CASE WHEN condition = 'dialogue_only' THEN brier END)
            AS dialogue_brier,
        MAX(CASE WHEN condition = 'probe_informed' THEN brier END)
            AS probe_brier
    FROM scored
    GROUP BY run_id, case_id, sample_id, model_route_id
)
SELECT
    run_id,
    case_id,
    sample_id,
    model_route_id,
    dialogue_brier - probe_brier AS delta_brier
FROM paired
WHERE dialogue_brier IS NOT NULL AND probe_brier IS NOT NULL;
```

Rules:

- Decision and training commands never attach the criterion root.
- The criterion evaluator asserts `committed_at_utc < revealed_at_utc` for every join.
- Final benchmark aggregation first handles repeats within case/model under the frozen method, then treats cases as the independent units.
- Negative-control and missingness views are materialized alongside the primary pair.
- Simulation training queries import only `artifacts/simulation/.../public`.
- Privileged simulation truth is attached only when a declared secondary metric requires it.
- Analysis selects named columns rather than `*` in final metric code.
- Every metric declares `source = benchmark_decision`, `benchmark_criterion`, `simulation_public`, `simulation_privileged`, or an explicit combination.
- Simulation public/truth joins use all turn-key fields and assert one-to-one cardinality.
- Final query text or analysis-code hash is saved in the report manifest.
- W&B exports are never used as the source for final statistics.

## 21. Artifact Lineage

```d2
direction: right

source: "Source and Inputs" {
  git: "Git Commit and Dirty Flag"
  lock: "Pixi Lock Hash"
  benchmark: "Frozen Benchmark and Review Hash"
  task_splits: "Task/Profile/Split Hashes"
  model_route: "Pinned Model Route and Prompt Hashes"
  config: "Resolved Hydra Config Hash"
}

primary: "Primary Benchmark Evidence" {
  manifest: "Benchmark Run Manifest"
  decisions: "Decision Dataset Hashes"
  phase_seal: "Decision-Phase Seal Hash"
  criteria: "Criterion Dataset Hashes"
  paired: "Paired Case Metric Hash"
}

secondary: "Secondary Simulation Evidence" {
  manifest: "Simulation Run Manifest"
  public: "Public Trajectory Hash"
  truth: "Privileged Truth Hash"
  policy: "Policy Artifact Hash"
  calibration: "Calibration Artifact Hash"
}

analysis: "Evaluation Evidence" {
  code: "Analysis Code Hash"
  primary_metrics: "Primary Effect, Controls, Missingness"
  secondary_metrics: "Secondary Metrics and Intervals"
  gate: "Retain / Reject / Inconclusive"
  report: "Tables, Figures, Dissertation Claim"
}

external: "Optional Dashboard" {
  wandb: "W&B Run ID"
}

source.git -> primary.manifest
source.lock -> primary.manifest
source.benchmark -> primary.manifest
source.model_route -> primary.manifest
source.config -> primary.manifest
primary.manifest -> primary.decisions
primary.decisions -> primary.phase_seal: "commit before reveal"
primary.phase_seal -> primary.criteria
primary.decisions -> primary.paired
primary.criteria -> primary.paired
primary.paired -> analysis.primary_metrics

source.git -> secondary.manifest
source.lock -> secondary.manifest
source.task_splits -> secondary.manifest
source.config -> secondary.manifest
secondary.manifest -> secondary.public
secondary.manifest -> secondary.truth
secondary.public -> secondary.policy: "training source"
secondary.public -> analysis.secondary_metrics
secondary.truth -> analysis.secondary_metrics: "declared truth metrics"
secondary.policy -> analysis.secondary_metrics
secondary.calibration -> analysis.secondary_metrics

analysis.code -> analysis.primary_metrics
analysis.code -> analysis.secondary_metrics
analysis.primary_metrics -> analysis.gate
analysis.secondary_metrics -> analysis.gate
analysis.gate -> analysis.report
primary.manifest -> external.wandb: "metadata and hashes"
secondary.manifest -> external.wandb: "metadata and hashes"
analysis.primary_metrics -> external.wandb: "aggregate metrics"
analysis.secondary_metrics -> external.wandb: "aggregate metrics"
external.wandb -> analysis.report: "never canonical" {
  style.stroke-dash: 4
}
```

### 21.1 Policy Artifact

`policy.artifact.v1` consists of:

- `metadata.json`: policy type/version, training config/data hashes, observation schema, feature roles, action mapping, hyperparameters, fallback policy, and evaluation status.
- `weights.npz`: NumPy arrays for bandit weights, Q-table, or visitation counts.
- `sha256`: content hash computed over canonical metadata plus array-file hash.

Heuristic policies have versioned rule/config metadata and may have no weights file.

### 21.2 Calibration Artifact

CBFM and tracker calibration artifacts include calibration split hash, feature normalizers, fitted parameters and constraints, optimization diagnostics, code/config hashes, and fit metrics. They never contain held-out outcomes.

### 21.3 Report Manifest

Every final table or figure resolves to:

- Frozen benchmark version, review, split, metric, decision-phase, criterion, and paired-dataset hashes when used.
- Frozen simulation run and dataset-manifest hashes when used.
- Pinned model/provider route, prompt, sampling, and recorded-response hashes for LLM conditions.
- Policy and calibration hashes.
- Metric definition and source classification.
- Analysis-code/config hash.
- Planned cases/samples, complete pairs, exclusions, missingness, and denominators.
- Statistical method and seed.
- Output file hash.

## 22. W&B Data Contract

W&B is an optional experiment index and dashboard.

Allowed:

- Fully resolved synthetic experiment configuration after secret redaction.
- Git/Pixi/config/benchmark/task/split/model-route hashes.
- Aggregate paired Brier effects, negative-control effects, missingness, case counts, costs, and intervals.
- Aggregate returns, calibration scores, action frequencies, and secondary intervals.
- Selected policy-safe synthetic examples.
- Approved policy, metrics, and report artifacts or local references.

Disallowed by default:

- Complete raw trajectories.
- Raw benchmark public/evidence responses or generated code by default.
- Criterion prompts, responses, executions, labels, or row-level joins.
- Privileged simulator rows.
- Demo responses.
- Future pilot responses or identity.
- API keys and provider authorization data.
- Raw hidden reasoning or rejected model content.

Every W&B run stores the corresponding local `run_id`, benchmark version where applicable, and run-manifest SHA-256. A W&B run without a resolvable complete local manifest is excluded from final reporting. W&B sweeps cannot change frozen benchmark cases, controls, labels, exclusions, splits, or primary metrics.

## 23. Leakage Controls

The local POC cannot rely on cloud IAM separation, so it layers practical controls:

1. Public benchmark projections omit criterion identifiers, paths, content, results, rubrics, and labels.
2. Criterion and simulator-private models have distinct Pydantic/Arrow builders and roots.
3. Static import tests prohibit evaluator-private imports in trackers, policies, graph, simulator, training, and public benchmark runners.
4. The decision command accepts only `benchmark.public_manifest.v1`; the criterion reader is not constructed during that phase.
5. A global decision seal accounts for every planned sample before criterion writing begins.
6. Criterion records must reference valid commit and phase seals and have later reveal timestamps.
7. Tracker/policy runtime spies record exact inputs and recursively reject `criterion_*`, `true_*`, `simulator_*`, and `transition_draw_*`.
8. Condition predictions have identical initial-state hashes and version-compatible tracker/policy metadata.
9. Public, evidence, and criterion generation use separate field allowlists, request hashes, and cache namespaces.
10. Unrelated/corrupted control assignments are frozen before held-out generation and cannot query criterion labels.
11. Task-family split validation rejects shared templates and declared variants across forbidden roles.
12. Trainer CLI accepts simulation public datasets only and cannot open benchmark criterion roots.
13. Policy feature-role manifests enumerate columns explicitly; no reader infers features from dtypes or `SELECT *`.
14. Simulator truth cannot populate benchmark labels or criterion records.
15. W&B sink applies an allowlist and receives no raw criterion rows.
16. Final reproducibility audit records all column roles, joins, exclusions, denominators, and access phases.

These controls prevent accidental and scientific leakage. They are not a claim that two directories on one researcher's laptop provide adversarial isolation.

## 24. Retention, Deletion, and Backup

| Data | Retention during dissertation | Final handling |
|---|---|---|
| Authored benchmark cases/probes/tests/reviews/splits/limitations | Immutable by benchmark version in Git | Include publishable subset and checksums; document any restricted provider terms |
| Benchmark generated responses and executions | Retain through assessment and reproducibility review | Archive with manifests; do not treat as human data |
| Benchmark decisions, criteria, and paired metrics | Retain through assessment and institutional requirement | Include canonical schemas, hashes, and permitted records |
| Authored tasks/configs/schemas | Versioned in Git | Include in reproduction package |
| Secondary synthetic trajectories | Retain through assessment and institutional requirement | Archive with checksums where permitted |
| Superseded/failed synthetic runs | Retain until final analysis freeze | Delete or archive separately from final evidence |
| Policy/calibration artifacts | Retain with source runs | Include selected artifacts and hashes |
| Demo SQLite/JSONL | Short-lived local operational data | Purge before distribution; not a dissertation dataset |
| Secrets | Never persisted in project data | Rotate/delete through provider controls |
| W&B copies | Convenience mirror | Retention follows account policy; local copy remains canonical |
| Future human data | Not collected | Requires separately approved schedule |

Backup strategy for the POC:

- Git remote for source, authored benchmark definitions/reviews, tasks, configs, and schemas.
- At least two copies of final frozen benchmark and secondary artifact bundles, each verified by manifest hashes.
- A scripted restore/reproduction check on a clean directory.
- SQLite backup only for demo continuity; losing it must not destroy scientific results.

Deletion commands must refuse paths outside configured `.local` or `artifacts` roots and support a dry run. No broad production deletion service is required.

## 25. Schema Evolution

Schemas use explicit names and integer versions:

- `demo.turn_event.v1`
- `benchmark.manifest.v1`
- `benchmark.public_manifest.v1`
- `benchmark.evaluator_manifest.v1`
- `benchmark.case_public.v1`
- `benchmark.criterion_spec.v1`
- `benchmark.review.v1`
- `benchmark.run_manifest.v1`
- `benchmark.channel_record.v1`
- `benchmark.evidence_execution.v1`
- `benchmark.decision_sample_status.v1`
- `benchmark.condition_prediction.v1`
- `benchmark.condition_commit_set.v1`
- `benchmark.decision_phase_manifest.v1`
- `benchmark.criterion_record.v1`
- `benchmark.condition_score.v1`
- `benchmark.paired_case_metric.v1`
- `experiment.run_manifest.v1`
- `experiment.dataset_manifest.v1`
- `trajectory.public.v1`
- `trajectory.privileged.v1`
- `policy.artifact.v1`
- `evaluation.metrics.v1`

Rules:

- Additive nullable fields may remain within a version only before final experiment freeze and with updated schema fingerprint.
- Removing, renaming, changing meaning/type, or changing unit requires a new schema version.
- Readers fail on unsupported future versions; they do not silently drop fields.
- Migrations read an old immutable artifact and write a new artifact with parent hash and migration-code hash.
- Final frozen artifacts are never migrated in place.
- Any semantic correction to a frozen case, probe, test, rubric, label rationale, review, control, split, or primary metric creates a new benchmark version and reruns affected conditions.
- Generated records always retain their original benchmark version; they are never relabelled onto a newer version.
- Arrow schema snapshots live in `data/schemas/` and are checked in CI.
- Action-map changes require a new policy artifact compatibility version even if the row schema is unchanged.

## 26. Future Human Pilot Boundary

This POC schema is not sufficient for a student pilot. Before recruitment, a separate approved design must define:

- Participant pseudonyms and the separately held identity link.
- Consent version, timestamp, scope, and withdrawal status.
- Inclusion/exclusion and age-related safeguards.
- Pre/post assessments and validated cognitive-load instrument data.
- Data controller/processor roles and approved regions.
- Retention, deletion, export, incident, and withdrawal workflows.
- Access roles and audit history.
- Which text may be sent to OpenRouter or any other processor.
- A frozen study build with no synthetic truth endpoint.

Current demo events must not be relabeled as pilot data after collection. Ethics and data-protection approval must precede, not follow, human research collection.

## 27. Validation Matrix

| Artifact | Required automated validation |
|---|---|
| Task/profile/split fixture | Schema, unique IDs/versions, referential integrity, split disjointness |
| Benchmark root manifest | Frozen status, all file hashes, minimum matrix, exact conditions, metric/control/model plan |
| Public benchmark projection | No criterion fields/paths, task-family splits, control references, content hashes |
| Criterion specifications | Oracle/rubric/review hashes, structural-difference record, no simulator-derived label |
| Benchmark reviews | Independent reviewer, exact reviewed hashes, resolved adjudication |
| Benchmark run manifest | Source/config/benchmark hashes, planned matrix, pinned route, seeds, budgets, output roots, no secrets |
| Channel records | Channel allowlist, request/response hashes, pinned route, no hidden reasoning, cache isolation |
| Evidence/criterion executions | Trusted-fixture allowlist or external sandbox, normalized status, explicit missingness |
| Sample status | Exact planned-key coverage; complete, pre-criterion missing, or invalid |
| Condition predictions | Four-condition uniqueness, identical state hash, finite probabilities, policy/tracker compatibility |
| Commit sets and phase seal | Prediction hashes, complete-set cardinality, global sample accounting, atomic publication |
| Criterion records | Eligible seal references, reveal ordering, one row per sample, nullable-label reason |
| Paired case metrics | Common criterion join, recomputed Brier, controls, exclusions, nested-repeat handling |
| JSONL event | Pydantic schema, unique keys, hash chain, state continuity, no prohibited fields |
| SQLite checkpoint | Restart/resume, provider-version compatibility, no private state/particles/secrets |
| Run manifest | Hashes, seeds, versions, split role, frozen-held-out rule, no secrets |
| Public Parquet | Arrow schema, finite/range checks, key uniqueness, feature allowlist, row lifecycle |
| Privileged Parquet | Arrow schema, key uniqueness, public-key parity, range checks |
| Dataset manifest | File hashes, schema fingerprints, expected rows/episodes, complete status |
| Policy artifact | Array shapes, finite values, feature/action compatibility, training data hash |
| Metrics/report | Source classification, input hashes, intervals, exclusions, output hash |
| W&B run | Local run ID/hash resolution and allowlisted payload scan |

## 28. Requirement Traceability

| Data design area | Primary PRD requirements |
|---|---|
| Interactive events and recovery | FR-VS-008, FR-GRF-002 through FR-GRF-005, FR-TRJ-001 |
| Benchmark definitions and reviews | FR-BMK-001 through FR-BMK-004, FR-BMK-007, FR-BMK-008, FR-BMK-013, FR-BMK-014, FR-BMK-016 |
| Condition pairing and criterion isolation | FR-BMK-005, FR-BMK-006, FR-BMK-015 |
| Benchmark metrics and controls | FR-BMK-009 through FR-BMK-012 |
| Experiment manifests and seeds | FR-EXP-001 through FR-EXP-006, NFR-REP-001 through NFR-REP-005 |
| Public/privileged trajectories | FR-SIM-002, FR-SIM-006, FR-TRJ-002 through FR-TRJ-004, NFR-SEC-002 |
| Policies and training targets | FR-POL-002, FR-POL-005 through FR-POL-009 |
| CBFM and tracker lineage | FR-CBFM-005 through FR-CBFM-009, FR-TRK-005 through FR-TRK-008 |
| W&B boundary | FR-TRJ-005, FR-TRJ-006, NFR-SEC-001, NFR-PRV-002 |
| Demo privacy | FR-UI-002, FR-UI-004, FR-UI-006, NFR-SEC-003, NFR-PRV-001 |
| Generated-code sandbox | SEC-SBX-001 through SEC-SBX-004, FR-BMK-003, FR-BMK-008 |

## 29. Acceptance Criteria

This data design is implementation-ready when:

- The browser path has a deterministic session/turn identity and one-event commit rule.
- JSONL/SQLite split-commit recovery is tested and does not duplicate tracker updates.
- SQLite remains provider-managed and is not promoted into a dissertation data store.
- Authored benchmark public and evaluator projections are separate and content-addressed.
- Every planned benchmark sample is accounted for before the global decision seal.
- Eligible samples have exactly four condition predictions with one shared initial-state hash.
- Criterion writers cannot be constructed before the decision seal and cannot mutate decision datasets.
- Pre-criterion-missing samples receive no criterion capability; criterion failures remain nullable labels rather than false labels.
- One evaluator-only criterion record scores every condition for a case/sample.
- Prediction/criterion timing, hashes, common-label joins, and Brier calculations are mechanically validated.
- Paired outputs report case, repeat, missingness, control, exclusion, and denominator fields without inflating the independent sample size.
- Recorded responses reproduce the benchmark report without OpenRouter, sandbox, or W&B access.
- Public policy data and simulator truth have separate schemas, roots, writers, and readers.
- Feature-role manifests prevent rewards, next state, metadata, or truth from becoming action-time features accidentally.
- Parquet publication validates keys, ranges, lifecycle, expected counts, phase ordering, and checksums before a run becomes complete.
- Every final metric declares its data classification and resolves to frozen local hashes.
- DuckDB reproduces analysis without importing W&B data.
- W&B remains optional, allowlisted, and linked back to canonical local artifacts.
- Demo responses are local, deletable, and excluded from synthetic research datasets.
- Criterion demonstrated mastery is explicitly scoped to held-out probe performance and is never represented as validated human mental state.
- Current schemas contain no human-study identity, consent, or participant lifecycle data.
- All D2 diagrams compile and every referenced PRD requirement ID resolves.
