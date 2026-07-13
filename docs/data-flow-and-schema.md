# Data Flow and Storage Schema Design

## Master's POC: Socratic Tutoring and RL Evaluation

| Field | Value |
|---|---|
| Status | Approved POC data design |
| Revision date | 2026-07-12 |
| Product requirements | `docs/PRD.md` |
| High-level architecture | `docs/HLD.md` |
| Low-level design | `docs/LLD.md` |
| Interactive checkpoint store | Local SQLite through LangGraph's supported checkpointer |
| Demo event store | Append-only JSONL |
| Canonical experiment store | Versioned local Parquet datasets and content-addressed files |
| Analysis engine | DuckDB over Parquet |
| Experiment dashboard | Optional Weights & Biases; never canonical |

> **Sequential-rescope status:** this document is aligned with the revised PRD, engineering plan, HLD, and LLD. `docs/implementation-plan.md` remains on the earlier enterprise plan until its separate revision stage.

## 1. Purpose and Scope

This document defines how data enters, moves through, persists in, and leaves the Master's POC. It covers:

- Interactive browser-turn data.
- LangGraph checkpoints and JSONL commit evidence.
- Deterministic simulation, training, and evaluation data.
- Public policy-safe and privileged simulator schemas.
- Local artifact lineage and optional W&B synchronization.
- Validation, publication, retention, deletion, and schema evolution.

This is not a production database design. There is no PostgreSQL, Redis, object-store service, MLflow registry, distributed queue, or human-study database. The design preserves only the data boundaries needed to demonstrate the tutoring loop and support defensible simulation experiments.

## 2. Data Principles

1. **Local canonical evidence.** Final claims must reproduce from repository-controlled configs plus local artifacts without W&B or network access.
2. **Separate observation from truth.** Policy-safe turns and privileged simulator truth use different schemas, directories, readers, and validation commands.
3. **Explicit column roles.** Observation features, training targets, metadata, and evaluation-only truth are classified separately.
4. **Append rather than mutate.** Demo events and final experiment datasets are immutable after publication.
5. **Checkpoint is not research data.** SQLite supports session recovery; it is not the source for dissertation analysis.
6. **One event per logical turn.** `(session_id, turn_id)` is the interactive idempotency key.
7. **Hash every scientific artifact.** Reports resolve to configs, data, policy, code, and analysis versions.
8. **No hidden reasoning.** Raw private chain-of-thought and provider-internal reasoning are never stored.
9. **Synthetic-only dissertation data.** A future human pilot requires a new approved data design, not reuse of this local layout.
10. **Honest trust boundary.** Separate local directories prevent accidental leakage; they are not equivalent to isolated production security domains.

## 3. Data Classification

| Class | Examples | Allowed locations | Prohibited destinations |
|---|---|---|---|
| Repository-controlled public | Task fixtures, profile definitions, split manifests, schemas, prompt templates | Git repository | Secret stores and raw participant stores |
| Policy-safe synthetic | Observations, evidence, tracker summaries, actions, rewards, terminations | Public Parquet, selected W&B summaries | Interactive prompt payloads unless needed |
| Privileged synthetic | True mastery, misconceptions, true `B_t`/`F_t`, transition draws, simulator parameters | Privileged Parquet and controlled evaluator | Policies, React demo API, OpenRouter, W&B Tables |
| Local demo content | Submitted explanation, generated prompt, evidence, tracker transition | Local JSONL and SQLite checkpoint | W&B and OpenRouter except minimum prompt context |
| Operational metadata | Latency, token counts, cost, provider/model IDs, errors | JSONL, manifests, metrics files | Source control when generated |
| Secret | OpenRouter and W&B keys | Environment/host secret mechanism | Logs, configs, JSONL, Parquet, browser, Git |
| Prohibited | Hidden model reasoning, credentials in prompts, unrestricted sandbox contents | None | Every store |
| Future human restricted | Consent, identity, study responses, withdrawal records | Not defined in this POC | Current SQLite, JSONL, Parquet, W&B by default |

`true_bandwidth` and `true_friction` are privileged simulator state. `estimated_bandwidth` and `estimated_friction` are policy-safe tracker summaries only when named as estimates and produced without reading simulator truth.

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
|-- runs/{run_id}/
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
`-- schemas/
```

This path refines the single-file shorthand shown in the LLD. A trajectory is a logical Parquet dataset rooted at `trajectories/public` or `trajectories/privileged`; it may contain one file for a small run and partitioned files for final matrices.

## 5. Level-1 Data Flow

```d2
direction: right

people: "People" {
  demo_user: "Demo User"
  researcher: "Researcher"
  evaluator: "Dissertation Evaluator"
}

interactive: "Interactive POC" {
  react: "React Client"
  api: "FastAPI"
  graph: "LangGraph Turn"
  domain: "Evidence / Tracker / Policy / Renderer"
}

experiments: "Offline Experiments" {
  hydra: "Hydra Configuration"
  runner: "Gymnasium Runner"
  trainer: "Bandit / Q Trainer"
  analysis: "DuckDB and Statistics"
}

local: "Canonical Local Data" {
  fixtures: "Versioned Tasks, Profiles, Splits"
  sqlite: "SQLite Checkpoints"
  jsonl: "Demo Turn JSONL"
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
experiments.runner -> optional.wandb: "metrics and approved references"

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
- Gymnasium constructs public and privileged rows from one transition but writes them through separate typed builders.
- Policies receive only the `PolicyObservation` object, not a raw row or `info` dictionary.
- Trainers may read public observation, action, reward-target, and next-observation columns; they cannot mount or open the privileged root.
- Evaluators must declare whether a metric uses public or privileged inputs.
- W&B receives aggregate synthetic metrics, resolved config, approved policy artifacts, and selected policy-safe examples only.
- OpenRouter receives only the context required to render the selected directive.

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

## 9. Offline Experiment Data Flow

### 9.1 Run Publication Flow

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

### 9.2 Run State Machine

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

## 10. Run Manifest Schema

`manifest.json` uses schema `experiment.run_manifest.v1`:

| Group | Required fields |
|---|---|
| Identity | `run_id`, `created_at_utc`, `purpose`, `condition_id` |
| Source | Git commit, dirty flag, Pixi lock SHA-256, schema versions |
| Configuration | Resolved-config path and SHA-256, Hydra overrides |
| Inputs | Task, profile, split manifest paths/versions/hashes |
| Seeds | Root seed, derivation version, episode seed-plan hash |
| Components | Environment, transition, CBFM, tracker, policy, action-map versions |
| Generation | Template/prompt version and optional pinned model/provider/sampling values |
| Outputs | Public root, privileged root, policy/calibration paths, expected episodes |
| Governance | Split role and `held_out_frozen` flag |

Validation occurs before the first reset:

- Referenced files exist and hashes match.
- Calibration, policy-selection, validation, and held-out IDs are disjoint.
- Held-out runs use a previously frozen policy/configuration.
- Action mapping and observation schema match the policy artifact.
- Required seeds are present and stable.
- Secrets are absent from the resolved configuration.

## 11. Public Trajectory Schema

The logical dataset is `trajectory.public.v1`. It contains policy-safe metadata, observations, actions, observable evidence, learning targets, and outcomes. A column's presence does not automatically make it a policy feature.

### 11.1 Public Column Groups

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

### 11.2 Feature Role Manifest

Each policy artifact contains exact lists:

```json
{
  "observation_columns": ["tracker_mastery_before", "..."],
  "action_column": "action_index",
  "reward_column": "reward_total",
  "next_observation_columns": ["tracker_mastery_after", "..."],
  "metadata_columns": ["episode_id", "turn_index", "..."],
  "forbidden_prefixes": ["true_", "simulator_", "transition_draw_"]
}
```

The dataset reader projects these lists explicitly. It never uses `SELECT *` or infers model features from numeric dtype. Reward and next-state columns are training targets, not action-time features.

## 12. Privileged Truth Schema

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

## 13. Dataset Relationship Diagram

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

## 14. Dataset Publication and Validation

### 14.1 Write Protocol

1. Create a new `run_id` directory and save resolved config/manifest before rollout.
2. Write each Parquet part with a `.tmp` suffix in the target filesystem.
3. Validate Arrow schema, row keys, finite/range constraints, and expected episode membership.
4. Close and atomically rename the part into its immutable final name.
5. Validate all parts as one logical dataset.
6. Compute sorted per-file hashes and one dataset hash over relative path, file hash, and row count.
7. Write `dataset_manifest.json` atomically.
8. Change run status to `complete` only when public and required privileged datasets both pass.

An interrupted run remains `running` or `failed`; evaluators read only `complete` or `frozen` manifests.

### 14.2 Required Checks

- Unique `(run_id, episode_id, turn_index)` in each dataset.
- Exact public-to-privileged key equality for synthetic runs.
- Contiguous turn indices per episode beginning at zero.
- Exactly one terminal/truncation row per completed episode.
- No row after termination/truncation.
- All probabilities finite and in `[0,1]`.
- Action index agrees with persisted directive mapping.
- Propensity is present and valid for stochastic behavior policies.
- Reward total agrees with named component aggregation within tolerance.
- Public schema contains no forbidden privileged columns.
- Split/task/profile membership agrees with the frozen manifest.
- Expected episode and row counts agree with the seed plan.

### 14.3 Content Hashes

File-level SHA-256 detects byte changes. Dataset hashes combine sorted part paths, hashes, row counts, and schema fingerprints. Semantic replay equivalence is tested separately because Parquet metadata or writer versions can change bytes without changing rows.

## 15. DuckDB Analysis Views

Analysis commands create temporary views, not a persistent analytical database:

```sql
CREATE VIEW public_turns AS
SELECT *
FROM read_parquet('artifacts/runs/*/trajectories/public/**/*.parquet', hive_partitioning = true);
```

Privileged data is attached only in an evaluation command whose metric declaration requires it:

```sql
CREATE VIEW privileged_truth AS
SELECT *
FROM read_parquet('artifacts/runs/*/trajectories/privileged/**/*.parquet', hive_partitioning = true);
```

Rules:

- Training queries import only the public root.
- Analysis selects named columns rather than `*` in final metric code.
- Every metric declares `source = public`, `privileged`, or `both`.
- Public/truth joins use all three turn-key fields and assert one-to-one cardinality.
- Final query text or analysis-code hash is saved in the report manifest.
- W&B exports are never used as the source for final statistics.

## 16. Artifact Lineage

```d2
direction: right

source: "Source and Inputs" {
  git: "Git Commit and Dirty Flag"
  lock: "Pixi Lock Hash"
  tasks: "Task/Profile/Split Hashes"
  config: "Resolved Hydra Config Hash"
}

run: "Run Evidence" {
  manifest: "Run Manifest"
  public: "Public Dataset Hash"
  truth: "Privileged Dataset Hash"
  policy: "Policy Artifact Hash"
  calibration: "Calibration Artifact Hash"
}

analysis: "Evaluation Evidence" {
  code: "Analysis Code Hash"
  metrics: "Metric Result and Intervals"
  gate: "Retain / Reject / Inconclusive"
  report: "Tables, Figures, Dissertation Claim"
}

external: "Optional Dashboard" {
  wandb: "W&B Run ID"
}

source.git -> run.manifest
source.lock -> run.manifest
source.tasks -> run.manifest
source.config -> run.manifest
run.manifest -> run.public
run.manifest -> run.truth
run.public -> run.policy: "training source"
run.public -> analysis.metrics: "public metrics"
run.truth -> analysis.metrics: "declared truth metrics"
run.policy -> analysis.metrics
run.calibration -> analysis.metrics
analysis.code -> analysis.metrics
analysis.metrics -> analysis.gate
analysis.gate -> analysis.report
run.manifest -> external.wandb: "metadata"
analysis.metrics -> external.wandb: "aggregate metrics"
external.wandb -> analysis.report: "never canonical" {
  style.stroke-dash: 4
}
```

### 16.1 Policy Artifact

`policy.artifact.v1` consists of:

- `metadata.json`: policy type/version, training config/data hashes, observation schema, feature roles, action mapping, hyperparameters, fallback policy, and evaluation status.
- `weights.npz`: NumPy arrays for bandit weights, Q-table, or visitation counts.
- `sha256`: content hash computed over canonical metadata plus array-file hash.

Heuristic policies have versioned rule/config metadata and may have no weights file.

### 16.2 Calibration Artifact

CBFM and tracker calibration artifacts include calibration split hash, feature normalizers, fitted parameters and constraints, optimization diagnostics, code/config hashes, and fit metrics. They never contain held-out outcomes.

### 16.3 Report Manifest

Every final table or figure resolves to:

- Frozen run and dataset-manifest hashes.
- Policy and calibration hashes.
- Metric definition and source classification.
- Analysis-code/config hash.
- Exclusions and missing pairs.
- Statistical method and seed.
- Output file hash.

## 17. W&B Data Contract

W&B is an optional experiment index and dashboard.

Allowed:

- Fully resolved synthetic experiment configuration after secret redaction.
- Git/Pixi/config/task/split hashes.
- Aggregate returns, calibration scores, action frequencies, costs, and intervals.
- Selected policy-safe synthetic examples.
- Approved policy, metrics, and report artifacts or local references.

Disallowed by default:

- Complete raw trajectories.
- Privileged simulator rows.
- Demo responses.
- Future pilot responses or identity.
- API keys and provider authorization data.
- Raw hidden reasoning or rejected model content.

Every W&B run stores the corresponding local `run_id` and manifest SHA-256. A W&B run without a resolvable complete local manifest is excluded from final reporting.

## 18. Leakage Controls

The local POC cannot rely on cloud IAM separation, so it layers practical controls:

1. Distinct Pydantic/Arrow builders for public and privileged rows.
2. Distinct dataset roots with no common row model.
3. Trainer CLI accepts `--public-dataset` only.
4. Policy feature-role manifests enumerate columns explicitly.
5. Static import test prohibits simulator-private imports in policy and graph packages.
6. Recursive schema scan rejects `true_*`, `simulator_*`, and `transition_draw_*` in public data.
7. Runtime policy spy records exact observation objects.
8. Evaluation tests compare observations across policies for parity.
9. W&B sink applies an allowlist rather than serializing arbitrary rows/config objects.
10. Final reproducibility audit records all column roles.

These controls prevent accidental and scientific leakage. They are not a claim that two directories on one researcher's laptop provide adversarial isolation.

## 19. Retention, Deletion, and Backup

| Data | Retention during dissertation | Final handling |
|---|---|---|
| Authored tasks/configs/schemas | Versioned in Git | Include in reproduction package |
| Synthetic final trajectories | Retain through assessment and institutional requirement | Archive with checksums where permitted |
| Superseded/failed synthetic runs | Retain until final analysis freeze | Delete or archive separately from final evidence |
| Policy/calibration artifacts | Retain with source runs | Include selected artifacts and hashes |
| Demo SQLite/JSONL | Short-lived local operational data | Purge before distribution; not a dissertation dataset |
| Secrets | Never persisted in project data | Rotate/delete through provider controls |
| W&B copies | Convenience mirror | Retention follows account policy; local copy remains canonical |
| Future human data | Not collected | Requires separately approved schedule |

Backup strategy for the POC:

- Git remote for source, authored data, configs, and schemas.
- At least two copies of final frozen synthetic artifact bundles, each verified by manifest hashes.
- A scripted restore/reproduction check on a clean directory.
- SQLite backup only for demo continuity; losing it must not destroy scientific results.

Deletion commands must refuse paths outside configured `.local` or `artifacts` roots and support a dry run. No broad production deletion service is required.

## 20. Schema Evolution

Schemas use explicit names and integer versions:

- `demo.turn_event.v1`
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
- Arrow schema snapshots live in `data/schemas/` and are checked in CI.
- Action-map changes require a new policy artifact compatibility version even if the row schema is unchanged.

## 21. Future Human Pilot Boundary

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

## 22. Validation Matrix

| Artifact | Required automated validation |
|---|---|
| Task/profile/split fixture | Schema, unique IDs/versions, referential integrity, split disjointness |
| JSONL event | Pydantic schema, unique keys, hash chain, state continuity, no prohibited fields |
| SQLite checkpoint | Restart/resume, provider-version compatibility, no private state/particles/secrets |
| Run manifest | Hashes, seeds, versions, split role, frozen-held-out rule, no secrets |
| Public Parquet | Arrow schema, finite/range checks, key uniqueness, feature allowlist, row lifecycle |
| Privileged Parquet | Arrow schema, key uniqueness, public-key parity, range checks |
| Dataset manifest | File hashes, schema fingerprints, expected rows/episodes, complete status |
| Policy artifact | Array shapes, finite values, feature/action compatibility, training data hash |
| Metrics/report | Source classification, input hashes, intervals, exclusions, output hash |
| W&B run | Local run ID/hash resolution and allowlisted payload scan |

## 23. Requirement Traceability

| Data design area | Primary PRD requirements |
|---|---|
| Interactive events and recovery | FR-VS-008, FR-GRF-002 through FR-GRF-005, FR-TRJ-001 |
| Experiment manifests and seeds | FR-EXP-001 through FR-EXP-006, NFR-REP-001 through NFR-REP-005 |
| Public/privileged trajectories | FR-SIM-002, FR-SIM-006, FR-TRJ-002 through FR-TRJ-004, NFR-SEC-002 |
| Policies and training targets | FR-POL-002, FR-POL-005 through FR-POL-009 |
| CBFM and tracker lineage | FR-CBFM-005 through FR-CBFM-009, FR-TRK-005 through FR-TRK-008 |
| W&B boundary | FR-TRJ-005, FR-TRJ-006, NFR-SEC-001, NFR-PRV-002 |
| Demo privacy | FR-UI-002, FR-UI-004, FR-UI-006, NFR-SEC-003, NFR-PRV-001 |
| Optional sandbox | SEC-SBX-001 through SEC-SBX-004 |

## 24. Acceptance Criteria

This data design is implementation-ready when:

- The browser path has a deterministic session/turn identity and one-event commit rule.
- JSONL/SQLite split-commit recovery is tested and does not duplicate tracker updates.
- SQLite remains provider-managed and is not promoted into a dissertation data store.
- Public policy data and simulator truth have separate schemas, roots, writers, and readers.
- Feature-role manifests prevent rewards, next state, metadata, or truth from becoming action-time features accidentally.
- Parquet publication validates keys, ranges, lifecycle, expected counts, and checksums before a run becomes complete.
- Every final metric declares its data classification and resolves to frozen local hashes.
- DuckDB reproduces analysis without importing W&B data.
- W&B remains optional, allowlisted, and linked back to canonical local artifacts.
- Demo responses are local, deletable, and excluded from synthetic research datasets.
- Current schemas contain no human-study identity, consent, or participant lifecycle data.
- All D2 diagrams compile and every referenced PRD requirement ID resolves.
