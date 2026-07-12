<!-- docs/data-flow-and-schema.md -->

# Data Flow and Database Schema Design

## Socratic Multi-Agent Tutoring and Evaluation Platform

| Field | Value |
|---|---|
| Status | Draft for engineering review |
| Product requirements | `docs/PRD.md` |
| High-level architecture | `docs/HLD.md` |
| Low-level design | `docs/LLD.md` |
| Operational database | PostgreSQL |
| Scientific data format | Apache Parquet in S3-compatible object storage |
| Experiment lineage | MLflow with immutable artifact references |
| Queue | Redis Streams; never a system of record |

## 1. Purpose

This document defines how data enters, moves through, persists within, and exits the Socratic tutoring platform. It specifies:

- Data ownership and classification.
- Runtime, research, training, and evaluation flows.
- PostgreSQL schemas, entities, relationships, constraints, and indexes.
- LangGraph checkpoint and MLflow isolation.
- Public and privileged trajectory schemas.
- Object-store layout and publication protocol.
- Transaction, idempotency, retention, deletion, backup, and schema-evolution rules.

The design must preserve the central validity constraint: simulator truth may support controlled evaluation, but it must never become a policy feature or human-facing production datum.

## 2. Data Design Principles

1. **Separate by trust, not only by column.** Policy-safe and privileged simulator trajectories use different schemas, objects, prefixes, and access policies.
2. **Immutable scientific inputs.** Signed manifests, split definitions, trajectories, calibration artifacts, policy artifacts, and final reports are append-only.
3. **Relational metadata, columnar observations.** PostgreSQL stores identities, state machines, references, and audit records. Parquet stores high-volume turn data.
4. **Framework schemas are isolated.** LangGraph and MLflow own their internal tables; application migrations do not modify them.
5. **Redis is transport.** Job and live-event messages are recoverable from PostgreSQL checkpoints and object-store artifacts.
6. **No hidden reasoning.** Code, test outcomes, structured evidence, and public text may be stored; raw private chain-of-thought may not.
7. **Every result has lineage.** A metric or report must resolve to code, manifest, data partitions, policy, prompts, models, and analysis version.
8. **Human data is a separate lifecycle.** Simulation tables contain no PII; future human records require approved retention, deletion, and access policies.

## 3. Data Classification

| Classification | Examples | Allowed stores | Default access |
|---|---|---|---|
| Public-safe research | Policy observations, directives, prompt-load features, structured evidence, rewards, aggregate metrics | PostgreSQL metadata, public Parquet prefix, MLflow | Researcher and evaluator |
| Privileged synthetic | True mastery, true misconceptions, true `B_t`/`F_t`, simulator parameters, particle snapshots | Privileged object prefix, restricted metadata references | Simulator and authorized evaluation only |
| Operational confidential | Provider request IDs, costs, errors, checkpoint references, audit events | PostgreSQL, redacted telemetry | Operator and service identities |
| Secret | API keys, database credentials, signing keys | Secrets manager only | Workload identity |
| Future human restricted | Identity subject, consent, student input, tracker history | Approved regional stores only | Student, approved study team, restricted services |
| Prohibited | Raw hidden model reasoning, credentials in prompts, unrestricted sandbox filesystem | None | None |

`B_t` and `F_t` are privileged when they are simulator truth and policy-safe only when they are tracker estimates explicitly named `estimated_bandwidth` and `estimated_friction`.

## 4. Level-1 Data Flow

```d2
direction: right

actors: "Actors" {
  researcher: "Researcher"
  evaluator: "Evaluator"
  operator: "Operator"
  future_student: "Future Student"
}

edge: "Application Edge" {
  web: "Glass Box Client"
  api: "FastAPI Gateway"
  websocket: "WebSocket Hub"
}

runtime: "Serving Plane" {
  graph: "LangGraph Runtime"
  tracker: "Epistemic Tracker"
  policy: "Policy Adapter"
  env: "Synthetic Environment"
  writer: "Trajectory Writer"
}

research: "Research Plane" {
  coordinator: "Experiment Coordinator"
  rollout: "Rollout Workers"
  trainer: "Policy Trainer"
  evaluation: "Evaluation Workers"
  reports: "Report Builder"
}

stores: "Systems of Record" {
  postgres: "PostgreSQL\nOperational Metadata"
  checkpoints: "PostgreSQL\nLangGraph Schema"
  public_objects: "Object Storage\nPolicy-Safe Parquet"
  privileged_objects: "Restricted Object Storage\nSimulator Truth"
  mlflow: "MLflow\nRuns and Policies"
}

transport: "Transport and Telemetry" {
  redis: "Redis Streams"
  tracing: "Redacted Tracing"
}

actors.researcher -> edge.web: "experiment manifests and inspection"
actors.evaluator -> edge.web: "reports and labeled synthetic evidence"
actors.operator -> edge.api: "promotion and operations"
actors.future_student -> edge.websocket: "approved future interaction"

edge.web -> edge.api: "REST commands and queries"
edge.web <-> edge.websocket: "live events and input"
edge.api -> runtime.graph: "session command"
edge.api -> research.coordinator: "experiment command"

runtime.graph -> runtime.tracker: "evidence and tutor control"
runtime.tracker -> runtime.policy: "tracker summaries via projection"
runtime.policy -> runtime.graph: "PolicyDecision"
runtime.graph -> runtime.env: "TutorControl in synthetic mode"
runtime.env -> runtime.writer: "public and privileged turn views"
runtime.graph -> stores.checkpoints: "durable workflow checkpoint"
runtime.writer -> stores.public_objects: "policy-safe trajectory"
runtime.writer -> stores.privileged_objects: "separate simulator-truth trajectory"
runtime.writer -> stores.postgres: "turn index and commit receipt"

research.coordinator -> stores.postgres: "experiment and shard metadata"
research.coordinator -> transport.redis: "durable shard command"
transport.redis -> research.rollout: "claimed work"
research.rollout -> stores.public_objects: "trajectory partitions"
research.rollout -> stores.privileged_objects: "restricted truth partitions"
stores.public_objects -> research.trainer: "allowlisted training columns"
research.trainer -> stores.mlflow: "candidate policy and run lineage"
stores.public_objects -> research.evaluation: "observable outcomes"
stores.privileged_objects -> research.evaluation: "controlled simulator metrics"
stores.mlflow -> research.evaluation: "candidate and incumbent policies"
research.evaluation -> stores.postgres: "gate and metric records"
research.evaluation -> research.reports: "validated analysis outputs"
research.reports -> stores.public_objects: "versioned reports"

runtime.graph -> transport.tracing: "redacted runtime trace"
research.coordinator -> transport.tracing: "job metrics and costs"
```

### 4.1 Boundary Rules

- `Policy Adapter` reads tracker summaries and public policy artifacts only.
- `Policy Trainer` receives an allowlisted public trajectory projection; privileged objects are not mounted into its workload.
- `Evaluation Workers` may receive controlled privileged access only for metrics declared to use simulator truth.
- The Glass Box receives privileged synthetic data only through an administrator-authorized endpoint that is absent from human deployments.
- WebSocket delivery and Redis acknowledgement never substitute for durable database or object commits.

## 5. Canonical Data Flows

### 5.1 Experiment Submission

1. A researcher submits an `ExperimentManifest` or content-addressed manifest reference.
2. The API authenticates the caller and computes the canonical manifest hash.
3. The coordinator validates schemas, task/profile/split disjointness, artifact references, budgets, and allowed models.
4. PostgreSQL stores an immutable manifest artifact reference and creates an `experiment` row.
5. Conditions and deterministic shard identities are inserted in one transaction.
6. An outbox event publishes shard commands to Redis Streams.
7. Workers claim commands through consumer groups and update shard leases in PostgreSQL.

### 5.2 Synthetic Turn

1. LangGraph loads the latest checkpoint and environment runtime handle.
2. The tracker produces policy-safe summaries.
3. The policy adapter emits `PolicyObservation` and `PolicyDecision`.
4. Tutor generation and guardrail processing produce final `TutorControl`.
5. The simulator updates private fast state, emits evidence, and applies slow state transitions.
6. The tracker applies its subjective update independently.
7. The trajectory builder creates two records with a shared `turn_id`:
   - Public-safe turn record.
   - Privileged simulator record.
8. The writer publishes both objects, writes checksums and references, then commits the graph checkpoint.
9. A committed event is emitted to the client or rollout coordinator.

### 5.3 Policy Training

1. The trainer receives a signed training manifest and explicit public partition references.
2. A dataset view enforces the policy feature allowlist.
3. The trainer verifies artifact checksums, schema versions, split authorization, and feature provenance.
4. Training writes diagnostics and a candidate policy object.
5. MLflow records the candidate, parent policy, data references, code revision, hyperparameters, and metrics.
6. PostgreSQL stores an application-level `policy_version` and immutable MLflow/object references.

### 5.4 Evaluation and Reporting

1. Evaluation workers receive a gate definition, candidate policy, held-out manifest, and metric-source declaration.
2. Public metrics read only public trajectories.
3. Simulator-calibration metrics receive a narrowly scoped privileged object prefix.
4. Expert and LLM judge grades are stored as structured rating artifacts with blinding metadata.
5. Metric estimates, intervals, exclusions, and provenance are stored transactionally.
6. Gate logic produces `accepted`, `rejected`, or `inconclusive`.
7. Report artifacts reference immutable metrics and source partitions.

### 5.5 Future Human Turn

Human mode contains no privileged trajectory branch. Student identity and consent data are separated from tutoring content by distinct tables and access roles. Human activation remains blocked until its optional schema, retention, and deletion workflow receive governance approval.

## 6. PostgreSQL Organization

### 6.1 Database Schemas

| Schema | Owner | Contents | Migration owner |
|---|---|---|---|
| `app` | Application | Sessions, experiments, artifacts, policies, metrics, audit | Alembic application migrations |
| `catalog` | Application | Tasks, concepts, profiles, splits, versioned domain metadata | Alembic application migrations |
| `langgraph` | LangGraph integration | Checkpoints and provider-managed runtime tables | LangGraph package/setup only |
| `mlflow` | MLflow service | Tracking and registry internals | MLflow migrations only |
| `auth` | Future identity integration | Optional identity/consent mappings | Separate approved migration set |

Application tables store `thread_id`, `mlflow_run_id`, and artifact URIs as opaque references. They do not create foreign keys into `langgraph` or `mlflow` internal tables.

### 6.2 Type Conventions

- Primary IDs: `uuid`, generated as UUIDv7 by the application.
- Time: `timestamptz`, always UTC.
- Version: non-empty `text`; semantic or content version depending on artifact.
- Hash: lowercase hexadecimal `text` with algorithm column or fixed SHA-256 check.
- Unit interval: `double precision` with `CHECK (value >= 0 AND value <= 1 AND isfinite(value))`.
- Money: integer micro-units plus ISO currency, never binary floating point.
- Token counts and sequence numbers: `bigint` with non-negative checks.
- Flexible validated manifests: `jsonb` plus `schema_version` and content hash.
- Status fields: PostgreSQL `text` with checks or application enums; avoid database enum types during rapid research evolution.

### 6.3 Common Columns

Mutable operational tables include `created_at`, `updated_at`, and optimistic `row_version`. Immutable scientific rows include `created_at` and content hash but no update timestamp. Soft deletion is not used for immutable research artifacts; lifecycle state is represented explicitly.

## 7. Operational Database Schema

```d2
direction: right

session: "app.session" {
  shape: sql_table
  id: "uuid PK"
  mode: "text NOT NULL"
  status: "text NOT NULL"
  thread_id: "text UNIQUE NOT NULL"
  policy_version_id: "uuid FK NOT NULL"
  task_version_id: "uuid FK NOT NULL"
  profile_version_id: "uuid FK NULL"
  owner_subject: "text NULL"
  latest_turn_index: "int NOT NULL"
  latest_checkpoint_id: "text NULL"
  budget_state: "jsonb NOT NULL"
  created_at: "timestamptz NOT NULL"
  completed_at: "timestamptz NULL"
}

session_turn: "app.session_turn" {
  shape: sql_table
  id: "uuid PK"
  session_id: "uuid FK NOT NULL"
  episode_id: "uuid NOT NULL"
  turn_index: "int NOT NULL"
  public_artifact_id: "uuid FK NOT NULL"
  privileged_artifact_id: "uuid FK NULL"
  checkpoint_id: "text NOT NULL"
  commit_checksum: "text NOT NULL"
  committed_at: "timestamptz NOT NULL"
  unique_turn: "UNIQUE(session_id, turn_index)"
}

artifact: "app.artifact" {
  shape: sql_table
  id: "uuid PK"
  kind: "text NOT NULL"
  classification: "text NOT NULL"
  schema_name: "text NOT NULL"
  schema_version: "text NOT NULL"
  uri: "text UNIQUE NOT NULL"
  sha256: "text NOT NULL"
  size_bytes: "bigint NOT NULL"
  state: "text NOT NULL"
  created_at: "timestamptz NOT NULL"
}

idempotency: "app.idempotency_record" {
  shape: sql_table
  scope: "text PK"
  idempotency_key: "text PK"
  request_hash: "text NOT NULL"
  status: "text NOT NULL"
  response_ref: "jsonb NULL"
  expires_at: "timestamptz NOT NULL"
  created_at: "timestamptz NOT NULL"
}

outbox: "app.outbox_event" {
  shape: sql_table
  id: "uuid PK"
  aggregate_type: "text NOT NULL"
  aggregate_id: "uuid NOT NULL"
  event_type: "text NOT NULL"
  payload: "jsonb NOT NULL"
  created_at: "timestamptz NOT NULL"
  published_at: "timestamptz NULL"
  attempt_count: "int NOT NULL"
}

audit: "app.audit_event" {
  shape: sql_table
  id: "uuid PK"
  actor_type: "text NOT NULL"
  actor_id: "text NOT NULL"
  action: "text NOT NULL"
  resource_type: "text NOT NULL"
  resource_id: "text NOT NULL"
  correlation_id: "text NOT NULL"
  details: "jsonb NOT NULL"
  occurred_at: "timestamptz NOT NULL"
}

policy_version: "app.policy_version" {
  shape: sql_table
  id: "uuid PK"
  policy_id: "uuid FK NOT NULL"
  version: "text NOT NULL"
  artifact_id: "uuid FK NOT NULL"
  status: "text NOT NULL"
  action_space_version: "text NOT NULL"
  observation_schema_version: "text NOT NULL"
  mlflow_run_id: "text NULL"
  created_at: "timestamptz NOT NULL"
  unique_version: "UNIQUE(policy_id, version)"
}

task_version: "catalog.task_version (external)" {
  shape: sql_table
  id: "uuid PK"
}

profile_version: "catalog.profile_version (external)" {
  shape: sql_table
  id: "uuid PK"
}

session.policy_version_id -> policy_version.id: "frozen policy"
session.task_version_id -> task_version.id: "assigned task"
session.profile_version_id -> profile_version.id: "synthetic profile"
session_turn.session_id -> session.id: "contains"
session_turn.public_artifact_id -> artifact.id: "public turn"
session_turn.privileged_artifact_id -> artifact.id: "synthetic truth"
policy_version.artifact_id -> artifact.id: "policy object"
```

The diagram references `catalog.task_version` and `catalog.profile_version`, defined in the catalog schema below.

### 7.1 Session Constraints

- `mode` is one of `deterministic`, `llm_simulation`, `glass_box`, or `human`.
- `profile_version_id` is required for synthetic modes and prohibited for human mode unless it references a non-personal pedagogical configuration rather than simulator truth.
- `privileged_artifact_id` is required for configured synthetic evaluation and always null in human mode.
- A session's `policy_version_id` is immutable after creation.
- `latest_turn_index` changes only after a matching `session_turn` commit.

### 7.2 Operational Indexes

| Table | Index | Purpose |
|---|---|---|
| `session` | `(status, updated_at)` | Worker recovery and stale-session scan |
| `session` | `(owner_subject, created_at DESC)` partial where non-null | Future user session list |
| `session_turn` | unique `(session_id, turn_index)` | Idempotent turn commit |
| `session_turn` | `(episode_id, turn_index)` | Episode reconstruction |
| `artifact` | unique `(uri)` and `(sha256, classification, kind)` | Deduplication and integrity |
| `artifact` | `(state, created_at)` | Cleanup of incomplete objects |
| `idempotency_record` | `(expires_at)` | Retention cleanup |
| `outbox_event` | `(published_at, created_at)` partial where unpublished | Publisher polling |
| `audit_event` | `(resource_type, resource_id, occurred_at DESC)` | Resource audit history |

## 8. Catalog Database Schema

```d2
direction: right

task: "catalog.task" {
  shape: sql_table
  id: "uuid PK"
  stable_key: "text UNIQUE NOT NULL"
  title: "text NOT NULL"
  domain: "text NOT NULL"
  created_at: "timestamptz NOT NULL"
}

task_version: "catalog.task_version" {
  shape: sql_table
  id: "uuid PK"
  task_id: "uuid FK NOT NULL"
  version: "text NOT NULL"
  prompt_artifact_id: "uuid FK NOT NULL"
  test_bundle_artifact_id: "uuid FK NOT NULL"
  difficulty: "double precision NOT NULL"
  content_hash: "text NOT NULL"
  created_at: "timestamptz NOT NULL"
  unique_version: "UNIQUE(task_id, version)"
}

concept: "catalog.concept" {
  shape: sql_table
  id: "uuid PK"
  stable_key: "text UNIQUE NOT NULL"
  name: "text NOT NULL"
  description: "text NOT NULL"
}

task_concept: "catalog.task_concept" {
  shape: sql_table
  task_version_id: "uuid PK, FK"
  concept_id: "uuid PK, FK"
  role: "text NOT NULL"
  weight: "double precision NOT NULL"
}

prerequisite: "catalog.concept_prerequisite" {
  shape: sql_table
  graph_version: "text PK"
  concept_id: "uuid PK, FK"
  prerequisite_concept_id: "uuid PK, FK"
  strength: "double precision NOT NULL"
}

misconception: "catalog.misconception" {
  shape: sql_table
  id: "uuid PK"
  stable_key: "text UNIQUE NOT NULL"
  concept_id: "uuid FK NOT NULL"
  description: "text NOT NULL"
  diagnostic_artifact_id: "uuid FK NULL"
}

profile: "catalog.profile" {
  shape: sql_table
  id: "uuid PK"
  stable_key: "text UNIQUE NOT NULL"
  label: "text NOT NULL"
  synthetic_only: "boolean NOT NULL"
}

profile_version: "catalog.profile_version" {
  shape: sql_table
  id: "uuid PK"
  profile_id: "uuid FK NOT NULL"
  version: "text NOT NULL"
  parameters: "jsonb NOT NULL"
  content_hash: "text NOT NULL"
  created_at: "timestamptz NOT NULL"
  unique_version: "UNIQUE(profile_id, version)"
}

profile_mastery: "catalog.profile_mastery" {
  shape: sql_table
  profile_version_id: "uuid PK, FK"
  concept_id: "uuid PK, FK"
  initial_mastery: "double precision NOT NULL"
}

profile_misconception: "catalog.profile_misconception" {
  shape: sql_table
  profile_version_id: "uuid PK, FK"
  misconception_id: "uuid PK, FK"
  active_probability: "double precision NOT NULL"
}

split_member: "catalog.split_member" {
  shape: sql_table
  split_manifest_hash: "text PK"
  member_type: "text PK"
  member_id: "uuid PK"
  split: "text NOT NULL"
}

task_version.task_id -> task.id: "versions"
task_concept.task_version_id -> task_version.id: "covers"
task_concept.concept_id -> concept.id: "targets"
prerequisite.concept_id -> concept.id: "advanced concept"
prerequisite.prerequisite_concept_id -> concept.id: "requires"
misconception.concept_id -> concept.id: "misunderstands"
profile_version.profile_id -> profile.id: "versions"
profile_mastery.profile_version_id -> profile_version.id: "initial state"
profile_mastery.concept_id -> concept.id: "mastery of"
profile_misconception.profile_version_id -> profile_version.id: "initial state"
profile_misconception.misconception_id -> misconception.id: "probability of"
```

### 8.1 Catalog Integrity

- Task, profile, concept, and misconception stable keys are never reused for different meanings.
- Published versions are immutable; corrections create new versions.
- Prerequisite graphs are validated as directed acyclic graphs before publication.
- Unit-interval checks apply to difficulty, weights, mastery, strength, and probabilities.
- `split_member` validation prevents the same member appearing in incompatible partitions for one manifest.
- Test bundles and diagnostic probes are artifacts with checksums and leakage metadata.

## 9. Experiment, Policy, and Evaluation Schema

```d2
direction: right

experiment: "app.experiment" {
  shape: sql_table
  id: "uuid PK"
  name: "text NOT NULL"
  status: "text NOT NULL"
  manifest_artifact_id: "uuid FK NOT NULL"
  manifest_hash: "text UNIQUE NOT NULL"
  split_manifest_hash: "text NOT NULL"
  code_revision: "text NOT NULL"
  submitted_by: "text NOT NULL"
  created_at: "timestamptz NOT NULL"
  completed_at: "timestamptz NULL"
}

condition: "app.experiment_condition" {
  shape: sql_table
  id: "uuid PK"
  experiment_id: "uuid FK NOT NULL"
  stable_key: "text NOT NULL"
  policy_version_id: "uuid FK NOT NULL"
  config: "jsonb NOT NULL"
  config_hash: "text NOT NULL"
  unique_condition: "UNIQUE(experiment_id, stable_key)"
}

shard: "app.experiment_shard" {
  shape: sql_table
  id: "uuid PK"
  experiment_id: "uuid FK NOT NULL"
  condition_id: "uuid FK NOT NULL"
  split: "text NOT NULL"
  shard_index: "int NOT NULL"
  status: "text NOT NULL"
  lease_owner: "text NULL"
  lease_expires_at: "timestamptz NULL"
  attempt_count: "int NOT NULL"
  output_partition_id: "uuid FK NULL"
  unique_shard: "UNIQUE(experiment_id, condition_id, split, shard_index)"
}

partition: "app.trajectory_partition" {
  shape: sql_table
  id: "uuid PK"
  experiment_id: "uuid FK NOT NULL"
  condition_id: "uuid FK NOT NULL"
  split: "text NOT NULL"
  public_artifact_id: "uuid FK NOT NULL"
  privileged_artifact_id: "uuid FK NULL"
  row_count: "bigint NOT NULL"
  schema_version: "text NOT NULL"
  validation_status: "text NOT NULL"
  created_at: "timestamptz NOT NULL"
}

policy: "app.policy" {
  shape: sql_table
  id: "uuid PK"
  stable_key: "text UNIQUE NOT NULL"
  policy_class: "text NOT NULL"
  created_at: "timestamptz NOT NULL"
}

policy_version: "app.policy_version" {
  shape: sql_table
  id: "uuid PK"
  policy_id: "uuid FK NOT NULL"
  version: "text NOT NULL"
  artifact_id: "uuid FK NOT NULL"
  training_experiment_id: "uuid FK NULL"
  mlflow_run_id: "text NULL"
  status: "text NOT NULL"
  created_at: "timestamptz NOT NULL"
}

evaluation_run: "app.evaluation_run" {
  shape: sql_table
  id: "uuid PK"
  experiment_id: "uuid FK NOT NULL"
  candidate_policy_version_id: "uuid FK NULL"
  gate_definition_version: "text NOT NULL"
  status: "text NOT NULL"
  analysis_revision: "text NOT NULL"
  created_at: "timestamptz NOT NULL"
  completed_at: "timestamptz NULL"
}

metric: "app.metric_result" {
  shape: sql_table
  id: "uuid PK"
  evaluation_run_id: "uuid FK NOT NULL"
  metric_key: "text NOT NULL"
  scope: "jsonb NOT NULL"
  estimate: "double precision NULL"
  lower_bound: "double precision NULL"
  upper_bound: "double precision NULL"
  sample_count: "bigint NOT NULL"
  artifact_id: "uuid FK NULL"
  unique_metric: "UNIQUE(evaluation_run_id, metric_key, scope)"
}

gate: "app.gate_result" {
  shape: sql_table
  id: "uuid PK"
  evaluation_run_id: "uuid FK NOT NULL"
  gate_key: "text NOT NULL"
  decision: "text NOT NULL"
  rationale: "text NOT NULL"
  evidence_artifact_id: "uuid FK NOT NULL"
  decided_at: "timestamptz NOT NULL"
  unique_gate: "UNIQUE(evaluation_run_id, gate_key)"
}

promotion: "app.policy_promotion" {
  shape: sql_table
  id: "uuid PK"
  policy_version_id: "uuid FK NOT NULL"
  target_alias: "text NOT NULL"
  previous_policy_version_id: "uuid FK NULL"
  decision: "text NOT NULL"
  actor_id: "text NOT NULL"
  reason: "text NOT NULL"
  gate_snapshot_hash: "text NOT NULL"
  created_at: "timestamptz NOT NULL"
}

artifact: "app.artifact (external)" {
  shape: sql_table
  id: "uuid PK"
}

experiment.manifest_artifact_id -> artifact.id: "signed manifest"
condition.experiment_id -> experiment.id: "belongs to"
condition.policy_version_id -> policy_version.id: "evaluates"
shard.experiment_id -> experiment.id: "belongs to"
shard.condition_id -> condition.id: "executes"
shard.output_partition_id -> partition.id: "produces"
partition.experiment_id -> experiment.id: "belongs to"
partition.condition_id -> condition.id: "contains condition"
partition.public_artifact_id -> artifact.id: "public parquet"
partition.privileged_artifact_id -> artifact.id: "restricted parquet"
policy_version.policy_id -> policy.id: "versions"
policy_version.training_experiment_id -> experiment.id: "trained by"
policy_version.artifact_id -> artifact.id: "serialized policy"
evaluation_run.experiment_id -> experiment.id: "evaluates"
evaluation_run.candidate_policy_version_id -> policy_version.id: "candidate"
metric.evaluation_run_id -> evaluation_run.id: "measured by"
gate.evaluation_run_id -> evaluation_run.id: "decided by"
promotion.policy_version_id -> policy_version.id: "promotes"
promotion.previous_policy_version_id -> policy_version.id: "replaces or restores"
```

### 9.1 State Machines

#### Experiment

```text
submitted -> validating -> queued -> running -> evaluating -> completed
                          |          |           |
                          +--------> failed <----+
                          +--------> cancelled
```

#### Shard

```text
pending -> leased -> running -> publishing -> completed
             |          |            |
             +-------> retryable <----+
                           |
                           +-> failed
```

#### Policy Version

```text
candidate -> evaluating -> eligible -> promoted
                  |           |           |
                  +-> rejected+           +-> superseded
                              +-----------> archived
```

Transitions use compare-and-swap updates on `row_version`. Invalid transitions fail transactionally and emit audit events.

### 9.2 Research Indexes

| Table | Index | Purpose |
|---|---|---|
| `experiment` | `(status, created_at)` | Scheduling and operations |
| `experiment_condition` | unique `(experiment_id, stable_key)` | Stable comparison key |
| `experiment_shard` | `(status, lease_expires_at)` | Claim and reclaim work |
| `experiment_shard` | unique `(experiment_id, condition_id, split, shard_index)` | Deterministic shard idempotency |
| `trajectory_partition` | `(experiment_id, condition_id, split)` | Dataset discovery |
| `policy_version` | `(policy_id, created_at DESC)` | Version history |
| `evaluation_run` | `(candidate_policy_version_id, status)` | Promotion gate lookup |
| `metric_result` | `(metric_key, evaluation_run_id)` | Report queries |
| `gate_result` | `(gate_key, decision)` | Gate analysis |
| `policy_promotion` | `(target_alias, created_at DESC)` | Current alias and rollback history |

## 10. LangGraph Checkpoint Data

The `langgraph` schema is managed by the installed PostgreSQL checkpointer. Application code treats it as an implementation-owned store with these conceptual fields:

- Thread identifier.
- Checkpoint identifier and parent.
- Checkpoint namespace.
- Serialized policy-safe graph state.
- Pending node writes.
- Checkpoint metadata and creation time.

Application rules:

- `app.session.thread_id` is the stable cursor.
- `app.session.latest_checkpoint_id` is an advisory recovery pointer, not a foreign key.
- Graph state contains tracker summaries and artifact references, not particle arrays, Parquet rows, secrets, or raw hidden reasoning.
- Checkpoint serialization must not use unrestricted pickle fallback in deployed environments.
- LangGraph migration/setup runs before application readiness but outside Alembic's application revision chain.

## 11. MLflow Data

MLflow owns its tracking and registry schema. The application stores only:

- `mlflow_run_id`.
- Registered model/policy name and version where applicable.
- Immutable policy artifact reference and checksum.
- Dataset input references.
- Promotion/gate state in application tables.

MLflow is used for exploration and lineage, while application policy promotion remains authoritative in `app.policy_promotion`. A deleted or unavailable MLflow UI must not erase the canonical policy object or gate evidence.

## 12. Public Trajectory Schema

The canonical public turn dataset is Parquet schema version `trajectory.public.v1`.

### 12.1 Identity and Provenance

| Field | Type | Null | Description |
|---|---|---|---|
| `schema_version` | string | No | `trajectory.public.v1` |
| `experiment_id` | fixed string/UUID | No | Experiment identity |
| `condition_id` | fixed string/UUID | No | Comparison condition |
| `shard_id` | fixed string/UUID | No | Rollout shard |
| `session_id` | fixed string/UUID | Yes | Interactive session if applicable |
| `episode_id` | fixed string/UUID | No | Episode identity |
| `turn_id` | fixed string/UUID | No | Shared public/privileged join key |
| `turn_index` | int32 | No | Zero-based turn index |
| `split` | dictionary string | No | Calibration, selection, validation, or held-out |
| `task_version_id` | fixed string/UUID | No | Immutable task version |
| `profile_version_id` | fixed string/UUID | Yes | Synthetic profile identity; may be blinded in evaluator exports |
| `root_seed` | uint64 | No | Episode root seed |
| `graph_version` | string | No | Graph implementation version |
| `code_revision` | string | No | Source revision |

### 12.2 Policy and Tutor Control

| Field | Type | Null | Description |
|---|---|---|---|
| `policy_id` | fixed string/UUID | No | Policy family |
| `policy_version` | string | No | Frozen version |
| `action_space_version` | string | No | Directive mapping version |
| `directive` | dictionary string | No | Selected action |
| `target_concept` | string | No | Policy target |
| `action_propensity` | float64 | Yes | Chosen-action probability |
| `rendered_prompt` | string | No | Final guardrail-approved prompt |
| `prompt_template_version` | string | No | Template lineage |
| `generator_model` | string | No | Template or model identifier |
| `estimated_tutor_difficulty` | float64 | No | Realized prompt difficulty estimate |

### 12.3 Policy Observation

| Field | Type | Null | Description |
|---|---|---|---|
| `obs_target_mastery` | float64 | No | Tracker estimate |
| `obs_prerequisite_mastery_min` | float64 | No | Tracker-derived minimum or sentinel encoding |
| `obs_misconception_probability` | float64 | No | Tracker marginal |
| `obs_estimated_bandwidth` | float64 | No | Tracker estimate, never simulator truth |
| `obs_estimated_friction` | float64 | No | Tracker estimate, never simulator truth |
| `obs_mastery_entropy` | float64 | No | Planning uncertainty |
| `obs_bandwidth_variance` | float64 | No | Posterior dispersion |
| `obs_repeated_failures` | int32 | No | Turn counter |
| `obs_hints_used` | int32 | No | Turn counter |
| `obs_recent_guardrail_events` | int32 | No | Bounded history feature |

Full mastery and misconception summary maps are stored as normalized list-of-struct fields or separate concept-level Parquet datasets, not dynamic top-level columns.

### 12.4 Prompt Load and Evidence

| Field | Type | Null | Description |
|---|---|---|---|
| `load_token_count` | float64 | No | Normalized token feature |
| `load_code_span` | float64 | No | Normalized code feature |
| `load_ast_depth` | float64 | No | Normalized AST feature |
| `load_concept_novelty` | float64 | No | Normalized novelty feature |
| `load_entropy` | float64 | No | Normalized entropy feature |
| `load_concept_count` | int32 | No | Concepts implicated |
| `load_aggregate` | float64 | No | Weighted prompt load |
| `load_normalizer_version` | string | No | Calibration lineage |
| `student_text` | string | No | Public response |
| `execution_passed` | boolean | Yes | Structured code outcome |
| `probe_passed` | boolean | Yes | Private diagnostic final result, not reasoning |
| `guardrail_leakage` | boolean | No | Leakage event |
| `guardrail_attempts` | int32 | No | Candidate count |
| `active_concepts` | list<string> | No | Evidence mask |
| `active_misconceptions` | list<string> | No | Evidence mask |
| `load_proxy` | struct | No | Observable proxy fields and label |

Large stdout/stderr and test details use bounded structured artifacts referenced by ID. They are not embedded without limits.

### 12.5 Reward, Outcome, and Operations

| Field | Type | Null | Description |
|---|---|---|---|
| `reward_mastery_gain` | float64 | No | Simulator reward component |
| `reward_overload_penalty` | float64 | No | Reward component |
| `reward_bandwidth_penalty` | float64 | No | Reward component |
| `reward_leakage_penalty` | float64 | No | Reward component |
| `reward_efficiency_penalty` | float64 | No | Optional component, zero if disabled |
| `reward_total` | float64 | No | Checked sum |
| `terminated` | boolean | No | Natural terminal flag |
| `truncated` | boolean | No | Administrative terminal flag |
| `termination_reason` | dictionary string | Yes | Typed reason |
| `latency_ms` | int64 | No | End-to-end turn latency |
| `input_tokens` | int64 | No | Provider usage |
| `output_tokens` | int64 | No | Provider usage |
| `cost_microunits` | int64 | No | Normalized cost |
| `currency` | string | No | ISO currency |
| `public_checksum` | string | No | Canonical row/envelope checksum |

All numeric columns reject NaN and infinity during validation.

## 13. Privileged Synthetic Trajectory Schema

The privileged dataset is `trajectory.privileged.v1`. It joins to the public record only by `turn_id` and provenance identifiers.

| Field | Type | Description |
|---|---|---|
| `schema_version` | string | `trajectory.privileged.v1` |
| `experiment_id` | UUID string | Experiment identity |
| `episode_id` | UUID string | Episode identity |
| `turn_id` | UUID string | Join key |
| `turn_index` | int32 | Turn index |
| `true_mastery_before` | list<struct<concept,value>> | Private pre-turn mastery |
| `true_mastery_after` | list<struct<concept,value>> | Private next-turn mastery |
| `true_misconceptions_before` | list<struct<id,active>> | Private pre-turn flags |
| `true_misconceptions_after` | list<struct<id,active>> | Private next-turn flags |
| `true_bandwidth_before` | float64 | Simulator `B` before prompt |
| `true_bandwidth_post_prompt` | float64 | Simulator fast-projected `B` |
| `true_friction_post_prompt` | float64 | Simulator overload `F` |
| `true_underchallenge` | float64 | Simulator diagnostic |
| `sampled_overload` | boolean | Seeded overload event |
| `simulator_parameters_hash` | string | Parameter-set lineage |
| `transition_random_draw_refs` | struct | Optional reproducibility metadata, not hidden reasoning |
| `privileged_checksum` | string | Integrity checksum |

This schema is forbidden to policy loaders. Primary educational-outcome reports must label metrics that depend on it.

## 14. Concept-Level and Rating Datasets

To avoid dynamic map columns in analytic queries, these optional long-form datasets are published:

### `trajectory.concept_estimate.v1`

| Field | Type |
|---|---|
| `experiment_id`, `episode_id`, `turn_id` | UUID string |
| `concept_id` | string |
| `estimated_mastery` | float64 |
| `mastery_entropy` | float64 |
| `active_evidence` | boolean |

### `trajectory.misconception_estimate.v1`

| Field | Type |
|---|---|
| `experiment_id`, `episode_id`, `turn_id` | UUID string |
| `misconception_id` | string |
| `posterior_probability` | float64 |
| `active_evidence` | boolean |

### `evaluation.rating.v1`

| Field | Type |
|---|---|
| `evaluation_run_id`, `rating_id` | UUID string |
| `blinded_item_id` | string |
| `rater_type` | expert or LLM |
| `rater_version` | string |
| `rubric_version` | string |
| `dimension_scores` | list<struct<dimension,score>> |
| `structured_rationale` | string, bounded and redacted |
| `schema_valid` | boolean |
| `created_at` | timestamp UTC |

Raters receive no condition label, policy name, true state, or aggregate outcome.

## 15. Object Storage Layout

Objects use immutable content-addressed paths. Example layout:

```text
s3://research-brain-{environment}/
|-- manifests/
|   `-- sha256/{hash}/manifest.json
|-- catalog/
|   |-- tasks/{task_id}/{version}/...
|   |-- probes/{probe_id}/{version}/...
|   `-- test-bundles/{artifact_hash}/...
|-- trajectories-public/
|   `-- schema=trajectory.public.v1/
|       `-- experiment_id={uuid}/
|           `-- split={split}/
|               `-- condition_id={uuid}/part-{shard_id}.parquet
|-- trajectories-privileged/
|   `-- schema=trajectory.privileged.v1/
|       `-- experiment_id={uuid}/
|           `-- split={split}/
|               `-- condition_id={uuid}/part-{shard_id}.parquet
|-- policies/
|   `-- {policy_id}/{version}/{sha256}/...
|-- calibration/
|   `-- cbfm/{version}/{sha256}/...
|-- evaluations/
|   `-- {evaluation_run_id}/metrics-and-gates/...
|-- reports/
|   `-- {experiment_id}/{report_version}/{sha256}/...
`-- temporary/
    `-- {upload_id}/...
```

Public and privileged prefixes use different IAM policies and preferably different encryption keys. Training workloads receive no credentials for `trajectories-privileged`.

### 15.1 Publication Protocol

1. Serialize using the registered Arrow schema.
2. Validate row counts, finite values, required fields, unique keys, and feature allowlists.
3. Compute SHA-256 over canonical bytes.
4. Upload to `temporary/{upload_id}` with checksum metadata.
5. Read back metadata and verify checksum/size.
6. Copy or promote to the content-addressed final key.
7. Insert `app.artifact` as `available` and associated partition metadata in one transaction.
8. Delete the temporary object asynchronously.

Readers accept only `available` artifacts referenced by committed metadata.

## 16. Turn-Level Data Flow

```d2
direction: right

source: "Turn Sources" {
  tracker: "TrackerEstimates"
  policy: "PolicyObservation and Decision"
  tutor: "TutorControl"
  evidence: "ObservationEvidence"
  true_state: "Simulator True State"
  operations: "Latency, Usage, Cost"
}

transform: "Trajectory Builder" {
  validate: "Validate IDs, Versions, and Finite Values"
  public_projection: "Build Policy-Safe Turn"
  privileged_projection: "Build Privileged Synthetic Turn"
  checksum: "Canonicalize and Checksum"
}

publish: "Commit Pipeline" {
  public_object: "Publish Public Parquet"
  privileged_object: "Publish Restricted Parquet"
  index: "Commit Artifact and Turn Index"
  checkpoint: "Commit Graph Checkpoint Reference"
  event: "Emit turn_committed"
}

consumers: "Consumers" {
  training: "Policy Training"
  public_eval: "Observable Evaluation"
  privileged_eval: "Controlled Simulator Evaluation"
  glass_box: "Synthetic Glass Box"
}

source.tracker -> transform.validate: "policy-safe estimates"
source.policy -> transform.validate: "action and propensity"
source.tutor -> transform.validate: "realized intervention"
source.evidence -> transform.validate: "observable evidence"
source.operations -> transform.validate: "resource metadata"
source.true_state -> transform.privileged_projection: "restricted only"

transform.validate -> transform.public_projection: "accepted common fields"
transform.validate -> transform.privileged_projection: "accepted identities"
transform.public_projection -> transform.checksum: "public envelope"
transform.privileged_projection -> transform.checksum: "privileged envelope"

transform.checksum -> publish.public_object: "public content"
transform.checksum -> publish.privileged_object: "privileged content"
publish.public_object -> publish.index: "artifact receipt"
publish.privileged_object -> publish.index: "restricted artifact receipt"
publish.index -> publish.checkpoint: "turn commit receipt"
publish.checkpoint -> publish.event: "durable commit complete"

publish.public_object -> consumers.training: "allowlisted columns"
publish.public_object -> consumers.public_eval: "observable metrics"
publish.privileged_object -> consumers.privileged_eval: "declared truth metrics"
publish.public_object -> consumers.glass_box: "public synthetic view"
publish.privileged_object -> consumers.glass_box: "authorized labeled truth view"
```

## 17. Lineage from Manifest to Report

```d2
direction: down

manifest: "Signed Experiment Manifest\ncontent hash"
catalog: "Versioned Tasks, Profiles, Splits, and Probes"
code: "Code Revision and Dependency Lock"
prompts: "Prompt and Model Versions"
calibration: "Frozen CBFM / Tracker Artifacts"

experiment: "Experiment and Condition Records"
shards: "Deterministic Shard Records"
public_data: "Public Trajectory Partitions"
privileged_data: "Privileged Synthetic Partitions"
training_run: "MLflow Training Run"
policy: "Candidate Policy Artifact"
evaluation: "Evaluation Run"
metrics: "Metric and Gate Records"
report: "Versioned Report Artifact"

manifest -> experiment: "defines"
catalog -> manifest: "referenced by immutable IDs"
code -> experiment: "pins implementation"
prompts -> manifest: "pins generation"
calibration -> manifest: "pins cognitive and tracker configuration"

experiment -> shards: "expands into"
shards -> public_data: "produce"
shards -> privileged_data: "produce under restricted policy"
public_data -> training_run: "policy-safe input"
training_run -> policy: "produces"
policy -> evaluation: "candidate"
public_data -> evaluation: "observable evidence"
privileged_data -> evaluation: "declared simulator evidence"
evaluation -> metrics: "computes estimates and decisions"
metrics -> report: "renders"

manifest -> report: "provenance"
code -> report: "analysis revision"
public_data -> report: "source partition hashes"
privileged_data -> report: "restricted metric-source declaration"
policy -> report: "policy version"
```

Every arrow becomes a stored reference or hash. A report is invalid if any required source object cannot be resolved and checksum-verified.

## 18. Transaction and Consistency Design

### 18.1 Experiment Creation Transaction

In one PostgreSQL transaction:

1. Insert or resolve the manifest artifact.
2. Insert the experiment.
3. Insert conditions.
4. Insert deterministic shards.
5. Insert outbox events.

The outbox publisher sends Redis messages after commit and marks events published. Duplicate publication is expected and handled by shard identity.

### 18.2 Shard Lease Transaction

A worker claims a shard using `SELECT ... FOR UPDATE SKIP LOCKED` or a compare-and-swap update on status/lease expiry. It records owner, expiry, attempt, and heartbeat. Completion requires an available output partition. Expired leases are reclaimable with the same shard ID and seeds.

### 18.3 Turn Commit Transaction

Object publication precedes the PostgreSQL turn-index transaction. The database transaction:

1. Inserts or resolves artifact rows by checksum and classification.
2. Inserts `session_turn` under its unique `(session_id, turn_index)` constraint.
3. Updates session cursor, budget, and latest checkpoint reference using optimistic versioning.
4. Inserts an outbox event.

If the transaction fails after object publication, unreferenced content-addressed objects are safe and later garbage-collected. Conflicting turn content is never overwritten.

### 18.4 Policy Promotion Transaction

Promotion locks the target alias, verifies mandatory gate records and candidate status, records the prior policy, inserts an immutable promotion event, updates the alias pointer, and emits an outbox/audit event. Rollback creates a new promotion record pointing to the prior version; history is never rewritten.

## 19. Data Quality Rules

### 19.1 Ingestion Validation

- Required identity and provenance fields are non-null.
- All IDs conform to expected UUID form.
- Unit-interval and finite-number constraints pass.
- `reward_total` equals the decomposed signed sum within tolerance.
- Exactly one of valid terminal/truncation combinations is recorded.
- Action belongs to the declared action-space version.
- Policy observation contains only allowlisted fields.
- Public and privileged records share identity and turn index but differ in classification/schema.
- No raw hidden-reasoning fields or known secret patterns appear.
- Row uniqueness holds for `(episode_id, turn_index)` within a condition.

### 19.2 Partition Validation

- Parquet physical schema matches its registry version.
- Partition path values match row values.
- Row count and checksum match metadata.
- Calibration, policy-selection, validation, and held-out members are disjoint under the signed split manifest.
- Every primary condition has its matched-seed counterpart or a recorded exclusion.
- Truncation, missingness, and provider-failure rates are reported before analysis.

### 19.3 Feature Leakage Validation

Training readers enumerate allowed columns rather than excluding forbidden columns. CI and job startup reject any feature whose lineage classification is `privileged_synthetic`, `future_human_restricted`, or `prohibited`. Nested fields are inspected recursively.

## 20. Retention and Deletion

### 20.1 Simulation Defaults

| Data | Default retention | Disposal |
|---|---|---|
| Signed manifests and final split definitions | Project lifetime plus institutional archive period | Archive; do not mutate |
| Primary trajectories and evaluation artifacts | Through dissertation reproducibility period | Archive or lifecycle to cold storage |
| Temporary uploads | 24 hours | Automated deletion if unreferenced |
| Failed/incomplete shard artifacts | 14 days | Delete after incident/debug window |
| LangGraph synthetic checkpoints | 90 days after experiment completion unless selected for evidence | Batch deletion by thread |
| Redis stream entries | Until durable acknowledgement plus bounded operational window | Trim by acknowledged ID and age |
| Redacted traces | Provider/configured retention, minimized | Automated expiry |
| Audit and promotion records | Project lifetime | Archive, append-only |

### 20.2 Future Human Data

No default is assumed. Before activation, governance must define lawful basis/consent, collection minimization, regional storage, access roles, participant withdrawal, deletion propagation, backup expiry, trace redaction, and study-specific retention. Human deletion must cover application rows, checkpoints, object artifacts, caches, tracing, and derived identifiable exports.

## 21. Access Control

| Role/workload | PostgreSQL | Public objects | Privileged objects | MLflow | Secrets |
|---|---|---|---|---|---|
| API service | Session and command metadata | Signed read links | No direct read by default | Policy metadata read | Scoped runtime secrets |
| Graph worker | Session/checkpoint metadata | Turn write/read | Synthetic turn write only | Promoted policy read | Model/sandbox credentials |
| Rollout worker | Shard lease/status | Partition write | Restricted partition write | No promotion rights | Scoped provider credentials |
| Trainer | Training job metadata | Allowlisted partition read | No access | Candidate write | Training-scoped credentials |
| Evaluator | Evaluation metadata | Held-out read | Scoped declared-metric read | Candidate read, gate write through API | Judge credentials if needed |
| Researcher | Experiment/report metadata | Authorized read | No default raw access | Run read | None |
| Privileged evaluator | Audited metadata access | Read | Time-bound audited read | Run read | None |
| Operator | Operational metadata | Lifecycle administration | No content access by default | Promotion through application API | Secret administration without value read where possible |

Database roles use least privilege. Services do not share one owner credential.

## 22. Encryption and Integrity

- TLS is required for all database, Redis, object, MLflow, and provider connections.
- PostgreSQL and object storage use encryption at rest.
- Privileged object prefixes use a distinct key and access policy where supported.
- Manifest, trajectory, policy, calibration, metric, and report artifacts use SHA-256 integrity checks.
- Signed manifests and promotion records use an application signing key managed outside the database.
- Artifact metadata records encryption classification but never stores key material.
- Restore procedures verify checksums before restored artifacts become available.

## 23. Backup and Recovery

### 23.1 PostgreSQL

- Automated snapshots and point-in-time recovery for staging/production-like environments.
- Recovery point objective: 15 minutes for operational metadata.
- Recovery time objective: 4 hours for dissertation operations.
- Quarterly or pre-experiment restore rehearsal.
- `langgraph`, `app`, and `catalog` schemas restored consistently from the same database recovery point.

### 23.2 Object Storage

- Versioning enabled for canonical buckets.
- Lifecycle protection for manifests, primary trajectories, policies, gates, and reports.
- Cross-region replication is optional for the dissertation but recommended for future human deployment subject to data residency.
- PostgreSQL artifact indexes can be rebuilt from object manifests, but the database remains authoritative for lifecycle state.

### 23.3 Redis and Telemetry

Redis loss may require republishing pending work from PostgreSQL/outbox state. Telemetry loss does not invalidate committed scientific artifacts, although observability gaps are recorded.

## 24. Schema Evolution and Migrations

### 24.1 PostgreSQL

- Alembic owns only `app`, `catalog`, and approved future `auth` objects.
- Expand-and-contract migrations are used for deployed compatibility.
- Destructive changes require backup, artifact compatibility analysis, and explicit approval.
- Application startup checks a supported migration range and fails readiness when incompatible.

### 24.2 Parquet

- Every dataset declares a semantic schema name and version.
- Additive nullable fields may remain within a compatible minor version.
- Renames, type changes, semantic changes, or required fields create a new major schema version.
- Readers specify supported versions and do not coerce unknown privileged fields into policy features.
- Migration jobs write new objects; they never overwrite source artifacts.

### 24.3 Contracts and Lineage

JSON Schema, Arrow Schema, SQL migration revision, action-space version, observation version, and policy artifact version are tracked independently. A compatibility matrix is stored with each release.

## 25. Monitoring and Alerts

Alert on:

- Unpublished outbox backlog.
- Expired shard leases and retry spikes.
- Artifact checksum or schema validation failures.
- Public/privileged join mismatches.
- Feature-leakage validation failures.
- Database connection saturation, replication lag, or backup failure.
- Checkpoint growth and stale active sessions.
- Object-store incomplete upload accumulation.
- Missing mandatory gate records for promotion attempts.
- Human-mode records appearing while the mode is disabled.
- Secret-like patterns or prohibited fields detected in trajectories/traces.

## 26. Requirement Traceability

| Data design area | Primary PRD requirements |
|---|---|
| Experiment metadata and splits | FR-EXP-001 through FR-EXP-006, NFR-REP-001 through NFR-REP-005 |
| Simulator truth separation | FR-SIM-002, FR-SIM-006, FR-TRJ-002, NFR-SEC-002 |
| CBFM feature and calibration lineage | FR-CBFM-001, FR-CBFM-005 through FR-CBFM-009 |
| Tracker summaries and evidence masks | FR-TRK-002, FR-TRK-005 through FR-TRK-009 |
| Policy artifacts and propensities | FR-POL-005 through FR-POL-009 |
| Structured evidence and no hidden reasoning | FR-STU-004, FR-STU-005, FR-STU-007 |
| Immutable trajectories | FR-TRJ-001 through FR-TRJ-003 |
| Training and promotion | FR-RL-001 through FR-RL-005 |
| Evaluation metrics and gates | FR-EVL-001 through FR-EVL-009 |
| Security, privacy, retention | SEC-SBX-001 through SEC-SBX-005, NFR-SEC-001 through NFR-PRV-002 |
| Observability and cost | NFR-OBS-001 through NFR-COST-002 |

## 27. Acceptance Criteria

This design is complete when:

- Operational, catalog, research, LangGraph, and MLflow ownership boundaries are explicit.
- All PostgreSQL entities have stable identities, constraints, lifecycle rules, and required indexes.
- Public and privileged trajectories use separate schemas, objects, credentials, and consumers.
- Policy training is technically unable to mount or deserialize privileged simulator truth.
- Manifest-to-report lineage is resolvable through immutable hashes and references.
- Turn, shard, experiment, and promotion transactions define idempotency and crash recovery.
- Redis loss can be recovered from PostgreSQL and object-store state.
- Schema migration rules prevent silent semantic changes.
- Simulation retention is defined and future human retention remains governance-gated.
- Backup, restore, integrity, access, and monitoring rules are specified.
- Every D2 data-flow, lineage, and database schema diagram compiles successfully.
