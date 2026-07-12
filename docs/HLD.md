<!-- docs/HLD.md -->

# High-Level Design

## Socratic Multi-Agent Tutoring and Evaluation Platform

| Field | Value |
|---|---|
| Status | Draft for engineering review |
| Design level | System and service architecture |
| Product requirements | `docs/PRD.md` |
| Engineering strategy | `docs/engineering-plan.md` |
| Initial mode | Simulation-first research platform |
| Future mode | Human tutoring after governance approval |

## 1. Purpose

This document defines the high-level architecture for the Socratic Multi-Agent Tutoring and Evaluation Platform. It translates the product requirements into system boundaries, deployable runtimes, data ownership, integration contracts, security zones, scaling rules, and operational behavior.

The design optimizes for two distinct workloads:

- Low-latency, resumable LangGraph sessions used by the Glass Box and future human clients.
- Reproducible, horizontally parallel simulation and evaluation jobs used to train and compare tutoring policies.

The same versioned domain packages support both workloads, but serving and training do not share mutable runtime state.

## 2. Architectural Goals

The architecture shall:

- Preserve strict separation between simulator truth, tracker belief, graph workflow state, and policy observations.
- Run deterministic surrogate episodes without network dependencies.
- Support LLM-backed episodes without coupling core mathematics to a model provider.
- Scale simulation rollouts independently from interactive sessions.
- Make every reported result traceable to immutable inputs and versioned artifacts.
- Isolate untrusted code outside the application host.
- Resume graph sessions and batch jobs after process failure.
- Keep policy promotion explicit and auditable.
- Permit future human interaction without exposing simulator-only capabilities.

## 3. Architecture Decisions

| ID | Decision | Rationale | Consequence |
|---|---|---|---|
| ADR-HLD-001 | Use a modular monorepo. | Shared Pydantic contracts and mathematical semantics must evolve atomically during research. | Deployables share a release but scale independently. |
| ADR-HLD-002 | Deploy serving, rollout, training, evaluation, and sandbox workloads separately. | Their latency, security, and compute profiles differ. | Runtime boundaries are explicit without separate source repositories. |
| ADR-HLD-003 | Run Gymnasium environments in rollout-worker processes. | Remote environment calls would weaken seeded replay and add avoidable latency. | Training remains independent of the live API and graph checkpoint database. |
| ADR-HLD-004 | Use immutable Parquet trajectories as the training boundary. | Scientific results need inspectable, replayable inputs. | Training does not consume live session state. |
| ADR-HLD-005 | Use PostgreSQL for operational metadata and LangGraph checkpoints. | Sessions, checkpoints, promotions, and manifests require transactions and durable queries. | Large arrays and trajectory payloads are stored elsewhere. |
| ADR-HLD-006 | Use S3-compatible object storage for trajectories and large artifacts. | Columnar scientific data and model artifacts exceed checkpoint/database concerns. | Objects are content-addressed and referenced from metadata. |
| ADR-HLD-007 | Use Redis Streams for durable work dispatch and WebSockets for live client events. | Pub/Sub alone does not provide durable job acknowledgement or replay. | Consumers use groups, bounded retries, and dead-letter handling. |
| ADR-HLD-008 | Freeze one policy artifact per episode. | Mid-episode policy changes invalidate comparisons and complicate recovery. | New policies require explicit offline evaluation and promotion. |
| ADR-HLD-009 | Use external microVM sandboxes for untrusted code. | AST filtering and host subprocesses are not security boundaries. | Local execution is allowed only for trusted tests in development. |
| ADR-HLD-010 | Use LangSmith for graph/LLM diagnostics and MLflow for scientific lineage. | Operational traces and canonical research artifacts have different retention and query needs. | Neither tool replaces the primary operational or artifact stores. |
| ADR-HLD-011 | Disable human mode by default. | Simulation success does not satisfy ethics, privacy, or educational-efficacy requirements. | Human endpoints require a separate feature gate and deployment configuration. |

## 4. System Context

```d2
direction: right

users: "Users and Stakeholders" {
  researcher: "Researcher"
  evaluator: "Evaluator / Reviewer"
  operator: "System Operator"
  student: "Future Student"
}

platform: "Socratic Tutoring Platform" {
  glass_box: "Glass Box Web Client"
  api: "FastAPI Gateway"
  graph: "LangGraph Runtime"
  research: "Simulation, Training, and Evaluation"
}

external: "External Systems" {
  identity: "Institutional Identity Provider"
  llm: "LLM Providers"
  sandbox: "MicroVM Sandbox Provider"
  telemetry: "LangSmith / Observability Backends"
  object_store: "S3-Compatible Object Storage"
}

users.researcher -> platform.glass_box: "configures and inspects experiments"
users.evaluator -> platform.glass_box: "reviews synthetic evidence and reports"
users.operator -> platform.api: "operates deployments and promotions"
users.student -> platform.glass_box: "future approved tutoring session"

platform.glass_box -> platform.api: "HTTPS and WebSocket"
platform.api -> platform.graph: "start, resume, and inspect sessions"
platform.api -> platform.research: "submit and inspect batch jobs"

platform.api -> external.identity: "OIDC / OAuth2 in human mode"
platform.graph -> external.llm: "typed generation requests"
platform.graph -> external.sandbox: "isolated code execution"
platform.graph -> external.telemetry: "redacted graph traces"
platform.research -> external.object_store: "trajectories and artifacts"
platform.research -> external.telemetry: "metrics and run lineage"
```

### 4.1 Context Boundaries

- Researchers and evaluators access synthetic truth only through privileged, labeled Glass Box views.
- Future students access only public prompts, their submissions, and safe feedback.
- Model providers receive the minimum prompt context required by their role.
- Sandbox providers receive code, tests, limits, and an idempotency key, but no unrelated learner profile.
- External telemetry receives redacted traces and identifiers that are safe for the configured environment.

## 5. Logical System Architecture

```d2
direction: right

clients: "Client Layer" {
  web: "React Glass Box"
  cli: "Research CLI"
}

edge: "Application Edge" {
  api: "FastAPI Gateway"
  auth: "Authentication and Authorization"
  ws: "WebSocket Session Hub"
}

serving: "Interactive Serving Plane" {
  graph_worker: "LangGraph Worker"
  policy_adapter: "Policy Adapter"
  model_gateway: "Model Gateway"
  guardrail: "Pedagogical Guardrail"
  tracker: "Epistemic Tracker"
  trajectory_writer: "Trajectory Writer"
}

research: "Research Compute Plane" {
  coordinator: "Experiment Coordinator"
  rollout: "Rollout Workers"
  trainer: "Policy Trainer"
  evaluator: "Evaluation Workers"
  report_builder: "Report Builder"
}

domain: "Shared Domain Packages" {
  contracts: "Versioned Pydantic Contracts"
  env: "Gymnasium POMDP Environment"
  cbfm: "CBFM and Transition Kernels"
  policies: "Policy Implementations"
  prompts: "Versioned Prompts and Templates"
}

data: "Data Plane" {
  postgres: "PostgreSQL\nSessions, Checkpoints, Metadata"
  redis: "Redis Streams\nJobs and Live Events"
  objects: "Object Storage\nParquet and Artifacts"
  mlflow: "MLflow\nRuns and Policy Registry"
}

external: "External Execution" {
  llm: "LLM Providers"
  sandbox: "MicroVM Sandbox"
  langsmith: "LangSmith"
  metrics: "OpenTelemetry / Prometheus / Sentry"
}

clients.web -> edge.api: "REST"
clients.web -> edge.ws: "WebSocket"
clients.cli -> edge.api: "experiment and artifact API"

edge.api -> edge.auth: "authorize request"
edge.api -> serving.graph_worker: "start or resume graph"
edge.api -> research.coordinator: "submit batch manifest"
edge.ws <-> serving.graph_worker: "stream events and resume input"

serving.graph_worker -> serving.policy_adapter: "PolicyObservation"
serving.policy_adapter -> serving.graph_worker: "PolicyDecision"
serving.graph_worker -> serving.model_gateway: "generation request"
serving.graph_worker -> serving.guardrail: "candidate tutor prompt"
serving.graph_worker -> serving.tracker: "ObservationEvidence"
serving.graph_worker -> serving.trajectory_writer: "committed turn"

research.coordinator -> data.redis: "durable rollout and evaluation jobs"
data.redis -> research.rollout: "claimed job shard"
research.rollout -> domain.env: "construct local environment"
research.rollout -> data.objects: "immutable trajectories"
data.objects -> research.trainer: "training partitions"
research.trainer -> data.mlflow: "candidate policy artifact"
data.objects -> research.evaluator: "held-out trajectories"
data.mlflow -> research.evaluator: "candidate and incumbent policies"
research.evaluator -> research.report_builder: "gate metrics"
research.report_builder -> data.objects: "versioned reports"

serving.graph_worker -> data.postgres: "checkpoints and session metadata"
serving.trajectory_writer -> data.objects: "trajectory records"
serving.policy_adapter -> data.mlflow: "load frozen promoted policy"
edge.ws -> data.redis: "ephemeral live-event fan-out"

serving.model_gateway -> external.llm: "bounded provider calls"
serving.graph_worker -> external.sandbox: "isolated execution request"
serving.graph_worker -> external.langsmith: "redacted graph trace"
edge.api -> external.metrics: "service telemetry"
serving.graph_worker -> external.metrics: "node and provider telemetry"
research.coordinator -> external.metrics: "job and cost telemetry"

serving.graph_worker -> domain.contracts: "validate graph state"
serving.graph_worker -> domain.prompts: "resolve prompt versions"
serving.tracker -> domain.cbfm: "subjective fast-state model"
serving.policy_adapter -> domain.policies: "invoke selected policy"
domain.env -> domain.cbfm: "true simulator dynamics"
domain.env -> domain.contracts: "validate observations and evidence"
```

### 5.1 Architectural Planes

#### Client Layer

The React Glass Box supports synthetic experiment inspection and future student interaction. The research CLI provides repeatable automation for manifests, batches, artifact retrieval, and report generation. Neither client directly accesses databases or model providers.

#### Application Edge

FastAPI is the only public application entry point. It validates request schemas, enforces role and mode authorization, creates correlation IDs, applies rate and budget limits, and routes work to serving or research runtimes. WebSockets carry live turn events and future human input; durable state remains in PostgreSQL rather than the socket process.

#### Interactive Serving Plane

LangGraph workers execute resumable sessions. They are optimized for I/O concurrency and bounded external calls. The graph owns workflow transitions, while specialist packages own domain behavior. Workers are stateless between invocations except for in-process caches; PostgreSQL checkpoints and artifact references are authoritative.

#### Research Compute Plane

The experiment coordinator validates a manifest and produces immutable job shards. Rollout workers construct Gymnasium environments in process and write trajectories. Trainers consume approved trajectory partitions or environment factories. Evaluators run held-out gates. Report builders produce versioned tables, figures, and decision records.

#### Shared Domain Packages

Shared packages contain contracts, mathematics, policies, prompts, and environment logic. They are libraries, not remotely deployed services. This preserves deterministic execution and prevents distributed calls inside each environment step.

#### Data Plane

PostgreSQL stores operational records and durable graph checkpoints. Redis Streams carries bounded work and event notifications, but is not the system of record. Object storage contains canonical trajectories, large tracker artifacts, reports, and policy files. MLflow indexes scientific runs and controls policy lifecycle metadata.

## 6. Component Responsibilities

| Component | Responsibilities | Explicit exclusions |
|---|---|---|
| FastAPI gateway | Request validation, authentication hooks, authorization, session APIs, experiment APIs, quotas, correlation IDs | Graph logic, training, code execution |
| WebSocket hub | Live event delivery, heartbeat, reconnect cursors, validated human input forwarding | Durable session state |
| LangGraph worker | Node routing, checkpoint lifecycle, bounded retries, interrupt/resume, turn commit | Owning latent state or training policies |
| Gymnasium environment | True state, fast/slow transitions, reward, termination, evidence orchestration | Exposing truth to policies |
| Epistemic tracker | Particle belief, evidence correction, propagation, uncertainty summaries | Mutating simulator truth |
| Policy adapter | Observation projection, feature audit, frozen artifact loading, decision validation | Rendering prompts or online learning |
| Model gateway | Provider-neutral requests, timeouts, usage, retries, structured response validation | Choosing pedagogical actions |
| Guardrail | Leakage checks, rejection reasons, bounded rewrite routing, safe fallback selection | Updating learner beliefs |
| Sandbox adapter | Isolated execution requests and normalized telemetry | Host-process execution in deployed modes |
| Trajectory writer | Immutable turn records, privileged/public separation, object-store commit | Training or report interpretation |
| Experiment coordinator | Manifest validation, split checks, budgets, shard lifecycle | Executing environment transitions |
| Rollout worker | Local environment execution, matched seeds, trajectory production | Calling live session APIs |
| Policy trainer | Bandit/Q-learning fitting, artifact production, training diagnostics | Serving alias mutation |
| Evaluation worker | Held-out metrics, ablations, confidence intervals, gate outcomes | Retuning on held-out data |
| Report builder | Reproducible figures, tables, provenance, accepted/rejected/inconclusive decision records | Altering source results |

## 7. State Ownership and Trust Boundaries

### 7.1 State Classes

| State class | Owner | Store | Permitted consumers |
|---|---|---|---|
| Simulator truth | Gymnasium environment | Privileged trajectory partition; optional encrypted debug artifact | Simulator and authorized evaluation only |
| Tracker particles | Epistemic tracker | External particle artifact when required; otherwise worker memory | Tracker only |
| Tracker summaries | Epistemic tracker | Graph checkpoint and public trajectory columns | Policy adapter, tutor, evaluator, synthetic UI |
| Graph workflow state | LangGraph | PostgreSQL checkpointer | Graph runtime and authorized operators |
| Policy observation | Policy adapter | Public trajectory columns | Active policy and evaluation |
| Policy artifact | Trainer/registry | Object storage plus MLflow metadata | Policy adapter and evaluators |
| Human submission | Future client/API | Approved operational store and checkpoint | Graph, sandbox, tracker under approved policy |

### 7.2 Enforcement

- `TrueStudentState` is defined in a simulator-private package, not the shared policy contract package.
- Policy loaders accept a `PolicyObservation` schema and reject extra fields.
- Privileged trajectory data uses a separate object prefix, schema, and authorization policy.
- Human-mode builds exclude synthetic truth endpoints and controls.
- Graph field ownership is validated at node boundaries; only the tracker may commit tutor-facing belief fields.
- Training feature manifests are compared with an allowlist before model fitting and evaluation.

## 8. Primary Runtime Flows

### 8.1 Synthetic Interactive Episode

1. The API validates an episode request and resolves a promoted policy version.
2. The graph worker initializes a thread and a simulator-private true state.
3. The tracker emits initial posterior summaries.
4. The policy adapter projects an allowed observation and returns a directive/target decision.
5. The tutor generator renders a candidate prompt through the model gateway or template engine.
6. The guardrail accepts the prompt, requests a bounded rewrite, or selects a safe fallback.
7. The environment computes the realized `TutorControl`, prompt load, `F_t`, and `B_t`.
8. The student simulator generates public output and private diagnostic evidence; code runs in the sandbox when needed.
9. The tracker corrects its belief using active evidence and propagates to the next turn.
10. The graph commits a checkpoint and trajectory turn before emitting the client event.

### 8.2 Batch Rollout and Training

1. The coordinator validates and signs an experiment manifest.
2. It creates deterministic shards keyed by condition, profile, task, repetition, and seed.
3. Rollout workers claim shards from Redis Streams and instantiate local environments.
4. Workers commit immutable trajectory partitions and mark shard completion transactionally.
5. The trainer loads only approved public trajectory columns or constructs training environments locally.
6. It writes a candidate policy artifact and training diagnostics to object storage/MLflow.
7. Evaluation workers run feature audits, replay checks, held-out gates, and cost/safety comparisons.
8. An authorized operator explicitly promotes or rejects the candidate.

### 8.3 Future Human Session

1. The API authenticates the student and verifies human-mode approval and entitlement.
2. LangGraph loads tracker summaries and a frozen policy; no simulator truth exists.
3. The graph produces and guardrails a prompt, then calls a dynamic interrupt.
4. PostgreSQL stores the checkpoint; the WebSocket hub publishes the prompt.
5. The client submits text or code with a turn idempotency key.
6. Code executes in a microVM and returns normalized evidence.
7. The graph resumes using the same thread ID, updates the tracker, and continues.

## 9. External Interfaces

### 9.1 Client API Families

| API family | Purpose | Interaction style |
|---|---|---|
| `/v1/sessions` | Create, inspect, resume, and end interactive sessions | REST plus WebSocket events |
| `/v1/experiments` | Validate manifests and submit/cancel batch experiments | REST and asynchronous job status |
| `/v1/artifacts` | Retrieve authorized reports, trajectories, and provenance | REST with signed object references |
| `/v1/policies` | List candidates, inspect gates, promote, or roll back | Administrative REST |
| `/v1/health` | Liveness, readiness, and dependency status | Operational REST |

Exact payloads, status codes, event schemas, and idempotency behavior are specified in the LLD.

### 9.2 Provider Interfaces

All external adapters implement typed protocols and normalize provider-specific behavior:

- `ModelGateway`: generation, structured classification, usage, latency, retry metadata.
- `SandboxGateway`: source, tests, limits, idempotency key, normalized execution result.
- `ArtifactStore`: content-addressed put/get, atomic publish, checksums, signed access.
- `PolicyRegistry`: candidate registration, gate attachment, promotion, rollback, resolution.
- `TraceSink`: redacted spans, metrics, errors, and environment tags.

## 10. Deployment Architecture

```d2
direction: down

internet: "User Network" {
  browser: "Browser / Research Client"
}

cloud: "Application Cloud Account" {
  edge: "Public Edge" {
    load_balancer: "TLS Load Balancer"
    api_instances: "FastAPI Instances\nStateless"
  }

  private_compute: "Private Compute" {
    graph_workers: "LangGraph Workers\nAutoscaled on concurrency"
    batch_workers: "Rollout / Evaluation Workers\nScale to zero"
    trainer: "Training Job\nCPU by default"
  }

  managed_data: "Managed Data Services" {
    postgres: "PostgreSQL\nMulti-AZ in production"
    redis: "Redis Streams"
    object_store: "Object Storage"
    secrets: "Secrets Manager"
  }

  observability: "Observability" {
    otel: "OpenTelemetry Collector"
    metrics: "Prometheus / Grafana"
    errors: "Sentry"
    mlflow: "MLflow Tracking"
  }
}

providers: "Restricted External Providers" {
  llm: "LLM API"
  sandbox: "Ephemeral MicroVM Sandbox"
  langsmith: "LangSmith"
}

internet.browser -> cloud.edge.load_balancer: "TLS 1.2+"
cloud.edge.load_balancer -> cloud.edge.api_instances: "HTTPS / WSS"
cloud.edge.api_instances -> cloud.private_compute.graph_workers: "private service request"
cloud.edge.api_instances -> cloud.managed_data.redis: "job and event command"

cloud.private_compute.graph_workers -> cloud.managed_data.postgres: "checkpoint transaction"
cloud.private_compute.graph_workers -> cloud.managed_data.object_store: "trajectory append"
cloud.private_compute.batch_workers -> cloud.managed_data.redis: "claim and acknowledge shards"
cloud.private_compute.batch_workers -> cloud.managed_data.object_store: "trajectory partitions"
cloud.private_compute.trainer -> cloud.managed_data.object_store: "read trajectories, write policies"
cloud.private_compute.trainer -> cloud.observability.mlflow: "run and policy metadata"

cloud.edge.api_instances -> cloud.managed_data.secrets: "runtime secret lookup"
cloud.private_compute.graph_workers -> cloud.managed_data.secrets: "provider credentials"
cloud.private_compute.graph_workers -> providers.llm: "egress allowlist"
cloud.private_compute.graph_workers -> providers.sandbox: "egress allowlist"
cloud.private_compute.graph_workers -> providers.langsmith: "redacted trace egress"

cloud.edge.api_instances -> cloud.observability.otel: "service spans"
cloud.private_compute.graph_workers -> cloud.observability.otel: "graph and provider spans"
cloud.private_compute.batch_workers -> cloud.observability.otel: "job spans"
cloud.observability.otel -> cloud.observability.metrics: "metrics and alerts"
cloud.observability.otel -> cloud.observability.errors: "error events"
```

### 10.1 Environment Strategy

#### Local

- Docker Compose runs API, graph worker, PostgreSQL, Redis, MLflow, and the web client.
- SQLite checkpointing may be used for isolated unit/integration development.
- Template providers and trusted sandbox fixtures are the default.
- DuckDB queries local Parquet artifacts.

#### Staging

- Mirrors production service boundaries with reduced capacity.
- Uses synthetic data only.
- Requires an external microVM sandbox for any untrusted execution.
- Exercises provider fallbacks, checkpoint recovery, policy rollback, and redaction.

#### Dissertation Batch

- Modal jobs provide scale-to-zero rollout and evaluation workers.
- CPU workers are used for tabular policies and statistical analysis.
- Concurrency is limited by provider quotas and experiment budgets, not available container count alone.

#### Production-Like Human Pilot

- Requires managed PostgreSQL, durable backups, role-based access, approved identity integration, retention/deletion workflows, and incident response.
- Human mode is a separate configuration and deployment approval, not a runtime toggle available to ordinary users.

## 11. Scaling and Capacity Model

### 11.1 Scaling Units

| Workload | Scaling unit | Primary constraint | Strategy |
|---|---|---|---|
| API | Request or socket connection | Connection count and request latency | Horizontal stateless replicas |
| LangGraph | Active graph run | LLM latency and provider quotas | Autoscale workers; cap per-provider concurrency |
| Rollout | Episode shard | API budget, environment duration | Scale-to-zero batch workers with bounded parallelism |
| Training | Policy/configuration pair | Trajectory volume | Single CPU job initially; Ray only after evidence of need |
| Evaluation | Metric/gate shard | Judge calls and bootstrap cost | Parallel independent jobs |
| Sandbox | Code submission | MicroVM startup and provider quota | Provider-managed ephemeral instances |

### 11.2 Backpressure

- API requests are rejected with a typed quota response before unbounded queuing.
- Redis consumer groups limit in-flight shards per worker.
- Provider adapters enforce global and per-experiment semaphores.
- Batch coordinators pause new shard dispatch as budgets approach limits.
- WebSocket event buffers are bounded; clients recover from durable checkpoint cursors rather than receiving an unlimited backlog.

## 12. Reliability and Failure Handling

| Failure | Detection | Recovery | Data integrity rule |
|---|---|---|---|
| Graph worker termination | Missing heartbeat or request failure | Resume from last PostgreSQL checkpoint | Committed turns are never regenerated silently |
| LLM timeout/rate limit | Adapter timeout and typed provider error | Bounded retry, alternate allowed provider, or template fallback | All attempts and selected output are recorded |
| Guardrail schema failure | Pydantic validation error | Treat as rejection; retry or safe fallback | Invalid output never passes |
| Sandbox timeout/failure | Typed sandbox status | Record evidence and continue/truncate according to manifest | No host fallback in deployed modes |
| Redis worker failure | Pending-entry timeout | Reclaim shard with same deterministic shard ID | Object commit and shard acknowledgement are idempotent |
| Object upload interruption | Missing checksum/final marker | Retry content-addressed upload | Readers ignore incomplete objects |
| PostgreSQL outage | Readiness failure and transaction error | Stop accepting resumable work; retry after recovery | No in-memory-only successful commit |
| Trainer interruption | Job checkpoint or shard state | Resume from durable boundary | Candidate artifact is published only after completion |
| Evaluation failure | Missing gate result | Rerun failed gate shard | Policy cannot promote with incomplete required gates |
| WebSocket disconnect | Heartbeat timeout | Client reconnects using session/checkpoint cursor | Socket delivery is not considered a state commit |

All external side effects use stable idempotency keys derived from experiment/session, turn, node, operation type, and attempt policy.

## 13. Security Architecture

### 13.1 Security Zones

- **Public zone:** TLS load balancer and authenticated API endpoints.
- **Application zone:** API and graph workers with no direct public ingress.
- **Research zone:** batch workers and trainers with access only to approved artifact prefixes and metadata.
- **Privileged synthetic zone:** true-state artifacts and synthetic administrative views.
- **Sandbox zone:** disposable external microVMs with no trust relationship to application hosts.
- **Data zone:** managed databases, object storage, secrets, and audit records.

### 13.2 Mandatory Controls

- Deny-by-default role and mode authorization.
- Separate object-store prefixes and credentials for public versus privileged trajectories.
- Workload identities instead of shared long-lived cloud credentials.
- Secret retrieval at runtime; no provider keys in checkpoints, prompts, or artifacts.
- Egress allowlists for LLM, sandbox, LangSmith, and approved telemetry endpoints.
- Request size, rate, token, execution, and cost limits.
- Redaction before traces leave the application account.
- Immutable audit events for policy promotion, rollback, experiment submission, and privileged artifact access.
- No raw hidden reasoning in storage, UI, logs, or evaluator inputs.

## 14. Observability Architecture

### 14.1 Correlation Model

Every operation carries:

- `trace_id`
- `experiment_id` or `session_id`
- `episode_id`
- `turn_id` where applicable
- `graph_version`
- `policy_version`
- `model_version`
- `environment`

These identifiers correlate API spans, graph nodes, provider calls, sandbox executions, trajectories, MLflow runs, and reports.

### 14.2 Systems of Record

| Concern | System | Retention role |
|---|---|---|
| Graph and LLM debugging | LangSmith | Sampled/redacted diagnostic trace |
| Distributed service telemetry | OpenTelemetry | Correlation and export |
| Metrics and alerts | Prometheus/Grafana | Operational trend and alerting |
| Exceptions | Sentry | Error triage |
| Scientific run lineage | MLflow | Parameters, metrics, policy lifecycle |
| Canonical research data | Object storage | Immutable trajectories, reports, artifacts |
| Operational truth | PostgreSQL | Sessions, checkpoints, jobs, promotions, audit metadata |

LangSmith is not the canonical trajectory store, and Redis is not the canonical job-result store.

## 15. Data Lifecycle at a Glance

1. A versioned manifest creates experiment and shard metadata in PostgreSQL.
2. Workers produce immutable public and privileged trajectory partitions.
3. Object checksums and schema validation complete before shard acknowledgement.
4. Trainers consume only approved public features.
5. Evaluators combine public outcomes, controlled privileged fields, and external ratings according to each metric's declared source.
6. MLflow records run lineage and references immutable objects.
7. Reports are content-addressed and linked to source partitions and analysis code.
8. Retention policies archive or delete artifacts by classification; future human data uses a separately approved policy.

The detailed entities, relationships, partitions, and retention rules are defined in `docs/data-flow-and-schema.md`.

## 16. Requirement Traceability

| Architecture area | Primary PRD requirements |
|---|---|
| Manifest and experiment control | FR-EXP-001 through FR-EXP-006, NFR-REP-001 through NFR-REP-005 |
| Local POMDP environment | FR-SIM-001 through FR-SIM-008, FR-CBFM-001 through FR-CBFM-009 |
| Tracker and state ownership | FR-TRK-001 through FR-TRK-009, FR-GRF-002, FR-TRJ-002 |
| Policy and generation | FR-POL-001 through FR-POL-009, FR-GEN-001 through FR-GEN-003 |
| Guardrail and sandbox | FR-GRD-001 through FR-GRD-006, SEC-SBX-001 through SEC-SBX-005 |
| Graph runtime | FR-GRF-001 through FR-GRF-007, NFR-REL-001 through NFR-REL-003 |
| Training and policy lifecycle | FR-TRJ-001 through FR-RL-005 |
| Evaluation and reports | FR-EVL-001 through FR-EVL-009 |
| Glass Box and future human mode | FR-UI-001 through FR-UI-006, NFR-ACC-001 through NFR-ACC-003 |
| Security and privacy | NFR-SEC-001 through NFR-PRV-002 |
| Observability and cost | NFR-OBS-001 through NFR-COST-002 |

## 17. HLD Acceptance Criteria

This design is ready for low-level specification when:

- Every deployable runtime has one primary responsibility and documented exclusions.
- Simulator truth cannot cross into policy or human-facing interfaces by design.
- Interactive serving and research training share contracts but no mutable runtime state.
- The synthetic turn order matches the POMDP timing in the proposal.
- PostgreSQL, Redis Streams, object storage, MLflow, and telemetry have non-overlapping systems-of-record responsibilities.
- All untrusted code paths terminate in an external sandbox boundary.
- Recovery behavior is defined for graph, queue, provider, sandbox, database, artifact, and client failures.
- Policy promotion is offline, gated, explicit, and reversible.
- Human mode remains separately governed and disabled by default.
- All D2 diagrams compile successfully.
