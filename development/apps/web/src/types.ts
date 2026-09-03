export type HealthResponse = {
  status: "ok";
  service: string;
  version: string;
};

export type EvidenceCategory =
  | "correct"
  | "incorrect"
  | "misconception"
  | "conflicting"
  | "uncertain"
  | "incomplete"
  | "empty";

export type TutorAction = "transfer" | "hint" | "clarify" | "probe" | "encourage";

export type Evidence = {
  category: EvidenceCategory;
  confidence: number;
  rationale: string;
  observed_signals: string[];
};

export type TrackerState = {
  concept: string;
  mastery_probability: number;
  observations: number;
  last_evidence: EvidenceCategory | null;
};

export type PolicyDecision = {
  action: TutorAction;
  rationale: string;
};

export type TaskView = {
  task_id: string;
  version: string;
  title: string;
  concept: string;
  instructions: string;
  starter_code: string;
};

export type TurnRecord = {
  turn_number: number;
  idempotency_key: string;
  student_response: string;
  evidence: Evidence;
  tracker_before: TrackerState;
  tracker_after: TrackerState;
  decision: PolicyDecision;
  tutor_prompt: string;
  guardrail: {
    safe: boolean;
    output_prompt: string;
    violations: string[];
  };
  completed_at: string;
};

export type SessionSnapshot = {
  session_id: string;
  task: TaskView;
  initial_prompt: string;
  tracker: TrackerState;
  turns: TurnRecord[];
  created_at: string;
  updated_at: string;
};

export type ReplayCondition =
  | "dialogue_only"
  | "probe_informed"
  | "unrelated_probe"
  | "corrupted_probe";

export type CommittedPrediction = {
  condition: ReplayCondition;
  label: string;
  tracker_score: number;
  policy_threshold: number;
  predicts_success: boolean;
  committed_at_utc: string;
  record_hash: string;
};

export type ReplayEvidence = {
  condition: "probe_informed" | "unrelated_probe";
  label: string;
  response_summary: string;
  execution_summary: string;
  passed_checks: number;
  failed_checks: number;
};

export type RecordedOutcome = {
  summary: string;
  test_warning: string;
  passed_checks: number;
  failed_checks: number;
  revealed_at_utc: string;
};

export type BenchmarkReplaySnapshot = {
  replay_id: string;
  phase: "predictions_committed" | "outcome_revealed";
  source: "recorded_artifact";
  source_data_hash: string;
  source_manifest_hash: string;
  benchmark_version: string;
  case_id: string;
  evaluation_model: string;
  public_summary: string;
  evidence: ReplayEvidence[];
  predictions: CommittedPrediction[];
  outcome: RecordedOutcome | null;
  interpretation: string | null;
  predictions_sealed_before_outcome_reveal: true;
  model_calls_made: 0;
  sandbox_calls_made: 0;
  post_reveal_policy_calls_made: 0;
  claim_boundary: string;
};

export type ExperimentFinding = {
  study_id: string;
  label: string;
  question: string;
  result: string;
  interpretation: string;
  sample: string;
  evidence_type:
    | "external_model_benchmark"
    | "glass_box_simulation"
    | "external_learner_records";
  status: string;
  source_report_hash: string;
};

export type ExperimentSummary = {
  schema_version: 1;
  schema_id: "demo.experiment_summary.v1";
  overarching_question: string;
  findings: ExperimentFinding[];
  claim_boundary: string;
};

export type LiveEvaluationStatus = {
  enabled: boolean;
  available: boolean;
  case_id: string;
  evaluation_model: string;
  reason: string | null;
  claim_boundary: string;
};

export type LiveExecution = {
  passed_checks: number;
  failed_checks: number;
  passed_all_checks: boolean;
  sandbox_id: string | null;
};

export type LivePrediction = {
  condition: "dialogue_only" | "probe_informed";
  label: string;
  tracker_score: number;
  policy_threshold: number;
  predicts_success: boolean;
  committed_at_utc: string;
  record_hash: string;
};

export type LiveEvaluationSnapshot = {
  run_id: string;
  case_id: string;
  source: "live_demo";
  canonical: false;
  phase: "predictions_committed" | "outcome_revealed" | "failed";
  evaluation_model: string;
  public_response: string | null;
  public_assessment: {
    category: EvidenceCategory;
    method: "transparent_development_rule";
    rationale: string;
  } | null;
  evidence_response: string | null;
  evidence_execution: LiveExecution | null;
  predictions: LivePrediction[];
  commitment_hash: string | null;
  committed_at_utc: string | null;
  outcome: {
    response: string;
    execution: LiveExecution;
    revealed_at_utc: string;
  } | null;
  predictions_sealed_before_outcome_reveal: boolean;
  model_calls_made: number;
  sandbox_calls_made: number;
  provider_latency_ms: number;
  failure: {
    stage:
      | "public_generation"
      | "evidence_generation"
      | "evidence_extraction"
      | "evidence_execution"
      | "criterion_generation"
      | "criterion_extraction"
      | "criterion_execution";
    message: string;
  } | null;
  claim_boundary: string;
};
