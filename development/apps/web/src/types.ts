export type HealthResponse = {
  status: "ok";
  service: string;
  version: string;
};

export type EvidenceCategory =
  | "correct"
  | "misconception"
  | "conflicting"
  | "uncertain"
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
