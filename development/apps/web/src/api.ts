import type {
  BenchmarkReplaySnapshot,
  ExperimentSummary,
  HealthResponse,
  LiveEvaluationSnapshot,
  LiveEvaluationStatus,
  SessionSnapshot,
  TaskView,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (typeof body.detail === "string") {
        message = body.detail;
      }
    } catch {
      // Preserve the status-based fallback when a response is not JSON.
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return readJson<HealthResponse>(await fetch("/api/health", { signal }));
}

export async function createSession(
  idempotencyKey: string,
  taskId = "mutable-list-aliasing",
  signal?: AbortSignal,
): Promise<SessionSnapshot> {
  const response = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ idempotency_key: idempotencyKey, task_id: taskId }),
    signal,
  });
  return readJson<SessionSnapshot>(response);
}

export async function listTasks(signal?: AbortSignal): Promise<TaskView[]> {
  return readJson<TaskView[]>(await fetch("/api/tasks", { signal }));
}

export async function getSession(
  sessionId: string,
  signal?: AbortSignal,
): Promise<SessionSnapshot> {
  return readJson<SessionSnapshot>(
    await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, { signal }),
  );
}

export async function submitTurn(
  sessionId: string,
  responseText: string,
  idempotencyKey: string,
): Promise<SessionSnapshot> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/turns`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      response_text: responseText,
      idempotency_key: idempotencyKey,
    }),
  });
  return readJson<SessionSnapshot>(response);
}

export async function getExperimentSummary(signal?: AbortSignal): Promise<ExperimentSummary> {
  return readJson<ExperimentSummary>(await fetch("/api/experiment-summary", { signal }));
}

export async function startBenchmarkReplay(
  idempotencyKey: string,
  signal?: AbortSignal,
): Promise<BenchmarkReplaySnapshot> {
  const response = await fetch("/api/benchmark-replays", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ idempotency_key: idempotencyKey }),
    signal,
  });
  return readJson<BenchmarkReplaySnapshot>(response);
}

export async function revealBenchmarkOutcome(
  replayId: string,
): Promise<BenchmarkReplaySnapshot> {
  return readJson<BenchmarkReplaySnapshot>(
    await fetch(`/api/benchmark-replays/${encodeURIComponent(replayId)}/reveal`, {
      method: "POST",
    }),
  );
}

export async function getLiveEvaluationStatus(
  signal?: AbortSignal,
): Promise<LiveEvaluationStatus> {
  return readJson<LiveEvaluationStatus>(await fetch("/api/live-evaluations/status", { signal }));
}

export async function startLiveEvaluation(
  idempotencyKey: string,
  caseId: string,
): Promise<LiveEvaluationSnapshot> {
  const response = await fetch("/api/live-evaluations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ idempotency_key: idempotencyKey, case_id: caseId }),
  });
  return readJson<LiveEvaluationSnapshot>(response);
}

export async function revealLiveEvaluation(runId: string): Promise<LiveEvaluationSnapshot> {
  return readJson<LiveEvaluationSnapshot>(
    await fetch(`/api/live-evaluations/${encodeURIComponent(runId)}/reveal`, {
      method: "POST",
    }),
  );
}
