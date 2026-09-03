import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";

import { createSession, listTasks, submitTurn } from "./api";
import { ExperimentView } from "./ExperimentView";
import { newIdempotencyKey } from "./ids";
import type { SessionSnapshot, TaskView } from "./types";

type SessionState =
  | { kind: "loading" }
  | { kind: "ready"; session: SessionSnapshot }
  | { kind: "error"; message: string };

type FailedSubmission = {
  responseText: string;
  idempotencyKey: string;
};

function readableLabel(value: string): string {
  return value.replaceAll("-", " ").replace(/^./, (character) => character.toUpperCase());
}

export function App() {
  const initialCreateKey = useRef(newIdempotencyKey());
  const selectedTaskId = useRef<string | null>(null);
  const [sessionState, setSessionState] = useState<SessionState>({ kind: "loading" });
  const [availableTasks, setAvailableTasks] = useState<TaskView[]>([]);
  const [responseText, setResponseText] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [failedSubmission, setFailedSubmission] = useState<FailedSubmission | null>(null);
  const [submissionError, setSubmissionError] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<"tutor" | "experiment">("tutor");
  const conversationRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    void listTasks(controller.signal)
      .then(async (tasks) => {
        const firstTask = tasks.at(0);
        if (firstTask === undefined) {
          throw new Error("No practice tasks are available");
        }
        setAvailableTasks(tasks);
        selectedTaskId.current = firstTask.task_id;
        return createSession(initialCreateKey.current, firstTask.task_id, controller.signal);
      })
      .then((session) => setSessionState({ kind: "ready", session }))
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
        setSessionState({
          kind: "error",
          message: error instanceof Error ? error.message : "Unable to start a session",
        });
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (sessionState.kind !== "ready") {
      return;
    }
    const conversation = conversationRef.current;
    if (conversation !== null) {
      conversation.scrollTop = conversation.scrollHeight;
    }
  }, [sessionState]);

  async function sendSubmission(request: FailedSubmission): Promise<void> {
    if (sessionState.kind !== "ready") {
      return;
    }
    setIsSubmitting(true);
    setSubmissionError(null);
    try {
      const session = await submitTurn(
        sessionState.session.session_id,
        request.responseText,
        request.idempotencyKey,
      );
      setSessionState({ kind: "ready", session });
      setResponseText("");
      setFailedSubmission(null);
    } catch (error: unknown) {
      setFailedSubmission(request);
      setSubmissionError(error instanceof Error ? error.message : "Unable to submit response");
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const request = {
      responseText,
      idempotencyKey: newIdempotencyKey(),
    };
    void sendSubmission(request);
  }

  function handleReset(): void {
    const taskId = selectedTaskId.current;
    if (taskId === null) {
      window.location.reload();
      return;
    }
    setSessionState({ kind: "loading" });
    setResponseText("");
    setFailedSubmission(null);
    setSubmissionError(null);
    void createSession(newIdempotencyKey(), taskId)
      .then((session) => setSessionState({ kind: "ready", session }))
      .catch((error: unknown) =>
        setSessionState({
          kind: "error",
          message: error instanceof Error ? error.message : "Unable to reset the session",
        }),
      );
  }

  function handleTaskChange(taskId: string): void {
    selectedTaskId.current = taskId;
    setSessionState({ kind: "loading" });
    setResponseText("");
    setFailedSubmission(null);
    setSubmissionError(null);
    void createSession(newIdempotencyKey(), taskId)
      .then((session) => setSessionState({ kind: "ready", session }))
      .catch((error: unknown) =>
        setSessionState({
          kind: "error",
          message: error instanceof Error ? error.message : "Unable to change task",
        }),
      );
  }

  if (sessionState.kind === "loading") {
    return (
      <main className="centered-state">
        <p role="status">Starting tutoring session...</p>
      </main>
    );
  }

  if (sessionState.kind === "error") {
    return (
      <main className="centered-state">
        <p className="error-message" role="alert">
          {sessionState.message}
        </p>
        <button className="secondary-button" type="button" onClick={handleReset}>
          Retry session
        </button>
      </main>
    );
  }

  const { session } = sessionState;
  const latestTurn = session.turns.at(-1);
  const masteryPercent = Math.round(session.tracker.mastery_probability * 100);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Research proof of concept</p>
          <h1>Socratic Tutor</h1>
        </div>
        <div className="header-controls">
          <nav className="view-tabs" aria-label="Demo view">
            <button
              type="button"
              aria-current={activeView === "tutor" ? "page" : undefined}
              onClick={() => setActiveView("tutor")}
            >
              Tutor
            </button>
            <button
              type="button"
              aria-current={activeView === "experiment" ? "page" : undefined}
              onClick={() => setActiveView("experiment")}
            >
              Experiment
            </button>
          </nav>
          {activeView === "tutor" && (
            <div className="header-actions">
              <label>
                <span>Practice task</span>
                <span className="select-control">
                  <select
                    aria-label="Practice task"
                    value={session.task.task_id}
                    onChange={(event) => handleTaskChange(event.target.value)}
                    disabled={isSubmitting}
                  >
                    {availableTasks.map((task) => (
                      <option key={task.task_id} value={task.task_id}>
                        {task.title}
                      </option>
                    ))}
                  </select>
                </span>
              </label>
              <button className="secondary-button" type="button" onClick={handleReset}>
                Reset session
              </button>
            </div>
          )}
        </div>
      </header>

      {activeView === "experiment" ? (
        <ExperimentView />
      ) : (
        <main className="workspace-grid">
        <section className="task-pane" aria-labelledby="task-title">
          <div className="section-heading">
            <span>Python task</span>
            <span>v{session.task.version}</span>
          </div>
          <h2 id="task-title">{session.task.title}</h2>
          <p className="task-instructions">{session.task.instructions}</p>
          <pre className="code-block">
            <code>{session.task.starter_code}</code>
          </pre>

          <div className="state-summary" aria-label="Current tutor state">
            <div>
              <span className="metric-label">Demo tracker score</span>
              <strong>{masteryPercent}%</strong>
              <progress max="100" value={masteryPercent} aria-label="Demo tracker score" />
            </div>
            <div>
              <span className="metric-label">Evidence</span>
              <strong>{latestTurn ? readableLabel(latestTurn.evidence.category) : "Pending"}</strong>
              <span className="metric-detail">
                {latestTurn ? latestTurn.evidence.rationale : "No response assessed"}
              </span>
            </div>
            <div>
              <span className="metric-label">Tutor action</span>
              <strong>{latestTurn ? readableLabel(latestTurn.decision.action) : "Initial prompt"}</strong>
              <span className="metric-detail">
                {latestTurn ? latestTurn.decision.rationale : "Waiting for first response"}
              </span>
            </div>
          </div>
        </section>

        <section className="dialogue-pane" aria-labelledby="dialogue-title">
          <div className="section-heading">
            <span id="dialogue-title">Tutoring dialogue</span>
            <span>{session.turns.length} turns</span>
          </div>

          <div className="conversation" aria-live="polite" ref={conversationRef}>
            <article className="message tutor-message">
              <span>Tutor</span>
              <p>{session.initial_prompt}</p>
            </article>
            {session.turns.map((turn) => (
              <div className="turn-group" key={turn.idempotency_key}>
                <article className="message student-message">
                  <span>You</span>
                  <p>{turn.student_response || "No response"}</p>
                </article>
                <article className="message tutor-message">
                  <span>Tutor</span>
                  <p>{turn.tutor_prompt}</p>
                </article>
              </div>
            ))}
          </div>

          <form className="response-form" onSubmit={handleSubmit}>
            <label htmlFor="student-response">Your reasoning</label>
            <textarea
              id="student-response"
              value={responseText}
              rows={5}
              maxLength={10_000}
              onChange={(event) => {
                setResponseText(event.target.value);
                setFailedSubmission(null);
                setSubmissionError(null);
              }}
              disabled={isSubmitting}
            />
            {submissionError && (
              <div className="inline-error" role="alert">
                <span>{submissionError}</span>
                {failedSubmission && (
                  <button
                    className="retry-button"
                    type="button"
                    onClick={() => void sendSubmission(failedSubmission)}
                    disabled={isSubmitting}
                  >
                    Retry
                  </button>
                )}
              </div>
            )}
            <div className="form-actions">
              <span>{responseText.length.toLocaleString()} / 10,000</span>
              <button className="primary-button" type="submit" disabled={isSubmitting}>
                {isSubmitting ? "Submitting..." : "Submit response"}
              </button>
            </div>
          </form>
        </section>
        </main>
      )}
    </div>
  );
}
