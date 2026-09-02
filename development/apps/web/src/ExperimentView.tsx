import { useEffect, useRef, useState } from "react";

import {
  getExperimentSummary,
  getLiveEvaluationStatus,
  revealBenchmarkOutcome,
  startBenchmarkReplay,
} from "./api";
import { newIdempotencyKey } from "./ids";
import { LiveEvaluationPanel } from "./LiveEvaluationPanel";
import type {
  BenchmarkReplaySnapshot,
  ExperimentSummary,
  LiveEvaluationStatus,
} from "./types";

type ExperimentMode = "recorded" | "live";

const unavailableLiveStatus: LiveEvaluationStatus = {
  enabled: false,
  available: false,
  case_id: "dev-aliasing-001",
  evaluation_model: "cerebras/gpt-oss-120b",
  reason: "The live status could not be loaded. The recorded replay remains available.",
  claim_boundary: "Illustration only. Live runs are not canonical research evidence.",
};

type ExperimentState =
  | { kind: "loading" }
  | {
      kind: "ready";
      replay: BenchmarkReplaySnapshot;
      summary: ExperimentSummary;
      liveStatus: LiveEvaluationStatus;
    }
  | { kind: "error"; message: string };

function scoreLabel(score: number): string {
  return `${Math.round(score * 100)}%`;
}

function evidenceTypeLabel(value: string): string {
  const labels: Record<string, string> = {
    external_model_benchmark: "Recorded model responses",
    glass_box_simulation: "Controlled simulation",
    external_learner_records: "Historical learner records",
  };
  return labels[value] ?? value;
}

export function ExperimentView() {
  const replayKey = useRef(newIdempotencyKey());
  const [state, setState] = useState<ExperimentState>({ kind: "loading" });
  const [mode, setMode] = useState<ExperimentMode>("recorded");
  const [isRevealing, setIsRevealing] = useState(false);
  const [revealError, setRevealError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void Promise.all([
      startBenchmarkReplay(replayKey.current, controller.signal),
      getExperimentSummary(controller.signal),
      getLiveEvaluationStatus(controller.signal).catch(() => unavailableLiveStatus),
    ])
      .then(([replay, summary, liveStatus]) =>
        setState({ kind: "ready", replay, summary, liveStatus }),
      )
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
        setState({
          kind: "error",
          message: error instanceof Error ? error.message : "Unable to load experiment evidence",
        });
      });
    return () => controller.abort();
  }, []);

  async function revealOutcome(): Promise<void> {
    if (state.kind !== "ready") {
      return;
    }
    setIsRevealing(true);
    setRevealError(null);
    try {
      const replay = await revealBenchmarkOutcome(state.replay.replay_id);
      setState({
        kind: "ready",
        replay,
        summary: state.summary,
        liveStatus: state.liveStatus,
      });
    } catch (error: unknown) {
      setRevealError(error instanceof Error ? error.message : "Unable to reveal the result");
    } finally {
      setIsRevealing(false);
    }
  }

  if (state.kind === "loading") {
    return (
      <main className="experiment-state">
        <p role="status">Loading recorded experiment...</p>
      </main>
    );
  }

  if (state.kind === "error") {
    return (
      <main className="experiment-state">
        <p className="error-message" role="alert">
          {state.message}
        </p>
      </main>
    );
  }

  const { replay, summary, liveStatus } = state;
  const showingLive = mode === "live";

  return (
    <main className="experiment-page">
      <section className="experiment-intro" aria-labelledby="experiment-title">
        <div>
          <p className="eyebrow">
            {showingLive ? "Optional live illustration" : "Recorded research evidence"}
          </p>
          <h2 id="experiment-title">When is another coding check worth requesting?</h2>
          <p>{summary.overarching_question}</p>
        </div>
        <dl className="experiment-meta">
          <div>
            <dt>Case</dt>
            <dd>{showingLive ? liveStatus.case_id : replay.case_id}</dd>
          </div>
          <div>
            <dt>Evaluation model</dt>
            <dd>{showingLive ? liveStatus.evaluation_model : replay.evaluation_model}</dd>
          </div>
          <div>
            <dt>{showingLive ? "Live activity" : "Replay activity"}</dt>
            <dd>
              {showingLive
                ? liveStatus.available
                  ? "Up to 3 model requests, 2 remote code runs"
                  : "Disabled"
                : "0 model calls, 0 code runs"}
            </dd>
          </div>
        </dl>
      </section>

      <div className="experiment-mode-tabs" role="tablist" aria-label="Experiment source">
        <button
          type="button"
          role="tab"
          aria-selected={!showingLive}
          aria-controls="recorded-replay-panel"
          onClick={() => setMode("recorded")}
        >
          Recorded replay
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={showingLive}
          aria-controls="live-evaluation-panel"
          onClick={() => setMode("live")}
        >
          Live illustration
        </button>
      </div>

      <div className="experiment-layout">
        <section
          id="recorded-replay-panel"
          className="replay-panel"
          aria-labelledby="replay-title"
          hidden={showingLive}
        >
          <div className="experiment-section-heading">
            <div>
              <span>Worked example</span>
              <h3 id="replay-title">A prediction fixed before the later result</h3>
            </div>
            <span className="record-status">Offline replay</span>
          </div>

          <section className="experiment-step" aria-labelledby="observed-answer-title">
            <span className="step-number">01</span>
            <div>
              <h4 id="observed-answer-title">Observed answer</h4>
              <p>{replay.public_summary}</p>
            </div>
          </section>

          <section className="experiment-step" aria-labelledby="evidence-title">
            <span className="step-number">02</span>
            <div>
              <h4 id="evidence-title">Recorded coding checks</h4>
              <div className="evidence-grid">
                {replay.evidence.map((evidence) => (
                  <article className="evidence-item" key={evidence.condition}>
                    <div className="evidence-title-row">
                      <h5>{evidence.label}</h5>
                      <span>
                        {evidence.passed_checks} passed, {evidence.failed_checks} failed
                      </span>
                    </div>
                    <p>{evidence.response_summary}</p>
                    <p className="muted-copy">{evidence.execution_summary}</p>
                  </article>
                ))}
              </div>
            </div>
          </section>

          <section className="experiment-step" aria-labelledby="predictions-title">
            <span className="step-number">03</span>
            <div>
              <h4 id="predictions-title">Predictions fixed before the later task</h4>
              <div className="prediction-table-wrap">
                <table className="prediction-table">
                  <thead>
                    <tr>
                      <th scope="col">Information used</th>
                      <th scope="col">Score</th>
                      <th scope="col">Prediction</th>
                    </tr>
                  </thead>
                  <tbody>
                    {replay.predictions.map((prediction) => (
                      <tr key={prediction.condition}>
                        <td>{prediction.label}</td>
                        <td>{scoreLabel(prediction.tracker_score)}</td>
                        <td>
                          <span
                            className={
                              prediction.predicts_success
                                ? "prediction-result predicts-pass"
                                : "prediction-result predicts-fail"
                            }
                          >
                            {prediction.predicts_success ? "Pass" : "Fail"}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </section>

          <section className="experiment-step outcome-step" aria-labelledby="outcome-title">
            <span className="step-number">04</span>
            <div>
              <h4 id="outcome-title">Later task result</h4>
              {replay.outcome === null ? (
                <div className="hidden-outcome">
                  <p>The recorded result remains hidden. The four predictions above are already fixed.</p>
                  <button
                    className="primary-button"
                    type="button"
                    onClick={() => void revealOutcome()}
                    disabled={isRevealing}
                  >
                    {isRevealing ? "Revealing..." : "Reveal recorded result"}
                  </button>
                  {revealError && (
                    <p className="inline-reveal-error" role="alert">
                      {revealError}
                    </p>
                  )}
                </div>
              ) : (
                <div className="revealed-outcome" aria-live="polite">
                  <div className="outcome-result">
                    <strong>
                      {replay.outcome.passed_checks} checks passed, {replay.outcome.failed_checks}{" "}
                      failed
                    </strong>
                    <span>Recorded outcome</span>
                  </div>
                  <p>{replay.outcome.summary}</p>
                  <p className="test-warning">
                    <strong>Important test limit:</strong> {replay.outcome.test_warning}
                  </p>
                  <p className="interpretation-copy">{replay.interpretation}</p>
                </div>
              )}
            </div>
          </section>

          <p className="claim-boundary">{replay.claim_boundary}</p>
        </section>

        <div id="live-evaluation-panel" hidden={!showingLive}>
          <LiveEvaluationPanel
            status={liveStatus}
            onUseRecorded={() => setMode("recorded")}
          />
        </div>

        <aside className="findings-panel" aria-labelledby="findings-title">
          <div className="experiment-section-heading">
            <div>
              <span>Results so far</span>
              <h3 id="findings-title">What the completed studies found</h3>
            </div>
          </div>
          <div className="finding-list">
            {summary.findings.map((finding) => (
              <article className="finding-item" key={finding.study_id}>
                <div className="finding-label-row">
                  <span>{finding.label}</span>
                  <span>{evidenceTypeLabel(finding.evidence_type)}</span>
                </div>
                <h4>{finding.question}</h4>
                <p className="finding-result">{finding.result}</p>
                <p>{finding.interpretation}</p>
                <footer>
                  <span>{finding.sample}</span>
                  <span>{finding.status}</span>
                </footer>
              </article>
            ))}
          </div>
          <p className="claim-boundary summary-boundary">{summary.claim_boundary}</p>
        </aside>
      </div>
    </main>
  );
}
