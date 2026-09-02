import { useRef, useState } from "react";

import { revealLiveEvaluation, startLiveEvaluation } from "./api";
import { newIdempotencyKey } from "./ids";
import type { LiveEvaluationSnapshot, LiveEvaluationStatus } from "./types";

type LiveEvaluationPanelProps = {
  status: LiveEvaluationStatus;
  onUseRecorded: () => void;
};

function scoreLabel(score: number): string {
  return score.toFixed(2);
}

function categoryLabel(category: string): string {
  return category.charAt(0).toUpperCase() + category.slice(1);
}

export function LiveEvaluationPanel({ status, onUseRecorded }: LiveEvaluationPanelProps) {
  const idempotencyKey = useRef(newIdempotencyKey());
  const [snapshot, setSnapshot] = useState<LiveEvaluationSnapshot | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [isRevealing, setIsRevealing] = useState(false);
  const [requestError, setRequestError] = useState<string | null>(null);

  async function start(): Promise<void> {
    setIsStarting(true);
    setRequestError(null);
    try {
      setSnapshot(
        await startLiveEvaluation(idempotencyKey.current, status.case_id),
      );
    } catch (error: unknown) {
      setRequestError(error instanceof Error ? error.message : "The live run could not start.");
    } finally {
      setIsStarting(false);
    }
  }

  async function reveal(): Promise<void> {
    if (snapshot === null) {
      return;
    }
    setIsRevealing(true);
    setRequestError(null);
    try {
      setSnapshot(await revealLiveEvaluation(snapshot.run_id));
    } catch (error: unknown) {
      setRequestError(error instanceof Error ? error.message : "The later result is unavailable.");
    } finally {
      setIsRevealing(false);
    }
  }

  return (
    <section className="replay-panel" aria-labelledby="live-evaluation-title">
      <div className="experiment-section-heading">
        <div>
          <span>Fresh development run</span>
          <h3 id="live-evaluation-title">Generate, commit, then reveal</h3>
        </div>
        <span className="record-status live-record-status">Illustration only</span>
      </div>

      {!status.available ? (
        <div className="live-unavailable">
          <h4>Live illustration unavailable</h4>
          <p>{status.reason}</p>
          <button className="secondary-button" type="button" onClick={onUseRecorded}>
            Use recorded replay
          </button>
        </div>
      ) : snapshot === null ? (
        <div className="live-start">
          <p>
            This creates fresh responses for {status.case_id}, runs generated Python in Modal,
            and fixes two predictions before requesting the later task.
          </p>
          <button
            className="primary-button"
            type="button"
            onClick={() => void start()}
            disabled={isStarting}
          >
            {isStarting ? "Running live checks..." : "Start live run"}
          </button>
          <button className="secondary-button" type="button" onClick={onUseRecorded}>
            Use recorded replay
          </button>
        </div>
      ) : (
        <>
          {snapshot.failure !== null && (
            <div className="live-failure" role="alert">
              <strong>Live run stopped</strong>
              <p>{snapshot.failure.message}</p>
              <button className="secondary-button" type="button" onClick={onUseRecorded}>
                Use recorded replay
              </button>
            </div>
          )}

          {snapshot.public_response !== null && (
            <section className="experiment-step" aria-labelledby="live-answer-title">
              <span className="step-number">01</span>
              <div>
                <h4 id="live-answer-title">The model's first answer</h4>
                <pre className="live-response">{snapshot.public_response}</pre>
                {snapshot.public_assessment !== null && (
                  <p className="live-assessment">
                    <strong>Phrase-matching assessment: {categoryLabel(snapshot.public_assessment.category)}.</strong>{" "}
                    {snapshot.public_assessment.rationale}
                  </p>
                )}
              </div>
            </section>
          )}

          {snapshot.evidence_response !== null && (
            <section className="experiment-step" aria-labelledby="live-evidence-title">
              <span className="step-number">02</span>
              <div>
                <h4 id="live-evidence-title">The model's answer to a separate coding check</h4>
                <pre className="live-response">{snapshot.evidence_response}</pre>
                {snapshot.evidence_execution !== null && (
                  <p className="live-check-result">
                    {snapshot.evidence_execution.passed_checks} checks passed,{" "}
                    {snapshot.evidence_execution.failed_checks} failed in Modal.
                  </p>
                )}
                <p>
                  These counts describe the authored tests. A passing result does not establish
                  that the tests cover every requirement; test reliability is not estimated here.
                </p>
              </div>
            </section>
          )}

          {snapshot.predictions.length > 0 && (
            <section className="experiment-step" aria-labelledby="live-predictions-title">
              <span className="step-number">03</span>
              <div>
                <h4 id="live-predictions-title">Predictions fixed before the later task</h4>
                <p>
                  These are simple rule-based scores on a 0–1 scale, not validated probabilities
                  of success. This live run uses the demo tracker, not the Bayesian research method.
                </p>
                <div className="prediction-table-wrap">
                  <table className="prediction-table">
                    <thead>
                      <tr>
                        <th scope="col">Information used</th>
                        <th scope="col">Score</th>
                        <th scope="col">Pass threshold</th>
                        <th scope="col">Prediction</th>
                      </tr>
                    </thead>
                    <tbody>
                      {snapshot.predictions.map((prediction) => (
                        <tr key={prediction.condition}>
                          <td>{prediction.label}</td>
                          <td>{scoreLabel(prediction.tracker_score)}</td>
                          <td>{scoreLabel(prediction.policy_threshold)}</td>
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
          )}

          {snapshot.phase !== "failed" && (
            <section className="experiment-step outcome-step" aria-labelledby="live-outcome-title">
              <span className="step-number">04</span>
              <div>
                <h4 id="live-outcome-title">Later task result</h4>
                {snapshot.outcome === null ? (
                  <div className="hidden-outcome">
                    <p>The predictions above are fixed. The later task has not been requested.</p>
                    <button
                      className="primary-button"
                      type="button"
                      onClick={() => void reveal()}
                      disabled={isRevealing}
                    >
                      {isRevealing ? "Requesting result..." : "Reveal live result"}
                    </button>
                  </div>
                ) : (
                  <div className="revealed-outcome" aria-live="polite">
                    <div className="outcome-result">
                      <strong>
                        {snapshot.outcome.execution.passed_checks} checks passed,{" "}
                        {snapshot.outcome.execution.failed_checks} failed
                      </strong>
                      <span>Fresh outcome</span>
                    </div>
                    <pre className="live-response">{snapshot.outcome.response}</pre>
                  </div>
                )}
              </div>
            </section>
          )}

          <dl className="live-activity">
            <div>
              <dt>Model requests</dt>
              <dd>{snapshot.model_calls_made} of 3</dd>
            </div>
            <div>
              <dt>Remote code runs</dt>
              <dd>{snapshot.sandbox_calls_made} of 2</dd>
            </div>
            <div>
              <dt>Provider time</dt>
              <dd>{snapshot.provider_latency_ms} ms</dd>
            </div>
          </dl>
        </>
      )}

      {requestError !== null && (
        <p className="inline-reveal-error" role="alert">
          {requestError}
        </p>
      )}
      <p className="claim-boundary">{status.claim_boundary}</p>
    </section>
  );
}
