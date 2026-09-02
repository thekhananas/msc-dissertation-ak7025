import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, vi } from "vitest";

import { ExperimentView } from "./ExperimentView";
import type {
  BenchmarkReplaySnapshot,
  ExperimentSummary,
  LiveEvaluationSnapshot,
  LiveEvaluationStatus,
} from "./types";

const committedReplay: BenchmarkReplaySnapshot = {
  replay_id: "a".repeat(64),
  phase: "predictions_committed",
  source: "recorded_artifact",
  source_data_hash: "b".repeat(64),
  source_manifest_hash: "c".repeat(64),
  benchmark_version: "v1",
  case_id: "h-c2m1-02",
  evaluation_model: "cerebras/gpt-oss-120b",
  public_summary: "The answer contained two conflicting conclusions.",
  evidence: [
    {
      condition: "probe_informed",
      label: "Related coding task",
      response_summary: "The model returned the expected list.",
      execution_summary: "The function passed its automatic check.",
      passed_checks: 1,
      failed_checks: 0,
    },
  ],
  predictions: [
    {
      condition: "dialogue_only",
      label: "First answer only",
      tracker_score: 0.4,
      policy_threshold: 0.6,
      predicts_success: false,
      committed_at_utc: "2026-09-01T07:47:56Z",
      record_hash: "d".repeat(64),
    },
  ],
  outcome: null,
  interpretation: null,
  predictions_sealed_before_outcome_reveal: true,
  model_calls_made: 0,
  sandbox_calls_made: 0,
  post_reveal_policy_calls_made: 0,
  claim_boundary: "This replay does not measure student learning.",
};

const revealedReplay: BenchmarkReplaySnapshot = {
  ...committedReplay,
  phase: "outcome_revealed",
  outcome: {
    summary: "The later answer passed both recorded checks.",
    test_warning: "The checks did not test whether the supplied list changed.",
    passed_checks: 2,
    failed_checks: 0,
    revealed_at_utc: "2026-09-01T09:03:39Z",
  },
  interpretation: "The example does not show that relevant evidence caused the change.",
};

const experimentSummary: ExperimentSummary = {
  schema_version: 1,
  schema_id: "demo.experiment_summary.v1",
  overarching_question: "When should a tutor request executable evidence?",
  findings: [
    {
      study_id: "m7b",
      label: "Selective probing study",
      question: "Did the proposed policy choose better probes?",
      result: "It beat random selection but not the stronger simple policies.",
      interpretation: "The planned overall advantage was unsupported.",
      sample: "2,000 episodes in each held-out environment.",
      evidence_type: "glass_box_simulation",
      status: "Canonical primary result",
      source_report_hash: "e".repeat(64),
    },
  ],
  claim_boundary: "These findings do not show improved human learning.",
};

const liveStatus: LiveEvaluationStatus = {
  enabled: true,
  available: true,
  case_id: "dev-aliasing-001",
  evaluation_model: "cerebras/gpt-oss-120b",
  reason: null,
  claim_boundary: "Illustration only. This is not canonical research evidence.",
};

const committedLive: LiveEvaluationSnapshot = {
  run_id: "f".repeat(64),
  case_id: "dev-aliasing-001",
  source: "live_demo",
  canonical: false,
  phase: "predictions_committed",
  evaluation_model: "cerebras/gpt-oss-120b",
  public_response: "Both names refer to the same list.",
  public_assessment: {
    category: "correct",
    method: "transparent_development_rule",
    rationale: "The answer contains the expected idea for this development case.",
  },
  evidence_response: "```python\ndef add_marker(items):\n    return items\n```",
  evidence_execution: {
    passed_checks: 2,
    failed_checks: 0,
    passed_all_checks: true,
    sandbox_id: "sb-live-1",
  },
  predictions: [
    {
      condition: "dialogue_only",
      label: "Visible answer only",
      tracker_score: 0.7,
      policy_threshold: 0.6,
      predicts_success: true,
      committed_at_utc: "2026-09-11T10:00:00Z",
      record_hash: "1".repeat(64),
    },
    {
      condition: "probe_informed",
      label: "Visible answer and coding check",
      tracker_score: 0.9,
      policy_threshold: 0.6,
      predicts_success: true,
      committed_at_utc: "2026-09-11T10:00:00Z",
      record_hash: "2".repeat(64),
    },
  ],
  commitment_hash: "3".repeat(64),
  committed_at_utc: "2026-09-11T10:00:00Z",
  outcome: null,
  predictions_sealed_before_outcome_reveal: true,
  model_calls_made: 2,
  sandbox_calls_made: 1,
  provider_latency_ms: 500,
  failure: null,
  claim_boundary: liveStatus.claim_boundary,
};

const revealedLive: LiveEvaluationSnapshot = {
  ...committedLive,
  phase: "outcome_revealed",
  outcome: {
    response: "```python\ndef snapshot_then_append(items, value):\n    return [], items\n```",
    execution: {
      passed_checks: 1,
      failed_checks: 0,
      passed_all_checks: true,
      sandbox_id: "sb-live-2",
    },
    revealed_at_utc: "2026-09-11T10:01:00Z",
  },
  model_calls_made: 3,
  sandbox_calls_made: 2,
  provider_latency_ms: 750,
};

function jsonResponse(payload: object, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

test("keeps the later result hidden until the recorded outcome is revealed", async () => {
  const fetchMock = vi
    .spyOn(globalThis, "fetch")
    .mockResolvedValueOnce(jsonResponse(committedReplay, 201))
    .mockResolvedValueOnce(jsonResponse(experimentSummary))
    .mockResolvedValueOnce(jsonResponse(liveStatus))
    .mockResolvedValueOnce(jsonResponse(revealedReplay));

  render(<ExperimentView />);

  expect(await screen.findByText("The answer contained two conflicting conclusions.")).toBeVisible();
  expect(screen.getByText("Related coding task")).toBeVisible();
  expect(screen.getByText("40%")).toBeVisible();
  expect(screen.getByText(/beat random selection/i)).toBeVisible();
  expect(screen.queryByText("The later answer passed both recorded checks.")).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "Reveal recorded result" }));

  expect(await screen.findByText("The later answer passed both recorded checks.")).toBeVisible();
  expect(screen.getByText(/does not show that relevant evidence caused/i)).toBeVisible();
  expect(fetchMock.mock.calls[3]?.[0]).toBe(
    `/api/benchmark-replays/${committedReplay.replay_id}/reveal`,
  );
});

test("keeps the fresh outcome hidden until the live predictions are fixed", async () => {
  const fetchMock = vi
    .spyOn(globalThis, "fetch")
    .mockResolvedValueOnce(jsonResponse(committedReplay, 201))
    .mockResolvedValueOnce(jsonResponse(experimentSummary))
    .mockResolvedValueOnce(jsonResponse(liveStatus))
    .mockResolvedValueOnce(jsonResponse(committedLive, 201))
    .mockResolvedValueOnce(jsonResponse(revealedLive));

  render(<ExperimentView />);

  fireEvent.click(await screen.findByRole("tab", { name: "Live illustration" }));
  expect(screen.getByText("Illustration only")).toBeVisible();
  expect(screen.getByRole("button", { name: "Use recorded replay" })).toBeVisible();

  fireEvent.click(screen.getByRole("button", { name: "Start live run" }));

  expect(await screen.findByText("Both names refer to the same list.")).toBeVisible();
  expect(screen.getByText("0.70")).toBeVisible();
  expect(screen.getByText("0.90")).toBeVisible();
  expect(screen.queryByText("70%")).not.toBeInTheDocument();
  expect(screen.getByText(/not validated probabilities/)).toBeVisible();
  expect(screen.getByText("2 of 3")).toBeVisible();
  expect(screen.queryByText(/snapshot_then_append/)).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "Reveal live result" }));

  expect(await screen.findByText(/snapshot_then_append/)).toBeVisible();
  expect(screen.getByText("3 of 3")).toBeVisible();
  expect(fetchMock.mock.calls[4]?.[0]).toBe(
    `/api/live-evaluations/${committedLive.run_id}/reveal`,
  );
});

test("keeps the recorded evidence available when live status cannot load", async () => {
  vi.spyOn(globalThis, "fetch")
    .mockResolvedValueOnce(jsonResponse(committedReplay, 201))
    .mockResolvedValueOnce(jsonResponse(experimentSummary))
    .mockRejectedValueOnce(new Error("Live route unavailable"));

  render(<ExperimentView />);

  expect(await screen.findByText("The answer contained two conflicting conclusions.")).toBeVisible();
  fireEvent.click(screen.getByRole("tab", { name: "Live illustration" }));
  expect(screen.getByText(/recorded replay remains available/i)).toBeVisible();
  expect(screen.getByRole("button", { name: "Use recorded replay" })).toBeVisible();
});
