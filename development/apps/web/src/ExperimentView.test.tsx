import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, vi } from "vitest";

import { ExperimentView } from "./ExperimentView";
import type { BenchmarkReplaySnapshot, ExperimentSummary } from "./types";

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
  expect(fetchMock.mock.calls[2]?.[0]).toBe(
    `/api/benchmark-replays/${committedReplay.replay_id}/reveal`,
  );
});
