import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, vi } from "vitest";

import { App } from "./App";
import type { SessionSnapshot } from "./types";

const tasks = [
  {
    task_id: "mutable-list-aliasing",
    version: "1.0.0",
    title: "Shared list references",
    concept: "mutable-list-aliasing",
    instructions: "Predict exactly what the program prints, then explain why.",
    starter_code: "numbers = [1, 2]\nalias = numbers\nalias.append(3)\nprint(numbers)",
  },
  {
    task_id: "none-versus-falsy",
    version: "1.0.0",
    title: "Missing values and valid zeroes",
    concept: "none-versus-falsy",
    instructions: "Predict both printed lines, then explain the condition.",
    starter_code: "if score is None:\n    return 'missing'",
  },
];

const baseSession: SessionSnapshot = {
  session_id: "session-1",
  task: {
    task_id: "mutable-list-aliasing",
    version: "1.0.0",
    title: "Shared list references",
    concept: "mutable-list-aliasing",
    instructions: "Predict exactly what the program prints, then explain why.",
    starter_code: "numbers = [1, 2]\nalias = numbers\nalias.append(3)\nprint(numbers)",
  },
  initial_prompt: "What do you think it prints?",
  tracker: {
    concept: "mutable-list-aliasing",
    mastery_probability: 0.5,
    observations: 0,
    last_evidence: null,
  },
  turns: [],
  created_at: "2026-07-14T12:00:00Z",
  updated_at: "2026-07-14T12:00:00Z",
};

const noneSession: SessionSnapshot = {
  ...baseSession,
  session_id: "session-2",
  task: tasks[1],
  initial_prompt: "What are the two printed lines?",
  tracker: {
    concept: "none-versus-falsy",
    mastery_probability: 0.5,
    observations: 0,
    last_evidence: null,
  },
};

const completedSession: SessionSnapshot = {
  ...baseSession,
  tracker: {
    concept: "mutable-list-aliasing",
    mastery_probability: 0.7,
    observations: 1,
    last_evidence: "correct",
  },
  turns: [
    {
      turn_number: 1,
      idempotency_key: "turn-key",
      student_response: "It prints [1, 2, 3] because both names reference the same list.",
      evidence: {
        category: "correct",
        confidence: 0.9,
        rationale: "The predicted output and explanation both match the reviewed rules.",
        observed_signals: ["[1, 2, 3]", "same list", "reference"],
      },
      tracker_before: baseSession.tracker,
      tracker_after: {
        concept: "mutable-list-aliasing",
        mastery_probability: 0.7,
        observations: 1,
        last_evidence: "correct",
      },
      decision: {
        action: "transfer",
        rationale: "Correct evidence supports moving to a nearby transfer question.",
      },
      tutor_prompt: "What would happen if the change were made through the other name?",
      guardrail: {
        safe: true,
        output_prompt: "What would happen if the change were made through the other name?",
        violations: [],
      },
      completed_at: "2026-07-14T12:01:00Z",
    },
  ],
  updated_at: "2026-07-14T12:01:00Z",
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

test("loads the task and initial tutoring prompt", async () => {
  vi.spyOn(globalThis, "fetch")
    .mockResolvedValueOnce(jsonResponse(tasks))
    .mockResolvedValueOnce(jsonResponse(baseSession, 201));

  render(<App />);

  expect(screen.getByText("Starting tutoring session...")).toBeInTheDocument();
  expect(await screen.findByRole("heading", { name: "Shared list references" })).toBeInTheDocument();
  expect(screen.getByText("What do you think it prints?")).toBeInTheDocument();
  expect(screen.getByText("50%")).toBeInTheDocument();
  expect(screen.getByLabelText("Practice task")).toHaveValue("mutable-list-aliasing");
});

test("submits a response and displays tracker and policy results", async () => {
  vi.spyOn(globalThis, "fetch")
    .mockResolvedValueOnce(jsonResponse(tasks))
    .mockResolvedValueOnce(jsonResponse(baseSession, 201))
    .mockResolvedValueOnce(jsonResponse(completedSession));
  render(<App />);
  const response = await screen.findByLabelText("Your reasoning");

  fireEvent.change(response, {
    target: { value: "It prints [1, 2, 3] because both names reference the same list." },
  });
  fireEvent.click(screen.getByRole("button", { name: "Submit response" }));

  expect(await screen.findByText("70%")).toBeInTheDocument();
  expect(screen.getByText("Correct")).toBeInTheDocument();
  expect(screen.getByText("Transfer")).toBeInTheDocument();
  expect(screen.getByText(/change were made through the other name/i)).toBeInTheDocument();
});

test("retries a failed submission with the same idempotency key", async () => {
  const fetchMock = vi
    .spyOn(globalThis, "fetch")
    .mockResolvedValueOnce(jsonResponse(tasks))
    .mockResolvedValueOnce(jsonResponse(baseSession, 201))
    .mockRejectedValueOnce(new Error("Connection interrupted"))
    .mockResolvedValueOnce(jsonResponse(completedSession));
  render(<App />);
  const response = await screen.findByLabelText("Your reasoning");
  fireEvent.change(response, { target: { value: "my response" } });
  fireEvent.click(screen.getByRole("button", { name: "Submit response" }));

  expect(await screen.findByRole("alert")).toHaveTextContent("Connection interrupted");
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  await waitFor(() => expect(screen.getByText("70%")).toBeInTheDocument());

  const firstBody = JSON.parse(String(fetchMock.mock.calls[2]?.[1]?.body)) as {
    idempotency_key: string;
  };
  const retryBody = JSON.parse(String(fetchMock.mock.calls[3]?.[1]?.body)) as {
    idempotency_key: string;
  };
  expect(retryBody.idempotency_key).toBe(firstBody.idempotency_key);
});

test("shows a recoverable error when session creation fails", async () => {
  vi.spyOn(globalThis, "fetch")
    .mockResolvedValueOnce(jsonResponse(tasks))
    .mockRejectedValueOnce(new Error("Backend unavailable"));

  render(<App />);

  expect(await screen.findByRole("alert")).toHaveTextContent("Backend unavailable");
  expect(screen.getByRole("button", { name: "Retry session" })).toBeInTheDocument();
});

test("switches to a different authored task", async () => {
  const fetchMock = vi
    .spyOn(globalThis, "fetch")
    .mockResolvedValueOnce(jsonResponse(tasks))
    .mockResolvedValueOnce(jsonResponse(baseSession, 201))
    .mockResolvedValueOnce(jsonResponse(noneSession, 201));
  render(<App />);

  const selector = await screen.findByLabelText("Practice task");
  fireEvent.change(selector, { target: { value: "none-versus-falsy" } });

  expect(
    await screen.findByRole("heading", { name: "Missing values and valid zeroes" }),
  ).toBeInTheDocument();
  const requestBody = JSON.parse(String(fetchMock.mock.calls[2]?.[1]?.body)) as {
    task_id: string;
  };
  expect(requestBody.task_id).toBe("none-versus-falsy");
});
