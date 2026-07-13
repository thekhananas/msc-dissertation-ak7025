import { render, screen } from "@testing-library/react";
import { afterEach, vi } from "vitest";

import { App } from "./App";

afterEach(() => {
  vi.restoreAllMocks();
});

test("shows backend health", async () => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(
    new Response(
      JSON.stringify({
        status: "ok",
        service: "socratic-tutor-api",
        version: "0.1.0",
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ),
  );

  render(<App />);

  expect(screen.getByText("Checking backend...")).toBeInTheDocument();
  expect(await screen.findByText("socratic-tutor-api")).toBeInTheDocument();
  expect(screen.getByText("0.1.0")).toBeInTheDocument();
});

test("shows an unavailable state", async () => {
  vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("offline"));

  render(<App />);

  expect(await screen.findByRole("alert")).toHaveTextContent("Backend unavailable");
});
