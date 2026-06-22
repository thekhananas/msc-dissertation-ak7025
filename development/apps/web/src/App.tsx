import { useEffect, useState } from "react";

import { getHealth } from "./api";
import type { HealthResponse } from "./types";

type HealthState =
  | { kind: "loading" }
  | { kind: "ready"; health: HealthResponse }
  | { kind: "error" };

export function App() {
  const [healthState, setHealthState] = useState<HealthState>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    void getHealth(controller.signal)
      .then((health) => setHealthState({ kind: "ready", health }))
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
        setHealthState({ kind: "error" });
      });
    return () => controller.abort();
  }, []);

  return (
    <main className="workspace-shell">
      <header className="workspace-header">
        <p className="eyebrow">Research proof of concept</p>
        <h1>Socratic Tutor</h1>
      </header>

      <section className="status-panel" aria-labelledby="workspace-status">
        <div>
          <h2 id="workspace-status">Workspace status</h2>
          <p>The tutoring vertical slice will be built here after the foundation gate passes.</p>
        </div>

        {healthState.kind === "loading" && <p role="status">Checking backend...</p>}
        {healthState.kind === "error" && (
          <p className="status-error" role="alert">
            Backend unavailable
          </p>
        )}
        {healthState.kind === "ready" && (
          <dl className="health-grid">
            <div>
              <dt>Status</dt>
              <dd className="status-ok">{healthState.health.status}</dd>
            </div>
            <div>
              <dt>Service</dt>
              <dd>{healthState.health.service}</dd>
            </div>
            <div>
              <dt>Version</dt>
              <dd>{healthState.health.version}</dd>
            </div>
          </dl>
        )}
      </section>
    </main>
  );
}
