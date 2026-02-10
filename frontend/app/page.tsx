"use client";

import { useMemo, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export default function Page() {
  const [simulationId, setSimulationId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  const wsUrl = useMemo(() => {
    if (!simulationId) return "";
    return `${API_BASE.replace("http", "ws")}/ws/simulations/${simulationId}`;
  }, [simulationId]);

  async function createAndStart() {
    setLoading(true);
    try {
      const create = await fetch(`${API_BASE}/api/v1/simulations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          goal: "Bootstrap simulation",
          max_steps: 8,
          tick_interval_ms: 800,
          agents: [
            { name: "Planner", role: "planner" },
            { name: "Researcher", role: "researcher" },
            { name: "Critic", role: "critic" }
          ]
        })
      });
      const createJson = await create.json();
      const id = createJson.simulation.id as number;
      setSimulationId(id);
      await fetch(`${API_BASE}/api/v1/simulations/${id}/start`, { method: "POST" });
    } finally {
      setLoading(false);
    }
  }

  async function stop() {
    if (!simulationId) return;
    await fetch(`${API_BASE}/api/v1/simulations/${simulationId}/stop`, { method: "POST" });
  }

  return (
    <main style={{ padding: 24, fontFamily: "sans-serif" }}>
      <h1>AI Agent Community Simulator</h1>
      <p>Project skeleton based on deep-research-report.md</p>
      <div style={{ display: "flex", gap: 12 }}>
        <button onClick={createAndStart} disabled={loading}>
          {loading ? "Starting..." : "Create & Start"}
        </button>
        <button onClick={stop} disabled={!simulationId}>Stop</button>
      </div>
      <p>Simulation ID: {simulationId ?? "-"}</p>
      <p>WebSocket: {wsUrl || "-"}</p>
    </main>
  );
}
