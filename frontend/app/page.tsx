"use client";

/*
  Main resource-community simulation UI.
  Responsibilities:
  - Create/start/pause/stop scenario runs
  - Poll events/report endpoints
  - Parse structured round markers from agent messages
  - Visualize realism signals, negotiation process, and final recommendation
*/

import { useCallback, useEffect, useMemo, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
const RESOURCE_SCENARIO_ID = "three-agent-resource-allocation";

type SimulationStatus = "idle" | "created" | "running" | "paused" | "stopped" | "completed";
type MechanismOption = "equal" | "contribution" | "dictator" | "majority_vote";
type TransparencyOption = "full" | "hidden";

type SimulationEvent = {
  id: number;
  step: number;
  event_type: string;
  source_agent?: string | null;
  target_agent?: string | null;
  content: string;
};

type SimulationReportData = {
  id: number;
  simulation_id: number;
  version: number;
  generator: string;
  title: string;
  markdown: string;
  report_json: Record<string, unknown>;
  created_at: string;
};

type AgentCritique = {
  consistency_score?: number;
  confidence?: number;
  loss_aversion_signal?: number;
  inertia_signal?: number;
  short_term_bias?: number;
  alignment_check?: string;
  revised_after_check?: boolean;
};

type AgentPeerReference = {
  agent?: string;
  quoted_message_snippet?: string;
  referenced_effort?: number;
  referenced_willingness_units?: number;
};

type AgentProposal = {
  effort?: number;
  effective_effort?: number;
  willingness_units?: number;
  allocation_proposal_units?: Record<string, number>;
  message?: string;
  influenced_by?: string;
  adjustment_reason?: string;
  previous_effort?: number;
  effort_delta?: number;
  previous_willingness_units?: number;
  willingness_delta?: number;
  mood?: number;
  stress?: number;
  reputation?: number;
  effort_cap?: number;
  thinking_style?: string;
  self_critique?: AgentCritique;
  peer_reference?: AgentPeerReference;
};

type ResourceScenarioRoundReport = {
  round: number;
  target_resource: number;
  selected_mechanism: string;
  transparency: "full" | "hidden";
  punishment: boolean;
  efforts: Record<string, number>;
  effective_efforts?: Record<string, number>;
  weighted_contributions: Record<string, number>;
  commitment_units: Record<string, number>;
  allocation_plan_units: Record<string, number>;
  net_payoff_units?: Record<string, number>;
  total_output: number;
  cumulative_output?: number;
  average_satisfaction: number;
  conflict_count: number;
  conflict: boolean;
  free_rider_agents: string[];
  free_rider_present: boolean;
  cooperation_alive: boolean;
  trust_mean?: number;
  low_credibility_turn?: boolean;
  target_gap: number;
  round_conclusion: string;
  causal_consistency_score?: number;
  public_stability_index?: number;
  social_welfare_index?: number;
  institution_cost_units?: number;
  moods?: Record<string, number>;
  stresses?: Record<string, number>;
  effort_cap_ratio?: Record<string, number>;
  agent_proposals?: Record<string, AgentProposal>;
};

type ResourceDiscussionItem = {
  id: number;
  step: number;
  round: number;
  source_agent: string;
  target_agent: string | null;
  text: string;
};

type ResourceRoundView = {
  round: number;
  report: ResourceScenarioRoundReport | null;
  discussions: ResourceDiscussionItem[];
};

const AGENT_META: Record<string, { role: string; behavior: string }> = {
  A: {
    role: "High-Efficiency",
    behavior: "High productivity, sensitive to inefficient allocation.",
  },
  B: {
    role: "Fairness-Sensitive",
    behavior: "Focuses on procedural fairness and collective stability.",
  },
  C: {
    role: "Self-Interested",
    behavior: "Maximizes own payoff, negotiates aggressively.",
  },
};

const MECHANISM_GUIDE: Record<MechanismOption, string> = {
  equal: "Fixed 33.3/33.3/33.3 split. Individual contribution does not directly change allocation this round.",
  contribution:
    "Allocation follows weighted contribution share. Higher effective contribution typically receives larger payoff.",
  dictator: "Round leader (A/B/C rotating by round) applies their own allocation proposal as final plan.",
  majority_vote: "Agents vote between equal vs contribution; 2 of 3 votes decide the active mechanism.",
};

const TRANSPARENCY_GUIDE: Record<TransparencyOption, string> = {
  full: "Agents observe peer signals with less distortion, making trust updates more grounded.",
  hidden: "Peer contribution is partially opaque and noisy, usually increasing suspicion and trust decay risk.",
};

const PUNISHMENT_GUIDE = {
  enabled:
    "Low-satisfaction states can escalate into strike-like effort caps and additional institution cost for control.",
  disabled:
    "Direct punitive channel is off; system still has fatigue/stress effects but weaker enforcement pressure.",
};

const PRESSURE_GUIDE = {
  enabled:
    "Strongly amplifies conflict and revision dynamics. Use only for stress experiments, not for neutral baseline runs.",
  disabled: "No extra stress amplification. Dynamics are driven by mechanism/transparency/punishment only.",
};

const CORE_RULES = [
  "Target task is fixed at 100 resource units per run.",
  "Agent multipliers are fixed: A x1.5, B x1.0, C x1.0.",
  "Each round each agent proposes effort (0-10), willingness (0-100), and allocation (A/B/C summing to 100).",
  "Effective contribution is adjusted by mood, stress, trust, and environment signal.",
  "Total output is gross contribution minus governance cost and accumulates by round.",
];

const REASONABLENESS_CHECKLIST = [
  "Effort in [0, 10], willingness in [0, 100], allocation close to 100 total.",
  "cumulative_output should be monotonic and match per-round total_output sum.",
  "Under hidden transparency, trust should usually decay faster than full transparency.",
  "Pressure mode should visibly increase conflict and revision intensity.",
];

function asRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

function readNumber(value: unknown, fallback = 0): number {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

function readString(value: unknown, fallback = "-"): string {
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed.length) return trimmed;
  }
  return fallback;
}

function clampPercent(value: number): number {
  return Math.max(0, Math.min(100, value));
}

function signed(value: number, digits = 2): string {
  const prefix = value >= 0 ? "+" : "";
  return `${prefix}${value.toFixed(digits)}`;
}

function stripEventTimestamp(content: string): string {
  const marker = " @ ";
  const idx = content.lastIndexOf(marker);
  if (idx < 0) return content.trim();
  return content.slice(0, idx).trim();
}

function parseResourceRoundFromText(content: string): number | null {
  const matched = content.match(/\[Resource Round (\d+)\]/i);
  if (!matched) return null;
  const value = Number(matched[1]);
  return Number.isFinite(value) && value > 0 ? value : null;
}

function parseResourceDiscussionText(content: string): string {
  const stripped = stripEventTimestamp(content);
  const lines = stripped
    .split("\n")
    .map((line) => line.trim())
    .filter(
      (line) =>
        line.length > 0 &&
        !line.startsWith("RESOURCE_ROUND_REPORT ") &&
        !line.startsWith("RESOURCE_FINAL_RECOMMENDATION ") &&
        !line.startsWith("RESOURCE_SELF_CRITIQUE ")
    )
    .map((line) => line.replace(/\[Resource Round \d+\]\s*/gi, ""));
  return lines.join(" ").trim();
}

function parseResourceRoundReports(events: SimulationEvent[]): Record<number, ResourceScenarioRoundReport> {
  const result: Record<number, ResourceScenarioRoundReport> = {};
  for (const event of events) {
    const stripped = stripEventTimestamp(event.content);
    for (const raw of stripped.split("\n")) {
      const line = raw.trim();
      if (!line.startsWith("RESOURCE_ROUND_REPORT ")) continue;
      const encoded = line.slice("RESOURCE_ROUND_REPORT ".length).trim();
      try {
        const parsed = JSON.parse(encoded) as ResourceScenarioRoundReport;
        const round = Number(parsed.round);
        if (Number.isFinite(round) && round >= 1) {
          result[round] = parsed;
        }
      } catch {
        // Ignore malformed structured lines.
      }
    }
  }
  return result;
}

function parseResourceFinalRecommendation(events: SimulationEvent[]): Record<string, unknown> | null {
  let latest: Record<string, unknown> | null = null;
  for (const event of events) {
    const stripped = stripEventTimestamp(event.content);
    for (const raw of stripped.split("\n")) {
      const line = raw.trim();
      if (!line.startsWith("RESOURCE_FINAL_RECOMMENDATION ")) continue;
      const encoded = line.slice("RESOURCE_FINAL_RECOMMENDATION ".length).trim();
      try {
        const parsed = JSON.parse(encoded);
        const record = asRecord(parsed);
        if (record) latest = record;
      } catch {
        // Ignore malformed recommendation lines.
      }
    }
  }
  return latest;
}

function formatTriplet(values: Record<string, number> | undefined, digits = 2): string {
  if (!values) return "-";
  const a = readNumber(values.A, 0);
  const b = readNumber(values.B, 0);
  const c = readNumber(values.C, 0);
  return `A ${a.toFixed(digits)} | B ${b.toFixed(digits)} | C ${c.toFixed(digits)}`;
}

function toNumberInput(value: string, fallback: number, min: number, max: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, Math.floor(parsed)));
}

function parseOptionalPositiveInt(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) return null;
  const normalized = Math.floor(parsed);
  if (normalized < 0) return null;
  return normalized;
}

function getGradeText(grade: string): string {
  if (grade === "high_realism") return "High Realism";
  if (grade === "moderate_realism") return "Moderate Realism";
  if (grade === "low_realism") return "Low Realism";
  return "Unknown";
}

function getToneByGrade(grade: string): string {
  if (grade === "high_realism") return "bar-fill";
  if (grade === "moderate_realism") return "bar-fill warn";
  return "bar-fill low";
}

function getMoodLabel(mood: number, stress: number): string {
  if (stress >= 0.72) return "High tension";
  if (mood >= 0.35 && stress <= 0.4) return "Constructive";
  if (mood <= -0.3) return "Defensive";
  return "Guarded";
}

export default function Page() {
  const [simulationId, setSimulationId] = useState<number | null>(null);
  const [simulationStatus, setSimulationStatus] = useState<SimulationStatus>("idle");
  const [step, setStep] = useState(0);
  const [wsStatus, setWsStatus] = useState<"idle" | "connected" | "disconnected">("idle");
  const [events, setEvents] = useState<SimulationEvent[]>([]);
  const [finalReport, setFinalReport] = useState<SimulationReportData | null>(null);

  const [maxSteps, setMaxSteps] = useState("8");
  const [tickIntervalMs, setTickIntervalMs] = useState("500");
  const [mechanism, setMechanism] = useState<MechanismOption>("majority_vote");
  const [transparency, setTransparency] = useState<TransparencyOption>("full");
  const [punishment, setPunishment] = useState(true);
  const [pressureRegime, setPressureRegime] = useState(false);
  const [runSeed, setRunSeed] = useState("");

  const [loading, setLoading] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
  const [error, setError] = useState("");

  const wsUrl = useMemo(() => {
    if (!simulationId) return "";
    return `${API_BASE.replace("http", "ws")}/ws/simulations/${simulationId}`;
  }, [simulationId]);

  const resourceRoundReports = useMemo(() => parseResourceRoundReports(events), [events]);
  const resourceFinalRecommendation = useMemo(() => parseResourceFinalRecommendation(events), [events]);

  const resourceRoundViews = useMemo<ResourceRoundView[]>(() => {
    const discussionsByRound: Record<number, ResourceDiscussionItem[]> = {};
    for (const event of events) {
      if (event.event_type !== "message") continue;
      const source = event.source_agent || "";
      if (!["A", "B", "C"].includes(source)) continue;
      const round = parseResourceRoundFromText(event.content);
      if (!round) continue;
      if (!discussionsByRound[round]) discussionsByRound[round] = [];
      discussionsByRound[round].push({
        id: event.id,
        step: event.step,
        round,
        source_agent: source,
        target_agent: event.target_agent || null,
        text: parseResourceDiscussionText(event.content),
      });
    }

    const rounds = new Set<number>();
    for (const key of Object.keys(resourceRoundReports)) rounds.add(Number(key));
    for (const key of Object.keys(discussionsByRound)) rounds.add(Number(key));

    return Array.from(rounds)
      .sort((a, b) => a - b)
      .map((round) => ({
        round,
        report: resourceRoundReports[round] || null,
        discussions: (discussionsByRound[round] || []).sort((a, b) => a.id - b.id),
      }));
  }, [events, resourceRoundReports]);

  const roundCards = useMemo(() => [...resourceRoundViews].reverse(), [resourceRoundViews]);
  const latestRound = resourceRoundViews.length ? resourceRoundViews[resourceRoundViews.length - 1].report : null;
  const recentTimeline = useMemo(() => events.slice(-140), [events]);

  const fatalEvent = useMemo(() => {
    const found = [...events].reverse().find((event) => event.content.includes("RESOURCE_SCENARIO_FATAL"));
    return found || null;
  }, [events]);

  const resourceSummary = useMemo(() => {
    if (!finalReport) return null;
    const reportJson = asRecord(finalReport.report_json);
    if (!reportJson) return null;
    return asRecord(reportJson.resource_allocation);
  }, [finalReport]);

  const realism = useMemo(() => {
    if (!resourceSummary) return null;
    return asRecord(resourceSummary.realism);
  }, [resourceSummary]);

  const realismComponents = useMemo(() => {
    if (!realism) return null;
    return asRecord(realism.components);
  }, [realism]);

  const realismScore = readNumber(realism?.score, 0);
  const realismGrade = readString(realism?.grade, "unknown");
  const realismPassed = Boolean(realism?.standard_passed);
  const causalConsistencyRate = readNumber(
    resourceSummary?.causal_consistency_rate,
    readNumber(realismComponents?.causal_consistency_score, 0)
  );
  const institutionalSensitivity = readNumber(
    resourceSummary?.institutional_sensitivity_score,
    readNumber(realismComponents?.institutional_sensitivity_score, 0)
  );
  const narrativeConsistency = readNumber(
    resourceSummary?.narrative_data_consistency_score,
    readNumber(realismComponents?.narrative_data_consistency_score, 0)
  );
  const roundCount = readNumber(resourceSummary?.round_count, resourceRoundViews.length);
  const totalOutput = readNumber(
    resourceSummary?.total_output,
    resourceRoundViews.reduce((acc, item) => acc + readNumber(item.report?.total_output, 0), 0)
  );
  const meanOutput = readNumber(resourceSummary?.mean_output, roundCount > 0 ? totalOutput / roundCount : 0);
  const conflictRounds = readNumber(
    resourceSummary?.conflict_rounds,
    resourceRoundViews.filter((item) => Boolean(item.report?.conflict)).length
  );
  const lowRealismSignal = readNumber(realismComponents?.conflict_rate, 0) < 0.08;

  const agentSnapshots = useMemo(() => {
    return ["A", "B", "C"].map((token) => {
      const proposal = latestRound?.agent_proposals?.[token];
      const mood = readNumber(proposal?.mood, readNumber(latestRound?.moods?.[token], 0));
      const stress = readNumber(proposal?.stress, readNumber(latestRound?.stresses?.[token], 0));
      const effort = readNumber(proposal?.effort, readNumber(latestRound?.efforts?.[token], 0));
      const effectiveEffort = readNumber(
        proposal?.effective_effort,
        readNumber(latestRound?.effective_efforts?.[token], effort)
      );
      const willingness = readNumber(
        proposal?.willingness_units,
        readNumber(latestRound?.commitment_units?.[token], 0)
      );
      const payoff = readNumber(latestRound?.net_payoff_units?.[token], 0);
      const influence = readString(proposal?.influenced_by, "NONE");
      const deltaEffort = readNumber(proposal?.effort_delta, 0);
      const deltaWilling = readNumber(proposal?.willingness_delta, 0);
      const consistency = readNumber(proposal?.self_critique?.consistency_score, 0);
      const confidence = readNumber(proposal?.self_critique?.confidence, 0);
      return {
        token,
        role: AGENT_META[token].role,
        behavior: AGENT_META[token].behavior,
        mood,
        stress,
        effort,
        effectiveEffort,
        willingness,
        payoff,
        influence,
        deltaEffort,
        deltaWilling,
        consistency,
        confidence,
      };
    });
  }, [latestRound]);

  const expectedDynamics = useMemo(() => {
    const lines: string[] = [];
    if (mechanism === "equal") {
      lines.push("Equal mechanism tends to reduce incentive for high-efficiency agents to over-contribute.");
    } else if (mechanism === "contribution") {
      lines.push("Contribution mechanism rewards effective effort and usually improves output when trust is stable.");
    } else if (mechanism === "dictator") {
      lines.push("Dictator mechanism often creates larger payoff disparity and can trigger fairness conflict.");
    } else {
      lines.push("Majority vote can switch mechanism by round, so volatility depends on vote alignment.");
    }

    if (transparency === "hidden") {
      lines.push("Hidden transparency should increase belief uncertainty and can accelerate trust decline.");
    } else {
      lines.push("Full transparency should make peer reactions more evidence-driven and reduce misinterpretation.");
    }

    if (punishment) {
      lines.push("Punishment ON can enforce short-term compliance but increases institution cost under conflict.");
    } else {
      lines.push("Punishment OFF avoids extra enforcement cost but relies more on voluntary coordination.");
    }
    if (pressureRegime) {
      lines.push("Pressure regime ON should create stronger conflict-negotiation-revision loops.");
    } else {
      lines.push("Pressure regime OFF is better for neutral baseline comparisons.");
    }

    return lines;
  }, [mechanism, transparency, punishment, pressureRegime]);

  const loadSimulationSnapshot = useCallback(async (id: number) => {
    const [eventResp, simResp, reportResp] = await Promise.all([
      fetch(`${API_BASE}/api/v1/simulations/${id}/events?limit=500`),
      fetch(`${API_BASE}/api/v1/simulations/${id}`),
      fetch(`${API_BASE}/api/v1/simulations/${id}/report`),
    ]);

    if (!eventResp.ok || !simResp.ok || !reportResp.ok) {
      throw new Error("Failed to load simulation snapshot");
    }

    const eventData = (await eventResp.json()) as SimulationEvent[];
    const simData = await simResp.json();
    const reportData = await reportResp.json();

    setEvents(eventData);
    setSimulationStatus((simData.simulation?.status || "idle") as SimulationStatus);
    setStep(Number(simData.simulation?.step || 0));
    setFinalReport((reportData.report || null) as SimulationReportData | null);
  }, []);

  const injectRunConfig = useCallback(
    async (id: number) => {
      const transparencyText = transparency === "full" ? "full transparency" : "hidden transparency";
      const punishmentText = punishment ? "punishment enabled" : "no punishment";
      const mechanismText = mechanism === "majority_vote" ? "majority vote" : mechanism;
      const seedValue = parseOptionalPositiveInt(runSeed);
      const seedText = seedValue === null ? "run_seed=auto" : `run_seed=${seedValue}`;
      const pressureText = pressureRegime ? "pressure_regime=on" : "pressure_regime=off";
      const content = `Use ${mechanismText} mechanism with ${transparencyText} and ${punishmentText}. ${pressureText}. ${seedText}.`;
      const response = await fetch(`${API_BASE}/api/v1/simulations/${id}/inject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      });
      if (!response.ok) {
        throw new Error("Failed to inject scenario configuration");
      }
    },
    [mechanism, pressureRegime, punishment, runSeed, transparency]
  );

  async function createAndStartResourceRun() {
    setLoading(true);
    setError("");
    try {
      const payload = {
        max_steps: toNumberInput(maxSteps, 8, 1, 200),
        tick_interval_ms: toNumberInput(tickIntervalMs, 500, 10, 10000),
      };
      const createdResp = await fetch(`${API_BASE}/api/v1/scenarios/${RESOURCE_SCENARIO_ID}/simulations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!createdResp.ok) throw new Error("Failed to create resource scenario simulation");
      const created = await createdResp.json();
      const id = Number(created.simulation?.id);
      if (!id) throw new Error("Invalid simulation id");

      setSimulationId(id);
      setSimulationStatus("created");
      setStep(0);
      setEvents([]);
      setFinalReport(null);

      await injectRunConfig(id);
      const startResp = await fetch(`${API_BASE}/api/v1/simulations/${id}/start`, { method: "POST" });
      if (!startResp.ok) throw new Error("Failed to start simulation");

      setSimulationStatus("running");
      await loadSimulationSnapshot(id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  async function pauseSimulation() {
    if (!simulationId) return;
    setError("");
    const resp = await fetch(`${API_BASE}/api/v1/simulations/${simulationId}/pause`, { method: "POST" });
    if (!resp.ok) {
      setError("Failed to pause simulation");
      return;
    }
    setSimulationStatus("paused");
  }

  async function resumeSimulation() {
    if (!simulationId) return;
    setError("");
    const resp = await fetch(`${API_BASE}/api/v1/simulations/${simulationId}/resume`, { method: "POST" });
    if (!resp.ok) {
      setError("Failed to resume simulation");
      return;
    }
    setSimulationStatus("running");
  }

  async function stopSimulation() {
    if (!simulationId) return;
    setError("");
    const resp = await fetch(`${API_BASE}/api/v1/simulations/${simulationId}/stop`, { method: "POST" });
    if (!resp.ok) {
      setError("Failed to stop simulation");
      return;
    }
    setSimulationStatus("stopped");
    await loadSimulationSnapshot(simulationId);
  }

  async function refreshSnapshot() {
    if (!simulationId) return;
    setError("");
    try {
      await loadSimulationSnapshot(simulationId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to refresh");
    }
  }

  async function regenerateFinalReport() {
    if (!simulationId) return;
    setReportLoading(true);
    setError("");
    try {
      const resp = await fetch(`${API_BASE}/api/v1/simulations/${simulationId}/report/regenerate`, {
        method: "POST",
      });
      if (!resp.ok) throw new Error("Failed to regenerate report");
      const data = await resp.json();
      setFinalReport((data.report || null) as SimulationReportData | null);
      await loadSimulationSnapshot(simulationId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to regenerate report");
    } finally {
      setReportLoading(false);
    }
  }

  useEffect(() => {
    if (!simulationId || !wsUrl) return;
    const ws = new WebSocket(wsUrl);
    setWsStatus("idle");

    ws.onopen = () => setWsStatus("connected");
    ws.onclose = () => setWsStatus("disconnected");
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data) as
        | { type: "tick"; step: number }
        | { type: "event"; event: SimulationEvent }
        | { type: "status"; status: SimulationStatus };

      if (data.type === "tick") {
        setStep(data.step);
        return;
      }
      if (data.type === "event") {
        setEvents((prev) => [...prev.slice(-499), data.event]);
        if (data.event.event_type === "report") {
          loadSimulationSnapshot(simulationId).catch(() => {});
        }
        return;
      }
      if (data.type === "status") {
        setSimulationStatus(data.status);
        if (data.status === "completed" || data.status === "stopped") {
          loadSimulationSnapshot(simulationId).catch(() => {});
        }
      }
    };

    const timer = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send("ping");
    }, 10000);

    return () => {
      clearInterval(timer);
      ws.close();
    };
  }, [loadSimulationSnapshot, simulationId, wsUrl]);

  return (
    <main>
      <div className="community-page">
        <header className="hero">
          <div className="hero-topline">AI Agent Community Simulator</div>
          <h1 className="hero-title">Virtual Community Reality Board</h1>
          <p className="hero-sub">
            This board tracks every negotiation round as a causal chain: peer messages, numeric proposal revision,
            trust/stress dynamics, and final collective outcome. The goal is to make the simulated community feel like
            a real social system under pressure.
          </p>
          <div className="hero-meta">
            <div className="hero-metric">
              <div className="hero-metric-label">Scenario</div>
              <div className="hero-metric-value">Resource Allocation</div>
            </div>
            <div className="hero-metric">
              <div className="hero-metric-label">Simulation</div>
              <div className="hero-metric-value">{simulationId ?? "-"}</div>
            </div>
            <div className="hero-metric">
              <div className="hero-metric-label">Live Step</div>
              <div className="hero-metric-value">{step}</div>
            </div>
            <div className="hero-metric">
              <div className="hero-metric-label">WebSocket</div>
              <div className="hero-metric-value">{wsStatus}</div>
            </div>
          </div>
        </header>

        <div className="page-grid">
          <div className="stack-col">
            <section className="panel">
              <div className="panel-head">
                <h2>Run Control</h2>
                <span className="status-chip">
                  <span className={`status-dot ${simulationStatus}`} />
                  {simulationStatus}
                </span>
              </div>
              <div className="panel-body">
                <div className="form-grid">
                  <label className="field">
                    Max steps
                    <input
                      type="number"
                      min={1}
                      max={200}
                      value={maxSteps}
                      onChange={(e) => setMaxSteps(e.target.value)}
                    />
                    <span className="field-hint">Suggested: 8 for iteration, 14 for stress audit upper bound.</span>
                  </label>
                  <label className="field">
                    Tick (ms)
                    <input
                      type="number"
                      min={10}
                      max={10000}
                      value={tickIntervalMs}
                      onChange={(e) => setTickIntervalMs(e.target.value)}
                    />
                  </label>
                  <label className="field">
                    Mechanism
                    <select
                      value={mechanism}
                      onChange={(e) =>
                        setMechanism(e.target.value as MechanismOption)
                      }
                    >
                      <option value="equal">equal</option>
                      <option value="contribution">contribution</option>
                      <option value="dictator">dictator</option>
                      <option value="majority_vote">majority_vote</option>
                    </select>
                  </label>
                  <label className="field">
                    Transparency
                    <select value={transparency} onChange={(e) => setTransparency(e.target.value as TransparencyOption)}>
                      <option value="full">full</option>
                      <option value="hidden">hidden</option>
                    </select>
                  </label>
                  <label className="field">
                    Run seed
                    <input
                      type="number"
                      min={0}
                      value={runSeed}
                      onChange={(e) => setRunSeed(e.target.value)}
                      placeholder="auto"
                    />
                  </label>
                  <label className="field full toggle-line">
                    <input type="checkbox" checked={punishment} onChange={(e) => setPunishment(e.target.checked)} />
                    Punishment enabled
                  </label>
                  <label className="field full toggle-line">
                    <input
                      type="checkbox"
                      checked={pressureRegime}
                      onChange={(e) => setPressureRegime(e.target.checked)}
                    />
                    Pressure regime enabled (explicit)
                  </label>
                </div>

                <div className="button-row">
                  <button className="btn primary" onClick={createAndStartResourceRun} disabled={loading}>
                    {loading ? "Starting..." : "Create and Start"}
                  </button>
                  <button className="btn secondary" onClick={refreshSnapshot} disabled={!simulationId}>
                    Refresh Snapshot
                  </button>
                  <button className="btn secondary" onClick={pauseSimulation} disabled={!simulationId || simulationStatus !== "running"}>
                    Pause
                  </button>
                  <button className="btn secondary" onClick={resumeSimulation} disabled={!simulationId || simulationStatus !== "paused"}>
                    Resume
                  </button>
                  <button
                    className="btn secondary"
                    onClick={stopSimulation}
                    disabled={!simulationId || !["running", "paused"].includes(simulationStatus)}
                  >
                    Stop
                  </button>
                  <button
                    className="btn secondary"
                    onClick={regenerateFinalReport}
                    disabled={!simulationId || !["stopped", "completed"].includes(simulationStatus) || reportLoading}
                  >
                    {reportLoading ? "Regenerating..." : "Regenerate Report"}
                  </button>
                </div>

                <div className="kv">
                  <div className="kv-item">
                    <span>Scenario</span>
                    <strong>{RESOURCE_SCENARIO_ID}</strong>
                  </div>
                  <div className="kv-item">
                    <span>Simulation ID</span>
                    <strong>{simulationId ?? "-"}</strong>
                  </div>
                  <div className="kv-item">
                    <span>Events captured</span>
                    <strong>{events.length}</strong>
                  </div>
                </div>

                {error ? <div className="error-box">Error: {error}</div> : null}
                {fatalEvent ? <div className="error-box">Fatal: {stripEventTimestamp(fatalEvent.content)}</div> : null}
              </div>
            </section>

            <section className="panel">
              <div className="panel-head">
                <h3>Rules Snapshot</h3>
                <span className="status-chip">Model Contract</span>
              </div>
              <div className="panel-body">
                <div className="guide-grid compact">
                  <article className="guide-card">
                    <h4>Current run parameters</h4>
                    <div className="guide-kv">
                      <div className="guide-kv-row">
                        <span>Mechanism ({mechanism})</span>
                        <strong>{MECHANISM_GUIDE[mechanism]}</strong>
                      </div>
                      <div className="guide-kv-row">
                        <span>Transparency ({transparency})</span>
                        <strong>{TRANSPARENCY_GUIDE[transparency]}</strong>
                      </div>
                      <div className="guide-kv-row">
                        <span>Punishment ({punishment ? "on" : "off"})</span>
                        <strong>{punishment ? PUNISHMENT_GUIDE.enabled : PUNISHMENT_GUIDE.disabled}</strong>
                      </div>
                      <div className="guide-kv-row">
                        <span>Pressure ({pressureRegime ? "on" : "off"})</span>
                        <strong>{pressureRegime ? PRESSURE_GUIDE.enabled : PRESSURE_GUIDE.disabled}</strong>
                      </div>
                      <div className="guide-kv-row">
                        <span>Execution window / seed</span>
                        <strong>
                          max_steps={toNumberInput(maxSteps, 8, 1, 200)} | tick={toNumberInput(tickIntervalMs, 500, 10, 10000)}ms |
                          seed={parseOptionalPositiveInt(runSeed) ?? "auto"}
                        </strong>
                      </div>
                    </div>
                  </article>

                  <article className="guide-card">
                    <h4>Expected dynamics</h4>
                    <ul className="guide-list">
                      {expectedDynamics.map((line) => (
                        <li key={`expected-${line}`}>{line}</li>
                      ))}
                    </ul>
                  </article>
                </div>

                <details className="guide-fold">
                  <summary>Show full rule contract and audit checklist</summary>
                  <div className="guide-fold-content">
                    <h4>Fixed game rules</h4>
                    <ul className="guide-list">
                      {CORE_RULES.map((line) => (
                        <li key={`rule-${line}`}>{line}</li>
                      ))}
                    </ul>
                    <h4>Data reasonableness checks</h4>
                    <ul className="guide-list">
                      {REASONABLENESS_CHECKLIST.map((line) => (
                        <li key={`check-${line}`}>{line}</li>
                      ))}
                    </ul>
                    <div className="guide-note">
                      Latest round snapshot: mechanism={readString(latestRound?.selected_mechanism, "-")}, output=
                      {readNumber(latestRound?.total_output, 0).toFixed(2)}, trust={readNumber(latestRound?.trust_mean, 0).toFixed(3)},
                      conflict={latestRound?.conflict ? "yes" : "no"}.
                    </div>
                  </div>
                </details>
              </div>
            </section>

            <section className="panel">
              <div className="panel-head">
                <h3>Authenticity Signal</h3>
                <span className="status-chip">{getGradeText(realismGrade)}</span>
              </div>
              <div className="panel-body">
                <div className="auth-grid">
                  <div className="auth-card">
                    <div className="auth-title">Realism score</div>
                    <div className="auth-value">{realismScore.toFixed(1)} / 100</div>
                    <div className="bar-track">
                      <div className={getToneByGrade(realismGrade)} style={{ width: `${clampPercent(realismScore)}%` }} />
                    </div>
                  </div>
                  <div className="auth-card">
                    <div className="auth-title">Standard passed</div>
                    <div className="auth-value">{realismPassed ? "YES" : "NO"}</div>
                    <div className="bar-track">
                      <div
                        className={realismPassed ? "bar-fill" : "bar-fill low"}
                        style={{ width: `${realismPassed ? 100 : 38}%` }}
                      />
                    </div>
                  </div>
                  <div className="auth-card">
                    <div className="auth-title">Causal consistency</div>
                    <div className="auth-value">{(causalConsistencyRate * 100).toFixed(1)}%</div>
                    <div className="bar-track">
                      <div className="bar-fill" style={{ width: `${clampPercent(causalConsistencyRate * 100)}%` }} />
                    </div>
                  </div>
                  <div className="auth-card">
                    <div className="auth-title">Institution sensitivity</div>
                    <div className="auth-value">{(institutionalSensitivity * 100).toFixed(1)}%</div>
                    <div className="bar-track">
                      <div
                        className={institutionalSensitivity >= 0.4 ? "bar-fill" : "bar-fill warn"}
                        style={{ width: `${clampPercent(institutionalSensitivity * 100)}%` }}
                      />
                    </div>
                  </div>
                  <div className="auth-card">
                    <div className="auth-title">Narrative-data match</div>
                    <div className="auth-value">{(narrativeConsistency * 100).toFixed(1)}%</div>
                    <div className="bar-track">
                      <div className="bar-fill" style={{ width: `${clampPercent(narrativeConsistency * 100)}%` }} />
                    </div>
                  </div>
                  <div className="auth-card">
                    <div className="auth-title">Conflict rounds</div>
                    <div className="auth-value">
                      {conflictRounds.toFixed(0)} / {Math.max(roundCount, 0).toFixed(0)}
                    </div>
                    <div className="bar-track">
                      <div
                        className={lowRealismSignal ? "bar-fill low" : "bar-fill warn"}
                        style={{
                          width: `${clampPercent(roundCount > 0 ? (conflictRounds / roundCount) * 100 : 0)}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <section className="panel">
              <div className="panel-head">
                <h3>Event Timeline</h3>
                <span className="status-chip">Last {recentTimeline.length}</span>
              </div>
              <div className="panel-body">
                <div className="timeline">
                  {recentTimeline.length ? (
                    recentTimeline.map((event) => (
                      <div key={`timeline-${event.id}`} className="timeline-item">
                        [{event.step}] {event.event_type} {event.source_agent ?? "-"}
                        {event.target_agent ? ` -> ${event.target_agent}` : ""}: {stripEventTimestamp(event.content)}
                      </div>
                    ))
                  ) : (
                    <div>No events yet.</div>
                  )}
                </div>
              </div>
            </section>

            <section className="panel">
              <div className="panel-head">
                <h3>Conclusion Report</h3>
                <span className="status-chip">{finalReport ? `v${finalReport.version}` : "not ready"}</span>
              </div>
              <div className="panel-body">
                {finalReport ? (
                  <>
                    <div className="kv">
                      <div className="kv-item">
                        <span>Title</span>
                        <strong>{finalReport.title}</strong>
                      </div>
                      <div className="kv-item">
                        <span>Generated</span>
                        <strong>{finalReport.created_at}</strong>
                      </div>
                    </div>
                    <pre className="report-box">{finalReport.markdown}</pre>
                  </>
                ) : (
                  <div>No report yet. Stop or complete a run to generate one.</div>
                )}
              </div>
            </section>
          </div>

          <div className="stack-col">
            <section className="panel">
              <div className="panel-head">
                <h2>Community Pulse</h2>
                <span className="status-chip">
                  Rounds {roundCount.toFixed(0)} | Output {meanOutput.toFixed(2)} / round
                </span>
              </div>
              <div className="panel-body board-grid">
                <div className="agent-grid">
                  {agentSnapshots.map((agent) => {
                    const moodPercent = clampPercent((agent.mood + 1) * 50);
                    const stressPercent = clampPercent(agent.stress * 100);
                    return (
                      <article key={`agent-${agent.token}`} className="agent-card">
                        <div className="agent-head">
                          <div>
                            <div className="agent-name">
                              Agent {agent.token}
                              {agent.influence !== "NONE" ? ` -> listens to ${agent.influence}` : ""}
                            </div>
                            <div className="agent-role">{agent.role}</div>
                          </div>
                          <span className="agent-badge">{getMoodLabel(agent.mood, agent.stress)}</span>
                        </div>
                        <div className="agent-stats">
                          <div>{agent.behavior}</div>
                          <div className="agent-stat-row">
                            <span>Effort / effective</span>
                            <strong>
                              {agent.effort.toFixed(2)} / {agent.effectiveEffort.toFixed(2)}
                            </strong>
                          </div>
                          <div className="agent-stat-row">
                            <span>Willingness units</span>
                            <strong>{agent.willingness.toFixed(2)}</strong>
                          </div>
                          <div className="agent-stat-row">
                            <span>Payoff</span>
                            <strong>{agent.payoff.toFixed(2)}</strong>
                          </div>
                          <div className="agent-stat-row">
                            <span>Delta (effort / willingness)</span>
                            <strong>
                              {signed(agent.deltaEffort, 2)} / {signed(agent.deltaWilling, 2)}
                            </strong>
                          </div>
                          <div className="agent-stat-row">
                            <span>Self-check (consistency / confidence)</span>
                            <strong>
                              {(agent.consistency * 100).toFixed(0)}% / {(agent.confidence * 100).toFixed(0)}%
                            </strong>
                          </div>
                          <div>
                            Mood
                            <div className="bar-track">
                              <div className="bar-fill" style={{ width: `${moodPercent}%` }} />
                            </div>
                          </div>
                          <div>
                            Stress
                            <div className="bar-track">
                              <div
                                className={stressPercent >= 70 ? "bar-fill warn" : "bar-fill"}
                                style={{ width: `${stressPercent}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      </article>
                    );
                  })}
                </div>

                <div className="kv">
                  <div className="kv-item">
                    <span>Total output</span>
                    <strong>{totalOutput.toFixed(2)}</strong>
                  </div>
                  <div className="kv-item">
                    <span>Mean output per round</span>
                    <strong>{meanOutput.toFixed(2)}</strong>
                  </div>
                  <div className="kv-item">
                    <span>Latest mechanism</span>
                    <strong>{readString(latestRound?.selected_mechanism, "-")}</strong>
                  </div>
                  <div className="kv-item">
                    <span>Latest trust mean</span>
                    <strong>{readNumber(latestRound?.trust_mean, 0).toFixed(3)}</strong>
                  </div>
                </div>
              </div>
            </section>

            <section className="panel round-section">
              <div className="panel-head">
                <h2>Negotiation Reality Replay</h2>
                <span className="status-chip">{roundCards.length} rounds parsed</span>
              </div>
              <div className="panel-body board-grid">
                {roundCards.length ? (
                  roundCards.map((item) => (
                    <article key={`round-${item.round}`} className="round-card">
                      <div className="round-head">
                        <strong>Round {item.round}</strong>
                        <span className="round-pill">Mechanism: {readString(item.report?.selected_mechanism, "-")}</span>
                        <span className="round-pill">
                          Output: {readNumber(item.report?.total_output, 0).toFixed(2)}
                        </span>
                        <span className="round-pill">
                          Stability: {readNumber(item.report?.public_stability_index, 0).toFixed(3)}
                        </span>
                        <span className="round-pill">
                          Welfare: {readNumber(item.report?.social_welfare_index, 0).toFixed(3)}
                        </span>
                        <span className={`round-pill ${item.report?.conflict ? "risk" : ""}`}>
                          Conflict: {item.report?.conflict ? "yes" : "no"}
                        </span>
                      </div>
                      <div className="round-body">
                        <div className="message-stream">
                          <h4 style={{ margin: 0, fontSize: "0.95rem" }}>Dialogue Stream</h4>
                          {item.discussions.length ? (
                            item.discussions.map((discussion) => (
                              <div key={`discussion-${discussion.id}`} className="message-item">
                                <strong>{discussion.source_agent}</strong>
                                {discussion.target_agent ? ` -> ${discussion.target_agent}` : ""}:{" "}
                                {discussion.text || "(empty message)"}
                              </div>
                            ))
                          ) : (
                            <div className="message-item">No message captured for this round.</div>
                          )}
                        </div>
                        <div className="evidence-box">
                          <h4 style={{ margin: 0, fontSize: "0.95rem" }}>Data and Causal Evidence</h4>
                          {item.report ? (
                            <>
                              <div className="triplet">
                                <div className="triplet-row">
                                  <span>Effort</span>
                                  <strong>{formatTriplet(item.report.efforts, 2)}</strong>
                                </div>
                                <div className="triplet-row">
                                  <span>Effective effort</span>
                                  <strong>{formatTriplet(item.report.effective_efforts, 2)}</strong>
                                </div>
                                <div className="triplet-row">
                                  <span>Contribution</span>
                                  <strong>{formatTriplet(item.report.weighted_contributions, 2)}</strong>
                                </div>
                                <div className="triplet-row">
                                  <span>Willingness</span>
                                  <strong>{formatTriplet(item.report.commitment_units, 1)}</strong>
                                </div>
                                <div className="triplet-row">
                                  <span>Allocation</span>
                                  <strong>{formatTriplet(item.report.allocation_plan_units, 1)}</strong>
                                </div>
                                <div className="triplet-row">
                                  <span>Net payoff</span>
                                  <strong>{formatTriplet(item.report.net_payoff_units, 2)}</strong>
                                </div>
                                <div className="triplet-row">
                                  <span>Gap to target</span>
                                  <strong>{readNumber(item.report.target_gap, 0).toFixed(2)}</strong>
                                </div>
                                <div className="triplet-row">
                                  <span>Causal score</span>
                                  <strong>{readNumber(item.report.causal_consistency_score, 0).toFixed(3)}</strong>
                                </div>
                              </div>
                              <div className="causal-grid">
                                {["A", "B", "C"].map((token) => {
                                  const proposal = item.report?.agent_proposals?.[token];
                                  const quote = readString(proposal?.peer_reference?.quoted_message_snippet, "No quote");
                                  return (
                                    <div key={`causal-${item.round}-${token}`} className="causal-item">
                                      <strong>Agent {token}</strong>: influenced_by {readString(proposal?.influenced_by, "NONE")}
                                      {" | "}delta effort {signed(readNumber(proposal?.effort_delta, 0), 2)}
                                      {" | "}delta willingness {signed(readNumber(proposal?.willingness_delta, 0), 2)}
                                      <br />
                                      Reason: {readString(proposal?.adjustment_reason, "No explicit adjustment reason")}
                                      <br />
                                      Critique: consistency {(readNumber(proposal?.self_critique?.consistency_score, 0) * 100).toFixed(0)}%
                                      , confidence {(readNumber(proposal?.self_critique?.confidence, 0) * 100).toFixed(0)}%
                                      , revised {proposal?.self_critique?.revised_after_check ? "yes" : "no"}
                                      <br />
                                      Peer quote: {quote}
                                    </div>
                                  );
                                })}
                              </div>
                              <div style={{ marginTop: 8, fontSize: 12 }}>
                                <strong>Round conclusion:</strong> {readString(item.report.round_conclusion, "-")}
                              </div>
                            </>
                          ) : (
                            <div style={{ marginTop: 8 }}>No structured round report found.</div>
                          )}
                        </div>
                      </div>
                    </article>
                  ))
                ) : (
                  <div>No negotiation data yet. Start a run to visualize the community dynamics.</div>
                )}
              </div>
            </section>

            <section className="panel">
              <div className="panel-head">
                <h3>Observer Final Recommendation</h3>
                <span className="status-chip">Evidence-backed summary</span>
              </div>
              <div className="panel-body">
                {resourceFinalRecommendation ? (
                  <div className="kv">
                    <div className="kv-item">
                      <span>Recommended mechanism</span>
                      <strong>{readString(resourceFinalRecommendation.recommended_mechanism, "-")}</strong>
                    </div>
                    <div className="kv-item">
                      <span>Total output</span>
                      <strong>{readNumber(resourceFinalRecommendation.total_output, 0).toFixed(2)}</strong>
                    </div>
                    <div className="kv-item">
                      <span>Conflict rounds</span>
                      <strong>{readNumber(resourceFinalRecommendation.conflict_rounds, 0).toFixed(0)}</strong>
                    </div>
                    <div className="kv-item">
                      <span>Free-rider rounds</span>
                      <strong>{readNumber(resourceFinalRecommendation.free_rider_rounds, 0).toFixed(0)}</strong>
                    </div>
                    <div className="kv-item">
                      <span>Rationale</span>
                      <strong>{readString(resourceFinalRecommendation.rationale, "-")}</strong>
                    </div>
                  </div>
                ) : (
                  <div>No final recommendation marker yet.</div>
                )}
              </div>
            </section>
          </div>
        </div>
      </div>
    </main>
  );
}
