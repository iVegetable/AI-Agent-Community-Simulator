"""Core simulation orchestrator.

Coordinates per-step agent decisions, routing, resource-scenario mechanics,
state updates, persistence, and websocket broadcasts.
"""

import asyncio
import json
import logging
import re
from collections.abc import Callable
from typing import Any

from sqlmodel import Session, select

from .agents.adapters.base import AgentDecision, DecisionAdapter, DecisionContext, InboxMessage
from .agents.adapters.mock_adapter import MockDecisionAdapter
from .agents.adapters.openai_adapter import OpenAIDecisionAdapter
from .agents.adapters.rule_based import RuleBasedDecisionAdapter
from .config import settings
from .models import Agent, Event, Simulation, SimulationStatus
from .reporting import generate_simulation_report
from .utils import utc_iso_now

logger = logging.getLogger(__name__)


class Orchestrator:
    """Owns active simulation tasks and executes each run loop."""

    def __init__(self, ws_manager) -> None:
        self.ws_manager = ws_manager
        self._tasks: dict[int, asyncio.Task] = {}
        self._resource_config_cache: dict[int, dict[str, Any]] = {}
        self._resource_round_reports_cache: dict[int, list[dict[str, Any]]] = {}
        self._resource_strike_next_round: dict[int, set[str]] = {}
        self._resource_social_state_cache: dict[int, dict[str, Any]] = {}

    def _reset_resource_state(self, simulation_id: int) -> None:
        self._resource_config_cache.pop(simulation_id, None)
        self._resource_round_reports_cache.pop(simulation_id, None)
        self._resource_strike_next_round.pop(simulation_id, None)
        self._resource_social_state_cache.pop(simulation_id, None)

    async def start(self, simulation_id: int, session_factory: Callable[[], Session]) -> None:
        if simulation_id in self._tasks and not self._tasks[simulation_id].done():
            return
        self._tasks[simulation_id] = asyncio.create_task(self._run(simulation_id, session_factory))

    async def stop(self, simulation_id: int) -> None:
        task = self._tasks.get(simulation_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._reset_resource_state(simulation_id)

    async def _run(self, simulation_id: int, session_factory: Callable[[], Session]) -> None:
        adapter = self._pick_adapter()
        fallback_adapter = RuleBasedDecisionAdapter()
        while True:
            planning_data: list[tuple[int, str, DecisionContext]] = []
            next_step = 0
            sleep_ms = settings.tick_interval_ms
            sim_max_steps = 0
            resource_mode = False
            resource_role_map: dict[str, Agent] | None = None
            resource_config: dict[str, Any] | None = None
            resource_target_reached = False

            with session_factory() as session:
                sim = session.get(Simulation, simulation_id)
                if not sim:
                    self._reset_resource_state(simulation_id)
                    return
                if sim.status != SimulationStatus.running:
                    self._reset_resource_state(simulation_id)
                    return
                if sim.step >= sim.max_steps:
                    sim.status = SimulationStatus.completed
                    session.add(sim)
                    session.commit()
                    _, report_event = generate_simulation_report(session, simulation_id, force=False)
                    if report_event:
                        await self.ws_manager.broadcast(
                            simulation_id,
                            {
                                "type": "event",
                                "event": {
                                    "id": report_event.id,
                                    "step": report_event.step,
                                    "event_type": report_event.event_type,
                                    "source_agent": report_event.source_agent,
                                    "target_agent": report_event.target_agent,
                                    "content": report_event.content,
                                },
                            },
                        )
                    self._reset_resource_state(simulation_id)
                    return

                next_step = sim.step + 1
                sleep_ms = sim.tick_interval_ms
                sim_max_steps = sim.max_steps
                agents = list(session.exec(select(Agent).where(Agent.simulation_id == sim.id)).all())
                resource_role_map = self._resource_role_agent_map(agents)
                resource_mode = resource_role_map is not None
                if resource_mode:
                    resource_config = self._resource_run_config(
                        session=session,
                        simulation_id=simulation_id,
                        goal=sim.goal,
                    )
                partner_counts = self._partner_counts(session, sim.id)
                for agent in agents:
                    goal_text = sim.goal
                    if resource_mode and resource_config:
                        goal_text = self._resource_goal_prompt(
                            base_goal=sim.goal,
                            simulation_id=simulation_id,
                            step=next_step,
                            agent_name=agent.name,
                            run_config=resource_config,
                        )
                    ctx = DecisionContext(
                        goal=goal_text,
                        step=next_step,
                        agent_name=agent.name,
                        agent_role=agent.role,
                        peers=[a.name for a in agents if a.id != agent.id],
                        memory_history=self._memory_history(agent.memory),
                        inbox=self._build_inbox(session, sim.id, sim.step, agent.name),
                    )
                    planning_data.append((agent.id or 0, agent.name, ctx))

            decisions: dict[int, tuple[str, Any]] = {}
            for agent_id, agent_name, ctx in planning_data:
                try:
                    decision = await adapter.decide(ctx)
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {str(exc)[:160]}"
                    if resource_mode:
                        await self._resource_fatal_stop(
                            simulation_id=simulation_id,
                            session_factory=session_factory,
                            step=next_step,
                            reason=f"AI decision failed for {agent_name}: {reason}",
                        )
                        return
                    logger.warning(
                        "fallback to rule adapter simulation=%s step=%s agent=%s reason=%s",
                        simulation_id,
                        ctx.step,
                        agent_name,
                        reason,
                    )
                    decision = await fallback_adapter.decide(ctx)
                    decision.memory_append.append("fallback:rule")
                    decision.memory_append.append(f"fallback_reason:{reason}")
                    decision.content = f"{decision.content} (fallback:{type(exc).__name__})"
                decision = self._normalize_decision(
                    decision=decision,
                    source_agent=agent_name,
                    peers=ctx.peers,
                    inbox=ctx.inbox,
                    partner_counts=partner_counts,
                )
                decisions[agent_id] = (agent_name, decision)

            if resource_mode and resource_role_map and resource_config:
                try:
                    decisions, resource_target_reached = self._decorate_resource_ai_decisions(
                        simulation_id=simulation_id,
                        step=next_step,
                        max_steps=sim_max_steps,
                        decisions=decisions,
                        role_map=resource_role_map,
                        run_config=resource_config,
                    )
                except Exception as exc:
                    await self._resource_fatal_stop(
                        simulation_id=simulation_id,
                        session_factory=session_factory,
                        step=next_step,
                        reason=f"Structured proposal parse/validation failed: {type(exc).__name__}: {str(exc)[:200]}",
                    )
                    return

            new_events: list[Event] = []
            new_event_payloads: list[dict[str, Any]] = []
            agent_updates: list[dict[str, Any]] = []
            report_event: Event | None = None
            report_event_payload: dict[str, Any] | None = None
            with session_factory() as session:
                sim = session.get(Simulation, simulation_id)
                if not sim:
                    self._resource_config_cache.pop(simulation_id, None)
                    self._resource_round_reports_cache.pop(simulation_id, None)
                    self._resource_strike_next_round.pop(simulation_id, None)
                    self._resource_social_state_cache.pop(simulation_id, None)
                    return
                if sim.status != SimulationStatus.running:
                    self._resource_config_cache.pop(simulation_id, None)
                    self._resource_round_reports_cache.pop(simulation_id, None)
                    self._resource_strike_next_round.pop(simulation_id, None)
                    self._resource_social_state_cache.pop(simulation_id, None)
                    return
                if sim.step >= sim.max_steps:
                    sim.status = SimulationStatus.completed
                    session.add(sim)
                    session.commit()
                    _, report_event = generate_simulation_report(session, simulation_id, force=False)
                    if report_event:
                        await self.ws_manager.broadcast(
                            simulation_id,
                            {
                                "type": "event",
                                "event": {
                                    "id": report_event.id,
                                    "step": report_event.step,
                                    "event_type": report_event.event_type,
                                    "source_agent": report_event.source_agent,
                                    "target_agent": report_event.target_agent,
                                    "content": report_event.content,
                                },
                            },
                        )
                    self._resource_config_cache.pop(simulation_id, None)
                    self._resource_round_reports_cache.pop(simulation_id, None)
                    self._resource_strike_next_round.pop(simulation_id, None)
                    self._resource_social_state_cache.pop(simulation_id, None)
                    return

                sim.step = next_step
                session.add(sim)
                for agent_id, value in decisions.items():
                    agent_name, decision = value
                    agent = session.get(Agent, agent_id)
                    if not agent:
                        continue
                    if not decision.memory_append:
                        summary = decision.content[:140].replace("\n", " ").strip()
                        decision.memory_append = [f"{decision.event_type}:{summary}"]
                    event = Event(
                        simulation_id=sim.id,
                        step=next_step,
                        event_type=decision.event_type,
                        source_agent=agent_name,
                        target_agent=decision.target_agent,
                        content=f"{decision.content} @ {utc_iso_now()}",
                    )
                    session.add(event)
                    new_events.append(event)
                    self._append_memory(agent, decision.memory_append)
                    session.add(agent)
                    history = self._memory_history(agent.memory)
                    agent_updates.append(
                        {
                            "id": agent.id,
                            "name": agent.name,
                            "role": agent.role,
                            "memory_size": len(history),
                            "last_memory": history[-1] if history else "",
                        }
                    )
                session.commit()
                for event in new_events:
                    session.refresh(event)
                    new_event_payloads.append(
                        {
                            "id": event.id,
                            "step": event.step,
                            "event_type": event.event_type,
                            "source_agent": event.source_agent,
                            "target_agent": event.target_agent,
                            "content": event.content,
                        }
                    )
                if resource_target_reached:
                    sim.status = SimulationStatus.completed
                    session.add(sim)
                    session.commit()
                    _, report_event = generate_simulation_report(session, simulation_id, force=False)
                    if report_event:
                        session.refresh(report_event)
                        report_event_payload = {
                            "id": report_event.id,
                            "step": report_event.step,
                            "event_type": report_event.event_type,
                            "source_agent": report_event.source_agent,
                            "target_agent": report_event.target_agent,
                            "content": report_event.content,
                        }

            for event_payload in new_event_payloads:
                await self.ws_manager.broadcast(
                    simulation_id,
                    {
                        "type": "event",
                        "event": event_payload,
                    },
                )
            for agent_update in agent_updates:
                await self.ws_manager.broadcast(
                    simulation_id,
                    {
                        "type": "agent_update",
                        "agent": agent_update,
                    },
                )
            if report_event_payload:
                await self.ws_manager.broadcast(
                    simulation_id,
                    {
                        "type": "event",
                        "event": report_event_payload,
                    },
                )
                await self.ws_manager.broadcast(
                    simulation_id,
                    {"type": "status", "status": SimulationStatus.completed.value},
                )
                self._resource_config_cache.pop(simulation_id, None)
                self._resource_round_reports_cache.pop(simulation_id, None)
                self._resource_strike_next_round.pop(simulation_id, None)
                self._resource_social_state_cache.pop(simulation_id, None)
                return
            await self.ws_manager.broadcast(simulation_id, {"type": "tick", "step": next_step})
            await asyncio.sleep(sleep_ms / 1000)

    async def _resource_fatal_stop(
        self,
        simulation_id: int,
        session_factory: Callable[[], Session],
        step: int,
        reason: str,
    ) -> None:
        with session_factory() as session:
            sim = session.get(Simulation, simulation_id)
            if not sim:
                return
            sim.status = SimulationStatus.stopped
            session.add(sim)
            fatal_event = Event(
                simulation_id=simulation_id,
                step=max(sim.step, step),
                event_type="action",
                source_agent="system",
                target_agent=None,
                content=f"RESOURCE_SCENARIO_FATAL: {reason}",
            )
            session.add(fatal_event)
            session.commit()
            session.refresh(fatal_event)
            await self.ws_manager.broadcast(
                simulation_id,
                {
                    "type": "event",
                    "event": {
                        "id": fatal_event.id,
                        "step": fatal_event.step,
                        "event_type": fatal_event.event_type,
                        "source_agent": fatal_event.source_agent,
                        "target_agent": fatal_event.target_agent,
                        "content": fatal_event.content,
                    },
                },
            )
            await self.ws_manager.broadcast(
                simulation_id,
                {"type": "status", "status": sim.status.value},
            )
        self._reset_resource_state(simulation_id)

    @staticmethod
    def _parse_pressure_regime(text: str) -> bool | None:
        lowered = text.lower()
        off_markers = (
            "pressure_regime=off",
            "pressure_regime=false",
            "pressure regime off",
            "pressure mode off",
            "disable pressure",
            "pressure disabled",
            "no pressure regime",
        )
        on_markers = (
            "pressure_regime=on",
            "pressure_regime=true",
            "pressure regime on",
            "pressure mode on",
            "enable pressure",
            "pressure enabled",
        )
        if any(marker in lowered for marker in off_markers):
            return False
        if any(marker in lowered for marker in on_markers):
            return True
        return None

    @staticmethod
    def _extract_seed(text: str) -> int | None:
        lowered = text.lower()
        patterns = (
            r"\b(?:run_)?seed\s*(?:=|:)?\s*(-?\d+)\b",
            r"\bseed\s+(-?\d+)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if not match:
                continue
            try:
                value = int(match.group(1))
            except Exception:
                continue
            return abs(value)
        return None

    @staticmethod
    def _resource_seed(simulation_id: int, run_config: dict[str, Any]) -> int:
        try:
            value = int(run_config.get("seed", simulation_id))
        except Exception:
            value = simulation_id
        return abs(value)

    def _resource_role_agent_map(self, agents: list[Agent]) -> dict[str, Agent] | None:
        selected: dict[str, Agent] = {}
        for agent in agents:
            role = agent.role.strip().lower()
            if role == "high-efficiency" and "A" not in selected:
                selected["A"] = agent
            elif role == "fairness-sensitive" and "B" not in selected:
                selected["B"] = agent
            elif role == "self-interested" and "C" not in selected:
                selected["C"] = agent
        if {"A", "B", "C"} - set(selected.keys()):
            return None
        return selected

    def _resource_run_config(
        self,
        session: Session,
        simulation_id: int,
        goal: str,
    ) -> dict[str, Any]:
        cached = self._resource_config_cache.get(simulation_id)
        if cached:
            return cached
        texts = [goal.lower()]
        stmt = (
            select(Event)
            .where(Event.simulation_id == simulation_id)
            .where(Event.event_type == "intervention")
        )
        for event in session.exec(stmt).all():
            texts.append((event.content or "").lower())
        merged = " ".join(texts)

        mechanism = "majority_vote"
        if "dictator" in merged:
            mechanism = "dictator"
        elif "contribution" in merged:
            mechanism = "contribution"
        elif "equal" in merged or "average" in merged:
            mechanism = "equal"
        elif "majority vote" in merged or "majority_vote" in merged:
            mechanism = "majority_vote"

        transparency = "hidden" if "hidden" in merged or "opaque" in merged else "full"
        punishment = True
        if (
            "no punishment" in merged
            or "without punishment" in merged
            or "punishment off" in merged
            or "disable punishment" in merged
        ):
            punishment = False
        elif "punishment" in merged or "strike" in merged:
            punishment = True
        pressure_regime = self._parse_pressure_regime(merged)
        seed = self._extract_seed(merged)

        config = {
            "target_resource": 100.0,
            "mechanism": mechanism,
            "transparency": transparency,
            "punishment": punishment,
            "pressure_regime": bool(pressure_regime) if pressure_regime is not None else False,
            "seed": seed if seed is not None else simulation_id,
        }
        self._resource_config_cache[simulation_id] = config
        return config

    def _resource_social_state(self, simulation_id: int) -> dict[str, dict[str, Any]]:
        cached = self._resource_social_state_cache.get(simulation_id)
        if cached:
            return cached
        state = {
            "A": {
                "mood": 0.10,
                "stress": 0.25,
                "effort_cap": 1.0,
                "reputation": 0.78,
                "alliance_target": "B",
                "trust": {"A": 1.0, "B": 0.55, "C": 0.45},
                "beliefs": {
                    "A": {"credibility": 1.0, "fairness": 1.0, "reliability": 1.0},
                    "B": {"credibility": 0.58, "fairness": 0.62, "reliability": 0.57},
                    "C": {"credibility": 0.46, "fairness": 0.44, "reliability": 0.49},
                },
                "last_utility_norm": 0.0,
                "profile": {
                    "thinking_style": "analytical",
                    "fairness_weight": 0.35,
                    "self_interest_weight": 0.65,
                    "risk_aversion": 0.40,
                    "bounded_rationality": {
                        "inertia": 0.42,
                        "loss_aversion": 0.58,
                        "myopia": 0.54,
                    },
                },
            },
            "B": {
                "mood": 0.05,
                "stress": 0.30,
                "effort_cap": 1.0,
                "reputation": 0.76,
                "alliance_target": "A",
                "trust": {"A": 0.55, "B": 1.0, "C": 0.55},
                "beliefs": {
                    "A": {"credibility": 0.60, "fairness": 0.57, "reliability": 0.56},
                    "B": {"credibility": 1.0, "fairness": 1.0, "reliability": 1.0},
                    "C": {"credibility": 0.53, "fairness": 0.50, "reliability": 0.54},
                },
                "last_utility_norm": 0.0,
                "profile": {
                    "thinking_style": "normative",
                    "fairness_weight": 0.90,
                    "self_interest_weight": 0.35,
                    "risk_aversion": 0.55,
                    "bounded_rationality": {
                        "inertia": 0.63,
                        "loss_aversion": 0.67,
                        "myopia": 0.42,
                    },
                },
            },
            "C": {
                "mood": 0.00,
                "stress": 0.28,
                "effort_cap": 1.0,
                "reputation": 0.71,
                "alliance_target": "A",
                "trust": {"A": 0.45, "B": 0.55, "C": 1.0},
                "beliefs": {
                    "A": {"credibility": 0.51, "fairness": 0.48, "reliability": 0.52},
                    "B": {"credibility": 0.55, "fairness": 0.59, "reliability": 0.57},
                    "C": {"credibility": 1.0, "fairness": 1.0, "reliability": 1.0},
                },
                "last_utility_norm": 0.0,
                "profile": {
                    "thinking_style": "opportunistic",
                    "fairness_weight": 0.25,
                    "self_interest_weight": 0.95,
                    "risk_aversion": 0.30,
                    "bounded_rationality": {
                        "inertia": 0.34,
                        "loss_aversion": 0.49,
                        "myopia": 0.74,
                    },
                },
            },
        }
        self._resource_social_state_cache[simulation_id] = state
        return state

    def _resource_effort_cap(
        self,
        mood: float,
        stress: float,
        punishment_enabled: bool,
        *,
        dissatisfaction: float = 0.0,
        conflict: bool = False,
    ) -> float:
        # Fatigue always exists; punishment changes how fast fatigue turns into strike-like behavior.
        fatigue_cap = 1.0
        if stress > 0.82:
            fatigue_cap = 0.55
        elif stress > 0.68 or mood < -0.62:
            fatigue_cap = 0.72
        elif stress > 0.52 or mood < -0.35:
            fatigue_cap = 0.86

        if not punishment_enabled:
            relaxed_cap = max(0.80, fatigue_cap)
            if dissatisfaction > 0.80 and conflict and stress > 0.90:
                return min(relaxed_cap, 0.75)
            return relaxed_cap

        if dissatisfaction > 0.62 and conflict:
            return min(fatigue_cap, 0.58)
        if dissatisfaction > 0.45 or stress > 0.70:
            return min(fatigue_cap, 0.72)
        return fatigue_cap

    def _resource_round_signal(
        self,
        simulation_id: int,
        step: int,
        run_config: dict[str, Any],
    ) -> dict[str, Any]:
        # Deterministic pseudo-random signal to keep runs reproducible without external RNG state.
        run_seed = self._resource_seed(simulation_id, run_config)
        base = (run_seed * 131 + step * 47 + 17) % 100
        phase = base / 100.0
        transparency = str(run_config.get("transparency", "full")).lower()
        hidden_mode = transparency == "hidden"
        punishment = bool(run_config.get("punishment", True))
        pressure_regime = bool(run_config.get("pressure_regime", False))

        shock_catalog = (
            ("demand_surge", 0.07, 0.00, 0.02, 0.03, 0.04),
            ("supply_disruption", -0.10, 0.04, 0.08, 0.05, 0.07),
            ("policy_audit", -0.03, 0.06, 0.10, 0.08, 0.03),
            ("coordination_breakdown", -0.07, 0.08, 0.12, 0.10, 0.08),
            ("stability_window", 0.05, -0.02, -0.01, -0.03, -0.02),
        )
        shock_idx = ((run_seed * 3) + step * 5 + 1) % len(shock_catalog)
        shock_type, prod_impact, trust_impact, urgency_impact, enforcement_impact, noise_impact = shock_catalog[shock_idx]

        demand_shock = (phase - 0.5) * 0.18 + prod_impact
        micro_noise = (((run_seed + step * 3) % 9) - 4) / 120.0
        productivity_factor = 1.0 + demand_shock + micro_noise
        productivity_factor = max(0.70, min(1.30, productivity_factor))

        trust_shock = max(0.0, trust_impact)
        if hidden_mode:
            trust_shock += 0.03 + 0.05 * abs(demand_shock)
        if base % 7 == 0:
            trust_shock += 0.02
        if pressure_regime:
            trust_shock += 0.035 + 0.04 * abs(demand_shock)

        enforcement_pressure = max(0.0, enforcement_impact)
        if punishment:
            enforcement_pressure += 0.03 + 0.03 * (1.0 if abs(demand_shock) > 0.08 else 0.0)
        else:
            enforcement_pressure *= 0.40
        if pressure_regime:
            enforcement_pressure += 0.03

        urgency = 0.40 + 0.50 * abs(demand_shock) + urgency_impact
        urgency = max(0.0, min(1.0, urgency))
        if pressure_regime:
            urgency = min(1.0, urgency + 0.08)

        communication_noise = max(0.0, 0.03 + noise_impact + (0.04 if hidden_mode else 0.01))
        communication_noise = max(0.0, min(0.25, communication_noise))
        if pressure_regime:
            communication_noise = max(
                communication_noise,
                min(0.25, 0.11 + 0.06 * abs(demand_shock)),
            )
        stability_shock = max(0.0, min(0.30, trust_shock + max(0.0, -prod_impact) * 0.5))

        return {
            "shock_type": shock_type,
            "shock_severity": round(abs(prod_impact) + max(0.0, trust_impact), 4),
            "productivity_factor": round(productivity_factor, 4),
            "demand_shock": round(demand_shock, 4),
            "trust_shock": round(trust_shock, 4),
            "enforcement_pressure": round(enforcement_pressure, 4),
            "urgency": round(urgency, 4),
            "communication_noise": round(communication_noise, 4),
            "stability_shock": round(stability_shock, 4),
        }

    def _resource_perception_noise(
        self,
        run_seed: int,
        step: int,
        observer_token: str,
        subject_token: str,
        amplitude: float,
    ) -> float:
        if amplitude <= 1e-8:
            return 0.0
        raw = ((run_seed * 89) + (step * 37) + (ord(observer_token[0]) * 13) + (ord(subject_token[0]) * 7)) % 100
        centered = raw / 100.0 - 0.5
        return centered * 2.0 * amplitude

    def _resource_goal_prompt(
        self,
        base_goal: str,
        simulation_id: int,
        step: int,
        agent_name: str,
        run_config: dict[str, Any],
    ) -> str:
        history = self._resource_round_reports_cache.get(simulation_id, [])
        last_round = history[-1] if history else None
        strike_next = self._resource_strike_next_round.get(simulation_id, set())
        forced_strike = agent_name in strike_next
        state = self._resource_social_state(simulation_id)
        round_signal = self._resource_round_signal(simulation_id, step, run_config)
        run_seed = self._resource_seed(simulation_id, run_config)
        agent_state = state.get(agent_name, {})
        trust = agent_state.get("trust", {})
        beliefs = agent_state.get("beliefs", {})
        profile = agent_state.get("profile", {})
        effort_cap = float(agent_state.get("effort_cap", 1.0))
        reputation = float(agent_state.get("reputation", 0.6))
        alliance_target = str(agent_state.get("alliance_target", "NONE"))
        preface = [
            "RESOURCE_SCENARIO",
            f"Task goal: {base_goal}",
            f"Round: {step}",
            "Agents: A(high-efficiency), B(fairness-sensitive), C(self-interested).",
            f"Target resource units: {float(run_config['target_resource']):.0f}",
            f"Allocation mechanism: {run_config['mechanism']}",
            f"Transparency: {run_config['transparency']}",
            f"Punishment enabled: {bool(run_config['punishment'])}",
            f"Pressure regime: {bool(run_config.get('pressure_regime', False))}",
            f"Run seed: {run_seed}",
            "V2-Lite game rule: behavior is affected by mood/stress/trust and role profile.",
            (
                "Environment signal this round: "
                f"shock_type={round_signal.get('shock_type', 'none')}, "
                f"shock_severity={float(round_signal.get('shock_severity', 0.0)):.2f}, "
                f"productivity_factor={round_signal['productivity_factor']:.2f}, "
                f"demand_shock={round_signal['demand_shock']:+.2f}, "
                f"trust_shock={round_signal['trust_shock']:.2f}, "
                f"enforcement_pressure={round_signal['enforcement_pressure']:.2f}, "
                f"urgency={round_signal['urgency']:.2f}, "
                f"communication_noise={float(round_signal.get('communication_noise', 0.0)):.2f}."
            ),
            (
                "Your social state: "
                f"mood={float(agent_state.get('mood', 0.0)):.3f}, "
                f"stress={float(agent_state.get('stress', 0.0)):.3f}, "
                f"effort_cap_ratio={effort_cap:.2f}, "
                f"reputation={reputation:.2f}, "
                f"alliance_target={alliance_target}, "
                f"trust(A/B/C)={float(trust.get('A', 0.5)):.2f}/"
                f"{float(trust.get('B', 0.5)):.2f}/"
                f"{float(trust.get('C', 0.5)):.2f}."
            ),
            f"Forced strike this round: {forced_strike}.",
            (
                "Your profile: "
                f"thinking_style={profile.get('thinking_style', 'analytical')}, "
                f"fairness_weight={float(profile.get('fairness_weight', 0.5)):.2f}, "
                f"self_interest_weight={float(profile.get('self_interest_weight', 0.5)):.2f}, "
                f"risk_aversion={float(profile.get('risk_aversion', 0.5)):.2f}."
            ),
            (
                "Bounded rationality: "
                f"inertia={float(profile.get('bounded_rationality', {}).get('inertia', 0.5)):.2f}, "
                f"loss_aversion={float(profile.get('bounded_rationality', {}).get('loss_aversion', 0.5)):.2f}, "
                f"myopia={float(profile.get('bounded_rationality', {}).get('myopia', 0.5)):.2f}."
            ),
            "You MUST include numbers in your message: effort(0-10), willingness units(0-100), and allocation plan A/B/C summing to 100.",
            "Your content MUST include one line: RESOURCE_AGENT_PROPOSAL {json}.",
            (
                'JSON minimum keys: effort, willingness_units, allocation_proposal_units({"A":..,"B":..,"C":..), '
                "vote_mechanism(equal|contribution), message, self_critique."
            ),
            "Optional keys for richer reasoning: influenced_by, adjustment_reason, previous_effort, previous_willingness_units, effort_delta, willingness_delta, peer_reference.",
            "Two-stage decision rule: produce proposal first, then self-critique consistency and possible revision.",
            "Prefer reacting to peers and changing numbers when useful, but keep decisions autonomous.",
        ]
        if bool(run_config.get("pressure_regime", False)) and step > 1:
            preface.append(
                "Pressure-mode requirement: choose influenced_by from A/B/C (not self), include peer_reference with quoted snippet, and revise at least one numeric field from previous round."
            )
        if isinstance(beliefs, dict):
            belief_fragments = []
            for token in ("A", "B", "C"):
                b = beliefs.get(token, {})
                if not isinstance(b, dict):
                    continue
                belief_fragments.append(
                    f"{token}(cred={float(b.get('credibility', 0.5)):.2f},fair={float(b.get('fairness', 0.5)):.2f},rel={float(b.get('reliability', 0.5)):.2f})"
                )
            if belief_fragments:
                preface.append("Current beliefs about peers: " + " | ".join(belief_fragments))
        if last_round:
            proposals = last_round.get("agent_proposals", {})
            own = proposals.get(agent_name, {}) if isinstance(proposals, dict) else {}
            peers = []
            observer_token = str(agent_name).strip().upper()
            if isinstance(proposals, dict):
                for name, payload in proposals.items():
                    if name == agent_name or not isinstance(payload, dict):
                        continue
                    subject_token = str(name).strip().upper()
                    try:
                        noise = self._resource_perception_noise(
                            run_seed=run_seed,
                            step=step,
                            observer_token=observer_token,
                            subject_token=subject_token,
                            amplitude=float(round_signal.get("communication_noise", 0.0)),
                        )
                    except Exception:
                        noise = 0.0
                    base_effort = float(payload.get("effort", 0.0))
                    base_will = float(payload.get("willingness_units", 0.0))
                    perceived_effort = max(0.0, min(10.0, base_effort * (1.0 + noise)))
                    perceived_will = max(0.0, min(100.0, base_will * (1.0 + noise)))
                    peers.append(
                        f"{name}: effort={perceived_effort:.2f}, willingness={perceived_will:.2f}, msg={payload.get('message', '')}"
                    )
            preface.append(
                "Last round summary: "
                f"output={last_round.get('total_output', 0.0)}, "
                f"stability={last_round.get('public_stability_index', 0.0)}, "
                f"conflict={last_round.get('conflict', False)}, "
                f"gap={last_round.get('target_gap', 0.0)}."
            )
            prev_signal = last_round.get("environment_signal", {})
            if isinstance(prev_signal, dict):
                preface.append(
                    "Previous environment signal: "
                    f"productivity_factor={float(prev_signal.get('productivity_factor', 1.0)):.2f}, "
                    f"demand_shock={float(prev_signal.get('demand_shock', 0.0)):+.2f}, "
                    f"trust_shock={float(prev_signal.get('trust_shock', 0.0)):.2f}."
                )
            if own:
                preface.append(
                    "Your previous proposal: "
                    f"effort={own.get('effort', 0)}, "
                    f"willingness={own.get('willingness_units', 0)}, "
                    f"allocation={own.get('allocation_proposal_units', {})}."
                )
            if peers:
                preface.append("Peer proposals last round: " + " | ".join(peers))
            preface.append(
                "Prefer reacting to peer proposals and revise numbers when your strategy changes."
            )
            preface.append("Use previous round context as signal, but keep autonomous decisions.")
        return "\n".join(preface)

    def _parse_resource_agent_proposal(
        self,
        content: str,
        agent_name: str,
        step: int,
        previous_round_proposals: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        marker = "RESOURCE_AGENT_PROPOSAL "
        payload: dict[str, Any] | None = None
        for raw in content.splitlines():
            line = raw.strip()
            if not line.startswith(marker):
                continue
            encoded = line[len(marker):].strip()
            try:
                parsed = json.loads(encoded)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                payload = parsed
                break
        if payload is None:
            raise ValueError(
                f"{agent_name} missing RESOURCE_AGENT_PROPOSAL marker at round {step}"
            )

        if "effort" not in payload:
            raise ValueError(f"{agent_name} missing effort in proposal")
        if "willingness_units" not in payload:
            raise ValueError(f"{agent_name} missing willingness_units in proposal")
        if "allocation_proposal_units" not in payload:
            raise ValueError(f"{agent_name} missing allocation_proposal_units in proposal")
        if "vote_mechanism" not in payload:
            raise ValueError(f"{agent_name} missing vote_mechanism in proposal")
        if "message" not in payload:
            raise ValueError(f"{agent_name} missing message in proposal")
        try:
            effort_value = float(payload["effort"])
        except Exception as exc:
            raise ValueError(f"{agent_name} invalid effort type/value") from exc
        effort_value = max(0.0, min(10.0, effort_value))
        if not (0.0 <= effort_value <= 10.0):
            raise ValueError(f"{agent_name} effort out of range")

        try:
            willingness_value = float(payload["willingness_units"])
        except Exception as exc:
            raise ValueError(f"{agent_name} invalid willingness_units type/value") from exc
        willingness_value = max(0.0, min(100.0, willingness_value))

        vote = str(payload["vote_mechanism"]).strip().lower()
        if vote not in {"equal", "contribution"}:
            raise ValueError(f"{agent_name} vote_mechanism must be equal|contribution")

        proposal = payload["allocation_proposal_units"]
        if not isinstance(proposal, dict):
            raise ValueError(f"{agent_name} allocation_proposal_units must be object")
        raw_alloc: dict[str, float] = {}
        for key in ("A", "B", "C"):
            if key not in proposal:
                raise ValueError(f"{agent_name} allocation missing key {key}")
            try:
                raw_alloc[key] = max(0.0, float(proposal[key]))
            except Exception as exc:
                raise ValueError(f"{agent_name} allocation value for {key} invalid") from exc
        total_alloc = sum(raw_alloc.values())
        if total_alloc <= 1e-8:
            raise ValueError(f"{agent_name} allocation sum must be > 0")
        allocation = {key: value / total_alloc * 100.0 for key, value in raw_alloc.items()}

        message_text = str(payload["message"]).strip()
        if not message_text:
            raise ValueError(f"{agent_name} proposal message is empty")
        influenced_by = str(payload.get("influenced_by", "NONE")).strip().upper()
        if influenced_by not in {"A", "B", "C", "NONE"}:
            influenced_by = "NONE"
        if influenced_by == agent_name:
            influenced_by = "NONE"
        adjustment_reason = str(payload.get("adjustment_reason", "")).strip()
        if not adjustment_reason:
            adjustment_reason = "Autonomous adjustment based on current round context."

        previous_effort = effort_value
        previous_willingness = willingness_value
        if step > 1 and previous_round_proposals:
            own_prev = previous_round_proposals.get(agent_name)
            if own_prev:
                previous_effort = max(0.0, min(10.0, float(own_prev.get("effort", effort_value))))
                previous_willingness = max(0.0, min(100.0, float(own_prev.get("willingness_units", willingness_value))))
        if "previous_effort" in payload:
            try:
                previous_effort = max(0.0, min(10.0, float(payload["previous_effort"])))
            except Exception:
                pass
        if "previous_willingness_units" in payload:
            try:
                previous_willingness = max(0.0, min(100.0, float(payload["previous_willingness_units"])))
            except Exception:
                pass
        effort_delta = effort_value - previous_effort
        willingness_delta = willingness_value - previous_willingness
        if "effort_delta" in payload:
            try:
                effort_delta = float(payload["effort_delta"])
            except Exception:
                pass
        if "willingness_delta" in payload:
            try:
                willingness_delta = float(payload["willingness_delta"])
            except Exception:
                pass

        peer_reference_raw = payload.get("peer_reference", {})
        if not isinstance(peer_reference_raw, dict):
            peer_reference_raw = {}
        peer_ref_agent = str(peer_reference_raw.get("agent", influenced_by)).strip().upper()
        if peer_ref_agent not in {"A", "B", "C", "NONE"}:
            peer_ref_agent = "NONE"
        quoted_snippet = str(peer_reference_raw.get("quoted_message_snippet", "")).strip()
        if not quoted_snippet:
            quoted_snippet = "No explicit peer quote."
        try:
            referenced_effort = float(peer_reference_raw.get("referenced_effort", 0.0))
            referenced_willingness = float(peer_reference_raw.get("referenced_willingness_units", 0.0))
        except Exception:
            referenced_effort = 0.0
            referenced_willingness = 0.0
        self_critique_raw = payload.get("self_critique", {})
        if not isinstance(self_critique_raw, dict):
            self_critique_raw = {}
        try:
            consistency_score = max(0.0, min(1.0, float(self_critique_raw.get("consistency_score", 0.55))))
        except Exception:
            consistency_score = 0.55
        try:
            confidence = max(0.0, min(1.0, float(self_critique_raw.get("confidence", 0.55))))
        except Exception:
            confidence = 0.55
        try:
            loss_aversion_signal = max(0.0, min(1.0, float(self_critique_raw.get("loss_aversion_signal", 0.5))))
        except Exception:
            loss_aversion_signal = 0.5
        try:
            inertia_signal = max(0.0, min(1.0, float(self_critique_raw.get("inertia_signal", 0.5))))
        except Exception:
            inertia_signal = 0.5
        try:
            short_term_bias = max(0.0, min(1.0, float(self_critique_raw.get("short_term_bias", 0.5))))
        except Exception:
            short_term_bias = 0.5
        alignment_check = str(self_critique_raw.get("alignment_check", "")).strip()
        if not alignment_check:
            alignment_check = "Self-check completed with bounded-rational strategy."
        revised_after_check = bool(self_critique_raw.get("revised_after_check", False))
        return {
            "agent": agent_name,
            "effort": round(effort_value, 4),
            "willingness_units": round(willingness_value, 4),
            "allocation_proposal_units": {k: round(v, 4) for k, v in allocation.items()},
            "vote_mechanism": vote,
            "message": message_text,
            "influenced_by": influenced_by,
            "adjustment_reason": adjustment_reason,
            "previous_effort": round(previous_effort, 4),
            "effort_delta": round(effort_delta, 4),
            "previous_willingness_units": round(previous_willingness, 4),
            "willingness_delta": round(willingness_delta, 4),
            "peer_reference": {
                "agent": peer_ref_agent,
                "quoted_message_snippet": quoted_snippet,
                "referenced_effort": round(referenced_effort, 4),
                "referenced_willingness_units": round(referenced_willingness, 4),
            },
            "self_critique": {
                "consistency_score": round(consistency_score, 4),
                "confidence": round(confidence, 4),
                "loss_aversion_signal": round(loss_aversion_signal, 4),
                "inertia_signal": round(inertia_signal, 4),
                "short_term_bias": round(short_term_bias, 4),
                "alignment_check": alignment_check,
                "revised_after_check": revised_after_check,
            },
            "source": "marker",
        }

    def _decorate_resource_ai_decisions(
        self,
        simulation_id: int,
        step: int,
        max_steps: int,
        decisions: dict[int, tuple[str, AgentDecision]],
        role_map: dict[str, Agent],
        run_config: dict[str, Any],
    ) -> tuple[dict[int, tuple[str, AgentDecision]], bool]:
        by_name: dict[str, tuple[int, AgentDecision]] = {}
        for agent_id, value in decisions.items():
            by_name[value[0]] = (agent_id, value[1])

        role_tokens = ("A", "B", "C")
        run_seed = self._resource_seed(simulation_id, run_config)
        token_by_agent_name = {role_map[token].name: token for token in role_tokens}
        history = self._resource_round_reports_cache.get(simulation_id, [])
        previous_round_proposals: dict[str, dict[str, Any]] | None = None
        if history:
            previous_round = history[-1]
            candidate = previous_round.get("agent_proposals")
            if isinstance(candidate, dict):
                previous_round_proposals = {
                    str(k): v
                    for k, v in candidate.items()
                    if isinstance(v, dict)
                }

        social_state = self._resource_social_state(simulation_id)
        forced_strike_tokens = set(self._resource_strike_next_round.get(simulation_id, set()))
        alliance_targets: dict[str, str] = {}
        for token in role_tokens:
            peer_candidates = [peer for peer in role_tokens if peer != token]
            beliefs = social_state[token].get("beliefs", {})
            best_peer = "NONE"
            best_score = -1.0
            for peer in peer_candidates:
                trust_score = float(social_state[token]["trust"].get(peer, 0.5))
                peer_rep = float(social_state[peer].get("reputation", 0.6))
                belief = beliefs.get(peer, {}) if isinstance(beliefs, dict) else {}
                credibility = float(belief.get("credibility", trust_score))
                reliability = float(belief.get("reliability", trust_score))
                fairness = float(belief.get("fairness", 0.5))
                score = 0.35 * trust_score + 0.25 * credibility + 0.20 * reliability + 0.20 * peer_rep - 0.08 * abs(0.5 - fairness)
                if score > best_score:
                    best_score = score
                    best_peer = peer
            alliance_targets[token] = best_peer if best_score >= 0.42 else "NONE"
            social_state[token]["alliance_target"] = alliance_targets[token]
        proposals: dict[str, dict[str, Any]] = {}
        for token in role_tokens:
            agent = role_map[token]
            current = by_name.get(agent.name)
            decision = current[1] if current else AgentDecision(event_type="message", content="")
            parsed = self._parse_resource_agent_proposal(
                decision.content,
                token,
                step,
                previous_round_proposals=previous_round_proposals,
            )
            influenced = str(parsed.get("influenced_by", "NONE")).upper()
            if step > 1 and influenced == "NONE":
                inferred = "NONE"
                target_name = str(getattr(decision, "target_agent", "") or "").strip()
                if target_name and target_name in token_by_agent_name:
                    candidate = token_by_agent_name[target_name]
                    if candidate != token:
                        inferred = candidate
                if inferred == "NONE" and previous_round_proposals:
                    peer_candidates = [peer for peer in role_tokens if peer != token]
                    if peer_candidates:
                        inferred = max(
                            peer_candidates,
                            key=lambda peer: float(
                                previous_round_proposals.get(peer, {}).get("willingness_units", 0.0)
                            ),
                        )
                parsed["influenced_by"] = inferred
                if inferred != "NONE":
                    reason = str(parsed.get("adjustment_reason", "")).strip()
                    if "inferred peer influence" not in reason.lower():
                        parsed["adjustment_reason"] = (
                            f"{reason} Inferred peer influence from negotiation context ({inferred})."
                        ).strip()
            effort_cap = float(social_state[token].get("effort_cap", 1.0))
            parsed["effort"] = min(float(parsed["effort"]), 10.0 * effort_cap)
            if token in forced_strike_tokens:
                parsed["effort"] = 0.0
                parsed["willingness_units"] = min(float(parsed["willingness_units"]), 25.0)
                parsed["adjustment_reason"] = (
                    f"{parsed.get('adjustment_reason', '').strip()} Forced strike due to prior dissatisfaction."
                ).strip()
                message_text = str(parsed.get("message", "")).strip()
                if "strike" not in message_text.lower():
                    parsed["message"] = f"{message_text} I enter a strike mode this round.".strip()
            proposals[token] = parsed

        pressure_regime = bool(run_config.get("pressure_regime", False))
        # In pressure mode, force observable negotiation adaptation if an agent barely changed.
        if pressure_regime and step > 1:
            for token in role_tokens:
                proposal = proposals[token]
                own_prev = (previous_round_proposals or {}).get(token, {})
                prev_effort = max(0.0, min(10.0, float(own_prev.get("effort", proposal["effort"]))))
                prev_willingness = max(
                    0.0,
                    min(100.0, float(own_prev.get("willingness_units", proposal["willingness_units"]))),
                )
                delta_effort = float(proposal["effort"]) - prev_effort
                delta_willingness = float(proposal["willingness_units"]) - prev_willingness
                if abs(delta_effort) >= 0.20 or abs(delta_willingness) >= 2.0:
                    continue

                influenced = str(proposal.get("influenced_by", "NONE")).upper()
                if influenced not in role_tokens or influenced == token:
                    trust_map = social_state[token].get("trust", {})
                    peer_candidates = [peer for peer in role_tokens if peer != token]
                    influenced = min(
                        peer_candidates,
                        key=lambda peer: float(trust_map.get(peer, 0.5)),
                    )
                    proposal["influenced_by"] = influenced

                peer_prev = (previous_round_proposals or {}).get(influenced, {})
                if not isinstance(peer_prev, dict):
                    peer_prev = {}
                peer_ref = proposal.get("peer_reference", {})
                if not isinstance(peer_ref, dict):
                    peer_ref = {}
                peer_ref["agent"] = influenced
                if not str(peer_ref.get("quoted_message_snippet", "")).strip():
                    peer_ref["quoted_message_snippet"] = str(
                        peer_prev.get("message", f"{influenced} previous proposal reference.")
                    )[:180]
                try:
                    peer_ref["referenced_effort"] = float(peer_prev.get("effort", 0.0))
                except Exception:
                    peer_ref["referenced_effort"] = 0.0
                try:
                    peer_ref["referenced_willingness_units"] = float(
                        peer_prev.get("willingness_units", 0.0)
                    )
                except Exception:
                    peer_ref["referenced_willingness_units"] = 0.0
                proposal["peer_reference"] = peer_ref

                trust_to_influencer = float(social_state[token].get("trust", {}).get(influenced, 0.5))
                direction = -1.0 if trust_to_influencer < 0.52 else 1.0
                drift_seed = (
                    (run_seed * 17)
                    + (step * 11)
                    + (ord(token[0]) * 7)
                    + (ord(influenced[0]) * 3)
                ) % 100
                drift = drift_seed / 100.0
                effort_shift = direction * (0.22 + 0.26 * drift)
                willingness_shift = direction * (2.4 + 3.2 * drift)
                proposal["effort"] = max(0.0, min(10.0, float(proposal["effort"]) + effort_shift))
                proposal["willingness_units"] = max(
                    0.0,
                    min(100.0, float(proposal["willingness_units"]) + willingness_shift),
                )
                proposal["previous_effort"] = round(prev_effort, 4)
                proposal["previous_willingness_units"] = round(prev_willingness, 4)
                proposal["effort_delta"] = round(float(proposal["effort"]) - prev_effort, 4)
                proposal["willingness_delta"] = round(
                    float(proposal["willingness_units"]) - prev_willingness,
                    4,
                )
                reason_text = str(proposal.get("adjustment_reason", "")).strip()
                proposal["adjustment_reason"] = (
                    f"{reason_text} Pressure-mode revision after peer challenge from {influenced}."
                ).strip()
                critique = proposal.get("self_critique", {})
                if not isinstance(critique, dict):
                    critique = {}
                critique["revised_after_check"] = True
                critique["consistency_score"] = max(
                    0.58,
                    min(1.0, float(critique.get("consistency_score", 0.60))),
                )
                proposal["self_critique"] = critique

        multipliers = {"A": 1.5, "B": 1.0, "C": 1.0}
        transparency_mode = str(run_config.get("transparency", "full")).lower()
        punishment_enabled = bool(run_config.get("punishment", True))
        hidden_mode = transparency_mode == "hidden"
        target_resource = float(run_config["target_resource"])
        round_signal = self._resource_round_signal(simulation_id, step, run_config)
        productivity_factor = float(round_signal["productivity_factor"])
        trust_shock = float(round_signal["trust_shock"])
        enforcement_pressure = float(round_signal["enforcement_pressure"])
        urgency = float(round_signal["urgency"])
        communication_noise = float(round_signal.get("communication_noise", 0.0))

        previous_net_payoff: dict[str, float] = {}
        previous_efforts: dict[str, float] = {}
        previous_breaches: set[str] = set()
        if history:
            prev_round = history[-1]
            prev_net = prev_round.get("net_payoff_units", {})
            if isinstance(prev_net, dict):
                for token in role_tokens:
                    try:
                        previous_net_payoff[token] = float(prev_net.get(token, 0.0))
                    except Exception:
                        previous_net_payoff[token] = 0.0
            prev_effort_map = prev_round.get("efforts", {})
            if isinstance(prev_effort_map, dict):
                for token in role_tokens:
                    try:
                        previous_efforts[token] = float(prev_effort_map.get(token, 0.0))
                    except Exception:
                        previous_efforts[token] = 0.0
            breaches = prev_round.get("breach_agents", [])
            if isinstance(breaches, list):
                previous_breaches = {str(item).upper() for item in breaches}

        effective_efforts: dict[str, float] = {}
        contributions: dict[str, float] = {}
        for token in role_tokens:
            mood = float(social_state[token].get("mood", 0.0))
            stress = float(social_state[token].get("stress", 0.0))
            effort = max(0.0, float(proposals[token]["effort"]))
            risk_aversion = float(social_state[token]["profile"].get("risk_aversion", 0.5))
            bounded = social_state[token]["profile"].get("bounded_rationality", {})
            inertia = float(bounded.get("inertia", 0.5))
            loss_aversion = float(bounded.get("loss_aversion", 0.5))
            myopia = float(bounded.get("myopia", 0.5))
            prior_effort = float(previous_efforts.get(token, effort))
            effort = inertia * prior_effort + (1.0 - inertia) * effort
            prior_net = float(previous_net_payoff.get(token, 0.0))
            if prior_net < 0:
                effort -= min(2.0, abs(prior_net) * (0.10 + 0.35 * loss_aversion))
            if prior_net > 0:
                effort += min(1.2, prior_net * (0.06 + 0.12 * myopia))
            if target_resource > 0:
                projected_gap = (target_resource - sum(float(proposals[t]["willingness_units"]) for t in role_tokens)) / target_resource
                if projected_gap > 0.15:
                    effort += max(0.0, projected_gap) * (0.40 + 0.35 * myopia)
            effort = max(0.0, min(10.0, effort))
            mood_factor = max(0.55, 0.92 + 0.16 * mood)
            stress_factor = max(0.42, 1.0 - 0.30 * stress)
            if hidden_mode:
                mood_factor -= 0.03 * risk_aversion
            if token in previous_breaches:
                mood_factor -= 0.04
            market_factor = max(0.70, min(1.30, productivity_factor + 0.04 * (0.5 - risk_aversion)))
            effective_efforts[token] = effort
            contributions[token] = effort * multipliers[token] * mood_factor * stress_factor * market_factor

        gross_output = sum(contributions.values())
        base_governance_cost_rate = 0.012 + 0.06 * enforcement_pressure
        if punishment_enabled:
            base_governance_cost_rate += 0.02
        if hidden_mode:
            base_governance_cost_rate += 0.01
        governance_cost_units = max(0.0, gross_output * base_governance_cost_rate)
        total_output = max(0.0, gross_output - governance_cost_units)
        total_contribution = sum(contributions.values())

        selected_mechanism = str(run_config["mechanism"])
        votes = {token: str(proposals[token]["vote_mechanism"]) for token in role_tokens}
        if selected_mechanism == "majority_vote":
            contribution_votes = sum(1 for token in role_tokens if votes[token] == "contribution")
            selected_mechanism = "contribution" if contribution_votes >= 2 else "equal"

        if selected_mechanism == "equal":
            allocation_plan_units = {"A": 100.0 / 3.0, "B": 100.0 / 3.0, "C": 100.0 / 3.0}
        elif selected_mechanism == "contribution":
            if total_contribution <= 1e-8:
                allocation_plan_units = {"A": 100.0 / 3.0, "B": 100.0 / 3.0, "C": 100.0 / 3.0}
            else:
                allocation_plan_units = {
                    token: contributions[token] / total_contribution * 100.0
                    for token in role_tokens
                }
        else:
            agenda_token = role_tokens[(step - 1) % len(role_tokens)]
            allocation_plan_units = dict(proposals[agenda_token]["allocation_proposal_units"])

        commitment_units = {token: float(proposals[token]["willingness_units"]) for token in role_tokens}
        target_gap = target_resource - sum(commitment_units.values())

        payoffs_output_units = {
            token: allocation_plan_units[token] / 100.0 * total_output
            for token in role_tokens
        }
        effort_costs = {
            token: (0.09 + 0.04 * float(social_state[token]["profile"].get("risk_aversion", 0.5)))
            * (effective_efforts[token] ** 2)
            for token in role_tokens
        }
        net_payoffs = {token: payoffs_output_units[token] - effort_costs[token] for token in role_tokens}
        total_payoff = sum(payoffs_output_units.values()) or 1.0

        satisfactions: dict[str, float] = {}
        fairness_gaps: dict[str, float] = {}
        trust_values: list[float] = []
        for token in role_tokens:
            payoff_share = payoffs_output_units[token] / total_payoff
            contribution_share = 0.0 if total_contribution <= 1e-8 else contributions[token] / total_contribution
            perceived_contribution_share = contribution_share
            opacity_penalty = 0.0
            if hidden_mode:
                perceived_contribution_share = 0.55 * (1.0 / 3.0) + 0.45 * contribution_share
                opacity_penalty = 0.05 + 0.10 * abs(payoff_share - 1.0 / 3.0)
            fairness_gap = abs(payoff_share - perceived_contribution_share)
            fairness_gaps[token] = fairness_gap
            profile = social_state[token]["profile"]
            utility_norm = net_payoffs[token] / max(1.0, total_output / 3.0)
            fairness_weight = float(profile["fairness_weight"])
            self_interest_weight = float(profile["self_interest_weight"])
            risk_aversion = float(profile.get("risk_aversion", 0.5))
            bounded = profile.get("bounded_rationality", {})
            inertia = float(bounded.get("inertia", 0.5))
            loss_aversion = float(bounded.get("loss_aversion", 0.5))
            myopia = float(bounded.get("myopia", 0.5))
            effort_burden = effective_efforts[token] / 10.0
            coordination_pressure = max(0.0, target_gap / max(1.0, target_resource)) * urgency
            alliance_peer = alliance_targets.get(token, "NONE")
            alliance_bonus = 0.0
            if alliance_peer in role_tokens and alliance_peer != token:
                if str(proposals[token]["influenced_by"]).upper() == alliance_peer:
                    alliance_bonus += 0.03
                else:
                    alliance_bonus -= 0.02 * (0.2 + myopia)
            governance_burden = governance_cost_units / max(1.0, gross_output)
            sat = (
                0.54
                + 0.22 * utility_norm
                - fairness_gap * (0.16 + 0.48 * fairness_weight)
                + (payoff_share - 1.0 / 3.0) * (0.25 * self_interest_weight)
                - effort_burden * (0.06 + 0.08 * risk_aversion)
                - governance_burden * (0.05 + 0.08 * loss_aversion)
                - float(social_state[token]["stress"]) * 0.10
                - coordination_pressure * (0.04 + 0.05 * risk_aversion)
                - communication_noise * (0.05 + 0.05 * inertia)
                - opacity_penalty
                + alliance_bonus
            )
            satisfactions[token] = max(0.0, min(1.0, sat))
            trust_values.extend(
                float(v)
                for k, v in social_state[token]["trust"].items()
                if k in role_tokens and k != token
            )

        sat_values = list(satisfactions.values())
        sat_mean = sum(sat_values) / len(sat_values)
        sat_dispersion = max(sat_values) - min(sat_values)
        fairness_dispersion = max(fairness_gaps.values()) - min(fairness_gaps.values())
        low_sat_agents = sum(1 for value in sat_values if value < 0.38)
        trust_mean = sum(trust_values) / len(trust_values) if trust_values else 0.5
        target_pressure = max(0.0, target_gap / max(1.0, target_resource))
        trust_floor = (0.42 if hidden_mode else 0.28) + 0.30 * trust_shock
        revision_ratio = (
            sum(
                1
                for token in role_tokens
                if abs(float(proposals[token]["effort_delta"])) > 0.15
                or abs(float(proposals[token]["willingness_delta"])) > 1.5
            )
            / len(role_tokens)
        )
        influence_ratio = (
            sum(
                1
                for token in role_tokens
                if str(proposals[token]["influenced_by"]).upper() in role_tokens
                and str(proposals[token]["influenced_by"]).upper() != token
            )
            / len(role_tokens)
        )
        conflict = (
            (low_sat_agents >= 2 and (sat_mean < 0.42 or fairness_dispersion > 0.14))
            or (low_sat_agents >= 1 and sat_mean < 0.32)
            or trust_mean < trust_floor
            or (target_pressure > 0.28 and sat_mean < 0.46)
            or (trust_shock > 0.06 and sat_mean < 0.50)
            or (
                pressure_regime
                and step > 1
                and (
                    fairness_dispersion > 0.06
                    or trust_mean < 0.62
                    or revision_ratio < 0.67
                    or influence_ratio < 0.67
                    or communication_noise > 0.09
                )
            )
        )

        free_rider_agents: list[str] = []
        for token in role_tokens:
            contrib_share = 0.0 if total_contribution <= 1e-8 else contributions[token] / total_contribution
            payoff_share = payoffs_output_units[token] / total_payoff
            min_gap = 0.08 if hidden_mode else 0.10
            if contrib_share <= 0.18 and payoff_share - contrib_share >= min_gap:
                free_rider_agents.append(token)

        institution_cost_units = governance_cost_units
        if conflict:
            institution_cost_units += gross_output * (0.025 + 0.03 * float(round_signal.get("stability_shock", 0.0)))
        if punishment_enabled:
            institution_cost_units += gross_output * (0.01 + 0.04 * enforcement_pressure)
        if free_rider_agents:
            institution_cost_units += gross_output * 0.01 * len(free_rider_agents)
        net_public_output = max(0.0, gross_output - institution_cost_units)
        total_output = net_public_output
        payoff_den = sum(payoffs_output_units.values()) or 1.0
        payoff_scale = total_output / payoff_den if payoff_den > 1e-8 else 1.0
        payoffs_output_units = {token: value * payoff_scale for token, value in payoffs_output_units.items()}
        total_payoff = sum(payoffs_output_units.values()) or 1.0
        net_payoffs = {token: payoffs_output_units[token] - effort_costs[token] for token in role_tokens}

        stress_values_now = [float(social_state[token].get("stress", 0.0)) for token in role_tokens]
        mean_stress_now = sum(stress_values_now) / len(stress_values_now) if stress_values_now else 0.0
        public_stability_index = max(
            0.0,
            min(
                1.0,
                0.46 * trust_mean + 0.24 * (1.0 - (1.0 if conflict else 0.0)) + 0.20 * (1.0 - mean_stress_now) + 0.10 * (1.0 - communication_noise),
            ),
        )
        social_welfare_index = max(
            0.0,
            min(
                1.0,
                0.62 * (total_output / max(1.0, target_resource / 2.0)) + 0.38 * public_stability_index,
            ),
        )

        adjusted_agents = 0
        for token in role_tokens:
            influenced = str(proposals[token]["influenced_by"]).upper()
            effort_delta = abs(float(proposals[token]["effort_delta"]))
            willing_delta = abs(float(proposals[token]["willingness_delta"]))
            if effort_delta > 0.20 or willing_delta > 1.50 or (influenced in role_tokens and influenced != token):
                adjusted_agents += 1
        behavioral_adjustment_ratio = adjusted_agents / len(role_tokens)

        breach_agents: list[str] = []
        compliance_ratio: dict[str, float] = {}
        for token in role_tokens:
            expected_effort = max(0.5, float(proposals[token]["willingness_units"]) / 10.0)
            actual_effort = max(0.0, float(effective_efforts[token]))
            ratio = max(0.0, min(2.0, actual_effort / expected_effort))
            compliance_ratio[token] = ratio
            if float(proposals[token]["willingness_units"]) >= 35.0 and ratio < 0.55:
                breach_agents.append(token)

        causal_consistency: dict[str, dict[str, Any]] = {}
        causal_consistency_pass = 0
        for token in role_tokens:
            influenced = str(proposals[token]["influenced_by"]).upper()
            peer_ref = proposals[token].get("peer_reference", {})
            peer_ref_agent = str(peer_ref.get("agent", "NONE")).upper() if isinstance(peer_ref, dict) else "NONE"
            effort_delta_abs = abs(float(proposals[token]["effort_delta"]))
            willingness_delta_abs = abs(float(proposals[token]["willingness_delta"]))
            delta_signal = effort_delta_abs + willingness_delta_abs / 10.0
            critique = proposals[token].get("self_critique", {})
            consistency_score = float(critique.get("consistency_score", 0.55)) if isinstance(critique, dict) else 0.55
            confidence = float(critique.get("confidence", 0.55)) if isinstance(critique, dict) else 0.55
            revised_after_check = bool(critique.get("revised_after_check", False)) if isinstance(critique, dict) else False
            alignment_check = (
                str(critique.get("alignment_check", "")).strip() if isinstance(critique, dict) else ""
            )
            reference_match = influenced in role_tokens and peer_ref_agent == influenced
            kept_after_consideration = (
                (not revised_after_check)
                and bool(alignment_check)
                and confidence >= 0.45
                and consistency_score >= 0.50
            )
            if step <= 1:
                # Round 1 has no prior peer proposal; only validate internal self-critique coherence.
                passed = bool(consistency_score >= 0.45 and confidence >= 0.40)
                reason = "initial_round_self_consistency"
            elif influenced in role_tokens:
                if reference_match and delta_signal >= 0.10 and consistency_score >= 0.50:
                    passed = True
                    reason = "peer_reference_and_numeric_adjustment"
                elif reference_match and delta_signal <= 0.35 and kept_after_consideration:
                    # Agents can rationally keep numbers unchanged after evaluating peer signals.
                    passed = True
                    reason = "peer_reference_and_reasoned_stability"
                elif (not reference_match) and delta_signal >= 0.18 and confidence >= 0.55:
                    passed = True
                    reason = "unmatched_reference_but_clear_adjustment"
                else:
                    passed = False
                    reason = "influenced_without_supported_change"
            else:
                if delta_signal <= 0.45 and consistency_score >= 0.45:
                    passed = True
                    reason = "autonomous_consistent"
                elif kept_after_consideration:
                    passed = True
                    reason = "autonomous_reasoned_stability"
                else:
                    passed = False
                    reason = "autonomous_inconsistent"
            if passed:
                causal_consistency_pass += 1
            causal_consistency[token] = {
                "passed": passed,
                "delta_signal": round(delta_signal, 4),
                "influenced_by": influenced,
                "reference_match": reference_match,
                "reason": reason,
                "self_consistency_score": round(consistency_score, 4),
            }
        causal_consistency_score = causal_consistency_pass / len(role_tokens)

        cooperation_alive = (
            all(effective_efforts[token] >= 0.8 for token in role_tokens)
            and not (conflict and sat_mean < 0.40)
            and total_output > 0
        )
        previous_cumulative = 0.0
        if history:
            previous_cumulative = float(history[-1].get("cumulative_output", 0.0))
        cumulative_output = previous_cumulative + total_output
        target_achieved = cumulative_output >= target_resource

        updated_social_state: dict[str, dict[str, Any]] = {}
        for token in role_tokens:
            prev_state = social_state[token]
            mood = float(prev_state["mood"])
            stress = float(prev_state["stress"])
            trust_map = dict(prev_state["trust"])
            belief_map = dict(prev_state.get("beliefs", {}))
            profile = prev_state["profile"]
            fairness_weight = float(profile["fairness_weight"])
            self_interest_weight = float(profile["self_interest_weight"])
            risk_aversion = float(profile.get("risk_aversion", 0.5))
            bounded = profile.get("bounded_rationality", {})
            inertia = float(bounded.get("inertia", 0.5))
            loss_aversion = float(bounded.get("loss_aversion", 0.5))
            myopia = float(bounded.get("myopia", 0.5))
            reputation = float(prev_state.get("reputation", 0.6))
            dissatisfaction = 1.0 - satisfactions[token]
            effort_burden = effective_efforts[token] / 10.0

            mood += (satisfactions[token] - 0.5) * 0.22
            mood += 0.05 if net_payoffs[token] > 0 else -0.07
            mood -= fairness_gaps[token] * (0.08 + 0.12 * fairness_weight)
            mood += (reputation - 0.6) * 0.03
            mood -= effort_burden * 0.04
            if hidden_mode:
                mood -= 0.04 * risk_aversion
            mood -= 0.03 * myopia
            if token in free_rider_agents:
                mood += 0.02 * self_interest_weight
            if token in breach_agents:
                mood -= 0.05 + 0.05 * loss_aversion
            ally = alliance_targets.get(token, "NONE")
            if ally in role_tokens and ally != token:
                if ally in breach_agents:
                    mood -= 0.04
                elif str(proposals[token]["influenced_by"]).upper() == ally:
                    mood += 0.03
            if conflict:
                mood -= 0.06

            stress += 0.06 * dissatisfaction
            stress += 0.03 * effort_burden
            stress += 0.03 * target_pressure
            stress += 0.04 * urgency
            if hidden_mode:
                stress += 0.03
            if conflict:
                stress += 0.05
            stress += trust_shock * (0.15 + 0.20 * risk_aversion)
            stress += enforcement_pressure * 0.06
            stress += communication_noise * (0.08 + 0.08 * inertia)
            if token in breach_agents:
                stress += 0.04 * loss_aversion
            if satisfactions[token] >= 0.62:
                stress -= 0.06
            if net_payoffs[token] > 0:
                stress -= 0.02
            if token in free_rider_agents and payoffs_output_units[token] > total_payoff / 3.0:
                stress -= 0.02
            if ally in role_tokens and ally != token and ally not in breach_agents:
                stress -= 0.02

            influenced_by = str(proposals[token]["influenced_by"]).upper()
            effort_delta = abs(float(proposals[token]["effort_delta"]))
            willing_delta = abs(float(proposals[token]["willingness_delta"]))
            for peer in role_tokens:
                if peer == token:
                    continue
                base = float(trust_map.get(peer, 0.5))
                trust_map[peer] = base + (0.5 - base) * 0.05
                if hidden_mode:
                    trust_map[peer] -= 0.02
                trust_map[peer] -= trust_shock * 0.35
                if peer in breach_agents:
                    trust_map[peer] -= 0.05
            if influenced_by in role_tokens and influenced_by != token:
                reaction_strength = max(effort_delta / 4.0, willing_delta / 20.0)
                if reaction_strength >= 0.12:
                    trust_map[influenced_by] = float(trust_map.get(influenced_by, 0.5)) + 0.06
                else:
                    trust_map[influenced_by] = float(trust_map.get(influenced_by, 0.5)) - 0.03
            if ally in role_tokens and ally != token:
                trust_map[ally] = float(trust_map.get(ally, 0.5)) + (0.04 if ally not in breach_agents else -0.03)

            beneficiary = max(allocation_plan_units.items(), key=lambda item: item[1])[0]
            if fairness_gaps[token] > 0.16 and beneficiary != token:
                trust_map[beneficiary] = float(trust_map.get(beneficiary, 0.5)) - 0.07
            if conflict:
                for peer in role_tokens:
                    if peer == token:
                        continue
                    trust_map[peer] = float(trust_map.get(peer, 0.5)) - 0.015
            if token not in free_rider_agents and free_rider_agents:
                for free_rider in free_rider_agents:
                    if free_rider != token:
                        trust_map[free_rider] = float(trust_map.get(free_rider, 0.5)) - 0.03

            mood = max(-1.0, min(1.0, mood))
            stress_floor = 0.10 if hidden_mode else 0.04
            stress = max(stress_floor, min(1.0, stress))
            for peer in role_tokens:
                if peer == token:
                    trust_map[peer] = 1.0
                else:
                    trust_map[peer] = max(0.0, min(1.0, float(trust_map.get(peer, 0.5))))

            for peer in role_tokens:
                prev_belief = belief_map.get(peer, {})
                if not isinstance(prev_belief, dict):
                    prev_belief = {}
                cred = float(prev_belief.get("credibility", 0.5))
                fairness = float(prev_belief.get("fairness", 0.5))
                reliability = float(prev_belief.get("reliability", 0.5))
                if peer == token:
                    belief_map[peer] = {"credibility": 1.0, "fairness": 1.0, "reliability": 1.0}
                    continue
                trust_peer = float(trust_map.get(peer, 0.5))
                compliance_peer = float(compliance_ratio.get(peer, 0.8))
                alloc_to_token = float(proposals[peer]["allocation_proposal_units"].get(token, 33.0))
                fairness_signal = max(0.0, min(1.0, alloc_to_token / 45.0))
                if peer in breach_agents:
                    reliability -= 0.10
                    cred -= 0.08
                else:
                    reliability += 0.06 * (compliance_peer - 0.6)
                cred = cred + 0.25 * (trust_peer - cred)
                fairness = fairness + 0.22 * (fairness_signal - fairness)
                belief_map[peer] = {
                    "credibility": max(0.0, min(1.0, cred)),
                    "fairness": max(0.0, min(1.0, fairness)),
                    "reliability": max(0.0, min(1.0, reliability)),
                }

            updated_reputation = reputation + 0.18 * (compliance_ratio[token] - reputation)
            if token in breach_agents:
                updated_reputation -= 0.10
            if net_payoffs[token] > 0:
                updated_reputation += 0.03
            updated_reputation = max(0.0, min(1.0, updated_reputation))
            utility_norm = net_payoffs[token] / max(1.0, total_output / 3.0)

            effort_cap = self._resource_effort_cap(
                mood,
                stress,
                punishment_enabled,
                dissatisfaction=max(0.0, min(1.0, dissatisfaction)),
                conflict=conflict,
            )
            updated_social_state[token] = {
                "mood": mood,
                "stress": stress,
                "effort_cap": effort_cap,
                "trust": trust_map,
                "beliefs": belief_map,
                "reputation": updated_reputation,
                "alliance_target": alliance_targets.get(token, "NONE"),
                "last_utility_norm": round(utility_norm, 4),
                "profile": prev_state["profile"],
            }
        self._resource_social_state_cache[simulation_id] = updated_social_state

        round_payload = {
            "round": step,
            "target_resource": target_resource,
            "selected_mechanism": selected_mechanism,
            "transparency": str(run_config["transparency"]),
            "punishment": bool(run_config["punishment"]),
            "environment_signal": dict(round_signal),
            "gross_output": round(gross_output, 4),
            "institution_cost_units": round(institution_cost_units, 4),
            "public_stability_index": round(public_stability_index, 4),
            "social_welfare_index": round(social_welfare_index, 4),
            "efforts": {k: round(float(proposals[k]["effort"]), 3) for k in role_tokens},
            "effective_efforts": {k: round(effective_efforts[k], 3) for k in role_tokens},
            "weighted_contributions": {k: round(contributions[k], 3) for k in role_tokens},
            "commitment_units": {k: round(commitment_units[k], 3) for k in role_tokens},
            "allocation_plan_units": {k: round(allocation_plan_units[k], 3) for k in role_tokens},
            "effort_cost_units": {k: round(effort_costs[k], 3) for k in role_tokens},
            "net_payoff_units": {k: round(net_payoffs[k], 3) for k in role_tokens},
            "total_output": round(total_output, 4),
            "cumulative_output": round(cumulative_output, 4),
            "target_achieved": target_achieved,
            "average_satisfaction": round(sum(satisfactions.values()) / 3.0, 4),
            "satisfaction_dispersion": round(sat_dispersion, 4),
            "fairness_dispersion": round(fairness_dispersion, 4),
            "conflict_count": 1 if conflict else 0,
            "conflict": conflict,
            "free_rider_agents": free_rider_agents,
            "free_rider_present": bool(free_rider_agents),
            "cooperation_alive": cooperation_alive,
            "trust_mean": round(trust_mean, 4),
            "low_credibility_turn": hidden_mode and (conflict or trust_mean < 0.45),
            "behavioral_adjustment_ratio": round(behavioral_adjustment_ratio, 4),
            "causal_consistency_score": round(causal_consistency_score, 4),
            "causal_consistency": causal_consistency,
            "alliance_targets": dict(alliance_targets),
            "breach_agents": sorted(breach_agents),
            "compliance_ratio": {k: round(v, 4) for k, v in compliance_ratio.items()},
            "mechanism_votes": dict(votes),
            "target_gap": round(target_gap, 4),
            "round_conclusion": (
                f"Round {step}: mechanism={selected_mechanism}, output={total_output:.2f}, "
                f"stability={public_stability_index:.2f}, welfare={social_welfare_index:.2f}, "
                f"cooperation={'alive' if cooperation_alive else 'fragile'}, "
                f"conflict={'yes' if conflict else 'no'}, trust_mean={trust_mean:.2f}, "
                f"cumulative_output={cumulative_output:.2f}, target_achieved={'yes' if target_achieved else 'no'}."
            ),
            "satisfactions": {k: round(satisfactions[k], 4) for k in role_tokens},
            "moods": {k: round(updated_social_state[k]["mood"], 4) for k in role_tokens},
            "stresses": {k: round(updated_social_state[k]["stress"], 4) for k in role_tokens},
            "effort_cap_ratio": {k: round(updated_social_state[k]["effort_cap"], 4) for k in role_tokens},
            "trust_matrix": {
                k: {kk: round(float(updated_social_state[k]["trust"][kk]), 4) for kk in role_tokens}
                for k in role_tokens
            },
            "agent_proposals": {
                k: {
                    "effort": round(float(proposals[k]["effort"]), 3),
                    "effective_effort": round(float(effective_efforts[k]), 3),
                    "willingness_units": round(float(proposals[k]["willingness_units"]), 3),
                    "allocation_proposal_units": {
                        kk: round(float(vv), 3)
                        for kk, vv in dict(proposals[k]["allocation_proposal_units"]).items()
                    },
                    "message": str(proposals[k]["message"]),
                    "influenced_by": str(proposals[k]["influenced_by"]),
                    "adjustment_reason": str(proposals[k]["adjustment_reason"]),
                    "previous_effort": round(float(proposals[k]["previous_effort"]), 3),
                    "effort_delta": round(float(proposals[k]["effort_delta"]), 3),
                    "previous_willingness_units": round(float(proposals[k]["previous_willingness_units"]), 3),
                    "willingness_delta": round(float(proposals[k]["willingness_delta"]), 3),
                    "peer_reference": dict(proposals[k]["peer_reference"]),
                    "self_critique": dict(proposals[k].get("self_critique", {})),
                    "mood": round(updated_social_state[k]["mood"], 4),
                    "stress": round(updated_social_state[k]["stress"], 4),
                    "effort_cap": round(updated_social_state[k]["effort_cap"], 4),
                    "reputation": round(float(updated_social_state[k].get("reputation", 0.6)), 4),
                    "alliance_target": str(updated_social_state[k].get("alliance_target", "NONE")),
                    "beliefs": dict(updated_social_state[k].get("beliefs", {})),
                    "thinking_style": str(updated_social_state[k]["profile"]["thinking_style"]),
                }
                for k in role_tokens
            },
        }

        strike_next: set[str] = set()
        for token in role_tokens:
            if updated_social_state[token]["effort_cap"] <= 0.70:
                strike_next.add(token)
        round_payload["strike_scheduled_next_round"] = sorted(strike_next)
        self._resource_strike_next_round[simulation_id] = strike_next

        history = self._resource_round_reports_cache.get(simulation_id, [])
        history.append(round_payload)
        self._resource_round_reports_cache[simulation_id] = history[-500:]

        resource_names = {role_map[k].name for k in role_tokens}
        token_by_name = {role_map[k].name: k for k in role_tokens}
        patched: dict[int, tuple[str, AgentDecision]] = {}
        for token in role_tokens:
            agent = role_map[token]
            current = by_name.get(agent.name)
            preferred_target = current[1].target_agent if current else None
            target_name: str | None = None
            influenced = str(proposals[token]["influenced_by"]).upper()
            alliance_peer = str(alliance_targets.get(token, "NONE")).upper()
            if influenced in role_tokens and influenced != token:
                target_name = role_map[influenced].name
            elif alliance_peer in role_tokens and alliance_peer != token and not conflict:
                target_name = role_map[alliance_peer].name
            elif isinstance(preferred_target, str):
                target_name = preferred_target

            if target_name not in resource_names or target_name == agent.name:
                candidates = [name for name in resource_names if name != agent.name]
                if candidates:
                    ranked = sorted(
                        candidates,
                        key=lambda n: float(updated_social_state[token]["trust"].get(token_by_name.get(n, ""), 0.5)),
                    )
                    if conflict:
                        target_name = ranked[0]
                    else:
                        target_name = ranked[-1]
                    # Add deterministic exploration to avoid rigid bilateral loops.
                    if len(ranked) > 1:
                        probe = ((run_seed * 13) + (step * 17) + ord(token[0]) * 7) % 100 / 100.0
                        stress = float(updated_social_state[token]["stress"])
                        explore_chance = min(0.50, 0.12 + 0.25 * stress)
                        if probe < explore_chance:
                            if conflict:
                                target_name = ranked[-1]
                            else:
                                target_name = ranked[0]
                else:
                    target_name = None

            message_body = str(proposals[token]["message"])
            if f"[Resource Round {step}]" not in message_body:
                message_body = f"[Resource Round {step}] {message_body}"
            critique_payload = dict(proposals[token].get("self_critique", {}))
            message_body = (
                f"{message_body}\n"
                f"RESOURCE_SELF_CRITIQUE {json.dumps(critique_payload, ensure_ascii=True)}"
            )
            if token == "A":
                message_body = f"RESOURCE_ROUND_REPORT {json.dumps(round_payload, ensure_ascii=True)}\n{message_body}"
            if token == "C" and (step >= max_steps or target_achieved):
                final_payload = self._build_resource_final_conclusion(
                    trace=self._resource_round_reports_cache.get(simulation_id, []),
                    target_resource=target_resource,
                )
                message_body = (
                    f"{message_body}\n"
                    f"RESOURCE_FINAL_RECOMMENDATION {json.dumps(final_payload, ensure_ascii=True)}"
                )
            patched[agent.id or 0] = (
                agent.name,
                AgentDecision(
                    event_type="message",
                    content=message_body,
                    target_agent=target_name,
                    memory_append=[
                        f"resource_round:{step}",
                        f"resource_mechanism:{selected_mechanism}",
                        f"resource_effort:{token}:{effective_efforts[token]:.3f}",
                        f"resource_willingness:{token}:{commitment_units[token]:.3f}",
                        f"resource_influenced_by:{token}:{proposals[token]['influenced_by']}",
                        f"resource_causal_pass:{token}:{int(bool(causal_consistency[token]['passed']))}",
                        f"resource_critique_score:{token}:{float(proposals[token].get('self_critique', {}).get('consistency_score', 0.0)):.3f}",
                        f"resource_mood:{token}:{updated_social_state[token]['mood']:.3f}",
                        f"resource_stress:{token}:{updated_social_state[token]['stress']:.3f}",
                        f"resource_parse_source:{token}:{proposals[token].get('source','default')}",
                    ],
                ),
            )
        return patched, target_achieved

    def _build_resource_final_conclusion(
        self,
        trace: list[dict[str, Any]],
        target_resource: float,
    ) -> dict[str, Any]:
        if not trace:
            return {
                "round_count": 0,
                "recommended_mechanism": "unknown",
                "rationale": "No trace data available.",
            }
        mechanism_counts: dict[str, int] = {}
        total_output = 0.0
        total_stability = 0.0
        total_welfare = 0.0
        conflict_rounds = 0
        free_rider_rounds = 0
        for item in trace:
            mechanism = str(item.get("selected_mechanism", "unknown"))
            mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
            total_output += float(item.get("total_output", item.get("output", 0.0)))
            total_stability += float(item.get("public_stability_index", 0.0))
            total_welfare += float(item.get("social_welfare_index", 0.0))
            if bool(item.get("conflict", False)):
                conflict_rounds += 1
            free_riders = item.get("free_rider_agents", [])
            if isinstance(free_riders, list) and free_riders:
                free_rider_rounds += 1
        dominant_mechanism = max(
            mechanism_counts.items(),
            key=lambda pair: (pair[1], pair[0]),
        )[0]
        conflict_ratio = conflict_rounds / len(trace)
        mean_stability = total_stability / len(trace)
        mean_welfare = total_welfare / len(trace)
        if dominant_mechanism == "contribution" and conflict_ratio <= 0.4:
            rationale = "Contribution-weighted allocation balanced output and conflict while preserving collective stability."
        elif dominant_mechanism == "equal" and conflict_ratio > 0.4:
            rationale = "Equal allocation dominated but conflict stayed high; transparency/punishment and reputation repair should be increased."
        else:
            rationale = "Mixed mechanism outcomes suggest tuning transparency, alliance incentives, and governance cost controls before policy freeze."
        final_round = trace[-1]
        final_alloc = dict(final_round.get("allocation_plan_units", {}))
        final_will = dict(final_round.get("commitment_units", {}))
        total_alloc = sum(float(v) for v in final_alloc.values()) or 1.0
        total_will = sum(float(v) for v in final_will.values()) or 1.0
        return {
            "round_count": len(trace),
            "total_output": round(total_output, 2),
            "mean_output": round(total_output / len(trace), 2),
            "mean_public_stability": round(mean_stability, 4),
            "mean_social_welfare": round(mean_welfare, 4),
            "conflict_rounds": conflict_rounds,
            "free_rider_rounds": free_rider_rounds,
            "recommended_mechanism": dominant_mechanism,
            "final_round_allocation_target100_units": {
                key: round((float(value) / total_alloc) * target_resource, 3)
                for key, value in final_alloc.items()
            },
            "final_willingness_target100_units": {
                key: round((float(value) / total_will) * target_resource, 3)
                for key, value in final_will.items()
            },
            "rationale": rationale,
        }

    def _normalize_decision(
        self,
        decision: AgentDecision,
        source_agent: str,
        peers: list[str],
        inbox: list[InboxMessage],
        partner_counts: dict[tuple[str, str], int],
    ) -> AgentDecision:
        if decision.event_type != "message":
            return decision
        target = decision.target_agent
        if target and target in peers and target != source_agent:
            return decision
        # Prefer replying to latest sender if possible.
        if inbox:
            latest_sender = inbox[-1].source_agent
            if latest_sender in peers and latest_sender != source_agent:
                decision.target_agent = latest_sender
                decision.memory_append.append(f"route:inbox:{latest_sender}")
                return decision
        selected = self._select_target_agent(source_agent, peers, partner_counts)
        decision.target_agent = selected
        decision.memory_append.append(f"route:auto:{selected}")
        return decision

    def _select_target_agent(
        self,
        source_agent: str,
        peers: list[str],
        partner_counts: dict[tuple[str, str], int],
    ) -> str | None:
        if not peers:
            return None
        candidates = [p for p in peers if p != source_agent]
        if not candidates:
            return None
        # Minimize total bilateral interactions to avoid long-term exclusion.
        scored: list[tuple[int, str]] = []
        for peer in candidates:
            score = partner_counts.get((source_agent, peer), 0) + partner_counts.get((peer, source_agent), 0)
            scored.append((score, peer))
        scored.sort(key=lambda x: (x[0], x[1]))
        return scored[0][1]

    def _partner_counts(self, session: Session, simulation_id: int) -> dict[tuple[str, str], int]:
        stmt = (
            select(Event)
            .where(Event.simulation_id == simulation_id)
            .where(Event.source_agent.is_not(None))
            .where(Event.target_agent.is_not(None))
        )
        rows = session.exec(stmt).all()
        counts: dict[tuple[str, str], int] = {}
        for row in rows:
            if not row.source_agent or not row.target_agent:
                continue
            key = (row.source_agent, row.target_agent)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _pick_adapter(self) -> DecisionAdapter:
        if settings.ai_mode == "mock":
            return MockDecisionAdapter()
        if settings.ai_mode == "openai":
            try:
                return OpenAIDecisionAdapter()
            except Exception:
                return RuleBasedDecisionAdapter()
        return RuleBasedDecisionAdapter()

    def _build_inbox(self, session: Session, simulation_id: int, step: int, agent_name: str) -> list[InboxMessage]:
        stmt = (
            select(Event)
            .where(Event.simulation_id == simulation_id)
            .where(Event.step == step)
            .where(Event.target_agent == agent_name)
        )
        events = session.exec(stmt).all()
        return [InboxMessage(source_agent=e.source_agent or "unknown", content=e.content) for e in events]

    def _memory_history(self, memory: dict) -> list[str]:
        if not isinstance(memory, dict):
            return []
        history = memory.get("history", [])
        return history if isinstance(history, list) else []

    def _append_memory(self, agent: Agent, entries: list[str]) -> None:
        existing = agent.memory if isinstance(agent.memory, dict) else {}
        memory: dict[str, Any] = {**existing}
        existing_history = memory.get("history", [])
        history = list(existing_history) if isinstance(existing_history, list) else []
        history.extend(entries)
        memory["history"] = history[-50:]
        agent.memory = memory


