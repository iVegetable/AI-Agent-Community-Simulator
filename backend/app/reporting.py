"""Simulation summarizer and realism audit report builder.

Generates markdown + JSON reports from persisted events, including
resource-allocation realism grading and supporting diagnostics.
"""

import json
import math
from typing import Any

from sqlmodel import Session, select

from .analysis import detect_emergent_patterns
from .models import Agent, Event, Simulation, SimulationReport
from .scoring import compute_kpi
from .utils import utc_iso_now

RISK_WORDS = ("risk", "issue", "concern", "failure", "blocker", "tradeoff")
RESOURCE_ROUND_PREFIX = "RESOURCE_ROUND_REPORT "
RESOURCE_FINAL_PREFIX = "RESOURCE_FINAL_RECOMMENDATION "


def get_latest_report(session: Session, simulation_id: int) -> SimulationReport | None:
    """Return newest stored report for a simulation, if any."""

    stmt = (
        select(SimulationReport)
        .where(SimulationReport.simulation_id == simulation_id)
        .order_by(SimulationReport.id.desc())
        .limit(1)
    )
    return session.exec(stmt).first()


def generate_simulation_report(
    session: Session,
    simulation_id: int,
    *,
    force: bool = False,
) -> tuple[SimulationReport, Event | None]:
    """Build and persist a report, optionally forcing a new version."""

    existing = get_latest_report(session, simulation_id)
    if existing and not force:
        return existing, None

    sim = session.get(Simulation, simulation_id)
    if not sim:
        raise ValueError("Simulation not found")
    agents = session.exec(select(Agent).where(Agent.simulation_id == simulation_id)).all()
    events = session.exec(select(Event).where(Event.simulation_id == simulation_id)).all()

    payload = _build_report_payload(sim=sim, agents=list(agents), events=list(events))
    version = (existing.version + 1) if existing and force else (1 if not existing else existing.version)
    report = SimulationReport(
        simulation_id=simulation_id,
        version=version,
        generator="summarizer-v1",
        title=str(payload["title"]),
        markdown=_report_payload_to_markdown(payload),
        report_json=payload,
        created_at=str(payload["generated_at"]),
    )
    session.add(report)
    event = Event(
        simulation_id=simulation_id,
        step=sim.step,
        event_type="report",
        source_agent="summarizer",
        target_agent=None,
        content=f"Conclusion report v{version} generated. Overall KPI={payload['kpi']['aggregate_overall']:.2f}",
    )
    session.add(event)
    session.commit()
    session.refresh(report)
    session.refresh(event)
    return report, event


def _build_report_payload(sim: Simulation, agents: list[Agent], events: list[Event]) -> dict[str, Any]:
    kpi = compute_kpi(events=events, agent_names=[agent.name for agent in agents])
    emergent_patterns = detect_emergent_patterns(events=events, agent_names=[agent.name for agent in agents])

    message_events = [event for event in events if event.event_type == "message"]
    intervention_events = [event for event in events if event.event_type == "intervention"]
    fallback_events = [event for event in message_events if "(fallback:" in event.content]
    risk_messages = []
    for event in message_events:
        text = event.content.lower()
        if any(word in text for word in RISK_WORDS):
            risk_messages.append(event)

    executive_summary = _executive_summary(
        sim=sim,
        message_count=len(message_events),
        intervention_count=len(intervention_events),
        fallback_count=len(fallback_events),
        kpi_overall=float(kpi["aggregate"]["overall"]),
        top_pattern=(emergent_patterns[-1]["label"] if emergent_patterns else "none"),
    )
    key_findings = _key_findings(kpi, emergent_patterns, message_events, intervention_events)
    risk_register = _risk_register(risk_messages)
    action_items = _action_items(kpi, risk_register, intervention_events)
    evidence = _key_evidence(message_events)
    resource_allocation = _resource_allocation_summary(events)
    if resource_allocation:
        key_findings.append(
            "Resource scenario observer summary: "
            f"total_output={resource_allocation['total_output']:.2f}, "
            f"conflict_rounds={resource_allocation['conflict_rounds']}, "
            f"recommended_mechanism={resource_allocation['recommended_mechanism']}."
        )
        realism = resource_allocation.get("realism", {})
        if isinstance(realism, dict) and "score" in realism:
            key_findings.append(
                "Resource realism score: "
                f"{float(realism['score']):.1f}/100 "
                f"(grade={str(realism.get('grade', 'unknown'))})."
            )
            key_findings.append(
                "Resource realism standard passed: "
                f"{bool(realism.get('standard_passed', False))}."
            )

    payload: dict[str, Any] = {
        "title": f"Simulation {sim.id} Conclusion Report",
        "generated_at": utc_iso_now(),
        "simulation": {
            "id": sim.id,
            "goal": sim.goal,
            "status": sim.status.value,
            "step": sim.step,
            "max_steps": sim.max_steps,
        },
        "kpi": {
            "aggregate_overall": float(kpi["aggregate"]["overall"]),
            "trend_delta": float(kpi["trend_delta"]),
            "dimensions": dict(kpi["aggregate"]["dimensions"]),
        },
        "executive_summary": executive_summary,
        "key_findings": key_findings,
        "risk_register": risk_register,
        "action_items": action_items,
        "evidence": evidence,
        "emergent_patterns": emergent_patterns[-5:],
    }
    if resource_allocation:
        payload["resource_allocation"] = resource_allocation
    return payload


def _executive_summary(
    sim: Simulation,
    message_count: int,
    intervention_count: int,
    fallback_count: int,
    kpi_overall: float,
    top_pattern: str,
) -> str:
    return (
        f"Simulation {sim.id} completed {sim.step}/{sim.max_steps} steps on goal '{sim.goal}'. "
        f"The run produced {message_count} messages and {intervention_count} interventions. "
        f"Overall KPI is {kpi_overall:.2f} with {fallback_count} fallback messages. "
        f"Latest dominant interaction pattern is '{top_pattern}'."
    )


def _key_findings(
    kpi: dict[str, Any],
    patterns: list[dict[str, Any]],
    message_events: list[Event],
    intervention_events: list[Event],
) -> list[str]:
    dims = kpi["aggregate"]["dimensions"]
    strongest = max(dims.items(), key=lambda item: float(item[1]))
    weakest = min(dims.items(), key=lambda item: float(item[1]))
    findings = [
        f"Strongest KPI dimension: {strongest[0]} ({float(strongest[1]):.2f}).",
        f"Weakest KPI dimension: {weakest[0]} ({float(weakest[1]):.2f}).",
        f"KPI trend delta across steps: {float(kpi['trend_delta']):+.2f}.",
        f"Interventions observed: {len(intervention_events)}; total messages: {len(message_events)}.",
    ]
    if patterns:
        findings.append(f"Detected emergent patterns: {', '.join(str(item['label']) for item in patterns[-3:])}.")
    return findings


def _risk_register(risk_messages: list[Event]) -> list[dict[str, Any]]:
    if not risk_messages:
        return [
            {
                "name": "No explicit risk signal",
                "severity": "low",
                "evidence": "No risk-related keywords found in message events.",
                "owner": "planner",
            }
        ]
    register: list[dict[str, Any]] = []
    for event in risk_messages[:5]:
        owner = (event.source_agent or "unknown").lower()
        snippet = event.content[:180].strip()
        severity = "medium"
        lowered = snippet.lower()
        if "critical" in lowered or "failure" in lowered:
            severity = "high"
        register.append(
            {
                "name": f"Risk signal at step {event.step}",
                "severity": severity,
                "evidence": snippet,
                "owner": owner,
            }
        )
    return register


def _action_items(
    kpi: dict[str, Any],
    risk_register: list[dict[str, Any]],
    intervention_events: list[Event],
) -> list[str]:
    dims = kpi["aggregate"]["dimensions"]
    actions = [
        f"Raise {min(dims, key=dims.get)} by defining measurable acceptance criteria for each agent output.",
        "Convert top risk signals into mitigation owners and due dates in next run.",
    ]
    if intervention_events:
        actions.append("Run A/B on intervention timing (early vs late step) to isolate causal effect on KPI.")
    else:
        actions.append("Add at least one structured intervention in next run to test policy resilience.")
    return actions


def _key_evidence(message_events: list[Event]) -> list[str]:
    evidence: list[str] = []
    for event in message_events:
        if "(fallback:" in event.content:
            continue
        source = event.source_agent or "unknown"
        target = event.target_agent or "all"
        snippet = event.content[:140].replace("\n", " ").strip()
        evidence.append(f"[step {event.step}] {source} -> {target}: {snippet}")
        if len(evidence) >= 6:
            break
    if not evidence:
        evidence.append("No high-quality message evidence captured; review run configuration.")
    return evidence


def _resource_allocation_summary(events: list[Event]) -> dict[str, Any] | None:
    round_reports: list[dict[str, Any]] = []
    final_recommendation: dict[str, Any] | None = None
    for event in events:
        content = _strip_event_timestamp(event.content)
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith(RESOURCE_ROUND_PREFIX):
                encoded = stripped[len(RESOURCE_ROUND_PREFIX) :].strip()
                try:
                    parsed = json.loads(encoded)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    round_reports.append(parsed)
                continue
            if stripped.startswith(RESOURCE_FINAL_PREFIX):
                encoded = stripped[len(RESOURCE_FINAL_PREFIX) :].strip()
                try:
                    parsed = json.loads(encoded)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    final_recommendation = parsed

    if not round_reports:
        return None

    total_output = sum(
        float(item.get("total_output", item.get("output", 0.0)))
        for item in round_reports
    )
    outputs = [float(item.get("total_output", item.get("output", 0.0))) for item in round_reports]
    conflict_rounds = sum(1 for item in round_reports if bool(item.get("conflict", False)))
    free_rider_rounds = 0
    strike_rounds = 0
    mechanism_counts: dict[str, int] = {}
    trust_means: list[float] = []
    mean_stresses: list[float] = []
    mean_moods: list[float] = []
    trust_variabilities: list[float] = []
    adjustment_ratios: list[float] = []
    satisfaction_dispersions: list[float] = []
    fairness_dispersions: list[float] = []
    allocation_shift_l1: list[float] = []
    willingness_shift_l1: list[float] = []
    public_stability_values: list[float] = []
    social_welfare_values: list[float] = []
    institution_cost_rates: list[float] = []
    shock_severities: list[float] = []
    shock_types: list[str] = []
    causal_agent_checks = 0
    causal_agent_pass = 0
    narrative_checks = 0
    narrative_pass = 0
    influenced_non_none = 0
    influenced_total = 0
    influence_unique_pairs: set[tuple[str, str]] = set()
    prev_alloc: dict[str, float] | None = None
    prev_will: dict[str, float] | None = None
    for item in round_reports:
        free_riders = item.get("free_rider_agents", [])
        if isinstance(free_riders, list) and free_riders:
            free_rider_rounds += 1
        mechanism = str(item.get("selected_mechanism", "unknown"))
        mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
        if "trust_mean" in item:
            try:
                trust_means.append(float(item.get("trust_mean", 0.0)))
            except Exception:
                pass
        stresses = item.get("stresses", {})
        if isinstance(stresses, dict) and stresses:
            vals = [float(v) for v in stresses.values()]
            mean_stresses.append(sum(vals) / len(vals))
        moods = item.get("moods", {})
        if isinstance(moods, dict) and moods:
            vals = [float(v) for v in moods.values()]
            mean_moods.append(sum(vals) / len(vals))
        if "public_stability_index" in item:
            try:
                public_stability_values.append(float(item.get("public_stability_index", 0.0)))
            except Exception:
                pass
        if "social_welfare_index" in item:
            try:
                social_welfare_values.append(float(item.get("social_welfare_index", 0.0)))
            except Exception:
                pass
        gross_output = float(item.get("gross_output", item.get("total_output", 0.0)))
        institution_cost = float(item.get("institution_cost_units", 0.0))
        if gross_output > 1e-8:
            institution_cost_rates.append(max(0.0, institution_cost / gross_output))
        env = item.get("environment_signal", {})
        if isinstance(env, dict):
            try:
                shock_severities.append(float(env.get("shock_severity", 0.0)))
            except Exception:
                pass
            shock_types.append(str(env.get("shock_type", "none")))
        causal = item.get("causal_consistency", {})
        if isinstance(causal, dict):
            for _, value in causal.items():
                if not isinstance(value, dict):
                    continue
                if "passed" in value:
                    causal_agent_checks += 1
                    if bool(value.get("passed", False)):
                        causal_agent_pass += 1
        elif "causal_consistency_score" in item:
            try:
                score = float(item.get("causal_consistency_score", 0.0))
                causal_agent_checks += 3
                causal_agent_pass += int(round(max(0.0, min(1.0, score)) * 3))
            except Exception:
                pass

        round_conclusion = str(item.get("round_conclusion", "")).lower()
        if round_conclusion:
            expected_conflict = "yes" if bool(item.get("conflict", False)) else "no"
            mention_conflict = f"conflict={expected_conflict}" in round_conclusion
            try:
                trust_mean = float(item.get("trust_mean", 0.0))
                mention_trust = f"trust_mean={trust_mean:.2f}".lower() in round_conclusion
            except Exception:
                mention_trust = False
            narrative_checks += 2
            narrative_pass += int(mention_conflict) + int(mention_trust)
        caps = item.get("effort_cap_ratio", {})
        if isinstance(caps, dict) and caps:
            try:
                if any(float(v) < 0.70 for v in caps.values()):
                    strike_rounds += 1
            except Exception:
                pass
        trust_matrix = item.get("trust_matrix", {})
        if isinstance(trust_matrix, dict) and trust_matrix:
            trust_vals: list[float] = []
            for src, dsts in trust_matrix.items():
                if not isinstance(dsts, dict):
                    continue
                for dst, value in dsts.items():
                    if str(src).upper() == str(dst).upper():
                        continue
                    try:
                        trust_vals.append(float(value))
                    except Exception:
                        pass
            if trust_vals:
                trust_variabilities.append(max(trust_vals) - min(trust_vals))
        if "behavioral_adjustment_ratio" in item:
            try:
                adjustment_ratios.append(float(item.get("behavioral_adjustment_ratio", 0.0)))
            except Exception:
                pass
        if "satisfaction_dispersion" in item:
            try:
                satisfaction_dispersions.append(float(item.get("satisfaction_dispersion", 0.0)))
            except Exception:
                pass
        elif isinstance(item.get("satisfactions"), dict):
            sat_vals = [float(v) for v in dict(item.get("satisfactions", {})).values()]
            if sat_vals:
                satisfaction_dispersions.append(max(sat_vals) - min(sat_vals))
        if "fairness_dispersion" in item:
            try:
                fairness_dispersions.append(float(item.get("fairness_dispersion", 0.0)))
            except Exception:
                pass
        alloc_now = item.get("allocation_plan_units", {})
        if isinstance(alloc_now, dict):
            candidate = {}
            for key in ("A", "B", "C"):
                try:
                    candidate[key] = float(alloc_now[key])
                except Exception:
                    candidate = {}
                    break
            if candidate:
                if prev_alloc is not None:
                    shift = sum(abs(candidate[k] - prev_alloc[k]) for k in ("A", "B", "C")) / 300.0
                    allocation_shift_l1.append(shift)
                prev_alloc = candidate
        will_now = item.get("commitment_units", {})
        if isinstance(will_now, dict):
            candidate = {}
            for key in ("A", "B", "C"):
                try:
                    candidate[key] = float(will_now[key])
                except Exception:
                    candidate = {}
                    break
            if candidate:
                if prev_will is not None:
                    shift = sum(abs(candidate[k] - prev_will[k]) for k in ("A", "B", "C")) / 300.0
                    willingness_shift_l1.append(shift)
                prev_will = candidate
        proposals = item.get("agent_proposals", {})
        if isinstance(proposals, dict):
            for agent, proposal in proposals.items():
                if not isinstance(proposal, dict):
                    continue
                influenced_total += 1
                influenced_by = str(proposal.get("influenced_by", "NONE")).upper()
                if influenced_by in {"A", "B", "C"} and influenced_by != str(agent).upper():
                    influenced_non_none += 1
                    influence_unique_pairs.add((str(agent).upper(), influenced_by))

    route_counts: dict[str, int] = {}
    for event in events:
        if event.event_type != "message":
            continue
        src = (event.source_agent or "").strip().upper()
        dst = (event.target_agent or "").strip().upper()
        if src in {"A", "B", "C"} and dst in {"A", "B", "C"} and src != dst:
            key = f"{src}->{dst}"
            route_counts[key] = route_counts.get(key, 0) + 1

    dominant_mechanism = max(
        mechanism_counts.items(),
        key=lambda pair: (pair[1], pair[0]),
    )[0]
    final_round = round_reports[-1]
    allocation = final_round.get("allocation_plan_units", final_round.get("payoffs", {}))
    final_round_allocation = dict(allocation) if isinstance(allocation, dict) else {}
    final_willingness = final_round.get("commitment_units", {})
    final_round_willingness = dict(final_willingness) if isinstance(final_willingness, dict) else {}
    causal_consistency_rate = (
        (causal_agent_pass / causal_agent_checks) if causal_agent_checks > 0 else 0.0
    )
    narrative_data_consistency_score = (
        (narrative_pass / narrative_checks) if narrative_checks > 0 else 0.0
    )
    institutional_sensitivity_score = _institutional_sensitivity_score(
        shock_severities=shock_severities,
        outputs=outputs,
        trust_means=trust_means,
        institution_cost_rates=institution_cost_rates,
        shock_types=shock_types,
    )

    summary = {
        "round_count": len(round_reports),
        "total_output": round(total_output, 2),
        "mean_output": round(total_output / len(round_reports), 2),
        "conflict_rounds": conflict_rounds,
        "free_rider_rounds": free_rider_rounds,
        "strike_rounds": strike_rounds,
        "recommended_mechanism": dominant_mechanism,
        "mean_behavioral_adjustment_ratio": round(_safe_mean(adjustment_ratios), 4),
        "mean_satisfaction_dispersion": round(_safe_mean(satisfaction_dispersions), 4),
        "mean_fairness_dispersion": round(_safe_mean(fairness_dispersions), 4),
        "mean_trust_variability": round(_safe_mean(trust_variabilities), 4),
        "allocation_revision_intensity": round(_safe_mean(allocation_shift_l1), 4),
        "willingness_revision_intensity": round(_safe_mean(willingness_shift_l1), 4),
        "mean_public_stability_index": round(_safe_mean(public_stability_values), 4),
        "mean_social_welfare_index": round(_safe_mean(social_welfare_values), 4),
        "mean_institution_cost_rate": round(_safe_mean(institution_cost_rates), 4),
        "causal_consistency_rate": round(causal_consistency_rate, 4),
        "institutional_sensitivity_score": round(institutional_sensitivity_score, 4),
        "narrative_data_consistency_score": round(narrative_data_consistency_score, 4),
        "final_round_allocation": final_round_allocation,
        "final_round_willingness": final_round_willingness,
        "final_round_conclusion": str(final_round.get("round_conclusion", "")),
        "realism": _compute_resource_realism(
            outputs=outputs,
            conflict_rounds=conflict_rounds,
            round_count=len(round_reports),
            route_counts=route_counts,
            trust_means=trust_means,
            mean_stresses=mean_stresses,
            mean_moods=mean_moods,
            influenced_non_none=influenced_non_none,
            influenced_total=influenced_total,
            influence_unique_pairs_count=len(influence_unique_pairs),
            strike_rounds=strike_rounds,
            adjustment_ratios=adjustment_ratios,
            satisfaction_dispersions=satisfaction_dispersions,
            fairness_dispersions=fairness_dispersions,
            trust_variabilities=trust_variabilities,
            allocation_revision_intensity=allocation_shift_l1,
            willingness_revision_intensity=willingness_shift_l1,
            causal_consistency_rate=causal_consistency_rate,
            institutional_sensitivity_score=institutional_sensitivity_score,
            narrative_data_consistency_score=narrative_data_consistency_score,
            mean_public_stability=_safe_mean(public_stability_values),
            mean_social_welfare=_safe_mean(social_welfare_values),
        ),
    }
    if final_recommendation:
        if "recommended_mechanism" in final_recommendation:
            summary["recommended_mechanism"] = str(final_recommendation["recommended_mechanism"])
        summary["observer_final_recommendation"] = final_recommendation
    return summary


def _strip_event_timestamp(content: str) -> str:
    if " @ " not in content:
        return content.strip()
    return content.rsplit(" @ ", 1)[0].strip()


def _report_payload_to_markdown(payload: dict[str, Any]) -> str:
    simulation = payload["simulation"]
    kpi = payload["kpi"]
    lines = [
        f"# {payload['title']}",
        "",
        f"- Generated At: {payload['generated_at']}",
        f"- Simulation ID: {simulation['id']}",
        f"- Status: {simulation['status']}",
        f"- Goal: {simulation['goal']}",
        "",
        "## Executive Summary",
        payload["executive_summary"],
        "",
        "## KPI Snapshot",
        f"- Overall KPI: {kpi['aggregate_overall']:.2f}",
        f"- Trend Delta: {kpi['trend_delta']:+.2f}",
        "- Dimensions:",
    ]
    for key, value in kpi["dimensions"].items():
        lines.append(f"  - {key}: {float(value):.2f}")
    lines.extend(["", "## Key Findings"])
    for item in payload["key_findings"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Risk Register"])
    for risk in payload["risk_register"]:
        lines.append(
            f"- [{risk['severity']}] {risk['name']} (owner: {risk['owner']}): {risk['evidence']}"
        )
    lines.extend(["", "## Action Items"])
    for item in payload["action_items"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Evidence"])
    for item in payload["evidence"]:
        lines.append(f"- {item}")
    resource = payload.get("resource_allocation")
    if isinstance(resource, dict):
        lines.extend(["", "## Resource Allocation Conclusion"])
        lines.append(f"- Rounds observed: {int(resource.get('round_count', 0))}")
        lines.append(f"- Total output: {float(resource.get('total_output', 0.0)):.2f}")
        lines.append(f"- Mean output per round: {float(resource.get('mean_output', 0.0)):.2f}")
        lines.append(f"- Conflict rounds: {int(resource.get('conflict_rounds', 0))}")
        lines.append(f"- Free-rider rounds: {int(resource.get('free_rider_rounds', 0))}")
        lines.append(f"- Mean public stability index: {float(resource.get('mean_public_stability_index', 0.0)):.3f}")
        lines.append(f"- Mean social welfare index: {float(resource.get('mean_social_welfare_index', 0.0)):.3f}")
        lines.append(f"- Causal consistency rate: {float(resource.get('causal_consistency_rate', 0.0)):.3f}")
        lines.append(f"- Institutional sensitivity score: {float(resource.get('institutional_sensitivity_score', 0.0)):.3f}")
        lines.append(f"- Narrative-data consistency score: {float(resource.get('narrative_data_consistency_score', 0.0)):.3f}")
        lines.append(f"- Recommended mechanism: {resource.get('recommended_mechanism', 'unknown')}")
        willingness = resource.get("final_round_willingness", {})
        if isinstance(willingness, dict) and willingness:
            willingness_text = " | ".join(
                f"{agent}:{float(value):.2f}" for agent, value in willingness.items()
            )
            lines.append(f"- Final willingness (target=100): {willingness_text}")
        allocation = resource.get("final_round_allocation", {})
        if isinstance(allocation, dict) and allocation:
            allocation_text = " | ".join(
                f"{agent}:{float(value):.2f}" for agent, value in allocation.items()
            )
            lines.append(f"- Final round allocation (target=100): {allocation_text}")
        final_conclusion = str(resource.get("final_round_conclusion", "")).strip()
        if final_conclusion:
            lines.append(f"- Final round conclusion: {final_conclusion}")
        observer_final = resource.get("observer_final_recommendation")
        if isinstance(observer_final, dict) and observer_final:
            rationale = str(observer_final.get("rationale", "")).strip()
            if rationale:
                lines.append(f"- Observer rationale: {rationale}")
        realism = resource.get("realism")
        if isinstance(realism, dict):
            lines.extend(["", "## Resource Realism Metrics"])
            lines.append(f"- Realism score: {float(realism.get('score', 0.0)):.1f}/100")
            lines.append(f"- Grade: {realism.get('grade', 'unknown')}")
            lines.append(f"- Standard passed: {bool(realism.get('standard_passed', False))}")
            lines.append(f"- Standard threshold: {float(realism.get('standard_threshold', 0.0)):.1f}")
            components = realism.get("components", {})
            if isinstance(components, dict):
                for key, value in components.items():
                    lines.append(f"- {key}: {float(value):.3f}")
            interpretation = str(realism.get("interpretation", "")).strip()
            if interpretation:
                lines.append(f"- Interpretation: {interpretation}")
    lines.extend(["", "## Emergent Patterns"])
    for item in payload["emergent_patterns"]:
        lines.append(
            f"- step {item['step']}: {item['label']} (score {item['score']}) - {item['reason']}"
        )
    return "\n".join(lines)


def _compute_resource_realism(
    *,
    outputs: list[float],
    conflict_rounds: int,
    round_count: int,
    route_counts: dict[str, int],
    trust_means: list[float],
    mean_stresses: list[float],
    mean_moods: list[float],
    influenced_non_none: int,
    influenced_total: int,
    influence_unique_pairs_count: int,
    strike_rounds: int,
    adjustment_ratios: list[float],
    satisfaction_dispersions: list[float],
    fairness_dispersions: list[float],
    trust_variabilities: list[float],
    allocation_revision_intensity: list[float],
    willingness_revision_intensity: list[float],
    causal_consistency_rate: float,
    institutional_sensitivity_score: float,
    narrative_data_consistency_score: float,
    mean_public_stability: float,
    mean_social_welfare: float,
) -> dict[str, Any]:
    if round_count <= 0:
        return {
            "score": 0.0,
            "grade": "insufficient_data",
            "standard_passed": False,
            "standard_threshold": 68.0,
            "components": {},
            "interpretation": "No rounds captured.",
        }

    mean_output = _safe_mean(outputs)
    if len(outputs) > 1 and mean_output > 1e-8:
        variance = sum((x - mean_output) ** 2 for x in outputs) / len(outputs)
        cv = math.sqrt(variance) / mean_output
    else:
        cv = 0.0
    conflict_rate = conflict_rounds / round_count

    route_total = sum(route_counts.values())
    route_entropy = 0.0
    if route_total > 0:
        probs = [count / route_total for count in route_counts.values() if count > 0]
        route_entropy = -sum(p * math.log(p) for p in probs)
    route_entropy_norm = 0.0
    if route_total > 0:
        route_entropy_norm = route_entropy / math.log(6.0)
        route_entropy_norm = max(0.0, min(1.0, route_entropy_norm))

    trust_drift = 0.0
    if len(trust_means) >= 2:
        trust_drift = abs(trust_means[-1] - trust_means[0])

    stress_mean = _safe_mean(mean_stresses)
    mood_variability = 0.0
    if len(mean_moods) >= 2:
        mood_variability = max(mean_moods) - min(mean_moods)

    trust_variability_mean = _safe_mean(trust_variabilities)
    adjustment_mean = _safe_mean(adjustment_ratios)
    sat_dispersion_mean = _safe_mean(satisfaction_dispersions)
    fairness_dispersion_mean = _safe_mean(fairness_dispersions)
    allocation_revision_mean = _safe_mean(allocation_revision_intensity)
    willingness_revision_mean = _safe_mean(willingness_revision_intensity)
    revision_intensity = 0.5 * allocation_revision_mean + 0.5 * willingness_revision_mean
    strike_rate = strike_rounds / round_count

    autonomy_ratio = influenced_non_none / influenced_total if influenced_total > 0 else 0.0
    influence_diversity = min(1.0, influence_unique_pairs_count / 6.0)

    score_conflict = _saturating_score(conflict_rate, scale=0.28)
    score_volatility = _saturating_score(cv, scale=0.22)
    score_routes = route_entropy_norm
    score_trust = _saturating_score(trust_variability_mean, scale=0.14)
    score_stress = _saturating_score(stress_mean, scale=0.45)
    score_adjustment = _saturating_score(adjustment_mean, scale=0.60)
    score_revision = _saturating_score(revision_intensity, scale=0.12)
    score_strike = _saturating_score(strike_rate, scale=0.30)
    score_public_stability = _saturating_score(mean_public_stability, scale=0.70)
    score_social_welfare = _saturating_score(mean_social_welfare, scale=0.70)
    score_autonomy = max(0.0, min(1.0, 0.65 * autonomy_ratio + 0.35 * influence_diversity))

    behavior_dynamic_score = (
        0.13 * score_conflict
        + 0.13 * score_volatility
        + 0.10 * score_routes
        + 0.12 * score_trust
        + 0.10 * score_stress
        + 0.13 * score_adjustment
        + 0.12 * score_revision
        + 0.07 * score_strike
        + 0.05 * score_autonomy
        + 0.03 * score_public_stability
        + 0.02 * score_social_welfare
    )
    behavior_dynamic_score = max(0.0, min(1.0, behavior_dynamic_score))
    causal_consistency_score = max(0.0, min(1.0, causal_consistency_rate))
    institution_score = max(0.0, min(1.0, institutional_sensitivity_score))
    narrative_score = max(0.0, min(1.0, narrative_data_consistency_score))

    realism_index = (
        0.50 * behavior_dynamic_score
        + 0.20 * institution_score
        + 0.20 * causal_consistency_score
        + 0.10 * narrative_score
    )
    realism_score = round(realism_index * 100.0, 2)
    standard_threshold = 68.0
    standard_passed = (
        realism_score >= standard_threshold
        and behavior_dynamic_score >= 0.55
        and conflict_rate >= 0.08
        and causal_consistency_score >= 0.85
        and institution_score >= 0.40
        and narrative_score >= 0.75
    )

    if realism_score >= 72:
        grade = "high_realism"
    elif realism_score >= 58:
        grade = "moderate_realism"
    else:
        grade = "low_realism"

    if standard_passed:
        interpretation = "Dynamics pass realism standard: social friction, adaptation, and trust shifts are all observable."
    else:
        weak_dims: list[str] = []
        if conflict_rate < 0.05:
            weak_dims.append("insufficient conflict")
        if adjustment_mean < 0.30:
            weak_dims.append("low behavioral adjustment")
        if revision_intensity < 0.03:
            weak_dims.append("low proposal revision")
        if route_entropy_norm < 0.35:
            weak_dims.append("low communication diversity")
        if trust_variability_mean < 0.06:
            weak_dims.append("weak trust dynamics")
        if causal_consistency_score < 0.85:
            weak_dims.append("causal consistency below 0.85")
        if institution_score < 0.40:
            weak_dims.append("institutional sensitivity too weak")
        if narrative_score < 0.75:
            weak_dims.append("narrative-data consistency too weak")
        if not weak_dims:
            weak_dims.append("global realism score below threshold")
        interpretation = "Needs calibration: " + ", ".join(weak_dims) + "."

    return {
        "score": realism_score,
        "grade": grade,
        "standard_passed": standard_passed,
        "standard_threshold": standard_threshold,
        "components": {
            "conflict_rate": round(conflict_rate, 4),
            "output_volatility_cv": round(cv, 4),
            "route_entropy_norm": round(route_entropy_norm, 4),
            "trust_drift": round(trust_drift, 4),
            "trust_variability_mean": round(trust_variability_mean, 4),
            "mean_stress": round(stress_mean, 4),
            "mood_variability": round(mood_variability, 4),
            "adjustment_ratio_mean": round(adjustment_mean, 4),
            "satisfaction_dispersion_mean": round(sat_dispersion_mean, 4),
            "fairness_dispersion_mean": round(fairness_dispersion_mean, 4),
            "allocation_revision_intensity": round(allocation_revision_mean, 4),
            "willingness_revision_intensity": round(willingness_revision_mean, 4),
            "revision_intensity": round(revision_intensity, 4),
            "strike_rate": round(strike_rate, 4),
            "autonomy_ratio": round(autonomy_ratio, 4),
            "influence_diversity": round(influence_diversity, 4),
            "behavior_dynamic_score": round(behavior_dynamic_score, 4),
            "institutional_sensitivity_score": round(institution_score, 4),
            "causal_consistency_score": round(causal_consistency_score, 4),
            "narrative_data_consistency_score": round(narrative_score, 4),
            "mean_public_stability": round(mean_public_stability, 4),
            "mean_social_welfare": round(mean_social_welfare, 4),
        },
        "interpretation": interpretation,
    }


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _saturating_score(value: float, *, scale: float) -> float:
    if scale <= 1e-8:
        return 0.0
    clamped = max(0.0, value)
    return max(0.0, min(1.0, 1.0 - math.exp(-clamped / scale)))


def _institutional_sensitivity_score(
    *,
    shock_severities: list[float],
    outputs: list[float],
    trust_means: list[float],
    institution_cost_rates: list[float],
    shock_types: list[str],
) -> float:
    if len(shock_severities) < 2:
        return 0.0
    n = len(shock_severities)
    aligned_outputs = outputs[:n] if outputs else []
    aligned_trust = trust_means[:n] if trust_means else []
    aligned_cost = institution_cost_rates[:n] if institution_cost_rates else []

    output_resp = abs(_pearson_corr(shock_severities[: len(aligned_outputs)], aligned_outputs))
    trust_resp = abs(_pearson_corr(shock_severities[: len(aligned_trust)], aligned_trust))
    cost_resp = _pearson_corr(shock_severities[: len(aligned_cost)], aligned_cost)
    cost_resp = max(0.0, cost_resp)

    output_volatility = 0.0
    if aligned_outputs:
        output_mean = _safe_mean(aligned_outputs)
        if output_mean > 1e-8:
            output_var = sum((x - output_mean) ** 2 for x in aligned_outputs) / len(aligned_outputs)
            output_volatility = max(0.0, min(1.0, math.sqrt(output_var) / output_mean))

    trust_volatility = 0.0
    if aligned_trust:
        trust_volatility = max(0.0, min(1.0, max(aligned_trust) - min(aligned_trust)))
    mean_cost_rate = _safe_mean(aligned_cost)
    governance_activation = _saturating_score(mean_cost_rate, scale=0.06)

    unique_types = {item for item in shock_types if item and item != "none"}
    shock_diversity = max(0.0, min(1.0, len(unique_types) / 5.0))
    trust_activation = _saturating_score(trust_volatility, scale=0.20)
    output_activation = _saturating_score(output_volatility, scale=0.30)

    return max(
        0.0,
        min(
            1.0,
            0.12 * output_resp
            + 0.08 * trust_resp
            + 0.15 * cost_resp
            + 0.40 * governance_activation
            + 0.10 * output_activation
            + 0.10 * trust_activation
            + 0.05 * shock_diversity,
        ),
    )


def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    if not xs or not ys:
        return 0.0
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    a = xs[:n]
    b = ys[:n]
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((va - mean_a) * (vb - mean_b) for va, vb in zip(a, b, strict=False))
    var_a = sum((va - mean_a) ** 2 for va in a)
    var_b = sum((vb - mean_b) ** 2 for vb in b)
    denom = math.sqrt(var_a * var_b)
    if denom <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, cov / denom))
