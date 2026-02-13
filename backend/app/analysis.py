"""Heuristic emergent-pattern detection over simulation event streams.

Used by summary/report APIs to label run phases such as fallback behavior,
critique spikes, balanced exchanges, and early exploration.
"""

from collections import defaultdict

from .models import Event


def detect_emergent_patterns(events: list[Event], agent_names: list[str]) -> list[dict[str, object]]:
    """Return recent step-level pattern labels extracted from message events."""

    by_step: dict[int, list[Event]] = defaultdict(list)
    for event in events:
        by_step[event.step].append(event)

    patterns: list[dict[str, object]] = []
    risk_words = ("risk", "issue", "concern", "blocker", "failure")

    for step in sorted(by_step.keys()):
        step_events = by_step[step]
        messages = [e for e in step_events if e.event_type == "message"]
        if not messages:
            continue

        fallback_hits = [e for e in messages if "(fallback:" in e.content]
        if fallback_hits:
            patterns.append(
                {
                    "step": step,
                    "label": "fallback_mode",
                    "score": round(len(fallback_hits) / len(messages), 2),
                    "reason": "Model output failed schema/runtime checks and rule fallback was used.",
                }
            )
            continue

        critic_risk_msgs = 0
        for event in messages:
            content = event.content.lower()
            if event.source_agent and event.source_agent.lower() == "critic":
                if any(word in content for word in risk_words):
                    critic_risk_msgs += 1
        if critic_risk_msgs >= 1:
            patterns.append(
                {
                    "step": step,
                    "label": "critique_spike",
                    "score": round(critic_risk_msgs / len(messages), 2),
                    "reason": "Critic produced concentrated risk/challenge feedback.",
                }
            )
            continue

        source_agents = {e.source_agent for e in messages if e.source_agent}
        target_agents = {e.target_agent for e in messages if e.target_agent}
        active_ratio = 0.0
        if agent_names:
            active_ratio = len(source_agents.intersection(set(agent_names))) / len(agent_names)
        if active_ratio >= 0.75 and len(target_agents) >= min(2, len(agent_names)):
            patterns.append(
                {
                    "step": step,
                    "label": "balanced_exchange",
                    "score": round(active_ratio, 2),
                    "reason": "Most agents participated and messages were not one-sided.",
                }
            )
            continue

        patterns.append(
            {
                "step": step,
                "label": "exploration",
                "score": round(active_ratio, 2),
                "reason": "Agents are still exploring context before convergence.",
            }
        )

    return patterns[-12:]
