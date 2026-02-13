"""Objective KPI scoring for simulation progress and collaboration quality.

The scoring pipeline computes step-level and aggregate dimensions, then
produces a weighted overall score used by API summaries and experiment reports.
"""

from collections import defaultdict
from math import sqrt

from .models import Event

KPI_WEIGHTS = {
    "progress": 0.30,
    "coordination": 0.25,
    "risk_management": 0.20,
    "quality": 0.15,
    "stability": 0.10,
}

PROGRESS_WORDS = (
    "plan",
    "next",
    "timeline",
    "milestone",
    "deliver",
    "execute",
    "launch",
    "ship",
    "roadmap",
)
RISK_WORDS = (
    "risk",
    "issue",
    "concern",
    "blocker",
    "tradeoff",
    "mitigate",
    "failure",
)
QUALITY_WORDS = (
    "because",
    "data",
    "metric",
    "evidence",
    "assumption",
    "hypothesis",
    "constraint",
)


def compute_kpi(events: list[Event], agent_names: list[str]) -> dict[str, object]:
    """Compute KPI timeline, current score, aggregate score, and trend."""

    by_step: dict[int, list[Event]] = defaultdict(list)
    for event in events:
        by_step[event.step].append(event)

    if not by_step:
        zero_dimensions = {
            "progress": 0.0,
            "coordination": 0.0,
            "risk_management": 0.0,
            "quality": 0.0,
            "stability": 0.0,
        }
        zero_step = {
            "step": 0,
            "overall": 0.0,
            "dimensions": zero_dimensions,
        }
        return {
            "weights": KPI_WEIGHTS,
            "current": zero_step,
            "aggregate": {"overall": 0.0, "dimensions": zero_dimensions},
            "trend_delta": 0.0,
            "history": [],
        }

    history: list[dict[str, object]] = []
    for step in sorted(by_step.keys()):
        step_events = by_step[step]
        dimensions = _score_step(step_events, agent_names)
        overall = _weighted_score(dimensions)
        history.append(
            {
                "step": step,
                "overall": overall,
                "dimensions": dimensions,
            }
        )

    avg_dimensions = {
        key: round(sum(item["dimensions"][key] for item in history) / len(history), 2)  # type: ignore[index]
        for key in KPI_WEIGHTS
    }
    latest_step = history[-1]
    aggregate = {
        "overall": round(sum(item["overall"] for item in history) / len(history), 2),  # type: ignore[index]
        "dimensions": avg_dimensions,
    }
    trend_delta = round(float(latest_step["overall"]) - float(history[0]["overall"]), 2)
    return {
        "weights": KPI_WEIGHTS,
        "current": latest_step,
        "aggregate": aggregate,
        "trend_delta": trend_delta,
        "history": history,
    }


def _score_step(step_events: list[Event], agent_names: list[str]) -> dict[str, float]:
    messages = [event for event in step_events if event.event_type == "message"]
    actions = [event for event in step_events if event.event_type == "action"]

    message_count = len(messages)
    action_count = len(actions)
    if message_count == 0 and action_count == 0:
        return {
            "progress": 0.0,
            "coordination": 0.0,
            "risk_management": 0.0,
            "quality": 0.0,
            "stability": 0.0,
        }

    normalized_goal_words = _keyword_density(messages, PROGRESS_WORDS)
    action_ratio = action_count / max(1, len(step_events))
    progress = _clamp(0.6 * normalized_goal_words + 0.4 * action_ratio)

    participating_sources = {event.source_agent for event in step_events if event.source_agent}
    participation_ratio = len(participating_sources.intersection(set(agent_names))) / max(1, len(agent_names))
    targeted_ratio = sum(1 for event in messages if event.target_agent) / max(1, message_count)
    distribution_balance = _source_balance(step_events, agent_names)
    coordination = _clamp(0.4 * participation_ratio + 0.3 * targeted_ratio + 0.3 * distribution_balance)

    risk_density = _keyword_density(messages, RISK_WORDS)
    critic_participation = 1.0 if any((event.source_agent or "").lower() == "critic" for event in messages) else 0.0
    risk_management = _clamp(0.65 * risk_density + 0.35 * critic_participation)

    quality_density = _keyword_density(messages, QUALITY_WORDS)
    avg_words = _average_words(messages)
    length_score = _clamp(avg_words / 20.0)
    quality = _clamp(0.7 * quality_density + 0.3 * length_score)

    fallback_ratio = sum(1 for event in step_events if "(fallback:" in event.content) / max(1, len(step_events))
    stability = _clamp(1.0 - fallback_ratio)

    return {
        "progress": round(progress * 100, 2),
        "coordination": round(coordination * 100, 2),
        "risk_management": round(risk_management * 100, 2),
        "quality": round(quality * 100, 2),
        "stability": round(stability * 100, 2),
    }


def _source_balance(step_events: list[Event], agent_names: list[str]) -> float:
    if not agent_names:
        return 0.0
    counts = []
    for name in agent_names:
        count = sum(1 for event in step_events if event.source_agent == name)
        counts.append(count)
    mean = sum(counts) / len(counts)
    if mean <= 0:
        return 0.0
    variance = sum((count - mean) ** 2 for count in counts) / len(counts)
    std = sqrt(variance)
    return _clamp(1.0 - (std / (mean + 1e-9)))


def _keyword_density(messages: list[Event], keywords: tuple[str, ...]) -> float:
    if not messages:
        return 0.0
    hits = 0
    for event in messages:
        text = event.content.lower()
        if any(word in text for word in keywords):
            hits += 1
    return _clamp(hits / len(messages))


def _average_words(messages: list[Event]) -> float:
    if not messages:
        return 0.0
    counts = [len(event.content.split()) for event in messages]
    return sum(counts) / len(counts)


def _weighted_score(dimensions: dict[str, float]) -> float:
    total = 0.0
    for key, weight in KPI_WEIGHTS.items():
        total += dimensions.get(key, 0.0) * weight
    return round(total, 2)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
