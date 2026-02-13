"""Unit tests for KPI scoring behavior and fallback penalties."""

from app.models import Event
from app.scoring import compute_kpi


def test_compute_kpi_returns_history_and_current():
    events = [
        Event(step=1, simulation_id=1, event_type="message", source_agent="Planner", target_agent="Critic", content="Plan next milestone and timeline."),
        Event(step=1, simulation_id=1, event_type="message", source_agent="Critic", target_agent="Planner", content="Key risk is unclear scope."),
        Event(step=2, simulation_id=1, event_type="message", source_agent="Planner", target_agent="Critic", content="Fallback path (fallback:BadRequestError)"),
    ]
    result = compute_kpi(events=events, agent_names=["Planner", "Critic"])
    assert "current" in result
    assert "history" in result
    assert len(result["history"]) == 2
    assert result["current"]["step"] == 2


def test_compute_kpi_fallback_reduces_stability():
    events = [
        Event(step=1, simulation_id=1, event_type="message", source_agent="A", target_agent="B", content="Normal message"),
        Event(step=2, simulation_id=1, event_type="message", source_agent="A", target_agent="B", content="Fallback path (fallback:BadRequestError)"),
    ]
    result = compute_kpi(events=events, agent_names=["A", "B"])
    history = result["history"]
    assert history[0]["dimensions"]["stability"] > history[1]["dimensions"]["stability"]
