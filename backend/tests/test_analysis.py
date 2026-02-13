"""Unit tests for emergent pattern labeling heuristics."""

from app.analysis import detect_emergent_patterns
from app.models import Event


def test_detect_emergent_patterns_fallback_mode():
    events = [
        Event(step=1, simulation_id=1, event_type="message", source_agent="A", target_agent="B", content="hello (fallback:BadRequestError)"),
        Event(step=1, simulation_id=1, event_type="message", source_agent="B", target_agent="A", content="ack"),
    ]
    patterns = detect_emergent_patterns(events=events, agent_names=["A", "B"])
    assert patterns
    assert patterns[0]["label"] == "fallback_mode"


def test_detect_emergent_patterns_balanced_exchange():
    events = [
        Event(step=2, simulation_id=1, event_type="message", source_agent="A", target_agent="B", content="plan"),
        Event(step=2, simulation_id=1, event_type="message", source_agent="B", target_agent="C", content="agree"),
        Event(step=2, simulation_id=1, event_type="message", source_agent="C", target_agent="A", content="next step"),
    ]
    patterns = detect_emergent_patterns(events=events, agent_names=["A", "B", "C"])
    assert patterns
    assert patterns[0]["label"] == "balanced_exchange"
