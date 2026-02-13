"""Unit tests for message target normalization and routing fallback."""

from app.agents.adapters.base import AgentDecision, InboxMessage
from app.orchestrator import Orchestrator


class _DummyWS:
    async def broadcast(self, *_args, **_kwargs):
        return None


def test_normalize_message_uses_inbox_sender_first():
    orch = Orchestrator(_DummyWS())
    decision = AgentDecision(event_type="message", content="x", target_agent=None, memory_append=[])
    out = orch._normalize_decision(
        decision=decision,
        source_agent="Planner",
        peers=["Researcher", "Critic"],
        inbox=[InboxMessage(source_agent="Critic", content="ping")],
        partner_counts={("Planner", "Researcher"): 5, ("Planner", "Critic"): 0},
    )
    assert out.target_agent == "Critic"


def test_normalize_message_routes_to_least_interacted_peer():
    orch = Orchestrator(_DummyWS())
    decision = AgentDecision(event_type="message", content="x", target_agent=None, memory_append=[])
    out = orch._normalize_decision(
        decision=decision,
        source_agent="Planner",
        peers=["Researcher", "Critic"],
        inbox=[],
        partner_counts={("Planner", "Researcher"): 4, ("Researcher", "Planner"): 4, ("Planner", "Critic"): 0},
    )
    assert out.target_agent == "Critic"
