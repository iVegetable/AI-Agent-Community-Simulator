"""Shared data contracts for pluggable agent decision adapters."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InboxMessage:
    """One inbound message visible to an agent at a simulation step."""

    source_agent: str
    content: str


@dataclass
class DecisionContext:
    """Complete input context an adapter receives when deciding."""

    goal: str
    step: int
    agent_name: str
    agent_role: str
    peers: list[str]
    memory_history: list[str]
    inbox: list[InboxMessage]


@dataclass
class AgentDecision:
    """Normalized adapter output consumed by the orchestrator."""

    event_type: str
    content: str
    target_agent: Optional[str] = None
    memory_append: list[str] = field(default_factory=list)


class DecisionAdapter:
    """Minimal interface implemented by all decision backends."""

    async def decide(self, ctx: DecisionContext) -> AgentDecision:
        raise NotImplementedError
