"""Pydantic request schemas for frontend-exposed API endpoints."""

from pydantic import BaseModel, ConfigDict, Field


class AgentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: str


class SimulationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    max_steps: int = 30
    tick_interval_ms: int = 800
    agents: list[AgentCreate] = Field(default_factory=list)


class InterventionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=2000)


class ScenarioSimulationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int | None = Field(default=None, ge=1, le=500)
    tick_interval_ms: int | None = Field(default=None, ge=10, le=10000)
