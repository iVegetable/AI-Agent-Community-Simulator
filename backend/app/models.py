from enum import Enum
from typing import Any, Optional

from sqlmodel import Field, JSON, SQLModel


class SimulationStatus(str, Enum):
    created = "created"
    running = "running"
    stopped = "stopped"
    completed = "completed"


class Simulation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    goal: str
    status: SimulationStatus = Field(default=SimulationStatus.created)
    step: int = Field(default=0)
    max_steps: int = Field(default=30)
    tick_interval_ms: int = Field(default=800)


class Agent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    simulation_id: int = Field(index=True)
    name: str
    role: str
    memory: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)


class Event(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    simulation_id: int = Field(index=True)
    step: int = Field(default=0)
    event_type: str
    source_agent: Optional[str] = None
    target_agent: Optional[str] = None
    content: str = ""
