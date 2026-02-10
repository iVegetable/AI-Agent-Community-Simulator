from pydantic import BaseModel, Field


class AgentCreate(BaseModel):
    name: str
    role: str


class SimulationCreate(BaseModel):
    goal: str
    max_steps: int = 30
    tick_interval_ms: int = 800
    agents: list[AgentCreate] = Field(default_factory=list)
