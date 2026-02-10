import asyncio
from sqlmodel import Session

from app.db import engine, init_db
from app.models import Agent, Simulation, SimulationStatus
from app.messaging import ConnectionManager
from app.orchestrator import Orchestrator


def test_orchestrator_runs_mock():
    init_db()
    with Session(engine) as session:
        sim = Simulation(goal="test", status=SimulationStatus.running, max_steps=2, tick_interval_ms=10)
        session.add(sim)
        session.commit()
        session.refresh(sim)

        agent = Agent(simulation_id=sim.id, name="A", role="planner", memory={"history": []})
        session.add(agent)
        session.commit()

    ws = ConnectionManager()
    orch = Orchestrator(ws)

    def session_factory():
        return Session(engine)

    async def run() -> None:
        await orch.start(sim.id, session_factory)
        await asyncio.sleep(0.1)
        await orch.stop(sim.id)

    asyncio.run(run())
