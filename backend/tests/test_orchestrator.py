"""Runtime orchestrator behavior tests with mock adapters."""

import asyncio
from sqlmodel import Session
from sqlmodel import select

from app.db import engine
from app.models import Agent, Event, Simulation, SimulationStatus
from app.messaging import ConnectionManager
from app.orchestrator import Orchestrator


def test_orchestrator_runs_mock():
    with Session(engine) as session:
        sim = Simulation(goal="test", status=SimulationStatus.running, max_steps=2, tick_interval_ms=10)
        session.add(sim)
        session.commit()
        session.refresh(sim)
        sim_id = sim.id

        agent = Agent(simulation_id=sim.id, name="A", role="planner", memory={"history": []})
        session.add(agent)
        session.commit()

    ws = ConnectionManager()
    orch = Orchestrator(ws)

    def session_factory():
        return Session(engine)

    async def run() -> None:
        await orch.start(sim_id, session_factory)
        await asyncio.sleep(0.1)
        await orch.stop(sim_id)

    asyncio.run(run())

    with Session(engine) as session:
        events = session.exec(select(Event).where(Event.simulation_id == sim_id)).all()
        assert len(events) >= 1


def test_resource_run_config_parses_explicit_pressure_and_seed():
    with Session(engine) as session:
        sim = Simulation(
            goal="resource config parse",
            status=SimulationStatus.created,
            max_steps=4,
            tick_interval_ms=10,
        )
        session.add(sim)
        session.commit()
        session.refresh(sim)
        sim_id = int(sim.id or 0)
        session.add(
            Event(
                simulation_id=sim_id,
                step=0,
                event_type="intervention",
                source_agent="system",
                target_agent="A",
                content=(
                    "Use contribution mechanism with hidden transparency and punishment enabled. "
                    "pressure_regime=on. run_seed=4242."
                ),
            )
        )
        session.commit()

        orch = Orchestrator(ConnectionManager())
        config = orch._resource_run_config(session, sim_id, sim.goal)
        assert config["mechanism"] == "contribution"
        assert config["transparency"] == "hidden"
        assert config["punishment"] is True
        assert config["pressure_regime"] is True
        assert int(config["seed"]) == 4242


def test_resource_run_config_no_implicit_pressure_regime():
    with Session(engine) as session:
        sim = Simulation(
            goal="resource config parse default",
            status=SimulationStatus.created,
            max_steps=4,
            tick_interval_ms=10,
        )
        session.add(sim)
        session.commit()
        session.refresh(sim)
        sim_id = int(sim.id or 0)
        session.add(
            Event(
                simulation_id=sim_id,
                step=0,
                event_type="intervention",
                source_agent="system",
                target_agent="A",
                content="Use contribution mechanism with hidden transparency and punishment enabled.",
            )
        )
        session.commit()

        orch = Orchestrator(ConnectionManager())
        config = orch._resource_run_config(session, sim_id, sim.goal)
        assert config["mechanism"] == "contribution"
        assert config["transparency"] == "hidden"
        assert config["punishment"] is True
        assert config["pressure_regime"] is False
