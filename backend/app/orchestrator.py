import asyncio
from collections.abc import Callable

from sqlmodel import Session, select

from .models import Agent, Event, Simulation, SimulationStatus
from .utils import utc_iso_now


class Orchestrator:
    def __init__(self, ws_manager) -> None:
        self.ws_manager = ws_manager
        self._tasks: dict[int, asyncio.Task] = {}

    async def start(self, simulation_id: int, session_factory: Callable[[], Session]) -> None:
        if simulation_id in self._tasks and not self._tasks[simulation_id].done():
            return
        self._tasks[simulation_id] = asyncio.create_task(self._run(simulation_id, session_factory))

    async def stop(self, simulation_id: int) -> None:
        task = self._tasks.get(simulation_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _run(self, simulation_id: int, session_factory: Callable[[], Session]) -> None:
        while True:
            with session_factory() as session:
                sim = session.get(Simulation, simulation_id)
                if not sim:
                    return
                if sim.status != SimulationStatus.running:
                    return
                if sim.step >= sim.max_steps:
                    sim.status = SimulationStatus.completed
                    session.add(sim)
                    session.commit()
                    return

                sim.step += 1
                session.add(sim)

                agents = session.exec(select(Agent).where(Agent.simulation_id == sim.id)).all()
                for agent in agents:
                    evt = Event(
                        simulation_id=sim.id,
                        step=sim.step,
                        event_type="action",
                        source_agent=agent.name,
                        content=f"{agent.role} step {sim.step} @ {utc_iso_now()}",
                    )
                    session.add(evt)

                session.commit()
                step = sim.step
                sleep_ms = sim.tick_interval_ms

            await self.ws_manager.broadcast(simulation_id, {"type": "tick", "step": step})
            await asyncio.sleep(sleep_ms / 1000)
