"""FastAPI application entrypoint and REST/WebSocket surface.

This module intentionally exposes only the APIs that are used by the
current frontend application.
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

from .config import settings
from .db import engine, init_db
from .messaging import ConnectionManager
from .models import Agent, Event, Simulation, SimulationReport, SimulationStatus
from .orchestrator import Orchestrator
from .reporting import generate_simulation_report, get_latest_report
from .scenarios import get_scenario
from .schemas import InterventionCreate, ScenarioSimulationCreate, SimulationCreate
from .utils import utc_iso_now


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="AI Agent Community Simulator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ws_manager = ConnectionManager()
orchestrator = Orchestrator(ws_manager)


def get_session():
    with Session(engine) as session:
        yield session


@app.post("/api/v1/scenarios/{scenario_id}/simulations")
def create_simulation_from_scenario(
    scenario_id: str,
    payload: ScenarioSimulationCreate,
    session: Session = Depends(get_session),
):
    scenario = get_scenario(scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    request = _scenario_payload_to_simulation_create(scenario, payload)
    sim = _create_simulation_entities(session, request)
    return {"scenario_id": scenario_id, "simulation": sim.model_dump()}


@app.post("/api/v1/simulations/{simulation_id}/start")
async def start_simulation(simulation_id: int, session: Session = Depends(get_session)):
    sim = session.get(Simulation, simulation_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    sim.status = SimulationStatus.running
    session.add(sim)
    session.commit()
    await orchestrator.start(simulation_id, lambda: Session(engine))
    return {"ok": True}


@app.post("/api/v1/simulations/{simulation_id}/pause")
async def pause_simulation(simulation_id: int, session: Session = Depends(get_session)):
    sim = session.get(Simulation, simulation_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    if sim.status == SimulationStatus.paused:
        return {"ok": True, "status": sim.status}
    if sim.status != SimulationStatus.running:
        raise HTTPException(status_code=409, detail="Only running simulations can be paused")
    await orchestrator.stop(simulation_id)
    sim.status = SimulationStatus.paused
    session.add(sim)
    session.commit()
    await ws_manager.broadcast(
        simulation_id,
        {
            "type": "status",
            "status": sim.status,
        },
    )
    return {"ok": True, "status": sim.status}


@app.post("/api/v1/simulations/{simulation_id}/resume")
async def resume_simulation(simulation_id: int, session: Session = Depends(get_session)):
    sim = session.get(Simulation, simulation_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    if sim.status == SimulationStatus.running:
        return {"ok": True, "status": sim.status}
    if sim.status in {SimulationStatus.completed, SimulationStatus.stopped}:
        raise HTTPException(status_code=409, detail=f"Cannot resume simulation in status={sim.status}")
    sim.status = SimulationStatus.running
    session.add(sim)
    session.commit()
    await orchestrator.start(simulation_id, lambda: Session(engine))
    await ws_manager.broadcast(
        simulation_id,
        {
            "type": "status",
            "status": sim.status,
        },
    )
    return {"ok": True, "status": sim.status}


@app.post("/api/v1/simulations/{simulation_id}/stop")
async def stop_simulation(simulation_id: int, session: Session = Depends(get_session)):
    sim = session.get(Simulation, simulation_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    await orchestrator.stop(simulation_id)
    sim.status = SimulationStatus.stopped
    session.add(sim)
    session.commit()
    report, report_event = generate_simulation_report(session, simulation_id, force=False)
    if report_event:
        await ws_manager.broadcast(
            simulation_id,
            {
                "type": "event",
                "event": {
                    "id": report_event.id,
                    "step": report_event.step,
                    "event_type": report_event.event_type,
                    "source_agent": report_event.source_agent,
                    "target_agent": report_event.target_agent,
                    "content": report_event.content,
                },
            },
        )
    return {"ok": True, "report_id": report.id}


@app.post("/api/v1/simulations/{simulation_id}/inject")
async def inject_intervention(
    simulation_id: int,
    payload: InterventionCreate,
    session: Session = Depends(get_session),
):
    step, targets, created, agent_updates = _create_intervention_events(
        simulation_id=simulation_id,
        payload=payload,
        session=session,
    )

    for event in created:
        await ws_manager.broadcast(
            simulation_id,
            {
                "type": "event",
                "event": {
                    "id": event.id,
                    "step": event.step,
                    "event_type": event.event_type,
                    "source_agent": event.source_agent,
                    "target_agent": event.target_agent,
                    "content": event.content,
                },
            },
        )
    for agent_update in agent_updates:
        await ws_manager.broadcast(
            simulation_id,
            {
                "type": "agent_update",
                "agent": agent_update,
            },
        )
    return {
        "ok": True,
        "simulation_id": simulation_id,
        "step": step,
        "targets": targets,
        "created_count": len(created),
    }


@app.get("/api/v1/simulations/{simulation_id}")
def get_simulation(simulation_id: int, session: Session = Depends(get_session)):
    sim = session.get(Simulation, simulation_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    agents = session.exec(select(Agent).where(Agent.simulation_id == simulation_id)).all()
    return {"simulation": sim.model_dump(), "agents": [a.model_dump() for a in agents]}


@app.get("/api/v1/simulations/{simulation_id}/events")
def list_events(simulation_id: int, limit: int = 200, session: Session = Depends(get_session)):
    sim = session.get(Simulation, simulation_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    stmt = select(Event).where(Event.simulation_id == simulation_id).order_by(Event.id.desc()).limit(limit)
    rows = session.exec(stmt).all()
    rows.reverse()
    return [row.model_dump() for row in rows]


@app.get("/api/v1/simulations/{simulation_id}/report")
def get_simulation_report(simulation_id: int, session: Session = Depends(get_session)):
    sim = session.get(Simulation, simulation_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    report = get_latest_report(session, simulation_id)
    if not report and sim.status in {SimulationStatus.stopped, SimulationStatus.completed}:
        report, _ = generate_simulation_report(session, simulation_id, force=False)
    if not report:
        return {"simulation_id": simulation_id, "status": sim.status, "report": None}
    return {
        "simulation_id": simulation_id,
        "status": sim.status,
        "report": _serialize_report(report),
    }


@app.post("/api/v1/simulations/{simulation_id}/report/regenerate")
def regenerate_simulation_report(simulation_id: int, session: Session = Depends(get_session)):
    sim = session.get(Simulation, simulation_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    if sim.status not in {SimulationStatus.stopped, SimulationStatus.completed}:
        raise HTTPException(status_code=409, detail="Report regeneration requires completed/stopped simulation")
    report, _ = generate_simulation_report(session, simulation_id, force=True)
    return {
        "simulation_id": simulation_id,
        "status": sim.status,
        "report": _serialize_report(report),
    }


@app.websocket("/ws/simulations/{simulation_id}")
async def simulation_ws(websocket: WebSocket, simulation_id: int):
    await ws_manager.connect(simulation_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(simulation_id, websocket)


def _create_simulation_entities(session: Session, payload: SimulationCreate) -> Simulation:
    sim = Simulation(
        goal=payload.goal,
        status=SimulationStatus.created,
        max_steps=payload.max_steps,
        tick_interval_ms=payload.tick_interval_ms,
    )
    session.add(sim)
    session.commit()
    session.refresh(sim)
    for agent_payload in payload.agents:
        session.add(
            Agent(
                simulation_id=sim.id,
                name=agent_payload.name,
                role=agent_payload.role,
                memory={"history": []},
            )
        )
    session.commit()
    session.refresh(sim)
    return sim


def _scenario_payload_to_simulation_create(
    scenario: dict[str, Any],
    payload: ScenarioSimulationCreate,
) -> SimulationCreate:
    return SimulationCreate(
        goal=scenario["goal"],
        max_steps=payload.max_steps if payload.max_steps is not None else scenario["max_steps"],
        tick_interval_ms=(
            payload.tick_interval_ms if payload.tick_interval_ms is not None else scenario["tick_interval_ms"]
        ),
        agents=scenario["agents"],
    )


def _create_intervention_events(
    simulation_id: int,
    payload: InterventionCreate,
    session: Session,
) -> tuple[int, list[str], list[Event], list[dict[str, Any]]]:
    sim = session.get(Simulation, simulation_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    if sim.status in {SimulationStatus.completed, SimulationStatus.stopped}:
        raise HTTPException(status_code=409, detail=f"Cannot inject into simulation in status={sim.status}")

    agents = session.exec(select(Agent).where(Agent.simulation_id == simulation_id)).all()
    if not agents:
        raise HTTPException(status_code=400, detail="No agents in simulation")
    by_name = {agent.name: agent for agent in agents}
    targets = [agent.name for agent in agents]

    created: list[Event] = []
    agent_updates: list[dict[str, Any]] = []
    for target in targets:
        agent = by_name[target]
        event = Event(
            simulation_id=simulation_id,
            step=sim.step,
            event_type="intervention",
            source_agent="system",
            target_agent=target,
            content=f"{payload.content} @ {utc_iso_now()}",
        )
        session.add(event)
        created.append(event)
        _append_memory(agent, [f"intervention:{payload.content}"])
        session.add(agent)
        history = _memory_history(agent.memory)
        agent_updates.append(
            {
                "id": agent.id,
                "name": agent.name,
                "role": agent.role,
                "memory_size": len(history),
                "last_memory": history[-1] if history else "",
            }
        )

    session.commit()
    for event in created:
        session.refresh(event)
    return sim.step, targets, created, agent_updates


def _serialize_report(report: SimulationReport) -> dict[str, Any]:
    return {
        "id": report.id,
        "simulation_id": report.simulation_id,
        "version": report.version,
        "generator": report.generator,
        "title": report.title,
        "markdown": report.markdown,
        "report_json": report.report_json,
        "created_at": report.created_at,
    }


def _memory_history(memory: dict[str, Any]) -> list[str]:
    if not isinstance(memory, dict):
        return []
    history = memory.get("history", [])
    return history if isinstance(history, list) else []


def _append_memory(agent: Agent, entries: list[str]) -> None:
    existing = agent.memory if isinstance(agent.memory, dict) else {}
    memory: dict[str, Any] = {**existing}
    existing_history = memory.get("history", [])
    history = list(existing_history) if isinstance(existing_history, list) else []
    history.extend(entries)
    memory["history"] = history[-50:]
    agent.memory = memory
