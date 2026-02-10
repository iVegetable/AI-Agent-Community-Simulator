from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

from .config import settings
from .db import engine, init_db
from .messaging import ConnectionManager
from .models import Agent, Simulation, SimulationStatus
from .orchestrator import Orchestrator
from .schemas import SimulationCreate

app = FastAPI(title="AI Agent Community Simulator")

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


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/simulations")
def create_simulation(payload: SimulationCreate, session: Session = Depends(get_session)):
    sim = Simulation(
        goal=payload.goal,
        status=SimulationStatus.created,
        max_steps=payload.max_steps,
        tick_interval_ms=payload.tick_interval_ms,
    )
    session.add(sim)
    session.commit()
    session.refresh(sim)

    for a in payload.agents:
        session.add(Agent(simulation_id=sim.id, name=a.name, role=a.role, memory={"history": []}))
    session.commit()

    return {"simulation": sim.model_dump()}


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


@app.post("/api/v1/simulations/{simulation_id}/stop")
async def stop_simulation(simulation_id: int, session: Session = Depends(get_session)):
    sim = session.get(Simulation, simulation_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    sim.status = SimulationStatus.stopped
    session.add(sim)
    session.commit()
    await orchestrator.stop(simulation_id)
    return {"ok": True}


@app.get("/api/v1/simulations/{simulation_id}")
def get_simulation(simulation_id: int, session: Session = Depends(get_session)):
    sim = session.get(Simulation, simulation_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    agents = session.exec(select(Agent).where(Agent.simulation_id == simulation_id)).all()
    return {"simulation": sim.model_dump(), "agents": [a.model_dump() for a in agents]}


@app.websocket("/ws/simulations/{simulation_id}")
async def simulation_ws(websocket: WebSocket, simulation_id: int):
    await ws_manager.connect(simulation_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(simulation_id, websocket)
