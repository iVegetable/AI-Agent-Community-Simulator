"""Tests for pause/inject/resume flow and intervention validation."""

import time

from fastapi.testclient import TestClient

from app.main import app


def test_pause_inject_resume_flow():
    payload = {"max_steps": 8, "tick_interval_ms": 40}
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/scenarios/three-agent-resource-allocation/simulations",
            json=payload,
        )
        assert created.status_code == 200
        sim_id = created.json()["simulation"]["id"]

        started = client.post(f"/api/v1/simulations/{sim_id}/start")
        assert started.status_code == 200
        time.sleep(0.15)

        paused = client.post(f"/api/v1/simulations/{sim_id}/pause")
        assert paused.status_code == 200
        assert paused.json()["status"] == "paused"

        injected = client.post(
            f"/api/v1/simulations/{sim_id}/inject",
            json={"content": "Focus on fairness constraints"},
        )
        assert injected.status_code == 200
        body = injected.json()
        assert body["created_count"] == 3
        assert body["targets"] == ["A", "B", "C"]

        events = client.get(f"/api/v1/simulations/{sim_id}/events?limit=50")
        assert events.status_code == 200
        intervention_events = [event for event in events.json() if event["event_type"] == "intervention"]
        assert len(intervention_events) >= 3
        targets = {event["target_agent"] for event in intervention_events[-3:]}
        assert targets == {"A", "B", "C"}

        resumed = client.post(f"/api/v1/simulations/{sim_id}/resume")
        assert resumed.status_code == 200
        assert resumed.json()["status"] == "running"
        time.sleep(0.2)

        simulation = client.get(f"/api/v1/simulations/{sim_id}")
        assert simulation.status_code == 200
        assert simulation.json()["simulation"]["step"] >= 1

        stopped = client.post(f"/api/v1/simulations/{sim_id}/stop")
        assert stopped.status_code == 200


def test_inject_stopped_simulation_returns_409():
    payload = {"max_steps": 3, "tick_interval_ms": 50}
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/scenarios/three-agent-resource-allocation/simulations",
            json=payload,
        )
        sim_id = created.json()["simulation"]["id"]
        stopped = client.post(f"/api/v1/simulations/{sim_id}/stop")
        assert stopped.status_code == 200
        injected = client.post(
            f"/api/v1/simulations/{sim_id}/inject",
            json={"content": "hello"},
        )
        assert injected.status_code == 409
