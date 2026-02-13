"""Smoke tests for frontend-exposed API liveness."""

from fastapi.testclient import TestClient

from app.main import app


def test_frontend_exposed_api_smoke():
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/scenarios/three-agent-resource-allocation/simulations",
            json={"max_steps": 2, "tick_interval_ms": 20},
        )
        assert created.status_code == 200
        simulation_id = int(created.json()["simulation"]["id"])
        state = client.get(f"/api/v1/simulations/{simulation_id}")
        assert state.status_code == 200
