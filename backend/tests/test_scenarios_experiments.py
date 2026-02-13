"""Contract tests for retained/removed API surface after frontend-only trimming."""

from fastapi.testclient import TestClient

from app.main import app


def test_create_simulation_from_frontend_scenario():
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/scenarios/three-agent-resource-allocation/simulations",
            json={"max_steps": 3, "tick_interval_ms": 30},
        )
        assert created.status_code == 200
        sim_id = int(created.json()["simulation"]["id"])

        state = client.get(f"/api/v1/simulations/{sim_id}")
        assert state.status_code == 200
        assert len(state.json()["agents"]) == 3


def test_removed_non_frontend_routes_return_404():
    with TestClient(app) as client:
        assert client.get("/health").status_code == 404
        assert client.post("/api/v1/simulations", json={}).status_code == 404
        assert client.get("/api/v1/scenarios").status_code == 404
        assert client.get("/api/v1/scenarios/three-agent-resource-allocation").status_code == 404
        assert client.get("/api/v1/simulations/1/summary").status_code == 404
        assert client.post("/api/v1/experiments/compare", json={}).status_code == 404
        assert client.post("/api/v1/experiments/compare/export", json={}).status_code == 404
        assert client.post("/api/v1/experiments/batch-compare", json={}).status_code == 404
        assert client.post("/api/v1/experiments/resource-allocation/compare", json={}).status_code == 404
