from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_create_and_start_stop():
    payload = {
        "goal": "test",
        "max_steps": 3,
        "tick_interval_ms": 50,
        "agents": [{"name": "P", "role": "planner"}],
    }
    r = client.post("/api/v1/simulations", json=payload)
    assert r.status_code == 200
    sim_id = r.json()["simulation"]["id"]

    r2 = client.post(f"/api/v1/simulations/{sim_id}/start")
    assert r2.status_code == 200

    r3 = client.post(f"/api/v1/simulations/{sim_id}/stop")
    assert r3.status_code == 200
