"""Core frontend-exposed simulation API flow tests."""

from fastapi.testclient import TestClient
from app.main import app


def test_create_and_start_stop():
    payload = {"max_steps": 3, "tick_interval_ms": 50}
    with TestClient(app) as client:
        r = client.post(
            "/api/v1/scenarios/three-agent-resource-allocation/simulations",
            json=payload,
        )
        assert r.status_code == 200
        sim_id = r.json()["simulation"]["id"]
        assert sim_id is not None

        r2 = client.post(f"/api/v1/simulations/{sim_id}/start")
        assert r2.status_code == 200

        r3 = client.post(f"/api/v1/simulations/{sim_id}/stop")
        assert r3.status_code == 200
        assert r3.json()["report_id"] is not None

        r4 = client.get(f"/api/v1/simulations/{sim_id}/events?limit=20")
        assert r4.status_code == 200
        events = r4.json()
        assert isinstance(events, list)
        assert len(events) >= 1

        report_resp = client.get(f"/api/v1/simulations/{sim_id}/report")
        assert report_resp.status_code == 200
        report = report_resp.json()["report"]
        assert report is not None
        assert "## Executive Summary" in report["markdown"]
