"""End-to-end regression test for resource scenario without fatal stop events."""

import time

from fastapi.testclient import TestClient

from app.main import app


def test_resource_scenario_runs_to_completion_without_fatal():
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/scenarios/three-agent-resource-allocation/simulations",
            json={"max_steps": 6, "tick_interval_ms": 30},
        )
        assert created.status_code == 200
        simulation_id = int(created.json()["simulation"]["id"])

        injected = client.post(
            f"/api/v1/simulations/{simulation_id}/inject",
            json={"content": "Use contribution mechanism with hidden transparency and punishment enabled."},
        )
        assert injected.status_code == 200

        started = client.post(f"/api/v1/simulations/{simulation_id}/start")
        assert started.status_code == 200

        deadline = time.time() + 6.0
        status = "running"
        while time.time() < deadline:
            state_resp = client.get(f"/api/v1/simulations/{simulation_id}")
            assert state_resp.status_code == 200
            status = str(state_resp.json()["simulation"]["status"])
            if status in {"completed", "stopped"}:
                break
            time.sleep(0.05)
        assert status == "completed"

        events_resp = client.get(f"/api/v1/simulations/{simulation_id}/events?limit=800")
        assert events_resp.status_code == 200
        events = events_resp.json()
        assert len(events) > 0

        assert not any("RESOURCE_SCENARIO_FATAL" in str(event.get("content", "")) for event in events)
        assert any("RESOURCE_ROUND_REPORT" in str(event.get("content", "")) for event in events)
        assert any("RESOURCE_FINAL_RECOMMENDATION" in str(event.get("content", "")) for event in events)

        report_resp = client.get(f"/api/v1/simulations/{simulation_id}/report")
        assert report_resp.status_code == 200
        report = report_resp.json()["report"]
        assert report is not None
        report_json = report.get("report_json", {})
        resource = report_json.get("resource_allocation", {})
        assert int(resource.get("round_count", 0)) >= 1
        assert "realism" in resource

