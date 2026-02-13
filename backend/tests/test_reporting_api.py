"""Report generation and regeneration API tests."""

import time

from fastapi.testclient import TestClient

from app.main import app


def test_report_generated_after_stop():
    payload = {"max_steps": 4, "tick_interval_ms": 40}
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/scenarios/three-agent-resource-allocation/simulations",
            json=payload,
        )
        sim_id = created.json()["simulation"]["id"]
        client.post(f"/api/v1/simulations/{sim_id}/start")
        time.sleep(0.2)
        stopped = client.post(f"/api/v1/simulations/{sim_id}/stop")
        assert stopped.status_code == 200
        assert stopped.json()["report_id"] is not None

        report_resp = client.get(f"/api/v1/simulations/{sim_id}/report")
        assert report_resp.status_code == 200
        report = report_resp.json()["report"]
        assert report is not None
        assert report["title"].startswith("Simulation")
        assert "## Executive Summary" in report["markdown"]

        events = client.get(f"/api/v1/simulations/{sim_id}/events?limit=200").json()
        assert any(event["event_type"] == "report" for event in events)


def test_report_regenerate_increments_version():
    payload = {"max_steps": 2, "tick_interval_ms": 40}
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/scenarios/three-agent-resource-allocation/simulations",
            json=payload,
        )
        sim_id = created.json()["simulation"]["id"]
        client.post(f"/api/v1/simulations/{sim_id}/start")
        time.sleep(0.1)
        client.post(f"/api/v1/simulations/{sim_id}/stop")

        first = client.get(f"/api/v1/simulations/{sim_id}/report").json()["report"]
        regen = client.post(f"/api/v1/simulations/{sim_id}/report/regenerate")
        assert regen.status_code == 200
        second = regen.json()["report"]
        assert second["version"] == first["version"] + 1


def test_resource_scenario_report_includes_allocation_conclusion():
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/scenarios/three-agent-resource-allocation/simulations",
            json={"max_steps": 4, "tick_interval_ms": 30},
        )
        assert created.status_code == 200
        sim_id = created.json()["simulation"]["id"]

        started = client.post(f"/api/v1/simulations/{sim_id}/start")
        assert started.status_code == 200
        time.sleep(0.5)
        client.post(f"/api/v1/simulations/{sim_id}/stop")

        events = client.get(f"/api/v1/simulations/{sim_id}/events?limit=400").json()
        assert any(event["source_agent"] == "A" and event["event_type"] == "message" for event in events)
        assert any(event["source_agent"] == "B" and event["event_type"] == "message" for event in events)
        assert any(event["source_agent"] == "C" and event["event_type"] == "message" for event in events)
        assert any(
            "RESOURCE_ROUND_REPORT" in event["content"]
            for event in events
        )
        assert any(
            "RESOURCE_FINAL_RECOMMENDATION" in event["content"]
            for event in events
        )

        report_resp = client.get(f"/api/v1/simulations/{sim_id}/report")
        assert report_resp.status_code == 200
        report = report_resp.json()["report"]
        assert report is not None
        assert "## Resource Allocation Conclusion" in report["markdown"]
        report_json = report["report_json"]
        assert "resource_allocation" in report_json
        assert report_json["resource_allocation"]["round_count"] >= 1
        assert "final_round_willingness" in report_json["resource_allocation"]
        realism = report_json["resource_allocation"].get("realism", {})
        assert "standard_passed" in realism
        assert "standard_threshold" in realism
        assert "components" in realism
