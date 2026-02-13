"""Regression tests for removed non-frontend quantitative experiment APIs."""

from fastapi.testclient import TestClient

from app.main import app


def test_quant_resource_experiment_endpoint_removed():
    payload = {
        "runs": 2,
        "target_production": 100,
        "baseline": {
            "mechanism": "equal",
            "transparency": "full",
            "punishment": False,
            "learning": "adaptive",
            "rounds": 10,
        },
        "variant": {
            "mechanism": "contribution",
            "transparency": "full",
            "punishment": True,
            "learning": "adaptive",
            "rounds": 10,
        },
    }
    with TestClient(app) as client:
        response = client.post("/api/v1/experiments/resource-allocation/compare", json=payload)
        assert response.status_code == 404
