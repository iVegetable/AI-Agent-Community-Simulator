#!/usr/bin/env python
"""Acceptance gate for resource-scenario realism standards.

Runs pressure and baseline cases, computes pass ratios against fixed
thresholds, and exits non-zero when acceptance criteria are not met.
"""

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import SQLModel


PRESSURE_INTERVENTION = (
    "Use contribution mechanism with hidden transparency and punishment enabled. "
    "pressure_regime=on."
)
BASELINE_INTERVENTION = (
    "Use equal mechanism with full transparency and no punishment. "
    "pressure_regime=off."
)


@dataclass
class RunResult:
    simulation_id: int
    score: float
    grade: str
    standard_passed: bool
    causal_consistency_rate: float
    mean_output: float
    conflict_rounds: int
    institution_score: float


def _ensure_database_url() -> None:
    if os.getenv("DATABASE_URL"):
        return
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "backend" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_file = (data_dir / "dev.db").resolve()
    os.environ["DATABASE_URL"] = f"sqlite:///{db_file.as_posix()}"


def _run_one_case(
    client: TestClient,
    *,
    intervention: str,
    max_steps: int,
    max_wait_seconds: float,
) -> RunResult:
    created = client.post(
        "/api/v1/scenarios/three-agent-resource-allocation/simulations",
        json={"max_steps": max_steps, "tick_interval_ms": 20},
    )
    created.raise_for_status()
    simulation_id = int(created.json()["simulation"]["id"])

    injected = client.post(
        f"/api/v1/simulations/{simulation_id}/inject",
        json={"content": intervention},
    )
    injected.raise_for_status()
    started = client.post(f"/api/v1/simulations/{simulation_id}/start")
    started.raise_for_status()

    deadline = time.time() + max_wait_seconds
    while time.time() < deadline:
        state = client.get(f"/api/v1/simulations/{simulation_id}")
        state.raise_for_status()
        status = str(state.json()["simulation"]["status"])
        if status in {"completed", "stopped"}:
            break
        time.sleep(0.03)

    client.post(f"/api/v1/simulations/{simulation_id}/stop")
    report_resp = client.get(f"/api/v1/simulations/{simulation_id}/report")
    report_resp.raise_for_status()
    report_json = ((report_resp.json().get("report") or {}).get("report_json") or {})
    resource = report_json.get("resource_allocation") or {}
    realism = resource.get("realism") or {}

    return RunResult(
        simulation_id=simulation_id,
        score=float(realism.get("score", 0.0)),
        grade=str(realism.get("grade", "unknown")),
        standard_passed=bool(realism.get("standard_passed", False)),
        causal_consistency_rate=float(resource.get("causal_consistency_rate", 0.0)),
        mean_output=float(resource.get("mean_output", 0.0)),
        conflict_rounds=int(resource.get("conflict_rounds", 0)),
        institution_score=float(resource.get("institutional_sensitivity_score", 0.0)),
    )


def _safe_cv(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_val = sum(values) / len(values)
    if abs(mean_val) <= 1e-8:
        return 0.0
    return statistics.pstdev(values) / mean_val


def _summarize(rows: list[RunResult]) -> dict[str, float]:
    if not rows:
        return {
            "count": 0.0,
            "mean_score": 0.0,
            "high_ratio": 0.0,
            "low_ratio": 0.0,
            "standard_passed_ratio": 0.0,
            "causal_mean": 0.0,
            "causal_pass_ratio": 0.0,
            "score_cv": 0.0,
            "output_cv": 0.0,
            "institution_mean": 0.0,
        }
    n = len(rows)
    scores = [item.score for item in rows]
    outputs = [item.mean_output for item in rows]
    causals = [item.causal_consistency_rate for item in rows]
    return {
        "count": float(n),
        "mean_score": sum(scores) / n,
        "high_ratio": sum(1 for item in rows if item.grade == "high_realism") / n,
        "low_ratio": sum(1 for item in rows if item.grade == "low_realism") / n,
        "standard_passed_ratio": sum(1 for item in rows if item.standard_passed) / n,
        "causal_mean": sum(causals) / n,
        "causal_pass_ratio": sum(1 for item in rows if item.causal_consistency_rate >= 0.85) / n,
        "score_cv": _safe_cv(scores),
        "output_cv": _safe_cv(outputs),
        "institution_mean": sum(item.institution_score for item in rows) / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run realism acceptance checks for resource scenario.")
    parser.add_argument("--pressure-runs", type=int, default=20)
    parser.add_argument("--baseline-runs", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-wait-seconds", type=float, default=20.0)
    args = parser.parse_args()

    _ensure_database_url()
    repo_root = Path(__file__).resolve().parent.parent
    backend_path = str(repo_root / "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    ai_mode = (os.getenv("AI_MODE") or "mock").strip().lower() or "mock"
    os.environ["AI_MODE"] = ai_mode
    effective_wait_seconds = args.max_wait_seconds
    if ai_mode == "openai" and effective_wait_seconds < 90.0:
        effective_wait_seconds = 90.0

    from app import models  # noqa: F401
    from app.db import engine
    from app.main import app
    from app.config import settings

    settings.ai_mode = ai_mode
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)

    print(f"[acceptance] AI_MODE={ai_mode}")
    print(
        "[acceptance] "
        f"pressure_runs={args.pressure_runs}, baseline_runs={args.baseline_runs}, "
        f"max_steps={args.max_steps}, max_wait_seconds={effective_wait_seconds}"
    )

    pressure_rows: list[RunResult] = []
    baseline_rows: list[RunResult] = []
    with TestClient(app) as client:
        for idx in range(1, args.pressure_runs + 1):
            result = _run_one_case(
                client,
                intervention=PRESSURE_INTERVENTION,
                max_steps=args.max_steps,
                max_wait_seconds=effective_wait_seconds,
            )
            pressure_rows.append(result)
            print(
                f"[pressure {idx:02d}/{args.pressure_runs}] "
                f"id={result.simulation_id} score={result.score:.2f} grade={result.grade} "
                f"pass={result.standard_passed} causal={result.causal_consistency_rate:.3f}"
            )
        for idx in range(1, args.baseline_runs + 1):
            result = _run_one_case(
                client,
                intervention=BASELINE_INTERVENTION,
                max_steps=args.max_steps,
                max_wait_seconds=effective_wait_seconds,
            )
            baseline_rows.append(result)
            print(
                f"[baseline {idx:02d}/{args.baseline_runs}] "
                f"id={result.simulation_id} score={result.score:.2f} grade={result.grade} "
                f"pass={result.standard_passed} causal={result.causal_consistency_rate:.3f}"
            )

    pressure = _summarize(pressure_rows)
    baseline = _summarize(baseline_rows)

    checks = {
        "pressure_high_realism_ratio>=0.60": pressure["high_ratio"] >= 0.60,
        "pressure_standard_passed_ratio>=0.75": pressure["standard_passed_ratio"] >= 0.75,
        "message_number_result_consistency>=0.85": pressure["causal_pass_ratio"] >= 0.85,
        "baseline_low_or_low_score": (
            baseline["low_ratio"] >= 0.50
            or baseline["mean_score"] <= 60.0
        ),
        "variance_controlled_same_params": (
            pressure["score_cv"] <= 0.12
            and pressure["output_cv"] <= 0.35
        ),
    }
    all_passed = all(checks.values())

    print("\n=== SUMMARY ===")
    print(
        "Pressure: "
        f"high_ratio={pressure['high_ratio']:.3f}, "
        f"standard_passed_ratio={pressure['standard_passed_ratio']:.3f}, "
        f"causal_pass_ratio={pressure['causal_pass_ratio']:.3f}, "
        f"mean_score={pressure['mean_score']:.2f}, "
        f"score_cv={pressure['score_cv']:.3f}, "
        f"output_cv={pressure['output_cv']:.3f}"
    )
    print(
        "Baseline: "
        f"low_ratio={baseline['low_ratio']:.3f}, "
        f"mean_score={baseline['mean_score']:.2f}, "
        f"high_ratio={baseline['high_ratio']:.3f}"
    )
    print("Checks:")
    for name, passed in checks.items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Acceptance: {'PASS' if all_passed else 'FAIL'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
