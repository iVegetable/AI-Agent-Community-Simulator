#!/usr/bin/env python
"""Matrix benchmark runner for resource-scenario realism exploration.

Sweeps mechanism/transparency/punishment combinations and prints the
resulting realism score, grade, and selected operational signals.
"""

import os
import time
from dataclasses import dataclass

from fastapi.testclient import TestClient


@dataclass
class CaseResult:
    mechanism: str
    transparency: str
    punishment: bool
    max_steps: int
    score: float
    grade: str
    passed: bool
    conflict_rounds: int
    strike_rounds: int
    total_output: float


def _run_one(
    client: TestClient,
    mechanism: str,
    transparency: str,
    punishment: bool,
    max_steps: int,
) -> CaseResult:
    created = client.post(
        "/api/v1/scenarios/three-agent-resource-allocation/simulations",
        json={"max_steps": max_steps, "tick_interval_ms": 20},
    )
    created.raise_for_status()
    simulation_id = int(created.json()["simulation"]["id"])

    intervention = (
        f"Use {mechanism} mechanism with {transparency} transparency and "
        f"{'punishment enabled' if punishment else 'no punishment'}. "
        "pressure_regime=off."
    )
    inject = client.post(
        f"/api/v1/simulations/{simulation_id}/inject",
        json={"content": intervention},
    )
    inject.raise_for_status()
    started = client.post(f"/api/v1/simulations/{simulation_id}/start")
    started.raise_for_status()

    start = time.time()
    while time.time() - start < 15:
        state = client.get(f"/api/v1/simulations/{simulation_id}")
        state.raise_for_status()
        status = str(state.json()["simulation"]["status"])
        if status in {"completed", "stopped"}:
            break
        time.sleep(0.03)

    client.post(f"/api/v1/simulations/{simulation_id}/stop")
    report_resp = client.get(f"/api/v1/simulations/{simulation_id}/report")
    report_resp.raise_for_status()
    report_body = report_resp.json().get("report") or {}
    report_json = report_body.get("report_json", {})
    report = report_json.get("resource_allocation") or {}
    realism = report.get("realism") or {}
    return CaseResult(
        mechanism=mechanism,
        transparency=transparency,
        punishment=punishment,
        max_steps=max_steps,
        score=float(realism.get("score", 0.0)),
        grade=str(realism.get("grade", "unknown")),
        passed=bool(realism.get("standard_passed", False)),
        conflict_rounds=int(report.get("conflict_rounds", 0)),
        strike_rounds=int(report.get("strike_rounds", 0)),
        total_output=float(report.get("total_output", 0.0)),
    )


def main() -> None:
    # You can override this from shell: AI_MODE=openai python scripts/realism_benchmark.py
    ai_mode = os.getenv("AI_MODE", "mock").strip().lower() or "mock"
    if ai_mode == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("[benchmark] OPENAI_API_KEY missing; fallback to AI_MODE=mock")
        ai_mode = "mock"
    os.environ["AI_MODE"] = ai_mode
    from app.main import app  # noqa: WPS433

    print(f"[benchmark] AI_MODE={ai_mode}")
    mechanisms = ["equal", "contribution", "dictator", "majority_vote"]
    transparencies = ["full", "hidden"]
    punishments = [False, True]
    steps_list = [5, 8]

    results: list[CaseResult] = []
    with TestClient(app) as client:
        total_cases = len(mechanisms) * len(transparencies) * len(punishments) * len(steps_list)
        idx = 0
        for mechanism in mechanisms:
            for transparency in transparencies:
                for punishment in punishments:
                    for max_steps in steps_list:
                        idx += 1
                        result = _run_one(
                            client=client,
                            mechanism=mechanism,
                            transparency=transparency,
                            punishment=punishment,
                            max_steps=max_steps,
                        )
                        results.append(result)
                        print(
                            f"[{idx:02d}/{total_cases}] "
                            f"{mechanism}/{transparency}/p{int(punishment)}/s{max_steps} -> "
                            f"score={result.score:.2f}, grade={result.grade}, pass={result.passed}"
                        )

    results.sort(key=lambda item: item.score, reverse=True)
    passed = sum(1 for item in results if item.passed)
    high = sum(1 for item in results if item.grade == "high_realism")
    print("\n=== TOP 8 ===")
    for item in results[:8]:
        print(
            f"{item.mechanism}/{item.transparency}/p{int(item.punishment)}/s{item.max_steps} "
            f"score={item.score:.2f} grade={item.grade} pass={item.passed} "
            f"conflict={item.conflict_rounds} strike={item.strike_rounds} total_output={item.total_output:.2f}"
        )
    print(f"\nPass count: {passed}/{len(results)}")
    print(f"High realism count: {high}/{len(results)}")


if __name__ == "__main__":
    main()
