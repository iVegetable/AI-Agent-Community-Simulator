"""Built-in scenario template used by the current frontend simulation flow."""

from typing import Any


SCENARIO_LIBRARY: list[dict[str, Any]] = [
    {
        "id": "three-agent-resource-allocation",
        "title": "Three-Agent Resource Allocation Experiment",
        "description": (
            "Integrated scenario run where A/B/C debate resource allocation every round and "
            "an observer agent records quantitative evidence and final recommendation."
        ),
        "goal": (
            "Simulate three heterogeneous agents (A efficiency-focused, B fairness-sensitive, "
            "C self-interested) debating effort and payoff under allocation mechanism tradeoffs, "
            "and conclude the final willingness/allocation agreement with quantitative evidence."
        ),
        "max_steps": 8,
        "tick_interval_ms": 500,
        "stress_max_steps": 14,
        "agents": [
            {"name": "A", "role": "high-efficiency"},
            {"name": "B", "role": "fairness-sensitive"},
            {"name": "C", "role": "self-interested"},
        ],
        "suggested_interventions": [
            "Mechanism M1: Equal distribution (everyone gets production/3).",
            "Mechanism M2: Contribution-based distribution (weighted by effort x efficiency).",
            "Transparency condition: hidden contributions vs full transparency.",
            "Enable punishment channel after unfair allocation.",
        ],
        "expected_hypotheses": [
            "H1: Contribution-based mechanism increases total production.",
            "H2: Equal distribution improves short-term stability but reduces peak output.",
            "H3: Hidden contributions increase conflict probability.",
        ],
        "demo_protocol": {
            "recommended_runs": 5,
            "baseline_label": "equal_distribution",
            "variant_label": "contribution_based",
            "baseline_interventions": [
                {"content": "Use equal mechanism with full transparency and no punishment."}
            ],
            "variant_interventions": [
                {"content": "Use contribution mechanism with full transparency and punishment enabled."}
            ],
        },
    },
]


def get_scenario(scenario_id: str) -> dict[str, Any] | None:
    for item in SCENARIO_LIBRARY:
        if item["id"] == scenario_id:
            return dict(item)
    return None
