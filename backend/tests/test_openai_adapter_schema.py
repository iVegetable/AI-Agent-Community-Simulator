"""Schema completeness tests for strict OpenAI resource output format."""

from types import SimpleNamespace

from app.agents.adapters.openai_adapter import OpenAIDecisionAdapter
from app.agents.adapters.base import DecisionContext


def test_resource_schema_required_covers_all_properties_for_strict_mode():
    adapter = OpenAIDecisionAdapter.__new__(OpenAIDecisionAdapter)
    schema = adapter._resource_schema()

    assert schema["type"] == "json_schema"
    assert schema["schema"]["type"] == "object"

    root = schema["schema"]
    root_props = root["properties"]
    root_required = set(root["required"])
    assert set(root_props.keys()) == root_required

    resource = root_props["resource_proposal"]
    assert resource["type"] == "object"
    resource_props = resource["properties"]
    resource_required = set(resource["required"])
    assert set(resource_props.keys()) == resource_required


def test_decide_retries_on_invalid_json(monkeypatch):
    adapter = OpenAIDecisionAdapter.__new__(OpenAIDecisionAdapter)
    responses = [
        SimpleNamespace(output_text="{not-json"),
        SimpleNamespace(
            output_text='{"event_type":"message","content":"ok","target_agent":"B","memory_append":["m1"]}'
        ),
    ]
    ctx = DecisionContext(
        goal="regular simulation goal",
        step=1,
        agent_name="A",
        agent_role="planner",
        peers=["B"],
        memory_history=[],
        inbox=[],
    )

    async def fake_request(prompt, *, resource_mode):
        if not hasattr(fake_request, "prompts"):
            fake_request.prompts = []  # type: ignore[attr-defined]
        fake_request.prompts.append(prompt)  # type: ignore[attr-defined]
        assert resource_mode is False
        return responses.pop(0)

    monkeypatch.setattr(adapter, "_request_with_retry", fake_request)

    import asyncio

    decision = asyncio.run(adapter.decide(ctx))
    assert decision.event_type == "message"
    assert decision.target_agent == "B"
    assert decision.content == "ok"
    assert len(fake_request.prompts) == 2  # type: ignore[attr-defined]
    assert "Previous response failed validation" in fake_request.prompts[1]  # type: ignore[attr-defined]
