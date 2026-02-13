"""Deterministic adapter for local development and repeatable tests.

Generates structured proposals without external API calls, including
resource-scenario markers consumed by the orchestrator.
"""

import json
import re

from .base import AgentDecision, DecisionAdapter, DecisionContext


class MockDecisionAdapter(DecisionAdapter):
    """Simple deterministic policy with scenario-aware numeric outputs."""

    async def decide(self, ctx: DecisionContext) -> AgentDecision:
        if "RESOURCE_SCENARIO" in ctx.goal:
            token = str(ctx.agent_name).strip().upper()
            hidden_mode = "transparency: hidden" in ctx.goal.lower()
            punishment_enabled = "punishment enabled: true" in ctx.goal.lower()
            forced_strike = "forced strike this round: true" in ctx.goal.lower()

            last_conflict = False
            target_gap = 0.0
            last_summary = re.search(
                r"Last round summary:\s*output=([0-9]+(?:\.[0-9]+)?),\s*conflict=(true|false),\s*gap=([-+]?[0-9]+(?:\.[0-9]+)?)",
                ctx.goal,
                re.IGNORECASE,
            )
            if last_summary:
                last_conflict = last_summary.group(2).strip().lower() == "true"
                target_gap = float(last_summary.group(3))

            own_prev = re.search(
                r"Your previous proposal:\s*effort=([0-9]+(?:\.[0-9]+)?),\s*willingness=([0-9]+(?:\.[0-9]+)?)",
                ctx.goal,
                re.IGNORECASE,
            )

            base_effort = {"A": 8.3, "B": 6.3, "C": 5.4}.get(token, 5.0)
            conflict_penalty = {"A": 0.7, "B": 1.0, "C": 0.5}.get(token, 0.7)
            gap_push = {"A": 0.75, "B": 0.35, "C": 0.20}.get(token, 0.25)

            effort = base_effort - (ctx.step - 1) * 0.10
            if last_conflict:
                effort -= conflict_penalty
            if target_gap > 0:
                effort += min(2.2, (target_gap / 20.0) * gap_push)
            if hidden_mode and token == "B":
                effort -= 0.35
            if punishment_enabled and last_conflict and token == "C":
                effort += 0.25
            if forced_strike:
                effort = 0.0
            effort = max(0.0, min(10.0, effort))

            willingness = effort * 8.8 + {"A": 8.0, "B": 4.0, "C": 6.0}.get(token, 5.0)
            if hidden_mode and token == "B":
                willingness -= 6.0
            if last_conflict and token == "C":
                willingness += 5.0
            if forced_strike:
                willingness = min(willingness, 20.0)
            willingness = max(0.0, min(100.0, willingness))

            previous_effort = effort
            previous_willingness = willingness
            if own_prev:
                previous_effort = max(0.0, min(10.0, float(own_prev.group(1))))
                previous_willingness = max(0.0, min(100.0, float(own_prev.group(2))))
            elif ctx.step > 1:
                previous_effort = max(0.0, min(10.0, base_effort - (ctx.step - 2) * 0.10))
                previous_willingness = max(0.0, min(100.0, previous_effort * 8.8))

            if token == "A":
                vote = "contribution"
            elif token == "B":
                vote = "contribution" if (target_gap > 18 and not last_conflict and not hidden_mode) else "equal"
            else:
                vote = "contribution" if punishment_enabled or not hidden_mode else "equal"

            base_allocations = {
                "A": {"A": 50.0, "B": 30.0, "C": 20.0},
                "B": {"A": 34.0, "B": 33.0, "C": 33.0},
                "C": {"A": 22.0, "B": 20.0, "C": 58.0},
            }
            proposal = dict(base_allocations.get(token, {"A": 34.0, "B": 33.0, "C": 33.0}))
            influenced_by = "NONE"
            adjustment_reason = "Initial round baseline proposal."
            peer_reference = {
                "agent": "none",
                "quoted_message_snippet": "initial round no prior peer quote",
                "referenced_effort": 0.0,
                "referenced_willingness_units": 0.0,
            }
            peer_map: dict[str, dict[str, float | str]] = {}
            for m in re.finditer(
                r"([ABC]):\s*effort=([0-9]+(?:\.[0-9]+)?),\s*willingness=([0-9]+(?:\.[0-9]+)?),\s*msg=([^|]+)",
                ctx.goal,
                re.IGNORECASE,
            ):
                peer_map[m.group(1).upper()] = {
                    "effort": float(m.group(2)),
                    "willingness": float(m.group(3)),
                    "msg": str(m.group(4)).strip(),
                }
            if ctx.step >= 2:
                influenced_by = {"A": "B", "B": "C", "C": "A"}.get(token, "A")
                adjustment_reason = (
                    f"Adjusted effort/willingness after reading {influenced_by} message and round signal."
                )
                peer_state = peer_map.get(influenced_by, {})
                peer_reference = {
                    "agent": influenced_by,
                    "quoted_message_snippet": str(
                        peer_state.get("msg", f"{influenced_by} proposed effort and willingness last round.")
                    )[:180],
                    "referenced_effort": round(float(peer_state.get("effort", 0.0)), 3),
                    "referenced_willingness_units": round(float(peer_state.get("willingness", 0.0)), 3),
                }

            if last_conflict:
                if token == "B":
                    proposal = {"A": 34.0, "B": 33.0, "C": 33.0}
                elif token == "C":
                    proposal["C"] += 6.0
                    proposal["A"] -= 3.0
                    proposal["B"] -= 3.0
                else:
                    proposal["A"] += 4.0
                    proposal["B"] += 1.0
                    proposal["C"] -= 5.0
            if target_gap > 0 and token in {"A", "B"}:
                proposal["A"] += 2.0
                proposal["C"] -= 1.0
                proposal["B"] -= 1.0
            if hidden_mode and token == "C":
                proposal["C"] += 4.0
                proposal["A"] -= 2.0
                proposal["B"] -= 2.0
            if hidden_mode and token == "B":
                proposal = {
                    "A": proposal["A"] * 0.95 + 34.0 * 0.05,
                    "B": proposal["B"] * 0.95 + 33.0 * 0.05,
                    "C": proposal["C"] * 0.95 + 33.0 * 0.05,
                }
            if forced_strike:
                proposal["B"] += 2.0
                proposal["A"] -= 1.0
                proposal["C"] -= 1.0

            if influenced_by in {"A", "B", "C"}:
                proposal[influenced_by] = max(5.0, proposal[influenced_by] + 2.0)
                for alloc_key in ("A", "B", "C"):
                    if alloc_key == influenced_by:
                        continue
                    proposal[alloc_key] = max(5.0, proposal[alloc_key] - 1.0)

            total_alloc = proposal["A"] + proposal["B"] + proposal["C"]
            if total_alloc <= 1e-8:
                proposal = {"A": 34.0, "B": 33.0, "C": 33.0}
            else:
                proposal = {
                    "A": round(proposal["A"] / total_alloc * 100.0, 3),
                    "B": round(proposal["B"] / total_alloc * 100.0, 3),
                    "C": round(proposal["C"] / total_alloc * 100.0, 3),
                }
            effort_delta = effort - previous_effort
            willingness_delta = willingness - previous_willingness
            state_tags: list[str] = []
            if last_conflict:
                state_tags.append("last_conflict")
            if hidden_mode:
                state_tags.append("hidden_info")
            if forced_strike:
                state_tags.append("strike_mode")
            if target_gap > 0:
                state_tags.append("target_gap_positive")
            state_text = ",".join(state_tags) if state_tags else "baseline"
            payload = {
                "effort": round(effort, 3),
                "willingness_units": round(willingness, 3),
                "allocation_proposal_units": proposal,
                "vote_mechanism": vote,
                "influenced_by": influenced_by,
                "adjustment_reason": adjustment_reason,
                "previous_effort": round(previous_effort, 3),
                "effort_delta": round(effort_delta, 3),
                "previous_willingness_units": round(previous_willingness, 3),
                "willingness_delta": round(willingness_delta, 3),
                "peer_reference": peer_reference,
                "self_critique": {
                    "consistency_score": 0.72 if state_text != "baseline" else 0.64,
                    "confidence": 0.68 if not forced_strike else 0.55,
                    "loss_aversion_signal": 0.58 if "last_conflict" in state_text else 0.42,
                    "inertia_signal": 0.61 if token == "B" else 0.48,
                    "short_term_bias": 0.74 if token == "C" else 0.46,
                    "alignment_check": f"Checked proposal under state={state_text} and peer influence={influenced_by}.",
                    "revised_after_check": bool(abs(effort_delta) > 0.05 or abs(willingness_delta) > 1.0),
                },
                "message": (
                    f"I am {token}. I propose effort={effort:.1f}, "
                    f"willingness={willingness:.1f}, "
                    f"allocation A/B/C="
                    f"{proposal['A']:.1f}/{proposal['B']:.1f}/{proposal['C']:.1f}. "
                    f"Influenced by {influenced_by}. State={state_text}."
                ),
            }
            target = ctx.peers[ctx.step % len(ctx.peers)] if ctx.peers else None
            return AgentDecision(
                event_type="message",
                content=(
                    f"{payload['message']}\n"
                    f"RESOURCE_AGENT_PROPOSAL {json.dumps(payload, ensure_ascii=True)}"
                ),
                target_agent=target,
                memory_append=[f"mock-resource:{ctx.step}"],
            )
        if ctx.peers:
            target = ctx.peers[ctx.step % len(ctx.peers)]
            return AgentDecision(
                event_type="message",
                content=f"mock:{ctx.agent_role}:step-{ctx.step}",
                target_agent=target,
                memory_append=[f"mock:{ctx.step}"],
            )
        return AgentDecision(
            event_type="action",
            content=f"mock:{ctx.agent_role}:solo-step-{ctx.step}",
            memory_append=[f"mock-solo:{ctx.step}"],
        )
