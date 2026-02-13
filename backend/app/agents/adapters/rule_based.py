"""Rule-based fallback adapter used when model calls fail."""

from .base import AgentDecision, DecisionAdapter, DecisionContext


class RuleBasedDecisionAdapter(DecisionAdapter):
    """Generate lightweight deterministic replies and actions."""

    async def decide(self, ctx: DecisionContext) -> AgentDecision:
        role = ctx.agent_role.lower()
        if ctx.inbox:
            msg = ctx.inbox[-1]
            responses = [
                f"{ctx.agent_role} replies to {msg.source_agent}: acknowledged. I will convert this into concrete next actions.",
                f"{ctx.agent_role} to {msg.source_agent}: received. I will refine scope and reduce ambiguity.",
                f"{ctx.agent_role} answers {msg.source_agent}: good input. I will turn this into a concise execution plan.",
            ]
            content = responses[(ctx.step + len(ctx.agent_name)) % len(responses)]
            return AgentDecision(
                event_type="message",
                content=content,
                target_agent=msg.source_agent,
                memory_append=[f"inbox:{msg.source_agent}:{msg.content}", f"reply:{content}"],
            )

        if ctx.peers:
            target = ctx.peers[(ctx.step + len(ctx.agent_name)) % len(ctx.peers)]
            if role == "critic":
                templates = [
                    f"{ctx.agent_role} flags risk to {target}: verify assumptions before committing implementation.",
                    f"{ctx.agent_role} asks {target} to add safeguards and failure handling for the current plan.",
                ]
            elif role == "researcher":
                templates = [
                    f"{ctx.agent_role} asks {target} to validate facts and gather stronger references for goal progress.",
                    f"{ctx.agent_role} suggests to {target}: compare alternatives and share evidence-backed tradeoffs.",
                ]
            else:
                templates = [
                    f"{ctx.agent_role} suggests collaborating with {target} to advance goal: {ctx.goal[:60]}",
                    f"{ctx.agent_role} nudges {target}: align on milestones and deliver a step-by-step plan for {ctx.goal[:40]}",
                ]
            content = templates[ctx.step % len(templates)]
            return AgentDecision(
                event_type="message",
                content=content,
                target_agent=target,
                memory_append=[f"nudge:{target}:{ctx.step}"],
            )

        content = f"{ctx.agent_role} executes independent action at step {ctx.step}"
        return AgentDecision(
            event_type="action",
            content=content,
            memory_append=[f"action:{ctx.step}"],
        )
