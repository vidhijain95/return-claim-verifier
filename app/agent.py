"""The agent loop + the deterministic decision layer.

    planner (LLM or rules)  --next action-->  executor runs tool  --observation-->  planner ...
                                                   |
                                      finish -> guardrails -> decide()  -> decision

Design rule: the LLM produces EVIDENCE and a validity score; plain code turns that into
money-moving decisions. That keeps the blast radius of a wrong / prompt-injected model small
and makes the thresholds tunable from data (see eval/run_eval.py).
"""
from __future__ import annotations

import json
import time

from . import config
from .planner import finish_action, get_planner
from .tools import run_tool

# Flags that always send the claim to a human, whatever the score says.
HARD_ESCALATE = {
    "duplicate_photo", "frequent_claimant", "planner_error", "max_steps",
    "unverified_finish", "policy_unclear", "vision_error",
}
# Flags that mean "decided without needing the vision call".
EVIDENCE_FREE = {"duplicate_photo", "out_of_window", "order_not_found", "frequent_claimant"}

APPROVE, PICKUP, ESCALATE, REJECT = "auto_approve", "pickup_with_checklist", "escalate_to_human", "reject_ineligible"


def apply_guardrails(final: dict, trace: list[dict]) -> dict:
    """Never trust a verdict that is not backed by the evidence it should rest on."""
    flags = set(final["flags"])
    called = {t["tool"] for t in trace if "error" not in t["result"]}
    if final["policy_eligible"] is not False and not (flags & EVIDENCE_FREE) and "analyze_damage" not in called:
        final = {**final, "flags": sorted(flags | {"unverified_finish"}), "validity_score": min(final["validity_score"], 0.0)}
    return final


def decide(final: dict, price: float | None, approve_t: float | None = None,
           pickup_t: float | None = None, max_price: float | None = None) -> dict:
    """Pure function: assessment -> business decision. Thresholds default to config."""
    approve_t = config.APPROVE_THRESHOLD if approve_t is None else approve_t
    pickup_t = config.PICKUP_THRESHOLD if pickup_t is None else pickup_t
    max_price = config.AUTO_APPROVE_MAX_PRICE if max_price is None else max_price
    flags, score = set(final["flags"]), final["validity_score"]

    if "order_not_found" in flags or price is None:
        return {"decision": ESCALATE, "reason": "Order not found - needs a human."}
    if final["policy_eligible"] is False:
        return {"decision": REJECT, "reason": "Ineligible under a hard policy rule (e.g. outside the return window)."}
    hard = sorted(flags & HARD_ESCALATE)
    if hard:
        return {"decision": ESCALATE, "reason": f"Flag(s) {', '.join(hard)} always require human review."}
    if score >= approve_t and price <= max_price:
        return {"decision": APPROVE, "reason": f"Validity {score} >= {approve_t} and order value within auto-approve limit."}
    if score >= pickup_t:
        why = "order value above the auto-approve limit" if score >= approve_t else f"validity {score} below {approve_t}"
        return {"decision": PICKUP, "reason": f"Plausible claim, but {why}: pickup with inspection checklist."}
    return {"decision": ESCALATE, "reason": f"Validity {score} below {pickup_t}: human review (never auto-rejected on suspicion)."}


def run_agent(claim: dict, planner=None) -> dict:
    planner = planner or get_planner()
    started = time.perf_counter()
    trace: list[dict] = []
    seen: set[tuple[str, str]] = set()
    final = None

    for step in range(1, config.MAX_AGENT_STEPS + 1):
        try:
            action = planner.next_action(claim, trace)
        except Exception as exc:
            final = finish_action(0.0, None, ["planner_error"], f"Planner failed: {type(exc).__name__}: {exc}")
            break
        if action["type"] == "finish":
            final = action
            break
        key = (action["tool"], json.dumps(action["args"], sort_keys=True))
        if key in seen:
            result = {"error": "duplicate tool call ignored - reuse the earlier observation"}
        else:
            seen.add(key)
            result = run_tool(action["tool"], action["args"], claim)
        trace.append({"step": step, "tool": action["tool"], "args": action["args"],
                      "why": action.get("why", ""), "result": result})

    if final is None:
        final = finish_action(0.0, None, ["max_steps"], f"Planner did not finish within {config.MAX_AGENT_STEPS} steps.")
    final = apply_guardrails(final, trace)

    order = next((t["result"] for t in trace if t["tool"] == "get_order" and "error" not in t["result"]), None)
    outcome = decide(final, order["price"] if order else None)
    return {
        "decision": outcome["decision"],
        "validity_score": final["validity_score"],
        "reasoning": f"{final['reasoning']} -> {outcome['reason']}",
        "flags": final["flags"],
        "policy_eligible": final["policy_eligible"],
        "final": final,
        "trace": trace,
        "vision_calls": sum(1 for t in trace if t["tool"] == "analyze_damage" and "error" not in t["result"]),
        "steps": len(trace),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "planner": getattr(planner, "name", type(planner).__name__),
    }
