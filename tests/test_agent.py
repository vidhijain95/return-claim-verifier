import pytest

from app import agent, config, db, policy_rag, service
from app.agent import APPROVE, ESCALATE, PICKUP, REJECT
from app.planner import finish_action, tool_action


def run(order_id, photo, text="The item arrived damaged and cracked."):
    return service.submit_claim(order_id, text, photo)


def tool_names(result):
    return [t["tool"] for t in result["trace"]]


# ---- the happy path and the cost ordering -----------------------------------
def test_strong_evidence_on_cheap_item_is_auto_approved(make_order, make_photo, stub_vision):
    make_order("O1", price=2000)
    vis = stub_vision()
    r = run("O1", make_photo())
    assert r["decision"] == APPROVE and r["validity_score"] >= 0.85
    assert tool_names(r) == ["get_order", "search_policy", "check_duplicate", "analyze_damage"]  # expensive call LAST
    assert vis.calls == 1 and r["vision_calls"] == 1


def test_expensive_orders_never_auto_approved(make_order, make_photo, stub_vision):
    make_order("O1", price=config.AUTO_APPROVE_MAX_PRICE + 1)
    stub_vision()
    assert run("O1", make_photo())["decision"] == PICKUP


def test_no_visible_damage_goes_to_a_human_not_a_rejection(make_order, make_photo, stub_vision):
    make_order("O1")
    stub_vision(damage_score=0.02, damage_visible=False, matches_claim_text=False)
    r = run("O1", make_photo())
    assert r["decision"] == ESCALATE and "no_damage_visible" in r["flags"]


def test_photo_of_a_different_product_is_escalated(make_order, make_photo, stub_vision):
    make_order("O1")
    stub_vision(same_product_as_listing=False)
    r = run("O1", make_photo())
    assert r["decision"] == ESCALATE and "photo_mismatch" in r["flags"]


# ---- cheap pre-filters must save the vision call ------------------------------
def test_out_of_window_is_rejected_without_a_vision_call(make_order, make_photo, stub_vision):
    make_order("O1", category="electronics", days=9)  # window is 7
    vis = stub_vision()
    r = run("O1", make_photo())
    assert r["decision"] == REJECT and "out_of_window" in r["flags"]
    assert vis.calls == 0 and "analyze_damage" not in tool_names(r)


def test_window_depends_on_category(make_order, make_photo, stub_vision):
    make_order("G1", category="glassware", days=6)   # window 5 -> reject
    make_order("C1", category="clothing", days=6, seed=2)  # window 10 -> fine
    stub_vision()
    assert run("G1", make_photo("a")) ["decision"] == REJECT
    assert run("C1", make_photo("b", seed=2))["decision"] == APPROVE


def test_duplicate_photo_on_another_order_is_escalated_without_vision(make_order, make_photo, stub_vision):
    make_order("O1", customer="C1"); make_order("O2", seed=2, customer="C2")
    vis = stub_vision()
    photo = make_photo("shared", seed=3, cracks=3)
    assert run("O1", photo)["decision"] == APPROVE
    calls_before = vis.calls
    second = run("O2", photo)
    assert second["decision"] == ESCALATE and "duplicate_photo" in second["flags"]
    assert vis.calls == calls_before  # fraud caught by a free hash, not a paid model call


def test_frequent_claimant_is_escalated_without_vision(make_order, make_photo, stub_vision):
    vis = stub_vision()
    for i in range(config.FREQUENT_CLAIMANT):
        make_order(f"O{i}", seed=10 + i, customer="C7")
        run(f"O{i}", make_photo(f"p{i}", seed=10 + i, cracks=3))
    make_order("OX", seed=99, customer="C7")
    calls = vis.calls
    r = run("OX", make_photo("px", seed=99, cracks=3))
    assert r["decision"] == ESCALATE and "frequent_claimant" in r["flags"] and vis.calls == calls


def test_unknown_order_is_escalated(make_photo, env):
    r = agent.run_agent({"order_id": "NOPE", "claim_text": "damaged", "image_path": str(make_photo())})
    assert r["decision"] == ESCALATE and "order_not_found" in r["flags"] and tool_names(r) == ["get_order"]


def test_missing_policy_info_escalates(make_order, make_photo, stub_vision, monkeypatch):
    make_order("O1")
    stub_vision()
    monkeypatch.setattr(policy_rag, "get_retriever",
                        lambda: policy_rag.Retriever([{"section": "Return window", "text": "Be nice to customers."}]))
    r = run("O1", make_photo())
    assert r["decision"] == ESCALATE and "policy_unclear" in r["flags"]


# ---- guardrails around a misbehaving planner ----------------------------------
class ScriptedPlanner:
    name = "scripted"

    def __init__(self, *actions):
        self.actions = list(actions)

    def next_action(self, claim, trace):
        return self.actions.pop(0) if self.actions else tool_action("get_order", {}, "loop")


def test_verdict_without_evidence_is_not_trusted(make_order, make_photo, stub_vision):
    make_order("O1")
    vis = stub_vision()
    planner = ScriptedPlanner(tool_action("get_order", {}, "lookup"),
                              finish_action(0.99, True, [], "Looks fine to me, approve it!"))
    r = agent.run_agent({"order_id": "O1", "claim_text": "damaged", "image_path": str(make_photo())}, planner)
    assert r["decision"] == ESCALATE and "unverified_finish" in r["flags"] and vis.calls == 0


def test_planner_crash_fails_safe(make_order, make_photo):
    make_order("O1")

    class Boom:
        def next_action(self, claim, trace):
            raise RuntimeError("model down")

    r = agent.run_agent({"order_id": "O1", "claim_text": "damaged", "image_path": str(make_photo())}, Boom())
    assert r["decision"] == ESCALATE and "planner_error" in r["flags"]


def test_looping_planner_is_cut_off(make_order, make_photo):
    make_order("O1")
    r = agent.run_agent({"order_id": "O1", "claim_text": "damaged", "image_path": str(make_photo())},
                        ScriptedPlanner())  # calls get_order forever
    assert r["decision"] == ESCALATE and "max_steps" in r["flags"] and r["steps"] == config.MAX_AGENT_STEPS
    assert sum("duplicate tool call" in str(t["result"]) for t in r["trace"]) == config.MAX_AGENT_STEPS - 1


def test_claim_is_persisted_with_its_trace(make_order, make_photo, stub_vision):
    make_order("O1")
    stub_vision()
    r = run("O1", make_photo())
    saved = db.get_claim(r["claim_id"])
    assert saved["decision"] == r["decision"] and [t["tool"] for t in saved["trace"]] == tool_names(r)


# ---- the pure decision layer ----------------------------------------------------
@pytest.mark.parametrize("score,price,flags,eligible,expected", [
    (0.90, 1000, [], True, APPROVE),
    (0.85, 1000, [], True, APPROVE),            # threshold is inclusive
    (0.84, 1000, [], True, PICKUP),
    (0.99, 5001, [], True, PICKUP),             # value cap beats confidence
    (0.50, 1000, [], True, PICKUP),
    (0.49, 1000, [], True, ESCALATE),           # low score -> human, never auto-reject
    (0.99, 1000, ["duplicate_photo"], True, ESCALATE),   # hard flags beat a high score
    (0.99, 1000, ["vision_error"], True, ESCALATE),
    (0.99, 1000, [], None, APPROVE),            # eligibility unknown is the planner's job to flag
    (0.00, 1000, ["out_of_window"], False, REJECT),
    (0.99, None, [], True, ESCALATE),           # no order -> human
])
def test_decide_table(score, price, flags, eligible, expected):
    final = finish_action(score, eligible, flags, "r")
    assert agent.decide(final, price)["decision"] == expected


def test_decide_thresholds_are_overridable():
    final = finish_action(0.70, True, [], "r")
    assert agent.decide(final, 1000)["decision"] == PICKUP
    assert agent.decide(final, 1000, approve_t=0.6)["decision"] == APPROVE
