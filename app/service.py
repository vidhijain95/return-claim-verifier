"""One entry point used by both the API and the eval harness: run the agent, persist the result."""
from __future__ import annotations

from . import agent, db, tools


def submit_claim(order_id: str, claim_text: str, image_path, planner=None) -> dict:
    claim = {"order_id": order_id, "claim_text": claim_text, "image_path": str(image_path)}
    order = db.get_order(order_id)
    result = agent.run_agent(claim, planner)
    claim_id, created_at = db.insert_claim(
        order_id=order_id,
        customer_id=order["customer_id"] if order else None,
        claim_text=claim_text,
        image_path=image_path,
        image_phash=tools.hash_image(image_path),  # stored for EVERY claim so later claims can be matched
        decision=result["decision"],
        validity_score=result["validity_score"],
        reasoning=result["reasoning"],
        flags=result["flags"],
        trace=result["trace"],
        vision_calls=result["vision_calls"],
    )
    return {**result, "claim_id": claim_id, "order_id": order_id, "created_at": created_at}
