"""The four tools the agent can call.

Each tool takes the claim context first (injected by the executor, never chosen by the LLM,
so a prompt-injected model cannot point a tool at somebody else's order or file) and returns
a JSON-serialisable dict. Failures are returned as {"error": ...} so the agent can react.
"""
from __future__ import annotations

from datetime import date

import imagehash
from PIL import Image

from . import config, db, policy_rag, vision


def hash_image(path) -> str:
    with Image.open(path) as im:
        return str(imagehash.phash(im.convert("RGB")))


# ---- tool 1: cheap database lookup ------------------------------------------
def get_order(claim: dict, **_) -> dict:
    row = db.get_order(claim["order_id"])
    if not row:
        return {"error": f"order {claim['order_id']} not found"}
    days = (date.today() - date.fromisoformat(row["delivered_on"])).days
    return {
        "order_id": row["order_id"],
        "customer_id": row["customer_id"],
        "product_name": row["product_name"],
        "category": row["category"],
        "price": row["price"],
        "delivered_on": row["delivered_on"],
        "days_since_delivery": days,
        "customer_claims_90d": db.customer_claims_90d(row["customer_id"]),
    }


# ---- tool 2: RAG over the return policy -------------------------------------
def search_policy(claim: dict, query: str = "", k: int = 2, **_) -> dict:
    if not str(query).strip():
        return {"error": "query is required"}
    return {"results": policy_rag.get_retriever().search(str(query), k=int(k))}


# ---- tool 3: perceptual-hash duplicate check (classical algorithm, no AI) ----
def check_duplicate(claim: dict, **_) -> dict:
    h = imagehash.hex_to_hash(hash_image(claim["image_path"]))
    best_cross = best_same = None
    for row in db.all_claim_hashes():
        dist = h - imagehash.hex_to_hash(row["image_phash"])
        if dist > config.DUP_HASH_DISTANCE:
            continue
        slot = "same" if row["order_id"] == claim["order_id"] else "cross"
        current = best_same if slot == "same" else best_cross
        if current is None or dist < current["distance"]:
            found = {"claim_id": row["id"], "order_id": row["order_id"], "distance": int(dist)}
            if slot == "same":
                best_same = found
            else:
                best_cross = found
    return {
        "is_duplicate": best_cross is not None,  # same photo on ANOTHER order = fraud signal
        "matched_claim_id": best_cross["claim_id"] if best_cross else None,
        "matched_order_id": best_cross["order_id"] if best_cross else None,
        "distance": best_cross["distance"] if best_cross else None,
        "same_order_resubmission": best_same is not None,  # retry on the same order: not fraud
    }


# ---- tool 4: the expensive vision-LLM call ----------------------------------
def analyze_damage(claim: dict, **_) -> dict:
    order = db.get_order(claim["order_id"])
    if not order:
        return {"error": f"order {claim['order_id']} not found"}
    return vision.get_vision().analyze(
        order["listing_image"], claim["image_path"], claim["claim_text"], order["product_name"]
    )


TOOLS = {
    "get_order": get_order,
    "search_policy": search_policy,
    "check_duplicate": check_duplicate,
    "analyze_damage": analyze_damage,
}

# Shown to the LLM planner. Note what is NOT an argument: order id and image path.
TOOL_SPECS = [
    {
        "name": "get_order",
        "cost": "free",
        "description": "Order details for this claim: product, category, price, delivery date, "
        "days_since_delivery and how many claims this customer made in the last 90 days.",
        "args": {},
    },
    {
        "name": "search_policy",
        "cost": "free",
        "description": "Retrieve the most relevant sections of the company return policy "
        "(return windows per category, damage rules, auto-approval limits, inspection checklist).",
        "args": {"query": "natural-language search query"},
    },
    {
        "name": "check_duplicate",
        "cost": "free",
        "description": "Perceptual-hash check: has this exact photo (even resized or re-compressed) "
        "been submitted on a different order before?",
        "args": {},
    },
    {
        "name": "analyze_damage",
        "cost": "EXPENSIVE (vision LLM call)",
        "description": "Vision model compares the customer's photo with the listing photo and the claim text. "
        "Returns damage_visible, severity, damage_score (0-1), same_product_as_listing, matches_claim_text.",
        "args": {},
    },
]


def run_tool(name: str, args: dict | None, claim: dict) -> dict:
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool '{name}'"}
    try:
        return fn(claim, **(args or {}))
    except Exception as exc:  # tool failures are observations, not crashes
        return {"error": f"{type(exc).__name__}: {exc}"}
