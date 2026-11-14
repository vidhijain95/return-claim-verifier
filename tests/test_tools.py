import numpy as np
from PIL import Image

from app import db, synth, tools, vision


def _store_claim(order_id, photo, customer="C1"):
    db.insert_claim(order_id=order_id, customer_id=customer, claim_text="damaged", image_path=photo,
                    image_phash=tools.hash_image(photo), decision="escalate_to_human", validity_score=0.1,
                    reasoning="x", flags=[], trace=[], vision_calls=0)


def test_get_order_found_and_missing(make_order):
    make_order("O1", category="clothing", price=999, days=3, customer="C9")
    o = tools.get_order({"order_id": "O1"})
    assert (o["category"], o["price"], o["days_since_delivery"], o["customer_claims_90d"]) == ("clothing", 999, 3, 0)
    assert "error" in tools.get_order({"order_id": "NOPE"})


def test_customer_claim_count(make_order, make_photo):
    make_order("O1", customer="C9")
    for i in range(3):
        _store_claim("O1", make_photo(f"p{i}", seed=i), customer="C9")
    assert tools.get_order({"order_id": "O1"})["customer_claims_90d"] == 3


def test_search_policy_requires_query():
    assert "error" in tools.search_policy({}, query="  ")
    assert tools.search_policy({}, query="return window days electronics")["results"][0]["section"] == "Return window"


def test_duplicate_detected_across_orders_even_after_recompression(make_order, make_photo, env):
    make_order("O1"); make_order("O2", seed=2, customer="C2")
    original = make_photo("orig", seed=5, cracks=3)
    _store_claim("O1", original)

    resaved = env / "resaved.jpg"  # smaller + lossy copy, like a screenshot or re-upload
    Image.open(original).resize((200, 200)).save(resaved, quality=60)
    res = tools.check_duplicate({"order_id": "O2", "image_path": str(resaved)})
    assert res["is_duplicate"] and res["matched_order_id"] == "O1" and res["distance"] <= 5


def test_different_photo_is_not_duplicate(make_order, make_photo):
    make_order("O1"); make_order("O2", seed=2)
    _store_claim("O1", make_photo("a", seed=5, cracks=3))
    res = tools.check_duplicate({"order_id": "O2", "image_path": str(make_photo("b", seed=6, cracks=3))})
    assert not res["is_duplicate"]


def test_same_order_resubmission_is_not_fraud(make_order, make_photo):
    make_order("O1")
    photo = make_photo("a", seed=5, cracks=3)
    _store_claim("O1", photo)
    res = tools.check_duplicate({"order_id": "O1", "image_path": str(photo)})
    assert not res["is_duplicate"] and res["same_order_resubmission"]


def test_run_tool_survives_bad_calls():
    assert "unknown tool" in tools.run_tool("rm_rf", {}, {})["error"]
    assert "error" in tools.run_tool("search_policy", {"query": "x", "bogus": 1, "k": "not-a-number"}, {})


def test_mock_vision_separates_damaged_from_intact(tmp_path):
    mv = vision.MockVision()
    def avg(cracks, other=False):
        scores, same = [], []
        for s in range(25):
            listing, photo = tmp_path / f"l{s}.png", tmp_path / f"p{s}.png"
            synth.product_image(s).save(listing)
            synth.customer_photo(s, cracks=cracks, other_product=other).save(photo)
            r = mv.analyze(listing, photo, "it arrived damaged", "x")
            scores.append(r["damage_score"]); same.append(r["same_product_as_listing"])
        return float(np.mean(scores)), float(np.mean(same))
    assert avg(0)[0] < 0.1
    assert avg(3)[0] > 0.6
    assert avg(0)[1] > 0.95            # same product recognised
    assert avg(3, other=True)[1] < 0.3  # different product mostly rejected


def test_normalize_is_defensive():
    n = vision.normalize({"damage_score": "7", "damage_visible": "false", "severity": "catastrophic"}, "x")
    assert n["damage_score"] == 1.0 and n["damage_visible"] is False and n["severity"] == "severe"
    assert vision.normalize({}, "x")["same_product_as_listing"] is True


def test_parse_json_tolerates_fences_and_chatter():
    assert vision.parse_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
