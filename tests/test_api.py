import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config, synth
from app.main import app


@pytest.fixture
def client(env):
    with TestClient(app) as c:  # runs lifespan: creates tables and seeds demo orders in the temp DB
        yield c


def png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def post_claim(client, order_id, img, text="It arrived damaged and cracked.", filename="photo.png", ctype="image/png"):
    data = img if isinstance(img, bytes) else png_bytes(img)
    return client.post("/api/claims", data={"order_id": order_id, "claim_text": text},
                       files={"photo": (filename, data, ctype)})


def test_health_orders_and_listing_image(client):
    assert client.get("/health").json() == {"status": "ok", "provider": "mock"}
    orders = client.get("/api/orders").json()
    assert len(orders) == 24 and "listing_image" not in orders[0]  # no server paths leak
    img = client.get(f"/api/orders/{orders[0]['order_id']}/listing-image")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert client.get("/api/orders/NOPE/listing-image").status_code == 404
    assert client.get("/").status_code == 200


def test_damaged_photo_runs_the_full_agent_and_is_stored(client):
    r = post_claim(client, "ORD1001", synth.customer_photo(1000, cracks=5))
    assert r.status_code == 200
    body = r.json()
    assert [t["tool"] for t in body["trace"]] == ["get_order", "search_policy", "check_duplicate", "analyze_damage"]
    assert body["vision_calls"] == 1 and body["validity_score"] > 0.5
    detail = client.get(f"/api/claims/{body['claim_id']}").json()
    assert detail["decision"] == body["decision"] and "image_path" not in detail
    assert client.get("/api/claims").json()[0]["id"] == body["claim_id"]
    assert len(list(config.UPLOAD_DIR.iterdir())) == 1  # saved under a server-chosen name


def test_intact_photo_is_not_auto_approved(client):
    body = post_claim(client, "ORD1001", synth.customer_photo(1000, cracks=0)).json()
    assert body["decision"] == "escalate_to_human" and "no_damage_visible" in body["flags"]


def test_same_photo_on_a_second_order_is_caught_end_to_end(client):
    photo = png_bytes(synth.customer_photo(1000, cracks=5))
    assert post_claim(client, "ORD1001", photo).json()["vision_calls"] == 1
    second = post_claim(client, "ORD1002", photo).json()
    assert second["decision"] == "escalate_to_human" and "duplicate_photo" in second["flags"]
    assert second["vision_calls"] == 0


def test_input_validation(client):
    good = synth.customer_photo(1000, cracks=3)
    assert post_claim(client, "NOPE", good).status_code == 404
    assert post_claim(client, "ORD1001", good, text="no").status_code == 400
    assert post_claim(client, "ORD1001", b"definitely not an image", filename="a.png").status_code == 400
    gif = io.BytesIO(); Image.new("RGB", (8, 8)).save(gif, format="GIF")
    assert post_claim(client, "ORD1001", gif.getvalue(), filename="a.gif", ctype="image/gif").status_code == 400
    assert client.post("/api/claims", data={"order_id": "ORD1001"}).status_code == 422
    assert client.get("/api/claims/9999").status_code == 404
    assert list(config.UPLOAD_DIR.iterdir()) == []  # rejected uploads are never written to disk


def test_oversized_upload_is_rejected(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)
    assert post_claim(client, "ORD1001", synth.customer_photo(1000, cracks=3)).status_code == 413


def test_ui_has_no_unsafe_html_sinks():
    html = (config.BASE_DIR / "app" / "static" / "index.html").read_text()
    assert "innerHTML" not in html and "insertAdjacentHTML" not in html
