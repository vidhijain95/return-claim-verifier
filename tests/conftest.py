import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, db, synth, vision  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated DB + upload dirs, offline provider."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "LISTING_DIR", tmp_path / "listing")
    monkeypatch.setattr(config, "DEMO_DIR", tmp_path / "demo")
    monkeypatch.setattr(config, "PROVIDER", "mock")
    vision.reset()
    db.init_db()
    yield tmp_path
    vision.reset()


@pytest.fixture
def make_order(env):
    def _make(order_id="O1", category="electronics", price=2000.0, days=2, seed=1, customer="C1"):
        listing = env / f"{order_id}_listing.png"
        synth.product_image(seed).save(listing)
        delivered = (date.today() - timedelta(days=days)).isoformat()
        db.insert_order(order_id, customer, "Test item", category, price, delivered, listing)
        return order_id

    return _make


@pytest.fixture
def make_photo(env):
    def _make(name="photo", seed=1, cracks=0, other=False):
        path = env / f"{name}.png"
        synth.customer_photo(seed, cracks=cracks, other_product=other).save(path)
        return path

    return _make


class StubVision:
    """Deterministic vision provider that counts how often the expensive call happens."""

    name = "stub"

    def __init__(self, **overrides):
        base = {"damage_score": 0.95, "damage_visible": True, "damage_type": "cracked",
                "same_product_as_listing": True, "matches_claim_text": True}
        self.result = vision.normalize({**base, **overrides}, "stub")
        self.calls = 0

    def analyze(self, *args, **kwargs):
        self.calls += 1
        return self.result


@pytest.fixture
def stub_vision():
    def _install(**overrides):
        stub = StubVision(**overrides)
        vision.set_provider(stub)
        return stub

    return _install
