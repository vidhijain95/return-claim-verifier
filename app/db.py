"""SQLite persistence.

``orders`` is the stand-in commerce backend the agent calls into.
``claims`` stores every decision together with the full tool trace (audit trail).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id      TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL,
    product_name  TEXT NOT NULL,
    category      TEXT NOT NULL,
    price         REAL NOT NULL,
    delivered_on  TEXT NOT NULL,
    listing_image TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS claims (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id       TEXT NOT NULL,
    customer_id    TEXT,
    claim_text     TEXT NOT NULL,
    image_path     TEXT NOT NULL,
    image_phash    TEXT NOT NULL,
    decision       TEXT NOT NULL,
    validity_score REAL NOT NULL,
    reasoning      TEXT NOT NULL,
    flags          TEXT NOT NULL DEFAULT '[]',
    trace          TEXT NOT NULL DEFAULT '[]',
    vision_calls   INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claims_customer ON claims(customer_id, created_at);
"""


@contextmanager
def connect():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# ---- orders -----------------------------------------------------------------
def insert_order(order_id, customer_id, product_name, category, price, delivered_on, listing_image) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?)",
            (order_id, customer_id, product_name, category, float(price), delivered_on, str(listing_image)),
        )


def get_order(order_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    return dict(row) if row else None


def list_orders() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM orders ORDER BY order_id").fetchall()
    return [dict(r) for r in rows]


def count_orders() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]


# ---- claims -----------------------------------------------------------------
def insert_claim(*, order_id, customer_id, claim_text, image_path, image_phash, decision,
                 validity_score, reasoning, flags, trace, vision_calls) -> tuple[int, str]:
    created_at = _now()
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO claims (order_id, customer_id, claim_text, image_path, image_phash, decision,
                                   validity_score, reasoning, flags, trace, vision_calls, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (order_id, customer_id, claim_text, str(image_path), image_phash, decision,
             float(validity_score), reasoning, json.dumps(flags), json.dumps(trace),
             int(vision_calls), created_at),
        )
        return cur.lastrowid, created_at


def _claim_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["flags"] = json.loads(d["flags"])
    d["trace"] = json.loads(d["trace"])
    return d


def get_claim(claim_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
    return _claim_dict(row) if row else None


def list_claims(limit: int = 20) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM claims ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_claim_dict(r) for r in rows]


def all_claim_hashes() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT id, order_id, image_phash FROM claims").fetchall()
    return [dict(r) for r in rows]


def customer_claims_90d(customer_id: str) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat(timespec="seconds")
    with connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM claims WHERE customer_id = ? AND created_at >= ?",
            (customer_id, cutoff),
        ).fetchone()[0]
