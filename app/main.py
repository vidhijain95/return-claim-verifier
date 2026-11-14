"""FastAPI app: upload a claim photo -> the agent verifies it -> decision + full reasoning trace."""
from __future__ import annotations

import io
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import config, db, seed, service

STATIC_DIR = Path(__file__).parent / "static"
EXTENSIONS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    if db.count_orders() == 0:
        seed.seed_orders()
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Return Claim Verifier", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _public_order(o: dict) -> dict:
    days = (date.today() - date.fromisoformat(o["delivered_on"])).days
    return {k: o[k] for k in ("order_id", "product_name", "category", "price", "delivered_on")} | {"days_since_delivery": days}


def _validated_extension(data: bytes) -> str:
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.verify()
    except Exception:
        raise HTTPException(400, "The uploaded file is not a valid image.")
    if fmt not in EXTENSIONS:
        raise HTTPException(400, "Only JPEG, PNG or WEBP images are accepted.")
    return EXTENSIONS[fmt]


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "provider": config.PROVIDER}


@app.get("/api/orders")
def list_orders():
    return [_public_order(o) for o in db.list_orders()]


@app.get("/api/orders/{order_id}/listing-image")
def listing_image(order_id: str):
    order = db.get_order(order_id)
    if not order or not Path(order["listing_image"]).exists():
        raise HTTPException(404, "Order or listing image not found.")
    return FileResponse(order["listing_image"])


@app.post("/api/claims")
def create_claim(order_id: str = Form(...), claim_text: str = Form(...), photo: UploadFile = File(...)):
    claim_text = claim_text.strip()
    if not 5 <= len(claim_text) <= 1000:
        raise HTTPException(400, "claim_text must be between 5 and 1000 characters.")
    if db.get_order(order_id) is None:
        raise HTTPException(404, f"Order {order_id} not found.")
    data = photo.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is too large.")
    ext = _validated_extension(data)

    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = config.UPLOAD_DIR / f"{uuid.uuid4().hex}.{ext}"  # never trust the client's filename
    path.write_bytes(data)
    return service.submit_claim(order_id, claim_text, path)


@app.get("/api/claims")
def list_claims(limit: int = 20):
    rows = db.list_claims(max(1, min(limit, 100)))
    return [{k: r[k] for k in ("id", "order_id", "decision", "validity_score", "flags", "vision_calls", "created_at")} for r in rows]


@app.get("/api/claims/{claim_id}")
def get_claim(claim_id: int):
    claim = db.get_claim(claim_id)
    if claim is None:
        raise HTTPException(404, "Claim not found.")
    claim.pop("image_path", None)  # internal server path
    return claim
