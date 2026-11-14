"""Central configuration.

Everything is read from environment variables (or a local .env file). Other modules
use ``config.NAME`` at call time, never ``from config import NAME``, so tests can
monkeypatch values.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Tiny .env loader so we don't need an extra dependency."""
    env_file = BASE_DIR / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# ---- paths -----------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "claims.db"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", DATA_DIR / "uploads"))
LISTING_DIR = Path(os.getenv("LISTING_DIR", DATA_DIR / "listing_images"))
DEMO_DIR = Path(os.getenv("DEMO_DIR", DATA_DIR / "demo_photos"))
POLICY_PATH = Path(os.getenv("POLICY_PATH", BASE_DIR / "data" / "return_policy.md"))

# ---- AI provider -----------------------------------------------------------
# "mock"   -> offline, deterministic (image-statistics vision + rule-based planner).
#             Lets you run the whole pipeline, tests and eval with no API key.
# "gemini" -> real vision LLM + LLM-driven tool-calling planner.
PROVIDER = os.getenv("LLM_PROVIDER", "mock").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ---- decision policy (tune these with eval/run_eval.py, don't guess) --------
APPROVE_THRESHOLD = float(os.getenv("APPROVE_THRESHOLD", "0.85"))
PICKUP_THRESHOLD = float(os.getenv("PICKUP_THRESHOLD", "0.50"))
AUTO_APPROVE_MAX_PRICE = float(os.getenv("AUTO_APPROVE_MAX_PRICE", "5000"))
FREQUENT_CLAIMANT = int(os.getenv("FREQUENT_CLAIMANT", "3"))  # prior claims in 90 days

# ---- tools / agent ---------------------------------------------------------
DUP_HASH_DISTANCE = int(os.getenv("DUP_HASH_DISTANCE", "5"))  # max Hamming distance (of 64 bits)
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "8"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
