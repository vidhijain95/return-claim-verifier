"""Evaluation harness: pick the approve-threshold with numbers, not vibes.

The agent runs ONCE per labelled case. Because ``agent.decide`` is a pure function, we can then
replay the stored assessments under many thresholds and price each outcome with a cost model.

    python -m eval.run_eval                      # synthetic cases, offline mock provider
    python -m eval.run_eval --fraud-rate 0.05    # fraud prevalence is the biggest lever - try it
    python -m eval.run_eval --csv my_cases.csv   # YOUR real photos + labels (see README)
    LLM_PROVIDER=gemini python -m eval.run_eval --csv my_cases.csv

IMPORTANT: with the default synthetic data this validates the PIPELINE (tools, guardrails,
threshold logic, cost model). It says nothing about real-world accuracy. Cost-model numbers are
ASSUMPTIONS - replace them with your business's real figures.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from datetime import date, timedelta
from pathlib import Path

from app import agent, config, db, service, synth, vision
from app.agent import APPROVE, ESCALATE, PICKUP, REJECT

PRICE_RANGE = {"electronics": (1500, 45000), "clothing": (400, 3500), "footwear": (800, 6000), "glassware": (300, 2500)}
CLAIM_TEXTS = [
    "The item arrived damaged.",
    "It was cracked when I opened the box.",
    "Product is broken and unusable.",
    "Received it with visible damage, please replace.",
]
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]


# ---- cases ------------------------------------------------------------------
@dataclass
class Case:
    case_id: str
    kind: str
    is_legit: bool
    category: str
    price: float
    days_since_delivery: int
    listing_image: str
    claim_image: str
    claim_text: str


def _weighted(rng: random.Random, options: list[tuple[str, float]]) -> str:
    return rng.choices([o[0] for o in options], weights=[o[1] for o in options])[0]


def build_synthetic_cases(n: int, fraud_rate: float, seed: int, out_dir: Path, prefix: str = "E") -> list[Case]:
    rng = random.Random(seed)
    img_dir = Path(out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    cases: list[Case] = []
    legit_photos: list[str] = []  # earlier legit claim photos a fraudster may reuse
    for i in range(n):
        if rng.random() < fraud_rate:
            kind = _weighted(rng, [("fraud_no_damage", .30), ("fraud_staged", .30), ("fraud_other_product", .15), ("fraud_duplicate", .25)])
        else:
            kind = _weighted(rng, [("legit_damage", .75), ("legit_minor", .25)])
        if kind == "fraud_duplicate" and not legit_photos:
            kind = "fraud_no_damage"

        category = rng.choice(list(PRICE_RANGE))
        lo, hi = PRICE_RANGE[category]
        case_id, img_seed = f"{prefix}{i:04d}", seed * 100_000 + i
        listing = img_dir / f"{case_id}_listing.png"
        synth.product_image(img_seed).save(listing)

        claim_img = img_dir / f"{case_id}_claim.png"
        if kind == "fraud_duplicate":
            claim_path = rng.choice(legit_photos)
        else:
            cracks, other = {
                "legit_damage": (rng.choice([2, 3, 3, 4, 5]), False),
                "legit_minor": (1, False),
                "fraud_no_damage": (0, False),
                "fraud_staged": (rng.choice([1, 2, 2, 3]), False),   # fraudster fakes damage on the real product
                "fraud_other_product": (rng.choice([2, 3, 4]), True),  # photo of someone else's damaged item
            }[kind]
            synth.customer_photo(img_seed, cracks=cracks, other_product=other).save(claim_img)
            claim_path = str(claim_img)
            if kind.startswith("legit"):
                legit_photos.append(claim_path)
        cases.append(Case(case_id, kind, kind.startswith("legit"), category, round(rng.uniform(lo, hi) / 10) * 10,
                          rng.randint(1, 4), str(listing), claim_path, rng.choice(CLAIM_TEXTS)))
    return cases


def write_cases_csv(cases: list[Case], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[x.name for x in fields(Case)])
        w.writeheader()
        for c in cases:
            w.writerow(asdict(c))


def load_cases_csv(path: Path) -> list[Case]:
    base = Path(path).parent
    cases = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            def resolve(p: str) -> str:
                return str(p if Path(p).is_absolute() else (base / p))
            cases.append(Case(
                case_id=r["case_id"], kind=r.get("kind") or "unlabelled",
                is_legit=str(r["is_legit"]).strip().lower() in {"1", "true", "yes"},
                category=r["category"], price=float(r["price"]), days_since_delivery=int(r["days_since_delivery"]),
                listing_image=resolve(r["listing_image"]), claim_image=resolve(r["claim_image"]),
                claim_text=r["claim_text"]))
    return cases


# ---- run the agent once per case --------------------------------------------
def run_cases(cases: list[Case], workdir: Path) -> list[dict]:
    """Run every case through the real agent against an isolated database."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    db_path = workdir / "eval.db"
    db_path.unlink(missing_ok=True)
    old_db = config.DB_PATH
    config.DB_PATH = db_path
    vision.reset()
    try:
        db.init_db()
        rows = []
        for c in cases:  # sequential on purpose: later claims can be duplicates of earlier ones
            db.insert_order(c.case_id, f"cust-{c.case_id}", f"{c.category} item", c.category, c.price,
                            (date.today() - timedelta(days=c.days_since_delivery)).isoformat(), c.listing_image)
            res = service.submit_claim(c.case_id, c.claim_text, c.claim_image)
            rows.append({"case": c, "final": res["final"], "vision_calls": res["vision_calls"],
                         "latency_ms": res["latency_ms"], "planner": res["planner"]})
        return rows
    finally:
        config.DB_PATH = old_db
        vision.reset()


# ---- cost model -------------------------------------------------------------
@dataclass
class CostModel:
    """ASSUMPTIONS (rupees). Replace with your business's real numbers."""
    review_cost: float = 60.0          # a human reviews one escalated claim
    pickup_extra_cost: float = 25.0    # extra handling for a pickup with an inspection checklist
    friction_pickup: float = 20.0      # an honest customer is inspected / slightly delayed
    friction_escalate: float = 60.0    # an honest customer waits for a human
    checklist_catch_rate: float = 0.70 # share of fraud a pickup inspection catches
    review_catch_rate: float = 0.95    # share of fraud a human reviewer catches
    wrongful_reject_factor: float = 0.5  # cost of wrongly rejecting an honest customer, x price
    vision_cost: float = 0.8           # one vision-LLM call


def case_cost(decision: str, is_legit: bool, price: float, vision_calls: int, m: CostModel) -> dict:
    leak = friction = ops = 0.0
    if decision == APPROVE:
        leak = 0.0 if is_legit else price
    elif decision == PICKUP:
        ops = m.pickup_extra_cost
        friction, leak = (m.friction_pickup, 0.0) if is_legit else (0.0, (1 - m.checklist_catch_rate) * price)
    elif decision == ESCALATE:
        ops = m.review_cost
        friction, leak = (m.friction_escalate, 0.0) if is_legit else (0.0, (1 - m.review_catch_rate) * price)
    elif decision == REJECT:
        friction = m.wrongful_reject_factor * price if is_legit else 0.0
    ops += vision_calls * m.vision_cost
    return {"leak": leak, "friction": friction, "ops": ops, "total": leak + friction + ops}


def evaluate(rows: list[dict], decide_fn, m: CostModel, use_vision: bool = True) -> dict:
    n = len(rows)
    dec, tot = Counter(), defaultdict(float)
    fraud_total = sum(1 for r in rows if not r["case"].is_legit)
    legit_total = n - fraud_total
    fraud_approved = legit_approved = legit_escalated = vision_calls = 0
    for r in rows:
        c = r["case"]
        d = decide_fn(r)
        calls = r["vision_calls"] if use_vision else 0
        cost = case_cost(d, c.is_legit, c.price, calls, m)
        dec[d] += 1
        vision_calls += calls
        for k, v in cost.items():
            tot[k] += v
        fraud_approved += d == APPROVE and not c.is_legit
        legit_approved += d == APPROVE and c.is_legit
        legit_escalated += d == ESCALATE and c.is_legit
    return {
        "auto_approve%": 100 * dec[APPROVE] / n, "pickup%": 100 * dec[PICKUP] / n,
        "escalate%": 100 * dec[ESCALATE] / n, "reject%": 100 * dec[REJECT] / n,
        "fraud_approved": fraud_approved, "fraud_total": fraud_total,
        "legit_auto_approved%": 100 * legit_approved / max(1, legit_total),
        "legit_escalated%": 100 * legit_escalated / max(1, legit_total),
        "vision_calls": vision_calls, "escalations": dec[ESCALATE],
        "fraud_leak_rs": tot["leak"], "friction_rs": tot["friction"], "ops_rs": tot["ops"],
        "total_cost_rs": tot["total"], "cost_per_claim_rs": tot["total"] / n,
    }


def sweep(rows: list[dict], m: CostModel, thresholds=THRESHOLDS) -> list[dict]:
    out = []
    for t in thresholds:
        res = evaluate(rows, lambda r, t=t: agent.decide(r["final"], r["case"].price, approve_t=t)["decision"], m)
        out.append({"approve_threshold": t, **res})
    return out


def baselines(rows: list[dict], m: CostModel) -> dict[str, dict]:
    return {
        "approve every claim (no verification)": evaluate(rows, lambda r: APPROVE, m, use_vision=False),
        "pickup-inspect every claim": evaluate(rows, lambda r: PICKUP, m, use_vision=False),
        "human-review every claim": evaluate(rows, lambda r: ESCALATE, m, use_vision=False),
    }


def break_even_review_cost(ai: dict, human_all: dict, m: CostModel) -> float | None:
    """Review cost per claim above which the AI system beats 'human-review everything'.

    Every cost except review_cost is independent of it, and each escalation pays review_cost once, so
    total = fixed + escalations * review_cost for both strategies. Solve for equality.
    Returns 0.0 if the AI already wins at any review cost, None if it can never win.
    """
    fixed_ai = ai["total_cost_rs"] - ai["escalations"] * m.review_cost
    fixed_hr = human_all["total_cost_rs"] - human_all["escalations"] * m.review_cost
    if fixed_ai <= fixed_hr:
        return 0.0
    extra = human_all["escalations"] - ai["escalations"]
    return (fixed_ai - fixed_hr) / extra if extra > 0 else None


def by_kind(rows: list[dict], approve_t: float) -> dict[str, Counter]:
    table: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        table[r["case"].kind][agent.decide(r["final"], r["case"].price, approve_t=approve_t)["decision"]] += 1
    return table


# ---- reporting --------------------------------------------------------------
def md_table(headers: list[str], body: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in body]
    return "\n".join(lines)


def build_report(rows, sweep_rows, base, best, m: CostModel, source: str) -> str:  # noqa: C901
    n = len(rows)
    fraud = sum(1 for r in rows if not r["case"].is_legit)
    planner = rows[0]["planner"] if rows else "?"
    calls = best["vision_calls"]
    sw = md_table(
        ["approve >=", "auto-approve %", "pickup %", "escalate %", "fraud auto-approved", "legit auto-approved %",
         "vision calls", "total cost (Rs)", "per claim (Rs)"],
        [[f"{r['approve_threshold']:.2f}" + (" **<- best**" if r is best else ""), f"{r['auto_approve%']:.0f}", f"{r['pickup%']:.0f}",
          f"{r['escalate%']:.0f}", f"{r['fraud_approved']}/{r['fraud_total']}", f"{r['legit_auto_approved%']:.0f}",
          r["vision_calls"], f"{r['total_cost_rs']:,.0f}", f"{r['cost_per_claim_rs']:.1f}"] for r in sweep_rows])
    bl = md_table(["strategy", "total cost (Rs)", "per claim (Rs)", "fraud auto-approved"],
                  [[k, f"{v['total_cost_rs']:,.0f}", f"{v['cost_per_claim_rs']:.1f}", f"{v['fraud_approved']}/{v['fraud_total']}"]
                   for k, v in base.items()])
    be = break_even_review_cost(best, base["human-review every claim"], m)
    if be is None:
        verdict = "The agent never beats human-review-all under these assumptions."
    elif be == 0.0:
        verdict = "The agent is cheaper than human-review-all at any review cost."
    else:
        cheaper = best["total_cost_rs"] < base["human-review every claim"]["total_cost_rs"]
        verdict = (f"Under the assumed review cost (Rs {m.review_cost:.0f}) the agent is "
                   f"{'cheaper' if cheaper else 'MORE EXPENSIVE'} than having humans review every claim. "
                   f"Break-even: the agent wins once one human review costs more than about Rs {be:,.0f}. "
                   f"It also needs {100 * best['escalations'] / n:.0f}% of the human reviews and settles "
                   f"{best['auto_approve%']:.0f}% of claims instantly - value this cost model does not price in.")
    kinds = by_kind(rows, best["approve_threshold"])
    kd = md_table(["case kind", "n", APPROVE, PICKUP, ESCALATE, REJECT],
                  [[k, sum(c.values()), c[APPROVE], c[PICKUP], c[ESCALATE], c[REJECT]] for k, c in sorted(kinds.items())])
    ma = ", ".join(f"{k}={v}" for k, v in asdict(m).items())
    return f"""# Evaluation report

- Source: **{source}** | cases: **{n}** ({fraud} fraudulent, {n - fraud} genuine) | planner: `{planner}`
- Cost-model ASSUMPTIONS (Rs): {ma}
- Vision calls made: **{calls} of {n}** claims (cheap pre-filters - out of window, duplicate photo, repeat claimant - skipped {n - calls}).
- Synthetic data validates the pipeline only; it is not evidence about real-world accuracy.
- Neighbouring thresholds often differ by only one or two fraud cases, so treat the best threshold as a *region*, not a point.

## Threshold sweep (vision-verified claims; pickup threshold fixed at {config.PICKUP_THRESHOLD})
{sw}

Lowest total cost at **approve >= {best['approve_threshold']:.2f}**.

## Baselines (no AI)
{bl}

**Reading the baselines:** {verdict}

## Where each kind of case ends up at the best threshold
{kd}
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=150, help="synthetic cases per dataset")
    p.add_argument("--repeats", type=int, default=3, help="independent synthetic datasets to pool (steadier threshold choice)")
    p.add_argument("--fraud-rate", type=float, default=0.20, help="share of fraudulent synthetic claims")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--csv", type=Path, help="evaluate your own labelled cases instead of synthetic ones")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    for f in fields(CostModel):
        p.add_argument(f"--{f.name.replace('_', '-')}", type=float, default=f.default)
    a = p.parse_args(argv)

    m = CostModel(**{f.name: getattr(a, f.name) for f in fields(CostModel)})
    a.out.mkdir(parents=True, exist_ok=True)
    if a.csv:
        cases, source = load_cases_csv(a.csv), f"csv:{a.csv}"
    else:
        cases = []
        for k in range(a.repeats):
            cases += build_synthetic_cases(a.n, a.fraud_rate, a.seed + k, a.out, prefix=f"S{a.seed + k}-")
        write_cases_csv(cases, a.out / "cases.csv")
        source = f"synthetic ({a.repeats} datasets x {a.n}, seeds {a.seed}..{a.seed + a.repeats - 1}, fraud_rate={a.fraud_rate})"

    rows = run_cases(cases, a.out)
    sweep_rows = sweep(rows, m)
    best = min(sweep_rows, key=lambda r: r["total_cost_rs"])
    report = build_report(rows, sweep_rows, baselines(rows, m), best, m, source)

    (a.out / "report.md").write_text(report)
    with open(a.out / "threshold_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0]))
        w.writeheader()
        w.writerows(sweep_rows)
    with open(a.out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "kind", "is_legit", "price", "validity_score", "flags", "policy_eligible", "vision_calls"])
        for r in rows:
            c, fin = r["case"], r["final"]
            w.writerow([c.case_id, c.kind, c.is_legit, c.price, fin["validity_score"], "|".join(fin["flags"]),
                        fin["policy_eligible"], r["vision_calls"]])
    print(report)
    print(f"Files written to {a.out}/ (report.md, threshold_sweep.csv, predictions.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
