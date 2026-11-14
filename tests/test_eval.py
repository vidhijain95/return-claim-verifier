from app import config
from app.agent import APPROVE, ESCALATE
from eval import run_eval as ev


def test_csv_roundtrip(tmp_path):
    cases = ev.build_synthetic_cases(12, 0.5, seed=3, out_dir=tmp_path)
    ev.write_cases_csv(cases, tmp_path / "cases.csv")
    assert ev.load_cases_csv(tmp_path / "cases.csv") == cases


def test_sweep_behaves_sensibly_and_restores_config(tmp_path):
    before = config.DB_PATH
    cases = ev.build_synthetic_cases(60, 0.3, seed=11, out_dir=tmp_path)
    rows = ev.run_cases(cases, tmp_path)
    assert config.DB_PATH == before                       # eval must not leak its temp DB
    m = ev.CostModel()
    sw = ev.sweep(rows, m)
    assert [r["approve_threshold"] for r in sw] == ev.THRESHOLDS
    # raising the bar can only reduce auto-approvals and fraud auto-approvals
    assert all(a["auto_approve%"] >= b["auto_approve%"] for a, b in zip(sw, sw[1:]))
    assert all(a["fraud_approved"] >= b["fraud_approved"] for a, b in zip(sw, sw[1:]))
    base = ev.baselines(rows, m)
    fraud = sum(1 for r in rows if not r["case"].is_legit)
    assert base["approve every claim (no verification)"]["fraud_approved"] == fraud
    assert base["human-review every claim"]["fraud_approved"] == 0
    assert min(r["total_cost_rs"] for r in sw) < base["approve every claim (no verification)"]["total_cost_rs"]


def test_cheap_prefilters_save_vision_calls(tmp_path):
    rows = ev.run_cases(ev.build_synthetic_cases(80, 0.5, seed=5, out_dir=tmp_path), tmp_path)
    dups = [r for r in rows if r["case"].kind == "fraud_duplicate"]
    assert dups and all(r["vision_calls"] == 0 and "duplicate_photo" in r["final"]["flags"] for r in dups)


def test_case_cost_rules():
    m = ev.CostModel()
    assert ev.case_cost(APPROVE, True, 1000, 1, m)["total"] == m.vision_cost
    assert ev.case_cost(APPROVE, False, 1000, 0, m)["leak"] == 1000
    assert ev.case_cost(ESCALATE, False, 1000, 0, m)["leak"] == (1 - m.review_catch_rate) * 1000


def test_break_even_review_cost():
    m = ev.CostModel(review_cost=10)
    ai = {"total_cost_rs": 10_000, "escalations": 40}      # fixed part (cost without reviews) = 9,600
    human = {"total_cost_rs": 9_000, "escalations": 100}   # fixed part = 8,000
    # equal when 9,600 + 40r == 8,000 + 100r  ->  r = 1,600 / 60
    assert ev.break_even_review_cost(ai, human, m) == 1_600 / 60
    cheaper_anyway = {"total_cost_rs": 1_000, "escalations": 10}   # fixed part 900 < 8,000
    assert ev.break_even_review_cost(cheaper_anyway, human, m) == 0.0
    fewer_reviews_for_humans = {"total_cost_rs": 5_000, "escalations": 10}  # fixed 4,900 < ai fixed, but ai reviews more
    assert ev.break_even_review_cost(ai, fewer_reviews_for_humans, m) is None


def test_cli_end_to_end_writes_the_report(tmp_path):
    assert ev.main(["--n", "30", "--repeats", "1", "--out", str(tmp_path)]) == 0
    assert "Threshold sweep" in (tmp_path / "report.md").read_text()
    assert (tmp_path / "threshold_sweep.csv").exists() and (tmp_path / "predictions.csv").exists()
    # the exported cases.csv can be fed back in as "your own data"
    assert ev.main(["--csv", str(tmp_path / "cases.csv"), "--out", str(tmp_path / "again")]) == 0
