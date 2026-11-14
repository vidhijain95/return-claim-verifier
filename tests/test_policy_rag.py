from app import policy_rag


def test_chunks_follow_policy_sections():
    chunks = policy_rag.chunk_markdown(policy_rag.config.POLICY_PATH.read_text())
    sections = {c["section"] for c in chunks}
    assert {"Return window", "Damaged on arrival", "Auto-approval limits",
            "Pickup with inspection checklist", "Fraud and duplicate evidence"} <= sections


def test_retrieves_the_right_section():
    r = policy_rag.get_retriever()
    assert r.search("return window days electronics")[0]["section"] == "Return window"
    assert r.search("pickup inspection checklist for delivery partner")[0]["section"] == "Pickup with inspection checklist"
    assert r.search("same photo used on another customer's claim")[0]["section"] == "Fraud and duplicate evidence"
    assert r.search("maximum order value for automatic approval")[0]["section"] == "Auto-approval limits"


def test_unrelated_query_returns_nothing():
    assert policy_rag.get_retriever().search("zzzz qqqq") == []


def test_parse_return_windows():
    top = policy_rag.get_retriever().search("return window days glassware")[0]
    assert policy_rag.parse_return_windows(top["text"]) == {
        "electronics": 7, "clothing": 10, "footwear": 10, "glassware": 5}
