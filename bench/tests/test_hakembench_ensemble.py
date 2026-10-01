"""AI-ensemble gold and the blind audit sample."""

from bench.hakembench import ensemble as E


def test_unanimity_needs_every_answer_present_and_equal():
    assert E.unanimous(["a", "a", "a"])
    assert not E.unanimous(["a", "a", None])
    assert not E.unanimous(["a", "b", "a"])


def test_allocation_is_proportional_and_capped():
    assert E.allocate({"x": 30, "y": 10}, 8) == {"x": 6, "y": 2}
    assert (
        E.allocate({"x": 1, "y": 100}, 10) == {"x": 0, "y": 10}
        or sum(E.allocate({"x": 1, "y": 100}, 10).values()) == 10
    )
    assert E.allocate({"x": 2}, 10) == {"x": 2}


def test_the_audit_is_stratified_and_leaves_out_seen_and_priority_questions():
    votes = {(f"u{i}", "q"): ["a", "a"] for i in range(40)}
    votes |= {(f"d{i}", "q"): ["a", "b"] for i in range(40)}
    votes |= {("k0", "q"): ["a", "a"], ("p0", E.PRIORITY): ["1", "2"]}
    tracks = {pair[0]: "spam" for pair in votes}
    sizes = {"adjudicated": 10, "unanimous": 7, "known_truth": 5}
    chosen = E.audit_sample(votes, tracks, {("k0", "q")}, {("u0", "q")}, sizes)
    assert len(chosen["adjudicated"]) == 10 and len(chosen["unanimous"]) == 7
    assert chosen["known_truth"] == [("k0", "q")]
    picked = {p for v in chosen.values() for p in v}
    assert ("u0", "q") not in picked and ("p0", E.PRIORITY) not in picked
    assert all(p[0].startswith("d") for p in chosen["adjudicated"])


def test_adjudication_sees_distinct_answers_and_reasons_without_labels():
    units = [{"id": "u", "track": "spam", "text": "t", "questions": [
        {"qid": "q", "question": "?", "type": "noul", "options": []},
        {"qid": "r", "question": "?", "type": "noul", "options": []}]}]  # fmt: skip
    votes = {("u", "q"): ["true", "false", "true"], ("u", "r"): ["true", "true"]}
    research = {"u": {"q": {"reason_tr": "bir"}}}
    second = {"u": {"q": {"reason_tr": "iki"}}}
    out = E.adjudication_units(units, votes, research, second)
    assert len(out) == 1 and [q["qid"] for q in out[0]["questions"]] == ["q"]
    q = out[0]["questions"][0]
    assert sorted(q["candidates"]) == ["false", "true"] and sorted(q["reasons"]) == ["bir", "iki"]


def test_the_audit_report_counts_disagreement_and_leaves_ambiguous_out():
    chosen = {"unanimous": [["a", "q"], ["b", "q"]], "adjudicated": [["c", "q"], ["d", "q"]]}
    owner = {("a", "q"): "x", ("b", "q"): None, ("c", "q"): "y", ("d", "q"): "x"}
    gold = {"a~q": {"answer": "x"}, "b~q": {"answer": "x"}, "c~q": {"answer": "x"},
            "d~q": {"answer": "x"}}  # fmt: skip
    out = E.audit_report(chosen, owner, gold, {"unanimous": 90, "adjudicated": 10})
    assert out["unanimous"]["ambiguous"] == 1 and out["unanimous"]["differ"] == 0
    assert out["adjudicated"]["rate"] == 0.5
    assert out["pooled_by_stratum_weight"] == 0.05
    assert E.wilson(0, 70)[1] < 0.06
