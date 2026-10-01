"""Triage: disagreements and a fifth of agreements go to the owner; the reviewer is measured."""

from bench.hakembench.triage import report, route


def unit(i, proposed="true", track="spam"):
    return {"id": f"u{i}", "track": track, "order": i,
            "questions": [{"qid": "q", "type": "noul", "proposed": proposed}]}  # fmt: skip


def test_disputes_and_a_fifth_of_agreements_go_to_the_owner():
    units = [unit(i) for i in range(100)]
    review = {f"u{i}": {"q": {"answer": "false" if i < 10 else "true", "flag": "none"}}
              for i in range(100)}  # fmt: skip
    review["u50"] = {"q": {"answer": None, "flag": "cant_tell"}}
    reasons = route(units, review, checked={"u99"})
    assert sum(r == "disputed" for r in reasons.values()) == 11
    assert reasons["u99"] == "checked"
    assert sum(r == "sampled" for r in reasons.values()) == round(0.2 * 88)


def test_the_reviewer_is_scored_on_catches_and_against_the_panel():
    units = [unit(0), unit(1), unit(2, proposed="false")]
    catches = {("u2", "q"): {"truth": "true", "shown": "false"}}
    review = {"u0": {"q": {"answer": "true", "flag": "none"}},
              "u1": {"q": {"answer": "false", "flag": "none"}},
              "u2": {"q": {"answer": "true", "flag": "none"}}}  # fmt: skip
    out = report(units, review, catches, {"u1": "disputed", "u2": "disputed"})
    assert out["reviewer_agrees_with_proposal"] == {"spam:noul": 0.5}
    assert out["reviewer_on_catch_items"]["caught"] == 1
    assert out["not_shown"] == 1
