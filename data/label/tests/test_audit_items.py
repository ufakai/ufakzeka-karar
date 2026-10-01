"""The owner's audit of flagged rows: the verdict that decides whether flags apply."""

from data.label.audit_items import verdict


def test_flags_apply_only_when_the_owner_sides_with_the_model_on_half():
    scores = {f"r{i}": {"probs": {"a": 0.9, "b": 0.1}, "target": {"a": 0.0, "b": 1.0}}
              for i in range(4)}  # fmt: skip
    audit = [{"id": f"r{i}"} for i in range(4)]
    labels = {"r0": {"answer": "a"}, "r1": {"answer": "a"}, "r2": {"answer": "b"},
              "r3": {"answer": None, "flag": "cant_tell"}}  # fmt: skip
    out = verdict(audit, labels, scores)
    assert out == {"answered": 3, "owner_with_model": 2, "owner_with_target": 1,
                   "flags_apply": True}  # fmt: skip
    labels["r1"] = {"answer": "b"}
    assert verdict(audit, labels, scores)["flags_apply"] is False
