"""Round 2's row scores: averaged probabilities, the flag, trivial rows, the weights."""

import json

import pytest

from model.head.scoring import (
    TRIVIAL_WEIGHT,
    confident_flags,
    load_scores,
    margin,
    memorised,
    trivial_rows,
    weights,
)


def row(task, label, probs):
    return {"task": task, "target": {k: float(k == label) for k in probs}, "probs": probs}


def test_scores_are_averaged_over_files(tmp_path):
    for i, p in enumerate((0.2, 0.6)):
        record = {"row_id": "r", "task": "t", "keys": ["a", "b"], "probs": [p, 1 - p],
                  "target": {"a": 1.0, "b": 0.0}}  # fmt: skip
        (tmp_path / f"{i}.jsonl").write_text(json.dumps(record) + "\n")
    rows = load_scores(sorted(tmp_path.glob("*.jsonl")))
    assert rows["r"]["probs"]["a"] == pytest.approx(0.4)
    assert margin(rows["r"]) == pytest.approx(-0.2)


def test_a_confidently_contradicted_label_is_flagged_and_a_merely_hard_one_is_not():
    rows = {f"a{i}": row("t", "a", {"a": 0.9, "b": 0.1}) for i in range(20)}
    rows |= {f"b{i}": row("t", "b", {"a": 0.2, "b": 0.8}) for i in range(20)}
    rows["wrong"] = row("t", "a", {"a": 0.05, "b": 0.95})  # the model is sure it is b
    rows["hard"] = row("t", "a", {"a": 0.45, "b": 0.55})  # unsure, under b's threshold
    flagged = confident_flags(rows)
    assert "wrong" in flagged and "hard" not in flagged
    assert not any(r.startswith(("a", "b")) and r != "wrong" for r in flagged)


def test_trivial_rows_are_read_inside_each_task():
    rows = {f"x{i}": row("x", "a", {"a": 0.5 + i / 40, "b": 0.5 - i / 40}) for i in range(20)}
    rows |= {f"y{i}": row("y", "a", {"a": 0.9, "b": 0.1}) for i in range(4)}
    easy = trivial_rows(rows, {})
    assert {f"x{i}" for i in range(15, 20)} <= easy and "x0" not in easy
    assert all(f"y{i}" in easy for i in range(4))  # a tie at the top counts as the quarter


def test_weights_keep_a_floor_of_easy_weight_and_flags_wait_for_the_audit():
    rows = {f"a{i}": row("t", "a", {"a": 0.95, "b": 0.05}) for i in range(10)}
    rows["wrong"] = row("t", "b", {"a": 0.97, "b": 0.03})
    out, report = weights(rows, {}, flags_apply=False)
    assert "wrong" not in out and report["flagged"] == 0
    assert all(v in (TRIVIAL_WEIGHT, 1.0) for v in out.values())
    out, report = weights(rows, {}, flags_apply=True)
    assert out["wrong"] == 0.25 and report["flagged"] == 1


def test_memorisation_is_called_only_when_trained_rows_look_far_easier():
    unseen = {f"u{i}": row("t", "a", {"a": 0.6, "b": 0.4}) for i in range(5)}
    fine = {f"s{i}": row("t", "a", {"a": 0.65, "b": 0.35}) for i in range(5)}
    learnt = {f"s{i}": row("t", "a", {"a": 0.99, "b": 0.01}) for i in range(5)}
    assert not memorised(fine, unseen)["memorised"]
    assert memorised(learnt, unseen)["memorised"]


def test_the_audit_is_a_uniform_draw_of_the_flags():
    from model.head.scoring import audit_sample

    rows = {f"{t}{i}": {"task": t} for t, n in (("x", 90), ("y", 10)) for i in range(n)}
    picked = audit_sample(set(rows), rows, count=50)
    assert len(set(picked)) == 50 and picked == audit_sample(set(rows), rows, count=50)
    assert sum(rows[r]["task"] == "x" for r in picked) > 35


def test_a_small_class_keeps_its_own_flags_beside_a_large_one():
    rows = {f"a{i}": row("t", "a", {"a": 0.9, "b": 0.05, "c": 0.05}) for i in range(60)}
    rows |= {f"b{i}": row("t", "b", {"a": 0.1, "b": 0.8, "c": 0.1}) for i in range(8)}
    rows |= {f"c{i}": row("t", "c", {"a": 0.1, "b": 0.1, "c": 0.8}) for i in range(8)}
    for i in range(6):  # many large-class rows the model reads as c
        rows[f"a{i}"] = row("t", "a", {"a": 0.02, "b": 0.03, "c": 0.95})
    rows["b0"] = row("t", "b", {"a": 0.1, "b": 0.05, "c": 0.85})  # one small-class row read as c
    flagged = confident_flags(rows)
    assert "b0" in flagged and {f"a{i}" for i in range(6)} <= flagged


def test_crossfit_weights_come_from_each_folds_held_out_scores(tmp_path):
    import argparse

    from model.head.scoring import crossfit

    files = {}
    for f in range(2):
        records, fold = [], []
        for i in range(20):
            rid = f"f{f}r{i}"
            records.append(
                {
                    "row_id": rid,
                    "task": "t",
                    "keys": ["a", "b"],
                    "probs": [0.9 - i / 40, 0.1 + i / 40],
                    "target": {"a": 1.0, "b": 0.0},
                }
            )
            fold.append({"row_id": rid, "question": {"type": "noul"}})
        for name, rows in (("scores_step9.jsonl", records), ("fold.jsonl", fold)):
            path = tmp_path / f"fold{f}_{name}"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            files[(f"r2-fold{f}", name)] = path
    args = argparse.Namespace(round="b", flags=False, out=tmp_path / "out")
    assert crossfit(args, ["r2-fold0", "r2-fold1"], lambda run, name: files[(run, name)],
                    lambda run: ["scores_step9.jsonl"]) == 0  # fmt: skip
    weights_b = json.loads((tmp_path / "out" / "weights_b.json").read_text())
    report = json.loads((tmp_path / "out" / "weights_b_report.json").read_text())
    assert report["rows"] == 40 and report["crossfit"] is True
    assert set(weights_b.values()) <= {0.5, 1.0} and len(weights_b) == report["trivial"]
