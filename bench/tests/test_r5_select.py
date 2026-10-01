"""r5's selection: the guardrail dev metrics and the rule, on made-up predictions."""

import json

import pytest

from bench import r5_select
from bench.r5_select import guard_metrics, guard_rows, select


def record(row_id, attack, p_true, task=r5_select.TASK):
    return {"row_id": row_id, "task": task, "keys": ["true", "false"],
            "probs": [p_true, 1 - p_true],
            "target": {"true": float(attack), "false": 1.0 - attack}}  # fmt: skip


def test_rows_are_read_from_the_dev_task_only_and_must_be_complete():
    records = [record("a", True, 0.9), record("b", False, 0.2), record("c", True, 0.9, "other")]
    assert guard_rows(records, {"a", "b"}) == [(True, True), (False, False)]
    with pytest.raises(ValueError, match="missing"):
        guard_rows(records, {"a", "b", "d"})


def test_metrics_count_recall_benign_passes_and_macro_f1():
    # 4 attacks, 3 caught; 6 benign, 5 passed.
    rows = [(True, True)] * 3 + [(True, False)] + [(False, False)] * 5 + [(False, True)]
    m = guard_metrics(rows, draws=200)
    assert m["attack_recall"] == pytest.approx(0.75)
    assert m["benign_pass_rate"] == pytest.approx(5 / 6)
    f1_attack, f1_benign = 2 * 3 / (6 + 1 + 1), 2 * 5 / (10 + 1 + 1)
    assert m["macro_f1"] == pytest.approx((f1_attack + f1_benign) / 2)
    low, high = m["macro_f1_interval"]
    assert low <= m["macro_f1"] <= high
    assert (m["attacks"], m["benign"], m["attacks_caught"], m["benign_passed"]) == (4, 6, 3, 5)


def run(recall, guard_f1, dev):
    return {"guard": {"attack_recall": recall, "macro_f1": guard_f1}, "dev_macro_f1": dev}


def test_the_recall_bar_comes_first_then_hakembench_dev():
    runs = {"a": run(0.95, 0.90, 0.60), "b": run(0.79, 0.99, 0.70), "c": run(0.80, 0.85, 0.65)}
    result = select(runs)
    assert result["qualifying"] == ["a", "c"]
    assert result["chosen"] == "c" and result["met_recall_bar"]


def test_a_dev_tie_goes_to_the_higher_guardrail_f1():
    runs = {"a": run(0.9, 0.80, 0.65), "b": run(0.9, 0.85, 0.65)}
    assert select(runs)["chosen"] == "b"


def test_with_no_qualifying_run_the_best_guardrail_f1_ships_and_says_so():
    runs = {"a": run(0.5, 0.70, 0.9), "b": run(0.7, 0.75, 0.6)}
    result = select(runs)
    assert result["chosen"] == "b" and not result["met_recall_bar"]
    assert "no run reached" in result["reason"]


def test_main_writes_the_selection_with_the_flag(tmp_path, monkeypatch):
    dev = tmp_path / "validation.jsonl"
    dev.write_text("".join(json.dumps({"row_id": i}) + "\n" for i in ("a", "b", "c", "d")))
    predictions = {
        "r5a-base-s1": [record("a", True, 0.9), record("b", True, 0.8), record("c", False, 0.1),
                        record("d", False, 0.6)],
        "r5b-base-s1": [record("a", True, 0.2), record("b", True, 0.3), record("c", False, 0.1),
                        record("d", False, 0.1)],
    }  # fmt: skip
    for name, records in predictions.items():
        path = tmp_path / name / "predictions_step9.jsonl"
        path.parent.mkdir()
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
    import model.head.accept as accept
    import model.head.compare as compare

    monkeypatch.setattr(accept, "last_step", lambda run, cache: "predictions_step9.jsonl")
    monkeypatch.setattr(compare, "fetch", lambda run, cache, name: tmp_path / run / name)
    monkeypatch.setattr(
        r5_select, "dev_macro_f1", lambda run, cache: {"r5a-base-s1": 0.6, "r5b-base-s1": 0.7}[run]
    )
    monkeypatch.setattr(r5_select, "DEV_FILE", dev)
    out = tmp_path / "selection.json"
    r5_select.main(["--runs", "r5a-base-s1,r5b-base-s1", "--draws", "50", "--out", str(out)])
    result = json.loads(out.read_text())
    assert result["chosen"] == "r5a-base-s1" and result["qualifying"] == ["r5a-base-s1"]
    assert r5_select.FLAG in result["flag"]
    assert result["runs"]["r5b-base-s1"]["guard"]["attack_recall"] == 0.0
