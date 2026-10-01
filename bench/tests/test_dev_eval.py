"""Scoring a model on HakemBench-dev beside the panel."""

import pytest

from bench.dev_eval import score


def test_model_and_panel_are_scored_on_the_same_items():
    dev = [{"id": f"r{i}", "task": "t", "target": {"true": 1.0, "false": 0.0}} for i in range(4)]
    model = {f"r{i}": {"true": 0.8, "false": 0.2} for i in range(3)}
    panel = {f"r{i}": {"true": 0.4, "false": 0.6} for i in range(4)}
    out = score(dev, model, panel, draws=50)
    assert out["rows"] == 3 and out["missing"] == 1
    assert out["model"]["accuracy"] == 1.0 and out["panel"]["accuracy"] == 0.0
    assert out["model_minus_panel_accuracy"]["difference"] == 1.0
    assert out["model"]["brier"] == pytest.approx(2 * 0.2**2)


def test_macro_f1_is_averaged_over_tasks():
    from bench.dev_eval import macro_f1_over_tasks

    dev = [{"id": "a", "task": "x", "target": {"t": 1.0, "f": 0.0}},
           {"id": "b", "task": "y", "target": {"t": 0.0, "f": 1.0}}]  # fmt: skip
    right = {"a": {"t": 0.9, "f": 0.1}, "b": {"t": 0.1, "f": 0.9}}
    assert macro_f1_over_tasks(dev, right) == 1.0
