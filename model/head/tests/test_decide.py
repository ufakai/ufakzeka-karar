"""The rule reads the paired-bootstrap interval; only a clear win keeps the rung."""

import json
import random

from model.head.decide import decide


def run(path, accuracy, seed, rows=300):
    rng = random.Random(seed)
    lines = []
    for task in ("a", "b"):
        for i in range(rows):
            gold = ["x", "y", "z"][i % 3]
            right = rng.random() < accuracy
            guess = gold if right else ["x", "y", "z"][(i + 1) % 3]
            keys = ["x", "y", "z"]
            probs = [0.8 if k == guess else 0.1 for k in keys]
            target = {k: float(k == gold) for k in keys}
            record = {"row_id": f"{task}{i}", "task": task, "keys": keys, "probs": probs}
            lines.append(json.dumps(record | {"target": target}))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_a_clear_win_keeps_the_converted_backbone(tmp_path):
    better = [run(tmp_path / f"a{s}.jsonl", 0.9, s) for s in range(3)]
    worse = [run(tmp_path / f"b{s}.jsonl", 0.6, 10 + s) for s in range(3)]
    out = decide(better, worse, draws=300)
    assert out["interval"][0] > 0 and out["verdict"] == "keep the converted backbone"
    assert out["rows_per_task"] == {"a": 300, "b": 300}


def test_equal_systems_keep_the_causal_backbone(tmp_path):
    first = [run(tmp_path / f"a{s}.jsonl", 0.8, s) for s in range(3)]
    second = [run(tmp_path / f"b{s}.jsonl", 0.8, 20 + s) for s in range(3)]
    out = decide(first, second, draws=300)
    assert out["verdict"].endswith("keep the causal backbone")
    assert out["interval"][0] < 0 < out["interval"][1]


def test_excluded_tasks_are_left_out_of_the_statistic(tmp_path):
    first = [run(tmp_path / f"a{s}.jsonl", 0.9, s) for s in range(2)]
    second = [run(tmp_path / f"b{s}.jsonl", 0.6, 10 + s) for s in range(2)]
    kept = decide(first, second, draws=50, exclude=frozenset({"b"}))
    alone = decide(first, second, draws=50, only=frozenset({"b"}))
    assert kept["tasks"] == ["a"] and list(kept["rows_per_task"]) == ["a"]
    assert alone["tasks"] == ["b"]
    both = decide(first, second, draws=50)
    assert abs(both["difference"] - (kept["difference"] + alone["difference"]) / 2) < 1e-9
