"""model/head/mix_counts.py on a fake data tree and run record."""

import hashlib
import json
from pathlib import Path

import pytest

from model.head import mix_counts as M


def tree(tmp_path: Path) -> tuple[Path, dict]:
    data = tmp_path / "built"
    rows = {
        "typed/a/train.jsonl": [{"task": "a", "row_id": f"a{i}", "state": f"a {i}"}
                                for i in range(5)],
        "r4/train.jsonl": [{"task": "b", "row_id": f"b{i}", "state": f"b {i}"} for i in range(4)]
        + [{"task": "c", "row_id": f"c{i}", "state": f"c {i}"} for i in range(3)],
    }  # fmt: skip
    hashes = {}
    for name, body in rows.items():
        path = data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in body), encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {
        "spec": {"name": "fake-s1", "task_fraction": {"c": 0.0}},
        "data_hashes": hashes,
        "task_fraction_rows": {"c": {"before": 3, "after": 0}},
        "mixed_rows": 9,
    }
    return data, record


def write(tmp_path: Path, record: dict) -> Path:
    path = tmp_path / "fake-s1.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_counts_per_task_and_file(tmp_path):
    data, record = tree(tmp_path)
    out = M.counts(write(tmp_path, record), data)
    assert out["rows_per_task"] == {"a": 5, "b": 4}
    assert out["rows_per_file"] == {"r4/train.jsonl": 4, "typed/a/train.jsonl": 5}
    assert out["mixed_rows"] == 9 and out["checked"]["rows_before_fraction"] == {"c": 3}


def test_refuses_a_changed_file_or_a_wrong_total(tmp_path):
    data, record = tree(tmp_path)
    with pytest.raises(SystemExit, match="mixed rows"):
        M.counts(write(tmp_path, {**record, "mixed_rows": 10}), data)
    (data / "typed/a/train.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="not the file"):
        M.counts(write(tmp_path, record), data)
