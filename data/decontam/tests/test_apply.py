"""Flagged rows leave the training file; the rest stay in order."""

import json

from data.decontam.apply import apply


def test_flagged_rows_are_removed_by_id(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"per_row": [
        {"row_id": "a", "flagged": False}, {"row_id": "b", "flagged": True},
        {"row_id": "c", "flagged": False}]}))  # fmt: skip
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps({"row_id": i, "x": 1}) + "\n" for i in "abc"))
    assert apply(report, rows, tmp_path / "ledger.txt") == {"rows": 3, "removed": 1, "kept": 2}
    assert [json.loads(x)["row_id"] for x in rows.read_text().splitlines()] == ["a", "c"]


def test_a_row_flagged_once_stays_out_after_a_rebuild(tmp_path):
    ledger = tmp_path / "ledger.txt"
    first = tmp_path / "first.json"
    first.write_text(json.dumps({"per_row": [{"row_id": "b", "flagged": True}]}))
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps({"row_id": i}) + "\n" for i in "ab"))
    apply(first, rows, ledger)
    # Rebuilt from raw; a new report made on the cleaned file no longer flags "b".
    rows.write_text("".join(json.dumps({"row_id": i}) + "\n" for i in "ab"))
    later = tmp_path / "later.json"
    later.write_text(json.dumps({"per_row": [{"row_id": "a", "flagged": False}]}))
    assert apply(later, rows, ledger)["removed"] == 1
