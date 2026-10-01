"""The on-disk format of an instrument dataset, and its checks.

Every converter under data/converters/ writes this format, so the training
code reads five datasets the same way and the checks below apply to all of
them.

    data/raw/instrument/<name>/
        meta.json
        train.jsonl  validation.jsonl  test.jsonl

A row is {"id", "text", "text_pair", "label"}: `text_pair` is null for
single-text tasks, `label` is an index into meta.json's `labels`.

The checks are about measurement hygiene, not style: a text that sits in
both train and test inflates every backbone's score, and a converter that
silently keeps such rows would make the instrument lie.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SPLITS = ("train", "validation", "test")

META_KEYS = {
    "name",  # folder name
    "task",  # one line: what is classified
    "source",  # URL the files were taken from
    "source_revision",  # commit hash, or sha256 of the downloaded files
    "license_id",  # as in data/licenses.py
    "labels",  # label names, index = label id
    "splits",  # {"train": n, "validation": n, "test": n}
    "split_origin",  # per split: "source" or how it was carved, with the seed
    "dropped",  # counts of rows the converter removed, by reason
    "notes",
}


@dataclass
class Row:
    id: str
    text: str
    text_pair: str | None
    label: int


@dataclass
class FormatReport:
    errors: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def example_key(row: Row) -> tuple[str, str]:
    """What counts as the same example: the text, ignoring case and outer space."""
    return (row.text.strip().casefold(), (row.text_pair or "").strip().casefold())


def read_split(path: Path) -> list[Row]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if set(raw) != {"id", "text", "text_pair", "label"}:
                raise ValueError(f"{path}:{number}: keys must be id, text, text_pair, label")
            rows.append(Row(raw["id"], raw["text"], raw["text_pair"], raw["label"]))
    return rows


def write_split(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {
                "id": row.id,
                "text": row.text,
                "text_pair": row.text_pair,
                "label": row.label,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def remove_leaks(splits: dict[str, list[Row]]) -> tuple[dict[str, list[Row]], dict[str, int]]:
    """Make the splits disjoint by text, and drop repeats inside each split.

    Test is kept whole. Validation loses what is also in test. Train loses
    what is in either. The evaluation splits are the ones other people's
    numbers refer to, so they are the ones left untouched.
    """
    dropped = {"duplicate_within_split": 0, "validation_in_test": 0, "train_in_eval": 0}
    seen_eval: set[tuple[str, str]] = set()
    cleaned: dict[str, list[Row]] = {}
    for name, reason in (
        ("test", None),
        ("validation", "validation_in_test"),
        ("train", "train_in_eval"),
    ):
        kept, seen_here = [], set()
        for row in splits[name]:
            key = example_key(row)
            if key in seen_here:
                dropped["duplicate_within_split"] += 1
            elif reason and key in seen_eval:
                dropped[reason] += 1
            else:
                seen_here.add(key)
                kept.append(row)
        cleaned[name] = kept
        seen_eval |= seen_here
    return {name: cleaned[name] for name in SPLITS}, dropped


def check_dataset(folder: Path) -> FormatReport:
    report = FormatReport()
    meta_path = folder / "meta.json"
    if not meta_path.is_file():
        report.errors.append(f"{meta_path} is missing")
        return report
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if set(meta) != META_KEYS:
        missing, extra = sorted(META_KEYS - set(meta)), sorted(set(meta) - META_KEYS)
        report.errors.append(f"meta.json keys: missing {missing}, unexpected {extra}")
        return report
    if meta["name"] != folder.name:
        report.errors.append(f"meta name {meta['name']!r} is not the folder name {folder.name!r}")
    labels = meta["labels"]
    if not isinstance(labels, list) or len(labels) < 2 or len(set(labels)) != len(labels):
        report.errors.append("labels must be two or more distinct names")
        return report

    ids: set[str] = set()
    keys: dict[str, set[tuple[str, str]]] = {}
    paired: set[bool] = set()
    for split in SPLITS:
        path = folder / f"{split}.jsonl"
        if not path.is_file():
            report.errors.append(f"{path.name} is missing")
            continue
        try:
            rows = read_split(path)
        except ValueError as exc:
            report.errors.append(str(exc))
            continue
        report.counts[split] = len(rows)
        if meta["splits"].get(split) != len(rows):
            report.errors.append(
                f"{split}: meta says {meta['splits'].get(split)}, file has {len(rows)}"
            )
        seen_labels = set()
        keys[split] = set()
        for row in rows:
            if not isinstance(row.id, str) or row.id in ids:
                report.errors.append(f"{split}: id {row.id!r} is missing or repeated")
            ids.add(row.id)
            if not isinstance(row.text, str) or not row.text.strip():
                report.errors.append(f"{split}: row {row.id} has no text")
                continue
            if row.text_pair is not None and not str(row.text_pair).strip():
                report.errors.append(f"{split}: row {row.id} has an empty text_pair")
            paired.add(row.text_pair is not None)
            if isinstance(row.label, bool) or not isinstance(row.label, int):
                report.errors.append(f"{split}: row {row.id} label must be an integer")
            elif not 0 <= row.label < len(labels):
                report.errors.append(f"{split}: row {row.id} label {row.label} is out of range")
            else:
                seen_labels.add(row.label)
            if example_key(row) in keys[split]:
                report.errors.append(f"{split}: row {row.id} repeats a text inside the split")
            keys[split].add(example_key(row))
        # A class that never occurs in train cannot be learned, and one that
        # never occurs in test makes macro-F1 undefined for it.
        if split in ("train", "test") and len(seen_labels) != len(labels):
            absent = [labels[i] for i in range(len(labels)) if i not in seen_labels]
            report.errors.append(f"{split}: no rows for labels {absent}")
    if len(paired) > 1:
        report.errors.append("some rows have a text_pair and some do not")

    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if a in keys and b in keys:
            shared = len(keys[a] & keys[b])
            if shared:
                report.errors.append(f"{shared} texts appear in both {a} and {b}")
    return report
