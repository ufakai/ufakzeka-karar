"""MiDe22 to the instrument format: the misinformation label of a Turkish tweet.

The source has one split, so train, validation and test are carved from it
with a fixed seed, stratified by label.
"""

from __future__ import annotations

from pathlib import Path

from data.converters.hub_common import (
    FetchFile,
    dataset_url,
    fetch_file,
    files_note,
    main_for,
    merge_counts,
    read_rows,
    sha256_of,
    stratified_carve,
    write_dataset,
)
from data.instrument_format import SPLITS, Row, remove_leaks

NAME = "mide22"
REPO = "ogozcelik/turkish-fake-news-detection"
REVISION = "6b6bb45712d746e28906913e953d5525f5c8c633"
LICENSE_ID = "MIT"
SOURCE_FILE = "mide22_all_tr.tsv"
COLUMNS = ("tweet", "label")
LABELS = ["False", "Other", "True"]
SHARES = {"train": 0.7, "validation": 0.1, "test": 0.2}
SEED = 1


def convert(out_root: Path, cache_dir: Path, *, fetch: FetchFile = fetch_file) -> Path:
    path = fetch(REPO, SOURCE_FILE, REVISION, cache_dir)
    shas = {SOURCE_FILE: sha256_of(path)}

    own = {"empty_text": 0, "unknown_label": 0, "duplicate_text": 0}
    pool: list[Row] = []
    seen: set[str] = set()
    for index, record in enumerate(read_rows(path, COLUMNS)):
        text = str(record["tweet"] or "").strip()
        label = str(record["label"] or "").strip()
        if not text:
            own["empty_text"] += 1
            continue
        if label not in LABELS:
            own["unknown_label"] += 1
            continue
        # The carve must not put two copies of one tweet in different splits,
        # so exact repeats go before the split is drawn, not after.
        if text in seen:
            own["duplicate_text"] += 1
            continue
        seen.add(text)
        pool.append(Row(f"{NAME}-{index:05d}", text, None, LABELS.index(label)))

    carved = stratified_carve(pool, SHARES, lambda row: row.label, SEED)
    splits, leaks = remove_leaks({split: carved[split] for split in SPLITS})
    origin = (
        f"carved from the single source split, {SHARES}, stratified by label, random.Random({SEED})"
    )
    meta = {
        "name": NAME,
        "task": "three-way misinformation label of a Turkish tweet: False, Other or True",
        "source": dataset_url(REPO),
        "source_revision": REVISION,
        "license_id": LICENSE_ID,
        "labels": LABELS,
        "splits": {split: len(rows) for split, rows in splits.items()},
        "split_origin": dict.fromkeys(splits, origin),
        "dropped": merge_counts(own, leaks),
        "notes": (
            "Row ids are the source file's row numbers, so a row keeps its id whatever split "
            "it lands in. Labels are the card's five-annotator labels, unchanged. "
            "Source files: " + files_note(shas)
        ),
    }
    return write_dataset(out_root, meta, splits)


if __name__ == "__main__":
    raise SystemExit(main_for(convert, __doc__ or NAME))
