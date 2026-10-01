"""TrCOLA to the instrument format: is a Turkish sentence acceptable or not.

Loaded from the standalone TrCOLA repository, which its authors publish
under CC BY 4.0. The TrGLUE bundle carries the same data under CC BY-SA
and is not used.
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
    write_dataset,
)
from data.instrument_format import Row, remove_leaks

NAME = "trcola"
REPO = "turkish-nlp-suite/TrCOLA"
REVISION = "84dc7a3641c0d6997b4970f0d4d8e26811904980"
LICENSE_ID = "CC-BY-4.0"
FILES = {
    "train": "data/train.jsonl",
    "validation": "data/validation.jsonl",
    "test": "data/test.jsonl",
}
# `id` is absent from the rows whose var_type is None, so it is not required
# and not used. `orig` is the uncorrupted original, kept by the authors for
# research and not part of the task.
COLUMNS = ("variation", "label")
LABELS = ["unacceptable", "acceptable"]


def convert(out_root: Path, cache_dir: Path, *, fetch: FetchFile = fetch_file) -> Path:
    paths = {split: fetch(REPO, name, REVISION, cache_dir) for split, name in FILES.items()}
    shas = {FILES[split]: sha256_of(path) for split, path in paths.items()}

    own = {"empty_text": 0, "unknown_label": 0}
    splits: dict[str, list[Row]] = {}
    for split, path in paths.items():
        rows: list[Row] = []
        for index, record in enumerate(read_rows(path, COLUMNS)):
            text = str(record["variation"] or "").strip()
            label = record["label"]
            if not text:
                own["empty_text"] += 1
                continue
            if isinstance(label, bool) or label not in (0, 1):
                own["unknown_label"] += 1
                continue
            rows.append(Row(f"{NAME}-{split}-{index:05d}", text, None, int(label)))
        splits[split] = rows

    splits, leaks = remove_leaks(splits)
    meta = {
        "name": NAME,
        "task": "binary linguistic acceptability of a Turkish sentence",
        "source": dataset_url(REPO),
        "source_revision": REVISION,
        "license_id": LICENSE_ID,
        "labels": LABELS,
        "splits": {split: len(rows) for split, rows in splits.items()},
        "split_origin": dict.fromkeys(splits, "source"),
        "dropped": merge_counts(own, leaks),
        "notes": (
            "The judged sentence is the `variation` column. Label 1 is acceptable and 0 is not: "
            "the card's worked example gives a morphological violation label 0, and every row "
            "with var_type None, where variation equals orig, carries label 1. The card writes "
            "the negative class as 'nonacceptable'. Source files: " + files_note(shas)
        ),
    }
    return write_dataset(out_root, meta, splits)


if __name__ == "__main__":
    raise SystemExit(main_for(convert, __doc__ or NAME))
