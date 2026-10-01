"""OffensEval-TR 2020 subtask A: offensive or not, in the instrument format.

The Hugging Face repository coltekin/offenseval2020_tr is a loader script,
which the datasets library no longer runs, so the original zip is downloaded
from the author's page and parsed here. Its README warns that the TSV files
use no quoting and no escapes, so the lines are split on tabs by hand rather
than read with a csv reader. The source has train and test only; validation
is carved from train.
"""

from __future__ import annotations

import functools
import random
import sys
import zipfile
from pathlib import Path

from data.converters.common import (
    Fetcher,
    cached_archive,
    http_get,
    run_cli,
    sha256_file,
    stratified_carve,
    write_dataset,
)
from data.instrument_format import Row, remove_leaks

NAME = "offenseval_tr"
SOURCE_URL = "https://coltekin.github.io/offensive-turkish/offenseval2020-turkish.zip"
ARCHIVE_NAME = "offenseval2020-turkish.zip"
# The archive as read on 2026-09-19. A different file is a different dataset, so the
# command line refuses it; a new upstream release needs a dated decision.
EXPECTED_SHA256 = "7977e96255dbc9b8d14893f1b14cbe3dec53c70358503c062c5a59720ec9c2f2"

_ROOT = "offenseval2020-turkish"
TRAIN_MEMBER = f"{_ROOT}/offenseval-tr-training-v1/offenseval-tr-training-v1.tsv"
TEST_MEMBER = f"{_ROOT}/offenseval-tr-testset-v1/offenseval-tr-testset-v1.tsv"
TEST_LABEL_MEMBER = f"{_ROOT}/offenseval-tr-testset-v1/offenseval-tr-labela-v1.tsv"

# The spelling used in the source files and in its annotation guide.
LABELS = ["NOT", "OFF"]

VALIDATION_FRACTION = 0.1
VALIDATION_SEED = 1


def _lines(archive: zipfile.ZipFile, member: str) -> list[str]:
    return [line for line in archive.read(member).decode("utf-8").split("\n") if line]


def _rows(archive: zipfile.ZipFile, member: str, columns: list[str]) -> list[list[str]]:
    """Read a tab separated member, checking the header names against `columns`."""
    lines = _lines(archive, member)
    if not lines or lines[0].split("\t") != columns:
        head = lines[0] if lines else ""
        raise RuntimeError(f"{member}: expected the header {columns}, found {head!r}")
    rows = [line.split("\t") for line in lines[1:]]
    bad = next((row for row in rows if len(row) != len(columns)), None)
    if bad is not None:
        raise RuntimeError(f"{member}: a line has {len(bad)} fields, not {len(columns)}")
    return rows


def _test_labels(archive: zipfile.ZipFile) -> dict[str, str]:
    """The gold test labels, a comma separated file of id,label with no header."""
    labels: dict[str, str] = {}
    for line in _lines(archive, TEST_LABEL_MEMBER):
        source_id, _, label = line.partition(",")
        if not label:
            raise RuntimeError(f"{TEST_LABEL_MEMBER}: {line!r} is not id,label")
        labels[source_id] = label
    return labels


def _to_rows(triples: list[tuple[str, str, str]], dropped: dict[str, int]) -> list[Row]:
    """Make Rows keyed by the source id from (id, tweet, label name) triples."""
    index_of = {label: index for index, label in enumerate(LABELS)}
    rows: list[Row] = []
    for source_id, tweet, label in triples:
        if label not in index_of:
            raise RuntimeError(
                f"row {source_id}: unknown label {label!r}, expected one of {LABELS}"
            )
        # The source already replaced mentions with @USER and links with URL.
        # Nothing else is touched here, not the case and not the emoji.
        text = tweet.strip()
        if not text:
            dropped["empty_text"] += 1
            continue
        rows.append(Row(source_id, text, None, index_of[label]))
    return rows


def _prefixed(split: str, rows: list[Row]) -> list[Row]:
    """Ids carry their split, so a train row and a test row cannot collide."""
    return [Row(f"{split}-{row.id}", row.text, row.text_pair, row.label) for row in rows]


def _parse(archive_path: Path, dropped: dict[str, int]) -> dict[str, list[Row]]:
    with zipfile.ZipFile(archive_path) as archive:
        train_lines = _rows(archive, TRAIN_MEMBER, ["id", "tweet", "subtask_a"])
        test_lines = _rows(archive, TEST_MEMBER, ["id", "tweet"])
        gold = _test_labels(archive)

    missing = [source_id for source_id, _ in test_lines if source_id not in gold]
    if missing:
        raise RuntimeError(f"{TEST_LABEL_MEMBER}: no label for test ids {missing[:5]}")

    train = _to_rows([(i, t, label) for i, t, label in train_lines], dropped)
    test = _to_rows([(i, t, gold[i]) for i, t in test_lines], dropped)
    rest, carved = stratified_carve(train, VALIDATION_FRACTION, random.Random(VALIDATION_SEED))
    return {
        "train": _prefixed("train", rest),
        "validation": _prefixed("validation", carved),
        "test": _prefixed("test", test),
    }


def convert(
    out_root: Path,
    cache_dir: Path,
    *,
    source_url: str = SOURCE_URL,
    fetch: Fetcher = http_get,
    expected_sha256: str | None = None,
) -> Path:
    archive = cached_archive(source_url, cache_dir / ARCHIVE_NAME, fetch)
    if expected_sha256 is not None and sha256_file(archive) != expected_sha256:
        raise RuntimeError(
            f"{archive} is not the pinned archive (sha256 {sha256_file(archive)}, "
            f"expected {expected_sha256})"
        )
    dropped = {"empty_text": 0}
    splits = _parse(archive, dropped)
    splits, leaked = remove_leaks(splits)
    dropped.update(leaked)

    empty = [split for split, rows in splits.items() if not rows]
    if empty:
        raise RuntimeError(f"{NAME}: no rows left for splits {empty}")

    carve = (
        f"carved from train: {round(VALIDATION_FRACTION * 100)} percent, stratified by label, "
        f"random.Random({VALIDATION_SEED})"
    )
    meta = {
        "task": "Whether a Turkish tweet is offensive, OffensEval 2020 subtask A.",
        "source": source_url,
        "source_revision": f"sha256:{sha256_file(archive)}",
        "license_id": "CC-BY-2.0",
        "labels": list(LABELS),
        "split_origin": {
            "train": "source file offenseval-tr-training-v1.tsv, less the validation carve",
            "validation": carve,
            "test": "source file offenseval-tr-testset-v1.tsv with the gold labels "
            "of offenseval-tr-labela-v1.tsv, joined on the id",
        },
        "dropped": dropped,
        "notes": (
            f"Version 1 of the corpus, members under {_ROOT}/ of {ARCHIVE_NAME}. Text is the "
            "tweet field with outer whitespace stripped; the source already replaced user "
            "mentions with @USER and links with URL, and nothing else is normalised. The "
            "validation split is carved before leak removal, so its final size can be under "
            "the tenth. Subtask A only: the archive carries no other subtask for Turkish."
        ),
    }
    return write_dataset(out_root, NAME, splits, meta)


if __name__ == "__main__":
    pinned = functools.partial(convert, expected_sha256=EXPECTED_SHA256)
    sys.exit(run_cli(pinned, __doc__.splitlines()[0]))
