"""MASSIVE 1.1, locale tr-TR: intent classification in the instrument format.

The Hugging Face repository AmazonScience/massive is a loader script, which
the datasets library no longer runs, so the original archive is downloaded
from Amazon's bucket and parsed here. The archive holds one JSON Lines file
per locale; only tr-TR is read. Splits come from each row's `partition`
field, the source's own division.

The corpus defines 60 intents. tr-TR has no test rows for cooking_query, so
that class is dropped and the written set is 59 way. The count is in
meta.json, not assumed here.
"""

from __future__ import annotations

import functools
import json
import sys
import tarfile
from pathlib import Path

from data.converters.common import (
    Fetcher,
    cached_archive,
    http_get,
    run_cli,
    sha256_file,
    write_dataset,
)
from data.instrument_format import SPLITS, Row, remove_leaks

NAME = "massive_tr"
SOURCE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
ARCHIVE_NAME = "amazon-massive-dataset-1.1.tar.gz"
# The archive as read on 2026-09-19. A different file is a different dataset, so the
# command line refuses it; a new upstream release needs a dated decision.
EXPECTED_SHA256 = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
LOCALE = "tr-TR"
MEMBER = f"1.1/data/{LOCALE}.jsonl"

# The source calls its middle split dev.
PARTITIONS = {"train": "train", "dev": "validation", "test": "test"}


def _read_member(archive: Path) -> list[dict]:
    with tarfile.open(archive, "r:gz") as tar:
        try:
            handle = tar.extractfile(MEMBER)
        except KeyError as exc:
            raise RuntimeError(f"{archive}: member {MEMBER} is missing") from exc
        if handle is None:
            raise RuntimeError(f"{archive}: member {MEMBER} is not a file")
        with handle:
            text = handle.read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _parse(records: list[dict], dropped: dict[str, int]) -> tuple[dict[str, list[Row]], list[str]]:
    """Turn the source rows into provisional Rows, indexed into the observed intents."""
    kept: list[tuple[str, str, str, str]] = []  # split, id, text, intent
    for record in records:
        if record["locale"] != LOCALE:
            dropped["other_locale"] += 1
            continue
        split = PARTITIONS.get(record["partition"])
        if split is None:
            dropped["unknown_partition"] += 1
            continue
        text = record["utt"].strip()
        if not text:
            dropped["empty_text"] += 1
            continue
        kept.append((split, record["id"], text, record["intent"]))

    labels = sorted({intent for _, _, _, intent in kept})
    index_of = {intent: index for index, intent in enumerate(labels)}
    splits: dict[str, list[Row]] = {split: [] for split in SPLITS}
    for split, source_id, text, intent in kept:
        # The source id is unique inside the locale; the prefix keeps it
        # readable and stays unique if a future version reuses ids per split.
        splits[split].append(Row(f"{split}-{source_id}", text, None, index_of[intent]))
    return splits, labels


def _drop_labels_missing_from_a_split(
    splits: dict[str, list[Row]], labels: list[str], dropped: dict[str, int]
) -> tuple[dict[str, list[Row]], list[str]]:
    """Remove intents that have no rows left in train or in test.

    A class absent from train cannot be learned and one absent from test makes
    macro-F1 undefined for it, so the format check rejects both. In tr-TR this
    removes cooking_query, which the source gives no test rows.
    """
    required = [{row.label for row in splits[split]} for split in ("train", "test")]
    keep = sorted(index for index in range(len(labels)) if all(index in seen for seen in required))
    if len(keep) == len(labels):
        return splits, labels

    new_labels = [labels[index] for index in keep]
    remap = {old: new for new, old in enumerate(keep)}
    cleaned: dict[str, list[Row]] = {}
    for split in SPLITS:
        rows = [row for row in splits[split] if row.label in remap]
        dropped["label_missing_from_a_split"] += len(splits[split]) - len(rows)
        cleaned[split] = [Row(r.id, r.text, r.text_pair, remap[r.label]) for r in rows]
    return cleaned, new_labels


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
    dropped = {
        "other_locale": 0,
        "unknown_partition": 0,
        "empty_text": 0,
        "label_missing_from_a_split": 0,
    }
    splits, labels = _parse(_read_member(archive), dropped)
    splits, leaked = remove_leaks(splits)
    dropped.update(leaked)
    # After leak removal, because a dropped duplicate can empty a class.
    splits, labels = _drop_labels_missing_from_a_split(splits, labels, dropped)

    empty = [split for split in SPLITS if not splits[split]]
    if empty:
        raise RuntimeError(f"{NAME}: no rows left for splits {empty}")

    meta = {
        "task": "Intent of a Turkish voice assistant utterance, one label per utterance.",
        "source": source_url,
        "source_revision": f"sha256:{sha256_file(archive)}",
        "license_id": "CC-BY-4.0",
        "labels": labels,
        "split_origin": {
            "train": "source partition train",
            "validation": "source partition dev",
            "test": "source partition test",
        },
        "dropped": dropped,
        "notes": (
            f"MASSIVE version 1.1, member {MEMBER} of {ARCHIVE_NAME}, locale {LOCALE} only. "
            "Text is the utt field with outer whitespace stripped; the slot annotations, "
            "worker ids and judgment scores in the source are not used. The source lists 60 "
            "intents; a label with no rows left in train or in test is dropped, which the "
            "labels list and the dropped counts show. MASSIVE is a localisation of SLURP, so "
            "the attribution covers both."
        ),
    }
    return write_dataset(out_root, NAME, splits, meta)


if __name__ == "__main__":
    pinned = functools.partial(convert, expected_sha256=EXPECTED_SHA256)
    sys.exit(run_cli(pinned, __doc__.splitlines()[0]))
