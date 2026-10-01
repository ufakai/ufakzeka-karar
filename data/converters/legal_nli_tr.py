"""Turkish legal NLI to the instrument format: premise and hypothesis pairs.

The pairs are summaries of Turkish commercial court rulings. The card says
the labels come from the distance between the Turkish Commercial Code
articles two rulings cite, arranged as a tree: rule-made, not model-made.

The source train split is 474,283 pairs and 1.9 GB of text, far more than
the instrument needs and more than fits in memory here. Leak removal runs
over all of it first, so no evaluation pair can survive in train, and only
the 20,000 pairs the subsample keeps are ever held as rows.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from data.converters.hub_common import (
    FetchFile,
    dataset_url,
    fetch_file,
    files_note,
    iter_parquet_rows,
    main_for,
    merge_counts,
    read_rows,
    sha256_of,
    stratified_take,
    write_dataset,
)
from data.instrument_format import SPLITS, Row, example_key, remove_leaks

NAME = "legal_nli_tr"
REPO = "Turkish-NLI/legal_nli_TR_V1"
REVISION = "67baa141cf4f6634c983d77eea193c5535611e5a"
LICENSE_ID = "Apache-2.0"
TRAIN_FILES = tuple(f"data/train-0000{shard}-of-00004.parquet" for shard in range(4))
FILES = {
    "train": TRAIN_FILES,
    "validation": ("data/validation-00000-of-00001.parquet",),
    "test": ("data/test-00000-of-00001.parquet",),
}
COLUMNS = ("premise", "hypothesis", "label")
LABELS = ["contradiction", "entailment", "neutral"]
TARGETS = {"train": 20_000, "validation": 2_000, "test": 3_000}
SEED = 1


def _digest(row: Row) -> bytes:
    """A short stand-in for the leak key, so 474,283 keys fit in memory."""
    text, pair = example_key(row)
    return hashlib.blake2b(f"{text}\x00{pair}".encode(), digest_size=16).digest()


def _make_row(split: str, index: int, record: Mapping[str, object]) -> Row | None:
    premise = str(record["premise"] or "").strip()
    hypothesis = str(record["hypothesis"] or "").strip()
    label = str(record["label"] or "").strip()
    if not premise or not hypothesis or label not in LABELS:
        return None
    return Row(f"{NAME}-{split}-{index:06d}", premise, hypothesis, LABELS.index(label))


def _read_eval(path: Path, split: str, own: dict[str, int]) -> list[Row]:
    rows: list[Row] = []
    for index, record in enumerate(read_rows(path, COLUMNS)):
        row = _make_row(split, index, record)
        if row is None:
            own["empty_or_unknown"] += 1
            continue
        rows.append(row)
    return rows


def convert(
    out_root: Path,
    cache_dir: Path,
    *,
    fetch: FetchFile = fetch_file,
    targets: Mapping[str, int] = TARGETS,
) -> Path:
    paths = {
        split: [fetch(REPO, name, REVISION, cache_dir) for name in names]
        for split, names in FILES.items()
    }
    shas = {
        name: sha256_of(path)
        for split, names in FILES.items()
        for name, path in zip(names, paths[split], strict=True)
    }

    own = {"empty_or_unknown": 0, "subsampled_out": 0}
    evaluation = {
        split: _read_eval(paths[split][0], split, own) for split in ("validation", "test")
    }
    # The evaluation splits are made disjoint first: their keys are what train
    # is then checked against, while train is still being streamed.
    evaluation, eval_leaks = remove_leaks({"train": [], **evaluation})
    eval_digests = {_digest(row) for split in ("validation", "test") for row in evaluation[split]}

    train_leaks = {"duplicate_within_split": 0, "train_in_eval": 0}
    seen: set[bytes] = set()
    kept: list[tuple[int, int]] = []
    for index, record in enumerate(iter_parquet_rows(paths["train"], COLUMNS)):
        row = _make_row("train", index, record)
        if row is None:
            own["empty_or_unknown"] += 1
            continue
        digest = _digest(row)
        if digest in seen:
            train_leaks["duplicate_within_split"] += 1
        elif digest in eval_digests:
            train_leaks["train_in_eval"] += 1
        else:
            seen.add(digest)
            kept.append((index, row.label))

    wanted = {index for index, _ in stratified_take(kept, targets["train"], lambda p: p[1], SEED)}
    own["subsampled_out"] += len(kept) - len(wanted)
    train: list[Row] = []
    for index, record in enumerate(iter_parquet_rows(paths["train"], COLUMNS)):
        if index not in wanted:
            continue
        row = _make_row("train", index, record)
        if row is not None:
            train.append(row)

    splits = {"train": train}
    for split in ("validation", "test"):
        rows = evaluation[split]
        splits[split] = stratified_take(rows, targets[split], lambda row: row.label, SEED)
        own["subsampled_out"] += len(rows) - len(splits[split])

    # The subsample only removes rows, so this second pass should find nothing.
    # It is the proof that it did not, and its counts are added to the rest.
    splits, final_leaks = remove_leaks({split: splits[split] for split in SPLITS})
    origin = {
        split: (
            f"source {split}, stratified subsample to {targets[split]} with random.Random({SEED}), "
            "after leak removal over the whole source split"
        )
        for split in SPLITS
    }
    meta = {
        "name": NAME,
        "task": "three-way inference over pairs of Turkish commercial court ruling summaries",
        "source": dataset_url(REPO),
        "source_revision": REVISION,
        "license_id": LICENSE_ID,
        "labels": LABELS,
        "splits": {split: len(rows) for split, rows in splits.items()},
        "split_origin": origin,
        "dropped": merge_counts(own, eval_leaks, train_leaks, final_leaks),
        "notes": (
            "text is the premise, text_pair the hypothesis. Labels are rule-made: the card "
            "derives them from the distance between the Turkish Commercial Code articles two "
            "rulings cite, placed in a tree, seven nearest as entailment and seven furthest as "
            "contradiction. No model wrote or labelled the pairs. Leak removal ran over all "
            "474,283 source train pairs before the subsample, so no evaluation pair can be in "
            "train. Source files: " + files_note(shas)
        ),
    }
    return write_dataset(out_root, meta, splits)


if __name__ == "__main__":
    raise SystemExit(main_for(convert, __doc__ or NAME))
