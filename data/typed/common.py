"""Pieces the one-hot typed converters share (data/typed/legal.py, data/typed/claims.py).

A converter reads its source into `Item`s (a source id, the text the model
reads, the label) per split, and this module does the rest the same way for
every set: pinned downloads checked by hash, a validation split carved by a
hash of an id when the source has none, exact duplicates collapsed, texts
labelled two ways dropped, validation texts already in train dropped, and the
rows written with counts a report can quote.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from data import hub
from schema.rows import TrainingRow

SPLITS = ("train", "validation")
VALIDATION_SHARE = 0.10


@dataclass(frozen=True)
class Item:
    source_id: str
    state: str
    label: str


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_pinned(
    files: dict[str, tuple[str, str]],
    folder: Path,
    *,
    download: Callable[[str, Path], None] = hub.download,
) -> dict[str, Path]:
    """Each file as a local copy, downloaded once and checked against its pinned sha256.

    `files` maps a local name to (url, sha256). hub.download sends the hub
    token to huggingface.co hosts only, so it is safe for other hosts too.
    """
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (url, expected) in files.items():
        path = folder / name
        if not path.is_file():
            download(url, path)
        actual = sha256_of(path)
        if actual != expected:
            raise ValueError(f"{path}: sha256 {actual} is not the pinned {expected}")
        paths[name] = path
    return paths


def hub_file_url(repo: str, revision: str, filename: str) -> str:
    return (
        f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(filename)}"
    )


def carve(key: str, salt: str, share: float = VALIDATION_SHARE) -> str:
    """'validation' for about `share` of keys, by a hash, so the carve never moves."""
    digest = hashlib.blake2b(f"{salt}:{key}".encode(), digest_size=8).digest()
    return "validation" if int.from_bytes(digest, "big") / 2**64 < share else "train"


def resolve(items: Iterable[Item], drops: Counter) -> list[Item]:
    """One item per distinct text; a text carrying two labels is dropped entirely.

    A human label is a one-hot target, so the same text labelled two ways
    (MiDe22-style duplicates, or a short summary shared by a civil and a
    criminal case) cannot be kept as either.
    """
    groups: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        groups[item.state].append(item)
    kept = []
    for group in groups.values():
        if len({item.label for item in group}) > 1:
            drops["text_with_two_labels"] += len(group)
            continue
        drops["duplicate_text"] += len(group) - 1
        kept.append(min(group, key=lambda item: item.source_id))
    return sorted(kept, key=lambda item: item.source_id)


def clean_splits(splits: dict[str, list[Item]], drops: dict[str, Counter]) -> dict[str, list[Item]]:
    """Resolve each split, then drop validation texts that train already has."""
    out = {split: resolve(splits.get(split, []), drops[split]) for split in SPLITS}
    in_train = {item.state for item in out["train"]}
    leaked = [item for item in out["validation"] if item.state in in_train]
    drops["validation"]["text_also_in_train"] += len(leaked)
    out["validation"] = [item for item in out["validation"] if item.state not in in_train]
    return out


def write_rows(
    splits: dict[str, list[Item]],
    make_row: Callable[[Item, str], TrainingRow],
    out_dir: Path,
) -> dict[str, dict]:
    """Write train.jsonl and validation.jsonl. Returns rows and class counts per split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        seen: set[str] = set()
        classes: Counter = Counter()
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item in splits[split]:
                row = make_row(item, split)
                if row.row_id in seen:
                    raise ValueError(f"{item.source_id}: two items became one row")
                seen.add(row.row_id)
                classes[item.label] += 1
                handle.write(row.model_dump_json() + "\n")
        counts[split] = {"rows": len(seen), "classes": dict(classes.most_common())}
    return counts


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
