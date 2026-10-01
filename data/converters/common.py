"""Shared plumbing for the converters that read an original archive over HTTPS.

Three jobs the converters have in common: fetch the archive once and hash it,
carve a validation split when the source has none, and write the instrument
folder so the format check runs on every conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx

from data.instrument_format import SPLITS, Row, check_dataset, write_split

# The response is streamed, so the archive never sits in memory whole.
_CHUNK = 1 << 20

# A converter takes this in place of the real download, so tests can pass one
# that raises and prove that a cached archive is not fetched again.
Fetcher = Callable[[str, Path], None]


def http_get(url: str, dest: Path) -> None:
    """Download `url` to `dest`, through a partial file so a break leaves no cache hit."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_bytes(_CHUNK):
                handle.write(chunk)
    partial.replace(dest)


def cached_archive(url: str, dest: Path, fetch: Fetcher = http_get) -> Path:
    """Return `dest`, downloading it from `url` only when it is not there yet."""
    if not dest.is_file():
        fetch(url, dest)
    return dest


def sha256_file(path: Path) -> str:
    """The hex digest of a file's bytes. Converters record it as the source revision."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def stratified_carve(
    rows: Sequence[Row], fraction: float, rng: random.Random
) -> tuple[list[Row], list[Row]]:
    """Split `rows` into (rest, carved), taking `fraction` of every label.

    Labels are visited in a fixed order and the rng is used once per label, so
    the carve depends only on the seed and the row order, not on dict order.
    A label with two or more rows always contributes one row, otherwise a rare
    class would be missing from the validation split it is meant to measure;
    at least one row of every label stays in the rest, which the format check
    requires of train.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be between 0 and 1, not {fraction}")
    by_label: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        by_label.setdefault(row.label, []).append(index)

    carved: set[int] = set()
    for label in sorted(by_label):
        group = by_label[label]
        take = min(len(group) - 1, max(1, round(len(group) * fraction)))
        if take > 0:
            carved.update(rng.sample(group, take))
    rest = [row for index, row in enumerate(rows) if index not in carved]
    taken = [row for index, row in enumerate(rows) if index in carved]
    return rest, taken


def write_dataset(out_root: Path, name: str, splits: dict[str, list[Row]], meta: dict) -> Path:
    """Write the three split files and meta.json, then fail if the folder is not valid.

    `meta` carries everything but `name` and `splits`, which are filled from
    what is actually written so the counts cannot drift from the files.
    """
    folder = out_root / name
    for split in SPLITS:
        write_split(folder / f"{split}.jsonl", splits[split])
    full = {"name": name, "splits": {split: len(splits[split]) for split in SPLITS}, **meta}
    text = json.dumps(full, ensure_ascii=False, indent=2, sort_keys=True)
    (folder / "meta.json").write_text(text + "\n", encoding="utf-8")

    report = check_dataset(folder)
    if not report.ok:
        joined = "\n  ".join(report.errors)
        raise RuntimeError(f"{name}: the written dataset failed the format check:\n  {joined}")
    return folder


def run_cli(
    convert: Callable[[Path, Path], Path], description: str, argv: Sequence[str] | None = None
) -> int:
    """The command line every converter module shares."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out-root", type=Path, default=Path("data/raw/instrument"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/_downloads"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    folder = convert(args.out_root, args.cache_dir)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    print(f"wrote {folder}")
    print(f"  revision {meta['source_revision']}")
    print(f"  labels   {len(meta['labels'])}")
    print(f"  splits   {meta['splits']}")
    print(f"  dropped  {meta['dropped']}")
    return 0
