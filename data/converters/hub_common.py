"""Shared pieces of the converters that read their sources from the Hugging Face Hub.

Every download is pinned to a full 40-character commit hash and cached under
one folder per repository and revision, so a rerun reads the same bytes or
fails. The rest is split arithmetic the converters share: a stratified carve
of one pool into three splits, a stratified subsample, and the final write
plus format check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from data.instrument_format import SPLITS, Row, check_dataset, write_split

HUB = "https://huggingface.co"


_FULL_HASH = re.compile(r"[0-9a-f]{40}")


class FetchFile(Protocol):
    """Hands back a local copy of one source file. Tests pass their own."""

    def __call__(self, repo_id: str, filename: str, revision: str, cache_dir: Path) -> Path: ...


def dataset_url(repo_id: str) -> str:
    return f"{HUB}/datasets/{repo_id}"


def _hf_download(repo_id: str, filename: str, revision: str, target_dir: Path) -> Path:
    # Imported inside the function so a test that supplies its own files never
    # loads the Hub client, and a missing network cannot be reached by accident.
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            local_dir=target_dir,
        )
    )


def fetch_file(
    repo_id: str,
    filename: str,
    revision: str,
    cache_dir: Path,
    *,
    download: Callable[[str, str, str, Path], Path] = _hf_download,
) -> Path:
    """The local copy of one Hub file at a pinned commit, downloaded once."""
    if not _FULL_HASH.fullmatch(revision):
        raise ValueError(f"{repo_id}: revision must be a full 40-character commit hash")
    target_dir = cache_dir / repo_id.replace("/", "__") / revision
    target = target_dir / filename
    if target.is_file():
        return target
    return download(repo_id, filename, revision, target_dir)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def files_note(shas: Mapping[str, str]) -> str:
    return "; ".join(f"{name} sha256 {sha}" for name, sha in shas.items())


def read_rows(path: Path, columns: Sequence[str]) -> list[dict[str, Any]]:
    """Read a whole parquet, csv, tsv or jsonl source file as dictionaries.

    Text columns of a csv or tsv are forced to strings. Left to itself the
    reader turns a column holding only "True" and "False" into booleans,
    which would silently rename two of MiDe22's three classes.
    """
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=list(columns))
        return table.to_pylist()
    if suffix in (".csv", ".tsv"):
        import pyarrow as pa
        import pyarrow.csv as pacsv

        parse = pacsv.ParseOptions(
            delimiter="\t" if suffix == ".tsv" else ",",
            # Tweets and ruling summaries contain line breaks inside quoted fields.
            newlines_in_values=True,
        )
        convert = pacsv.ConvertOptions(column_types={name: pa.string() for name in columns})
        table = pacsv.read_csv(path, parse_options=parse, convert_options=convert)
        missing = [name for name in columns if name not in table.column_names]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        return table.select(list(columns)).to_pylist()
    if suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        missing = sorted({name for name in columns for row in rows if name not in row})
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        return rows
    raise ValueError(f"{path}: unsupported file type {suffix!r}")


def iter_parquet_rows(
    paths: Sequence[Path], columns: Sequence[str], batch_size: int = 512
) -> Iterator[dict[str, Any]]:
    """Stream the rows of several parquet shards in file order.

    The legal NLI train split is 1.9 GB of text. Reading it whole does not
    fit in the memory of the machines this runs on, so it is read in batches.
    """
    import pyarrow.parquet as pq

    for path in paths:
        reader = pq.ParquetFile(path)
        for batch in reader.iter_batches(batch_size=batch_size, columns=list(columns)):
            yield from batch.to_pylist()


def apportion[K](
    total: int, weights: Mapping[K, float], caps: Mapping[K, int] | None = None
) -> dict[K, int]:
    """Split `total` between the keys by weight, integers, largest remainders first.

    A cap stops a key from being given more than it has. Ties go to the
    smaller key, so the result does not depend on dictionary order.
    """
    if total < 0:
        raise ValueError("total must not be negative")
    population = sum(weights.values())
    if population <= 0:
        return dict.fromkeys(weights, 0)
    exact = {key: total * weight / population for key, weight in weights.items()}
    limit = {key: (caps[key] if caps else total) for key in weights}
    quotas = {key: min(int(value), limit[key]) for key, value in exact.items()}
    order = sorted(weights, key=lambda key: (-(exact[key] - int(exact[key])), key))
    leftover = total - sum(quotas.values())
    while leftover > 0:
        room = [key for key in order if quotas[key] < limit[key]]
        if not room:
            break
        for key in room:
            if leftover == 0:
                break
            quotas[key] += 1
            leftover -= 1
    return quotas


def _positions_by_label[T](
    items: Sequence[T], label_of: Callable[[T], int]
) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for position, item in enumerate(items):
        groups.setdefault(label_of(item), []).append(position)
    return groups


def stratified_take[T](
    items: Sequence[T], count: int, label_of: Callable[[T], int], seed: int
) -> list[T]:
    """`count` of the items, in their original order, each label keeping its share."""
    if count >= len(items):
        return list(items)
    groups = _positions_by_label(items, label_of)
    sizes = {label: len(positions) for label, positions in groups.items()}
    quotas = apportion(count, sizes, sizes)
    rng = random.Random(seed)
    chosen: list[int] = []
    for label in sorted(groups):
        chosen.extend(rng.sample(groups[label], quotas[label]))
    return [items[position] for position in sorted(chosen)]


def stratified_carve[T](
    items: Sequence[T], shares: Mapping[str, float], label_of: Callable[[T], int], seed: int
) -> dict[str, list[T]]:
    """Carve one pool into named parts, splitting every label by the same shares."""
    groups = _positions_by_label(items, label_of)
    rng = random.Random(seed)
    parts: dict[str, list[int]] = {name: [] for name in shares}
    for label in sorted(groups):
        positions = list(groups[label])
        rng.shuffle(positions)
        counts = apportion(len(positions), shares)
        start = 0
        for name, size in counts.items():
            parts[name].extend(positions[start : start + size])
            start += size
    return {
        name: [items[position] for position in sorted(positions)]
        for name, positions in parts.items()
    }


def merge_counts(*counts: Mapping[str, int]) -> dict[str, int]:
    """Add drop counts together, keeping a zero entry so the reason stays visible."""
    merged: dict[str, int] = {}
    for part in counts:
        for reason, value in part.items():
            merged[reason] = merged.get(reason, 0) + value
    return merged


def write_dataset(out_root: Path, meta: Mapping[str, Any], splits: Mapping[str, list[Row]]) -> Path:
    """Write the three split files and meta.json, then check the folder."""
    folder = out_root / str(meta["name"])
    folder.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        write_split(folder / f"{name}.jsonl", splits[name])
    text = json.dumps(dict(meta), ensure_ascii=False, indent=2, sort_keys=True)
    (folder / "meta.json").write_text(text + "\n", encoding="utf-8")
    report = check_dataset(folder)
    if not report.ok:
        raise RuntimeError(f"{folder}: " + "; ".join(report.errors))
    return folder


def main_for(
    convert: Callable[[Path, Path], Path], description: str, argv: Sequence[str] | None = None
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out-root", type=Path, default=Path("data/raw/instrument"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/_downloads"))
    args = parser.parse_args(argv)
    folder = convert(args.out_root, args.cache_dir)
    print(folder)
    print((folder / "meta.json").read_text(encoding="utf-8"), end="")
    return 0
