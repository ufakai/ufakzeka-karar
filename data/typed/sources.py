"""Downloads and row helpers for the guardrail and relevance converters.

Every download is a file at a full 40-character commit, fetched through the
hub's file resolver (data/hub.py, which sends the token to the hub only) or
from a pinned GitHub commit, into data/raw/_downloads/<owner>__<repo>/<commit>/.
A path naming a test split is refused before any request is made: a source's
test split is never downloaded, read or written.

The laptop's disk is nearly full, so every download first checks that at
least MIN_FREE_BYTES would remain after it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

from data import hub
from data.decontam.ngrams import normalise
from schema.rows import TrainingRow

DOWNLOADS = Path("data/raw/_downloads")
BUILT = Path("data/built/typed")
MIN_FREE_BYTES = 300 * 1024 * 1024
VALIDATION_SHARE = 0.1
SPLITS = ("train", "validation")

_FULL_HASH = re.compile(r"[0-9a-f]{40}")
_TEST_PART = re.compile(r"(^|[/_.-])(test|tests)([/_.-]|$)", re.IGNORECASE)

Fetch = Callable[[str, Path], None]


def refuse_test(path: str) -> None:
    """Raise if a source path names a test split."""
    if _TEST_PART.search(path):
        raise ValueError(f"{path}: a source's test split is never downloaded or read")


def check_disk(target: Path, expected_bytes: int, *, free_bytes: int | None = None) -> None:
    """Stop when the download would leave less than MIN_FREE_BYTES on the volume."""
    if free_bytes is None:
        probe = target
        while not probe.exists():
            probe = probe.parent
        free_bytes = shutil.disk_usage(probe).free
    if free_bytes - expected_bytes < MIN_FREE_BYTES:
        raise OSError(
            f"{target}: {expected_bytes / 2**20:.0f} MB would leave "
            f"{(free_bytes - expected_bytes) / 2**20:.0f} MB free, under the "
            f"{MIN_FREE_BYTES / 2**20:.0f} MB floor"
        )


def local_path(repo: str, revision: str, path: str, root: Path = DOWNLOADS) -> Path:
    return root / repo.replace("/", "__") / revision / path


def hub_file(
    repo: str,
    revision: str,
    path: str,
    expected_bytes: int,
    root: Path = DOWNLOADS,
    *,
    fetch: Fetch = hub.download,
) -> Path:
    """One dataset file from the hub at a pinned commit, downloaded once."""
    refuse_test(path)
    if not _FULL_HASH.fullmatch(revision):
        raise ValueError(f"{repo}: revision must be a full 40-character commit hash")
    target = local_path(repo, revision, path, root)
    if target.is_file():
        return target
    check_disk(target, expected_bytes)
    fetch(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}", target)
    return target


def _plain_download(url: str, target: Path) -> None:
    """A download from a host other than the hub, so no token is ever attached."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "ufakzeka-karar"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    partial.replace(target)


def github_file(
    repo: str,
    revision: str,
    path: str,
    expected_bytes: int,
    root: Path = DOWNLOADS,
    *,
    fetch: Fetch = _plain_download,
) -> Path:
    """One file of a GitHub repository at a pinned commit, downloaded once."""
    refuse_test(path)
    if not _FULL_HASH.fullmatch(revision):
        raise ValueError(f"{repo}: revision must be a full 40-character commit hash")
    target = local_path(repo, revision, path, root)
    if target.is_file():
        return target
    check_disk(target, expected_bytes)
    fetch(f"https://raw.githubusercontent.com/{repo}/{revision}/{path}", target)
    return target


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_split(source_id: str, share: float = VALIDATION_SHARE) -> str:
    """Validation for a fixed share of source ids, chosen by a hash of the id."""
    digest = hashlib.blake2b(source_id.encode("utf-8"), digest_size=8).digest()
    return "validation" if int.from_bytes(digest, "big") % 10_000 < share * 10_000 else "train"


def dedupe_key(text: str) -> str:
    """Text compared across sources: NFKC, Turkish lower case, no punctuation, one space."""
    return normalise(text)


def read_jsonl(path: Path) -> list[dict]:
    # Split on newlines only: str.splitlines also breaks on U+2028 and similar
    # separators that occur inside JSON strings.
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def read_parquet(path: Path, columns: Iterable[str] | None = None) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=list(columns) if columns is not None else None)
    return table.to_pylist()


def noul_balance(rows: Iterable[TrainingRow]) -> dict[str, int]:
    counts = Counter("true" if row.target["true"] == 1.0 else "false" for row in rows)
    return {"true": counts["true"], "false": counts["false"]}


def unique_rows(rows: Iterable[TrainingRow]) -> list[TrainingRow]:
    """The first row of each row id; the same text and question asked twice is one row."""
    seen: set[str] = set()
    kept = []
    for row in rows:
        if row.row_id not in seen:
            seen.add(row.row_id)
            kept.append(row)
    return kept


def write_rows(rows: Iterable[TrainingRow], out_dir: Path) -> dict[str, int]:
    """Write train and validation files; one copy of each row id. Returns counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {split: (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") for split in SPLITS}
    seen: set[str] = set()
    counts = dict.fromkeys(SPLITS, 0)
    try:
        for row in rows:
            if row.split not in handles:
                raise ValueError(f"{row.split!r} rows are not written here")
            if row.row_id in seen:
                continue
            seen.add(row.row_id)
            handles[row.split].write(row.model_dump_json() + "\n")
            counts[row.split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def write_meta(out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (out_dir / "meta.json").write_text(body, encoding="utf-8")
