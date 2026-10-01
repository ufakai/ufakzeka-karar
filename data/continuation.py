"""The text for the continuation experiment (PLAN.md backbone item 4).

Continued causal pretraining of ufakzeka-1-base on the kind of text its
decisions are about, before the head is trained: court decisions and
legislation for the legal-aid track, company and community questions for
support routing. Two tiers of token shards in the format the conversion trainer
reads (uint16, no header, documents closed by the end-of-text id):

- legal: mrfg/turkish-court-decisions (CC0), Council of State, Emsal and Court
  of Cassation decisions from a few pinned shards. The Constitutional Court
  configurations are not read: the legal-rights task's validation rows are
  summaries of those rulings.
- questions: clips/mqa (CC0 packaging), the two Turkish configurations, through
  the same filters and masks as the build (data/label/texts.py).

Every document is masked for personal data and dropped if it overlaps an
evaluation set: the 70 registered reference sets (the index on the volume) and
our own validation and held-out texts, so continued pretraining never reads
what the head is later scored on. The rule is the decontamination's own: more
than half of a document's tokens in shared 8-grams, a whole reference item
inside it, or a near-duplicate by MinHash.
"""

from __future__ import annotations

import gc
import json
import multiprocessing
import random
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import closing
from pathlib import Path

import numpy as np

LEGAL_REPO = "mrfg/turkish-court-decisions"
LEGAL_REVISION = "e9062174adcdee8386b94cebd01df7264268e86b"
LEGAL_FILES = (
    "data/danistay/train-00000-of-00002.parquet",
    "data/danistay/train-00001-of-00002.parquet",
    "data/emsal/train-00000-of-00009.parquet",
    "data/emsal/train-00001-of-00009.parquet",
    "data/emsal/train-00002-of-00009.parquet",
    "data/yargitay/train-00000-of-00017.parquet",
    "data/yargitay/train-00001-of-00017.parquet",
)
MIN_CHARS = 200
MAX_CHARS = 20_000
SHARD_TOKENS = 50_000_000
OVERLAP_LIMIT = 0.5


def legal_documents(paths: Iterable[Path], seed: int) -> Iterator[tuple[str, str]]:
    """(id, raw text) for every decision in the shards, row groups in a seeded order.

    The text is cut to MAX_CHARS here and masked by `legal_text` in the workers,
    so the reader stays cheap and the regular expressions run in parallel.
    """
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    groups = [(p, g) for p in paths for g in range(pq.ParquetFile(p).metadata.num_row_groups)]
    rng.shuffle(groups)
    for path, group in groups:
        table = pq.ParquetFile(path).read_row_group(group, columns=["id", "text"]).to_pylist()
        for row in table:
            text = (row.get("text") or "").strip()
            if len(text) >= MIN_CHARS:
                yield str(row["id"]), text[:MAX_CHARS]


def legal_text(text: str) -> str:
    from data.label.texts import mask_personal_data

    return mask_personal_data(text)


def question_documents(paths: Iterable[Path], seed: int) -> Iterator[tuple[str, dict]]:
    """(id, row) for every question-and-answer row; `question_text` filters and masks it."""
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    groups = [(p, g) for p in paths for g in range(pq.ParquetFile(p).metadata.num_row_groups)]
    rng.shuffle(groups)
    for path, group in groups:
        columns = ["id", "name", "domain", "answers"]
        for row in pq.ParquetFile(path).read_row_group(group, columns=columns).to_pylist():
            yield str(row["id"]), row


def question_text(row: dict) -> str | None:
    """The build's own filters and masks (data/label/texts.py); None if a filter drops it."""
    from data.label.texts import faq_text

    return faq_text(row)


def evaluation_texts(root: Path) -> Iterator[tuple[str, str]]:
    """(id, state) of every validation and held-out row under a data root."""
    patterns = ("typed/*/validation.jsonl", "sss/validation.jsonl", "sss/heldout_task.jsonl",
                "synth/validation.jsonl")  # fmt: skip
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    state = row["state"]
                    text = state if isinstance(state, str) else json.dumps(state)
                    yield f"eval:{row['row_id']}", text


def overlap_reason(text: str, checks: list[tuple[str, object, object | None]]) -> str | None:
    """Why a document overlaps an evaluation set, or None if it does not."""
    from data.decontam.ngrams import gram_hashes, minhash_signature, tokens

    toks = tokens(text)
    hashes: set[int] | None = None
    signature, signed = None, False
    for name, index, lsh in checks:
        fraction, refs = index.overlap(text)
        if fraction > OVERLAP_LIMIT:
            return f"{name}:ngram"
        if refs:
            if hashes is None:
                hashes = set(gram_hashes(toks, index.n))
            if hashes and index.covered_references(hashes, refs):
                return f"{name}:covers"
        if lsh is not None:
            # One signature per document, not one per check: it is the costly part.
            if not signed:
                signature, signed = minhash_signature(text), True
            if signature is not None and lsh.query(signature):
                return f"{name}:minhash"
    return None


# What the worker processes read. Set before the pool forks, so each worker
# shares the parent's copy of the indexes instead of unpickling its own.
_WORK: dict = {}


def _screen(item: tuple[str, object]) -> tuple[str | None, str | None]:
    """(text, why it is dropped) for one document, in a worker."""
    _doc_id, payload = item
    prepare = _WORK["prepare"]
    text = prepare(payload) if prepare is not None else payload
    if text is None:
        return None, "filtered"
    return text, overlap_reason(text, _WORK["checks"])


def screened(
    documents: Iterable[tuple[str, object]],
    checks: list,
    prepare: Callable[[object], str | None] | None = None,
    workers: int = 1,
) -> Iterator[tuple[str | None, str | None]]:
    """(text, reason) per document, in the documents' order, across `workers` processes.

    The order is kept (imap, not imap_unordered), so a build is the same
    whatever the worker count.
    """
    _WORK.update(checks=checks, prepare=prepare)
    if workers <= 1:
        yield from map(_screen, documents)
        return
    # Frozen objects are left out of the collector's passes, so a collection in
    # a worker does not touch, and so copy, every page of the inherited indexes.
    gc.freeze()
    try:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            yield from pool.imap(_screen, documents, chunksize=16)
    finally:
        gc.unfreeze()


def write_tier(
    documents: Iterable[tuple[str, object]],
    encode_batch,
    separator_id: int,
    out_dir: Path,
    target_tokens: int,
    checks: list,
    batch: int = 256,
    prepare: Callable[[object], str | None] | None = None,
    workers: int = 1,
) -> dict:
    """Tokenised documents up to `target_tokens`, in uint16 shards; counts of what was dropped.

    `prepare` turns a reader's payload into the text (masking, filters), None
    to drop it; without it the payload is the text.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    buffer: list[int] = []
    written, shard = 0, 0

    def flush(final: bool = False) -> None:
        nonlocal buffer, shard
        while len(buffer) >= SHARD_TOKENS or (final and buffer):
            part, buffer = buffer[:SHARD_TOKENS], buffer[SHARD_TOKENS:]
            np.asarray(part, dtype=np.uint16).tofile(out_dir / f"part-{shard:03d}.bin")
            shard += 1

    pending: list[str] = []

    def take(texts: list[str]) -> None:
        nonlocal written
        for ids in encode_batch(texts):
            buffer.extend(ids)
            buffer.append(separator_id)
            written += len(ids) + 1

    # Closed on the early stop, so the pool is shut down there and not whenever
    # the suspended generator happens to be collected.
    with closing(screened(documents, checks, prepare, workers)) as stream:
        for text, reason in stream:
            counts["seen"] += 1
            if counts["seen"] % 20_000 == 0:
                print(f"{out_dir.name}: {written:,} tokens, {dict(counts)}", flush=True)
            if reason is not None:
                counts[f"dropped:{reason}"] += 1
                continue
            pending.append(text)
            counts["kept"] += 1
            if len(pending) >= batch:
                take(pending)
                pending = []
                flush()
                if written >= target_tokens:
                    break
    if pending and written < target_tokens:
        take(pending)
    flush(final=True)
    return {"tokens": written, "shards": shard, **dict(counts)}
