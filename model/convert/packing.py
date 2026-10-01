"""Packing documents into fixed windows, so no step trains on padding.

The plan trains at context 1024 with packing. Documents are concatenated with
a separator between them and cut into windows of exactly that length, so every
position in every batch carries real text and the GPU is never paid to attend
to padding.

Packing has a cost that only shows up with bidirectional attention: two
unrelated documents sharing a window can read each other, which teaches a
context inference never presents. The separator is what makes that avoidable.
`bidirectional.document_ids_for` turns the separators back into per-document
indices at batch time, and the mask keeps each document to itself, so
nothing here needs to track boundaries itself.

A document longer than the window simply spans several, and its continuation
carries no separator, which is correct: a continuation is the same document and
should be able to read the part of itself that shares its window.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import numpy as np


def pack(
    documents: Iterable[Sequence[int]],
    *,
    context: int,
    separator_id: int,
    dtype: np.dtype | type = np.int64,
) -> Iterator[np.ndarray]:
    """Yield windows of exactly `context` tokens, documents separated.

    The tail that cannot fill a window is dropped rather than padded. At this
    context a dropped tail is at most 1023 tokens against billions, and padding
    it would put positions in the batch that the objective must then be told to
    ignore, which is a bug waiting to happen for no measurable gain.
    """
    if context <= 0:
        raise ValueError(f"context must be positive, got {context}")

    buffer: list[int] = []
    for document in documents:
        buffer.extend(document)
        buffer.append(separator_id)
        while len(buffer) >= context:
            yield np.asarray(buffer[:context], dtype=dtype)
            del buffer[:context]


def packed_token_count(total_tokens: int, n_documents: int, context: int) -> int:
    """How many tokens survive packing, separators included, tail dropped.

    Used to turn a token budget into a step count before anything is read, so
    the schedule and the ledger agree with what the run will actually do.
    """
    if context <= 0:
        raise ValueError(f"context must be positive, got {context}")
    return ((total_tokens + n_documents) // context) * context
