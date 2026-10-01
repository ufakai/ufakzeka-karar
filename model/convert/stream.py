"""Choosing what the conversion reads next, and being able to say it again.

Two properties matter more than speed here. The mixture has to hold: over a run
the tiers must appear in the proportions the backbone was pretrained with,
or the conversion changes the diet as well as the objective. And the
order has to be reproducible from a checkpoint, because Modal caps a call at
24 hours and preempts GPU containers, so every long run is really a series of
runs that must continue rather than restart.

Which tier a step draws from is therefore a pure function of the seed and the
step number, with no state at all: a resume recomputes it rather than trusting
something it saved. Where inside that tier the step reads is a cursor, because
tiers are consumed at different rates, and those cursors are small integers
that ride in the checkpoint.
"""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


def tier_for_step(step: int, mixture: dict[str, float], seed: int) -> str:
    """Which tier step `step` draws from. Pure, so a resume needs nothing saved.

    The draw is a hash of the seed and the step rather than a generator's
    state, which means step one million can be answered without having walked
    the first 999,999.
    """
    if step < 0:
        raise ValueError(f"step cannot be negative, got {step}")
    live = {tier: weight for tier, weight in sorted(mixture.items()) if weight > 0}
    if not live:
        raise ValueError(f"mixture has no positive weight: {mixture}")

    digest = hashlib.blake2b(f"{seed}:{step}".encode(), digest_size=8).digest()
    draw = int.from_bytes(digest, "big") / float(1 << 64)

    total = sum(live.values())
    running = 0.0
    for tier, weight in live.items():
        running += weight / total
        if draw < running:
            return tier
    return next(reversed(live))


def scattered_index(cursor: int, size: int, seed: int) -> int:
    """Where the `cursor`-th read of a tier lands, visiting every window once.

    Reading windows 0, 1, 2 in file order made a batch out of several hundred
    consecutive windows of one shard, and a run that uses a third of a tier
    read only the first third of its files, so whatever order the shards were
    written in became a curriculum nobody chose.

    An affine map `(a * cursor + b) mod size` with `a` coprime to `size` is a
    permutation, so nothing repeats before everything has been seen, and it
    needs no state: the cursor in the checkpoint still says everything. `a` is
    taken near the golden ratio of `size`, which puts consecutive reads far
    apart and spreads any run of reads evenly over the whole tier.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if size == 1:
        return 0
    digest = hashlib.blake2b(f"scatter:{seed}:{size}".encode(), digest_size=8).digest()
    offset = int.from_bytes(digest, "big") % size
    stride = max(1, int(size * 0.6180339887498949))
    while math.gcd(stride, size) != 1:
        stride += 1
    return (stride * (cursor % size) + offset) % size


class Windows(Protocol):
    """A tier's windows, addressable by index and wrapping when exhausted."""

    def __len__(self) -> int: ...

    def window(self, index: int) -> np.ndarray: ...


@dataclass
class MixedStream:
    """Windows drawn tier by tier in the backbone's proportions.

    `mixture` is either one set of weights for the whole run, or a function of
    the training step. The second form is what the recipe actually asks for:
    the backbone leant on bulk web while stable and on the curated tier while
    annealing, and the conversion inherits both. An earlier version took
    only a dict, so the decay mixture was declared and never reached, and the
    tail of the run would have been trained on the wrong diet without anything
    saying so.

    `cursors` is the whole of the mutable state, so saving and restoring it is
    all a resume needs beyond the step number.
    """

    sources: dict[str, Windows]
    mixture: dict[str, float] | Callable[[int], dict[str, float]]
    seed: int
    cursors: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for weights in self._all_mixtures():
            missing = {t for t, w in weights.items() if w > 0} - set(self.sources)
            if missing:
                raise ValueError(f"mixture asks for tiers with no source: {sorted(missing)}")
        for tier in self.sources:
            self.cursors.setdefault(tier, 0)

    def _all_mixtures(self) -> list[dict[str, float]]:
        if callable(self.mixture):
            # Both ends of the run, so a tier missing from either is refused now
            # rather than at the step where the schedule first asks for it.
            return [self.mixture(0), self.mixture(10**9)]
        return [self.mixture]

    def mixture_at(self, step: int) -> dict[str, float]:
        return self.mixture(step) if callable(self.mixture) else self.mixture

    def take(self, step: int, weights: dict[str, float] | None = None) -> tuple[str, np.ndarray]:
        """The window for `step`, and the tier it came from."""
        tier = tier_for_step(step, weights or self.mixture_at(step), self.seed)
        source = self.sources[tier]
        if len(source) == 0:
            raise RuntimeError(f"tier {tier} has no windows")
        window = source.window(scattered_index(self.cursors[tier], len(source), self.seed))
        self.cursors[tier] += 1
        return tier, window

    def batch(self, step: int, size: int) -> tuple[np.ndarray, dict[str, int]]:
        """`size` windows for one step, and how many came from each tier.

        The mixture is resolved once from the training step, not from the
        window counter, so every row of a batch is drawn under the weights that
        step belongs to.
        """
        weights = self.mixture_at(step)
        rows, counts = [], dict.fromkeys(self.sources, 0)
        for index in range(size):
            tier, window = self.take(step * size + index, weights)
            rows.append(window)
            counts[tier] += 1
        return np.stack(rows), counts

    def state(self) -> dict[str, int]:
        return dict(sorted(self.cursors.items()))

    def load_state(self, cursors: dict[str, int]) -> None:
        unknown = set(cursors) - set(self.sources)
        if unknown:
            raise ValueError(f"saved cursors name unknown tiers: {sorted(unknown)}")
        self.cursors.update(cursors)


def staged_mixture(
    stable: dict[str, float], decay: dict[str, float], total_steps: int, decay_frac: float
) -> Callable[[int], dict[str, float]]:
    """The stable weights, then the decay weights over the tail.

    The boundary is the same fraction the learning rate and the masking rate
    step at, so the three change together rather than drifting apart.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not 0.0 <= decay_frac <= 1.0:
        raise ValueError(f"decay_frac must be between 0 and 1, got {decay_frac}")
    first_decay_step = int(total_steps * (1.0 - decay_frac))

    def at(step: int) -> dict[str, float]:
        return decay if step >= first_decay_step else stable

    return at


@dataclass(frozen=True)
class ArrayWindows:
    """Windows cut from one flat token array. The shape a .bin shard arrives in."""

    tokens: np.ndarray
    context: int

    def __len__(self) -> int:
        return len(self.tokens) // self.context

    def window(self, index: int) -> np.ndarray:
        if not 0 <= index < len(self):
            raise IndexError(f"window {index} outside 0..{len(self)}")
        start = index * self.context
        return self.tokens[start : start + self.context]


def observed_mixture(counts: Sequence[dict[str, int]]) -> dict[str, float]:
    """The proportions actually drawn, for checking a run against its plan."""
    total: dict[str, int] = {}
    for batch in counts:
        for tier, n in batch.items():
            total[tier] = total.get(tier, 0) + n
    drawn = sum(total.values())
    if not drawn:
        return {}
    return {tier: n / drawn for tier, n in sorted(total.items())}


@dataclass
class ShardedWindows:
    """Windows across many token shards, memory-mapped, never all resident.

    The naive version concatenates a tier into one array. Tier B is about 9.5B
    tokens, which is 19 GB as uint16 and 76 GB once NumPy widens it to int64,
    so that version dies before the first step. Here each shard is mapped on
    demand and only the window being used is materialised, which is two
    kilobytes.

    A window never straddles two shards. The tail of a shard shorter than one
    window is dropped, exactly as packing drops its own tail, so no window is
    ever half one file and half another.

    Shards are raw token streams, not .npy: the backbone's corpus was written
    with `tofile` as uint16, which has no header, so the dtype is supplied
    here and the length comes from the file size. Writing our own rebuilt
    shards the same way keeps one format across the whole corpus instead of
    two that must be told apart.
    """

    paths: list[Path]
    context: int
    dtype: np.dtype | type = np.uint16
    _maps: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    _counts: list[int] = field(default_factory=list, repr=False)
    _starts: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.context <= 0:
            raise ValueError(f"context must be positive, got {self.context}")
        if not self.paths:
            raise ValueError("no token shards given")
        width = np.dtype(self.dtype).itemsize
        running = 0
        for path in self.paths:
            size = path.stat().st_size
            if size % width:
                raise ValueError(f"{path.name} is {size} bytes, not a whole number of tokens")
            count = (size // width) // self.context
            self._counts.append(count)
            self._starts.append(running)
            running += count

    def __len__(self) -> int:
        return sum(self._counts)

    def window(self, index: int) -> np.ndarray:
        if not 0 <= index < len(self):
            raise IndexError(f"window {index} outside 0..{len(self)}")
        shard = bisect_right(self._starts, index) - 1
        offset = (index - self._starts[shard]) * self.context
        mapped = self._maps.get(shard)
        if mapped is None:
            mapped = np.memmap(self.paths[shard], dtype=self.dtype, mode="r")
            self._maps[shard] = mapped
        return np.asarray(mapped[offset : offset + self.context], dtype=np.int64)
