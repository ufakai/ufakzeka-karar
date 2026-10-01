"""Text normalisation, the 8-gram index and MinHash near-duplicate search.

Both halves work on the same normalised tokens, so a match cannot depend on
case, spacing or punctuation. Turkish casing needs care: Python lowercases
"I" to "i" and "İ" to "i" plus a combining dot, both wrong for Turkish, so the
two dotted and dotless capitals are mapped by hand before lowercasing.

The 8-gram rule is Tülu 3's (arXiv 2411.15124): an item overlaps
when more than half of its tokens sit inside 8-grams it shares with the other
side. Texts under eight tokens have no 8-gram and are compared whole instead.
"""

from __future__ import annotations

import functools
import hashlib
import pickle
import unicodedata
from collections import Counter
from collections.abc import Container, Iterable
from pathlib import Path

import numpy as np

NGRAM = 8
# More than this share of an item's tokens in shared 8-grams marks it.
OVERLAP_THRESHOLD = 0.5
# MinHash: 128 permutations over 5-token shingles, near-duplicate at 0.8.
NUM_PERM = 128
SHINGLE = 5
JACCARD_THRESHOLD = 0.8
# A reference id is "<set name>:<item id>", so the set is recoverable from it.
SET_SEPARATOR = ":"

_TURKISH_CAPITALS = str.maketrans({"İ": "i", "I": "ı"})


def _is_word_char(char: str) -> bool:
    return char.isalnum() or unicodedata.category(char).startswith("M")


def _is_punctuation(char: str) -> bool:
    return unicodedata.category(char)[0] in "PS"


def normalise(text: str) -> str:
    """NFKC, Turkish lowercasing, punctuation dropped unless inside a word, one space.

    Punctuation and symbols (emoji included) are kept only when a word
    character sits on both sides, so "jungkook'a", "e-posta" and "3.5" stay
    whole while quotes, brackets and sentence stops go.
    """
    text = unicodedata.normalize("NFKC", text).translate(_TURKISH_CAPITALS).lower()
    kept: list[str] = []
    last = len(text) - 1
    for position, char in enumerate(text):
        if _is_punctuation(char):
            inside = (
                0 < position < last
                and _is_word_char(text[position - 1])
                and _is_word_char(text[position + 1])
            )
            kept.append(char if inside else " ")
        else:
            kept.append(char)
    return " ".join("".join(kept).split())


def tokens(text: str) -> list[str]:
    """Whitespace tokens of the normalised text, a run of one repeated token kept once.

    The run collapse is ours, not Tülu 3's. Tweets in OffensEval-TR open with
    chains of up to twenty "@USER" placeholders, and without it any two such
    tweets share 8-grams made of nothing but placeholders. Measured on the
    local sets on 2026-09-22, it was the main source of flagged OffensEval rows.
    """
    toks: list[str] = []
    for token in normalise(text).split():
        if not toks or toks[-1] != token:
            toks.append(token)
    return toks


def _hash64(payload: str) -> int:
    return int.from_bytes(hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest(), "big")


def gram_hashes(toks: list[str], n: int = NGRAM) -> list[int]:
    """The 64-bit hash of every n-gram, in order: entry i covers tokens i to i+n-1."""
    return [_hash64(" ".join(toks[i : i + n])) for i in range(len(toks) - n + 1)]


def whole_hash(toks: list[str]) -> int:
    """Hash of a whole normalised text, for the short-text path."""
    return _hash64(" ".join(toks))


def ngram_set(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    """The n-grams of a text as token tuples. Readable; the index stores hashes instead."""
    toks = tokens(text)
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def covered_fraction(
    hashes: list[int], length: int, shared: Container[int], n: int = NGRAM
) -> float:
    """Share of `length` tokens that sit inside an n-gram whose hash is in `shared`."""
    if length == 0:
        return 0.0
    covered = np.zeros(length, dtype=bool)
    for position, value in enumerate(hashes):
        if value in shared:
            covered[position : position + n] = True
    return float(covered.sum()) / length


def set_name(ref_id: str) -> str:
    return ref_id.split(SET_SEPARATOR, 1)[0]


class NgramIndex:
    """Hashed 8-grams of the reference texts, each mapped to the references holding it.

    `grams` maps a 64-bit blake2b hash to the positions (in `ref_ids`) of the
    references that contain it; its keys are the set of hashed 8-grams. Each
    reference also keeps its own ordered hashes, so the reverse question, how
    much of a reference a training set covers, needs no second pass over the
    reference text. References are numbered to keep the lists small.
    """

    def __init__(self, n: int = NGRAM) -> None:
        self.n = n
        self.ref_ids: list[str] = []
        self.grams: dict[int, list[int]] = {}
        self.wholes: dict[int, list[int]] = {}
        self.ref_hashes: list[np.ndarray] = []
        self.ref_lengths: list[int] = []
        self.ref_whole: list[int] = []
        self.numbers: dict[str, int] = {}

    @classmethod
    def build(cls, items: Iterable[tuple[str, str]], n: int = NGRAM) -> NgramIndex:
        index = cls(n)
        for ref_id, text in items:
            index.add(ref_id, text)
        return index

    def __len__(self) -> int:
        return len(self.ref_ids)

    def add(self, ref_id: str, text: str) -> None:
        if SET_SEPARATOR not in ref_id:
            raise ValueError(f"reference id {ref_id!r} must be '<set>{SET_SEPARATOR}<item>'")
        if ref_id in self.numbers:
            raise ValueError(f"reference id {ref_id!r} added twice")
        number = len(self.ref_ids)
        self.numbers[ref_id] = number
        self.ref_ids.append(ref_id)
        toks = tokens(text)
        hashes = gram_hashes(toks, self.n)
        for value in dict.fromkeys(hashes):
            self.grams.setdefault(value, []).append(number)
        whole = whole_hash(toks)
        self.wholes.setdefault(whole, []).append(number)
        self.ref_hashes.append(np.asarray(hashes, dtype=np.uint64))
        self.ref_lengths.append(len(toks))
        self.ref_whole.append(whole)

    def overlap(self, text: str) -> tuple[float, list[str]]:
        """(share of the text's tokens inside 8-grams shared with the index, matched refs).

        A text under n tokens has no n-gram: it scores 1.0 when its whole
        normalised form equals a reference's, and 0.0 otherwise.
        """
        toks = tokens(text)
        if not toks:
            return 0.0, []
        if len(toks) < self.n:
            exact = self.wholes.get(whole_hash(toks), [])
            return (1.0 if exact else 0.0), sorted(self.ref_ids[i] for i in exact)
        hashes = gram_hashes(toks, self.n)
        fraction = covered_fraction(hashes, len(toks), self.grams, self.n)
        numbers: set[int] = set()
        for value in hashes:
            numbers.update(self.grams.get(value, ()))
        return fraction, sorted(self.ref_ids[i] for i in numbers)

    def is_contaminated(self, text: str) -> bool:
        return self.overlap(text)[0] > OVERLAP_THRESHOLD

    def covered_references(self, hashes: set[int], candidates: Iterable[str]) -> list[str]:
        """The candidates more than half of whose tokens sit in n-grams from `hashes`.

        Used on one training text and the references it shares n-grams with,
        to catch a short test item pasted inside a long row, which the row's
        own fraction hides.
        """
        found = []
        for ref_id in candidates:
            number = self.numbers[ref_id]
            ref_hashes = self.ref_hashes[number].tolist()
            share = covered_fraction(ref_hashes, self.ref_lengths[number], hashes, self.n)
            if share > OVERLAP_THRESHOLD:
                found.append(ref_id)
        return sorted(found)

    def set_sizes(self) -> Counter[str]:
        """How many items each reference set has in the index."""
        return Counter(set_name(ref_id) for ref_id in self.ref_ids)

    def reference_coverage(self, train_grams: set[int], train_wholes: set[int]) -> list[float]:
        """For every reference, in `ref_ids` order, the share of its tokens a training set covers.

        `train_grams` holds the hashed n-grams of every training text and
        `train_wholes` the hashes of their whole normalised forms. A reference
        under n tokens is covered only by an identical training text.
        """
        coverage: list[float] = []
        for hashes, length, whole in zip(
            self.ref_hashes, self.ref_lengths, self.ref_whole, strict=True
        ):
            if length < self.n:
                coverage.append(1.0 if length and whole in train_wholes else 0.0)
            else:
                coverage.append(covered_fraction(hashes.tolist(), length, train_grams, self.n))
        return coverage

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> NgramIndex:
        # Only ever load an index this repo built: unpickling runs code.
        with path.open("rb") as handle:
            index = pickle.load(handle)
        if not isinstance(index, cls):
            raise TypeError(f"{path} does not hold an NgramIndex")
        return index


# MinHash. Each permutation is h(x) = ((a x + b) mod p) mod 2^32 over 32-bit
# shingle hashes, with p the Mersenne prime 2^61 - 1. a and b are kept under
# 2^31 so a x + b stays under 2^64 and numpy's uint64 never wraps.
_MERSENNE = np.uint64((1 << 61) - 1)
_MAX_HASH = np.uint64((1 << 32) - 1)


@functools.lru_cache(maxsize=8)
def _permutations(num_perm: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 1 << 31, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, 1 << 31, size=num_perm, dtype=np.uint64)
    return a, b


def minhash_signature(
    text: str, num_perm: int = NUM_PERM, shingle: int = SHINGLE, seed: int = 1
) -> np.ndarray | None:
    """MinHash signature over the set of `shingle`-token shingles, or None under `shingle` tokens.

    Texts that short are left to the exact whole-text comparison.
    """
    toks = tokens(text)
    if len(toks) < shingle:
        return None
    shingles = {" ".join(toks[i : i + shingle]) for i in range(len(toks) - shingle + 1)}
    values = np.fromiter(
        (
            int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "big")
            for s in shingles
        ),
        dtype=np.uint64,
        count=len(shingles),
    )
    a, b = _permutations(num_perm, seed)
    permuted = (a[:, None] * values[None, :] + b[:, None]) % _MERSENNE & _MAX_HASH
    return permuted.min(axis=1)


def estimated_jaccard(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError("signatures differ in length")
    return float(np.mean(first == second))


def _integrate(function, low: float, high: float) -> float:
    grid = np.linspace(low, high, 501)
    return float(np.trapezoid(function(grid), grid))


@functools.lru_cache(maxsize=8)
def optimal_bands(threshold: float, num_perm: int) -> tuple[int, int]:
    """Bands and rows with bands * rows <= num_perm that best separate pairs at `threshold`.

    Minimises the equal-weighted area of false positives under the threshold
    and false negatives above it, the rule the datasketch library uses. The
    candidate stage only has to find pairs; every candidate is then checked
    against the threshold on its full signature.
    """
    best: tuple[float, int, int] | None = None
    for bands in range(1, num_perm + 1):
        for rows in range(1, num_perm // bands + 1):

            def hit(s, bands=bands, rows=rows):
                return 1.0 - (1.0 - s**rows) ** bands

            error = 0.5 * _integrate(hit, 0.0, threshold) + 0.5 * _integrate(
                lambda s, hit=hit: 1.0 - hit(s), threshold, 1.0
            )
            if best is None or error < best[0]:
                best = (error, bands, rows)
    assert best is not None
    return best[1], best[2]


class MinHashLSH:
    """Banded locality-sensitive hashing over MinHash signatures, verified on query."""

    def __init__(self, threshold: float = JACCARD_THRESHOLD, num_perm: int = NUM_PERM) -> None:
        self.threshold = threshold
        self.num_perm = num_perm
        self.bands, self.rows = optimal_bands(threshold, num_perm)
        self.tables: list[dict[bytes, list[str]]] = [{} for _ in range(self.bands)]
        self.signatures: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.signatures)

    def _band_keys(self, signature: np.ndarray) -> list[bytes]:
        if signature.shape != (self.num_perm,):
            raise ValueError(f"signature must have {self.num_perm} values")
        return [
            signature[band * self.rows : (band + 1) * self.rows].tobytes()
            for band in range(self.bands)
        ]

    def add(self, key: str, signature: np.ndarray) -> None:
        if key in self.signatures:
            raise ValueError(f"key {key!r} added twice")
        self.signatures[key] = signature
        for table, band_key in zip(self.tables, self._band_keys(signature), strict=True):
            table.setdefault(band_key, []).append(key)

    def query(self, signature: np.ndarray) -> list[tuple[str, float]]:
        """Stored keys whose estimated Jaccard with `signature` is at least the threshold."""
        candidates: set[str] = set()
        for table, band_key in zip(self.tables, self._band_keys(signature), strict=True):
            candidates.update(table.get(band_key, ()))
        found = []
        for key in candidates:
            jaccard = estimated_jaccard(signature, self.signatures[key])
            if jaccard >= self.threshold:
                found.append((key, jaccard))
        return sorted(found, key=lambda pair: (-pair[1], pair[0]))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> MinHashLSH:
        with path.open("rb") as handle:
            lsh = pickle.load(handle)
        if not isinstance(lsh, cls):
            raise TypeError(f"{path} does not hold a MinHashLSH")
        return lsh
