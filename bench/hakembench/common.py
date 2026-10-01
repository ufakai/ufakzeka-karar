"""Pieces the HakemBench candidate converters share (bench/hakembench/sources.py).

A candidate is one test text with one typed question and its source label.
This module does the same things for every track: downloads a pinned test
file into data/raw/_hakembench/ (gitignored, with its own manifest entry),
splits items into the public and private halves by a hash, draws items in a
fixed hash order, finds candidates whose text the training data already
holds, and writes the items and their provenance.

The training check. bench/dev.py removes training rows, so it asks of each
training row whether it carries a dev text. Here the test item is what goes,
so the question is asked of each candidate instead, with the rule the
decontamination already uses for a reference item (data/decontam/cli.py, the
set-level rule of Tülu 3, arXiv 2411.15124): a candidate is overlapped when a
training text equals it after normalisation, when the training texts together
hold more than half of its tokens inside shared 8-grams, or when a training
text is a MinHash near-duplicate of it. The training side is every built file
at once, not one file at a time, which is the stricter reading.

dev.py's other rule, a training row with more than half of its own tokens in
8-grams shared with the text, is recorded as row_ngram but does not drop a
candidate by itself: when that row covers under half of the candidate, most
of the candidate is text the training data does not hold. On the legal
summaries it fires on the court's closing formula ("... nedeniyle ...
hakkının ihlal edildiği iddiasına ilişkindir"), which nearly every summary
ends with.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.harness.items import Item
from data import hub
from data.decontam.ngrams import (
    OVERLAP_THRESHOLD,
    MinHashLSH,
    NgramIndex,
    covered_fraction,
    gram_hashes,
    minhash_signature,
    tokens,
    whole_hash,
)

CACHE = Path("data/raw/_hakembench")
CANDIDATES = Path("bench/hakembench/candidates")
HALVES = ("public", "private")
PRIVATE_SHARE = 0.5
# Rules that drop a candidate. row_ngram is reported only (module docstring).
DROP_RULES = ("whole", "covered", "near_duplicate")
_FULL_HASH = re.compile(r"[0-9a-f]{40}")

Fetch = Callable[[str, Path], None]


@dataclass(frozen=True)
class Candidate:
    """One test text, its question and gold label, and where it came from."""

    track: str
    state: str
    question_id: str
    question: dict[str, Any]
    gold: str | bool
    source: str
    licence: str
    revision: str
    source_file: str
    source_row_id: str
    # The text the half is drawn from; items that must share a half share it.
    split_key: str
    label_kind: str
    # Items drawn together (a question and its two passages); empty when alone.
    group: str = ""


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unit(salt: str, key: str) -> float:
    digest = hashlib.blake2b(f"{salt}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def half(key: str) -> str:
    """'private' or 'public' by a hash of the key, half each; never moves for a key."""
    return "private" if _unit("hakembench-half", key) < PRIVATE_SHARE else "public"


def unit_halves(units: dict[str, tuple[str, int]], salt: str = "hakembench-half",
                held: dict[str, dict[str, int]] | None = None,
                global_ties: bool = False) -> dict[str, str]:  # fmt: skip
    """A half per unit (a writer request, a template family), siblings never split.

    `units` maps a unit to (stratum, items). Within each stratum the units are taken in
    a fixed hash order and each goes to the half holding fewer of that stratum's items,
    so both halves see every stratum in about equal measure. `held` gives items a stratum
    already has in each half (units placed earlier, which never move). A tie goes to the
    public half, or with `global_ties` to the half holding fewer items overall, so strata
    with an odd number of units do not all lean the same way.
    """
    out: dict[str, str] = {}
    strata: dict[str, list[str]] = {}
    for unit, (stratum, _) in units.items():
        strata.setdefault(stratum, []).append(unit)
    total = {"public": 0, "private": 0}
    for stratum in sorted(strata):
        start = (held or {}).get(stratum, {})
        counts = {"public": start.get("public", 0), "private": start.get("private", 0)}
        for unit in sorted(strata[stratum], key=lambda u: _unit(salt, u)):
            if counts["public"] != counts["private"]:
                side = "public" if counts["public"] < counts["private"] else "private"
            elif global_ties and total["private"] < total["public"]:
                side = "private"
            else:
                side = "public"
            out[unit] = side
            counts[side] += units[unit][1]
            total[side] += units[unit][1]
    return out


def draw_order(key: str) -> float:
    """A fixed position for drawing items, independent of the half."""
    return _unit("hakembench-draw", key)


def item_id(candidate: Candidate) -> str:
    return f"{candidate.track}-{text_sha256(candidate.state)[:16]}"


def fetch_test(
    repo: str,
    revision: str,
    path: str,
    sha256: str,
    root: Path = CACHE,
    *,
    download: Fetch = hub.download,
) -> Path:
    """One test file from the hub at a pinned commit, downloaded once and checked by hash.

    hub.download sends the token to the hub's hosts only.
    """
    if not _FULL_HASH.fullmatch(revision):
        raise ValueError(f"{repo}: revision must be a full 40-character commit hash")
    target = root / repo.replace("/", "__") / revision / path
    if not target.is_file():
        download(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}", target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"{target}: sha256 {digest} is not the pinned {sha256}")
    return target


def water_fill(counts: dict[str, int], total: int) -> dict[str, int]:
    """Per class, how many to draw so `total` are drawn with the rarest classes whole.

    The largest cap c with sum(min(n, c)) <= total, then one more item for the
    largest classes, in name order, until `total` is reached or nothing is left.
    """
    total = min(total, sum(counts.values()))
    cap = 0
    while sum(min(n, cap + 1) for n in counts.values()) <= total:
        cap += 1
        if cap > max(counts.values(), default=0):
            break
    quota = {name: min(n, cap) for name, n in counts.items()}
    for name in sorted(counts, key=lambda k: (-counts[k], k)):
        if sum(quota.values()) >= total:
            break
        if quota[name] < counts[name]:
            quota[name] += 1
    return quota


def draw_per_class(
    candidates: Iterable[Candidate], quota: dict[str, int], label: Callable[[Candidate], str]
) -> list[Candidate]:
    """The first `quota[class]` candidates of each class in draw order."""
    taken: dict[str, int] = dict.fromkeys(quota, 0)
    out = []
    for candidate in sorted(candidates, key=lambda c: (draw_order(c.state), c.state)):
        name = label(candidate)
        if taken.get(name, 0) < quota.get(name, 0):
            taken[name] += 1
            out.append(candidate)
    return out


def row_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def training_texts(files: Iterable[Path], root: Path) -> Iterator[tuple[str, str]]:
    """(where, text) for every row of every file; where is the file under root and the row id."""
    for path in files:
        name = path.relative_to(root).as_posix()
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                yield f"{name}:{row.get('row_id', number)}", row_text(row["state"])


def training_files(root: Path) -> list[Path]:
    """Every jsonl under the built data root: train, validation and held-out files alike."""
    return sorted(root.rglob("*.jsonl"))


def overlapping(
    texts: dict[str, str], training: Iterable[tuple[str, str]]
) -> dict[str, dict[str, Any]]:
    """For each candidate key a training text touches: its rules, coverage and training rows.

    Rules:
    - whole: a training text equals it after normalisation (the only test for
      texts under eight tokens);
    - covered: the training texts together hold more than half of its tokens
      inside shared 8-grams (`coverage`); its row is the one training row
      that covers most of it;
    - near_duplicate: a training text is a MinHash near-duplicate (estimated
      Jaccard of 0.8 or more on 5-token shingles);
    - row_ngram: one training text has more than half of its own tokens in
      8-grams shared with the candidate (bench/dev.py's "ngram" rule).
    `dropped` is whether any of DROP_RULES holds. Keys no training text
    shares an 8-gram, a whole form or a near-duplicate with are left out.
    """
    index = NgramIndex.build((f"hb:{key}", text) for key, text in texts.items())
    lsh = MinHashLSH()
    for ref_id, text in zip(index.ref_ids, texts.values(), strict=True):
        signature = minhash_signature(text)
        if signature is not None:
            lsh.add(ref_id, signature)
    own = [set(hashes.tolist()) for hashes in index.ref_hashes]
    shared: set[int] = set()
    notes: dict[str, dict[str, Any]] = {}
    best: dict[str, float] = {}

    def note(ref_id: str, rule: str, where: str) -> None:
        entry = notes.setdefault(ref_id, {"rules": set(), "rows": {}})
        entry["rules"].add(rule)
        entry["rows"].setdefault(rule, where)

    for where, text in training:
        toks = tokens(text)
        if not toks:
            continue
        for number in index.wholes.get(whole_hash(toks), ()):
            note(index.ref_ids[number], "whole", where)
        signature = minhash_signature(text)
        if signature is not None:
            for ref_id, _ in lsh.query(signature):
                note(ref_id, "near_duplicate", where)
        if len(toks) < index.n:
            continue
        hashes = gram_hashes(toks, index.n)
        row_grams = {value for value in hashes if value in index.grams}
        if not row_grams:
            continue
        shared |= row_grams
        fraction, refs = index.overlap(text)
        for ref_id in refs:
            number = index.numbers[ref_id]
            share = covered_fraction(
                index.ref_hashes[number].tolist(), index.ref_lengths[number], row_grams, index.n
            )
            entry = notes.setdefault(ref_id, {"rules": set(), "rows": {}})
            if share > best.get(ref_id, 0.0):
                best[ref_id] = share
                entry["rows"]["covered"] = where
            if fraction > OVERLAP_THRESHOLD and (
                covered_fraction(hashes, len(toks), own[number], index.n) > OVERLAP_THRESHOLD
            ):
                note(ref_id, "row_ngram", where)

    coverage = dict(zip(index.ref_ids, index.reference_coverage(shared, set()), strict=True))
    found: dict[str, dict[str, Any]] = {}
    for ref_id, entry in notes.items():
        rules = set(entry["rules"])
        if coverage[ref_id] > OVERLAP_THRESHOLD:
            rules.add("covered")
        found[ref_id.split(":", 1)[1]] = {
            "rules": sorted(rules),
            "coverage": round(coverage[ref_id], 4),
            "rows": {rule: entry["rows"][rule] for rule in sorted(rules)},
            "dropped": any(rule in DROP_RULES for rule in rules),
        }
    return found


def to_item(candidate: Candidate) -> Item:
    return Item(
        id=item_id(candidate),
        track=candidate.track,
        state=candidate.state,
        questions={candidate.question_id: candidate.question},
        gold={candidate.question_id: candidate.gold},
    )


def meta_row(candidate: Candidate) -> dict[str, Any]:
    row = {
        "id": item_id(candidate),
        "track": candidate.track,
        "half": half(candidate.split_key),
        "source": candidate.source,
        "licence": candidate.licence,
        "source_revision": candidate.revision,
        "source_file": candidate.source_file,
        "source_row_id": candidate.source_row_id,
        "label_kind": candidate.label_kind,
        "text_sha256": text_sha256(candidate.state),
    }
    if candidate.group:
        row["group"] = candidate.group
    return row


def write_track(candidates: list[Candidate], out_dir: Path, name: str) -> dict[str, Any]:
    """Write <name>.jsonl (items) and <name>.meta.jsonl (provenance). Returns counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(candidates, key=item_id)
    ids = [item_id(c) for c in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name}: two candidates share a text")
    items = [to_item(c).model_dump_json() for c in ordered]
    metas = [json.dumps(meta_row(c), ensure_ascii=False) for c in ordered]
    (out_dir / f"{name}.jsonl").write_text("".join(i + "\n" for i in items), encoding="utf-8")
    (out_dir / f"{name}.meta.jsonl").write_text("".join(m + "\n" for m in metas), "utf-8")
    halves = {h: sum(half(c.split_key) == h for c in ordered) for h in HALVES}
    gold: dict[str, int] = {}
    for c in ordered:
        gold[str(c.gold).lower()] = gold.get(str(c.gold).lower(), 0) + 1
    return {"items": len(ordered), "halves": halves, "gold": dict(sorted(gold.items()))}
