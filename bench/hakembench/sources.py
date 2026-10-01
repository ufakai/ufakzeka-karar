"""HakemBench candidates from three untouched test splits.

    uv run --group instrument python -m bench.hakembench.sources --built data/built

Three tracks, each from the test split its training converter refuses, with
the training task's own question, filters and masks, so a test item reads like
a training row and is scored on the same question:

- hukuk: icgcihan/Turkish_Constutional_Court_Decisions test.json (CC BY 4.0),
  the court's summary of the complaint and the right it is filed under, the
  aym_rights question of data/typed/legal.py. Up to 200 items, drawn so the
  rare rights are whole and the common ones share the rest (water_fill),
  since the natural mix is two-fifths civil fair trial. The court's docket
  repeats itself, so many test summaries are already in training and fewer
  than 200 may remain.
- guvenlik: the Turkish rows of the test splits of
  3nesdeniz/turkish-prompt-injection-1k and 3nesdeniz/guardrail-hard-negatives
  (CC BY 4.0), the prompt_injection question of data/typed/guardrails.py,
  masked and deduplicated as there. Balanced 1:1 as far as the rarer class
  goes, at most 320.
- arama: PaDaS-Lab/webfaq-retrieval's Turkish test queries and qrels (CC BY
  4.0), joined to the raw corpus for site and page and filtered as
  data/typed/relevance.py does, with negatives found by its Pool among test
  pairs only. 160 questions, each with its own answer and one negative, 320
  items; a question's two items share a half and a group.

The training texts are every jsonl under --built. A candidate a training text
carries (bench/hakembench/common.py, `overlapping`) is dropped before the
draw, and the counts go to the report. Test files are downloaded to
data/raw/_hakembench/, never to the training converters' folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from bench.hakembench.common import (
    CACHE,
    CANDIDATES,
    DROP_RULES,
    Candidate,
    draw_per_class,
    fetch_test,
    overlapping,
    text_sha256,
    training_files,
    training_texts,
    water_fill,
    write_track,
)
from data.decontam.ngrams import normalise
from data.label.texts import mask_personal_data
from data.typed import guardrails, legal, relevance
from data.typed.common import Item as LegalItem
from data.typed.common import resolve
from data.typed.sources import DOWNLOADS, hub_file, read_jsonl, read_parquet

REPORT = Path("results/step8/candidates.json")
LICENCE = "CC-BY-4.0"

# Legal ----------------------------------------------------------------------

LEGAL_TRACK = "hukuk"
LEGAL_ITEMS = 200
LEGAL_FILES = {
    "test.json": "716e370a5d6becf05a8ce1fb73aab48156781a26b5f7b94a7227913242f2db72",
    "README.md": "38ad7475a3704a761cd8324202a5d58013fe172a7d7d4106f0f99f0bf77c953f",
}

# Guardrails -----------------------------------------------------------------

GUARD_TRACK = "guvenlik"
GUARD_MAX = 320
GUARD_SOURCES = (
    (
        guardrails.Source(
            "tpi1k",
            "3nesdeniz/turkish-prompt-injection-1k",
            "1cbd1152d9732f40148fbd5bb7cf0f58ddfe84c6",
            {"test": ("data/test.parquet", 16_130)},
        ),
        {
            "data/test.parquet": "9431b17735ab1eb1112881b95c81ccc2b4786a9fd36b9e822865762dc55d4728",
            "README.md": "a9ca9377b88214fad7d12bbcf66295bf628b26417e2267e042742d93d5fa7f87",
        },
    ),
    (
        guardrails.Source(
            "ghn",
            "3nesdeniz/guardrail-hard-negatives",
            "638188743abb089708163b2408a0ef6279721869",
            {"test": ("data/test.parquet", 27_073)},
            language="tr",
        ),
        {
            "data/test.parquet": "945a5d354241754a8e67b4e51a8760edf3784bdde55e0f41381662448d716783",
            "README.md": "972138da25bd94925926f40e89d355ba180c516cf82f60b81ffdaf001e5f6904",
        },
    ),
)

# Relevance ------------------------------------------------------------------

RELEVANCE_TRACK = "arama"
RELEVANCE_QUESTIONS = 160
# Negatives are built for this many questions, so the draw can pass over any
# the training check drops.
RELEVANCE_POOL = 480
RELEVANCE_SITE_CAP = 2
RELEVANCE_FILES = {
    "tur/queries-test.jsonl": "d819c1daf5d5bd06d9802c3438b2ef29d22971e3891a0d95c55655a83043156c",
    "tur/qrels-test.jsonl": "a03df09d7d3387fff5e61f1d83d2cb4556aeb023253c0b0c5b10b16d5d471976",
    "README.md": "0c254e7f764673c07cdcb81c58d721264df09f25ebda791381de49cd18be465c",
}


def hub_url(repo: str) -> str:
    return f"https://huggingface.co/datasets/{repo}"


# Legal ----------------------------------------------------------------------


def legal_candidates(records: list[dict], drops: Counter) -> list[Candidate]:
    """One candidate per distinct masked summary, with legal.py's items and resolve."""
    found = resolve(legal.items(records, drops), drops)
    return [legal_candidate(item) for item in found]


def legal_candidate(item: LegalItem) -> Candidate:
    return Candidate(
        track=LEGAL_TRACK,
        state=item.state,
        question_id=legal.TASK,
        question=legal.QUESTION,
        gold=item.label,
        source=hub_url(legal.REPO),
        licence=LICENCE,
        revision=legal.REVISION,
        source_file="test.json",
        source_row_id=item.source_id,
        split_key=item.state,
        label_kind="human",
    )


def select_legal(candidates: list[Candidate], total: int = LEGAL_ITEMS) -> list[Candidate]:
    counts = Counter(str(c.gold) for c in candidates)
    quota = water_fill(dict(counts), total)
    return draw_per_class(candidates, quota, lambda c: str(c.gold))


def load_legal(root: Path) -> list[dict]:
    paths = {
        name: fetch_test(legal.REPO, legal.REVISION, name, sha, root)
        for name, sha in LEGAL_FILES.items()
    }
    return json.loads(paths["test.json"].read_text(encoding="utf-8"))


# Guardrails -----------------------------------------------------------------


def guard_candidates(
    per_source: list[tuple[guardrails.Source, list[dict]]], drops: Counter
) -> list[Candidate]:
    """guardrails.py's filters, masks and cross-source dedupe, on test rows only."""
    found = []
    for source, records in per_source:
        found.extend(guardrails.injection_candidates(source, "test", records))
    kept, dropped = guardrails.dedupe(found)
    drops.update(dropped)
    sources = {source.key: source for source, _ in per_source}
    out = []
    for c in kept:
        source = sources[c.source]
        out.append(
            Candidate(
                track=GUARD_TRACK,
                state=c.text,
                question_id="prompt_injection",
                question=guardrails.INJECTION_QUESTION,
                gold=bool(c.label),
                source=hub_url(source.repo),
                licence=LICENCE,
                revision=source.revision,
                source_file=source.files["test"][0],
                source_row_id=c.source_id,
                split_key=c.text,
                label_kind="rule",
            )
        )
    return out


def select_guard(candidates: list[Candidate], most: int = GUARD_MAX) -> list[Candidate]:
    """Equal numbers of each class, as many as the rarer class allows, at most `most`."""
    counts = Counter(str(c.gold) for c in candidates)
    each = min(min(counts.values(), default=0), most // 2)
    if len(counts) < 2:
        each = 0
    return draw_per_class(candidates, dict.fromkeys(counts, each), lambda c: str(c.gold))


def load_guard(root: Path) -> list[tuple[guardrails.Source, list[dict]]]:
    out = []
    for source, files in GUARD_SOURCES:
        paths = {
            path: fetch_test(source.repo, source.revision, path, sha, root)
            for path, sha in files.items()
        }
        out.append((source, read_parquet(paths[source.files["test"][0]])))
    return out


# Relevance ------------------------------------------------------------------


def relevance_candidates(
    pairs: list[relevance.Pair],
    questions: int = RELEVANCE_POOL,
    cap: int = RELEVANCE_SITE_CAP,
    stats: Counter | None = None,
) -> list[tuple[Candidate, Candidate]]:
    """(positive, negative) per sampled test question, as relevance.py builds its rows.

    Filters and masks as relevance_rows; one pool, of test pairs only.
    """
    stats = stats if stats is not None else Counter()
    kept = []
    for pair in pairs:
        reason = relevance.keep(pair)
        if reason:
            stats[f"dropped_{reason}"] += 1
            continue
        kept.append(
            relevance.Pair(
                pair.qid,
                mask_personal_data(pair.question),
                mask_personal_data(pair.answer),
                pair.origin,
                pair.url,
            )
        )
    pool = relevance.Pool(kept)
    stats["pool"] = len(pool.pairs)
    out = []
    for i in relevance.sample(pool.pairs, questions, cap):
        found = pool.negative(i)
        if found is None:
            stats["no_negative"] += 1
            continue
        j, kind = found
        stats[f"negative_{kind}"] += 1
        pair, other = pool.pairs[i], pool.pairs[j]
        both = []
        for answer, gold, row in ((pair.answer, True, pair.qid), (other.answer, False, other.qid)):
            both.append(
                Candidate(
                    track=RELEVANCE_TRACK,
                    state=relevance.state(pair.question, answer),
                    question_id=relevance.TASK,
                    question=relevance.QUESTION,
                    gold=gold,
                    source=hub_url(relevance.RETRIEVAL_REPO),
                    licence=LICENCE,
                    revision=relevance.RETRIEVAL_REVISION,
                    source_file="tur/queries-test.jsonl, tur/qrels-test.jsonl, tur/corpus.jsonl",
                    source_row_id=f"query {pair.qid}, answer of query {row}",
                    split_key=normalise(pair.question),
                    label_kind="rule",
                    group=f"webfaq-tur-test:{pair.qid}",
                )
            )
        out.append((both[0], both[1]))
    return out


def select_relevance(
    pairs: list[tuple[Candidate, Candidate]], dropped: set[str], questions: int
) -> list[Candidate]:
    """The first `questions` pairs, in sample order, with neither text dropped."""
    out: list[Candidate] = []
    for positive, negative in pairs:
        if len(out) >= 2 * questions:
            break
        if positive.state in dropped or negative.state in dropped:
            continue
        if positive.state == negative.state:
            continue
        out.extend((positive, negative))
    return out


def load_relevance(root: Path) -> list[relevance.Pair]:
    local = {
        p: fetch_test(relevance.RETRIEVAL_REPO, relevance.RETRIEVAL_REVISION, p, sha, root)
        for p, sha in RELEVANCE_FILES.items()
    }
    corpus_path, corpus_size = relevance.RETRIEVAL_FILES["corpus"]
    corpus = hub_file(
        relevance.RETRIEVAL_REPO, relevance.RETRIEVAL_REVISION, corpus_path, corpus_size, DOWNLOADS
    )
    raw = [
        hub_file(relevance.RAW_REPO, relevance.RAW_PARQUET, path, size, DOWNLOADS)
        for path, size in relevance.RAW_FILES
    ]
    queries = read_jsonl(local["tur/queries-test.jsonl"])
    qrels = read_jsonl(local["tur/qrels-test.jsonl"])
    wanted = {r["corpus-id"] for r in qrels}
    answers = [d for d in read_jsonl(corpus) if d["_id"] in wanted]
    pairs = relevance.train_pairs(queries, answers, qrels)
    joined, unjoined = relevance.join_sites(pairs, relevance.iter_raw(raw))
    if unjoined:
        print(f"arama: {unjoined} test pairs did not join to the raw corpus", file=sys.stderr)
    return joined


# Build ----------------------------------------------------------------------


def dedupe_across(candidates: list[Candidate]) -> list[Candidate]:
    """The first candidate of each text; a text is one item."""
    seen: set[str] = set()
    out = []
    for c in candidates:
        if c.state not in seen:
            seen.add(c.state)
            out.append(c)
    return out


def overlap_summary(pool: list[Candidate], found: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Counts per rule over a track's pool, and each dropped or row_ngram candidate."""
    rows = []
    rules: Counter = Counter()
    for c in pool:
        entry = found.get(c.state)
        if entry is None or not entry["rules"]:
            continue
        rules.update(entry["rules"])
        rows.append({"source_row_id": c.source_row_id, "text_sha256": text_sha256(c.state),
                     **entry})  # fmt: skip
    return {
        "candidates": len(pool),
        "dropped": sum(r["dropped"] for r in rows),
        "row_ngram_only_kept": sum(not r["dropped"] for r in rows),
        "by_rule": dict(sorted(rules.items())),
        "rows": rows,
    }


def sources() -> list[dict[str, Any]]:
    """Every pinned test file: repository, revision, path and sha256."""
    pinned = [(legal.REPO, legal.REVISION, LEGAL_FILES)]
    pinned += [(source.repo, source.revision, files) for source, files in GUARD_SOURCES]
    pinned += [(relevance.RETRIEVAL_REPO, relevance.RETRIEVAL_REVISION, RELEVANCE_FILES)]
    return [{"source": hub_url(repo), "revision": revision, "path": path, "sha256": sha}
            for repo, revision, files in pinned for path, sha in files.items()]  # fmt: skip


def build(built: Path, out: Path = CANDIDATES, root: Path = CACHE) -> dict[str, Any]:
    files = training_files(built)
    if not files:
        raise FileNotFoundError(f"{built}: no training files, so the overlap check would be empty")
    names = [p.relative_to(built).as_posix() for p in files]
    report: dict[str, Any] = {
        "sources": sources(),
        "drop_rules": list(DROP_RULES),
        "training_files": names,
        "tracks": {},
    }

    legal_drops: Counter = Counter()
    legal_pool = legal_candidates(load_legal(root), legal_drops)
    guard_drops: Counter = Counter()
    guard_pool = guard_candidates(load_guard(root), guard_drops)
    relevance_stats: Counter = Counter()
    relevance_pool = relevance_candidates(load_relevance(root), stats=relevance_stats)
    relevance_flat = dedupe_across([c for pair in relevance_pool for c in pair])

    texts = {c.state: c.state for c in [*legal_pool, *guard_pool, *relevance_flat]}
    found = overlapping(texts, training_texts(files, built))
    dropped = {text for text, entry in found.items() if entry["dropped"]}

    legal_items = select_legal([c for c in legal_pool if c.state not in dropped])
    guard_items = select_guard([c for c in guard_pool if c.state not in dropped])
    relevance_items = select_relevance(relevance_pool, dropped, RELEVANCE_QUESTIONS)

    for track, pool, items, extra in (
        (LEGAL_TRACK, legal_pool, legal_items, {"source_drops": dict(legal_drops)}),
        (GUARD_TRACK, guard_pool, guard_items, {"source_drops": dict(guard_drops)}),
        (RELEVANCE_TRACK, relevance_flat, relevance_items, {"stats": dict(relevance_stats)}),
    ):
        counts = write_track(items, out, track)
        report["tracks"][track] = {
            **counts,
            "pool_gold": dict(sorted(Counter(str(c.gold).lower() for c in pool).items())),
            "overlap": overlap_summary(pool, found),
            **extra,
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--built", type=Path, default=Path("data/built"))
    parser.add_argument("--out", type=Path, default=CANDIDATES)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    report = build(args.built, args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")
    summary = {
        track: {k: v for k, v in r.items() if k != "overlap"}
        | {"overlap_dropped": r["overlap"]["dropped"]}
        for track, r in report["tracks"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
