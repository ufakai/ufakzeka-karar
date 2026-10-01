"""Passage relevance from WebFAQ, track "arama", task "webfaq_relevance".

A noul question: does this passage answer this question? The pairs are
Turkish FAQ entries from web pages' schema.org FAQPage markup, collected by
PaDaS-Lab (CC BY 4.0, card read 2026-09-23). A question's own answer on its
page is a positive; the same question with an answer from a different page is
a negative. The label comes from page structure, not from a person, so
label_kind is "rule".

Which pairs. The raw corpus (PaDaS-Lab/webfaq, config tur, 1.45 million
pairs) has no splits. Its curated subset, PaDaS-Lab/webfaq-retrieval, has
train and test splits per language, and its test split is an evaluation set
(MTEB's WebFAQRetrieval). Only pairs in the retrieval set's Turkish train
qrels are used, and its test queries and qrels are never downloaded. The
retrieval files carry no site or page, so each train pair is joined back to
the raw corpus by its exact question and answer text to recover them; every
train pair joined when this was written.

Filters, before sampling: the same as the other FAQ texts (data/label/texts.py
`blocked`): complaint and gambling sites, markup, mojibake, link-only answers,
text that is not Turkish; and answers under 20 or over 1,500 characters.
Personal data is masked in both question and answer.

Sampling. Validation is a tenth of the pairs by a hash of the query id, and
negatives are drawn only from the same split's pool, so no text crosses
splits. About TARGET_QUESTIONS questions are taken in a seeded order with at
most SITE_CAP per site, since a handful of travel sites hold much of the
corpus. Each question yields two rows, its own answer and one negative, so the
classes are balanced 1:1 and the question alone says nothing about the label.

Negatives, hardest first:

1. same_site: an answer from another page of the same site that shares the
   most (IDF-weighted) keywords with the question;
2. keyword: an answer from any other page that shares the most keywords;
3. same_site_any, then random: when nothing shares a keyword.

A candidate is refused when its answer is a near-duplicate of the positive
(character 5-gram Jaccard at or above 0.5, or one contains the other), or when
its own question is close to ours (word Jaccard at or above 0.5). Sites repeat
one question template across pages ("... otopark var mı?"), and an answer to
the same template may well answer ours; such pairs are not negatives.
Keywords are the first five letters of normalised words of at least three
letters outside a stop list, a crude stem for Turkish suffixes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from data.decontam.ngrams import normalise
from data.label.texts import TURKISH_WORDS, blocked, mask_personal_data
from data.typed.sources import (
    BUILT,
    DOWNLOADS,
    hash_split,
    hub_file,
    noul_balance,
    read_jsonl,
    sha256_of,
    unique_rows,
    write_meta,
    write_rows,
)
from schema.rows import TrainingRow

TRACK = "arama"
TASK = "webfaq_relevance"
RECIPE = "typed-webfaq-relevance-v1"

RAW_REPO = "PaDaS-Lab/webfaq"
RAW_MAIN = "692dd5d911a9f0172215e82e5fa1fb1fc2cc9c6e"
# The hub's parquet export of the main commit above (refs/convert/parquet).
RAW_PARQUET = "92be36774d8b618a7e8e8dca86092661f74e326a"
RAW_FILES = (("tur/default/0000.parquet", 148_363_699), ("tur/default/0001.parquet", 42_695_393))
RETRIEVAL_REPO = "PaDaS-Lab/webfaq-retrieval"
RETRIEVAL_REVISION = "1abff20b9ea37abeec151bd8bcfa78c31a48f9fb"
RETRIEVAL_FILES = {
    "queries": ("tur/queries-train.jsonl", 11_003_660),
    "qrels": ("tur/qrels-train.jsonl", 6_400_507),
    "corpus": ("tur/corpus.jsonl", 57_145_253),
}

TARGET_QUESTIONS = 10_000
SITE_CAP = 20
SEED = 1
MIN_ANSWER, MAX_ANSWER = 20, 1_500
NEAR_DUPLICATE = 0.5
SAME_TEMPLATE = 0.5
# Same-site candidates scored per question, and postings read per keyword.
SITE_SCAN = 400
POSTINGS_SCAN = 3_000
# Candidates tried per rule before moving to the next.
RANKED_CHECKS = 50
# A keyword on more than this share of a pool's answers carries no signal.
MAX_DF_SHARE = 0.02
STEM = 5
_STOP = (
    "nasıl nedir neden hangi hangisi kadar olan olarak için ile gibi daha çok var yok mıdır "
    "midir mudur müdür mısınız misiniz musunuz müsünüz nerede ne zaman kim kaç bir bu şu "
    "sizin bizim ben siz biz her olur oluyor olabilir yapılır yapabilirim yapmalıyım"
)
STOP_WORDS = frozenset(_STOP.split()) | TURKISH_WORDS

QUESTION = {
    "type": "noul",
    "instructions": "Metindeki pasaj, metindeki soruyu yanıtlıyor mu?",
    "criteria": {
        "true": "Pasaj bu sorunun cevabını veriyor.",
        "false": "Pasaj bu soruyu yanıtlamıyor; başka bir sorunun ya da konunun cevabı.",
    },
}


@dataclass(frozen=True)
class Pair:
    qid: str
    question: str
    answer: str
    origin: str
    url: str


def _key(question: str, answer: str) -> bytes:
    payload = question.strip() + "\x00" + answer.strip()
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()


def train_pairs(
    queries: Iterable[dict], corpus: Iterable[dict], qrels: Iterable[dict]
) -> list[tuple[str, str, str]]:
    """(query id, question, answer) for every train qrel."""
    texts = {q["_id"]: q["text"] for q in queries}
    docs = {d["_id"]: d["text"] for d in corpus}
    return [
        (str(r["query-id"]), texts[r["query-id"]], docs[r["corpus-id"]])
        for r in qrels
        if r.get("score", 1) > 0
    ]


def join_sites(
    pairs: list[tuple[str, str, str]], raw_rows: Iterable[dict]
) -> tuple[list[Pair], int]:
    """Attach site and page from the raw corpus by exact text. Returns pairs, unjoined count."""
    wanted = {_key(q, a): i for i, (_, q, a) in enumerate(pairs)}
    found: dict[int, tuple[str, str]] = {}
    for row in raw_rows:
        index = wanted.get(_key(row["question"], row["answer"]))
        if index is not None and index not in found:
            found[index] = (row["origin"] or "", row["url"] or "")
    joined = [
        Pair(qid, q.strip(), a.strip(), *found[i])
        for i, (qid, q, a) in enumerate(pairs)
        if i in found
    ]
    return joined, len(pairs) - len(joined)


def keep(pair: Pair) -> str | None:
    """Why a pair is dropped, or None to keep it."""
    if not MIN_ANSWER <= len(pair.answer) <= MAX_ANSWER:
        return "answer_length"
    domain = pair.origin.split("://", 1)[-1]
    if blocked({"domain": domain}, pair.question, pair.answer):
        return "blocked"
    return None


def stems(text: str) -> frozenset[str]:
    return frozenset(
        word[:STEM] for word in normalise(text).split() if len(word) >= 3 and word not in STOP_WORDS
    )


def shingles(text: str, n: int = 5) -> frozenset[str]:
    flat = normalise(text)
    return frozenset(flat[i : i + n] for i in range(max(1, len(flat) - n + 1)))


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _rng(text: str) -> random.Random:
    return random.Random(
        int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")
    )


class Pool:
    """One split's pairs, indexed for negative search."""

    def __init__(self, pairs: list[Pair]):
        self.pairs = pairs
        self.answer_stems = [stems(p.answer) for p in pairs]
        self.question_words = [frozenset(normalise(p.question).split()) for p in pairs]
        self.normal_answers = [normalise(p.answer) for p in pairs]
        self.by_site: dict[str, list[int]] = defaultdict(list)
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, pair in enumerate(pairs):
            self.by_site[pair.origin].append(i)
            for stem in self.answer_stems[i]:
                self.postings[stem].append(i)
        size = max(1, len(pairs))
        self.max_df = max(1, int(MAX_DF_SHARE * size))
        self.idf = {s: _idf(size, len(ids)) for s, ids in self.postings.items()}

    def acceptable(self, i: int, j: int, shingles_i: frozenset[str]) -> bool:
        """Whether pair j's answer can serve as a negative for pair i's question."""
        if i == j or (self.pairs[i].url and self.pairs[i].url == self.pairs[j].url):
            return False
        a, b = self.normal_answers[i], self.normal_answers[j]
        if a == b or a in b or b in a:
            return False
        if jaccard(self.question_words[i], self.question_words[j]) >= SAME_TEMPLATE:
            return False
        return jaccard(shingles_i, shingles(self.pairs[j].answer)) < NEAR_DUPLICATE

    def ranked(self, question_stems: frozenset[str], candidates: Iterable[int]) -> list[int]:
        """Candidates sharing at least one keyword, most IDF weight first, then by index."""
        scored = []
        for j in candidates:
            value = sum(self.idf.get(s, 0.0) for s in question_stems & self.answer_stems[j])
            if value > 0:
                scored.append((-value, j))
        return [j for _, j in sorted(scored)]

    def first_acceptable(self, i: int, ranked: Iterable[int], own: frozenset[str]) -> int | None:
        for checked, j in enumerate(ranked):
            if checked >= RANKED_CHECKS:
                return None
            if self.acceptable(i, j, own):
                return j
        return None

    def negative(self, i: int) -> tuple[int, str] | None:
        """The hardest acceptable negative for pair i, and which rule found it."""
        pair = self.pairs[i]
        rng = _rng(f"negative:{pair.qid}")
        q_stems = stems(pair.question)
        own = shingles(pair.answer)

        site = [j for j in self.by_site[pair.origin] if j != i]
        if len(site) > SITE_SCAN:
            site = rng.sample(site, SITE_SCAN)
        found = self.first_acceptable(i, self.ranked(q_stems, site), own)
        if found is not None:
            return found, "same_site"

        candidates: set[int] = set()
        for stem in sorted(q_stems):
            posting = self.postings.get(stem, [])
            if 0 < len(posting) <= self.max_df:
                candidates.update(posting[:POSTINGS_SCAN])
        others = [j for j in candidates if self.pairs[j].origin != pair.origin]
        found = self.first_acceptable(i, self.ranked(q_stems, others), own)
        if found is not None:
            return found, "keyword"

        rng.shuffle(site)
        found = self.first_acceptable(i, site, own)
        if found is not None:
            return found, "same_site_any"
        for _ in range(RANKED_CHECKS):
            j = rng.randrange(len(self.pairs))
            if self.pairs[j].origin != pair.origin and self.acceptable(i, j, own):
                return j, "random"
        return None


def _idf(size: int, df: int) -> float:
    return math.log(1 + size / df)


def sample(pairs: list[Pair], target: int, cap: int = SITE_CAP, seed: int = SEED) -> list[int]:
    """Indices of up to `target` pairs in a seeded order, at most `cap` per site."""
    order = sorted(
        range(len(pairs)),
        key=lambda i: hashlib.blake2b(f"{seed}:{pairs[i].qid}".encode(), digest_size=8).digest(),
    )
    per_site: Counter = Counter()
    chosen: list[int] = []
    for i in order:
        if len(chosen) >= target:
            break
        if per_site[pairs[i].origin] >= cap:
            continue
        per_site[pairs[i].origin] += 1
        chosen.append(i)
    return chosen


def state(question: str, answer: str) -> str:
    return f"Soru: {question}\nPasaj: {answer}"


def relevance_rows(
    pairs: list[Pair], target: int = TARGET_QUESTIONS, source: str = ""
) -> tuple[list[TrainingRow], Counter]:
    """Positive and negative rows per sampled question, split by a hash of the query id."""
    stats: Counter = Counter()
    kept: list[Pair] = []
    for pair in pairs:
        reason = keep(pair)
        if reason:
            stats[f"dropped_{reason}"] += 1
            continue
        kept.append(
            Pair(
                pair.qid,
                mask_personal_data(pair.question),
                mask_personal_data(pair.answer),
                pair.origin,
                pair.url,
            )
        )
    by_split: dict[str, list[Pair]] = defaultdict(list)
    for pair in kept:
        by_split[hash_split(f"webfaq-tur:{pair.qid}")].append(pair)
    rows: list[TrainingRow] = []
    for split in ("train", "validation"):
        pool = Pool(by_split[split])
        share = 0.9 if split == "train" else 0.1
        for i in sample(pool.pairs, round(target * share)):
            found = pool.negative(i)
            if found is None:
                stats["no_negative"] += 1
                continue
            j, kind = found
            stats[f"negative_{kind}"] += 1
            pair = pool.pairs[i]
            for answer, yes in ((pair.answer, 1.0), (pool.pairs[j].answer, 0.0)):
                rows.append(
                    TrainingRow(
                        track=TRACK,
                        task=TASK,
                        split=split,
                        origin="converted",
                        label_kind="rule",
                        source=source,
                        state=state(pair.question, answer),
                        question=QUESTION,
                        target={"true": yes, "false": 1.0 - yes},
                        recipe=RECIPE,
                    )
                )
        stats[f"pool_{split}"] = len(pool.pairs)
    return rows, stats


def iter_raw(paths: Iterable[Path]) -> Iterator[dict]:
    import pyarrow.parquet as pq

    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            columns=["origin", "url", "question", "answer"], batch_size=50_000
        ):
            yield from batch.to_pylist()


def build(out_root: Path = BUILT, root: Path = DOWNLOADS, target: int = TARGET_QUESTIONS) -> dict:
    hub_file(RETRIEVAL_REPO, RETRIEVAL_REVISION, "README.md", 200_000, root)
    hub_file(RAW_REPO, RAW_MAIN, "README.md", 200_000, root)
    local = {
        name: hub_file(RETRIEVAL_REPO, RETRIEVAL_REVISION, path, size, root)
        for name, (path, size) in RETRIEVAL_FILES.items()
    }
    raw = [hub_file(RAW_REPO, RAW_PARQUET, path, size, root) for path, size in RAW_FILES]
    pairs = train_pairs(
        read_jsonl(local["queries"]), read_jsonl(local["corpus"]), read_jsonl(local["qrels"])
    )
    joined, unjoined = join_sites(pairs, iter_raw(raw))
    source = f"{DOWNLOADS.as_posix()}/{RETRIEVAL_REPO.replace('/', '__')}/"
    rows, stats = relevance_rows(joined, target, source)
    rows = unique_rows(rows)
    out = out_root / TASK
    counts = write_rows(rows, out)
    meta = {
        "task": TASK,
        "track": TRACK,
        "label_kind": "rule",
        "files_sha256": {
            f"{RETRIEVAL_REPO}@{RETRIEVAL_REVISION}/{p}": sha256_of(local[n])
            for n, (p, _) in RETRIEVAL_FILES.items()
        }
        | {
            f"{RAW_REPO}@{RAW_PARQUET}/{p}": sha256_of(path)
            for (p, _), path in zip(RAW_FILES, raw, strict=True)
        },
        "train_pairs": len(pairs),
        "unjoined": unjoined,
        "stats": dict(stats),
        "counts": counts,
        "balance": {split: noul_balance(r for r in rows if r.split == split) for split in counts},
        "sites_joined": len({p.origin for p in joined}),
    }
    write_meta(out, meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WebFAQ Turkish pairs to relevance rows")
    parser.add_argument("--out", type=Path, default=BUILT)
    parser.add_argument("--questions", type=int, default=TARGET_QUESTIONS)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.out, target=args.questions), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
