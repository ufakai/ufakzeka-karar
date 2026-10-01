"""The decision model on public Turkish test sets it never trained on.

    python -m bench.external build --set mmlu_pro_tr
    python -m bench.external run --set mmlu_pro_tr --weights models/r3-base-s2/model.pt \
        --calibrator results/step7/r3-base-s2/calibrator.json \
        --out results/step9/external/mmlu_pro_tr-r3-base-s2.json
    python -m bench.external traces --set turkish_mmlu --traces PARQUET --out OUT.json

`build` reads one set at a pinned revision, turns every test row into one typed
question, checks the texts against every file under data/built with the rule of
bench/hakembench/common.overlapping (whole, covered, near-duplicate), drops any
item a rule touches, and writes the items to .cache/karar/external/<set>/,
outside the data tree and never committed: these are other people's test sets,
read to be measured on, as data/decontam/references.py reads them. The record
(repo, revision, licence as shown, split, rows, phrasing, overlap result, ids
only) goes to results/step9/external/<set>.build.json.

`run` answers the items the way the shipped model answers (bench/adapters/karar.py:
one pass per question, at most ten options a pass, the calibrator's temperature
per question type, the abstain map), in padded batches sorted by length so a
CPU container is used well; the logits are the ones the adapter would compute
(a test holds the two within float tolerance). It writes the rows (ids, gold,
probabilities; no text) beside --out, and the summary to --out.

The numbers: accuracy, macro F1, Brier (sum over options, 0 to 2) and top-label
smooth ECE (bench/metrics.py conventions), each with a 2,000-draw bootstrap
interval over items; the random baseline, the mean of 1/k; accuracy per
category with a Wilson interval.

`traces` scores another model's published per-question traces on exactly the
questions `build` kept, with the same metrics, when its question ids match.

Sets (SETS below): MMLU-Pro-TR (MIT) and the owner's Turkish MMLU (evaluation
only); the official test splits of the three instrument sets the model
trained on only through their train splits, with the training wording; and the
Cetvel classification tasks whose licence is on the allow-list of
data/licenses.py (CETVEL_TASKS lists every one, with the reason for each left
out).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from bench.harness.items import Item, load_items
from schema.questions import HEAD_OPTIONS_PER_PASS, ChoiceQuestion, NoulQuestion, Question

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(".cache/karar/external")
RESULTS = Path("results/step9/external")
BUILT = Path("data/built")
INSTRUMENT = Path("data/raw/instrument")
QUESTION_ID = "q"
QUESTION = TypeAdapter(Question)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DRAWS = 2000

# The plain Turkish of a multiple-choice exam: "which is the right answer?". The
# state already holds the question; this asks for the option, nothing more.
MC_INSTRUCTIONS = "Doğru cevap hangisi?"


# The sets ------------------------------------------------------------------------------


@dataclass(frozen=True)
class SetSpec:
    """One external test set, as recorded in its build file."""

    name: str
    title: str
    repo: str
    revision: str
    url: str
    licence_as_shown: str
    # The allow-list id (data/licenses.py), or None for the owner's evaluation-only set.
    licence_id: str | None
    split: str
    phrasing: str
    cetvel_task: str | None = None
    notes: tuple[str, ...] = ()
    gated: bool = False


SETS: dict[str, SetSpec] = {
    "mmlu_pro_tr": SetSpec(
        name="mmlu_pro_tr",
        title="MMLU-Pro-TR",
        repo="bezir/MMLU-pro-TR",
        revision="e9ca2d4d1bb18bb8307b7f3e0fb55830cba30dc2",
        url="https://huggingface.co/datasets/bezir/MMLU-pro-TR",
        licence_as_shown="license: mit (card metadata at the pinned revision); upstream "
        "TIGER-Lab/MMLU-Pro, license: mit",
        licence_id="MIT",
        split="test",
        phrasing="choice: state = the question text; criteria = one key per option, its letter "
        f'and text ("A) ..."), no description; instructions = "{MC_INSTRUCTIONS}"',
        notes=(
            "The card says the Turkish text was machine-translated from MMLU-Pro with human "
            "oversight; it is not native Turkish.",
            "Options: 3 to 10 per question, so every question is one pass.",
        ),
    ),  # fmt: skip
    "turkish_mmlu": SetSpec(
        name="turkish_mmlu",
        title="Turkish MMLU (alibayram)",
        repo="alibayram/turkish_mmlu",
        revision="30c94d45e29424a074dc754910fb44019284cc91",
        url="https://huggingface.co/datasets/alibayram/turkish_mmlu",
        licence_as_shown="license: cc-by-nc-nd-4.0 (card metadata); the card's own licence text "
        "reads CC BY-NC 4.0 (as the owner read it, 2026-09-27)",
        licence_id=None,
        split="mmlu",
        phrasing="choice: state = the question text (soru); criteria = one key per option "
        f'(secenekler), "A) ..."; instructions = "{MC_INSTRUCTIONS}"',
        notes=(
            "Not on the allow-list of data/licenses.py. Added by the owner for evaluation "
            "only: scores are published for non-commercial research only, no item is "
            "ever republished, the items stay outside the repo, and the rows hold ids and "
            "numbers only. Cite Zenodo DOI 10.5281/zenodo.13378019.",
            "The mmlu split is the 6,200 questions of the Turkish MMLU leaderboard.",
        ),
        gated=True,
    ),  # fmt: skip
    "offenseval_tr": SetSpec(
        name="offenseval_tr",
        title="OffensEval-TR 2020, subtask A",
        repo="coltekin/offenseval2020_tr",
        revision="sha256:7977e96255dbc9b8d14893f1b14cbe3dec53c70358503c062c5a59720ec9c2f2",
        url="https://coltekin.github.io/offensive-turkish/offenseval2020-turkish.zip",
        licence_as_shown="cc-by-2.0",
        licence_id="CC-BY-2.0",
        split="test",
        phrasing="noul, exactly the training question (data/typed/instrument.py)",
        cetvel_task="offenseval_tr",
        notes=(
            "Test split as converted by data/converters/offenseval_tr.py into "
            "data/raw/instrument/offenseval_tr/test.jsonl (the archive's test file with its "
            "gold labels, joined on the id).",
        ),
    ),  # fmt: skip
    "massive_tr": SetSpec(
        name="massive_tr",
        title="MASSIVE 1.1, tr-TR intents",
        repo="AmazonScience/massive",
        revision="sha256:4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577",
        url="https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz",
        licence_as_shown="cc-by-4.0",
        licence_id="CC-BY-4.0",
        split="test",
        phrasing="choice over all 59 intents with the training wording and option names "
        "(data/typed/instrument.py), answered in passes of ten; training rows carried "
        "the right intent and nine drawn distractors",
        notes=("Test split as converted by data/converters/massive_tr.py (partition test).",),
    ),  # fmt: skip
    "mide22": SetSpec(
        name="mide22",
        title="MiDe22 Turkish tweets",
        repo="ogozcelik/turkish-fake-news-detection",
        revision="6b6bb45712d746e28906913e953d5525f5c8c633",
        url="https://huggingface.co/datasets/ogozcelik/turkish-fake-news-detection",
        licence_as_shown="mit",
        licence_id="MIT",
        split="test",
        phrasing="choice of three, exactly the training question (data/typed/instrument.py)",
        notes=(
            "The source has one split; the test fifth was carved by data/converters/mide22.py "
            "(70/10/20, stratified, seed 1) before any training.",
        ),
    ),  # fmt: skip
    "xcopa_tr": SetSpec(
        name="xcopa_tr",
        title="XCOPA, Turkish",
        repo="cambridgeltl/xcopa",
        revision="042f78955ba48e6404616762fa6e05e839c3907a",
        url="https://huggingface.co/datasets/cambridgeltl/xcopa",
        licence_as_shown='license: cc-by-4.0 (card metadata); "Creative Commons Attribution 4.0 '
        'International (CC BY 4.0)" (card, Licensing Information)',
        licence_id="CC-BY-4.0",
        split="test",
        phrasing='choice of two: state = the premise; instructions = "Bunun sebebi hangisi?" '
        '(question cause) or "Bunun sonucu hangisi?" (effect); criteria = the two '
        "alternative sentences as keys",
        cetvel_task="xcopa_tr",
    ),  # fmt: skip
    "xfact_tr": SetSpec(
        name="xfact_tr",
        title="X-FACT, Turkish test claims",
        repo="utahnlp/x-fact (GitHub)",
        revision="c58b1ea78753977528485469428ff356a4b837e4",
        url="https://github.com/utahnlp/x-fact",
        licence_as_shown="MIT License, Copyright (c) 2021 Utah NLP (LICENSE at the pinned commit)",
        licence_id="MIT",
        split="test",
        phrasing="choice of Cetvel's four verdicts in plain Turkish: state = the claim; "
        "instructions = the FACTurk training question (data/typed/claims.py); gold by "
        "Cetvel's own map (true, false, complicated, anything else partly true)",
        cetvel_task="xfact_tr",
        notes=(
            "Cetvel loads the mirror mcemilg/x-fact, which carries no licence tag; the rows "
            "are read from the authors' own repository, whose LICENSE is MIT.",
            "All Turkish claims come from dogrulukpayi.com, one of the FACTurk sources the "
            "model trained on (verdict task); FACTurk was decontaminated against X-FACT, "
            "and the overlap check here is run again.",
        ),
    ),  # fmt: skip
}


# Cetvel's classification and multiple-choice tasks (github.com/KUIS-AI/cetvel, tasks/,
# at 6119c517e06cce23aaeac103ce3504c9380655e3, read 2026-09-27), the repo each loads, the
# licence that repo shows (hub card metadata and body at the revision in
# data/decontam/references.py, read 2026-09-27), and the verdict. Generation tasks
# (summaries, translation, question answering, grammar correction) are not typed
# decisions and are not listed.
CETVEL_COMMIT = "6119c517e06cce23aaeac103ce3504c9380655e3"
CETVEL_TASKS: dict[str, dict[str, str]] = {
    "belebele_tr": {"repo": "facebook/belebele", "licence": "cc-by-sa-4.0",
                    "verdict": "out: ShareAlike"},
    "bilmecebench": {"repo": "abrek/bilmecebench-lm-evaluation-harness", "licence": "none shown",
                     "verdict": "out: no licence"},
    "circumflex_tr": {"repo": "abrek/circumflex_tr", "licence": "none shown",
                      "verdict": "out: no licence"},
    "exams_tr": {"repo": "mhardalov/exams", "licence": "cc-by-sa-4.0",
                 "verdict": "out: ShareAlike"},
    "ironytr": {"repo": "mcemilg/IronyTR", "licence": "none shown (upstream teghub/IronyTR has "
                "no licence file)", "verdict": "out: no licence"},
    "news_cat": {"repo": "mcemilg/news-cat", "licence": "none shown",
                 "verdict": "out: no licence"},
    "mnli_tr": {"repo": "boun-tabi/nli_tr (multinli_tr)", "licence": "cc-by-3.0, cc-by-4.0, "
                "cc-by-sa-3.0, mit, other", "verdict": "out: ShareAlike and 'other' listed"},
    "snli_tr": {"repo": "boun-tabi/nli_tr (snli_tr)", "licence": "cc-by-3.0, cc-by-4.0, "
                "cc-by-sa-3.0, mit, other", "verdict": "out: ShareAlike and 'other' listed"},
    "xnli_tr": {"repo": "facebook/xnli", "licence": "none shown (card: 'More Information "
                "Needed')", "verdict": "out: no licence shown"},
    "offenseval_tr": {"repo": "offenseval2020_tr", "licence": "cc-by-2.0",
                      "verdict": "in, as the set offenseval_tr"},
    "sts_tr": {"repo": "emrecan/stsb-mt-turkish", "licence": "none shown",
               "verdict": "out: no licence"},
    "check_worthiness": {"repo": "mcemilg/TrClaim19", "licence": "none shown",
                         "verdict": "out: no licence"},
    "relevance_judgment": {"repo": "mcemilg/TrClaim19", "licence": "none shown",
                           "verdict": "out: no licence"},
    "turkce_atasozleri": {"repo": "abrek/turkce-atasozleri-lm-evaluation-harness",
                          "licence": "gpl-3.0", "verdict": "out: GPL is not on the allow-list"},
    "turkish_plu_goal_inference": {"repo": "mcemilg/turkish-plu-goal-inference",
                                   "licence": "none shown (upstream GGLAB-KU/turkish-plu has no "
                                   "licence file)", "verdict": "out: no licence"},
    "turkish_plu_next_event_prediction": {"repo": "mcemilg/turkish-plu-next-event-prediction",
                                          "licence": "none shown", "verdict": "out: no licence"},
    "turkish_plu_step_inference": {"repo": "mcemilg/turkish-plu-step-inference",
                                   "licence": "none shown", "verdict": "out: no licence"},
    "turkish_plu_step_ordering": {"repo": "mcemilg/turkish-plu-step-ordering",
                                  "licence": "none shown", "verdict": "out: no licence"},
    "turkishmmlu": {"repo": "AYueksel/TurkishMMLU", "licence": "none shown (upstream "
                    "ArdaYueksel/TurkishMMLU has no licence file)", "verdict": "out: no licence"},
    "xcopa_tr": {"repo": "cambridgeltl/xcopa (tr)", "licence": "cc-by-4.0",
                 "verdict": "in, as the set xcopa_tr"},
    "xfact_tr": {"repo": "mcemilg/x-fact (tr), a mirror of utahnlp/x-fact",
                 "licence": "mirror: none shown; upstream: MIT",
                 "verdict": "in, as the set xfact_tr, read from the upstream MIT repository"},
}  # fmt: skip


# Mapping rows to typed questions (no network) ------------------------------------------


@dataclass
class Built:
    """One test row as an item, with its category and the text the overlap check reads."""

    item: Item
    group: str
    text: str


def lettered(options: list[str]) -> dict[str, None]:
    """Option keys "A) text", in the source's order; the letter keeps equal texts apart."""
    if not 2 <= len(options) <= len(LETTERS):
        raise ValueError(f"{len(options)} options cannot be lettered")
    return {f"{LETTERS[i]}) {str(text).strip()}": None for i, text in enumerate(options)}


def choice_item(set_name: str, row_id: str, question_text: str, options: list[str],
                answer_index: int, group: str) -> Built:  # fmt: skip
    """A multiple-choice exam question as one typed choice question."""
    if not 0 <= answer_index < len(options):
        raise ValueError(f"{row_id}: answer {answer_index} outside {len(options)} options")
    criteria = lettered(options)
    question = {"type": "choice", "instructions": MC_INSTRUCTIONS, "criteria": criteria}
    gold = list(criteria)[answer_index]
    text = question_text.strip()
    item = Item(id=f"{set_name}:{row_id}", track=set_name, state=text,
                questions={QUESTION_ID: question}, gold={QUESTION_ID: gold})  # fmt: skip
    return Built(item, group, "\n".join([text, *(str(o).strip() for o in options)]))


def parse_options(raw: Any) -> list[str]:
    """MMLU-Pro-TR stores the options as the text of a Python list."""
    if isinstance(raw, list):
        return [str(o) for o in raw]
    value = ast.literal_eval(raw)
    if not isinstance(value, list):
        raise ValueError("options are not a list")
    return [str(o) for o in value]


def mmlu_pro_item(row: dict) -> Built:
    try:
        options = parse_options(row["options"])
    except (ValueError, SyntaxError) as error:
        raise ValueError(f"{row['question_id']}: options unreadable ({error})") from error
    index = int(row["answer_index"])
    if LETTERS[index] != str(row["answer"]).strip():
        raise ValueError(f"{row['question_id']}: answer letter and index disagree")
    return choice_item("mmlu_pro_tr", str(row["question_id"]), row["question"], options, index,
                       str(row["category"]))  # fmt: skip


def turkish_mmlu_item(row: dict, index: int) -> Built:
    """One row of the mmlu split; it has no id column, so the id is its row number."""
    group = str(row.get("konu") or row.get("bolum") or "none")
    return choice_item("turkish_mmlu", str(index), row["soru"], list(row["secenekler"]),
                       int(row["cevap"]), group)  # fmt: skip


def instrument_item(dataset: str, row: dict, labels: list[str]) -> Built:
    """A test row with exactly the question its train rows were trained with."""
    from data.typed.instrument import _question

    question, keys = _question(dataset, labels)
    gold_key = keys[int(row["label"])]
    gold: str | bool = gold_key == "true" if question["type"] == "noul" else gold_key
    if row.get("text_pair") is not None:
        raise ValueError(f"{dataset}: a text pair was not expected")
    group = labels[int(row["label"])]
    item = Item(id=f"{dataset}:{row['id']}", track=dataset, state=row["text"],
                questions={QUESTION_ID: question}, gold={QUESTION_ID: gold})  # fmt: skip
    return Built(item, group, row["text"])


XCOPA_ASKS = {"cause": "Bunun sebebi hangisi?", "effect": "Bunun sonucu hangisi?"}


def xcopa_item(row: dict) -> Built:
    choices = [row["choice1"].strip(), row["choice2"].strip()]
    if choices[0] == choices[1]:
        raise ValueError(f"{row['idx']}: the two alternatives are the same text")
    question = {"type": "choice", "instructions": XCOPA_ASKS[row["question"]],
                "criteria": dict.fromkeys(choices)}  # fmt: skip
    item = Item(id=f"xcopa_tr:{row['idx']}", track="xcopa_tr", state=row["premise"].strip(),
                questions={QUESTION_ID: question},
                gold={QUESTION_ID: choices[int(row["label"])]})  # fmt: skip
    return Built(item, row["question"], "\n".join([row["premise"], *choices]))


# Cetvel's four X-FACT choices (tasks/xfact/tr.yaml) in plain Turkish, and its label map
# (tasks/xfact/utils.py): true, false, complicated, and every other label as partly true.
XFACT_OPTIONS = {
    "doğru": "İddia doğru.",
    "yanlış": "İddia yanlış.",
    "karmaşık ya da sınıflandırması zor": "İddia doğru ya da yanlış diye sınıflandırılamayacak "
                                          "kadar karmaşık.",
    "kısmen doğru ya da yanıltıcı": "İddianın bir kısmı doğru ya da iddia yanıltıcı.",
}  # fmt: skip


def xfact_gold(label: str) -> str:
    label = label.strip()
    if label == "true":
        return "doğru"
    if label == "false":
        return "yanlış"
    if label == "complicated/hard to categorise":
        return "karmaşık ya da sınıflandırması zor"
    return "kısmen doğru ya da yanıltıcı"


def xfact_item(row: dict, index: int) -> Built:
    from data.typed.claims import INSTRUCTIONS

    question = {"type": "choice", "instructions": INSTRUCTIONS, "criteria": dict(XFACT_OPTIONS)}
    claim = row["claim"].strip()
    item = Item(id=f"xfact_tr:{index}", track="xfact_tr", state=claim,
                questions={QUESTION_ID: question},
                gold={QUESTION_ID: xfact_gold(row["label"])})  # fmt: skip
    return Built(item, row["label"].strip(), claim)


# Reading the sources (network, except the instrument sets) -----------------------------


def _hub_file(repo: str, filename: str, revision: str) -> Path:
    """A file at a pinned revision, in the user's own hub cache; a login is used if present."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def _parquet_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


# A reader yields one thunk per source row, so a row that cannot be mapped is refused
# and counted in build() without ending the read.
Thunk = Callable[[], Built]


def read_mmlu_pro(spec: SetSpec) -> Iterator[Thunk]:
    path = _hub_file(spec.repo, "data/test-00000-of-00001.parquet", spec.revision)
    for row in _parquet_rows(path):
        yield lambda row=row: mmlu_pro_item(row)


def read_turkish_mmlu(spec: SetSpec) -> Iterator[Thunk]:
    path = _hub_file(spec.repo, "data/mmlu-00000-of-00001.parquet", spec.revision)
    for index, row in enumerate(_parquet_rows(path)):
        yield lambda row=row, index=index: turkish_mmlu_item(row, index)


def read_instrument(dataset: str) -> Callable[[SetSpec], Iterator[Thunk]]:
    def read(spec: SetSpec) -> Iterator[Thunk]:
        folder = REPO_ROOT / INSTRUMENT / dataset
        labels = json.loads((folder / "meta.json").read_text(encoding="utf-8"))["labels"]
        with (folder / "test.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    yield lambda row=row: instrument_item(dataset, row, labels)

    return read


def read_xcopa(spec: SetSpec) -> Iterator[Thunk]:
    path = _hub_file(spec.repo, "tr/test-00000-of-00001.parquet", spec.revision)
    for row in _parquet_rows(path):
        yield lambda row=row: xcopa_item(row)


XFACT_FILE = "data/x-fact/test.all.tsv"


def xfact_rows(text: str) -> list[dict]:
    """The Turkish rows of the upstream test file.

    The file is tab-separated with no quoting; a row whose field count is off
    (a tab inside a text) is dropped and counted by the caller, never repaired.
    """
    lines = text.splitlines()
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) != len(header):
            continue
        row = dict(zip(header, parts, strict=True))
        if row["language"] == "tr":
            rows.append(row)
    return rows


def read_xfact(spec: SetSpec) -> Iterator[Thunk]:
    import urllib.request

    url = f"https://raw.githubusercontent.com/utahnlp/x-fact/{spec.revision}/{XFACT_FILE}"
    with urllib.request.urlopen(url, timeout=120) as response:
        text = response.read().decode("utf-8")
    for index, row in enumerate(xfact_rows(text)):
        yield lambda row=row, index=index: xfact_item(row, index)


READERS: dict[str, Callable[[SetSpec], Iterator[Thunk]]] = {
    "mmlu_pro_tr": read_mmlu_pro,
    "turkish_mmlu": read_turkish_mmlu,
    "offenseval_tr": read_instrument("offenseval_tr"),
    "massive_tr": read_instrument("massive_tr"),
    "mide22": read_instrument("mide22"),
    "xcopa_tr": read_xcopa,
    "xfact_tr": read_xfact,
}


# Build: items, overlap, record ----------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def overlap_check(built: list[Built], built_root: Path) -> dict[str, Any]:
    """bench/hakembench/common.overlapping of every item's text against every built file."""
    from bench.hakembench.common import overlapping, training_files, training_texts

    files = training_files(built_root)
    found = overlapping({b.item.id: b.text for b in built}, training_texts(files, built_root))
    dropped = sorted(key for key, note in found.items() if note["dropped"])
    by_rule = Counter(rule for key in dropped for rule in found[key]["rules"])
    return {
        "rule": "bench/hakembench/common.overlapping: dropped on whole (normalised equality), "
                "covered (over half the item's tokens in 8-grams the training texts share) or "
                "near_duplicate (MinHash Jaccard 0.8 on 5-token shingles); row_ngram recorded only",
        "training_side": f"every jsonl under {relative(built_root).as_posix()} (train, validation, "
                         "held-out and r4 files), texts of the state field",
        "files": {p.relative_to(built_root).as_posix(): sha256_file(p) for p in files},
        "items_checked": len(built),
        "items_touched": len(found),
        "items_dropped": len(dropped),
        "dropped_by_rule": dict(sorted(by_rule.items())),
        "dropped": {key: {"rules": found[key]["rules"], "coverage": found[key]["coverage"],
                          "rows": found[key]["rows"]} for key in dropped},
        "touched_not_dropped": {key: {"rules": note["rules"], "coverage": note["coverage"]}
                                for key, note in sorted(found.items()) if not note["dropped"]},
    }  # fmt: skip


def cache_dir(name: str, root: Path = CACHE) -> Path:
    return REPO_ROOT / root / name


def build(name: str, built_root: Path = BUILT, cache: Path = CACHE,
          results: Path = RESULTS) -> dict[str, Any]:  # fmt: skip
    """Read, map, check and write one set. Returns the record also written to results/."""
    spec = SETS[name]
    started = time.perf_counter()
    rows: list[Built] = []
    refused: list[dict[str, str]] = []
    for make in READERS[name](spec):
        try:
            rows.append(make())
        except ValueError as error:  # a row that cannot be a typed question is counted, not fixed
            refused.append({"error": str(error)})
    seen: set[str] = set()
    for b in rows:
        if b.item.id in seen:
            raise ValueError(f"{name}: duplicate item id {b.item.id}")
        seen.add(b.item.id)
    overlap = overlap_check(rows, REPO_ROOT / built_root)
    kept = [b for b in rows if b.item.id not in overlap["dropped"]]
    out = cache_dir(name, cache)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "items.jsonl").open("w", encoding="utf-8") as handle:
        for b in kept:
            handle.write(b.item.model_dump_json(exclude_none=True) + "\n")
    groups = {b.item.id: b.group for b in kept}
    (out / "groups.json").write_text(json.dumps(groups, ensure_ascii=False), encoding="utf-8")
    options = Counter(len(b.item.questions[QUESTION_ID].criteria or {}) if
                      isinstance(b.item.questions[QUESTION_ID], ChoiceQuestion) else 2
                      for b in kept)  # fmt: skip
    record = {
        "set": asdict(spec),
        "cetvel": (
            {"task": spec.cetvel_task, "commit": CETVEL_COMMIT} if spec.cetvel_task else None
        ),  # fmt: skip
        "rows_read": len(rows) + len(refused),
        "rows_refused": refused,
        "items_built": len(rows),
        "items_kept": len(kept),
        "options_per_question": dict(sorted(options.items())),
        "groups": dict(sorted(Counter(groups.values()).items())),
        "overlap": overlap,
        "items_file": f"{(cache / name / 'items.jsonl').as_posix()} (not committed)",
        "items_sha256": sha256_file(out / "items.jsonl"),
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.perf_counter() - started, 1),
    }
    target = REPO_ROOT / results / f"{name}.build.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def load_built(name: str, cache: Path = CACHE) -> tuple[list[Item], dict[str, str]]:
    folder = cache_dir(name, cache)
    groups = json.loads((folder / "groups.json").read_text(encoding="utf-8"))
    return load_items(folder / "items.jsonl"), groups


# Answering: the adapter's numbers, in batches -------------------------------------------


@dataclass
class Job:
    """One forward pass: a question, or one pass of ten options of a larger choice."""

    index: int  # the question's position in the run
    part: int  # the pass number within the question
    packed: Any
    tokens: int = 0


def passes(question: Question, encode: Callable[[str], list[int]]) -> list[Question]:
    """The passes model/head/infer.chunked_logits makes, in the same order."""
    from model.head.infer import _sub_question

    if not isinstance(question, ChoiceQuestion) or len(question.criteria) <= HEAD_OPTIONS_PER_PASS:
        return [question]
    keys = sorted(question.options, key=lambda key: (encode("\n" + key), key))
    groups = [
        keys[i : i + HEAD_OPTIONS_PER_PASS] for i in range(0, len(keys), HEAD_OPTIONS_PER_PASS)
    ]
    if len(groups) > 1 and len(groups[-1]) == 1:
        groups[-1] = [groups[-2][-1], *groups[-1]]
    return [_sub_question(question, keys_in_pass) for keys_in_pass in groups]


def batches(jobs: list[Job], max_tokens: int, max_rows: int, seed: int = 1) -> list[list[Job]]:
    """Jobs sorted by length into padded batches of at most max_tokens, in a shuffled order.

    The order is shuffled so time spent so far predicts the time left.
    """
    ordered = sorted(jobs, key=lambda job: job.tokens)
    out: list[list[Job]] = []
    current: list[Job] = []
    for job in ordered:
        width = max([job.tokens, *(j.tokens for j in current)])
        if current and (width * (len(current) + 1) > max_tokens or len(current) >= max_rows):
            out.append(current)
            current = []
        current.append(job)
    if current:
        out.append(current)
    random.Random(seed).shuffle(out)
    return out


def logits_for(question: Question, parts: list[dict[str, float]]) -> list[float]:
    """The adapter's logits in the answer's key order, from the passes' outputs."""
    if isinstance(question, ChoiceQuestion):
        merged: dict[str, float] = {}
        for by_key in parts:  # a borrowed option keeps its first pass's logit, as chunked_logits
            for key, value in by_key.items():
                merged.setdefault(key, value)
        return [merged[key] for key in question.options]
    (by_key,) = parts
    if isinstance(question, NoulQuestion):
        return [by_key["true"], by_key["false"]]
    return [by_key[str(k)] for k in range(len(question.criteria))]


@dataclass
class Progress:
    done_tokens: int = 0
    total_tokens: int = 0
    started: float = field(default_factory=time.perf_counter)

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def projected_seconds(self) -> float | None:
        if not self.done_tokens:
            return None
        return self.elapsed() * self.total_tokens / self.done_tokens


@dataclass
class Prepared:
    """One set's questions packed into passes and planned into batches."""

    items: list[Item]
    questions: list[Question]
    refused: list[dict[str, str]]
    plan: list[list[Job]]

    @property
    def tokens(self) -> int:
        return sum(max(j.tokens for j in b) * len(b) for b in self.plan)


def prepare(items: list[Item], encode: Callable[[str], list[int]], *, layout: str = "blind",
            max_tokens: int = 16384, max_rows: int = 64) -> Prepared:  # fmt: skip
    """Pack every question's passes; an item packing refuses is recorded, never guessed."""
    from model.head.pack import pack

    questions: list[Question] = []
    jobs: list[Job] = []
    refused: list[dict[str, str]] = []
    kept: list[Item] = []
    for item in items:
        question = item.questions[QUESTION_ID]
        try:
            packed = [pack(item.state, sub, encode, layout=layout)
                      for sub in passes(question, encode)]  # fmt: skip
        except ValueError as error:
            refused.append({"id": item.id, "error": str(error)})
            continue
        index = len(questions)
        questions.append(question)
        kept.append(item)
        jobs += [Job(index, part, p, len(p.ids)) for part, p in enumerate(packed)]
    return Prepared(kept, questions, refused, batches(jobs, max_tokens, max_rows))


def execute(model: Any, prepared: Prepared, progress: Progress,
            on_batch: Callable[[Progress], None] | None = None
            ) -> dict[int, dict[int, dict[str, float]]]:  # fmt: skip
    """Every planned batch through the network; the logits by question and pass."""
    import torch

    from model.head.pack import collate

    outputs: dict[int, dict[int, dict[str, float]]] = defaultdict(dict)
    model.eval()
    with torch.inference_mode():
        for batch_jobs in prepared.plan:
            batch = model.to_device(collate([j.packed for j in batch_jobs], model.pad_id), "cpu")
            out, _ = model(batch)
            values = out.float().tolist()
            for row, job in enumerate(batch_jobs):
                keys = batch.keys[row]
                outputs[job.index][job.part] = dict(zip(keys, values[row][: len(keys)],
                                                        strict=True))  # fmt: skip
            progress.done_tokens += max(j.tokens for j in batch_jobs) * len(batch_jobs)
            if on_batch is not None:
                on_batch(progress)
    return outputs


def finish(prepared: Prepared, outputs: dict[int, dict[int, dict[str, float]]],
           calibration: Any) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:  # fmt: skip
    """The served answer of each question, through the adapter's own code."""
    from bench.adapters.karar import KararAdapter

    # Only _answer is used: the temperature per type and the abstain map, as served.
    adapter = KararAdapter(model=object(), encode=lambda text: [], calibration=calibration,
                           revision="batch")  # fmt: skip
    rows, refused = [], list(prepared.refused)
    for index, (item, question) in enumerate(zip(prepared.items, prepared.questions,
                                                 strict=True)):  # fmt: skip
        parts = [outputs[index][part] for part in sorted(outputs[index])]
        logits = logits_for(question, parts)
        if not all(math.isfinite(v) for v in logits):
            refused.append({"id": item.id, "error": "non-finite logit"})
            continue
        answer, diagnostics = adapter._answer(question, logits, calibration)
        rows.append(row_of(item, question, answer, diagnostics, logits))
    return rows, refused


def answer_items(model: Any, encode: Callable[[str], list[int]], calibration: Any,
                 items: list[Item], *, max_tokens: int = 16384, max_rows: int = 64,
                 on_batch: Callable[[Progress], None] | None = None
                 ) -> tuple[list[dict[str, Any]], list[dict[str, str]], Progress]:  # fmt: skip
    """Every item's one question answered as the karar adapter answers it.

    Returns the rows (ids, gold, served and raw probabilities, the expected
    error), the items refused (never guessed), and the progress.
    """
    prepared = prepare(items, encode, layout=getattr(model, "layout", "blind"),
                       max_tokens=max_tokens, max_rows=max_rows)  # fmt: skip
    progress = Progress(total_tokens=prepared.tokens)
    outputs = execute(model, prepared, progress, on_batch)
    rows, refused = finish(prepared, outputs, calibration)
    return rows, refused, progress


def row_of(item: Item, question: Question, answer: dict, diagnostics: dict,
           logits: list[float]) -> dict[str, Any]:  # fmt: skip
    """One results row: ids and numbers, no text of the item."""
    gold = item.gold[QUESTION_ID]
    if isinstance(question, NoulQuestion):
        outcomes = ["true", "false"]
        probabilities = [answer["noul"], 1.0 - answer["noul"]]
        gold_index = 0 if gold is True else 1
    else:
        outcomes = list(answer["probabilities"])
        probabilities = [answer["probabilities"][k] for k in outcomes]
        gold_index = outcomes.index(str(gold))
    predicted = max(range(len(outcomes)), key=lambda k: probabilities[k])
    return {
        "id": item.id,
        "type": question.type,
        "k": len(outcomes),
        "gold": gold_index,
        "predicted": predicted,
        "correct": predicted == gold_index,
        "probabilities": probabilities,
        "raw_probabilities": diagnostics["raw_probabilities"],
        "logits": logits,
        "expected_error": diagnostics["expected_error"],
        "temperature": diagnostics["temperature"],
    }


# Scoring --------------------------------------------------------------------------------


def wilson(correct: int, n: int, z: float = 1.959964) -> list[float] | None:
    if n == 0:
        return None
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def scored(rows: list[dict[str, Any]], outcome_names: Callable[[dict], list[str]] | None = None):
    """The rows as bench.metrics.Scored; one item per row."""
    from bench.metrics import Question as MQ
    from bench.metrics import Scored

    questions = []
    for row in rows:
        names = outcome_names(row) if outcome_names else [str(k) for k in range(row["k"])]
        questions.append(MQ(item_id=row["id"], type=row["type"], outcomes=tuple(names),
                            probabilities=tuple(row["probabilities"]), gold=row["gold"],
                            confidence=1.0 - row.get("expected_error", 0.0)))  # fmt: skip
    return Scored.build(questions)


def statistic(with_f1: bool) -> Callable:
    from bench.metrics import accuracy as acc
    from bench.metrics import block_macro_f1, brier, top_label_smooth_ece

    def compute(s) -> dict[str, float]:
        out = {"accuracy": acc(s), "brier": brier(s), "smooth_ece": top_label_smooth_ece(s)}
        if with_f1:
            out["macro_f1"] = block_macro_f1(s)
        return out

    return compute


def summarise(rows: list[dict[str, Any]], groups: dict[str, str], *, with_f1: bool,
              draws: int = DRAWS) -> dict[str, Any]:  # fmt: skip
    """Metrics with bootstrap intervals, the random baseline, accuracy per category."""
    from bench.metrics import evaluate

    # Outcome names by position: macro F1 then counts classes per position, which is right
    # for the fixed-label sets; the lettered exam sets report no macro F1.
    report, _ = evaluate(scored(rows), statistic(with_f1), draws=draws, seed=0)
    per_group: dict[str, dict[str, Any]] = {}
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        entry = tally[groups.get(row["id"], "none")]
        entry[0] += int(row["correct"])
        entry[1] += 1
    for name, (right, n) in sorted(tally.items()):
        per_group[name] = {"n": n, "accuracy": right / n, "wilson_95": wilson(right, n)}
    return {
        "questions": report["questions"],
        "metrics": report["metrics"],
        "random_baseline_accuracy": sum(1.0 / row["k"] for row in rows) / len(rows),
        "per_group_accuracy": per_group,
        "definitions": {
            "accuracy": "share of questions whose most probable option is the gold one",
            "macro_f1": "unweighted mean F1 over the gold and predicted classes (bench/metrics.py)",
            "brier": "sum over options of the squared error of the served probabilities, 0 to 2",
            "smooth_ece": "top-label smooth ECE of the served maximum probability (relplot, "
                          "arXiv 2309.12236)",
            "intervals": f"2.5 and 97.5 percentiles of {draws} bootstrap draws over items, seed 0",
            "random_baseline_accuracy": "mean over questions of 1/k, k the number of options",
            "per_group_accuracy": "accuracy per category with a 95 percent Wilson interval",
        },
    }  # fmt: skip


def with_f1(name: str) -> bool:
    """Macro F1 needs fixed classes; lettered exam options and XCOPA's sentences are not."""
    return name not in ("mmlu_pro_tr", "turkish_mmlu", "xcopa_tr")


# Run records -----------------------------------------------------------------------------


def code_state() -> dict[str, Any]:
    """HEAD, whether the tree is clean outside results/, and the hashes of this code."""

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()  # fmt: skip
        except (OSError, subprocess.CalledProcessError):
            return ""

    status = git("status", "--porcelain", "--", ".", ":(exclude)results")
    files = ["bench/external.py", "bench/modal_external.py", "bench/adapters/karar.py",
             "model/head/pack.py", "model/head/infer.py", "model/head/head.py"]  # fmt: skip
    return {"git_head": git("rev-parse", "HEAD") or None, "tree_clean": not status,
            "code_sha256": {f: sha256_file(REPO_ROOT / f) for f in files
                            if (REPO_ROOT / f).exists()}}  # fmt: skip


def write_run(out: Path, record: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    """The summary at `out`, the rows at <out stem>.rows.jsonl; neither is overwritten."""
    rows_path = out.with_suffix(".rows.jsonl")
    for path in (out, rows_path):
        if path.exists():
            raise FileExistsError(f"{path} exists; results are never overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    record = {**record, "rows_file": rows_path.name, "rows_sha256": sha256_file(rows_path)}
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows_path


def run_record(name: str, rows: list[dict], refused: list[dict], groups: dict[str, str],
               model_info: dict[str, Any], runtime: dict[str, Any], *,
               draws: int = DRAWS, limit: int = 0) -> dict[str, Any]:  # fmt: skip
    build_path = REPO_ROOT / RESULTS / f"{name}.build.json"
    build_record = json.loads(build_path.read_text(encoding="utf-8")) if build_path.exists() else {}
    return {
        "set": name,
        "dataset": build_record.get("set"),
        "items_kept_after_overlap": build_record.get("items_kept"),
        "overlap_dropped": build_record.get("overlap", {}).get("items_dropped"),
        "build_file": build_path.relative_to(REPO_ROOT).as_posix(),
        "limit": limit or None,
        "model": model_info,
        "runtime": runtime,
        "unanswered": refused,
        "results": summarise(rows, groups, with_f1=with_f1(name), draws=draws),
        "code": code_state(),
    }


def relative(path: Path) -> Path:
    """A path under the repo as the repo names it, so no local home folder is recorded."""
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def model_info(weights: Path, calibrator: Path, run: str | None = None) -> dict[str, Any]:
    data = json.loads(calibrator.read_text(encoding="utf-8"))
    weights, calibrator = relative(weights), relative(calibrator)
    return {"run": run, "weights": weights.as_posix(), "weights_sha256": sha256_file(weights),
            "calibrator": calibrator.as_posix(), "calibrator_sha256": sha256_file(calibrator),
            "calibrator_fitted_on": data.get("weights_sha256"),
            "temperatures": data.get("temperatures"),
            "path": "bench/adapters/karar.py answer, fp32 torch, CPU, batched passes"}  # fmt: skip


def run_local(name: str, weights: Path, calibrator: Path, out: Path, *, limit: int = 0,
              threads: int = 0, max_tokens: int = 16384, draws: int = DRAWS,
              sample_seed: int = 1) -> dict[str, Any]:  # fmt: skip
    import torch

    from bench.adapters.karar import Calibration, load_model, sha256

    if threads:
        torch.set_num_threads(threads)
    items, groups = load_built(name)
    if limit:
        items = random.Random(sample_seed).sample(items, min(limit, len(items)))
    calibration = Calibration.from_file(calibrator, weights_sha256=sha256(weights))
    started = time.perf_counter()
    model, encode = load_model(weights)
    load_seconds = time.perf_counter() - started
    rows, refused, progress = answer_items(model, encode, calibration, items,
                                           max_tokens=max_tokens)  # fmt: skip
    runtime = {"host": f"{platform.system()} {platform.machine()} {platform.processor()}",
               "threads": torch.get_num_threads(), "load_seconds": round(load_seconds, 1),
               "answer_seconds": round(progress.elapsed(), 1),
               "padded_tokens": progress.total_tokens, "questions": len(rows),
               "seconds_per_question": round(progress.elapsed() / max(1, len(rows)), 4),
               "sample": f"{limit} items drawn with random.Random({sample_seed})" if limit
               else "every kept item"}  # fmt: skip
    record = run_record(name, rows, refused, groups, model_info(weights, calibrator), runtime,
                        draws=draws, limit=limit)  # fmt: skip
    write_run(out, record, rows)
    return record


# Another model's traces on the same questions -------------------------------------------


def score_traces(name: str, rows: list[dict[str, Any]], source: dict[str, Any],
                 out: Path, draws: int = DRAWS) -> dict[str, Any]:  # fmt: skip
    """Rows already mapped to our item ids (gold index, probabilities, k) scored as ours are."""
    items, groups = load_built(name)
    kept = {item.id for item in items}
    missing = sorted(kept - {row["id"] for row in rows})
    rows = [row for row in rows if row["id"] in kept]
    record = {"set": name, "source": source, "questions_ours": len(kept),
              "questions_matched": len(rows), "questions_missing": len(missing),
              "missing_ids": missing[:50],
              "results": summarise(rows, groups, with_f1=with_f1(name), draws=draws),
              "code": code_state()}  # fmt: skip
    write_run(out, record, rows)
    return record


# Command line ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.external")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("build", help="read, map, overlap-check and cache one set")
    p.add_argument("--set", required=True, choices=sorted(SETS))
    p = commands.add_parser("run", help="answer one built set with the decision model, locally")
    p.add_argument("--set", required=True, choices=sorted(SETS))
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--calibrator", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0, help="a seeded sample of N items (smoke)")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--draws", type=int, default=DRAWS)
    p = commands.add_parser("cetvel", help="print Cetvel's classification tasks and verdicts")
    args = parser.parse_args(argv)
    if args.command == "build":
        record = build(args.set)
        print(json.dumps({k: record[k] for k in ("items_built", "items_kept", "rows_read")}
                         | {"dropped": record["overlap"]["items_dropped"]}))  # fmt: skip
        return 0
    if args.command == "cetvel":
        print(json.dumps(CETVEL_TASKS, ensure_ascii=False, indent=2))
        return 0
    record = run_local(args.set, args.weights, args.calibrator, args.out, limit=args.limit,
                       threads=args.threads, draws=args.draws)  # fmt: skip
    print(json.dumps({"runtime": record["runtime"],
                      "metrics": record["results"]["metrics"]}, indent=2))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
