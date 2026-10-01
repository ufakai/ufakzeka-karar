"""The evaluation test material no training row may overlap (rule 2), and its local cache.

Each entry names where the items come from and which fields make an item's
text. Hub entries are read from each dataset's parquet export through the
hub's file resolver (data/hub.py), not the dataset viewer's rows API, whose
edge throttled the fetch after seven sets. Every hub
repo, config and split below was checked on 2026-09-22 against the viewer's
/splits, /info and /is-valid endpoints without reading any rows; the commit
seen then is kept as `revision`. The parquet export follows the current commit,
not a pinned one, so `fetch` records the commit it read beside the cache.

Reading a set here is for decontamination only. Several are ShareAlike (TrGLUE,
Belebele, XQuAD) and none is training data under rule 1.

Cetvel's sets are the ones its task files load (github.com/KUIS-AI/cetvel,
tasks/, read 2026-09-22 and again 2026-09-23 for the remaining tasks), each
at the split Cetvel evaluates on: TQuAD and XQuAD have no test split and
Cetvel scores their validation split. TabiBench's are the test splits of the
28 sets in the boun-tabilab hub collection "tabibench" (arXiv 2512.23065,
github.com/boun-tabi-LMG/TabiBERT). The sets added on 2026-09-23 were checked
against the viewer's /info endpoint, or, for the four repos that still use a
loading script (reciTAL/mlsum, boun-tabi/nli_tr, GEM/wiki_lingua,
csebuetnlp/xlsum), against the file listing of their parquet export, which
the viewer built before it stopped running scripts.

The cache is one jsonl of {"id", "text"} per set under .cache/karar/eval_refs/,
outside the data tree: these are sets we must never train on, read only to
detect overlap, and some carry licences our data may not, so they are not
data of ours and have no manifest entry. The cache is not committed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from data import hub
from data.decontam.ngrams import SET_SEPARATOR, MinHashLSH, NgramIndex, minhash_signature

CACHE_DIR = Path(".cache/karar/eval_refs")
HUB_API = "https://huggingface.co/api/datasets"
VERIFIED = "2026-09-22"

Source = Literal["local", "hub", "github_squad"]


@dataclass(frozen=True)
class RefSpec:
    """One evaluation set to decontaminate against.

    `repo` is a hub repo id, a local path, or a pinned raw URL, by `source`.
    An item's text is its non-empty `fields` joined by newlines; its id is
    `id_field` when given, else its row index.
    """

    benchmark: str
    source: Source
    repo: str
    split: str
    fields: tuple[str, ...]
    config: str | None = None
    id_field: str | None = None
    revision: str | None = None


def _instrument(name: str, benchmark: str) -> RefSpec:
    return RefSpec(
        benchmark=benchmark,
        source="local",
        repo=f"data/raw/instrument/{name}/test.jsonl",
        split="test",
        fields=("text", "text_pair"),
        id_field="id",
    )


_TRGLUE = "turkish-nlp-suite/TrGLUE"
_TRGLUE_REVISION = "c3f11352f79c9cc9bea42a9ff4309097e4204fbc"


def _trglue(config: str, split: str, fields: tuple[str, ...]) -> RefSpec:
    return RefSpec(
        benchmark="TrGLUE",
        source="hub",
        repo=_TRGLUE,
        config=config,
        split=split,
        fields=fields,
        revision=_TRGLUE_REVISION,
    )


def _cetvel(
    repo: str, split: str, fields: tuple[str, ...], revision: str, config: str = "default"
) -> RefSpec:
    return RefSpec(
        benchmark="Cetvel",
        source="hub",
        repo=repo,
        config=config,
        split=split,
        fields=fields,
        revision=revision,
    )


def _tabibench(name: str, fields: tuple[str, ...], revision: str) -> RefSpec:
    return RefSpec(
        benchmark="TabiBench",
        source="hub",
        repo=f"boun-tabilab/{name}",
        config="default",
        split="test",
        fields=fields,
        revision=revision,
    )


_PLU = ("sent1", "sent2", "ending0", "ending1", "ending2", "ending3")
_PAIR = ("anchor", "positive")
_QUERY_DOC = ("query", "doc")


REFERENCE_SETS: dict[str, RefSpec] = {
    # The step 1 instrument's own test splits. OffensEval-TR's is also
    # Cetvel's offenseval_tr test, and TrCOLA is TrGLUE's CoLA task.
    "massive_tr_test": _instrument("massive_tr", "instrument"),
    "offenseval_tr_test": _instrument("offenseval_tr", "instrument, Cetvel"),
    "trcola_test": _instrument("trcola", "instrument, TrGLUE"),
    "mide22_test": _instrument("mide22", "instrument"),
    "legal_nli_tr_test": _instrument("legal_nli_tr", "instrument"),
    # Cetvel.
    "xnli_tr_test": RefSpec(
        benchmark="Cetvel",
        source="hub",
        repo="facebook/xnli",
        config="tr",
        split="test",
        fields=("premise", "hypothesis"),
        revision="b8dd5d7af51114dbda02c0e3f6133f332186418e",
    ),
    "xcopa_tr_test": RefSpec(
        benchmark="Cetvel",
        source="hub",
        repo="cambridgeltl/xcopa",
        config="tr",
        split="test",
        fields=("premise", "choice1", "choice2"),
        id_field="idx",
        revision="042f78955ba48e6404616762fa6e05e839c3907a",
    ),
    "belebele_tur_latn_test": RefSpec(
        benchmark="Cetvel",
        source="hub",
        repo="facebook/belebele",
        config="tur_Latn",
        split="test",
        fields=(
            "flores_passage",
            "question",
            "mc_answer1",
            "mc_answer2",
            "mc_answer3",
            "mc_answer4",
        ),
        revision="7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
    ),
    "xquad_tr_validation": RefSpec(
        benchmark="Cetvel",
        source="hub",
        repo="google/xquad",
        config="xquad.tr",
        split="validation",
        fields=("context", "question"),
        id_field="id",
        revision="51adfef1c1287aab1d2d91b5bead9bcfb9c68583",
    ),
    "stsb_tr_test": RefSpec(
        benchmark="Cetvel",
        source="hub",
        repo="emrecan/stsb-mt-turkish",
        config="default",
        split="test",
        fields=("sentence1", "sentence2"),
        revision="27084881aeef3ce11e905bbc486b2823fd728fa5",
    ),
    "xfact_tr_test": RefSpec(
        benchmark="Cetvel",
        source="hub",
        repo="mcemilg/x-fact",
        config="tr",
        split="test",
        fields=("claim",),
        revision="8944a7cbbf38bff86bd836cd1710b2907d266426",
    ),
    # Cetvel loads TQuAD through mcemilg/tquad, a loading script the viewer
    # cannot run; the script reads this file, pinned here to a commit.
    "tquad_dev": RefSpec(
        benchmark="Cetvel",
        source="github_squad",
        repo=(
            "https://raw.githubusercontent.com/TQuad/turkish-nlp-qa-dataset/"
            "3bece60f62569182476fe00683793745ba92c32e/dev-v0.1.json"
        ),
        split="validation",
        fields=("context", "question"),
        id_field="id",
    ),
    # TrGLUE: every task's test split.
    "trglue_cola_test": _trglue("cola", "test", ("sentence",)),
    "trglue_mnli_test_matched": _trglue("mnli", "test_matched", ("premise", "hypothesis")),
    "trglue_mnli_test_mismatched": _trglue("mnli", "test_mismatched", ("premise", "hypothesis")),
    "trglue_mrpc_test": _trglue("mrpc", "test", ("sentence1", "sentence2")),
    "trglue_qnli_test": _trglue("qnli", "test", ("question", "sentence")),
    "trglue_qqp_test": _trglue("qqp", "test", ("question1", "question2")),
    "trglue_rte_test": _trglue("rte", "test", ("sentence1", "sentence2")),
    "trglue_sst2_test": _trglue("sst2", "test", ("sentence",)),
    "trglue_stsb_test": _trglue("stsb", "test", ("sentence1", "sentence2")),
    # Cetvel's remaining tasks, at the split each task file evaluates on: a
    # task with no test split is scored on its validation split, and several
    # sets hold their whole evaluation data in a split named train.
    "bilmecebench_test": _cetvel(
        "abrek/bilmecebench-lm-evaluation-harness",
        "test",
        ("riddle", "choice_0", "choice_1", "choice_2", "choice_3"),
        "698cdcbe86a053dfd66244b8897e7aca31096f7f",
    ),
    "circumflex_tr_train": _cetvel(
        "abrek/circumflex_tr",
        "train",
        ("word", "definition1", "definition2"),
        "157d80367f0cae958c6dd0e452b77f91ca0df658",
    ),
    "exams_tr_validation": _cetvel(
        "mhardalov/exams",
        "validation",
        ("question.stem", "question.choices.text[]"),
        "4ff10804abb3341f8815cacd778181177bba7edd",
        config="crosslingual_tr",
    ),
    "gecturk_test": _cetvel(
        "mcemilg/GECTurk-generation",
        "test",
        ("source", "target"),
        "36f6a61aca96cafc4149fa823510a3fa81b98ee3",
    ),
    "ironytr_validation": _cetvel(
        "mcemilg/IronyTR", "validation", ("text",), "659d7da0591e373110b19bca538d02a52c20edd2"
    ),
    "mkqa_tr_train": _cetvel(
        "mcemilg/mkqa_tr", "train", ("query",), "eb77fe065345a53c1fa15afbd4d9e6eb714c11fd"
    ),
    "mlsum_tu_test": _cetvel(
        "reciTAL/mlsum",
        "test",
        ("title", "text", "summary"),
        "0f8114e036015f81e877e8a1950ce2713d7afe8d",
        config="tu",
    ),
    "news_cat_test": _cetvel(
        "mcemilg/news-cat", "test", ("text",), "394d21ad7df8d1303288a82817b91960ea4f5081"
    ),
    "mnli_tr_validation_matched": _cetvel(
        "boun-tabi/nli_tr",
        "validation_matched",
        ("premise", "hypothesis"),
        "15a8962926fc0c3c4dc357c9b49e4501dd723676",
        config="multinli_tr",
    ),
    "snli_tr_test": _cetvel(
        "boun-tabi/nli_tr",
        "test",
        ("premise", "hypothesis"),
        "15a8962926fc0c3c4dc357c9b49e4501dd723676",
        config="snli_tr",
    ),
    "tr_wikihow_summ_test": _cetvel(
        "ardauzunoglu/tr-wikihow-summ",
        "test",
        ("text", "summary"),
        "e18b665eeba6ad0833f85899f9ce81532311544d",
    ),
    "trclaim19_train": _cetvel(
        "mcemilg/TrClaim19", "train", ("tweet", "claim"), "7c16e96ae2b4f6760dcab954704d0fc2dbd1d043"
    ),
    "turkce_atasozleri_train": _cetvel(
        "abrek/turkce-atasozleri-lm-evaluation-harness",
        "train",
        ("question", "choice1", "choice2", "choice3", "choice4"),
        "1a40e83b6ae09dabdec6e7441582419b5c0832b8",
    ),
    "turkish_plu_goal_inference_test": _cetvel(
        "mcemilg/turkish-plu-goal-inference",
        "test",
        _PLU,
        "a1d486a4394887ae3165b2d5a498eb5fba0653ea",
    ),
    "turkish_plu_next_event_prediction_test": _cetvel(
        "mcemilg/turkish-plu-next-event-prediction",
        "test",
        _PLU,
        "0ebf43f4bf4d6c7e96e3fbe6974c8e4c40fba1ab",
    ),
    "turkish_plu_step_inference_test": _cetvel(
        "mcemilg/turkish-plu-step-inference",
        "test",
        _PLU,
        "3020a6573609c077aa698992054e8e3953475be2",
    ),
    "turkish_plu_step_ordering_test": _cetvel(
        "mcemilg/turkish-plu-step-ordering",
        "test",
        _PLU,
        "e17ebf7917c06b3d40a2446ad062929af76cb6cb",
    ),
    # Cetvel loads the nine subject configs; "All" holds the same 900 items.
    "turkishmmlu_test": _cetvel(
        "AYueksel/TurkishMMLU",
        "test",
        ("question", "choices[]"),
        "fcbe5bb19af2714186fdd5752671e0748efa0024",
        config="All",
    ),
    "wiki_lingua_tr_test": _cetvel(
        "GEM/wiki_lingua",
        "test",
        ("source", "target"),
        "af5d0f00b59a6933165c97b384f50d8b563c314d",
        config="tr",
    ),
    "wmt16_tr_en_validation": _cetvel(
        "wmt/wmt16",
        "validation",
        ("translation.tr", "translation.en"),
        "41d8a4013aa1489f28fea60ec0932af246086482",
        config="tr-en",
    ),
    "xlsum_tr_test": _cetvel(
        "csebuetnlp/xlsum",
        "test",
        ("title", "text", "summary"),
        "30fece425f9a3866e04321773ca7a80056d55ca6",
        config="turkish",
    ),
    # TabiBench (arXiv 2512.23065): the test split of each of its 28 sets,
    # as published in the boun-tabilab hub collection "tabibench".
    "tabibench_product_reviews_test": _tabibench(
        "Turkish-Product-Reviews", ("text",), "958eec15a6bc6f72b3bcbd51c97e485be1f378fb"
    ),
    "tabibench_news_cat_test": _tabibench(
        "News-Cat", ("text",), "b934304df00dc094eda2c7eb0cfd8e7289db64b5"
    ),
    "tabibench_bil_tweet_news_test": _tabibench(
        "BilTweetNews-Sentiment-Analysis", ("text",), "93b095dcf6313a6f7797730a2a5f02a76839c123"
    ),
    "tabibench_gender_hate_speech_test": _tabibench(
        "Gender-Hate-Speech-TR", ("text",), "74ba32d803bceec37803250dea55cc6d360cfdd2"
    ),
    "tabibench_wikiner_test": _tabibench(
        "WikiNER-TR", ("tokens[]",), "e3195350902f2eb4f6435cf9ad2cf8718911e584"
    ),
    "tabibench_wikiann_test": _tabibench(
        "WikiANN-TR", ("tokens[]",), "fefa86a5d9dbe09d6343fa150c869d3790b15fab"
    ),
    "tabibench_pos_ud_boun_test": _tabibench(
        "PosUD-BOUN", ("tokens[]",), "503d3847a553ba9d72ad15cbb3808cac9c85c2ff"
    ),
    "tabibench_pos_ud_imst_test": _tabibench(
        "PosUD-IMST", ("tokens[]",), "256bdb1fd92edac27640b7ad5d43811875790cac"
    ),
    "tabibench_snli_tr_test": _tabibench(
        "SNLI-TR", ("premise", "hypothesis"), "c39b1c52f91b8f85e6b98e1d5eff273d7aca1d56"
    ),
    "tabibench_multinli_tr_test": _tabibench(
        "MultiNLI-TR", ("premise", "hypothesis"), "3d709fc175df32c2351a0821c6cf222380f619f2"
    ),
    "tabibench_xquad_tr_test": _tabibench(
        "XQuAD-TR", ("context", "question"), "eee8993f5831ce35b5aa9da1db734906a3b17485"
    ),
    "tabibench_tquad2_test": _tabibench(
        "TQuad-2", ("context", "question"), "0c55923517a3af1e52fc4072bd0aaeafdaa2ae7a"
    ),
    "tabibench_stsb_tr_test": _tabibench(
        "STSb-TR", ("sentence1", "sentence2"), "491218bcd2509ed22f9861d68ee895d487fb6925"
    ),
    "tabibench_sick_tr_test": _tabibench(
        "SICK-TR", ("sentence1", "sentence2"), "8a11986a8825ce1e1e5a83294f428e55e462efd9"
    ),
    "tabibench_wmt16_tr_test": _tabibench(
        "Wmt16-TR", _PAIR, "efe64297ead320e43f7a0c0883d359cc0e5ca420"
    ),
    "tabibench_msmarco_tr_test": _tabibench(
        "MsMarco-TR", _PAIR, "adb63dbb73074a393ba6525faf6ce9c568cb8bca"
    ),
    "tabibench_scifact_tr_test": _tabibench(
        "Scifact-TR", _PAIR, "35c8222a40b1a9a66ff17cd0d253c77b3ede6635"
    ),
    "tabibench_nfcorpus_tr_test": _tabibench(
        "NFCorpus-TR", _PAIR, "3868549124bbe876e8062410cd2df4fc15c17403"
    ),
    "tabibench_quora_tr_test": _tabibench(
        "Quora-TR", _PAIR, "2d0986f379b630216c27079b604a7d2a93303d4e"
    ),
    "tabibench_fiqa_tr_test": _tabibench(
        "Fiqa-TR", _PAIR, "ae208e04e9dd31af806cd1af9be82eb07574e212"
    ),
    "tabibench_apps_tr_test": _tabibench(
        "Apps-Retrieval-TR", _QUERY_DOC, "8812d33165bda3ccb3a4b2b7184655cc4e7c2eb6"
    ),
    "tabibench_cosqa_tr_test": _tabibench(
        "Cos-QA-TR", _QUERY_DOC, "16b6692df5c957807a88c07c6732ecc870d5bdf3"
    ),
    "tabibench_code_search_net_tr_test": _tabibench(
        "Code-Search-Net-21K-TR", _QUERY_DOC, "acaad22aff048fd28a972bd32fe802877bcce81c"
    ),
    "tabibench_stackoverflow_qa_tr_test": _tabibench(
        "Stackoverflow-QA-TR", _QUERY_DOC, "489a750408e2486d415e014ce009e3c0b6420ade"
    ),
    "tabibench_med_nli_tr_test": _tabibench(
        "Med-NLI-TR", ("sentence1", "sentence2"), "49d0d83153cea9b3b49c2f3d106e0668696a2434"
    ),
    "tabibench_pubmed_rct_tr_test": _tabibench(
        "Pubmed-RCT-10K-TR", ("text",), "1e86befe13737391ba9c08e6a8779ae0142b4fba"
    ),
    "tabibench_sci_cite_tr_test": _tabibench(
        "Sci-Cite-TR", ("text",), "39377d62b527e700a6faf45f8945dcf7c0d38a9f"
    ),
    "tabibench_thesis_abstract_test": _tabibench(
        "Thesis-Abstract-Classification-11K", ("text",), "09674029d0a9dd34db6f27ac808842adf7d728d4"
    ),
}

for _name in REFERENCE_SETS:
    if SET_SEPARATOR in _name:
        raise ValueError(f"set name {_name!r} must not contain {SET_SEPARATOR!r}")


def field_value(row: dict[str, Any], field: str) -> Any:
    """A field's value; "a.b" reads key b inside a, and a trailing "[]" joins a list of strings.

    The list form is for token lists and answer choices, joined by spaces. A
    list read without "[]" is not text and is skipped, as is a missing key.
    """
    joined = field.endswith("[]")
    value: Any = row
    for key in field.removesuffix("[]").split("."):
        value = value.get(key) if isinstance(value, dict) else None
    if joined:
        if not isinstance(value, list):
            return None
        return " ".join(part.strip() for part in value if isinstance(part, str) and part.strip())
    return value


def text_from(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    """The item's text: its non-empty string fields, in order, one per line."""
    parts = [field_value(row, field) for field in fields]
    return "\n".join(part.strip() for part in parts if isinstance(part, str) and part.strip())


GetJson = Callable[[str], Any]
Download = Callable[[str, Path], None]


def _hub_rows(
    spec: RefSpec, cache_dir: Path, get_json: GetJson, download: Download
) -> Iterator[tuple[int, dict[str, Any]]]:
    import pyarrow.parquet as pq

    files = hub.parquet_files(
        spec.repo,
        spec.config or "default",
        spec.split,
        cache_dir / "parquet",
        fetch_json=get_json,
        fetch_file=download,
    )
    index = 0
    for path in files:
        for row in pq.read_table(path).to_pylist():
            yield index, row
            index += 1


def _squad_rows(document: dict[str, Any]) -> Iterator[tuple[int, dict[str, Any]]]:
    index = 0
    for article in document["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                yield index, {"id": qa.get("id"), "context": paragraph["context"], **qa}
                index += 1


def _local_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                yield index, json.loads(line)


def _items(spec: RefSpec, rows: Iterator[tuple[int, dict[str, Any]]]) -> Iterator[dict[str, str]]:
    seen: set[str] = set()
    for index, row in rows:
        item_id = str(row[spec.id_field]) if spec.id_field else ""
        if not item_id or item_id == "None":
            item_id = str(index)
        if item_id in seen:
            item_id = f"{item_id}#{index}"
        seen.add(item_id)
        text = text_from(row, spec.fields)
        if text:
            yield {"id": item_id, "text": text}


def cache_path(name: str, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"{name}.jsonl"


def fetch(
    name: str,
    cache_dir: Path = CACHE_DIR,
    *,
    get_json: GetJson = hub.get_json,
    download: Download = hub.download,
) -> Path:
    """Write one set's items to `<cache_dir>/<name>.jsonl` and a meta file beside it.

    Local sets are read from disk. Hub and GitHub sets are downloaded, which
    only the orchestrator runs; tests pass their own `get_json`.
    """
    spec = REFERENCE_SETS[name]
    revision: str | None = None
    if spec.source == "local":
        rows = _local_rows(Path(spec.repo))
    elif spec.source == "hub":
        revision = get_json(f"{HUB_API}/{spec.repo}").get("sha")
        rows = _hub_rows(spec, cache_dir, get_json, download)
    elif spec.source == "github_squad":
        rows = _squad_rows(get_json(spec.repo))
    else:
        raise ValueError(f"{name}: unknown source {spec.source!r}")
    target = cache_path(name, cache_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".jsonl.partial")
    count = 0
    with partial.open("w", encoding="utf-8") as handle:
        for item in _items(spec, rows):
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    partial.replace(target)
    meta = {
        "name": name,
        "benchmark": spec.benchmark,
        "source": spec.source,
        "repo": spec.repo,
        "config": spec.config,
        "split": spec.split,
        "fields": list(spec.fields),
        "items": count,
        "revision_verified": spec.revision,
        "revision_read": revision,
        "fetched": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    target.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


def missing(cache_dir: Path = CACHE_DIR) -> list[str]:
    """Registered sets with no cache file yet."""
    return [name for name in REFERENCE_SETS if not cache_path(name, cache_dir).is_file()]


def load_all(cache_dir: Path = CACHE_DIR) -> list[tuple[str, str]]:
    """(reference id, text) for every item of every set fetched so far."""
    items: list[tuple[str, str]] = []
    for name in REFERENCE_SETS:
        path = cache_path(name, cache_dir)
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    items.append((f"{name}{SET_SEPARATOR}{item['id']}", item["text"]))
    return items


def build_index(cache_dir: Path = CACHE_DIR) -> NgramIndex:
    return NgramIndex.build(load_all(cache_dir))


def build_lsh(cache_dir: Path = CACHE_DIR) -> MinHashLSH:
    """MinHash LSH over every fetched item long enough to have a shingle."""
    lsh = MinHashLSH()
    for ref_id, text in load_all(cache_dir):
        signature = minhash_signature(text)
        if signature is not None:
            lsh.add(ref_id, signature)
    return lsh
