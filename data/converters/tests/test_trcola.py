"""TrCOLA conversion, on invented rows in the source schema. No network."""

import json

import pytest

from data.converters import trcola
from data.instrument_format import check_dataset, read_split

# Sentences written for this test. The source file has one JSON object per
# line, and the rows whose var_type is None carry no `id` field.
TRAIN = [
    {
        "orig": "Kedi bahçede uyuyor.",
        "variation": "Kedi bahçede uyuyor.",
        "var_type": "None",
        "label": 1,
    },
    {
        "id": "s1",
        "orig": "Kedi bahçede uyuyor.",
        "variation": "Kedi bahçede uyuyorlar.",
        "var_type": "Morphological Violation",
        "label": 0,
    },
    # A repeat of the first row.
    {
        "orig": "Kedi bahçede uyuyor.",
        "variation": "Kedi bahçede uyuyor.",
        "var_type": "None",
        "label": 1,
    },
    # Also in test, so it must leave train.
    {
        "orig": "Yarın sabah erken kalkacağım.",
        "variation": "Yarın sabah erken kalkacağım.",
        "var_type": "None",
        "label": 1,
    },
    {
        "id": "s2",
        "orig": "Boş satır.",
        "variation": "   ",
        "var_type": "Semantic Violation",
        "label": 0,
    },
    {
        "id": "s3",
        "orig": "Çocuklar parkta oynuyor.",
        "variation": "Çocuklar parkta oynadı.",
        "var_type": "x",
        "label": 7,
    },
    {
        "id": "s4",
        "orig": "Masanın üstünde kitaplar duruyor.",
        "variation": "Masanın üstünde kitaplar duruyorlar.",
        "var_type": "Syntactic Violation",
        "label": 0,
    },
    {
        "orig": "Deniz bugün çok sakin.",
        "variation": "Deniz bugün çok sakin.",
        "var_type": "None",
        "label": 1,
    },
]
VALIDATION = [
    {
        "orig": "Otobüs durağında bekliyoruz.",
        "variation": "Otobüs durağında bekliyoruz.",
        "var_type": "None",
        "label": 1,
    },
    {
        "id": "s5",
        "orig": "Kitabı masaya bıraktım.",
        "variation": "Kitabı masaya bıraktılar bıraktım.",
        "var_type": "Syntactic Violation",
        "label": 0,
    },
]
TEST = [
    {
        "orig": "Yarın sabah erken kalkacağım.",
        "variation": "Yarın sabah erken kalkacağım.",
        "var_type": "None",
        "label": 1,
    },
    {
        "orig": "Otobüs durağında bekliyoruz.",
        "variation": "Otobüs durağında bekliyoruz.",
        "var_type": "None",
        "label": 1,
    },
    {
        "id": "s6",
        "orig": "Ali okula gitti.",
        "variation": "Ali okula gitti gitmedi.",
        "var_type": "Semantic Violation",
        "label": 0,
    },
]


@pytest.fixture
def fetch(tmp_path):
    source = tmp_path / "source"
    for split, rows in (("train", TRAIN), ("validation", VALIDATION), ("test", TEST)):
        path = source / trcola.FILES[split]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )

    def local_fetch(repo_id, filename, revision, cache_dir):
        assert repo_id == trcola.REPO
        assert revision == trcola.REVISION
        return source / filename

    return local_fetch


def convert(tmp_path, fetch, name="out"):
    return trcola.convert(tmp_path / name, tmp_path / "cache", fetch=fetch)


def test_the_written_folder_passes_the_format_check(tmp_path, fetch):
    folder = convert(tmp_path, fetch)
    assert check_dataset(folder).ok


def test_duplicates_leaks_empty_text_and_unknown_labels_are_dropped_and_counted(tmp_path, fetch):
    folder = convert(tmp_path, fetch)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["splits"] == {"train": 4, "validation": 1, "test": 3}
    assert meta["dropped"] == {
        "empty_text": 1,
        "unknown_label": 1,
        "duplicate_within_split": 1,
        "validation_in_test": 1,
        "train_in_eval": 1,
    }
    texts = {
        split: [row.text for row in read_split(folder / f"{split}.jsonl")]
        for split in meta["splits"]
    }
    assert "Yarın sabah erken kalkacağım." not in texts["train"]
    assert texts["validation"] == ["Kitabı masaya bıraktılar bıraktım."]
    assert texts["train"].count("Kedi bahçede uyuyor.") == 1


def test_the_judged_sentence_is_the_variation_and_one_means_acceptable(tmp_path, fetch):
    folder = convert(tmp_path, fetch)
    rows = read_split(folder / "train.jsonl")
    by_text = {row.text: row.label for row in rows}
    assert by_text["Kedi bahçede uyuyor."] == 1
    assert by_text["Kedi bahçede uyuyorlar."] == 0
    assert "Masanın üstünde kitaplar duruyor." not in by_text
    assert all(row.text_pair is None for row in rows)


def test_meta_records_the_pinned_revision_the_labels_and_the_source_splits(tmp_path, fetch):
    folder = convert(tmp_path, fetch)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["name"] == "trcola"
    assert meta["source_revision"] == "84dc7a3641c0d6997b4970f0d4d8e26811904980"
    assert meta["source"] == "https://huggingface.co/datasets/turkish-nlp-suite/TrCOLA"
    assert meta["license_id"] == "CC-BY-4.0"
    assert meta["labels"] == ["unacceptable", "acceptable"]
    assert set(meta["split_origin"].values()) == {"source"}
    assert "sha256" in meta["notes"]


def test_ids_are_unique_and_the_run_repeats_exactly(tmp_path, fetch):
    first = convert(tmp_path, fetch, "one")
    second = convert(tmp_path, fetch, "two")
    ids = [
        row.id
        for split in ("train", "validation", "test")
        for row in read_split(first / f"{split}.jsonl")
    ]
    assert len(ids) == len(set(ids))
    for split in ("train", "validation", "test"):
        name = f"{split}.jsonl"
        assert (first / name).read_text(encoding="utf-8") == (second / name).read_text(
            encoding="utf-8"
        )
    assert (first / "meta.json").read_text(encoding="utf-8") == (second / "meta.json").read_text(
        encoding="utf-8"
    )
