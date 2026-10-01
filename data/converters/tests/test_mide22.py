"""MiDe22 conversion, on invented tweets in the source schema. No network."""

import csv
import json
from collections import Counter

import pytest

from data.converters import mide22
from data.instrument_format import check_dataset, read_split

# Short texts written for this test, ten per class, in the source's two
# columns. None of them is a real tweet.
STEMS = {
    "False": "Bu iddia için hiçbir kaynak gösterilmedi, sayı {}",
    "Other": "Konuyla ilgili görüşümü paylaşıyorum, sayı {}",
    "True": "Açıklama resmi kurumun sitesinde yayımlandı, sayı {}",
}


def source_rows():
    rows = [[STEMS[label].format(number), label] for label in STEMS for number in range(10)]
    # A repeat of the first row, an empty line, a label that is not one of the three,
    # and a text with a line break inside it.
    rows.append([STEMS["False"].format(0), "False"])
    rows.append(["", ""])
    rows.append(["Etiketi belirsiz bir gönderi", "Unknown"])
    rows.append(["İlk satır\nikinci satır", "Other"])
    return rows


@pytest.fixture
def fetch(tmp_path):
    path = tmp_path / "source" / mide22.SOURCE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["tweet", "label"])
        writer.writerows(source_rows())

    def local_fetch(repo_id, filename, revision, cache_dir):
        assert repo_id == mide22.REPO
        assert revision == mide22.REVISION
        assert filename == mide22.SOURCE_FILE
        return path

    return local_fetch


def convert(tmp_path, fetch, name="out"):
    return mide22.convert(tmp_path / name, tmp_path / "cache", fetch=fetch)


def labels_of(folder, split):
    return Counter(row.label for row in read_split(folder / f"{split}.jsonl"))


def test_the_written_folder_passes_the_format_check(tmp_path, fetch):
    assert check_dataset(convert(tmp_path, fetch)).ok


def test_the_carve_is_seventy_ten_twenty_and_stratified(tmp_path, fetch):
    folder = convert(tmp_path, fetch)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    # 31 kept texts: ten per class, plus the one with a line break, labelled Other.
    # Other's eleventh row goes to train, which holds the largest remainder.
    assert meta["splits"] == {"train": 22, "validation": 3, "test": 6}
    assert labels_of(folder, "train") == {0: 7, 1: 8, 2: 7}
    assert labels_of(folder, "validation") == {0: 1, 1: 1, 2: 1}
    assert labels_of(folder, "test") == {0: 2, 1: 2, 2: 2}
    assert meta["labels"] == ["False", "Other", "True"]


def test_the_duplicate_the_empty_text_and_the_unknown_label_are_dropped_and_counted(
    tmp_path, fetch
):
    folder = convert(tmp_path, fetch)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["dropped"] == {
        "empty_text": 1,
        "unknown_label": 1,
        "duplicate_text": 1,
        "duplicate_within_split": 0,
        "validation_in_test": 0,
        "train_in_eval": 0,
    }
    texts = [
        row.text
        for split in ("train", "validation", "test")
        for row in read_split(folder / f"{split}.jsonl")
    ]
    assert len(texts) == len(set(texts))
    assert "Etiketi belirsiz bir gönderi" not in texts
    assert "İlk satır\nikinci satır" in texts


def test_meta_records_the_pinned_revision_and_how_the_splits_were_made(tmp_path, fetch):
    folder = convert(tmp_path, fetch)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["name"] == "mide22"
    assert meta["source_revision"] == "6b6bb45712d746e28906913e953d5525f5c8c633"
    assert meta["source"] == "https://huggingface.co/datasets/ogozcelik/turkish-fake-news-detection"
    assert meta["license_id"] == "MIT"
    for origin in meta["split_origin"].values():
        assert "stratified by label" in origin and "random.Random(1)" in origin
    assert "sha256" in meta["notes"]


def test_ids_are_unique_and_the_carve_repeats_exactly(tmp_path, fetch):
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
