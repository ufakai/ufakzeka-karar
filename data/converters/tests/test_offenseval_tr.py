"""The OffensEval-TR converter and the helpers in data/converters/common.py.

Every tweet here is invented for the test. No row of the real corpus is in
the repository, and the tests never reach the network: the zip is written
into the cache directory first and the download is a function that fails.
"""

from __future__ import annotations

import hashlib
import json
import random
import zipfile
from pathlib import Path

import pytest

from data.converters import common, offenseval_tr
from data.instrument_format import Row, check_dataset, read_split

# Row 102 repeats row 101, row 103 repeats a test row, row 104 has no text.
SPECIAL_TRAIN = [
    ("101", "bugün hava çok güzel", "NOT"),
    ("102", "bugün hava çok güzel", "NOT"),
    ("103", "maç berabere bitti", "NOT"),
    ("104", "   ", "NOT"),
    ("105", "@USER seni gidi yalancı", "OFF"),
    ("106", " URL adresinden kayıt olun ", "NOT"),
]

# Filler so that the tenth carved from train is more than one row per label:
# twenty NOT and ten OFF rows reach the carve.
FILLER_TRAIN = [(f"2{i:02d}", f"kütüphaneden {i} numaralı kitabı aldım", "NOT") for i in range(16)]
FILLER_TRAIN += [(f"3{i:02d}", f"@USER {i} kere söyledim, saçmalama", "OFF") for i in range(9)]

TRAIN = SPECIAL_TRAIN + FILLER_TRAIN
TEST = [
    ("501", "maç berabere bitti", "NOT"),
    ("502", "yarın sabah erken kalkacağım", "NOT"),
    ("503", "@USER boş konuşuyorsun, sus artık", "OFF"),
]


def make_archive(path: Path, train: list[tuple[str, str, str]], test: list[tuple[str, str, str]]):
    """Write a zip in the layout of offenseval2020-turkish.zip.

    The TSV files use no quoting and no escapes, and the gold labels are a
    separate comma separated file without a header, as in the source.
    """
    train_tsv = "\n".join(["id\ttweet\tsubtask_a", *(f"{i}\t{t}\t{y}" for i, t, y in train)]) + "\n"
    test_tsv = "\n".join(["id\ttweet", *(f"{i}\t{t}" for i, t, _ in test)]) + "\n"
    labels_csv = "\n".join(f"{i},{y}" for i, _, y in test) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(offenseval_tr.TRAIN_MEMBER, train_tsv)
        archive.writestr(offenseval_tr.TEST_MEMBER, test_tsv)
        archive.writestr(offenseval_tr.TEST_LABEL_MEMBER, labels_csv)
    return path


def refuse(url: str, dest: Path) -> None:
    raise AssertionError(f"the converter tried to download {url}")


def convert_from(tmp_path: Path, train=TRAIN, test=TEST, out_name: str = "out") -> Path:
    cache = tmp_path / "cache"
    make_archive(cache / offenseval_tr.ARCHIVE_NAME, train, test)
    return offenseval_tr.convert(tmp_path / out_name, cache, fetch=refuse)


def meta_of(folder: Path) -> dict:
    return json.loads((folder / "meta.json").read_text(encoding="utf-8"))


def rows_of(folder: Path) -> dict[str, list[Row]]:
    return {
        split: read_split(folder / f"{split}.jsonl") for split in ("train", "validation", "test")
    }


def test_the_converted_folder_passes_the_format_check(tmp_path):
    folder = convert_from(tmp_path)
    assert folder == tmp_path / "out" / "offenseval_tr"
    report = check_dataset(folder)
    assert report.ok, report.errors


def test_splits_labels_and_ids(tmp_path):
    folder = convert_from(tmp_path)
    meta = meta_of(folder)
    assert meta["labels"] == ["NOT", "OFF"]
    assert meta["license_id"] == "CC-BY-2.0"
    assert meta["source"] == offenseval_tr.SOURCE_URL
    # Thirty non-empty train rows: three are carved off, two are leaks.
    assert meta["splits"] == {"train": 25, "validation": 3, "test": 3}

    rows = rows_of(folder)
    ids = [row.id for split in rows.values() for row in split]
    assert len(ids) == len(set(ids))
    assert all(row.id.startswith("train-") for row in rows["train"])
    assert all(row.id.startswith("validation-") for row in rows["validation"])
    assert [row.id for row in rows["test"]] == ["test-501", "test-502", "test-503"]


def test_the_duplicate_the_leak_and_the_empty_row_are_dropped_and_counted(tmp_path):
    folder = convert_from(tmp_path)
    dropped = meta_of(folder)["dropped"]
    assert dropped["empty_text"] == 1
    leaks = ("duplicate_within_split", "validation_in_test", "train_in_eval")
    # One copy of the repeated tweet and one of the tweet that is also in
    # test. Which counter takes them depends on where the carve put them.
    assert sum(dropped[reason] for reason in leaks) == 2

    rows = rows_of(folder)
    everywhere = [row.text for split in rows.values() for row in split]
    assert everywhere.count("bugün hava çok güzel") == 1
    assert everywhere.count("maç berabere bitti") == 1
    assert [row.text for row in rows["test"]].count("maç berabere bitti") == 1
    assert "" not in everywhere


def test_only_outer_whitespace_is_stripped(tmp_path):
    folder = convert_from(tmp_path)
    texts = [row.text for split in rows_of(folder).values() for row in split]
    assert " URL adresinden kayıt olun " not in texts
    assert "URL adresinden kayıt olun" in texts
    # Mentions, links and case are left as the source wrote them.
    assert "@USER seni gidi yalancı" in texts


def test_the_validation_carve_is_stratified_and_repeatable(tmp_path):
    folder = convert_from(tmp_path)
    meta = meta_of(folder)
    assert meta["split_origin"]["validation"] == (
        "carved from train: 10 percent, stratified by label, random.Random(1)"
    )
    validation = read_split(folder / "validation.jsonl")
    # Twenty NOT and ten OFF rows reach the carve, so the tenth is 2 and 1.
    assert sorted(row.label for row in validation) == [0, 0, 1]

    again = convert_from(tmp_path, out_name="again")
    assert [row.id for row in read_split(again / "validation.jsonl")] == [
        row.id for row in validation
    ]


def test_the_revision_is_the_archive_hash(tmp_path):
    folder = convert_from(tmp_path)
    revision = meta_of(folder)["source_revision"]
    assert revision.startswith("sha256:")
    archive = tmp_path / "cache" / offenseval_tr.ARCHIVE_NAME
    assert revision == f"sha256:{common.sha256_file(archive)}"


def test_the_archive_is_fetched_once_and_then_reused(tmp_path):
    prepared = make_archive(tmp_path / "prepared.zip", TRAIN, TEST)
    calls: list[str] = []

    def fetch(url: str, dest: Path) -> None:
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(prepared.read_bytes())

    cache = tmp_path / "cache"
    first = offenseval_tr.convert(tmp_path / "one", cache, fetch=fetch)
    second = offenseval_tr.convert(tmp_path / "two", cache, fetch=fetch)
    assert calls == [offenseval_tr.SOURCE_URL]
    assert (first / "train.jsonl").read_bytes() == (second / "train.jsonl").read_bytes()


def test_a_test_row_without_a_gold_label_is_an_error(tmp_path):
    cache = tmp_path / "cache"
    path = make_archive(cache / offenseval_tr.ARCHIVE_NAME, TRAIN, TEST)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members[offenseval_tr.TEST_LABEL_MEMBER] = b"501,NOT\n502,NOT\n"
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    with pytest.raises(RuntimeError, match="no label for test ids"):
        offenseval_tr.convert(tmp_path / "out", cache, fetch=refuse)


def test_an_unexpected_header_is_an_error(tmp_path):
    cache = tmp_path / "cache"
    path = cache / offenseval_tr.ARCHIVE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(offenseval_tr.TRAIN_MEMBER, "id\ttext\tlabel\n1\tmerhaba\tNOT\n")
        archive.writestr(offenseval_tr.TEST_MEMBER, "id\ttweet\n2\tmerhaba dünya\n")
        archive.writestr(offenseval_tr.TEST_LABEL_MEMBER, "2,NOT\n")
    with pytest.raises(RuntimeError, match="expected the header"):
        offenseval_tr.convert(tmp_path / "out", cache, fetch=refuse)


# data/converters/common.py


def rows(labels: list[int]) -> list[Row]:
    return [Row(str(i), f"cümle {i}", None, label) for i, label in enumerate(labels)]


def test_stratified_carve_takes_the_fraction_of_every_label():
    source = rows([0] * 40 + [1] * 20 + [2] * 10)
    rest, carved = common.stratified_carve(source, 0.1, random.Random(1))
    assert len(carved) == 7
    assert sorted(row.label for row in carved) == [0] * 4 + [1] * 2 + [2]
    assert len(rest) == 63
    assert {row.id for row in rest}.isdisjoint({row.id for row in carved})
    # The rest keeps the source order, which the ids follow.
    assert [row.id for row in rest] == sorted((row.id for row in rest), key=int)


def test_stratified_carve_is_a_function_of_the_seed():
    source = rows([0] * 40 + [1] * 20)
    first = common.stratified_carve(source, 0.1, random.Random(1))[1]
    again = common.stratified_carve(source, 0.1, random.Random(1))[1]
    other = common.stratified_carve(source, 0.1, random.Random(2))[1]
    assert [row.id for row in first] == [row.id for row in again]
    assert [row.id for row in first] != [row.id for row in other]


def test_stratified_carve_keeps_rare_labels_on_both_sides():
    rest, carved = common.stratified_carve(rows([0] * 9 + [1, 1, 2]), 0.1, random.Random(1))
    # A label with two rows gives one up, a label with one row gives none.
    assert sorted(row.label for row in carved) == [0, 1]
    assert sorted(row.label for row in rest) == [0] * 8 + [1, 2]


def test_stratified_carve_rejects_a_fraction_outside_the_unit_interval():
    with pytest.raises(ValueError, match="between 0 and 1"):
        common.stratified_carve(rows([0, 0]), 1.0, random.Random(1))


def test_sha256_file_hashes_the_bytes(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(b"ufak")
    assert common.sha256_file(path) == hashlib.sha256(b"ufak").hexdigest()


def test_cached_archive_downloads_only_when_the_file_is_absent(tmp_path):
    dest = tmp_path / "sub" / "archive.bin"
    calls: list[str] = []

    def fetch(url: str, target: Path) -> None:
        calls.append(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"payload")

    assert common.cached_archive("https://example.invalid/a", dest, fetch) == dest
    common.cached_archive("https://example.invalid/a", dest, fetch)
    assert calls == ["https://example.invalid/a"]
    assert dest.read_bytes() == b"payload"


def test_write_dataset_fills_the_counts_and_refuses_a_broken_folder(tmp_path):
    meta = {
        "task": "demo",
        "source": "https://example.invalid/demo",
        "source_revision": "sha256:0",
        "license_id": "MIT",
        "labels": ["a", "b"],
        "split_origin": {"train": "source", "validation": "source", "test": "source"},
        "dropped": {},
        "notes": "",
    }
    splits = {
        "train": [Row("t1", "bir", None, 0), Row("t2", "iki", None, 1)],
        "validation": [Row("v1", "üç", None, 0)],
        "test": [Row("s1", "dört", None, 0), Row("s2", "beş", None, 1)],
    }
    folder = common.write_dataset(tmp_path / "out", "demo", splits, meta)
    assert json.loads((folder / "meta.json").read_text(encoding="utf-8"))["splits"] == {
        "train": 2,
        "validation": 1,
        "test": 2,
    }

    splits["train"].append(Row("t3", "dört", None, 0))  # a text that is also in test
    with pytest.raises(RuntimeError, match="both train and test"):
        common.write_dataset(tmp_path / "bad", "demo", splits, meta)
