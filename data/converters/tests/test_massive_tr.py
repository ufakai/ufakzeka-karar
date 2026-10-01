"""The MASSIVE tr-TR converter, against a hand-made archive in the source's layout.

Every row here is invented for the test. No row of the real corpus is in the
repository, and the tests never reach the network: the archive is written
into the cache directory first and the download is a function that fails.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from data.converters import massive_tr
from data.instrument_format import check_dataset, read_split


def record(source_id: str, partition: str, intent: str, utt: str, locale: str = "tr-TR") -> dict:
    """One line of 1.1/data/tr-TR.jsonl, with the fields the source writes."""
    return {
        "id": source_id,
        "locale": locale,
        "partition": partition,
        "scenario": intent.split("_")[0],
        "intent": intent,
        "utt": utt,
        "annot_utt": utt,
        "worker_id": "1",
        "slot_method": [],
        "judgments": [],
    }


# alarm_set, news_query and weather_query appear in train and in test.
# Row 2 repeats row 1, row 3 repeats a test row, row 6 has no text.
RECORDS = [
    record("1", "train", "alarm_set", "sabah yedide beni uyandır"),
    record("2", "train", "alarm_set", "sabah yedide beni uyandır"),
    record("3", "train", "news_query", "bugünkü haberleri oku"),
    record("4", "train", "weather_query", "yarın hava nasıl olacak"),
    record("5", "train", "news_query", "spor haberlerini göster"),
    record("6", "train", "weather_query", "   "),
    record("7", "train", "alarm_set", "  akşam altıda alarm kur  "),
    record("8", "dev", "news_query", "ekonomi haberlerini aç"),
    record("9", "dev", "alarm_set", "alarmı iptal et"),
    record("10", "test", "news_query", "bugünkü haberleri oku"),
    record("11", "test", "alarm_set", "beni öğlen uyandır"),
    record("12", "test", "weather_query", "hafta sonu yağmur var mı"),
]

LABELS = ["alarm_set", "news_query", "weather_query"]


def make_archive(path: Path, records: list[dict]) -> Path:
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in records).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(massive_tr.MEMBER)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return path


def refuse(url: str, dest: Path) -> None:
    raise AssertionError(f"the converter tried to download {url}")


def convert_from(tmp_path: Path, records: list[dict], out_name: str = "out") -> Path:
    cache = tmp_path / "cache"
    make_archive(cache / massive_tr.ARCHIVE_NAME, records)
    return massive_tr.convert(tmp_path / out_name, cache, fetch=refuse)


def meta_of(folder: Path) -> dict:
    return json.loads((folder / "meta.json").read_text(encoding="utf-8"))


def test_the_converted_folder_passes_the_format_check(tmp_path):
    folder = convert_from(tmp_path, RECORDS)
    assert folder == tmp_path / "out" / "massive_tr"
    report = check_dataset(folder)
    assert report.ok, report.errors


def test_splits_labels_and_ids(tmp_path):
    folder = convert_from(tmp_path, RECORDS)
    meta = meta_of(folder)
    assert meta["labels"] == LABELS
    assert meta["name"] == "massive_tr"
    assert meta["license_id"] == "CC-BY-4.0"
    assert meta["source"] == massive_tr.SOURCE_URL
    # Six train rows survive the empty one, two of them are dropped below.
    assert meta["splits"] == {"train": 4, "validation": 2, "test": 3}
    assert meta["split_origin"]["validation"] == "source partition dev"

    rows = {split: read_split(folder / f"{split}.jsonl") for split in meta["splits"]}
    ids = [row.id for split in rows.values() for row in split]
    assert len(ids) == len(set(ids))
    assert [row.id for row in rows["validation"]] == ["validation-8", "validation-9"]
    assert [row.id for row in rows["test"]] == ["test-10", "test-11", "test-12"]


def test_the_duplicate_the_leak_and_the_empty_row_are_dropped_and_counted(tmp_path):
    folder = convert_from(tmp_path, RECORDS)
    assert meta_of(folder)["dropped"] == {
        "other_locale": 0,
        "unknown_partition": 0,
        "empty_text": 1,
        "label_missing_from_a_split": 0,
        "duplicate_within_split": 1,
        "validation_in_test": 0,
        "train_in_eval": 1,
    }
    train = read_split(folder / "train.jsonl")
    assert [row.id for row in train] == ["train-1", "train-4", "train-5", "train-7"]
    # The leaked text stays in test, and only there.
    texts = [
        row.text
        for split in ("train", "validation")
        for row in read_split(folder / f"{split}.jsonl")
    ]
    assert "bugünkü haberleri oku" not in texts


def test_only_outer_whitespace_is_stripped(tmp_path):
    folder = convert_from(tmp_path, RECORDS)
    rows = {row.id: row.text for row in read_split(folder / "train.jsonl")}
    assert rows["train-7"] == "akşam altıda alarm kur"


def test_a_label_with_no_test_rows_is_dropped(tmp_path):
    records = [*RECORDS, record("13", "train", "cooking_query", "kek tarifi ver")]
    folder = convert_from(tmp_path, records)
    meta = meta_of(folder)
    assert meta["labels"] == LABELS
    assert meta["dropped"]["label_missing_from_a_split"] == 1
    assert check_dataset(folder).ok


def test_other_locales_and_unknown_partitions_are_skipped(tmp_path):
    records = [
        *RECORDS,
        record("14", "train", "alarm_set", "wake me up at seven", locale="en-US"),
        record("15", "heldout", "alarm_set", "beni yedide uyandır"),
    ]
    dropped = meta_of(convert_from(tmp_path, records))["dropped"]
    assert dropped["other_locale"] == 1
    assert dropped["unknown_partition"] == 1


def test_the_revision_is_the_archive_hash(tmp_path):
    folder = convert_from(tmp_path, RECORDS)
    revision = meta_of(folder)["source_revision"]
    assert revision.startswith("sha256:")
    archive = tmp_path / "cache" / massive_tr.ARCHIVE_NAME
    assert revision == f"sha256:{massive_tr.sha256_file(archive)}"


def test_the_archive_is_fetched_once_and_then_reused(tmp_path):
    prepared = make_archive(tmp_path / "prepared.tar.gz", RECORDS)
    calls: list[str] = []

    def fetch(url: str, dest: Path) -> None:
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(prepared.read_bytes())

    cache = tmp_path / "cache"
    first = massive_tr.convert(tmp_path / "one", cache, fetch=fetch)
    second = massive_tr.convert(tmp_path / "two", cache, fetch=fetch)
    assert calls == [massive_tr.SOURCE_URL]
    assert (first / "train.jsonl").read_text(encoding="utf-8") == (
        second / "train.jsonl"
    ).read_text(encoding="utf-8")


def test_a_missing_member_is_an_error(tmp_path):
    cache = tmp_path / "cache"
    archive = cache / massive_tr.ARCHIVE_NAME
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("1.1/data/en-US.jsonl")
        info.size = 0
        tar.addfile(info, io.BytesIO(b""))
    with pytest.raises(RuntimeError, match="is missing"):
        massive_tr.convert(tmp_path / "out", cache, fetch=refuse)
