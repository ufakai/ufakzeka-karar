"""The helpers the Hub converters share: pinned fetch, apportioning, carving, writing."""

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from data.converters.hub_common import (  # noqa: E402
    apportion,
    dataset_url,
    fetch_file,
    files_note,
    iter_parquet_rows,
    merge_counts,
    read_rows,
    sha256_of,
    stratified_carve,
    stratified_take,
    write_dataset,
)
from data.instrument_format import Row, check_dataset

REVISION = "0123456789abcdef0123456789abcdef01234567"


def no_download(repo_id, filename, revision, target_dir):
    raise AssertionError(f"the tests must not reach the network: {repo_id} {filename}")


def test_a_cached_file_is_returned_without_downloading(tmp_path):
    cache = tmp_path / "downloads"
    target = cache / "owner__set" / REVISION / "data" / "train.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text("{}\n", encoding="utf-8")
    found = fetch_file("owner/set", "data/train.jsonl", REVISION, cache, download=no_download)
    assert found == target


def test_a_missing_file_is_downloaded_into_the_pinned_folder(tmp_path):
    calls = []

    def fake_download(repo_id, filename, revision, target_dir):
        calls.append((repo_id, filename, revision, target_dir))
        path = Path(target_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok", encoding="utf-8")
        return path

    cache = tmp_path / "downloads"
    path = fetch_file("owner/set", "a/b.csv", REVISION, cache, download=fake_download)
    assert path.read_text(encoding="utf-8") == "ok"
    assert calls == [("owner/set", "a/b.csv", REVISION, cache / "owner__set" / REVISION)]


def test_a_short_revision_is_refused(tmp_path):
    with pytest.raises(ValueError, match="40-character"):
        fetch_file("owner/set", "f.csv", "84dc7a3641", tmp_path, download=no_download)


def test_sha256_and_the_note_it_goes_into(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"veri")
    digest = sha256_of(path)
    assert len(digest) == 64
    assert files_note({"f.txt": digest}) == f"f.txt sha256 {digest}"
    assert dataset_url("owner/set") == "https://huggingface.co/datasets/owner/set"


def test_apportion_splits_by_weight_and_hands_leftovers_to_the_largest_remainder():
    assert apportion(10, {"a": 7, "b": 3}) == {"a": 7, "b": 3}
    # 5 * 1/3 is 1.666 each, so two of the three get the leftover, by remainder then key.
    assert apportion(5, {0: 1, 1: 1, 2: 1}) == {0: 2, 1: 2, 2: 1}
    assert sum(apportion(3000, {0: 2097, 1: 2097, 2: 806}).values()) == 3000
    assert apportion(4, {"x": 1.0, "y": 0.0}) == {"x": 4, "y": 0}


def test_apportion_never_hands_a_key_more_than_its_cap():
    quotas = apportion(6, {0: 2, 1: 10}, {0: 2, 1: 10})
    assert quotas == {0: 1, 1: 5}
    assert apportion(0, {0: 5}) == {0: 0}


def test_stratified_take_keeps_the_shares_the_order_and_the_seed():
    items = [(index, index % 3) for index in range(60)]
    taken = stratified_take(items, 12, lambda pair: pair[1], 1)
    assert len(taken) == 12
    assert Counter(label for _, label in taken) == {0: 4, 1: 4, 2: 4}
    assert taken == sorted(taken)
    assert taken == stratified_take(items, 12, lambda pair: pair[1], 1)
    assert taken != stratified_take(items, 12, lambda pair: pair[1], 2)
    assert stratified_take(items, 500, lambda pair: pair[1], 1) == items


def test_stratified_carve_splits_every_label_the_same_way():
    items = [(index, index % 3) for index in range(30)]
    parts = stratified_carve(
        items, {"train": 0.7, "validation": 0.1, "test": 0.2}, lambda p: p[1], 1
    )
    assert [len(parts[name]) for name in ("train", "validation", "test")] == [21, 3, 6]
    for name, per_label in (("train", 7), ("validation", 1), ("test", 2)):
        assert Counter(label for _, label in parts[name]) == {
            0: per_label,
            1: per_label,
            2: per_label,
        }
    assert parts["train"] == sorted(parts["train"])
    again = stratified_carve(
        items, {"train": 0.7, "validation": 0.1, "test": 0.2}, lambda p: p[1], 1
    )
    assert parts == again
    everything = [item for name in parts for item in parts[name]]
    assert sorted(everything) == items


def test_merge_counts_adds_and_keeps_zero_reasons():
    assert merge_counts({"a": 1, "b": 0}, {"a": 2, "c": 3}) == {"a": 3, "b": 0, "c": 3}


def test_read_rows_reads_the_three_source_file_types(tmp_path):
    parquet = tmp_path / "rows.parquet"
    pq.write_table(
        pa.table({"premise": ["Dava dilekçesi sunuldu."], "label": ["neutral"], "extra": [1]}),
        parquet,
    )
    assert read_rows(parquet, ("premise", "label")) == [
        {"premise": "Dava dilekçesi sunuldu.", "label": "neutral"}
    ]

    tsv = tmp_path / "rows.tsv"
    with tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["tweet", "label"])
        writer.writerow(["Haber doğru çıktı.", "True"])
        writer.writerow(["İlk satır\nikinci satır", "False"])
    rows = read_rows(tsv, ("tweet", "label"))
    # "True" and "False" must stay text, or two of the three classes get renamed.
    assert [row["label"] for row in rows] == ["True", "False"]
    assert rows[1]["tweet"] == "İlk satır\nikinci satır"

    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text(
        '{"variation": "Kedi uyuyor.", "label": 1}\n\n'
        '{"variation": "Kedi uyuyorlar.", "label": 0}\n',
        encoding="utf-8",
    )
    assert read_rows(jsonl, ("variation", "label")) == [
        {"variation": "Kedi uyuyor.", "label": 1},
        {"variation": "Kedi uyuyorlar.", "label": 0},
    ]
    with pytest.raises(ValueError, match="missing columns"):
        read_rows(jsonl, ("variation", "var_type"))
    with pytest.raises(ValueError, match="unsupported file type"):
        read_rows(tmp_path / "rows.txt", ("a",))


def test_iter_parquet_rows_walks_the_shards_in_order(tmp_path):
    paths = []
    for shard in range(3):
        path = tmp_path / f"shard-{shard}.parquet"
        pq.write_table(pa.table({"premise": [f"Metin {shard}"], "label": ["neutral"]}), path)
        paths.append(path)
    rows = list(iter_parquet_rows(paths, ("premise", "label"), batch_size=1))
    assert [row["premise"] for row in rows] == ["Metin 0", "Metin 1", "Metin 2"]


def meta_for(splits, **changes):
    meta = {
        "name": "demo",
        "task": "demo",
        "source": "https://example.invalid/demo",
        "source_revision": REVISION,
        "license_id": "MIT",
        "labels": ["hayır", "evet"],
        "splits": {name: len(rows) for name, rows in splits.items()},
        "split_origin": dict.fromkeys(splits, "source"),
        "dropped": {},
        "notes": "",
    }
    meta.update(changes)
    return meta


def test_write_dataset_writes_the_folder_and_checks_it(tmp_path):
    splits = {
        "train": [Row("t1", "evet cümlesi", None, 1), Row("t2", "hayır cümlesi", None, 0)],
        "validation": [Row("v1", "doğrulama cümlesi", None, 1)],
        "test": [Row("s1", "sınama evet", None, 1), Row("s2", "sınama hayır", None, 0)],
    }
    folder = write_dataset(tmp_path / "out", meta_for(splits), splits)
    assert folder == tmp_path / "out" / "demo"
    assert check_dataset(folder).ok
    assert (
        json.loads((folder / "meta.json").read_text(encoding="utf-8"))["source_revision"]
        == REVISION
    )
    assert len((folder / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 2


def test_write_dataset_raises_when_the_folder_does_not_check_out(tmp_path):
    splits = {
        "train": [Row("t1", "evet cümlesi", None, 1)],
        "validation": [],
        "test": [Row("s1", "sınama evet", None, 1)],
    }
    with pytest.raises(RuntimeError, match="no rows for labels"):
        write_dataset(tmp_path / "out", meta_for(splits), splits)
