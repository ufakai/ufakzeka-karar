"""Turkish legal NLI conversion, on invented pairs in the source schema. No network."""

import json
from collections import Counter

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from data.converters import legal_nli_tr  # noqa: E402
from data.instrument_format import check_dataset, read_split  # noqa: E402

LABELS = legal_nli_tr.LABELS
# Sentences written for this test, in the shape of a ruling summary but far
# shorter than a real one. None of them comes from the source.
PREMISE = "Davacı vekili dilekçesinde özetle {} numaralı sözleşmeden doğan alacağı talep etmiştir."
HYPOTHESIS = "Davalı taraf {} numaralı faturaya konu bedelin ödenmediğini beyan etmiştir."


def pair(number, label):
    return {
        "premise": PREMISE.format(number),
        "hypothesis": HYPOTHESIS.format(number),
        "label": label,
    }


def train_rows():
    rows = [pair(100 + index, LABELS[index % 3]) for index in range(12)]
    rows.append(pair(100, LABELS[0]))  # a repeat of the first pair
    rows.append(pair(900, "entailment"))  # also in test, so it must leave train
    rows.append({"premise": PREMISE.format(7), "hypothesis": "  ", "label": "neutral"})
    rows.append(pair(8, "related"))  # a label that is not one of the three
    return rows


def validation_rows():
    return [pair(200 + index, LABELS[index % 2]) for index in range(4)]


def sample_test_rows():
    rows = [pair(300 + index, LABELS[index % 3]) for index in range(6)]
    rows[0] = pair(900, "entailment")
    rows.append(pair(900, "entailment"))  # a repeat inside test
    return rows


def write_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({name: [row[name] for row in rows] for name in legal_nli_tr.COLUMNS})
    pq.write_table(table, path)


@pytest.fixture
def fetch(tmp_path):
    source = tmp_path / "source"
    shards = legal_nli_tr.FILES["train"]
    rows = train_rows()
    # The rows are spread over the four shards the source has, in file order.
    per_shard = [rows[0:6], rows[6:10], rows[10:14], rows[14:]]
    for name, shard_rows in zip(shards, per_shard, strict=True):
        write_parquet(source / name, shard_rows)
    write_parquet(source / legal_nli_tr.FILES["validation"][0], validation_rows())
    write_parquet(source / legal_nli_tr.FILES["test"][0], sample_test_rows())

    def local_fetch(repo_id, filename, revision, cache_dir):
        assert repo_id == legal_nli_tr.REPO
        assert revision == legal_nli_tr.REVISION
        return source / filename

    return local_fetch


TARGETS = {"train": 6, "validation": 2, "test": 3}


def convert(tmp_path, fetch, name="out", targets=TARGETS):
    return legal_nli_tr.convert(tmp_path / name, tmp_path / "cache", fetch=fetch, targets=targets)


def labels_of(folder, split):
    return Counter(row.label for row in read_split(folder / f"{split}.jsonl"))


def test_the_written_folder_passes_the_format_check(tmp_path, fetch):
    assert check_dataset(convert(tmp_path, fetch)).ok


def test_the_premise_is_the_text_and_the_hypothesis_is_the_pair(tmp_path, fetch):
    rows = read_split(convert(tmp_path, fetch) / "train.jsonl")
    assert all(row.text.startswith("Davacı vekili") for row in rows)
    assert all(row.text_pair is not None and row.text_pair.startswith("Davalı") for row in rows)
    assert LABELS == ["contradiction", "entailment", "neutral"]


def test_the_subsample_hits_the_targets_and_keeps_the_label_shares(tmp_path, fetch):
    folder = convert(tmp_path, fetch)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["splits"] == TARGETS
    assert labels_of(folder, "train") == {0: 2, 1: 2, 2: 2}
    assert labels_of(folder, "test") == {0: 1, 1: 1, 2: 1}
    assert sum(labels_of(folder, "validation").values()) == 2


def test_leaks_duplicates_and_unusable_rows_are_dropped_and_counted(tmp_path, fetch):
    folder = convert(tmp_path, fetch)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["dropped"]["empty_or_unknown"] == 2
    # One repeat in train, one in test, and the train copy of a test pair.
    assert meta["dropped"]["duplicate_within_split"] == 2
    assert meta["dropped"]["train_in_eval"] == 1
    assert meta["dropped"]["validation_in_test"] == 0
    assert meta["dropped"]["subsampled_out"] == 12 - 6 + (4 - 2) + (6 - 3)
    leaked = HYPOTHESIS.format(900)
    assert leaked not in [row.text_pair for row in read_split(folder / "train.jsonl")]


def test_leak_removal_covers_the_whole_source_train_split_not_the_subsample(tmp_path, fetch):
    # With a target of 12 the subsample keeps everything, and the pair shared
    # with test is still gone: it was removed before the subsample was drawn.
    folder = convert(tmp_path, fetch, targets={"train": 12, "validation": 4, "test": 6})
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["splits"]["train"] == 12
    assert meta["dropped"]["train_in_eval"] == 1
    assert meta["dropped"]["subsampled_out"] == 0


def test_meta_records_the_pinned_revision_and_the_subsample(tmp_path, fetch):
    folder = convert(tmp_path, fetch)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["name"] == "legal_nli_tr"
    assert meta["source_revision"] == "67baa141cf4f6634c983d77eea193c5535611e5a"
    assert meta["source"] == "https://huggingface.co/datasets/Turkish-NLI/legal_nli_TR_V1"
    assert meta["license_id"] == "Apache-2.0"
    for split, origin in meta["split_origin"].items():
        assert f"stratified subsample to {TARGETS[split]}" in origin
        assert "random.Random(1)" in origin
    assert "sha256" in meta["notes"]
    assert meta["notes"].count("sha256") == 6


def test_ids_are_unique_and_the_subsample_repeats_exactly(tmp_path, fetch):
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
