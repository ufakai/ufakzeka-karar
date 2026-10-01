"""The instrument dataset format: leak removal and the folder check."""

import json

from data.instrument_format import META_KEYS, Row, check_dataset, remove_leaks, write_split


def rows(prefix, texts, label=0):
    return [Row(f"{prefix}-{i}", text, None, label) for i, text in enumerate(texts)]


def test_leaks_are_removed_from_train_first_and_test_is_untouched():
    splits = {
        "train": rows("tr", ["a", "b", "c", "c", "SHARED  ", "valonly"]),
        "validation": rows("va", ["valonly", "shared", "v"]),
        "test": rows("te", ["shared", "t", "t"]),
    }
    cleaned, dropped = remove_leaks(splits)
    assert [r.text for r in cleaned["test"]] == ["shared", "t"]
    assert [r.text for r in cleaned["validation"]] == ["valonly", "v"]
    assert [r.text for r in cleaned["train"]] == ["a", "b", "c"]
    assert dropped == {"duplicate_within_split": 2, "validation_in_test": 1, "train_in_eval": 2}
    assert list(cleaned) == ["train", "validation", "test"]


def test_a_pair_is_the_same_example_only_when_both_texts_match():
    splits = {
        "train": [Row("1", "p", "h1", 0), Row("2", "p", "h2", 1)],
        "validation": [Row("3", "q", "h", 0)],
        "test": [Row("4", "p", "h1", 0)],
    }
    cleaned, dropped = remove_leaks(splits)
    assert [r.id for r in cleaned["train"]] == ["2"]
    assert dropped["train_in_eval"] == 1


def make_folder(tmp_path, split_rows_by_name, labels=("hayır", "evet"), **meta_changes):
    folder = tmp_path / "demo"
    for name, split_rows in split_rows_by_name.items():
        write_split(folder / f"{name}.jsonl", split_rows)
    meta = {
        "name": "demo",
        "task": "demo task",
        "source": "https://example.invalid/demo",
        "source_revision": "abc",
        "license_id": "MIT",
        "labels": list(labels),
        "splits": {name: len(r) for name, r in split_rows_by_name.items()},
        "split_origin": {name: "source" for name in split_rows_by_name},
        "dropped": {},
        "notes": "",
    }
    meta.update(meta_changes)
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return folder


def good_splits():
    return {
        "train": rows("tr", ["a", "b"], 0) + rows("tr1", ["c", "d"], 1),
        "validation": rows("va", ["e"], 0) + rows("va1", ["f"], 1),
        "test": rows("te", ["g"], 0) + rows("te1", ["h"], 1),
    }


def test_a_clean_folder_passes(tmp_path):
    report = check_dataset(make_folder(tmp_path, good_splits()))
    assert report.ok, report.errors
    assert report.counts == {"train": 4, "validation": 2, "test": 2}
    assert set(json.loads((tmp_path / "demo/meta.json").read_text())) == META_KEYS


def test_a_text_shared_between_train_and_test_fails(tmp_path):
    splits = good_splits()
    splits["train"].append(Row("leak", " G ", None, 0))
    report = check_dataset(make_folder(tmp_path, splits))
    assert any("both train and test" in e for e in report.errors)


def test_a_label_missing_from_test_fails(tmp_path):
    splits = good_splits()
    splits["test"] = rows("te", ["g", "h"], 0)
    report = check_dataset(make_folder(tmp_path, splits))
    assert any("test: no rows for labels ['evet']" in e for e in report.errors)


def test_bad_rows_are_reported(tmp_path):
    splits = good_splits()
    splits["train"] += [
        Row("tr-0", "repeated id", None, 0),
        Row("x1", "  ", None, 0),
        Row("x2", "out of range", None, 7),
        Row("x3", "bool label", None, True),
        Row("x4", "a", None, 0),
    ]
    errors = check_dataset(make_folder(tmp_path, splits)).errors
    for fragment in (
        "'tr-0' is missing or repeated",
        "x1 has no text",
        "label 7 is out of range",
        "x3 label must be an integer",
        "x4 repeats a text",
    ):
        assert any(fragment in e for e in errors), fragment


def test_meta_must_match_the_files(tmp_path):
    folder = make_folder(tmp_path, good_splits(), splits={"train": 99, "validation": 2, "test": 2})
    assert any("meta says 99" in e for e in check_dataset(folder).errors)


def test_meta_keys_are_fixed(tmp_path):
    folder = make_folder(tmp_path, good_splits(), licence="MIT")
    assert any("unexpected ['licence']" in e for e in check_dataset(folder).errors)


def test_mixed_single_and_pair_rows_fail(tmp_path):
    splits = good_splits()
    splits["train"].append(Row("p", "premise", "hypothesis", 0))
    assert any("text_pair" in e for e in check_dataset(make_folder(tmp_path, splits)).errors)


def test_missing_files_are_reported(tmp_path):
    folder = make_folder(tmp_path, good_splits())
    (folder / "validation.jsonl").unlink()
    assert any("validation.jsonl is missing" in e for e in check_dataset(folder).errors)
    assert not check_dataset(tmp_path / "nothing").ok
