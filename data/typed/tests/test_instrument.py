"""Step 1 sets become typed rows only when cleared, and never from test."""

import json

import pytest

from data.typed.instrument import MASSIVE_OPTIONS, cleared_for_training, typed_rows
from schema.rows import outcomes


def write_set(root, name, labels, rows, split="train"):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text(json.dumps({"labels": labels}), encoding="utf-8")
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (folder / f"{split}.jsonl").write_text(body, encoding="utf-8")


def test_each_set_becomes_the_question_type_it_should(tmp_path):
    write_set(tmp_path, "offenseval_tr", ["NOT", "OFF"],
              [{"id": "a", "text": "iyi günler", "text_pair": None, "label": 0},
               {"id": "b", "text": "hakaret", "text_pair": None, "label": 1}])  # fmt: skip
    rows = list(typed_rows("offenseval_tr", "train", root=tmp_path))
    assert [r.question.type for r in rows] == ["noul", "noul"]
    assert rows[0].target == {"true": 0.0, "false": 1.0}
    assert rows[1].target == {"true": 1.0, "false": 0.0}
    assert all(r.label_kind == "human" and r.origin == "converted" for r in rows)

    labels = list(MASSIVE_OPTIONS)
    wake = {"id": "m", "text": "yarın yedide uyandır", "text_pair": None, "label": 2}
    write_set(tmp_path, "massive_tr", labels, [wake])
    (row,) = typed_rows("massive_tr", "train", root=tmp_path)
    # 59 intents become the correct one plus nine seeded distractors.
    assert row.question.type == "choice" and len(outcomes(row.question)) == 10
    assert row.target["alarm kurma"] == 1.0 and sum(row.target.values()) == 1.0
    (again,) = typed_rows("massive_tr", "train", root=tmp_path)
    assert outcomes(again.question) == outcomes(row.question), "the draw is not reproducible"
    assert len(set(MASSIVE_OPTIONS.values())) == 59, "two intents share a Turkish name"

    write_set(tmp_path, "mide22", ["False", "Other", "True"],
              [{"id": "x", "text": "iddia", "text_pair": None, "label": 0}])  # fmt: skip
    (row,) = typed_rows("mide22", "train", root=tmp_path)
    assert row.target["yanlış bilgi"] == 1.0


def test_the_test_split_is_never_converted(tmp_path):
    with pytest.raises(ValueError, match="only train and validation"):
        list(typed_rows("offenseval_tr", "test", root=tmp_path))


def test_nothing_is_training_data_until_the_manifest_says_so(tmp_path):
    manifest = tmp_path / "MANIFEST.yaml"
    manifest.write_text(
        "version: 1\nfiles:\n"
        "  - path: data/raw/instrument/massive_tr/\n    use: measure\n"
        "  - path: data/raw/instrument/mide22/\n    use: both\n",
        encoding="utf-8",
    )
    assert not cleared_for_training("massive_tr", manifest)
    assert cleared_for_training("mide22", manifest)
    with pytest.raises(KeyError):
        cleared_for_training("trcola", manifest)


def test_the_real_manifest_clears_exactly_what_d78_promoted():
    """Nothing is training data without a dated decision; three sets were promoted."""
    for name in ("massive_tr", "offenseval_tr", "mide22"):
        assert cleared_for_training(name), f"{name} was promoted"
    for name in ("trcola", "legal_nli_tr"):
        assert not cleared_for_training(name), f"{name} is cleared without a decision"
