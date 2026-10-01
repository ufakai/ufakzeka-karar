"""Court summaries become ten-option right questions, and the test file is never read."""

import json

import pytest

from data.typed import legal
from schema.rows import outcomes

EDUCATION = "Başvuru, eğitim hakkının ihlal edildiği iddiasına ilişkindir."
DELAY = "Başvuru, makul sürede yargılanma hakkının ihlal edildiği iddiasına ilişkindir."


def ruling(link, summary, heading, outcome=1):
    return {
        "text": "Tam karar metni, sonucuyla birlikte.",
        "Haklar": heading,
        "Kararın Bağlantı Linki": link,
        "Başvuru Konusu": summary,
        "labels": outcome,
    }


def write(folder, name, records):
    (folder / name).write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def raw(tmp_path):
    folder = tmp_path / "raw"
    folder.mkdir()
    write(folder, "train.json", [
        ruling("a", "Başvuru, gözaltı süresinin uzunluğu nedeniyle kişi özgürlüğünün ihlal "
                    "edildiği iddiasına ilişkindir.", "Kişi özgürlüğü ve güvenliği hakkı"),
        ruling("b", "Başvuru, sendika üyeliği nedeniyle nakil iddiasına ilişkindir. "
                    "Başvurucu ali@example.com adresinden yazmıştır.", "Sendika hakkı"),
        # The same summary filed under a civil and a criminal heading: dropped.
        ruling("c", DELAY, "Adil yargılanma hakkı (Medeni Hak ve Yükümlülükler)"),
        ruling("d", DELAY, "Adil yargılanma hakkı (Suç İsnadı)"),
        # An exact duplicate with the same heading: kept once.
        ruling("e", EDUCATION, "Eğitim hakkı"),
        ruling("f", EDUCATION, "Eğitim hakkı"),
        ruling("g", "   ", "İfade özgürlüğü"),
    ])  # fmt: skip
    write(folder, "dev.json", [
        ruling("h", "Başvuru, gazetedeki yazı nedeniyle verilen ceza nedeniyle ifade "
                    "özgürlüğünün ihlal edildiği iddiasına ilişkindir.", "İfade özgürlüğü"),
        # Already in train: dropped from validation.
        ruling("i", EDUCATION, "Eğitim hakkı"),
    ])  # fmt: skip
    # A heading the converter does not know would stop the build if this were read.
    write(folder, "test.json", [ruling("t", "Test özeti.", "Bilinmeyen hak")])
    return folder


def test_rows_are_ten_option_one_hot_questions(raw, tmp_path):
    report = legal.build(raw, tmp_path / "out", download=False)
    train = read(tmp_path / "out" / "train.jsonl")
    validation = read(tmp_path / "out" / "validation.jsonl")
    assert report["splits"]["train"]["rows"] == len(train) == 3
    assert report["splits"]["validation"]["rows"] == len(validation) == 1
    for row in train + validation:
        assert row["question"]["type"] == "choice"
        assert list(row["question"]["criteria"]) == list(legal.OPTIONS)
        assert sorted(row["target"].values()) == [0.0] * 9 + [1.0]
        assert (row["track"], row["task"], row["label_kind"], row["origin"]) == (
            "hukuk", "aym_rights", "human", "converted",
        )  # fmt: skip
    by_state = {row["state"]: row for row in train}
    union = next(s for s in by_state if "sendika" in s)
    assert by_state[union]["target"][legal.ASSEMBLY] == 1.0
    assert "ali@example.com" not in union and "[e-posta]" in union
    education = next(s for s in by_state if "eğitim" in s)
    assert by_state[education]["target"][legal.OTHER] == 1.0
    assert validation[0]["target"][legal.EXPRESSION] == 1.0


def test_drops_are_counted(raw, tmp_path):
    report = legal.build(raw, tmp_path / "out", download=False)
    assert report["dropped"]["train"] == {
        "duplicate_text": 1,
        "empty_summary": 1,
        "source_rows": 7,
        "text_with_two_labels": 2,
    }
    assert report["dropped"]["validation"]["text_also_in_train"] == 1


def test_the_test_split_is_never_downloaded_or_read():
    assert "test.json" not in legal.SPLIT_FILES.values()
    assert "test.json" not in legal.PINNED
    assert set(legal.SPLIT_FILES) == {"train", "validation"}


def test_an_unknown_heading_stops_the_build(raw, tmp_path):
    write(raw, "dev.json", [ruling("x", "Özet.", "Yeni bir hak")])
    with pytest.raises(ValueError, match="has no option"):
        legal.build(raw, tmp_path / "out", download=False)


def test_the_option_table_fits_one_pass():
    assert len(legal.OPTIONS) == 10
    assert set(legal.HEADINGS.values()) == set(legal.OPTIONS)
    assert len(legal.HEADINGS) == 21
    row = legal.make_row(legal.Item("a", "Özet.", legal.PROPERTY), "train")
    assert len(outcomes(row.question)) == 10
