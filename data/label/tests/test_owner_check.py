"""The judges against the owner's labels: what is counted, what is left out."""

import json

import pytest

from data.label.owner_check import compare, load_labels

TEXT = "Soru: Kargom ne zaman gelir?\nCevap: Siparişiniz iki iş günü içinde kargoya verilir."
STUB = ("Soru: kızımın absans epilepsi\nCevap: kızımın absans epilepsi dr. ad doktorumuzun"
        " branşındaki cevaplarını görüntüleyin.")  # fmt: skip


def item(i, text=TEXT):
    return {"id": f"r{i}", "cell": "konu-noul", "text": text}


def build_row(a_true, c_true):
    judges = [{"judge": "A", "distribution": {"true": a_true, "false": 1 - a_true}},
              {"judge": "C", "distribution": {"true": c_true, "false": 1 - c_true}}]  # fmt: skip
    mean = (a_true + c_true) / 2
    return {"judges": judges, "target": {"true": mean, "false": 1 - mean}}


def test_judges_are_scored_on_answered_rows_only():
    items = [item(i) for i in range(4)] + [item(4, STUB), item(5), item(6)]
    build = {f"r{i}": build_row(0.9, 0.3) for i in range(7)}
    labels = {f"r{i}": {"answer": "true", "flag": "none"} for i in range(5)}
    labels["r5"] = {"answer": None, "flag": "cant_tell"}
    out = compare(items, build, labels, draws=100)
    assert out["counts"] == {"answered": 4, "dropped_by_filters": 1, "cant_tell": 1,
                             "unlabelled": 1}  # fmt: skip
    assert out["sources"]["A"]["accuracy"] == 1.0
    assert out["sources"]["C"]["accuracy"] == 0.0
    assert out["sources"]["A"]["brier"] == pytest.approx(2 * 0.1**2)
    assert out["accuracy_a_minus_c"]["difference"] == 1.0
    assert out["sources"]["panel"]["accuracy"] == 1.0  # the mean, 0.6, still says "true"


def test_labels_are_read_from_the_desk_export(tmp_path):
    doc = {"item": "r1", "answer": "false", "flag": "none", "note": "", "seconds": 12}
    (tmp_path / "r1.json").write_text(json.dumps(doc))
    assert load_labels(tmp_path)["r1"]["answer"] == "false"
