"""HakemBench-dev: which owner labels enter, and a training row carrying a dev text is caught."""

import json

from bench.dev import dev_rows, training_overlap

TEXT = "Soru: Kargom üç gündür gelmedi, siparişim nerede?\nCevap: Takip numaranızla kargo sayfasından bakabilirsiniz."  # noqa: E501


def item(i, text=TEXT):
    return {"id": f"r{i}", "cell": "konu-noul", "task": "konu-noul-x", "type": "noul",
            "text": text, "question": "Soru mu?",
            "options": [{"key": "true"}, {"key": "false"}]}  # fmt: skip


def test_only_answered_rows_enter_with_a_one_hot_target():
    labels = {"r0": {"answer": "false", "flag": "none"}, "r1": {"answer": None, "flag": "no_fit"},
              "r2": {"answer": None, "flag": "cant_tell"}}  # fmt: skip
    rows = dev_rows([item(0), item(1), item(2), item(3)], labels)
    assert [r["id"] for r in rows] == ["r0"]
    assert rows[0]["target"] == {"true": 0.0, "false": 1.0}


def test_a_training_row_carrying_a_dev_text_is_found(tmp_path):
    rows = dev_rows([item(0)], {"r0": {"answer": "true", "flag": "none"}})
    train = tmp_path / "train.jsonl"
    lines = [{"row_id": "a", "state": TEXT},
             {"row_id": "b", "state": "Başka bir metin, hiç ilgisi yok, tamamen farklı konu."},
             {"row_id": "c", "state": "Giriş cümlesi burada. " * 30 + TEXT}]  # fmt: skip
    train.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
    found = {f["row_id"]: (f["rule"], f["refs"]) for f in training_overlap(rows, [train])}
    assert found == {"a": ("whole", ["r0"]), "c": ("covers", ["r0"])}
