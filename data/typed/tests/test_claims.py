"""FACTurk claims become verdict questions; ratings are normalised and unusable rows counted."""

import csv
import io
import json
import zipfile

from data.typed import claims
from data.typed.common import carve

COLUMNS = [
    "author", "claim", "claim_id_lowconf", "claim_id_highconf", "content", "date_published",
    "normalised_rating", "organisation", "original_verdict", "report_id", "row_id", "title", "url",
]  # fmt: skip


def report(report_id, claim, organisation, verdict, rating, cluster=None):
    return {
        "author": "Yazar",
        "claim": claim,
        "claim_id_lowconf": cluster or f"cluster_{report_id}",
        "claim_id_highconf": cluster or f"cluster_{report_id}",
        "content": "Kuruluşun makale metni, okunmamalı.",
        "date_published": "",
        "normalised_rating": rating,
        "organisation": organisation,
        "original_verdict": verdict,
        "report_id": report_id,
        "row_id": "1",
        "title": claim,
        "url": "https://example.org",
    }


def write_zip(folder, records):
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(records)
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(folder / "FACTurk.zip", "w") as bundle:
        bundle.writestr(claims.CSV_NAME, "﻿" + text.getvalue())


def rows_of(folder):
    out = []
    for split in ("train", "validation"):
        path = folder / f"{split}.jsonl"
        out += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return out


def test_ratings_are_normalised_and_unusable_rows_counted(tmp_path):
    raw = tmp_path / "raw"
    write_zip(raw, [
        report("1", "Videonun depremden olduğu iddiası", "Teyit", "Yanlış", "Misleading"),
        report("2", "Bakanın istifa ettiği", "Malumatfuruş", "YANLIS", "false"),
        report("3", "Fotoğrafın 1951 yılından olduğu", "Malumatfuruş", "YANLIÞ", "Yanliþ"),
        report("4", "Köprünün açıldığı", "Malumatfuruş", "DOGRU", "Dogru"),
        report("5", "Anket sonucunun gerçek olduğu", "Teyit", "Sonuçlandırılamadı", "Unknown"),
        report("6", "Kaygusuz Abdal'ın Yedi Ulu Ozandan Biri Olduğu", "Malumatfuruş",
               "Unknown", "Unknown"),
        report("7", "Aşıların kısırlık yaptığı", "Yalansavar", "Yanlış", "False"),
        report("8", "Uydu görüntülerinin kasırgayı gösterdiği", "Doğrula", "Doğru", ""),
        report("9", "Zamların geri alındığı @kullanici123 paylaştı", "Teyit", "Karma", "Mixed"),
    ])  # fmt: skip
    result = claims.build(raw, tmp_path / "out", download=False)
    assert result["dropped"]["source"] == {
        "no_verdict_on_page": 1,
        "source_rows": 9,
        "unmapped_rating": 1,
        "verdict_fixed_by_scraper": 1,
    }
    rows = {row["state"]: row for row in rows_of(tmp_path / "out")}
    assert len(rows) == 6

    def top(state):
        target = rows[state]["target"]
        return max(target, key=target.get)

    assert top("Videonun depremden olduğu iddiası") == claims.FALSE  # misleading folds into false
    assert top("Bakanın istifa ettiği") == claims.FALSE
    assert top("Fotoğrafın 1951 yılından olduğu") == claims.FALSE
    assert top("Köprünün açıldığı") == claims.TRUE
    assert top("Anket sonucunun gerçek olduğu") == claims.UNSETTLED
    masked = next(state for state in rows if state.startswith("Zamların"))
    assert "kullanici123" not in masked and top(masked) == claims.MIXED
    for row in rows.values():
        assert (row["track"], row["task"], row["label_kind"]) == (
            "dogrulama", "facturk_verdict", "human",
        )  # fmt: skip
        assert list(row["question"]["criteria"]) == list(claims.OPTIONS)
        assert sorted(row["target"].values()) == [0.0] * 3 + [1.0]


def test_a_claim_cluster_stays_on_one_side_and_conflicts_drop(tmp_path):
    raw = tmp_path / "raw"
    records = [
        report(str(i), f"İddia numarası {i}", "Teyit", "Yanlış", "False", cluster=f"c{i % 40}")
        for i in range(400)
    ]
    records += [
        report("a", "Aynı iddia", "Teyit", "Yanlış", "False", cluster="x"),
        report("b", "Aynı iddia", "Doğruluk Payı", "Doğru", "True", cluster="x"),
    ]
    write_zip(raw, records)
    result = claims.build(raw, tmp_path / "out", download=False)
    assert (
        result["dropped"]["train"].get("text_with_two_labels", 0)
        + result["dropped"]["validation"].get("text_with_two_labels", 0)
        == 2
    )
    side = {}
    for split in ("train", "validation"):
        path = tmp_path / "out" / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            number = int(json.loads(line)["state"].rsplit(" ", 1)[1])
            assert side.setdefault(number % 40, split) == split, "a cluster straddles the splits"
    assert set(side.values()) == {"train", "validation"}


def test_fold_and_carve_are_stable():
    assert claims.fold("Yanliþ") == claims.fold("YANLIŞ") == claims.fold(" yanlış ") == "yanlis"
    assert claims.fold("Doğru") == claims.fold("Dogru") == "dogru"
    assert carve("cluster_7", "facturk_verdict") == carve("cluster_7", "facturk_verdict")
    share = sum(carve(f"k{i}", "s") == "validation" for i in range(10000)) / 10000
    assert 0.09 < share < 0.11
