"""r4 row building on small fixtures: votes, agreement filter, skip lists. No agents, no network."""

import json

import pytest

from data.label import r4
from schema.rows import TrainingRow


def test_a_distribution_is_renormalised_and_a_bad_one_refused():
    got = r4.distribution({"true": 0.7, "false": 0.29}, ["true", "false"])
    assert got["true"] == pytest.approx(0.7 / 0.99)
    assert r4.distribution({"true": 0.5, "false": 0.2}, ["true", "false"]) is None  # sums to 0.7
    assert r4.distribution({"true": 1.0, "maybe": 0.0}, ["true", "false"]) is None
    assert r4.distribution("true", ["true", "false"]) is None


def test_the_check_accepts_a_skip_list_and_names_what_is_missing(tmp_path):
    units = tmp_path / "labels" / "units" / "batch_00.json"
    units.parent.mkdir(parents=True)
    units.write_text(json.dumps([{"id": "g-1", "text": "a"}, {"id": "g-2", "text": "b"}]))
    out = tmp_path / "out.json"
    out.write_text(json.dumps({"g-1": {"true": 0.9, "false": 0.1}, "_skipped": ["g-2"]}))
    assert r4.check_output(units, out) == []
    out.write_text(json.dumps({"g-1": {"true": 0.9, "false": 0.1}}))
    assert r4.check_output(units, out) == ["g-2: missing"]


def test_written_rows_keep_agreement_only_and_average_intent_with_the_label(tmp_path, monkeypatch):
    monkeypatch.setattr(r4, "WORK", tmp_path)
    question = {"qid": "offenseval_tr", "type": "noul", "instructions": "Saldırgan mı?",
                "criteria": {"true": "evet", "false": "hayır"}}  # fmt: skip
    for track in ("guvenlik", "moderasyon"):
        root = tmp_path / "labels" / track
        (root / "out").mkdir(parents=True)
        kept = [{"id": f"{track[0]}-1", "text": f"{track} bir", "intended": True},
                {"id": f"{track[0]}-2", "text": f"{track} iki", "intended": False},
                {"id": f"{track[0]}-3", "text": f"{track} üç", "intended": True}]  # fmt: skip
        (root / "kept.json").write_text(json.dumps(kept))
        (root / "question.json").write_text(json.dumps(question))
        labels = {f"{track[0]}-1": {"true": 0.8, "false": 0.2},
                  f"{track[0]}-2": {"true": 0.9, "false": 0.1},  # differs from the intent
                  "_skipped": [f"{track[0]}-3"]}  # fmt: skip
        (root / "out" / "out_00.json").write_text(json.dumps(labels))
    report: dict = {}
    rows = r4.written_rows(report)
    assert len(rows) == 2
    row = rows[0]
    assert isinstance(row, TrainingRow)
    assert row.task == "offenseval_tr-r4"
    assert row.target["true"] == pytest.approx(0.9)  # mean of 1.0 (intent) and 0.8 (label)
    assert [v.judge for v in row.judges] == ["D", "E"]
    assert report["guvenlik"] == {"kept": 1, "differ (intended false)": 1, "no label": 1}


def test_the_validation_split_is_a_fixed_function_of_the_text():
    texts = [f"metin {n}" for n in range(2000)]
    splits = [r4.validation_split(t) for t in texts]
    assert splits == [r4.validation_split(t) for t in texts]
    assert 0.07 < splits.count("validation") / len(splits) < 0.13


def test_the_new_taxonomy_leaves_the_benchmarks_contexts_and_forms_out():
    from bench.hakembench import generate, topup

    assert not set(r4.MOD_FORMS) & (set(generate.MODERATION_FORMS) | set(topup.MODERATION_FORMS))
    benchmark_contexts = {
        "bankacılık müşteri asistanı", "e-posta özetleyici", "sağlık randevu asistanı",
        "sigorta hasar asistanı", "okul ve öğrenci asistanı", "muhasebe ve fatura aracı",
        "kargo takip botu", "seyahat planlayıcı", "İK ve işe alım asistanı",
        "sosyal medya moderasyon botu", "hukuk bürosu asistanı", "oyun topluluğu Discord botu",
        "akıllı ev asistanı", "havayolu rezervasyon asistanı", "belediye çağrı merkezi asistanı",
        "e-ticaret destek botu", "kurumsal belge arama (RAG) asistanı", "kod asistanı",
    }  # fmt: skip
    assert not set(r4.GUARD_CONTEXTS) & benchmark_contexts
