"""The support rubric: option wording, blind units, the gold rule and the owner check."""

import json

import pytest

from bench.hakembench import rubric as R


def item(uid="sss-1"):
    old = {
        "hassasiyet-noul": {"type": "noul", "instructions": "q1",
                            "criteria": {"true": "eski", "false": "eski"}},
        "insan_destegi-choice": {"type": "choice", "instructions": "q2", "criteria": {
            "Acil yönlendirme": "e", "Yönlendirme gereksiz": "e", "İsteğe bağlı yönlendirme": "e"}},
        "yanit_yeterliligi-choice": {"type": "choice", "instructions": "q3", "criteria": {
            "Kapsamlı çözüm": "e", "Kişisel deneyim": "e", "Kısmi yönlendirme": "e",
            "İlgisiz içerik": "e"}},
        "yanit_yeterliligi-score": {"type": "score", "instructions": "q4",
                                    "criteria": ["a", "b", "c"]},
    }  # fmt: skip
    return {"id": uid, "track": "sss", "state": "Soru: x\nCevap: y", "questions": old}


def test_rewritten_changes_only_the_descriptions():
    new = R.rewritten(item())
    for qid in R.QIDS:
        assert new["questions"][qid]["criteria"] == R.RUBRIC[qid]
        old = item()["questions"][qid]["criteria"]
        if isinstance(old, dict):
            assert list(new["questions"][qid]["criteria"]) == list(old)
        assert new["questions"][qid]["instructions"] == item()["questions"][qid]["instructions"]
    bad = item()
    bad["questions"]["insan_destegi-choice"]["criteria"] = {"Başka": "x"}
    with pytest.raises(ValueError):
        R.rewritten(bad)


def test_the_second_pass_sees_the_same_options_in_another_order():
    items = [R.rewritten(item(f"sss-{n}")) for n in range(20)]
    plain, shuffled = R.units(items), R.units(items, shuffle_seed=128)
    moved = 0
    for u, v in zip(plain, shuffled, strict=True):
        for q, r in zip(u["questions"], v["questions"], strict=True):
            assert sorted(o["key"] for o in q["options"]) == sorted(o["key"] for o in r["options"])
            moved += [o["key"] for o in q["options"]] != [o["key"] for o in r["options"]]
            if q["type"] == "score":
                assert q["options"] == r["options"]
    assert moved > 0


def test_gold_takes_agreement_then_adjudication():
    assert R.decide("1", "1", None) == ("1", "ai_rubric_agreed")
    assert R.decide("1", "2", "2") == ("2", "ai_rubric_adjudicated")
    assert R.decide("1", "2", None) == (None, "needs adjudication")
    assert R.decide(None, None, None)[0] is None


def test_the_check_counts_owner_answers_per_question(tmp_path):
    docs = [
        {"item": "sss-1", "qid": "hassasiyet-noul", "verdict": "label", "answer": "true",
         "proposed": None, "flag": "none"},
        {"item": "sss-1", "qid": "insan_destegi-choice", "verdict": "agree", "answer": None,
         "proposed": "Yönlendirme gereksiz", "flag": "none"},
        {"item": "sss-2", "qid": "hassasiyet-noul", "verdict": "flag", "answer": None,
         "proposed": None, "flag": "cant_tell"},
        {"item": "egitim-1", "qid": "cevap_puani", "verdict": "label", "answer": "2",
         "proposed": None, "flag": "none"},
    ]  # fmt: skip
    for n, d in enumerate(docs):
        (tmp_path / f"{n}.json").write_text(json.dumps(d))
    owner = R.owner_answers(tmp_path)
    assert owner == {
        "sss-1~hassasiyet-noul": "true",
        "sss-1~insan_destegi-choice": "Yönlendirme gereksiz",
    }
    gold = {"sss-1~hassasiyet-noul": {"answer": "true"},
            "sss-1~insan_destegi-choice": {"answer": "Acil yönlendirme"}}  # fmt: skip
    report = R.check(gold, owner)
    assert (report["agree"], report["of"]) == (1, 2)
    assert report["per_question"]["hassasiyet-noul"] == {"agree": 1, "of": 1}
