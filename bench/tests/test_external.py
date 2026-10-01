"""The external evaluation's mapping, batching and scoring, without the network."""

import json

import numpy as np
import pytest

from bench import external as ext
from bench.harness.items import Item
from data.licenses import ALLOWED_LICENSE_IDS, deny_markers


def mmlu_row(**over):
    row = {"question_id": 7, "question": " Hangisi asal sayıdır? ",
           "options": "['4', '6', '7', '9']", "answer": "C", "answer_index": 2,
           "category": "math"}  # fmt: skip
    return {**row, **over}


def test_mmlu_pro_row_is_one_lettered_choice_question():
    built = ext.mmlu_pro_item(mmlu_row())
    q = built.item.questions["q"]
    assert q.type == "choice" and q.instructions == "Doğru cevap hangisi?"
    assert list(q.criteria) == ["A) 4", "B) 6", "C) 7", "D) 9"]
    assert all(v is None for v in q.criteria.values())
    assert built.item.gold == {"q": "C) 7"}
    assert built.item.state == "Hangisi asal sayıdır?"
    assert built.item.id == "mmlu_pro_tr:7" and built.group == "math"
    # The overlap check reads the question and every option.
    assert built.text.splitlines() == ["Hangisi asal sayıdır?", "4", "6", "7", "9"]


def test_equal_option_texts_stay_apart_and_ten_options_fit():
    built = ext.mmlu_pro_item(mmlu_row(options=str(["x"] * 10), answer="J", answer_index=9))
    assert len(built.item.questions["q"].criteria) == 10
    assert built.item.gold["q"] == "J) x"


@pytest.mark.parametrize(
    "over, message",
    [({"answer": "B"}, "disagree"), ({"options": "nope"}, "unreadable"),
     ({"answer_index": 5, "answer": "F"}, "outside")],
)  # fmt: skip
def test_bad_source_rows_are_refused_not_repaired(over, message):
    with pytest.raises(ValueError, match=message):
        ext.mmlu_pro_item(mmlu_row(**over))


def test_turkish_mmlu_row_uses_its_row_number_and_subject():
    row = {"bolum": "KPSS", "konu": "Tarih", "soru": "Soru?", "cevap": 1,
           "secenekler": ["a", "b", "c", "d", "e"], "aciklama": "x"}  # fmt: skip
    built = ext.turkish_mmlu_item(row, 41)
    assert built.item.id == "turkish_mmlu:41" and built.group == "Tarih"
    assert built.item.gold == {"q": "B) b"}
    assert len(built.item.questions["q"].criteria) == 5


def test_instrument_items_ask_exactly_the_training_question():
    from data.typed.instrument import MASSIVE_OPTIONS, MIDE22_OPTIONS, _question

    off = ext.instrument_item("offenseval_tr", {"id": "o1", "text": "t", "label": 1},
                              ["NOT", "OFF"])  # fmt: skip
    assert (
        off.item.questions["q"].model_dump(exclude_none=True)
        == _question("offenseval_tr", ["NOT", "OFF"])[0]
    )
    assert off.item.gold == {"q": True} and off.group == "OFF"
    labels = sorted(MASSIVE_OPTIONS)
    mas = ext.instrument_item("massive_tr", {"id": "m1", "text": "alarm kur", "label": 2}, labels)
    q = mas.item.questions["q"]
    assert q.instructions == "Kullanıcı sesli asistandan ne istiyor?"
    assert len(q.criteria) == 59  # the whole intent set, answered in passes of ten
    assert mas.item.gold == {"q": MASSIVE_OPTIONS[labels[2]]}
    mide = ext.instrument_item("mide22", {"id": "d1", "text": "t", "label": 0},
                               ["False", "Other", "True"])  # fmt: skip
    assert mide.item.gold == {"q": MIDE22_OPTIONS["False"][0]}
    assert (
        mide.item.questions["q"].model_dump()["criteria"]
        == _question("mide22", ["False", "Other", "True"])[0]["criteria"]
    )


def test_xcopa_asks_for_the_cause_or_the_effect():
    row = {"premise": "Ürün paketlenmişti.", "choice1": "Kırılgandı.", "choice2": "Küçüktü.",
           "question": "effect", "label": 1, "idx": 3}  # fmt: skip
    built = ext.xcopa_item(row)
    q = built.item.questions["q"]
    assert q.instructions == "Bunun sonucu hangisi?"
    assert list(q.criteria) == ["Kırılgandı.", "Küçüktü."]
    assert built.item.gold == {"q": "Küçüktü."}
    cause = ext.xcopa_item({**row, "question": "cause"})
    assert cause.item.questions["q"].instructions == "Bunun sebebi hangisi?"


def test_xfact_follows_cetvels_label_map_and_drops_broken_rows():
    assert ext.xfact_gold("true") == "doğru"
    assert ext.xfact_gold("false") == "yanlış"
    assert ext.xfact_gold("complicated/hard to categorise") == "karmaşık ya da sınıflandırması zor"
    for other in ("mostly true", "mostly false", "partly true/misleading", "other"):
        assert ext.xfact_gold(other) == "kısmen doğru ya da yanıltıcı"
    text = ("language\tclaim\tlabel\n" "tr\tBir iddia.\ttrue\n" "en\tA claim.\tfalse\n"
            "tr\tbroken\trow\twith a tab\n" "tr\tBaşka iddia.\tmostly true\n")  # fmt: skip
    rows = ext.xfact_rows(text)
    assert [r["claim"] for r in rows] == ["Bir iddia.", "Başka iddia."]
    built = ext.xfact_item(rows[1], 1)
    assert built.item.gold == {"q": "kısmen doğru ya da yanıltıcı"}
    assert len(built.item.questions["q"].criteria) == 4


def test_every_set_but_the_owners_exception_is_on_the_allow_list():
    for name, spec in ext.SETS.items():
        if name == "turkish_mmlu":
            assert spec.licence_id is None and spec.gated
            assert deny_markers(spec.licence_as_shown)  # NC and ND stay visible
            continue
        assert spec.licence_id in ALLOWED_LICENSE_IDS, name
        assert not deny_markers(spec.licence_as_shown), name
    inside = {task for task, row in ext.CETVEL_TASKS.items() if row["verdict"].startswith("in")}
    assert inside == {"offenseval_tr", "xcopa_tr", "xfact_tr"}
    assert inside <= set(ext.SETS)
    assert set(ext.READERS) == set(ext.SETS)


def test_build_refuses_bad_rows_drops_overlap_and_writes_ids_only(tmp_path, monkeypatch):
    built_root = tmp_path / "built"
    (built_root / "typed").mkdir(parents=True)
    seen = "Bu cümle eğitim verisinde aynen geçiyor ve bu yüzden sınavdan düşmeli."
    (built_root / "typed" / "train.jsonl").write_text(
        json.dumps({"row_id": "r1", "state": seen}) + "\n", encoding="utf-8"
    )
    fresh = "Tamamen yeni bir öncül cümlesi, eğitim verisinde hiç bulunmuyor elbette."

    def rows(spec):
        yield lambda: ext.xcopa_item(
            {
                "premise": seen,
                "choice1": "a",
                "choice2": "b",
                "question": "cause",
                "label": 0,
                "idx": 1,
            }
        )
        yield lambda: ext.xcopa_item(
            {
                "premise": fresh,
                "choice1": "c",
                "choice2": "d",
                "question": "effect",
                "label": 1,
                "idx": 2,
            }
        )
        yield lambda: ext.xcopa_item({"premise": "p", "choice1": "e", "choice2": "e",
                                      "question": "effect", "label": 1, "idx": 3})  # fmt: skip

    monkeypatch.setitem(ext.READERS, "xcopa_tr", rows)
    record = ext.build("xcopa_tr", built_root=built_root, cache=tmp_path / "cache",
                       results=tmp_path / "results")  # fmt: skip
    assert record["rows_read"] == 3 and len(record["rows_refused"]) == 1
    assert record["overlap"]["items_dropped"] == 1 and "xcopa_tr:1" in record["overlap"]["dropped"]
    assert record["items_kept"] == 1
    items, groups = ext.load_built("xcopa_tr", tmp_path / "cache")
    assert [i.id for i in items] == ["xcopa_tr:2"] and groups == {"xcopa_tr:2": "effect"}
    written = (tmp_path / "results" / "xcopa_tr.build.json").read_text(encoding="utf-8")
    assert fresh not in written and seen not in written


def test_batches_cover_every_job_once_within_the_limits():
    jobs = [ext.Job(i, 0, None, tokens) for i, tokens in enumerate([5, 90, 12, 40, 40, 7, 64])]
    plan = ext.batches(jobs, max_tokens=100, max_rows=3)
    assert sorted(j.index for b in plan for j in b) == list(range(7))
    for b in plan:
        assert len(b) <= 3
        assert max(j.tokens for j in b) * len(b) <= 100 or len(b) == 1


def test_scoring_reports_accuracy_baseline_and_groups():
    answers = [(0, [0.7, 0.1, 0.1, 0.1]), (1, [0.2, 0.6, 0.1, 0.1]),
               (0, [0.4, 0.3, 0.2, 0.1]), (0, [0.9, 0.05, 0.03, 0.02])]  # fmt: skip
    rows = [{"id": f"s:{i}", "type": "choice", "k": 4, "gold": 0, "predicted": p,
             "correct": p == 0, "probabilities": probs, "expected_error": 0.3}
            for i, (p, probs) in enumerate(answers)]  # fmt: skip
    groups = {"s:0": "a", "s:1": "a", "s:2": "b", "s:3": "b"}
    out = ext.summarise(rows, groups, with_f1=False, draws=50)
    m = out["metrics"]
    assert m["accuracy"]["value"] == pytest.approx(0.75)
    expected_brier = np.mean([(0.3**2 + 0.03), (0.8**2 + 0.36 + 0.02), (0.36 + 0.09 + 0.05),
                              (0.01 + 0.0025 + 0.0009 + 0.0004)])  # fmt: skip
    assert m["brier"]["value"] == pytest.approx(expected_brier)
    assert "macro_f1" not in m and len(m["accuracy"]["interval"]) == 2
    assert out["random_baseline_accuracy"] == pytest.approx(0.25)
    assert out["per_group_accuracy"]["a"]["accuracy"] == 0.5
    assert out["per_group_accuracy"]["b"]["n"] == 2


def test_wilson_interval():
    low, high = ext.wilson(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-3) and high == pytest.approx(0.9433, abs=1e-3)
    assert ext.wilson(0, 0) is None


# The batched answers are the adapter's answers. ------------------------------------------

torch = pytest.importorskip("torch")


def test_batched_answers_equal_the_adapter_one_question_at_a_time():
    from bench.adapters.karar import Calibration, KararAdapter
    from model.head.tests.test_head import encode, head
    from schema.api import Request

    model = head(causal=True)
    cal = Calibration(temperature=1.3, map_x=(0.2, 0.9), map_y=(0.8, 0.1),
                      by_type=(("choice", 1.2), ("noul", 1.5)))  # fmt: skip
    many = {f"seçenek {i:02d}": None for i in range(12)}  # two passes, one option borrowed
    questions = [
        {"type": "choice", "instructions": "Hangisi?", "criteria": many},
        {"type": "choice", "instructions": "Hangi birim?",
         "criteria": {"teknik destek": "Arızalar", "fatura": None, "satış": None}},
        {"type": "noul", "instructions": "Acil mi?"},
    ]  # fmt: skip
    golds = ["seçenek 03", "fatura", False]
    items = [Item(id=f"t:{i}", track="t", state=f"Durum {i}: üç gündür internet yok.",
                  questions={"q": q}, gold={"q": g})
             for i, (q, g) in enumerate(zip(questions, golds, strict=True))]  # fmt: skip
    rows, refused, progress = ext.answer_items(model, encode, cal, items, max_tokens=4096)
    assert not refused and progress.done_tokens == progress.total_tokens
    adapter = KararAdapter(model=model, encode=encode, calibration=cal, revision="test")
    for item, row in zip(items, rows, strict=True):
        request = Request.model_validate({"state": item.state, "model": "x",
                                          "questions": item.questions})  # fmt: skip
        answer = adapter.answer(request).response.answers["q"]
        if answer.type == "noul":
            expected = [answer.noul, 1 - answer.noul]
        else:
            expected = [answer.probabilities[k] for k in item.questions["q"].criteria]
        assert row["probabilities"] == pytest.approx(expected, abs=1e-5)
        assert 1 - row["expected_error"] == pytest.approx(
            answer.confidence if answer.type != "noul" else 1 - answer.abstain, abs=1e-5
        )
    assert rows[2]["gold"] == 1  # noul outcomes are (true, false); gold False is index 1
    assert rows[0]["k"] == 12


def test_passes_match_the_served_chunking():
    from model.head.tests.test_head import encode

    question = ext.QUESTION.validate_python(
        {"type": "choice", "instructions": "?", "criteria": {f"o{i:02d}": None for i in range(21)}}
    )
    parts = ext.passes(question, encode)
    assert [len(p.criteria) for p in parts] == [10, 10, 2]  # the last pass borrows one
    assert set().union(*(set(p.criteria) for p in parts)) == set(question.criteria)
    small = ext.QUESTION.validate_python({"type": "noul", "instructions": "?"})
    assert ext.passes(small, encode) == [small]
