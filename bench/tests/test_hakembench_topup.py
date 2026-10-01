"""The v1.0 top-up: placeholders kept, brands anchored, exams whole, halves by unit."""

from bench.hakembench import topup as U
from bench.hakembench.common import unit_halves


def test_placeholders_and_plain_greetings_survive_the_mask():
    text = "Merhaba, geçen hafta [şirket] Eczanesi'nden aldığım ilacı [ilçe] şubesine götürdüm."
    assert U.mask(text) == text
    assert U.screen(text, "list") is None
    assert U.mask("Beni 0555 000 00 00 numarasından arayın") == "Beni [telefon] numarasından arayın"
    assert (
        U.mask("[şirket]'nın yeni menüsü burada, bakınız.")
        == "[şirket]'in yeni menüsü burada, bakınız."
    )


def test_unknown_brackets_and_real_brands_are_dropped_but_plain_words_are_not():
    assert (
        U.screen("Sayın [müşteri adı], siparişiniz yola çıktı efendim.", "list")
        == "unknown bracket"
    )
    assert U.screen("Garanti BBVA hesabınız askıya alındı, hemen giriş yapın.", "list").startswith(
        "real"
    )
    assert U.screen("Ürünün garanti süresi iki yıl, faturayı getirin lütfen.", "list") is None
    assert U.screen("Emniyet kemeri takılmadan araç teslim edilmez efendim.", "list") is None


def test_an_exam_loses_all_four_answers_when_one_fails():
    request = {"id": "r", "kind": "ders:Fizik", "shape": "exam"}
    good = {
        "soru": "Serbest düşmede hız nasıl değişir?",
        "cevaplar": [
            {"seviye": k, "metin": f"cevap numarası {k} burada yazıyor"} for k in range(4)
        ],
    }
    rows, _ = U.texts_of(request, good)
    assert [r["level"] for r in rows] == [0, 1, 2, 3]
    bad = {**good, "cevaplar": good["cevaplar"][:3] + [{"seviye": 3, "metin": "Garanti BBVA"}]}
    rows, why = U.texts_of(request, bad)
    assert rows == [] and sum(why.values()) == 1


def test_new_units_balance_against_what_each_half_already_holds():
    units = {f"e{i}": ("ders:Fizik", 4) for i in range(5)}
    out = unit_halves(units, held={"ders:Fizik": {"public": 12, "private": 8}}, global_ties=True)
    sides = sorted(out.values())
    assert sides.count("private") == 3 and sides.count("public") == 2


def test_global_ties_stop_odd_strata_from_all_leaning_public():
    units = {f"{k}{i}": (k, 10) for k in "abcd" for i in range(5)}
    lean = sorted(unit_halves(units).values()).count("public")
    even = sorted(unit_halves(units, global_ties=True).values()).count("public")
    assert lean == 12 and even == 10


def test_the_plan_uses_only_new_topics_and_the_placeholder_rule():
    requests = U.plan()
    per = {
        t: sum(r["track"] == t for r in requests) for t in ("egitim", "moderasyon", "sss", "hukuk")
    }
    assert per == {"egitim": 40, "moderasyon": 20, "sss": 32, "hukuk": 12}
    assert all(U.PLACEHOLDER_RULE in r["prompt"] for r in requests)
    assert len({r["id"] for r in requests}) == len(requests)


def test_gold_rule_lets_the_panel_stand_against_the_writer_family():
    d = U.decide
    assert d(["true"] * 3, ["true", "true"], "offenseval_tr", "moderasyon", None, None) == (
        "true",
        "ai_ensemble_unanimous",
    )
    assert d(["true"] * 3, ["false", "false"], "offenseval_tr", "moderasyon", None, None) == (
        "true",
        "llm_panel_unanimous",
    )
    assert d(["true", "false", "true"], ["false", "false"], "offenseval_tr", "moderasyon",
             None, None) == (None, "needs adjudication")  # fmt: skip
    assert d(["true", "false", "true"], ["false", "false"], "offenseval_tr", "moderasyon",
             None, "false") == ("false", "writer_family_adjudicated")  # fmt: skip
    # Support questions all take the owner's reading, even when every vote agrees.
    assert d(["true"] * 3, ["true", "true"], "hassasiyet-noul", "sss", None, None) == (
        None,
        "needs owner reading",
    )
    assert d(["true"] * 3, ["true", "true"], "hassasiyet-noul", "sss", "false", None) == (
        "false",
        "ai_owner_reading",
    )


def test_the_judges_vote_separately_and_a_tie_is_no_vote():
    meta = {"votes": {"q": {"judges": {"A": {"true": 0.9, "false": 0.1},
                                       "B": {"true": 0.5, "false": 0.5},
                                       "C": {"true": 0.2, "false": 0.8}}}}}  # fmt: skip
    assert U.judge_answers(meta, "q") == ["true", None, "false"]


def test_the_audit_takes_every_overruled_question_and_leaves_knowledge_questions_out():
    gold = {f"m{i}~offenseval_tr": {"answer": "true", "source": "ai_ensemble_unanimous"}
            for i in range(50)}  # fmt: skip
    gold |= {f"e{i}~cevap_puani": {"answer": "2", "source": "ai_ensemble_unanimous"}
             for i in range(50)}  # fmt: skip
    over = ["x1~offenseval_tr", "x2~cevap_puani"]
    sample = U.audit_sample(gold, over)
    assert sample["overruled"] == ["x1~offenseval_tr"]
    assert len(sample["unanimous"]) == 30
    assert not any(k.endswith("cevap_puani") for k in sample["unanimous"])
