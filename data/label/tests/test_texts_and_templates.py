"""Personal data is masked, and a flawed template never reaches a judge."""

import json
from collections import Counter

from data.label.generate import generator_messages, parse_template, to_question
from data.label.texts import faq_text, mask_personal_data
from schema.rows import outcomes


def test_personal_data_is_masked():
    text = ("Bize destek@example.invalid adresinden, 0555 000 00 00 ya da +90 (212) 555 12 12 "
            "numarasından ulaşın. IBAN: TR12 0006 1005 1978 6457 8413 26. TC 12345678901. "
            "Ayrıntı: https://firma.com/sss")  # fmt: skip
    masked = mask_personal_data(text)
    for leak in ("destek@", "0555", "555 12", "TR12", "12345678901", "https://"):
        assert leak not in masked, leak
    for tag in ("[e-posta]", "[telefon]", "[iban]", "[kimlik]", "[bağlantı]"):
        assert tag in masked
    # Ordinary numbers survive: prices, years, quantities.
    assert mask_personal_data("Fiyat 1.250 TL, 2024 modeli, 4.000 mAh") == (
        "Fiyat 1.250 TL, 2024 modeli, 4.000 mAh"
    )


def test_a_faq_pair_becomes_a_state_or_is_dropped():
    answer = {"text": "Siparişiniz 2-3 iş günü içinde kargoya verilir.", "is_accepted": True}
    row = {"name": "Kargo ne zaman gelir?", "answers": [answer]}
    assert faq_text(row).startswith("Soru: Kargo ne zaman gelir?\nCevap: Siparişiniz")
    assert faq_text({"name": "", "answers": []}) is None
    assert faq_text({"name": "Kısa?", "answers": [{"text": "Evet."}]}) is None


def good(**overrides):
    template = {
        "family": "konu", "type": "choice",
        "question": "Bu soru hangi konuyla ilgili?",
        "applies_when": "Bir şirketin sıkça sorulan sorular sayfasındaki her soru",
        "options": [{"name": "ödeme", "description": "Ödeme ve fatura"},
                    {"name": "teslimat", "description": "Kargo ve teslim süresi"},
                    {"name": "ürün özellikleri", "description": "Ürünün teknik bilgileri"},
                    {"name": "iade", "description": "İade ve değişim işlemleri"}],
    }  # fmt: skip
    return json.dumps({**template, **overrides}, ensure_ascii=False)


def test_a_clean_template_passes_and_becomes_a_typed_question():
    parsed = parse_template(good())
    assert parsed.template is not None, parsed.problems
    question = to_question(parsed.template)
    assert question["type"] == "choice" and len(question["criteria"]) == 4
    from pydantic import TypeAdapter

    from schema.questions import Question

    typed = TypeAdapter(Question).validate_python(question)
    assert outcomes(typed) == ["ödeme", "teslimat", "ürün özellikleri", "iade"]


def test_each_flaw_is_caught():
    none_of = good(options=[{"name": "ödeme"}, {"name": "teslimat"}, {"name": "hiçbiri"}])
    assert "other options" in " ".join(parse_template(none_of).problems)
    absolute = good(options=[{"name": "ödeme"}, {"name": "teslimat"}, {"name": "asla cevaplanmaz"}])
    assert "absolute" in " ".join(parse_template(absolute).problems)
    long = good(options=[{"name": "a"}, {"name": "b"},
                         {"name": "c", "description": "çok uzun bir açıklama " * 5}])  # fmt: skip
    assert "longer" in " ".join(parse_template(long).problems)
    leak = good(question="Bu soru teslimat ile mi ilgili, yoksa başka bir konuyla mı?")
    assert "named in the question" in " ".join(parse_template(leak).problems)
    assert parse_template("bu JSON değil").problems[0].startswith("not JSON")
    assert parse_template("```json\n" + good() + "\n```").template is not None
    too_few = good(options=[{"name": "a"}, {"name": "b"}])
    assert parse_template(too_few).template is None


def test_yes_no_and_score_templates():
    noul = {"family": "insan_destegi", "type": "noul", "applies_when": "Müşteri soruları"}
    noul |= {"question": "Bu soru bir temsilciye aktarılmalı mı?"}
    noul |= {"true": "Otomatik cevap yetmez", "false": "Cevap yeterli"}
    noul = parse_template(json.dumps(noul))
    assert noul.template is not None
    assert to_question(noul.template)["criteria"] == {
        "true": "Otomatik cevap yetmez",
        "false": "Cevap yeterli",
    }
    score = {"family": "aciliyet", "type": "score", "question": "Bu soru ne kadar acil?"}
    score |= {"applies_when": "Müşteri soruları", "levels": ["acil değil", "biraz acil", "acil"]}
    score = parse_template(json.dumps(score))
    assert score.template is not None and len(to_question(score.template)["criteria"]) == 3


def test_the_generator_request_names_the_family_and_type():
    messages = generator_messages("konu", "choice", ["Soru: a\nCevap: b"] * 3)
    user = messages[-1]["content"]
    assert "Örnek 3" in user and "Soru türü: choice" in user and "hiçbiri" in user


def test_option_names_must_be_natural_turkish():
    snake = good(options=[{"name": "ucret_veya_maliyet"}, {"name": "teslimat"},
                          {"name": "iade"}])  # fmt: skip
    assert any("not natural Turkish" in p for p in parse_template(snake).problems)


def test_family_and_type_come_from_the_request():
    renamed = parse_template(good(family="odeme_konusu"), "konu", "choice")
    assert renamed.template is not None and renamed.template.family == "konu"
    other_type = parse_template(good(), "konu", "score")
    assert other_type.template is None and "asked for score" in other_type.problems[0]


def test_forum_underlines_are_removed():
    row = {
        "name": "Bebek neden uyumaz?\n==========\n\n",
        "answers": [{"text": "Gündüz fazla uyuyorsa gece uyanık kalabilir, düzen önemli."}],
    }
    assert faq_text(row) == ("Soru: Bebek neden uyumaz?\nCevap: Gündüz fazla uyuyorsa gece "
                             "uyanık kalabilir, düzen önemli.")  # fmt: skip


def test_a_template_is_screened_on_its_first_rows():
    from data.label.pilot import screen_verdict
    from schema.rows import TrainingRow

    def row(top):
        return TrainingRow(
            track="sss", task="t", split="train", origin="generated", label_kind="rule",
            source="s", state="metin " + top, recipe="r",
            question={"type": "noul", "instructions": "Acil mi?",
                      "criteria": {"true": "acil", "false": "acil değil"}},
            target={"true": 1.0, "false": 0.0} if top == "true" else {"true": 0.0, "false": 1.0},
        )  # fmt: skip

    assert screen_verdict([row("true")]) == "rarely_applies"
    assert screen_verdict([row("false")] * 9 + [row("true")]) == "low_variance"
    assert screen_verdict([row("false")] * 5 + [row("true")] * 3) == "passed"


def test_the_rate_limit_reset_is_read_from_the_header():
    from data.hub import reset_after

    assert reset_after('"api";r=0;t=42') == 42
    assert reset_after(None) is None and reset_after('"api";r=3') is None


def test_the_sampler_spreads_over_row_groups_and_caps_sites(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from data.label import texts

    answer = [{"text": "Bu sorunun cevabı yeterince uzun bir açıklama içeriyor.",
               "name": "", "is_accepted": True}]  # fmt: skip
    rows = [{"id": str(i), "name": f"Soru {i} nasıl yapılır?", "domain": f"site{i // 50}",
             "answers": answer} for i in range(1000)]  # fmt: skip
    path = tmp_path / "0000.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=100)
    monkeypatch.setattr(texts, "shards", lambda config, cache: [path])
    sample = texts.sample_faq(30, per_domain=3, seed=1)
    assert len(sample) == 30
    assert max(Counter(r["domain"] for r in sample).values()) <= 3
    assert len({int(r["source_id"]) // 100 for r in sample}) >= 8  # many row groups


def test_seo_pages_are_dropped():
    echo = "şu anda cayman adaları için en çok rezerve edilen araç kiralama"
    assert faq_text({"name": echo + " .", "answers": [{"text": echo + "."}]}) is None
    answer = "Buradan bakabilirsiniz: [kiralama](https://ornek.com) sayfası."
    link = {"name": "Araç kiralama nerede yapılır, fiyatlar nedir?", "answers": [{"text": answer}]}
    assert faq_text(link) is None


def test_blocked_sources_markup_link_answers_and_other_languages_are_dropped():
    good = {"text": "Siparişiniz 2-3 iş günü içinde kargoya verilir."}
    base = {"name": "Kargo ne zaman gelir?", "answers": [good], "domain": "firma.com"}
    assert faq_text(base)
    assert faq_text(base | {"domain": "www.sikayetvar.com"}) is None
    assert faq_text(base | {"name": "Boom Casino'da hangi oyunlar var?"}) is None
    for brand in ("tipobet affiliate", "deneme bonusu", "cepbahis adresi ne oldu?"):
        assert faq_text(base | {"name": brand}) is None, brand
    for word in ("Diyabet hastası ne yemeli?", "Sohbet odası ücretli mi?"):
        assert faq_text(base | {"name": word}), word
    assert faq_text(base | {"name": "<span>Kargo</span> ne zaman gelir?"}) is None
    assert faq_text(base | {"answers": [{"text": "Bakınız: https://firma.com/kargo"}]}) is None
    english = {"name": "how do you say this in turkish? do you love me",
               "answers": [{"text": "beni seviyor musun diye sorulur"}]}  # fmt: skip
    assert faq_text(base | english) is None
    # A short answer with no link is a real answer, as on quiz sites.
    hardware = {"name": "Bu anakartta kaç ram slotu var?",
                "answers": [{"text": "İki adet DDR4 ram slotu bulunur."}]}  # fmt: skip
    assert faq_text(base | hardware)
    quiz = {"name": "Hangi seçenekte tamamı otçul canlılar vardır?",
            "answers": [{"text": "keçi, koyun, inek"}]}  # fmt: skip
    assert faq_text(base | quiz)


def test_loosely_grouped_phones_names_by_address_and_handles_are_masked():
    masked = mask_personal_data(
        "Tel: 0-212- 444 0 678. Merhaba Esra, kemal bey ve seval hanım. başak merhaba, "
        "sayın seçer, merhaba, tolga. @sekerfare21 yazdı"
    )
    for leak in ("444 0 678", "Esra", "kemal", "seval", "başak", "seçer", "tolga", "sekerfare"):
        assert leak not in masked, leak
    for plain in ("yıl 2021, 1500 kişi, fiyat 70 tl", "2024 12 12 12 30", "herkese merhaba",
                  "Merhaba arkadaşlar, nasılsınız", "Sayın Yetkili"):  # fmt: skip
        assert mask_personal_data(plain) == plain


def test_intent_options_must_be_actions_and_options_must_not_overlap():
    def template(family, names):
        return json.dumps({"family": family, "type": "choice", "question": "Amaç nedir acaba?",
                           "applies_when": "Her soru metni.",
                           "options": [{"name": n} for n in names]})  # fmt: skip

    topics = parse_template(template("niyet", ["ücret ve maliyet", "kural ve prosedür",
                                               "teknik sorun ve çözüm"]))  # fmt: skip
    assert "intent options are topics, not actions" in topics.problems
    actions = parse_template(template("niyet", ["fiyat öğrenmek", "şikayet etmek",
                                                "iade istemek"]))  # fmt: skip
    assert actions.template is not None
    overlapping = parse_template(template("konu", ["Kargo teslim süresi", "Kargo teslim süreci",
                                                   "Fatura"]))  # fmt: skip
    assert any(p.startswith("options overlap") for p in overlapping.problems)


def test_cut_questions_answered_by_a_pointer_are_dropped():
    from data.label.texts import blocked, faq_text

    cut = "merhaba [ad] bey oğlumun astım tedavisi"
    pointer = (cut + " prof. dr. ad soyad doktorumuzun Çocuk göğüs hastalıkları branşındaki"
               " cevaplarını görüntüleyin.")  # fmt: skip
    assert blocked({}, cut, pointer)
    seo = "başakşehir evden eve nakliyat hizmeti nedir"
    call = seo + " sorusuna doğru ve hızlı cevap almak için hemen firmamızı arayın"
    assert blocked({}, seo + "?", call)
    echo = "diş fırçalarken su kaçarsa oruç bozulur mu"
    assert blocked({}, echo, echo + " evet bozar")
    real = echo + ". Su boğaza kaçmadıysa oruç bozulmaz; ağzı çalkalayıp tükürmek yeterlidir."
    assert not blocked({}, echo + " ?", real)
    answer = "Bu yıl on dört gün yıllık izin hakkınız var, ocakta başlıyor."
    question = "İ".lower() + "zin günlerim ne zaman başlıyor, bu yıl kaç gün kullanırım?"
    row = {"name": question, "answers": [{"text": answer}]}
    assert "\u0307" not in faq_text(row)


def test_adult_sites_are_dropped_and_sexual_health_questions_are_not():
    from data.label.texts import blocked

    q = "### Hesap özetimde ödeme nasıl görünür?"
    assert blocked({}, q, "Bu sebeple Free Live Sex Cams ile yapılan işlemler SegPay görünür.")
    assert blocked({}, "karşıyaka escort var mı?", "ilanlarımızda tüm ilçelerde escort bulunur")
    health = "Kanser hücresi öpüşmeyle veya cinsel ilişkiyle bulaşır mı?"
    assert not blocked({}, health, "Hayır, kanser bulaşıcı bir hastalık değildir, merak etmeyin.")
    assert not blocked(
        {}, "Pornografi Bağımlılığı", "Bir terapist ile birlikte çalışmanızı öneririm."
    )
