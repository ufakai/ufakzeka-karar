"""Parliament candidates: splitting, filters, quotas, the verbatim check, the seed, the split."""

import datetime as dt
import hashlib
from collections import Counter

import pytest

from bench.hakembench import parliament as p
from data.label.texts import mask_personal_data


def turn(i, party="A", month=1, year=2020, source=p.TRANSCRIPT, text=None):
    text = text or (
        f"Bu yıl {i} numaralı bölgede tarım destekleri yüzde {i} oranında azaldı ve çiftçiler "
        f"zor durumda kaldı. Ayrıca {i} köyde sulama kanalı hâlâ yapılmadı ve üretim düştü."
    )
    return {"file_id": f"f{i // 10}", "speech_id_within_doc": i,
            "tarih": dt.datetime(year, month, 1 + i % 27), "tbmm_donemi": 27.0,
            "yasama_yili": 2.0, "meclis_turu": "TBMM", "source": source,
            "speaker_name_raw": f"KİŞİ {i}", "speaker_role_raw": "mp", "wiki_name": f"KISI {i}",
            "wiki_parti": party, "speech_text": text}  # fmt: skip


def texts_of(text, paragraphs=True):
    return [text[a:b] for a, b in p.split_sentences(text, paragraphs=paragraphs)]


# splitting -------------------------------------------------------------------------


def test_abbreviations_initials_and_titles_do_not_end_a_sentence():
    text = (
        "Sayın Dr. Ahmet Yılmaz ve Prof. Dr. Ayşe Kaya dün geldi. M. Ali Bey de geldi. "
        "Hz. Ali’nin sözünü andı."
    )
    assert texts_of(text) == [
        "Sayın Dr. Ahmet Yılmaz ve Prof. Dr. Ayşe Kaya dün geldi.",
        "M. Ali Bey de geldi.",
        "Hz. Ali’nin sözünü andı.",
    ]


def test_numbers_ordinals_decimals_and_dates():
    text = (
        "Yargıtay 2. Ceza Dairesi karar verdi. Destekler yüzde 38. Bu çiftçi nasıl geçinecek? "
        "Enflasyon 19.02.2015 tarihinde 3,5 puan arttı. II. Dünya Savaşı bitti."
    )
    assert texts_of(text) == [
        "Yargıtay 2. Ceza Dairesi karar verdi.",
        "Destekler yüzde 38.",
        "Bu çiftçi nasıl geçinecek?",
        "Enflasyon 19.02.2015 tarihinde 3,5 puan arttı.",
        "II. Dünya Savaşı bitti.",
    ]


def test_a_quotation_stays_whole():
    text = "Bakan dedi ki: “Millet memnun değil. Anayasa da öyle.” Getirebiliyor musunuz?"
    assert texts_of(text) == [
        "Bakan dedi ki: “Millet memnun değil. Anayasa da öyle.”",
        "Getirebiliyor musunuz?",
    ]


def test_line_breaks_end_sentences_in_transcripts_only():
    text = "Birinci satır burada biter\nikinci satır burada devam eder."
    assert len(texts_of(text, paragraphs=True)) == 2
    assert texts_of(text, paragraphs=False) == [text]


def test_spans_are_offsets_into_the_turn():
    text = "  Birinci cümle burada. İkinci cümle de burada!  "
    for start, end in p.split_sentences(text, paragraphs=True):
        assert text[start:end] == text[start:end].strip()
        assert text[start:end] in text


def test_a_turn_stops_at_another_speakers_line():
    text = "Korkudan! MUSA ÇAM (Devamla) – Çünkü baskı var. BAŞKAN – Teşekkürler."
    assert text[: p.own_end(text)] == "Korkudan! "
    assert p.own_end("Bakın, AKP iktidarı - dediğim gibi - bunu yaptı.") == len(
        "Bakın, AKP iktidarı - dediğim gibi - bunu yaptı."
    )


# filters -----------------------------------------------------------------------------

SUBSTANTIVE = "Geçen yıl ilaç fiyatları yüzde 37 arttı ve hastalar ilaç bulamıyor."


@pytest.mark.parametrize(
    ("sentence", "reason"),
    [
        ("Sayın Başkan, değerli milletvekilleri; bugün önemli bir kanunu görüşüyoruz.", "opening"),
        ("Değerli milletvekilleri, bu kanun teklifi ülkemiz için çok önemlidir.", "opening"),
        ("Teşekkür ederim Sayın Başkanım, hepinizi de ayrıca kutluyorum burada.", "thanks"),
        ("Ekranları başında bizi izleyen tüm vatandaşlarımızı saygıyla selamlıyorum efendim.",
         "thanks"),
        ("Yeni yasama yılımızın ülkemize ve milletimize hayırlı olsun diliyorum efendim.",
         "greeting"),
        ("Maddeyi oylarınıza sunuyorum, kabul edenler lütfen işaret etsinler efendim.", "vote"),
        ("Sayın milletvekilleri, birleşime on dakika ara veriyorum.", "opening"),
        ("Önergeyi okutuyorum ve ardından söz talebi olan arkadaşlara söz vereceğim.", "chair"),
        ("Bu ülke hepimizin ve hepimiz buradayız (AK PARTİ sıralarından alkışlar) diyoruz.",
         "stage_note"),
        ("Bu konuda söyleyeceğimiz çok şey var TBMM B: 58 19 . 2 . 2015 bunu biliyoruz.",
         "page_header"),
        ("ve bu nedenle bu kanunun geri çekilmesi gerektiğini düşünüyoruz arkadaşlar.", "start"),
        ("Bu kanunla birlikte ilaç fiyatları yüzde 37 arttı ve hastalar ilaç bulamıyor…",
         "unfinished"),
        ("İlaç fiyatları arttı.", "short"),
        (" ".join(["uzun"] * 41).capitalize() + ".", "long"),
        ("Sayın Yılmaz, geçen yıl ilaç fiyatlarının yüzde 37 arttığını söyledi.",
         "names_person"),
        (SUBSTANTIVE, None),
        ("Sayın Bakan, geçen yıl ilaç fiyatları yüzde 37 arttı, ne yapacaksınız?", None),
    ],
)  # fmt: skip
def test_filters(sentence, reason):
    assert p.reject(sentence) == reason


def test_ocr_words_run_together_are_caught_with_a_vocabulary():
    vocab = Counter({w: 10 for w in SUBSTANTIVE.lower().replace(".", "").split()})
    vocab.update({"hizmet": 10, "alanı": 10})
    glued = "Geçen yıl ilaç fiyatları yüzde 37 arttı ve hastalar hizmetalanı bulamıyor."
    assert p.reject(glued, vocab) == "ocr_glued"
    assert p.reject("Geçen yıl ilaç fiyatları yüzde 37 arttı ve6 hastalar bulamıyor.", vocab) == (
        "ocr_glued"
    )
    assert p.reject(SUBSTANTIVE, vocab) is None
    assert p.reject(glued) is None  # transcripts are not checked


def test_masks_follow_mask_personal_data_except_titles():
    text = "Bana ahmet@example.invalid adresinden ya da 0555 000 00 00 numarasından ulaşın."
    assert p.mask_names(text) == mask_personal_data(text)
    assert p.mask_names("Sayın Yılmaz geldi.") == mask_personal_data("Sayın Yılmaz geldi.")
    assert p.mask_names("Sayın Bakan ve Bakan Bey geldi.") == "Sayın Bakan ve Bakan Bey geldi."
    assert mask_personal_data("Sayın Bakan geldi.") != "Sayın Bakan geldi."
    assert p.names_person("Ahmet Bey dün buradaydı.")
    assert not p.names_person("Sayın Bakan dün buradaydı.")


# quotas ------------------------------------------------------------------------------


def test_party_cap_moves_the_excess_to_the_others():
    quotas = p.party_quotas({"A": 60, "B": 25, "C": 15}, 100)
    assert quotas == {"A": 35, "B": 35, "C": 30}
    assert sum(quotas.values()) == 100
    assert p.party_quotas({"A": 30, "B": 70}, 1300)["B"] == 455


def test_a_full_cell_passes_its_share_to_the_others():
    shares = p.allocate({"x": 50, "y": 50}, 10, {"x": 2, "y": 100})
    assert shares == {"x": 2, "y": 8}


def fixture_turns():
    turns = [turn(i, "A", month=1 + i % 12) for i in range(60)]
    turns += [turn(100 + i, "B", month=1 + i % 12) for i in range(25)]
    turns += [turn(200 + i, "C", month=1 + i % 12) for i in range(15)]
    return turns


def test_the_draw_caps_the_largest_party():
    result = p.draw(fixture_turns(), target=20)
    parties = Counter(c["meta"]["party"] for c in result.candidates)
    assert len(result.candidates) == 20
    assert max(parties.values()) <= 0.35 * 20
    assert parties["A"] == 7


# verbatim, seed, split ---------------------------------------------------------------


def test_every_candidate_is_its_turns_sentence():
    turns = fixture_turns()
    turns.append(turn(300, "C", source="turkronicles", text=(
        "Bu yıl bölgede tarım destekleri\nazaldı ve çiftçiler zor durumda kaldı. Ayrıca köyde "
        "sulama kanalı yapılmadı.")))  # fmt: skip
    by_id = {p.turn_id(t): t["speech_text"] for t in turns}
    result = p.draw(turns, target=40)
    for cand in result.candidates:
        p.verify(cand["meta"], cand["text"], by_id[cand["meta"]["turn_id"]])
    cand = result.candidates[0]
    with pytest.raises(AssertionError):
        p.verify(cand["meta"], cand["text"] + " Ek.", by_id[cand["meta"]["turn_id"]])


def test_line_breaks_joined_are_recorded():
    t = turn(1, source="turkronicles", text=(
        "Bu yıl bölgede tarım destekleri\nazaldı ve çiftçiler zor durumda kaldı."))  # fmt: skip
    cand = p.candidate(t, p.eligible(t), p.SEED)
    assert cand["text"] == "Bu yıl bölgede tarım destekleri azaldı ve çiftçiler zor durumda kaldı."
    assert cand["meta"]["line_breaks_joined"]
    p.verify(cand["meta"], cand["text"], t["speech_text"])


def test_the_draw_is_seeded():
    ids = [c["id"] for c in p.draw(fixture_turns(), target=20).candidates]
    again = [c["id"] for c in p.draw(list(reversed(fixture_turns())), target=20).candidates]
    other = [c["id"] for c in p.draw(fixture_turns(), target=20, seed=2).candidates]
    assert ids == again
    assert ids != other


def test_an_overlapping_candidate_is_replaced_from_the_reserve():
    first = p.draw(fixture_turns(), target=20)
    hit = first.candidates[0]["id"]
    result = p.draw(fixture_turns(), target=20, overlapping=lambda cands: {hit: "overlap_x"})
    assert len(result.candidates) == 20
    assert hit not in {c["id"] for c in result.candidates}
    assert result.dropped == [{"id": hit, "turn_id": first.candidates[0]["meta"]["turn_id"],
                               "reason": "overlap_x"}]  # fmt: skip


def test_split_is_the_parity_of_the_text_hash():
    for text in ("birinci", "ikinci", "üçüncü"):
        parity = int(hashlib.sha256(text.encode()).hexdigest(), 16) % 2
        assert p.split_of(text) == ("public" if parity == 0 else "private")
    splits = Counter(p.split_of(f"cümle {i}") for i in range(2000))
    assert 900 < splits["public"] < 1100


def test_items_and_meta_are_written(tmp_path):
    from bench.harness.items import load_items

    result = p.draw(fixture_turns(), target=20)
    items_path, meta_path = p.write(result, tmp_path)
    items = load_items(items_path)
    assert len(items) == 20
    assert all(i.gold == {} and set(i.questions) == {"dogrulanabilir", "kontrol_onceligi"}
               for i in items)  # fmt: skip
    assert items[0].questions["dogrulanabilir"].model_dump(exclude_none=True) == {
        "type": "noul", **{k: v for k, v in p.CHECKABLE.items() if k != "type"}}  # fmt: skip
    assert items[0].questions["kontrol_onceligi"].criteria == p.PRIORITY["criteria"]
    assert len(meta_path.read_text().splitlines()) == 20


# the file ----------------------------------------------------------------------------


def test_population_reads_a_parquet_and_keeps_current_plenary_turns(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [turn(1), turn(2, year=2009), {**turn(3), "meclis_turu": "DANISMA"},
            {**turn(4), "wiki_parti": None}]  # fmt: skip
    path = tmp_path / "core.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    counts = Counter()
    kept = p.population(p.read_turns(path), counts)
    assert [t["speech_id_within_doc"] for t in kept] == [1]
    assert counts == Counter({"rows_read": 4, "before_2011": 1, "not_plenary_chamber": 1,
                              "no_speaker_or_party": 1, "population_turns": 1})  # fmt: skip
