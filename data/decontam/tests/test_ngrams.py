"""Normalisation, the 8-gram rule, the short-text path and MinHash near-duplicates."""

import numpy as np
import pytest

from data.decontam.ngrams import (
    OVERLAP_THRESHOLD,
    MinHashLSH,
    NgramIndex,
    estimated_jaccard,
    gram_hashes,
    minhash_signature,
    ngram_set,
    normalise,
    optimal_bands,
    tokens,
    whole_hash,
)
from data.decontam.tests.fixtures import PARAGRAPH, UNRELATED, turkish_upper

# Twenty distinct tokens, so every 8-gram in it is unique.
REFERENCE = (
    "Belediye meclisi dün akşam yaptığı toplantıda şehir merkezindeki eski pazar "
    "yerinin yeniden düzenlenmesi için hazırlanan projeyi oy birliğiyle kabul etti."
)
FILLER = "kedi köpek elma armut kalem defter masa sandalye pencere kapı halı perde"


def test_turkish_casing():
    assert normalise("İSTANBUL") == "istanbul"
    assert normalise("IĞDIR") == "ığdır"
    assert normalise("Işık İçin") == "ışık için"
    # "I" followed by a combining dot composes to "İ" under NFKC first.
    assert normalise("I\u0307zmir") == "izmir"


def test_punctuation_and_spacing():
    text = '  "Merhaba,   dünya!"  (jungkook\'a)  e-posta 3.5 👏 @USER #etiket.  '
    assert normalise(text) == "merhaba dünya jungkook'a e-posta 3.5 user etiket"
    assert tokens("Ali, eve   geldi.") == ["ali", "eve", "geldi"]
    # Full-width letters fold to ASCII under NFKC.
    assert normalise("ＡＢＣ") == "abc"


def test_placeholder_chains_do_not_match_each_other():
    assert tokens("@USER @USER @USER @USER merhaba @USER") == ["user", "merhaba", "user"]
    chain = " ".join(["@USER"] * 20)
    index = NgramIndex.build(
        [("offenseval:1", chain + " Allah kabul etsin, hayırlı olsun kardeşim")]
    )
    fraction, _ = index.overlap(chain + " maç bu akşam saat dokuzda başlıyor değil mi")
    assert fraction == 0.0


def test_ngram_set():
    grams = ngram_set("bir iki üç dört beş altı yedi sekiz dokuz", n=8)
    assert grams == {
        ("bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz"),
        ("iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"),
    }
    assert ngram_set("çok kısa", n=8) == set()


def test_overlap_above_and_below_half():
    assert len(tokens(REFERENCE)) == 20
    index = NgramIndex.build([("ref:1", REFERENCE)])
    ref_tokens = tokens(REFERENCE)
    filler = FILLER.split()

    # Sixteen copied tokens and four new ones: 16 of 20 covered.
    above = " ".join(ref_tokens[:16] + filler[:4])
    fraction, refs = index.overlap(above)
    assert fraction == pytest.approx(0.8)
    assert refs == ["ref:1"]
    assert index.is_contaminated(above)

    # Eight copied tokens and twelve new ones: one shared 8-gram, 8 of 20.
    below = " ".join(ref_tokens[:8] + filler[:12])
    fraction, refs = index.overlap(below)
    assert fraction == pytest.approx(0.4)
    assert fraction <= OVERLAP_THRESHOLD
    assert refs == ["ref:1"]
    assert not index.is_contaminated(below)

    # Seven copied tokens share no 8-gram at all.
    fraction, refs = index.overlap(" ".join(ref_tokens[:7] + filler))
    assert (fraction, refs) == (0.0, [])


def test_overlap_ignores_case_and_punctuation():
    index = NgramIndex.build([("ref:1", REFERENCE)])
    fraction, _ = index.overlap(turkish_upper(REFERENCE).replace(" ", " , "))
    assert fraction == 1.0


def test_short_texts_compare_whole():
    index = NgramIndex.build([("massive:1", "Sessiz"), ("massive:2", "ışıkları kapat lütfen")])
    assert index.overlap("sessiz!") == (1.0, ["massive:1"])
    assert index.overlap("IŞIKLARI KAPAT, lütfen.") == (1.0, ["massive:2"])
    assert index.overlap("sessiz ol") == (0.0, [])
    assert index.overlap("") == (0.0, [])
    assert index.is_contaminated("Sessiz.")


def test_reference_ids_carry_their_set():
    index = NgramIndex()
    with pytest.raises(ValueError, match="must be"):
        index.add("no-separator", "metin")
    index.add("a:1", "metin")
    with pytest.raises(ValueError, match="twice"):
        index.add("a:1", "metin")
    index.add("b:1", "başka")
    assert index.set_sizes() == {"a": 1, "b": 1}


def test_reference_coverage_by_a_training_set():
    index = NgramIndex.build([("ref:long", REFERENCE), ("ref:short", "sessiz")])
    train = tokens(REFERENCE)[:14]
    coverage = index.reference_coverage(set(gram_hashes(train)), {whole_hash(["sessiz"])})
    assert coverage == [pytest.approx(14 / 20), 1.0]


def test_index_round_trips_through_pickle(tmp_path):
    index = NgramIndex.build([("ref:1", REFERENCE), ("ref:2", "Sessiz"), ("ref:3", PARAGRAPH)])
    path = tmp_path / "index.pkl"
    index.save(path)
    loaded = NgramIndex.load(path)
    assert loaded.ref_ids == index.ref_ids
    assert loaded.grams == index.grams
    for text in (REFERENCE, "sessiz", PARAGRAPH[:200], UNRELATED):
        assert loaded.overlap(text) == index.overlap(text)

    lsh = MinHashLSH()
    lsh.add("ref:3", minhash_signature(PARAGRAPH))
    lsh.save(tmp_path / "lsh.pkl")
    again = MinHashLSH.load(tmp_path / "lsh.pkl")
    assert again.query(minhash_signature(PARAGRAPH)) == [("ref:3", 1.0)]
    with pytest.raises(TypeError):
        NgramIndex.load(tmp_path / "lsh.pkl")


def test_minhash_is_deterministic_and_skips_short_texts():
    first, second = minhash_signature(PARAGRAPH), minhash_signature(PARAGRAPH)
    assert first.dtype == np.uint64 and first.shape == (128,)
    assert np.array_equal(first, second)
    assert minhash_signature("dört kelime var burada") is None


def test_bands_suit_a_threshold_of_about_point_eight():
    bands, rows = optimal_bands(0.8, 128)
    assert bands * rows <= 128
    # The S-curve's midpoint (1/b)^(1/r) should sit near the threshold.
    assert 0.7 <= (1 / bands) ** (1 / rows) <= 0.9


def test_minhash_finds_a_light_edit_and_not_an_unrelated_text():
    lsh = MinHashLSH()
    lsh.add("ref:paragraph", minhash_signature(PARAGRAPH))
    lsh.add("ref:other", minhash_signature(UNRELATED))

    # Casing and punctuation changes, one word swapped and one word added.
    edited = turkish_upper(PARAGRAPH.replace("simit", "poğaça")).replace(".", " ;") + " hep"
    matches = lsh.query(minhash_signature(edited))
    assert [key for key, _ in matches] == ["ref:paragraph"]
    assert matches[0][1] >= 0.8

    unrelated = (
        "Mutfakta annem akşam yemeği için mercimek çorbası pişirirken babam balkonda "
        "domates fidelerini suluyordu; kardeşim ise odasında yarınki matematik sınavına "
        "çalışıyor, arada bir pencereden bahçedeki kediye bakıyordu."
    )
    assert lsh.query(minhash_signature(unrelated)) == []


def test_candidates_below_the_threshold_are_rejected():
    lsh = MinHashLSH()
    lsh.add("ref:paragraph", minhash_signature(PARAGRAPH))
    half = " ".join(tokens(PARAGRAPH)[:44]) + " " + UNRELATED
    signature = minhash_signature(half)
    assert estimated_jaccard(signature, lsh.signatures["ref:paragraph"]) < 0.8
    assert lsh.query(signature) == []
