"""WebFAQ relevance rows: train pairs only, joined sites, hard but honest negatives. No network."""

from collections import Counter

from data.decontam.ngrams import normalise
from data.typed.relevance import (
    Pair,
    Pool,
    join_sites,
    keep,
    relevance_rows,
    sample,
    train_pairs,
)
from data.typed.sources import hash_split


def test_train_pairs_join_back_to_their_site():
    queries = [{"_id": 1, "text": "Kargo ücreti ne kadar?"}, {"_id": 2, "text": "İade var mı?"}]
    corpus = [
        {"_id": 10, "title": "", "text": "Kargo ücretsizdir."},
        {"_id": 11, "title": "", "text": "30 gün içinde iade edebilirsiniz."},
    ]
    qrels = [
        {"query-id": 1, "corpus-id": 10, "score": 1},
        {"query-id": 2, "corpus-id": 11, "score": 1},
    ]
    pairs = train_pairs(queries, corpus, qrels)
    raw = [
        {
            "origin": "https://a.com",
            "url": "https://a.com/sss",
            "question": "Kargo ücreti ne kadar?",
            "answer": "Kargo ücretsizdir.",
        },
        {
            "origin": "https://b.com",
            "url": "https://b.com/x",
            "question": "Başka?",
            "answer": "Yok.",
        },
    ]
    joined, unjoined = join_sites(pairs, raw)
    assert unjoined == 1
    assert joined == [
        Pair(
            "1",
            "Kargo ücreti ne kadar?",
            "Kargo ücretsizdir.",
            "https://a.com",
            "https://a.com/sss",
        )
    ]


def test_filters_drop_gambling_and_short_answers():
    ok = Pair(
        "1",
        "Siparişim ne zaman gelir?",
        "Siparişiniz iki iş günü içinde kargoya verilir.",
        "https://magaza.com",
        "u",
    )
    assert keep(ok) is None
    assert keep(Pair("2", ok.question, "Evet.", ok.origin, "u")) == "answer_length"
    bet = Pair(
        "3",
        "Deneme bonusu nasıl alınır?",
        "Üye olun ve deneme bonusu kazanın hemen.",
        "https://x.com",
        "u",
    )
    assert keep(bet) == "blocked"
    site = Pair("4", ok.question, ok.answer, "https://www.sikayetvar.com", "u")
    assert keep(site) == "blocked"


QUESTIONS = [
    (
        "Rezervasyonu nasıl iptal ederim?",
        "İptal için hesabınıza girip rezervasyonlarım sayfasından işlemi başlatın.",
    ),  # noqa: E501
    ("Ödeme seçenekleri nelerdir?", "Kredi kartı, havale ve kapıda nakit ödeme kabul ediyoruz."),  # noqa: E501
    (
        "Kargo kaç günde gelir?",
        "Siparişler iki iş günü içinde kargoya verilir, teslimat üç gün sürer.",
    ),  # noqa: E501
    (
        "Fatura bilgilerimi değiştirebilir miyim?",
        "Fatura adresinizi sipariş onaylanmadan önce profil ekranından düzenleyebilirsiniz.",
    ),  # noqa: E501
    (
        "Çocuklar için indirim var mı?",
        "Yedi yaş altı çocuklar ücretsizdir, on iki yaşa kadar yarı fiyat uygulanır.",
    ),  # noqa: E501
    (
        "Evcil hayvan kabul ediliyor mu?",
        "Küçük köpek ve kediler ek ücretle kabul edilir, önceden bildirmeniz gerekir.",
    ),  # noqa: E501
    (
        "Giriş saati kaçta başlar?",
        "Odalara giriş öğleden sonra ikide başlar, erken giriş talebe bağlıdır.",
    ),  # noqa: E501
    (
        "Havalimanı transferi sunuyor musunuz?",
        "Havalimanından tesise özel araçla transfer hizmetimiz ücretlidir.",
    ),  # noqa: E501
]


def faq(n_sites=6, per_site=8):
    """Sites with the same eight FAQ entries, each question marked with its site."""
    pairs = []
    for s in range(n_sites):
        for p in range(per_site):
            question, answer = QUESTIONS[(p + s) % len(QUESTIONS)]
            pairs.append(
                Pair(
                    f"{s}-{p}",
                    f"{question} (şube {s})",
                    f"{answer} Ayrıntı için {s}{p} numaralı rehbere bakın.",
                    f"https://site{s}.com.tr",
                    f"https://site{s}.com.tr/sayfa{p}",
                )
            )
    return pairs


def test_every_question_gets_its_answer_and_one_negative():
    pairs = faq()
    rows, stats = relevance_rows(pairs, target=40, source="src/")
    positives = [r for r in rows if r.target["true"] == 1.0]
    negatives = [r for r in rows if r.target["true"] == 0.0]
    assert len(positives) == len(negatives) > 0
    assert all(
        r.label_kind == "rule" and r.track == "arama" and r.question.type == "noul" for r in rows
    )
    questions = Counter(r.state.split("\nPasaj: ")[0] for r in rows)
    assert set(questions.values()) == {2}, "each question carries one positive and one negative"
    by_question = {}
    for r in rows:
        by_question.setdefault(r.state.split("\nPasaj: ")[0], []).append(r)
    for group in by_question.values():
        assert len({r.split for r in group}) == 1
        pos = next(r for r in group if r.target["true"] == 1.0).state.split("\nPasaj: ")[1]
        neg = next(r for r in group if r.target["true"] == 0.0).state.split("\nPasaj: ")[1]
        assert normalise(pos) != normalise(neg)
    assert stats["negative_same_site"] > 0


def test_near_duplicates_and_same_template_questions_are_never_negatives():
    base = Pair(
        "a",
        "Otelde otopark var mı?",
        "Evet, otelde ücretsiz otopark vardır, rezervasyon gerekmez.",  # noqa: E501
        "https://h.com",
        "https://h.com/1",
    )
    copy = Pair(
        "b",
        "Otopark hizmeti hakkında bilgi",
        "Evet, otelde ücretsiz otopark vardır; rezervasyon gerekmez.",  # noqa: E501
        "https://h.com",
        "https://h.com/2",
    )
    template = Pair(
        "c",
        "Otelde otopark var mı?",
        "Hayır, bu otelde otopark bulunmuyor, yakında park yeri var.",  # noqa: E501
        "https://h.com",
        "https://h.com/3",
    )
    other = Pair(
        "d",
        "Kahvaltı saat kaçta?",
        "Kahvaltı sabah yedide başlar, otopark girişinin yanındaki salonda verilir.",  # noqa: E501
        "https://h.com",
        "https://h.com/4",
    )
    pool = Pool([base, copy, template, other])
    j, kind = pool.negative(0)
    assert pool.pairs[j].qid == "d" and kind == "same_site"


def test_sampling_caps_each_site_and_is_reproducible():
    pairs = faq(n_sites=3, per_site=8)
    chosen = sample(pairs, target=100, cap=3)
    assert len(chosen) == 9
    assert Counter(pairs[i].origin for i in chosen) == {p.origin: 3 for p in pairs}
    assert sample(pairs, target=100, cap=3) == chosen


def test_hash_split_gives_about_a_tenth():
    share = sum(hash_split(f"id-{i}") == "validation" for i in range(20_000)) / 20_000
    assert 0.09 < share < 0.11
