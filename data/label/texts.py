"""Texts for generated decision rows: sampled from a CC0 corpus, masked for personal data.

The texts are Turkish question-and-answer pairs from clips/mqa (CC0-1.0, licence
read on the hub 2026-09-22 and again 2026-09-23), from two configurations:
company FAQ pages (tr-faq-question), close to the product's customer-service
decisions, and community question sites (tr-cqa-question), which carry people's
own questions, complaints and urgency that FAQ pages lack. A text
is one pair:

    Soru: <question>
    Cevap: <answer>

The configurations' parquet files are downloaded once through the hub's file
resolver, which the hub's rate-limit page (read 2026-09-23) recommends over its
API; many small reads of the dataset viewer's rows API were throttled.
Sampling takes a few random rows from each of many random row groups, because
the corpus stores one site's pages next to each other, and caps the pairs per
site: three for FAQ pages, which read alike within a site, and a tenth of the
share for the community configuration, which is a dozen sites of different
people's posts.

Personal data is masked before any text leaves this machine or enters a row
(PLAN.md: PII masking on every collected item). The masks are conservative
regular expressions for what can be matched reliably: e-mail addresses, URLs,
Turkish and international phone numbers, IBANs and 11-digit national identity
numbers, user handles, and a name next to a greeting ("merhaba Esra", "başak
merhaba", "sayın seçer") or before "Bey" or "Hanım", in any case, since most of
the corpus is lower case. Other names
are not masked by rule, because no rule separates a person's name from a brand
or a place; the data card says so.

A pair is dropped at sampling when it comes from a complaint site, which
PLAN.md rules out, or from a gambling page, most of which is generated SEO
text; when it carries markup or template placeholders; when its answer is
only a link; or when it is not Turkish.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

from data import hub

DATASET = "clips/mqa"
CONFIGS = ("tr-faq-question", "tr-cqa-question")
CACHE_DIR = Path(".cache/karar/mqa")
UNDERLINE = re.compile(r"^\s*[=\-]{3,}\s*$", re.MULTILINE)
BLOCKED_DOMAIN = re.compile(r"sikayet|şikayet|complaint|casino|bahis|kumar|iddaa")
GAMBLING = re.compile(
    r"casino|kumarhane|\bkumar|bahis|iddaa|free ?spin|freebet|deneme bonus|bedava bonus"
    r"|çevrim şartı|bonus çevir|yatırım bonus|hoş ?geldin bonus|kayıp bonus|slot oyun"
    r"|slot site|poker|rulet|jackpot|megaways|slot makine",
    re.IGNORECASE,
)
# Commercial adult sites and escort advertising: webcam-sex site FAQ templates,
# escort listings, explicit spam. Questions about sexual health are not caught:
# the patterns name the sites and their templates, not the topic.
ADULT = re.compile(
    r"sex ?cams?\b|live sex|cam sex|live girls|stripchat|drtuber|dirtychat|fuckfap|\bfuck"
    r"|intim\.webcam|private webcam|\bescort|sex ?finder|modellere paramın|ero chats"
    r"|\bxxx\b|\bporn(?!ografi bağımlılığı)",
    re.IGNORECASE,
)
# Betting brands carry "bet" (bets10, tipobet365, süperbetin); a word with "bet"
# in it is taken for one unless it holds a Turkish stem that has "bet" in it.
BET_BRAND = re.compile(r"\w*bet\w*", re.IGNORECASE)
_BET_STEMS = (
    "kaybet elbet sohbet rekabet muhabbet diyabet diabet nöbet şerbet beton beta betim "
    "betik gurbet akıbet ukubet mahabbet alfabe hibet"
)
BET_STEMS = tuple(_BET_STEMS.split())
# Text decoded with the wrong code page: CJK signs inside Turkish words.
MOJIBAKE = re.compile(r"[\u3000-\u9fff\uac00-\ud7af]")
MARKUP = re.compile(r"</?[a-z][a-z0-9]*(\s[^<>]*)?>|href=|\{\s*\w+\s*:|&[a-z]+;", re.IGNORECASE)
LINK = re.compile(r"(?:https?://|www\.)\S+")
_TURKISH = (
    "ve bir bu için ne mi mı mu mü nasıl da de ile çok daha var yok olarak gibi kaç nedir "
    "hangi ben sen o biz siz ama veya ki en şu olan olur"
)
# No short words that are also Turkish suffixes after an apostrophe ("'in", "'de").
_ENGLISH = "the is are does you what how and of this that for with your"
TURKISH_WORDS = frozenset(_TURKISH.split())
ENGLISH_WORDS = frozenset(_ENGLISH.split())

MASKS = (
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[e-posta]"),
    (re.compile(r"(?:https?://|www\.)\S+"), "[bağlantı]"),
    (re.compile(r"\bTR\s?\d{2}(?:\s?\d{4}){5}\s?\d{2}\b", re.IGNORECASE), "[iban]"),
    (re.compile(r"(?<!\d)[1-9]\d{10}(?!\d)"), "[kimlik]"),
    (
        re.compile(
            r"(?<![\w+])(?:\+?90[\s-]?)?\(?0?5\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"
        ),
        "[telefon]",
    ),  # fmt: skip
    (
        re.compile(
            r"(?<![\w+])(?:\+?90[\s-]?)?\(?0?[2-4]\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"
        ),
        "[telefon]",
    ),  # fmt: skip
    (re.compile(r"(?<!\d)0?\s?\(?850\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b"), "[telefon]"),
    # Any other grouping of a leading 0 or +90, an area code and seven digits, as
    # in "0-212- 444 0 678"; the leading digits keep dates and counts out.
    (
        re.compile(r"(?<![\w+])(?:\+?90|0)[\s-]*\(?[2-5]\d{2}\)?(?:[\s-]*\d){7}(?!\d)"),
        "[telefon]",
    ),  # fmt: skip
    (re.compile(r"(?<![\w.])@\w{3,}|(?<=\w)@\w{3,}(?![\w.])"), "@[kullanıcı]"),
    (re.compile(r"\b\w+\s+(bey|hanım|hanim)\b", re.IGNORECASE), r"[ad] \1"),
)
GREETING = re.compile(
    r"\b(merhaba|merhabalar|selam|sayın|değerli|sevgili|kıymetli)([\s,]+)(\w+)", re.IGNORECASE
)
GREETED = re.compile(r"\b(\w+)([\s,]+)(merhaba|merhabalar|selam)\b", re.IGNORECASE)
# Words that follow or precede a greeting and are not names.
_NOT_NAMES = (
    "arkadaşlar arkadaşım herkese herkes hocam hocalar yetkili yetkililer müşterimiz "
    "müşterilerimiz müşteri dostlar dostum beyler hanımlar ben biz bilimseverler kardeşim "
    "kardeşler abi abla doktor doktorum dr değerli sevgili sayın tekrar öncelikle ve "
    "iyi günler akşamlar size sizlere siz sen hepinize tüm bütün nasılsınız nasılsın kolay "
    "geçmiş sorunuz sorunuza sorunuzla teşekkürler teşekkür öncelikle şimdiden bu bir"
)
NOT_NAMES = frozenset(_NOT_NAMES.split())


def _greeting(match: re.Match) -> str:
    word = match.group(3)
    keep = word.lower() in NOT_NAMES or word.isdigit() or word.startswith("[")
    return match.group(0) if keep else f"{match.group(1)}{match.group(2)}[ad]"


def _greeted(match: re.Match) -> str:
    word = match.group(1)
    keep = word.lower() in NOT_NAMES or word.isdigit() or word.endswith("]")
    return match.group(0) if keep else f"[ad]{match.group(2)}{match.group(3)}"


def mask_personal_data(text: str) -> str:
    for pattern, replacement in MASKS:
        text = pattern.sub(replacement, text)
    text = GREETING.sub(_greeting, text)
    return GREETED.sub(_greeted, text)


def _clean(text: str) -> str:
    """Forum titles carry heading underlines ("=====") and blank runs; drop both.

    Some sources lowercased Turkish with a locale-blind function, which turns
    "İ" into "i" plus a combining dot; the dot is dropped.
    """
    text = text.replace("i\u0307", "i")
    return re.sub(r"\n{2,}", "\n", UNDERLINE.sub("", text)).strip()


# Pages whose "question" is a patient's message cut at a fixed length and whose
# "answer" repeats it and points to a doctor's page, or an SEO firm's page that
# answers every question with "call us". Found by the owner on the label desk.
POINTER = re.compile(
    r"doktorumuzun\b.{0,80}\bcevaplar[ıi]|sorusuna doğru ve hızlı cevap almak için",
    re.IGNORECASE,
)


def is_stub(question: str, answer: str) -> bool:
    """An answer that is the question again plus a pointer or a few words."""
    if POINTER.search(answer):
        return True
    q, a = _squash(question), _squash(answer)
    return len(q) >= 15 and a.startswith(q) and len(a) - len(q) < 40


def _squash(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())


def is_turkish(text: str) -> bool:
    """Not mostly English, and two Turkish function words or a Turkish letter."""
    words = re.findall(r"\w+", text.lower())
    turkish = sum(w in TURKISH_WORDS for w in words)
    english = sum(w in ENGLISH_WORDS for w in words)
    if english >= 2 and english > turkish:
        return False
    return turkish >= 2 or bool(re.search(r"[çğıöşüÇĞİÖŞÜ]", text))


def blocked(row: dict, question: str, answer: str) -> bool:
    """A pair from a source PLAN.md rules out, or one too broken to read."""
    if BLOCKED_DOMAIN.search((row.get("domain") or "").lower()):
        return True
    both = question + "\n" + answer
    if GAMBLING.search(both) or MARKUP.search(both) or MOJIBAKE.search(both) or ADULT.search(both):
        return True
    words = (m.group(0).lower() for m in BET_BRAND.finditer(both))
    if any(not any(stem in word for stem in BET_STEMS) for word in words):
        return True
    if LINK.search(answer) and len(_squash(LINK.sub("", answer))) < 15:
        return True
    if is_stub(question, answer):
        return True
    return not is_turkish(question + "\n" + answer)


def still_allowed(text: dict) -> bool:
    """A sampled text checked again under the current filters, for a sample drawn earlier."""
    question, _, answer = text["text"].partition("\nCevap: ")
    return not blocked(text, question.removeprefix("Soru: "), answer)


def faq_text(row: dict) -> str | None:
    """One question-and-answer pair as a state, or None if it is unusable."""
    question = _clean(row.get("name") or "")
    answers = [a for a in row.get("answers") or [] if (a.get("text") or "").strip()]
    if not question or not answers:
        return None
    answer = _clean(next((a for a in answers if a.get("is_accepted")), answers[0])["text"])
    # Generated SEO pages answer with their own question, or keep link markup.
    if _squash(answer) in _squash(question) or "](" in question + answer:
        return None
    if blocked(row, question, answer):
        return None
    text = mask_personal_data(f"Soru: {question}\nCevap: {answer}")
    return text if 60 <= len(text) <= 1500 else None


def shards(config: str, cache: Path = CACHE_DIR) -> list[Path]:
    return hub.parquet_files(DATASET, config, "train", cache)


def sample_faq(
    count: int,
    *,
    config: str = CONFIGS[0],
    per_domain: int = 3,
    seed: int = 1,
    per_group: int = 4,
    cache: Path = CACHE_DIR,
) -> list[dict]:
    """About `count` masked pairs, a few from each of many random row groups."""
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    files = [pq.ParquetFile(path) for path in shards(config, cache)]
    groups = [(f, g) for f in files for g in range(f.metadata.num_row_groups)]
    rng.shuffle(groups)
    by_domain: Counter[str] = Counter()
    out = []
    for file, group in groups:
        table = file.read_row_group(group, columns=["id", "name", "domain", "answers"]).to_pylist()
        for row in rng.sample(table, min(per_group, len(table))):
            domain = row.get("domain") or "?"
            text = faq_text(row)
            if text is None or by_domain[domain] >= per_domain:
                continue
            by_domain[domain] += 1
            out.append({"source_id": row["id"], "domain": domain, "text": text, "config": config})
            if len(out) >= count:
                return out
    return out


def sample_mixed(
    count: int, *, seed: int = 1, community_share: float = 0.5, **kwargs
) -> list[dict]:
    """Texts from both configurations, company FAQ pages and community questions."""
    community = round(count * community_share)
    rows = sample_faq(count - community, seed=seed, config=CONFIGS[0], **kwargs)
    rows += sample_faq(
        community, seed=seed, config=CONFIGS[1], per_domain=max(3, community // 10), **kwargs
    )
    return rows


def write_sample(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
