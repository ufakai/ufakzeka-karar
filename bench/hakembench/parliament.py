"""Check-worthiness candidates for HakemBench: sentences from parliamentary proceedings.

    python -m bench.hakembench.parliament --data data/built

Source: Zenodo record 19713325, the TBMM proceedings corpus with speaker turns,
party linkage and checked metadata, 1950 to 2023 (CC BY 4.0), its analytical
core file: floor speeches whose speaker is linked to a member of parliament
and a party at high or medium confidence. For 1950 to 2018 the text is OCR of
the printed proceedings (the Turkronicles corpus, CC BY 4.0); for the 27th term
(2018 to 2023) it comes from the parliament's own Word transcripts.

Nothing here is written or reworded by anything. Every candidate is one
sentence cut from one speaker turn; the only changes are line breaks inside
the sentence turned into spaces (the OCR half wraps lines mid-sentence) and
personal-data masks. The meta file keeps the character offsets, and the run
checks each candidate against its turn again after the draw.

The draw, seed 1:

1. Population: turns from 1 January 2011 in the plenary chambers, with a
   speaker and a party. The 32 turns the corpus files under the 1980s
   advisory assembly (DANISMA) are left out.
2. Each turn keeps only the text before the first speaker line inside it
   ("BAŞKAN – ...", "AD SOYAD (Devamla) – ..."), because the corpus's turns
   sometimes run on into the chair's or another member's words, and the
   attribution would then be wrong.
3. Sentences are split by a Turkish-aware rule: no break after an
   abbreviation ("Dr.", "Prof.", "Hz.", "Sn."), a single capital (an
   initial), a Roman numeral, or a short number followed by a capitalised
   word that does not open sentences ("2. Ceza Dairesi"); no break inside a
   quotation; in the transcripts, every line is a paragraph.
4. A sentence is skipped, by the first rule it meets: OCR noise (a page
   header, a word hyphenated across lines, punctuation with no space after
   it, a stage note such as "(AK PARTİ sıralarından alkışlar)", a speaker
   line, mostly capitals); procedure (an opening address to the chair or the
   members, thanks, greetings, commemoration and condolence formulas, chair
   procedure and speaking-time formulas, vote announcements); shape (not
   starting with a capital, a digit or a quote, not ending in a full stop,
   ending in an ellipsis, unbalanced quotes or brackets); length outside 8
   to 40 words; in the OCR half, words run together ("hizmetalanı", a digit
   inside a word, or two words the transcripts never use); a name the
   personal-data rules would mask (below).
5. A turn with at least one sentence left is eligible. Party quotas are
   proportional to the population's turns with no party above 35 percent
   (the excess goes to the others in proportion); within a party the quota is
   spread over (legislative year, month) cells in proportion to its turns
   there, both by largest remainder, no cell above its eligible turns. The
   shares follow the population, not the eligible turns, because the OCR of
   2011 to 2014 loses most of its sentences to the noise rules and would
   otherwise lose its share. The legislative year runs from 1 October, from
   the sitting date. Within a cell, turns are drawn in a seeded order, and
   each drawn turn gives one sentence at a seeded position among its eligible
   sentences.
6. Candidates whose text overlaps a training file (data/built train and
   validation files, by bench/dev.py's rules) or a FACTurk claim are dropped
   and counted, as are repeated texts; each cell draws a reserve so the
   quotas still fill.

Personal data: texts.mask_personal_data's patterns, with parliamentary
titles added to the words its name rules skip, so "Sayın Bakan" and "Bakan
Bey" are not names. Contact data (addresses, numbers, links) is masked. A
sentence in which the name rules would mask a word ("Sayın Yılmaz", "Ahmet
Bey") is skipped instead: in the chamber these are mostly members and
ministers, and the rule masks one word of the name ("Sayın [ad] Tayyip
Erdoğan"), which neither hides the person nor reads as a sentence. A name with
no title or greeting next to it is kept (no rule separates a person's name
from a place or a party). The speaker's name stays in the meta file, never in
the text.

Each item carries two questions: check-worthiness exactly as the training
rows ask it (data/label/synth.py), and a four-level check priority. Gold is
empty; the panel labels later. Items are split public or private by the
parity of their text's sha256.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from bench.dev import training_overlap
from bench.harness.items import Item
from data.label import texts
from data.label.synth import CHECKABLE

RECORD = "19713325"
SOURCE = f"https://zenodo.org/records/{RECORD}"
FILE_NAME = "tbmm_speeches_v1_floor_core.parquet"
URL = f"https://zenodo.org/api/records/{RECORD}/files/{FILE_NAME}/content"
SHA256 = "fc856032cea5273b1ddf63031633fe4d2ef2c486e446d46b4094d7c2af1f93c8"
RAW = Path(f"data/raw/_downloads/zenodo__{RECORD}")
OUT = Path("bench/hakembench/candidates")
REPORT = Path("results/step8/hakembench/dogrulama_candidates.json")
FACTURK = Path("data/raw/_downloads/altuncu__FACTurk/FACTurk.zip")
CACHE = Path(".cache/karar/hakembench")
LICENCE = "CC-BY-4.0"
TRACK = "dogrulama"

SEED = 1
TARGET = 1300
PARTY_CAP = 0.35
MIN_WORDS, MAX_WORDS = 8, 40
FROM = dt.datetime(2011, 1, 1)
CHAMBERS = ("TBMM", "MILLET")
# The corpus's own label for its Word-transcript half, where a line is a paragraph.
TRANSCRIPT = "tbmm_direct_scrape"
# Per cell, drawn beyond the quota in case a candidate is dropped for overlap.
RESERVE_SHARE = 0.25
RESERVE_MIN = 2

COLUMNS = [
    "file_id", "speech_id_within_doc", "tarih", "tbmm_donemi", "yasama_yili", "meclis_turu",
    "source", "speaker_name_raw", "speaker_role_raw", "wiki_name", "wiki_parti", "speech_text",
]  # fmt: skip

PRIORITY = {
    "type": "score",
    "instructions": "Bu cümledeki iddianın kamuoyu için doğruluk kontrolü ne kadar öncelikli?",
    "criteria": [
        "Kontrol gerektirmez: doğrulanabilir bir iddia yok ya da kişisel görüş.",
        "Düşük: doğrulanabilir ama önemsiz ya da zararsız bir iddia.",
        "Orta: kamuyu ilgilendiren, yanlışsa yanıltabilecek bir iddia.",
        "Yüksek: sağlık, güvenlik, seçim ya da ekonomi gibi alanlarda yanlışsa ciddi zarar "
        "verebilecek bir iddia.",
    ],
}
QUESTIONS = {"dogrulanabilir": CHECKABLE, "kontrol_onceligi": PRIORITY}

_LOWER = str.maketrans({"İ": "i", "I": "ı"})


def lower_tr(text: str) -> str:
    return text.translate(_LOWER).lower()


# sentence splitting -----------------------------------------------------------

_ABBREVIATIONS = (
    "dr prof doç av sn op uzm yrd öğr gör arş gn org korg orgl tümg tuğg alb yzb bnb tğm "
    "ütğm vb vs örn bkz yy no md mad st sy bşk genl müd hz aş şti ltd cad sok mah ecz dt vd "
    "mr ms"
)
ABBREVIATIONS = frozenset(_ABBREVIATIONS.split())
# After a number of up to three digits and a full stop, a break only before one
# of these words; otherwise the number is read as an ordinal ("2. Ceza Dairesi").
_STARTERS = (
    "bu şu o ve ama fakat ancak şimdi yani bir biz siz ben onlar sayın değerli peki evet hayır "
    "hem çünkü dolayısıyla oysa işte buna bunu bunun bunlar burada neden niye nasıl ne kim hangi "
    "eğer ayrıca üstelik bakın bakınız maalesef tabii elbette bugün yine hâlâ hala daha bütün "
    "tüm her geçen sonra önce şu anda arkadaşlar"
)
STARTERS = frozenset(_STARTERS.split())
ROMAN = re.compile(r"[IVX]{1,4}")
END = re.compile(r"(?:\.\.\.|…|[.!?])+[”\"’»)]*(?=\s)")
OPENERS = '“"‘«('
LAST_WORD = re.compile(r"(\w+)$")
NEXT_WORD = re.compile(r"\s*([^\s]+)")

CAPS = "A-ZÇĞİÖŞÜÂÎÛ"
SPEAKER = re.compile(
    rf"(?<![\w’'])((?:[{CAPS}][{CAPS}’'.]*\s+){{0,8}}[{CAPS}][{CAPS}’'.]+)"
    rf"\s*(\([^()\n]{{1,60}}\))?\s*[–—-]\s"
)
CHAIR_WORDS = frozenset({"BAŞKAN", "BAŞKANVEKİLİ", "REİS"})


def own_end(text: str) -> int:
    """Where the speaker's own words end: at the first speaker line inside the turn."""
    for match in SPEAKER.finditer(text):
        words = match.group(1).split()
        if match.group(2) or len(words) >= 2 or words[-1] in CHAIR_WORDS:
            return match.start()
    return len(text)


def _breaks(block: str, start: int, stop: re.Match) -> bool:
    """Whether the sentence that began at `start` ends with the stop `stop` matched."""
    rest = NEXT_WORD.match(block, stop.end())
    if rest is None:
        return False
    following = rest.group(1)
    head = following[0]
    if not (head.isupper() or head.isdigit() or head in OPENERS):
        return False
    segment = block[start : stop.end()]
    if segment.count("“") > segment.count("”"):
        return False
    if stop.group(0).rstrip('”"’»)') != ".":
        return True
    token = LAST_WORD.search(block[start : stop.start()])
    return _period_breaks(token.group(1) if token else "", following)


def _period_breaks(token: str, following: str) -> bool:
    if not token:
        return True
    if lower_tr(token) in ABBREVIATIONS:
        return False
    if len(token) == 1 and token.isalpha() and token.isupper():
        return False
    if ROMAN.fullmatch(token):
        return False
    if token.isdigit() and len(token) <= 3:
        word = lower_tr(following.strip('“"‘«(,'))
        return word in STARTERS
    return True


def split_sentences(text: str, *, paragraphs: bool) -> list[tuple[int, int]]:
    """Sentence spans (start, end) into `text`, whitespace trimmed.

    With `paragraphs`, a line break always ends a sentence (the transcripts);
    without, it is a space (the OCR half wraps lines inside sentences).
    """
    blocks = []
    if paragraphs:
        position = 0
        for line in text.split("\n"):
            blocks.append((position, line))
            position += len(line) + 1
    else:
        blocks.append((0, text))
    spans = []
    for offset, block in blocks:
        start = 0
        for match in END.finditer(block):
            if not _breaks(block, start, match):
                continue
            spans.append(_trim(block, start, match.end(), offset))
            start = match.end()
        spans.append(_trim(block, start, len(block), offset))
    return [s for s in spans if s[1] > s[0]]


def _trim(block: str, start: int, end: int, offset: int) -> tuple[int, int]:
    while start < end and block[start].isspace():
        start += 1
    while end > start and block[end - 1].isspace():
        end -= 1
    return offset + start, offset + end


# sentence filters ---------------------------------------------------------------

PAGE_HEADER = re.compile(r"TBMM\s*B\s*:\s*\d|\bO\s*:\s*\d")
# A whole running header of the printed proceedings, "TBMM B: 58 19 . 2 . 2015 O: 6167":
# its spaced date would otherwise split it across two sentences.
PAGE_HEADER_SPAN = re.compile(
    r"TBMM\s*B\s*:\s*\d+\s*\d{1,2}\s*\.\s*\d{1,2}\s*\.\s*\d{4}\s*O\s*:\s*\d+"
)
HYPHEN_WRAP = re.compile(r"\w-[ \t]*\n")
SOFT_HYPHEN = "­"
# A word the transcripts never use that splits into two words they use often,
# "hizmetalanı" or "söylediğimgibi": the OCR half dropped the space.
GLUE_MIN_LETTERS = 6
GLUE_PART_COUNT = 5
LETTERS = re.compile(r"[^\W\d_]+")
DIGIT_IN_WORD = re.compile(r"[^\W\d_]\d|\d[^\W\d_]")
# OCR that lost spaces: punctuation run into the next word ("birlikte,üç"), except
# after a capital or a digit ("T.C.", "3.5"); or a space before a stop ("bir .").
GLUED = re.compile(r"(?<![A-ZÇĞİÖŞÜ\d])[,;:.!?](?=[a-zçğıöşüâîûA-ZÇĞİÖŞÜ“])|\s[.,;:!?](?!\.)")
STAGE_NOTE = re.compile(
    r"\([^()]*(?:sıralarından|alkış|gürültü|mikrofon|uğultu|devamla|sesleri|kürsüye|gösterdi"
    r"|gösteriyor|divanı|müdahale|laf atma|ayağa kalk|oturuyor)[^()]*\)"
)
OPENING = re.compile(
    r"^(?:(?:çok )?(?:sayın|değerli|saygıdeğer|kıymetli|muhterem|aziz)\s+)+"
    r"(?:başkan|milletvekil|arkadaş|üye|divan|genel kurul|meclis|heyet|parlamenter|vekil)"
    r"|^başkanım\b|^genel kurulu(?:muzu)?\s"
)
PROCEDURAL = {
    "thanks": re.compile(
        r"teşekkür|sağ ?olun|sağ ?ol\b|saygılarımı|saygılarımla|saygıyla selamla|saygılar\w* sun"
        r"|selamlıyor|selamlarımı|minnettar|şükran"
    ),
    "greeting": re.compile(
        r"hayırlı olsun|hayırlı olmasını|hoş ?geldiniz|günaydın|iyi akşamlar|iyi geceler"
        r"|iyi günler dilerim|merhaba|başsağlığı|rahmet diliyorum|\banıyor(?:um|uz)\b"
        r"|şifa diliyorum|acil şifalar|geçmiş olsun|başarılar dil|bayram\w* kutlu"
        r"|tebrik ediyorum|kutluyorum"
    ),
    "chair": re.compile(
        r"birleşim|oturum|ara veriyorum|söz veriyorum|sözü \w+ veriyorum|söz vereceğim|söz aldım"
        r"|söz almış|söz istedim|söz talebi|buyurun|süreniz|ek süre|süre veriyorum|tamamlayın"
        r"|yoklama|yeter sayısı|gündeme geçiyoruz|gündemin|okutuyorum|kâtip üye|katip üye"
        r"|önergeyi|önergesi\w* üzerinde|önergemiz|katılıyor musunuz|katılmıyoruz|katılıyoruz"
        r"|iç ?tüzük|sıra sayılı|grubu adına|şahsı adına|şahsım adına|kişisel söz|yerinden"
        r"|sisteme gir|kürsüye davet|madde üzerinde|tümü üzerinde|bölüm üzerinde|danışma kurulu"
        r"|grup önerisi|grup önerimiz|ivedilik|yeterlilik|sataşma|açıklama yapmak|kayıtlara geç"
        r"|zapta geç|tutanaklara"
    ),
    "vote": re.compile(
        r"oylarınıza|oya sunuyorum|kabul edenler|kabul etmeyenler|etmeyenler|kabul edilmiştir"
        r"|kabul edilmemiştir|reddedilmiştir|oylama|oy birliği|oy çokluğu|açık oy|gizli oy"
        r"|elektronik cihazla"
    ),
}
# The order in which a skipped sentence is counted: the first rule it meets.
REASONS = (
    "page_header", "hyphen_wrap", "ocr_spacing", "stage_note", "speaker_line", "capitals",
    "opening", *PROCEDURAL, "start", "unfinished", "unbalanced", "short", "long", "ocr_glued",
    "names_person",
)  # fmt: skip


def words(text: str) -> list[str]:
    return text.split()


def vocabulary(turns: Iterable[dict]) -> Counter:
    """Word counts over the transcript half, which has no OCR."""
    counts: Counter = Counter()
    for turn in turns:
        if turn["source"] == TRANSCRIPT:
            counts.update(LETTERS.findall(lower_tr(turn["speech_text"])))
    return counts


def _splits(word: str, vocab: Counter) -> bool:
    """Whether `word` is two or more words the transcripts use often, run together."""
    # ends[i]: word[:i] is a run of such words, and how many.
    ends = {0: 0}
    for end in range(2, len(word) + 1):
        for start in range(0, end - 1):
            if start in ends and vocab[word[start:end]] >= GLUE_PART_COUNT:
                ends[end] = max(ends.get(end, 0), ends[start] + 1)
    return ends.get(len(word), 0) >= 2


def glued(text: str, vocab: Counter) -> bool:
    """OCR that lost spaces: a word run together from others, a digit inside a word, or
    two or more words the transcripts never use (one may be a rare name)."""
    if DIGIT_IN_WORD.search(text):
        return True
    unknown = 0
    for word in LETTERS.findall(lower_tr(text)):
        if vocab[word] or len(word) < 3:
            continue
        unknown += 1
        if unknown >= 2 or (len(word) >= GLUE_MIN_LETTERS and _splits(word, vocab)):
            return True
    return False


def reject(raw: str, vocab: Counter | None = None) -> str | None:
    """Why a sentence is skipped, or None when it is a candidate.

    With `vocab` (word counts from the transcripts), OCR text is also checked
    for words that lost the space between them.
    """
    if PAGE_HEADER.search(raw):
        return "page_header"
    if HYPHEN_WRAP.search(raw) or SOFT_HYPHEN in raw:
        return "hyphen_wrap"
    text = " ".join(raw.split())
    if GLUED.search(text):
        return "ocr_spacing"
    low = lower_tr(text)
    if STAGE_NOTE.search(low):
        return "stage_note"
    if own_end(text) < len(text):
        return "speaker_line"
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) > 0.5 * len(letters):
        return "capitals"
    if OPENING.search(low):
        return "opening"
    for reason, pattern in PROCEDURAL.items():
        if pattern.search(low):
            return reason
    if not (text[0].isupper() or text[0].isdigit() or text[0] in '“"‘«'):
        return "start"
    core = text.rstrip('”"’»')
    if not core or core[-1] not in ".!?" or core.endswith(("..", "…")) or "…" in core[-3:]:
        return "unfinished"
    if text.count("“") != text.count("”") or text.count("(") != text.count(")"):
        return "unbalanced"
    count = len(words(text))
    if count < MIN_WORDS:
        return "short"
    if count > MAX_WORDS:
        return "long"
    if vocab is not None and glued(text, vocab):
        return "ocr_glued"
    if names_person(text):
        return "names_person"
    return None


def eligible(
    turn: dict, counts: Counter | None = None, vocab: Counter | None = None
) -> list[tuple[int, int]]:
    """The spans of a turn's sentences that pass every filter; `vocab` checks OCR turns."""
    text = turn["speech_text"]
    vocab = None if turn["source"] == TRANSCRIPT else vocab
    own = text[: own_end(text)]
    counts = counts if counts is not None else Counter()
    counts["cut_at_speaker_line"] += own != text
    headers = [m.span() for m in PAGE_HEADER_SPAN.finditer(own)]
    spans = []
    for start, end in split_sentences(own, paragraphs=turn["source"] == TRANSCRIPT):
        counts["sentences"] += 1
        on_header = any(start < h_end and h_start < end for h_start, h_end in headers)
        reason = "page_header" if on_header else reject(own[start:end], vocab)
        if reason:
            counts[reason] += 1
        else:
            spans.append((start, end))
    counts["eligible_sentences"] += len(spans)
    return spans


# personal data ------------------------------------------------------------------

_TITLES = (
    "başkan başkanım başkanımız başkanvekili başkanvekilim başkanvekilleri bakan bakanım "
    "bakanımız bakanlar bakanlarım bakanlık bakanlığı milletvekili milletvekilleri milletvekilim "
    "milletvekillerimiz vekil vekilim vekiller vekillerimiz genel grup grubu cumhurbaşkanı "
    "cumhurbaşkanım cumhurbaşkanımız başbakan başbakanım başbakanımız meclis meclisimiz üye "
    "üyeler üyelerimiz komisyon komisyonumuz hükûmet hükumet hükümet kâtip katip divan heyet "
    "arkadaşlarım arkadaşlarımız arkadaşımız arkadaşım halkımız halkım vatandaşlar "
    "vatandaşlarımız vatandaşım yurttaşlar yurttaşlarımız yurttaşım milletim milletimiz "
    "kardeşlerim kardeşlerimiz dostlarım hocam hocamız hoca kurul valim vali valimiz kaymakam "
    "büyükelçi elçi parlamenter parlamenterler sayın değerli sevgili kıymetli saygıdeğer "
    "muhterem aziz çok gençler gençlerimiz şehitlerimiz"
)
TITLES = frozenset(_TITLES.split())


def _kept(word: str) -> bool:
    low = lower_tr(word)
    return (
        low in TITLES
        or word.lower() in texts.NOT_NAMES
        or low in texts.NOT_NAMES
        or word.isdigit()
        or word.startswith("[")
        or word.endswith("]")
    )


def _honorific(match: re.Match) -> str:
    word = match.group(0).split()[0]
    return match.group(0) if _kept(word) else f"[ad] {match.group(1)}"


def _greeting(match: re.Match) -> str:
    return match.group(0) if _kept(match.group(3)) else f"{match.group(1)}{match.group(2)}[ad]"


def _greeted(match: re.Match) -> str:
    return match.group(0) if _kept(match.group(1)) else f"[ad]{match.group(2)}{match.group(3)}"


def mask_names(text: str) -> str:
    """texts.mask_personal_data with parliamentary titles among the words its name rules skip."""
    for pattern, replacement in texts.MASKS:
        if replacement.startswith("[ad]"):
            text = pattern.sub(_honorific, text)
        else:
            text = pattern.sub(replacement, text)
    text = texts.GREETING.sub(_greeting, text)
    return texts.GREETED.sub(_greeted, text)


def names_person(text: str) -> bool:
    """Whether the name rules would mask a word: "Sayın Yılmaz", "Ahmet Bey".

    Such a sentence is skipped, not masked. In the chamber the rule mostly hits
    members and ministers, and masking only the word next to the title leaves
    the rest of the name ("Sayın [ad] Tayyip Erdoğan", "Sayın [ad] Bakanı"),
    which neither hides the person nor reads as a sentence.
    """
    return "[ad]" in mask_names(text) and "[ad]" not in text


# the draw -----------------------------------------------------------------------


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_of(text: str) -> str:
    """Public or private, by the parity of the text's sha256: stable per text, about half each."""
    return "public" if int(sha256(text), 16) % 2 == 0 else "private"


def legislative_year(date: dt.date) -> str:
    """The legislative year opens on 1 October (Constitution, article 93)."""
    first = date.year if date.month >= 10 else date.year - 1
    return f"{first}-{first + 1}"


def cell_of(turn: dict) -> str:
    date = turn["tarih"]
    return f"{legislative_year(date)}/{date.month:02d}"


def largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    """Integer shares of `total` proportional to `weights`; ties go by key order."""
    mass = sum(weights.values())
    if total <= 0 or mass <= 0:
        return {key: 0 for key in weights}
    exact = {key: total * weight / mass for key, weight in weights.items()}
    shares = {key: int(value) for key, value in exact.items()}
    left = total - sum(shares.values())
    order = sorted(weights, key=lambda key: (-(exact[key] - shares[key]), key))
    for key in order[:left]:
        shares[key] += 1
    return shares


def allocate(weights: dict[str, float], total: int, capacity: dict[str, int]) -> dict[str, int]:
    """Largest-remainder shares with a ceiling per key; what a full key cannot take is
    spread over the others in proportion to their weights. The sum falls short of
    `total` only when every key is full."""
    fixed: dict[str, int] = {}
    while True:
        free = {k: w for k, w in weights.items() if k not in fixed}
        shares = largest_remainder(free, total - sum(fixed.values()))
        over = {k for k, n in shares.items() if n > capacity[k]}
        if not over:
            return {**fixed, **shares}
        fixed.update({k: capacity[k] for k in over})


def party_quotas(counts: dict[str, int], total: int, cap: float = PARTY_CAP) -> dict[str, int]:
    """Quotas proportional to `counts`, no party above `cap` of `total`; the excess is
    spread over the parties below the cap in proportion to their counts."""
    limit = int(cap * total + 1e-9)
    return allocate(counts, total, dict.fromkeys(counts, limit))


@dataclass
class Draw:
    candidates: list[dict]
    counts: Counter = field(default_factory=Counter)
    quotas: dict[str, int] = field(default_factory=dict)
    shortfall: dict[str, int] = field(default_factory=dict)
    dropped: list[dict] = field(default_factory=list)
    # Turns per party in the population and among eligible turns.
    population: dict[str, int] = field(default_factory=dict)
    eligible: dict[str, int] = field(default_factory=dict)
    # Turns per calendar year: population, eligible.
    years: dict[str, list[int]] = field(default_factory=dict)


def turn_id(turn: dict) -> str:
    return f"{turn['file_id']}:{turn['speech_id_within_doc']}"


def candidate(turn: dict, spans: list[tuple[int, int]], seed: int) -> dict:
    """One sentence of the turn, at a seeded position among its eligible sentences."""
    tid = turn_id(turn)
    index = random.Random(f"{seed}:sentence:{tid}").randrange(len(spans))
    start, end = spans[index]
    raw = turn["speech_text"][start:end]
    plain = " ".join(raw.split())
    text = mask_names(plain)
    digest = sha256(text)
    date = turn["tarih"]
    return {
        "id": f"tbmm-{digest[:16]}",
        "text": text,
        "meta": {
            "id": f"tbmm-{digest[:16]}",
            "source": f"{SOURCE} ({FILE_NAME})",
            "licence": LICENCE,
            "record": RECORD,
            "turn_id": tid,
            "file_id": turn["file_id"],
            "speech_id_within_doc": turn["speech_id_within_doc"],
            "sitting": sitting_of(turn["file_id"]),
            "date": date.date().isoformat() if isinstance(date, dt.datetime) else str(date),
            "legislative_year": legislative_year(date),
            "term": _number(turn.get("tbmm_donemi")),
            "term_year": _number(turn.get("yasama_yili")),
            "chamber": turn["meclis_turu"],
            "corpus_source": turn["source"],
            "speaker": turn["speaker_name_raw"],
            "speaker_roster_name": turn.get("wiki_name"),
            "speaker_role": turn.get("speaker_role_raw"),
            "party": turn["wiki_parti"],
            "char_start": start,
            "char_end": end,
            "sentence_index": index,
            "eligible_sentences": len(spans),
            "line_breaks_joined": raw != plain,
            "masked": text != plain,
            "text_sha256": digest,
            "split": split_of(text),
        },
    }


SITTING = re.compile(r"_b(\d+)$")


def sitting_of(file_id: str) -> int | None:
    """The sitting (birleşim) number, which the transcript files carry in their id
    ("d27_y2_b21"); the OCR files have a bare number and no sitting."""
    match = SITTING.search(file_id)
    return int(match.group(1)) if match else None


def _number(value) -> int | None:
    if value is None or value != value:  # NaN
        return None
    return int(value)


def verify(meta: dict, text: str, turn_text: str) -> None:
    """The candidate is its turn's text at the recorded offsets, up to spaces and masks."""
    raw = turn_text[meta["char_start"] : meta["char_end"]]
    if not raw or raw != raw.strip() or raw not in turn_text:
        raise AssertionError(f"{meta['id']}: no sentence at the recorded offsets")
    if mask_names(" ".join(raw.split())) != text:
        raise AssertionError(f"{meta['id']}: text is not its turn's sentence")
    if sha256(text) != meta["text_sha256"]:
        raise AssertionError(f"{meta['id']}: text hash does not match")


def draw(
    turns: list[dict],
    *,
    seed: int = SEED,
    target: int = TARGET,
    cap: float = PARTY_CAP,
    overlapping: Callable[[list[dict]], dict[str, str]] = lambda cands: {},
) -> Draw:
    """The seeded draw: party quotas, cells, one sentence per turn, overlap dropped.

    Quotas follow the population's turns, not the eligible ones, so the years whose
    OCR loses more sentences keep their share; a cell never gets more than its
    eligible turns, and what it cannot take goes to the party's other cells.
    """
    counts: Counter = Counter()
    vocab = vocabulary(turns)
    population_cells: Counter = Counter()
    by_cell: dict[tuple[str, str], list[tuple[dict, list]]] = defaultdict(list)
    years: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for turn in sorted(turns, key=turn_id):
        key = (turn["wiki_parti"], cell_of(turn))
        population_cells[key] += 1
        spans = eligible(turn, counts, vocab)
        counts["turns"] += 1
        years[str(turn["tarih"].year)][0] += 1
        if spans:
            counts["eligible_turns"] += 1
            years[str(turn["tarih"].year)][1] += 1
            by_cell[key].append((turn, spans))
    per_party = Counter()
    for (party, _cell), n in population_cells.items():
        per_party[party] += n
    quotas = party_quotas(dict(sorted(per_party.items())), target, cap)
    pools: dict[tuple[str, str], list[dict]] = {}
    wanted: dict[tuple[str, str], int] = {}
    for party, quota in sorted(quotas.items()):
        cells = {c: n for (p, c), n in sorted(population_cells.items()) if p == party}
        room = {c: len(by_cell.get((party, c), [])) for c in cells}
        shares = allocate(cells, quota, room)
        if sum(shares.values()) < quota:
            counts[f"party_short:{party}"] = quota - sum(shares.values())
        for cell, q in shares.items():
            if q == 0:
                continue
            members = list(by_cell[(party, cell)])
            random.Random(f"{seed}:order:{party}:{cell}").shuffle(members)
            size = min(len(members), q + max(RESERVE_MIN, int(RESERVE_SHARE * q + 0.999)))
            pools[(party, cell)] = [candidate(t, s, seed) for t, s in members[:size]]
            wanted[(party, cell)] = q
    pooled = [c for pool in pools.values() for c in pool]
    dropped_ids = overlapping(pooled)
    out, seen, dropped, shortfall = [], set(), [], {}
    for key in sorted(pools):
        taken = 0
        for cand in pools[key]:
            if taken == wanted[key]:
                break
            reason = dropped_ids.get(cand["id"])
            if reason is None and cand["meta"]["text_sha256"] in seen:
                reason = "repeated_text"
            if reason:
                dropped.append({"id": cand["id"], "turn_id": cand["meta"]["turn_id"],
                                "reason": reason})  # fmt: skip
                continue
            seen.add(cand["meta"]["text_sha256"])
            out.append(cand)
            taken += 1
        if taken < wanted[key]:
            shortfall["/".join(key)] = wanted[key] - taken
    counts["pooled"] = len(pooled)
    counts["overlap_in_pool"] = len(dropped_ids)
    eligible_per_party = Counter()
    for (party, _cell), members in by_cell.items():
        eligible_per_party[party] += len(members)
    return Draw(out, counts, quotas, shortfall, dropped, dict(sorted(per_party.items())),
                dict(sorted(eligible_per_party.items())), dict(sorted(years.items())))  # fmt: skip


# input and output ---------------------------------------------------------------


def fetch(folder: Path = RAW) -> Path:
    from data.typed.common import fetch_pinned

    return fetch_pinned({FILE_NAME: (URL, SHA256)}, folder)[FILE_NAME]


def population(rows: Iterable[dict], counts: Counter) -> list[dict]:
    """Plenary turns from 2011 on with a speaker and a party."""
    out = []
    for row in rows:
        counts["rows_read"] += 1
        date = row.get("tarih")
        if date is None or date < FROM:
            counts["before_2011"] += 1
        elif row.get("meclis_turu") not in CHAMBERS:
            counts["not_plenary_chamber"] += 1
        elif not row.get("wiki_parti") or not row.get("speaker_name_raw"):
            counts["no_speaker_or_party"] += 1
        elif not (row.get("speech_text") or "").strip():
            counts["empty_text"] += 1
        else:
            out.append(row)
    counts["population_turns"] = len(out)
    return out


def read_turns(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=COLUMNS).to_pylist()


def facturk_rows(archive: Path, target: Path) -> Path:
    """FACTurk's claims as rows the overlap check reads (its `content` is never read)."""
    from data.typed.claims import read_csv

    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"row_id": f"facturk:{i}", "state": record["claim"]}, ensure_ascii=False)
        for i, record in enumerate(read_csv(archive))
        if (record.get("claim") or "").strip()
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def overlap_check(files: list[Path]) -> Callable[[list[dict]], dict[str, str]]:
    def check(cands: list[dict]) -> dict[str, str]:
        rows = [{"id": c["id"], "text": c["text"]} for c in cands]
        found: dict[str, str] = {}
        for match in training_overlap(rows, files):
            source = "facturk" if match["row_id"].startswith("facturk:") else "training"
            for ref in match["refs"]:
                found.setdefault(ref, f"overlap_{source}_{match['rule']}")
        return found

    return check


def write(result: Draw, out: Path) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    ordered = sorted(result.candidates, key=lambda c: c["id"])
    items = [
        Item(id=c["id"], track=TRACK, state=c["text"], questions=QUESTIONS).model_dump(
            mode="json", exclude_none=True
        )
        for c in ordered
    ]
    items_path, meta_path = out / f"{TRACK}.jsonl", out / f"{TRACK}.meta.jsonl"
    items_path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items),
                          encoding="utf-8")  # fmt: skip
    meta_path.write_text("".join(json.dumps(c["meta"], ensure_ascii=False) + "\n"
                                 for c in ordered), encoding="utf-8")  # fmt: skip
    return items_path, meta_path


def report(result: Draw, population_counts: Counter, files: list[Path], parquet: Path) -> dict:
    metas = [c["meta"] for c in result.candidates]
    return {
        "source": SOURCE,
        "file": FILE_NAME,
        "file_sha256": SHA256,
        "parquet": str(parquet),
        "seed": SEED,
        "target": TARGET,
        "party_cap": PARTY_CAP,
        "words": [MIN_WORDS, MAX_WORDS],
        "population": dict(population_counts),
        "sentences": {k: result.counts[k] for k in ("sentences", *REASONS, "eligible_sentences")},
        "turns": {k: result.counts[k] for k in ("turns", "cut_at_speaker_line", "eligible_turns")},
        "turns_per_year_population_eligible": result.years,
        "population_turns_per_party": result.population,
        "eligible_turns_per_party": result.eligible,
        "party_quotas": result.quotas,
        "party_short": {
            k.split(":", 1)[1]: v for k, v in result.counts.items() if k.startswith("party_short:")
        },
        "pooled": result.counts["pooled"],
        "overlap_in_pool": result.counts["overlap_in_pool"],
        "dropped_while_filling": dict(Counter(d["reason"] for d in result.dropped)),
        "shortfall": result.shortfall,
        "candidates": len(metas),
        "per_party": dict(sorted(Counter(m["party"] for m in metas).items())),
        "per_calendar_year": dict(sorted(Counter(m["date"][:4] for m in metas).items())),
        "per_legislative_year": dict(sorted(Counter(m["legislative_year"] for m in metas).items())),
        "per_corpus_source": dict(sorted(Counter(m["corpus_source"] for m in metas).items())),
        "per_split": dict(sorted(Counter(m["split"] for m in metas).items())),
        "masked": sum(m["masked"] for m in metas),
        "line_breaks_joined": sum(m["line_breaks_joined"] for m in metas),
        "overlap_files_checked": [str(p) for p in files],
    }


def main(argv: list[str] | None = None) -> int:
    from model.head.mixing import train_files, validation_files

    parser = argparse.ArgumentParser(prog="bench.hakembench.parliament")
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--data", type=Path, default=Path("data/built"))
    parser.add_argument("--facturk", type=Path, default=FACTURK)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    parquet = fetch(args.raw)
    files = [*train_files(args.data), *validation_files(args.data)]
    if not files:
        print(f"no training files under {args.data}; the overlap check cannot run")
        return 1
    files.append(facturk_rows(args.facturk, CACHE / "facturk_claims.jsonl"))

    population_counts: Counter = Counter()
    turns = population(read_turns(parquet), population_counts)
    result = draw(turns, overlapping=overlap_check(files))
    by_id = {turn_id(t): t["speech_text"] for t in read_turns_by_id(parquet, result)}
    for cand in result.candidates:
        verify(cand["meta"], cand["text"], by_id[cand["meta"]["turn_id"]])
    write(result, args.out)
    summary = report(result, population_counts, files, parquet)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")  # fmt: skip
    print(json.dumps({k: v for k, v in summary.items() if k != "overlap_files_checked"},
                     ensure_ascii=False, indent=2))  # fmt: skip
    return 0 if len(result.candidates) == TARGET else 1


def read_turns_by_id(parquet: Path, result: Draw) -> list[dict]:
    """The drawn turns read again from the file, for the verbatim check."""
    wanted = {c["meta"]["turn_id"] for c in result.candidates}
    return [row for row in read_turns(parquet) if turn_id(row) in wanted]


if __name__ == "__main__":
    sys.exit(main())
