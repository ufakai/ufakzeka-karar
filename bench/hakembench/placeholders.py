"""Masking placeholders and the case suffix after them.

A mask replaces a name but not the suffix the writer attached to it, so
"Kule Kafe'nin" became "[şirket]'nin". Turkish suffixes follow the last
vowel and the last sound of the word they attach to, so after a mask the
suffix should follow the placeholder's own word: "[şirket]'in", "[ad]'ın",
"[ilçe]'nin". `harmonise` rewrites the first case suffix after each
placeholder and keeps whatever follows it ("'ndaki" becomes "'teki").
"""

from __future__ import annotations

import re

# One token per entity type; the benchmark card lists them.
PLACEHOLDERS = (
    "ad", "şirket", "kurum", "ilçe", "ürün", "bağlantı", "telefon", "hesap no", "plaka", "adres",
    "kimlik no", "vergi no", "kullanıcı adı", "e-posta", "tarih", "tutar", "kod", "şifre",
)  # fmt: skip

BACK = set("aıou")
ROUND = set("ouöü")
VOWELS = set("aeıioöuü")
HARD = set("çfhkpsşt")

# First case morpheme after the apostrophe, longest alternatives first.
CASES = (
    ("ablative", re.compile(r"n?[dt][ae]n")),
    ("genitive", re.compile(r"n?[ıiuü]n")),
    ("locative", re.compile(r"n?[dt][ae]")),
    ("instrumental", re.compile(r"y?l[ae]")),
    ("dative", re.compile(r"[ny]?[ae]")),
    ("accusative", re.compile(r"[ny]?[ıiuü]")),
)
AFTER = re.compile(r"\[(" + "|".join(re.escape(p) for p in PLACEHOLDERS) + r")\]'([a-zçğıöşü]+)")


def _word(placeholder: str) -> str:
    return placeholder.split()[-1]


def _vowels(word: str) -> tuple[bool, bool]:
    last = [c for c in word if c in VOWELS][-1]
    return last in BACK, last in ROUND


def suffix(placeholder: str, case: str) -> str:
    """The case suffix that follows `placeholder`'s own word."""
    word = _word(placeholder)
    back, rounded = _vowels(word)
    a = "a" if back else "e"
    i = ("u" if rounded else "ı") if back else ("ü" if rounded else "i")
    ends_vowel = word[-1] in VOWELS
    d = "t" if word[-1] in HARD else "d"
    return {
        "genitive": ("n" if ends_vowel else "") + i + "n",
        "dative": ("y" if ends_vowel else "") + a,
        "accusative": ("y" if ends_vowel else "") + i,
        "locative": d + a,
        "ablative": d + a + "n",
        "instrumental": ("y" if ends_vowel else "") + "l" + a,
    }[case]


def harmonise(text: str) -> str:
    def fix(m: re.Match) -> str:
        placeholder, rest = m.group(1), m.group(2)
        for case, pattern in CASES:
            head = pattern.match(rest)
            if head:
                tail = _tail(placeholder, case, rest[head.end() :])
                return f"[{placeholder}]'{suffix(placeholder, case)}{tail}"
        return m.group(0)

    return AFTER.sub(fix, text)


def _tail(placeholder: str, case: str, tail: str) -> str:
    """The rest of the suffix, its vowels following the new case suffix; "-ki" never changes."""
    back, rounded = _vowels(suffix(placeholder, case))
    out = []
    for n, c in enumerate(tail):
        if tail[max(0, n - 1) : n + 1] == "ki" or tail[n : n + 2] == "ki":
            out.append(c)
            continue
        if c in "ae":
            c = "a" if back else "e"
        elif c in "ıiuü":
            c = ("u" if rounded else "ı") if back else ("ü" if rounded else "i")
        if c in VOWELS:
            back, rounded = c in BACK, c in ROUND
        out.append(c)
    return "".join(out)


def unharmonised(text: str) -> list[str]:
    """Placeholder suffixes that `harmonise` would change."""
    fixed = harmonise(text)
    if fixed == text:
        return []
    return [m.group(0) for m in AFTER.finditer(text) if harmonise(m.group(0)) != m.group(0)]
