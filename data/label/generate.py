"""Task templates written by the generator, checked before any judge sees them.

A template is one decision question with its options, written to apply to many
texts of one kind. The generator writes templates, not rows: the
pipeline instantiates each over sampled texts and the judges label every pair,
so generation is a small share of the cost and judges set it.

Every template is checked for the item-writing flaws that 2026 studies of
generated multiple choice measured before it is used:

- an option set that is too small or too large, or repeats itself;
- "all of the above", "none of the above", "both": options defined by other
  options, which blind options cannot represent and which are banned;
- absolute terms ("her zaman", "asla"), which make an option trivially wrong;
- one option much longer than the rest, the longest-answer cue;
- an option name repeated inside the question, which gives the answer away;
- a score rubric without enough levels, or a yes-or-no question without both
  meanings written out;
- two options whose words mostly coincide, or an intent question whose options
  are topics and not actions;
- a question nearly the same as an earlier template's, checked by the build.

Turkish words are compared by their first four letters, a crude stem that
lets "kişi" and "kişinin" meet. Overlaps these rules miss go to the one-call
template check of data/label/build.py.

A flagged template is dropped, not repaired, and the count goes in the report.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from data.label.judge import normalise

FAMILIES: dict[str, str] = {
    "konu": "sorunun hangi konu ya da birimle ilgili olduğu",
    "niyet": "soruyu soran kişinin ne yapmak ya da öğrenmek istediği",
    "yanit_yeterliligi": "verilen cevabın soruyu ne ölçüde karşıladığı",
    "aciliyet": "sorunun ne kadar hızlı ele alınması gerektiği",
    "bilgi_turu": "cevabın verdiği bilginin türü (fiyat, süreç, teknik özellik, kural gibi)",
    "insan_destegi": "sorunun bir insan temsilciye aktarılması gerekip gerekmediği",
    "hassasiyet": "konunun parasal, hukuki ya da sağlıkla ilgili bir risk taşıyıp taşımadığı",
    "hedef_kitle": "cevabın kime yönelik olduğu",
}

BANNED = re.compile(
    r"\b(hepsi|hiçbiri|yukarıdaki|her ikisi|ikisi de|tümü|hiçbirisi)\b", re.IGNORECASE
)
ABSOLUTE = re.compile(r"\b(her zaman|asla|hiçbir zaman|kesinlikle|daima)\b", re.IGNORECASE)
MAX_LENGTH_RATIO = 4.0
MAX_OPTION_OVERLAP = 0.5
MAX_QUESTION_OVERLAP = 0.6
# An intent is something the asker wants to do: a verb or a verbal noun.
ACTION = re.compile(r"(mek|mak|me|ma|mesi|ması)$")
_FILLER = "ve veya ile ya da bir için gibi olan"
FILLER = frozenset(_FILLER.split())


def stems(text: str) -> set[str]:
    return {w[:4] for w in re.findall(r"\w+", text.lower()) if w not in FILLER and len(w) > 1}


def overlap(a: str, b: str) -> float:
    """Jaccard similarity of two texts' crude stems."""
    x, y = stems(a), stems(b)
    return len(x & y) / len(x | y) if x | y else 0.0


class Option(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    description: str | None = Field(default=None, max_length=200)


class Template(BaseModel):
    family: str
    type: Literal["choice", "noul", "score"]
    question: str = Field(min_length=8, max_length=300)
    applies_when: str = Field(min_length=8, max_length=300)
    options: list[Option] = Field(default_factory=list)
    true: str | None = None
    false: str | None = None
    levels: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _shape(self) -> Template:
        if self.type == "choice" and not 3 <= len(self.options) <= 8:
            raise ValueError(f"a choice template needs 3 to 8 options, got {len(self.options)}")
        if self.type == "noul" and not (self.true and self.false):
            raise ValueError("a yes-or-no template needs both meanings")
        if self.type == "score" and not 3 <= len(self.levels) <= 5:
            raise ValueError(f"a score template needs 3 to 5 levels, got {len(self.levels)}")
        return self


def flaws(template: Template) -> list[str]:
    """Every item-writing flaw found, empty if the template is clean."""
    found = []
    texts = [o.name for o in template.options] + template.levels
    texts += [t for t in (template.true, template.false) if t]
    for text in texts + [o.description or "" for o in template.options]:
        if BANNED.search(text):
            found.append(f"option defined by other options: {text!r}")
        if ABSOLUTE.search(text):
            found.append(f"absolute term: {text!r}")
    for option in template.options:
        if "_" in option.name:
            found.append(f"option name is not natural Turkish: {option.name!r}")
    if template.type == "choice":
        names = [normalise(o.name).lower() for o in template.options]
        if len(set(names)) != len(names):
            found.append("repeated option")
        lengths = [
            len(normalise(o.name)) + len(normalise(o.description or "")) for o in template.options
        ]
        if min(lengths) and max(lengths) / min(lengths) > MAX_LENGTH_RATIO:
            found.append("one option far longer than the others")
        question = template.question.lower()
        for name in names:
            if len(name) > 3 and name in question:
                found.append(f"option named in the question: {name!r}")
        full = [f"{o.name} {o.description or ''}" for o in template.options]
        for i, a in enumerate(full):
            for b in full[i + 1 :]:
                if overlap(a, b) > MAX_OPTION_OVERLAP:
                    found.append(f"options overlap: {a[:30]!r} and {b[:30]!r}")
        if template.family == "niyet":
            actions = sum(bool(ACTION.search(name.split()[-1])) for name in names if name)
            if actions * 2 < len(names):
                found.append("intent options are topics, not actions")
    return found


def to_question(template: Template) -> dict[str, Any]:
    """The template as a typed question (schema/questions.py)."""
    if template.type == "choice":
        return {
            "type": "choice",
            "instructions": template.question,
            "criteria": {normalise(o.name): (normalise(o.description) if o.description else None)
                         for o in template.options},
        }  # fmt: skip
    if template.type == "noul":
        return {
            "type": "noul",
            "instructions": template.question,
            "criteria": {"true": normalise(template.true), "false": normalise(template.false)},
        }
    return {"type": "score", "instructions": template.question,
            "criteria": [normalise(level) for level in template.levels]}  # fmt: skip


SYSTEM = "Türkçe karar soruları yazan dikkatli bir uzmansın. Yalnızca istenen JSON nesnesini yaz."


def generator_messages(family: str, kind: str, examples: list[str]) -> list[dict]:
    """The request for one template: its family, its type, and three example texts."""
    shape = {
        "choice": '"options": [{"name": "...", "description": "..."}], 3 ile 8 arası seçenek',
        "noul": '"true": "evet ne demek", "false": "hayır ne demek"',
        "score": '"levels": ["en düşük düzey", "...", "en yüksek düzey"], 3 ile 5 düzey',
    }[kind]
    shown = "\n\n".join(f"Örnek {i + 1}:\n{text}" for i, text in enumerate(examples))
    user = f"""Aşağıdaki örnekler, şirketlerin sıkça sorulan sorular sayfalarından
ve topluluk soru-cevap sitelerinden alınmış soru ve cevaplardır.

{shown}

Bu türden metinlerin çoğuna uygulanabilecek TEK bir karar sorusu yaz.
Konu ailesi: {FAMILIES[family]}.
Soru türü: {kind}.

Kurallar:
- Soru, herhangi bir tek metne değil, bu tür metinlerin çoğuna uygulanabilmeli.
- Seçenekler birbirinden bağımsız ve birbirini dışlayan kategoriler olmalı.
- Cevap metinden metne değişmeli: metinlerin çoğunda aynı seçeneği seçtiren soru yazma.
- "Hepsi", "hiçbiri", "her ikisi" gibi başka seçeneklere dayanan seçenek yazma.
- "Her zaman", "asla" gibi mutlak ifadeler kullanma.
- Seçeneklerin uzunlukları birbirine yakın olsun.
- Seçenek adlarını soru metninde tekrarlama.
- Seçenek adlarını doğal Türkçe kelimelerle ve Türkçe harflerle yaz
  (ç, ğ, ı, ö, ş, ü); alt çizgi kullanma.
- Doğal, sade ve doğru bir Türkçe kullan.

Şu alanları içeren bir JSON nesnesi yaz:
"family": "{family}", "type": "{kind}", "question": "soru metni",
"applies_when": "bu sorunun hangi metinlere uygulanabileceği",
{shape}"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


@dataclass
class Parsed:
    template: Template | None
    problems: list[str]


def parse_template(content: str, family: str | None = None, kind: str | None = None) -> Parsed:
    """The generator's text as a checked template, or the reasons it was not one."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        return Parsed(None, [f"not JSON: {error}"])
    try:
        template = Template.model_validate(data)
    except ValidationError as error:
        return Parsed(None, [f"invalid template: {error.errors()[0]['msg']}"])
    # The family is ours to assign, never the generator's to rename;
    # a template of another type than asked is a different template and dropped.
    if family is not None:
        template = template.model_copy(update={"family": family})
    if kind is not None and template.type != kind:
        return Parsed(None, [f"asked for {kind}, got {template.type}"])
    if template.family not in FAMILIES:
        return Parsed(None, [f"unknown family {template.family!r}"])
    found = flaws(template)
    return Parsed(None if found else template, found)
