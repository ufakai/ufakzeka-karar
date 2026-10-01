"""Asking a judge LLM which option fits, and reading its answer as a distribution.

A judge sees the text, the question and the options, each option behind a
letter, and writes one letter. Its first-token log-probabilities over those
letters, renormalised, are its vote; the row's target is the arithmetic mean
of the votes (schema.rows.mean_vote).

Option text is normalised before it is shown: markdown symbols stripped,
whitespace collapsed, one line per option. The 2026 judge-bias study measured
formatting bias well above position bias (arXiv 2604.23178), so no
option may stand out by how it is written. Option order is rotated across the
judges on choice and noul questions; score levels keep their natural order,
because a shuffled rubric would no longer read as a scale.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from schema.questions import (
    HEAD_OPTIONS_PER_PASS,
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
)
from schema.rows import JudgeName, JudgeVote, outcomes, text_of

# One letter per option, so a question shown to a judge has at most ten options.
LETTERS = "ABCDEFGHIJ"
if len(LETTERS) != HEAD_OPTIONS_PER_PASS:
    raise ImportError("LETTERS must cover exactly the options one pass scores")

# A call is rejected when less than this much probability lands on the letters.
MIN_LETTER_MASS = 0.9
# The chat-completions limit on top log-probabilities per token.
MAX_TOP_LOGPROBS = 20

SYSTEM_PROMPT = (
    "Sen dikkatli ve tarafsız bir değerlendiricisin. Metni ve soruyu oku, en doğru "
    "seçeneğin harfini yaz. Yalnızca tek bir harf yaz, açıklama yapma."
)
USER_TEMPLATE = (
    "Metin:\n{state}\n\nSoru: {instructions}\n\nSeçenekler:\n{lines}\n\nCevap (yalnızca harf):"
)

NOUL_LABELS = {"true": "Evet", "false": "Hayır"}


class NoLogprobs(ValueError):
    """The response carries no first-token top log-probabilities."""


class LowLabelMass(ValueError):
    """Too little probability landed on the offered letters for the vote to mean anything."""

    def __init__(self, mass: float) -> None:
        super().__init__(f"letter mass {mass:.4f} is under {MIN_LETTER_MASS}")
        self.mass = mass
        # Set by label_row, so a build can count failures per judge.
        self.judge: str | None = None


@dataclass(frozen=True)
class JudgeSpec:
    name: JudgeName
    model: str
    provider: str
    # "logprobs": the first-token log-probabilities over the letters. "stated":
    # the judge writes a probability for each letter as JSON, for a family that
    # returns no log-probabilities. Stated probabilities are a weaker
    # measurement, which the scoring against human labels has to justify.
    mode: str = "logprobs"
    # Reasoning effort for a judge whose endpoint will not turn reasoning off.
    reasoning: str | None = "none"

    def __post_init__(self) -> None:
        if self.name not in ("A", "B", "C", "D"):
            raise ValueError(f"judge name must be A, B, C or D, got {self.name!r}")


_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_QUOTE = re.compile(r"^\s*>+\s?", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_EMPHASIS = re.compile(r"[*`~]+")
# An underscore inside a word, as in a snake_case name, is kept.
_UNDERSCORE = re.compile(r"(?<!\w)_+|_+(?!\w)")
_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Plain text on one line: markdown symbols stripped, whitespace collapsed."""
    text = _LINK.sub(r"\1", text)
    text = _HEADING.sub("", text)
    text = _QUOTE.sub("", text)
    text = _BULLET.sub("", text)
    text = _EMPHASIS.sub("", text)
    text = _UNDERSCORE.sub("", text)
    return _SPACE.sub(" ", text).strip()


def _check_order(question: Question, shown_order: list[str]) -> None:
    keys = outcomes(question)
    if sorted(shown_order) != sorted(keys) or len(shown_order) != len(keys):
        raise ValueError(f"shown_order {shown_order} is not a permutation of {keys}")
    if len(keys) > len(LETTERS):
        raise ValueError(f"a judge is shown at most {len(LETTERS)} options, got {len(keys)}")


def _described(label: str, description: str | None) -> str:
    description = normalise(description) if description else ""
    return f"{label}: {description}" if description else label


def option_lines(question: Question, shown_order: list[str]) -> list[str]:
    """What the judge sees for each option, in `shown_order`, one line each."""
    _check_order(question, shown_order)
    lines = []
    for letter, key in zip(LETTERS, shown_order, strict=False):
        if isinstance(question, ChoiceQuestion):
            text = _described(normalise(key), question.criteria[key])
        elif isinstance(question, NoulQuestion):
            criteria = question.criteria
            description = getattr(criteria, key) if criteria is not None else None
            text = _described(NOUL_LABELS[key], description)
        elif isinstance(question, ScoreQuestion):
            text = f"Düzey {key}: {normalise(question.criteria[int(key)])}"
        else:
            raise TypeError(f"unknown question type {type(question).__name__}")
        lines.append(f"{letter}) {text}")
    return lines


def judge_messages(state_text: str, question: Question, shown_order: list[str]) -> list[dict]:
    """The system and user messages for one judge call, in Turkish."""
    user = USER_TEMPLATE.format(
        state=state_text.strip(),
        instructions=text_of(question.instructions).strip(),
        lines="\n".join(option_lines(question, shown_order)),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


STATED_TEMPLATE = (
    "Metin:\n{state}\n\nSoru: {instructions}\n\nSeçenekler:\n{lines}\n\n"
    "Her seçeneğin doğru cevap olma olasılığını ver. Olasılıkların toplamı 1 olsun. "
    'Yalnızca JSON yaz, örneğin {{"A": 0.7, "B": 0.3}}.'
)


def stated_messages(state_text: str, question: Question, shown_order: list[str]) -> list[dict]:
    """The request for a judge that states its probabilities instead of exposing them."""
    user = STATED_TEMPLATE.format(
        state=state_text.strip(),
        instructions=text_of(question.instructions).strip(),
        lines="\n".join(option_lines(question, shown_order)),
    )
    return [{"role": "system", "content": SYSTEM_PROMPT_STATED}, {"role": "user", "content": user}]


SYSTEM_PROMPT_STATED = (
    "Sen dikkatli ve tarafsız bir değerlendiricisin. Metni ve soruyu oku ve her seçeneğin "
    "doğru olma olasılığını dürüstçe tahmin et. Emin değilsen olasılığı paylaştır."
)


class NoStatedDistribution(ValueError):
    """The judge did not return a usable JSON object of probabilities."""


def parse_stated(
    response: dict[str, Any], shown_order: list[str]
) -> tuple[dict[str, float], float]:
    """A stated distribution over the option keys, renormalised, and its missing mass.

    Letters the judge left out get 0; a letter outside the offered ones, a
    negative or non-numeric value, or a total far from one rejects the call.
    """
    letters = LETTERS[: len(shown_order)]
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise NoStatedDistribution("no message content") from error
    text = (content or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise NoStatedDistribution(f"no JSON object in {text[:80]!r}")
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise NoStatedDistribution(f"invalid JSON: {error}") from error
    if not isinstance(raw, dict) or not raw:
        raise NoStatedDistribution("the JSON is not a non-empty object")
    values = {}
    for key, value in raw.items():
        letter = _letter_of(key)
        if letter not in letters:
            raise NoStatedDistribution(f"unknown option {key!r}")
        if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
            raise NoStatedDistribution(f"bad probability {value!r} for {key!r}")
        values[letter] = values.get(letter, 0.0) + float(value)
    total = math.fsum(values.values())
    if not 0.9 <= total <= 1.1:
        raise LowLabelMass(total)
    missing = min(1.0, max(0.0, 1.0 - total))
    return {
        key: values.get(letter, 0.0) / total
        for letter, key in zip(letters, shown_order, strict=True)
    }, missing


def _letter_of(token: Any) -> str | None:
    """The option letter a token stands for, or None.

    " A", "a", "(B" and "C." all count. Only ASCII is accepted, because Python
    upper-cases the Turkish dotless "ı" to "I", which is an option letter.
    """
    if not isinstance(token, str):
        return None
    token = token.strip()
    token = token.removeprefix("(")
    token = token.removesuffix(")").removesuffix(".")
    token = token.strip()
    if len(token) != 1 or not token.isascii():
        return None
    return token.upper()


def parse_distribution(
    response: dict[str, Any], shown_order: list[str]
) -> tuple[dict[str, float], float]:
    """A judge's distribution over the option keys, and its off-letter mass.

    Probabilities of tokens that name the same letter are added. Every key is
    present; a letter absent from the top log-probabilities gets 0.
    """
    if len(shown_order) > len(LETTERS):
        raise ValueError(f"at most {len(LETTERS)} options, got {len(shown_order)}")
    if len(set(shown_order)) != len(shown_order):
        raise ValueError(f"shown_order repeats a key: {shown_order}")
    letters = LETTERS[: len(shown_order)]
    try:
        top = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError) as error:
        raise NoLogprobs("no first-token top_logprobs in the response") from error
    if not isinstance(top, list) or not top:
        raise NoLogprobs("top_logprobs is empty")

    mass_by_letter = dict.fromkeys(letters, 0.0)
    for entry in top:
        if not isinstance(entry, dict):
            raise NoLogprobs(f"malformed top_logprobs entry {entry!r}")
        logprob = entry.get("logprob")
        if isinstance(logprob, bool) or not isinstance(logprob, int | float):
            raise NoLogprobs(f"malformed logprob in {entry!r}")
        letter = _letter_of(entry.get("token"))
        if letter in mass_by_letter:
            mass_by_letter[letter] += math.exp(logprob)

    mass = math.fsum(mass_by_letter.values())
    if not mass >= MIN_LETTER_MASS:
        raise LowLabelMass(mass)
    off_letter_mass = min(1.0, max(0.0, 1.0 - mass))
    distribution = {
        key: mass_by_letter[letter] / mass for letter, key in zip(letters, shown_order, strict=True)
    }
    return distribution, off_letter_mass


def _rank(seed: str, key: str) -> bytes:
    return hashlib.blake2b(f"{seed}\x00{key}".encode(), digest_size=16).digest()


def shown_orders(
    question: Question, keys: list[str], judges: list[JudgeSpec], seed: str
) -> dict[str, list[str]]:
    """The option order each judge sees, reproducible from `seed` (the row id).

    Choice and noul: the keys are shuffled once by a hash of the seed and each
    key, and judge i sees rotation i of that order. The hash, not the random
    module, fixes the shuffle, so the orders do not change with the Python
    version. Score: every judge sees the levels in their natural order.
    """
    if sorted(keys) != sorted(outcomes(question)) or len(keys) != len(set(keys)):
        raise ValueError(f"keys {keys} are not the question's outcomes")
    names = [judge.name for judge in judges]
    if len(set(names)) != len(names):
        raise ValueError(f"a judge appears twice: {names}")
    if isinstance(question, ScoreQuestion):
        natural = sorted(keys, key=int)
        return {name: list(natural) for name in names}
    base = sorted(keys, key=lambda key: _rank(seed, key))
    orders = {}
    for i, name in enumerate(names):
        shift = i % len(base)
        orders[name] = base[shift:] + base[:shift]
    return orders


def _parsed(parse, response, order, judge: JudgeSpec):
    try:
        return parse(response, order)
    except LowLabelMass as error:
        error.judge = judge.name
        raise


def label_row(
    client: Any,
    judges: list[JudgeSpec],
    row_id: str,
    state_text: str,
    question: Question,
) -> list[JudgeVote]:
    """Every judge's vote on one row. Any failure is raised, so no partial row is written."""
    keys = outcomes(question)
    if len(keys) > len(LETTERS):
        raise ValueError(f"a judged row has at most {len(LETTERS)} options, got {len(keys)}")
    orders = shown_orders(question, keys, judges, row_id)
    top_logprobs = min(max(len(keys), 10), MAX_TOP_LOGPROBS)
    votes = []
    for judge in judges:
        order = orders[judge.name]
        if judge.mode == "stated":
            response = client.chat(
                judge.model,
                judge.provider,
                stated_messages(state_text, question, order),
                max_tokens=600,
                reasoning_effort=judge.reasoning,
                temperature=0.0,
                # Its endpoint will not turn reasoning off; the cost is counted.
                forbid_reasoning_tokens=False,
            )
            distribution, off_letter_mass = _parsed(parse_stated, response, order, judge)
        else:
            response = client.chat(
                judge.model,
                judge.provider,
                judge_messages(state_text, question, order),
                max_tokens=1,
                logprobs=True,
                top_logprobs=top_logprobs,
                reasoning_effort=judge.reasoning,
                temperature=0.0,
            )
            distribution, off_letter_mass = _parsed(parse_distribution, response, order, judge)
        votes.append(
            JudgeVote(
                judge=judge.name,
                shown_order=order,
                distribution=distribution,
                off_letter_mass=off_letter_mass,
            )
        )
    return votes
