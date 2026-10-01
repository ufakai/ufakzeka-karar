"""One training row: a state, one typed question, and the distribution to learn.

A row is one question (docs/PLAN.md, "one row per question in training", so
packing cannot bias answers). Its target is a probability distribution over
the question's outcomes, keyed the way the answer contract keys them
(schema/questions.py), so the same code reads a target and a prediction:

- choice: the option names, as in `ChoiceQuestion.criteria`;
- score: the level indices "0" to "K-1", as in `ScoreAnswer.probabilities`;
- noul: "true" and "false", as in `NoulCriteria`.

Where the target came from is kept beside it, because the whole recipe depends
on it: a human label from a converted set is a one-hot target; a
labelled row carries each judge's own distribution and the option order that
judge was shown, so the vote can be recomputed, audited for position bias, and
re-aggregated if the rule changes, without paying for the calls again.

A choice question may carry up to 255 options (the API limit), but a training
row carries at most ten (schema/questions.py HEAD_OPTIONS_PER_PASS), the most
one pass scores and about the most a judge's top log-probabilities cover. A
larger set is recast when the row is built, as the correct option plus
distractors drawn with a stored seed, so every row is reproducible and
can be labelled. At inference the server chunks a large set into tens and
rescores the winners, which blind options make consistent: an option's
score cannot depend on which others shared its pass.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schema.questions import (
    HEAD_OPTIONS_PER_PASS,
    PROBABILITY_SUM_TOLERANCE,
    ChoiceQuestion,
    JsonContent,
    NoulQuestion,
    Question,
    ScoreQuestion,
)

# Where a row may be used. "heldout_task" rows belong to tasks withheld entirely
# for the generalisation claim (PLAN.md: about 20 percent of tasks).
Split = Literal["train", "validation", "heldout_task"]
# How the target was produced.
LabelKind = Literal["human", "judges", "rule"]
# How the row itself came to exist.
Origin = Literal["converted", "generated", "authored"]
# Judges are named by letter in anything public (rule 4); their identity
# and provider live in the spend ledger.
JudgeName = Literal["A", "B", "C", "D", "E"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def outcomes(question: Question) -> list[str]:
    """The target keys a question has, in canonical order."""
    if isinstance(question, ChoiceQuestion):
        return question.options
    if isinstance(question, ScoreQuestion):
        return [str(level) for level in range(len(question.criteria))]
    if isinstance(question, NoulQuestion):
        return ["true", "false"]
    raise TypeError(f"unknown question type {type(question).__name__}")


def _check_target(target: dict[str, float], keys: list[str], where: str) -> None:
    if set(target) != set(keys):
        missing, extra = sorted(set(keys) - set(target)), sorted(set(target) - set(keys))
        raise ValueError(
            f"{where}: keys do not match the question: missing {missing}, extra {extra}"
        )
    if any(not (0.0 <= p <= 1.0) or math.isnan(p) for p in target.values()):
        raise ValueError(f"{where}: a probability lies outside [0, 1]")
    total = math.fsum(target.values())
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(f"{where}: probabilities sum to {total!r}, expected 1")


class JudgeVote(_Strict):
    """One judge's distribution for one row, and what it was shown."""

    judge: JudgeName
    # Option keys in the order the judge saw them. Needed to measure position
    # bias and to rotate orders across judges.
    shown_order: list[str]
    distribution: dict[str, float]
    # The probability mass the judge put outside the offered letters before
    # renormalisation. A large value means the judge did not answer the
    # question as asked, which the confident-learning pass needs to know.
    off_letter_mass: float = Field(ge=0.0, le=1.0)


class TrainingRow(_Strict):
    row_id: str = ""
    track: str
    task: str
    split: Split
    origin: Origin
    label_kind: LabelKind
    # data/MANIFEST.yaml source id; the licence is read from the manifest, never
    # restated here where it could drift.
    source: str
    state: JsonContent
    question: Question
    target: dict[str, float]
    judges: list[JudgeVote] = Field(default_factory=list)
    # Version of the prompt that generated the question, or of the converter.
    recipe: str

    @model_validator(mode="after")
    def _check(self) -> TrainingRow:
        keys = outcomes(self.question)
        if len(keys) > HEAD_OPTIONS_PER_PASS:
            raise ValueError(
                f"a training row carries at most {HEAD_OPTIONS_PER_PASS} options, got "
                f"{len(keys)}; recast larger sets when the row is built"
            )
        _check_target(self.target, keys, "target")
        if self.label_kind == "judges":
            if len(self.judges) < 2:
                raise ValueError("a judge-labelled row needs at least two judges")
            if len({vote.judge for vote in self.judges}) != len(self.judges):
                raise ValueError("a judge appears twice on one row")
            for vote in self.judges:
                _check_target(vote.distribution, keys, f"judge {vote.judge}")
                if sorted(vote.shown_order) != sorted(keys):
                    raise ValueError(f"judge {vote.judge}: shown_order is not the options")
            mean = mean_vote(self.judges, keys)
            if any(abs(mean[k] - self.target[k]) > 1e-6 for k in keys):
                raise ValueError("target is not the arithmetic mean of the judges' votes")
        elif self.judges:
            raise ValueError(f"a {self.label_kind} row carries no judge votes")
        if self.label_kind == "human" and max(self.target.values()) < 1.0 - 1e-9:
            raise ValueError("a human-labelled row must have a one-hot target")
        expected = row_id_for(self)
        if self.row_id and self.row_id != expected:
            raise ValueError(f"row_id {self.row_id!r} is not the content hash {expected!r}")
        self.row_id = expected
        return self


def mean_vote(votes: list[JudgeVote], keys: list[str]) -> dict[str, float]:
    """The plan's vote distribution: the arithmetic mean of the judges' vectors.

    Not the geometric mean, which averaging log-probabilities would give and
    which is sharper than any judge that disagrees.
    """
    return {key: math.fsum(vote.distribution[key] for vote in votes) / len(votes) for key in keys}


def text_of(state: JsonContent) -> str:
    """The state as one canonical string, for hashing and decontamination."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def row_id_for(row: TrainingRow) -> str:
    """A content hash of what the model reads, so a duplicate cannot hide under two ids.

    The target and the labels are left out on purpose: the same state and
    question labelled twice is one row seen twice, which is the duplication a
    split must not straddle.
    """
    payload = json.dumps(
        {
            "state": text_of(row.state),
            "question": row.question.model_dump(mode="json"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()
