"""Answering a question with more options than one pass scores.

The head scores at most ten options a pass. Options are blind to each other
and share their positions, so an option's logit is the same in any pass
that holds it: a larger set is split into passes in the canonical order (not
the caller's), every option's logit is read, and one softmax runs over all of
them. No option's probability is dropped, and the answer cannot depend on how
the set was listed or split. An earlier version kept only each pass's winner
and rescored the winners, which discarded the losers' probability and returned
fewer options than were asked.
"""

from __future__ import annotations

import torch
from pydantic import TypeAdapter

from model.head.pack import collate, pack, parts
from schema.questions import HEAD_OPTIONS_PER_PASS, ChoiceQuestion, Question

QUESTION = TypeAdapter(Question)


def _sub_question(question: ChoiceQuestion, keys: list[str]) -> ChoiceQuestion:
    return QUESTION.validate_python(
        {
            "type": "choice",
            "instructions": question.instructions,
            "criteria": {key: question.criteria[key] for key in keys},
        }
    )


def chunked_logits(
    model, state, question: ChoiceQuestion, encode, device="cpu", per_pass=HEAD_OPTIONS_PER_PASS
) -> dict[str, float]:
    """Every option's logit, in the caller's order, from passes of at most ten.

    The logits are what a temperature divides before the softmax (bench/adapters/karar.py).
    """
    keys = sorted(question.options, key=lambda key: (encode("\n" + key), key))
    passes = [keys[i : i + per_pass] for i in range(0, len(keys), per_pass)]
    if len(passes) > 1 and len(passes[-1]) == 1:
        # A question needs two options; the last pass borrows one it scores again.
        passes[-1] = [passes[-2][-1], *passes[-1]]
    if len(passes) > 1:
        # pack refuses two options that encode alike within one pass; encode every option
        # together first, so a pair split across passes is refused as well.
        parts(state, question, encode)
    model.eval()
    logits: dict[str, float] = {}
    with torch.no_grad():
        for keys_in_pass in passes:
            batch = model.to_device(
                # The model's own layout. Under "sequential" an option's score
                # depends on the options before it, so scores from different
                # passes are not strictly comparable there.
                collate([pack(state, _sub_question(question, keys_in_pass), encode,
                              layout=getattr(model, "layout", "blind"))],
                        model.pad_id), device,
            )  # fmt: skip
            out, _ = model(batch)
            for key, value in zip(batch.keys[0], out[0].float().tolist(), strict=True):
                logits.setdefault(key, value)
    return {key: logits[key] for key in question.options}


def chunked_choice(
    model, state, question: ChoiceQuestion, encode, device="cpu", per_pass=HEAD_OPTIONS_PER_PASS
) -> dict[str, float]:
    """Probabilities over every option, in the caller's order, from passes of at most ten."""
    logits = chunked_logits(model, state, question, encode, device, per_pass)
    probs = torch.softmax(torch.tensor(list(logits.values()), dtype=torch.float64), dim=0)
    return dict(zip(logits, probs.tolist(), strict=True))
