"""How one question becomes one sequence the backbone reads.

    [state and question: the prefix][option 1][option 2] ... [option n]

Three properties make the answer independent of option order, and
Set-LLM (arXiv 2505.15433) is the published form of the first two:

1. Shared positions. Every option starts at the position right after the
   prefix, so no option sits "later" than another.
2. Blind options. The prefix attends only to itself, so it is computed the same
   whatever options follow; each option attends to the prefix and to itself,
   never to another option.
3. A canonical order. Options are packed in the order of their token ids, not
   the caller's, and scores are handed back in the caller's order. 1 and 2
   already make the answer order-free in exact arithmetic; floating-point sums
   can still differ in the last bit when the same numbers are added in another
   grouping, and packing canonically removes even that, so the permutation test
   can demand bit-for-bit equality.

Segments name what each token belongs to: 0 for the prefix, i for option i
(from 1), -1 for padding. The mask and the pooling are both computed from them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from schema.questions import ChoiceQuestion, NoulQuestion, ScoreQuestion
from schema.rows import outcomes, text_of

PREFIX = 0
PAD = -1

Encode = Callable[[str], list[int]]


def option_texts(question) -> dict[str, str]:
    """What each outcome looks like to the model, keyed like the target."""
    if isinstance(question, ChoiceQuestion):
        return {
            name: f"Seçenek: {name}" + (f". {text}" if text else "")
            for name, text in question.criteria.items()
        }
    if isinstance(question, NoulQuestion):
        criteria = question.criteria
        yes = criteria.true if criteria and criteria.true else ""
        no = criteria.false if criteria and criteria.false else ""
        return {
            "true": "Cevap: evet" + (f". {yes}" if yes else ""),
            "false": "Cevap: hayır" + (f". {no}" if no else ""),
        }
    if isinstance(question, ScoreQuestion):
        # Levels are blind to each other too, so each carries its own number:
        # order is information the level's own text has to hold.
        return {str(k): f"Düzey {k}: {text}" for k, text in enumerate(question.criteria)}
    raise TypeError(f"unknown question type {type(question).__name__}")


def prefix_text(state, question) -> str:
    return f"{text_of(state)}\n\nSoru: {text_of(question.instructions)}"


@dataclass(frozen=True)
class Packed:
    ids: list[int]
    positions: list[int]
    segments: list[int]
    # Outcome keys in the caller's order, and for each the segment it was
    # packed into.
    keys: list[str]
    segment_of: dict[str, int]


@dataclass(frozen=True)
class Parts:
    """A question's encoded prefix and options, before they are laid out."""

    prefix: list[int]
    keys: list[str]
    encoded: dict[str, list[int]]


def parts(state, question, encode: Encode, *, max_prefix: int = 448, max_option: int = 48) -> Parts:
    """Encode once; `arrange` lays the parts out. Truncation keeps the start of each part.

    A prefix too long for `max_prefix` loses the end of its state first and keeps
    the question, up to half of the prefix (a longer question loses its own end):
    the question is what the options answer. A prefix that fits is encoded
    whole, exactly as before.
    """
    prefix = encode(prefix_text(state, question))
    if len(prefix) > max_prefix:
        asked = encode(f"\n\nSoru: {text_of(question.instructions)}")[: max_prefix // 2]
        prefix = encode(text_of(state))[: max_prefix - len(asked)] + asked
    texts = option_texts(question)
    keys = outcomes(question)
    encoded = {key: encode("\n" + texts[key])[:max_option] for key in keys}
    for key, tokens in encoded.items():
        if not tokens:
            raise ValueError(f"option {key!r} encodes to no tokens")
    if len({tuple(encoded[k]) for k in keys}) != len(keys):
        raise ValueError("two options encode to the same tokens and could not be told apart")
    return Parts(prefix, keys, encoded)


def arrange(p: Parts, layout: str = "blind", order: list[str] | None = None) -> Packed:
    """Lay the parts out as one sequence.

    "blind" (the shipped head): options in canonical token order, every option
    starting at the position after the prefix. "sequential" (the conventional
    head of step 6's claim 5): options one after another in `order` (the
    caller's order unless given), positions running on, each option able to see
    the ones before it through the mask.
    """
    if layout == "blind":
        laid = sorted(p.keys, key=lambda key: (p.encoded[key], key))
    elif layout == "sequential":
        laid = list(order) if order is not None else list(p.keys)
        if sorted(laid) != sorted(p.keys):
            raise ValueError("an order must name every option once")
    else:
        raise ValueError(f"unknown layout {layout!r}")
    n = len(p.prefix)
    ids, positions, segments = list(p.prefix), list(range(n)), [PREFIX] * n
    segment_of, start = {}, n
    for index, key in enumerate(laid, start=1):
        tokens = p.encoded[key]
        ids += tokens
        positions += list(range(start, start + len(tokens)))
        if layout == "sequential":
            start += len(tokens)
        segments += [index] * len(tokens)
        segment_of[key] = index
    return Packed(ids, positions, segments, p.keys, segment_of)


def pack(
    state,
    question,
    encode: Encode,
    *,
    max_prefix: int = 448,
    max_option: int = 48,
    layout: str = "blind",
) -> Packed:
    """One question as one sequence, in the caller's order where the layout keeps order."""
    return arrange(parts(state, question, encode, max_prefix=max_prefix, max_option=max_option),
                   layout)  # fmt: skip


def attention_mask(segments: torch.Tensor, causal: bool, layout: str = "blind") -> torch.Tensor:
    """Who may attend to whom, (batch, 1, length, length), True where allowed.

    A query in the prefix sees the prefix. A query in option i sees the prefix
    and option i. Nothing sees another option. With `causal`, attention inside
    the prefix and inside each option runs forward only, which is the only
    reading a causal backbone was trained for; the prefix always precedes an
    option, so an option still sees all of it. Padding sees only itself, so no
    row of the softmax is empty.

    The "sequential" layout lets option i also see options 1 to i-1,
    which lie before it: with a causal backbone this is the plain causal mask
    over the real tokens, the sequential head the blind one is compared with.
    The prefix still sees only itself.
    """
    q = segments[:, :, None]
    k = segments[:, None, :]
    real_q = q != PAD
    if layout == "sequential":
        allowed = real_q & (k != PAD) & ((k == PREFIX) | ((k >= 1) & (k <= q)))
    else:
        allowed = real_q & ((k == PREFIX) | (k == q)) & (k != PAD)
    if causal:
        length = segments.shape[1]
        index = torch.arange(length, device=segments.device)
        forward_only = index[None, :, None] >= index[None, None, :]
        allowed = allowed & forward_only
    eye = torch.eye(segments.shape[1], dtype=torch.bool, device=segments.device)[None]
    allowed = allowed | (eye & ~real_q)
    return allowed[:, None, :, :]


@dataclass
class Batch:
    ids: torch.Tensor  # (B, T)
    positions: torch.Tensor  # (B, T)
    segments: torch.Tensor  # (B, T)
    # Slot j pools segment j + 1: slots follow the canonical packing order, so
    # pooling and scoring run on the same numbers in the same arrangement
    # whatever order the caller gave. Padded slots hold segment 0 and are
    # marked invalid.
    option_segments: torch.Tensor  # (B, N)
    option_valid: torch.Tensor  # (B, N) bool
    # For the caller's k-th outcome, the canonical slot that scores it. Scores
    # go back through this index, which moves numbers without computing on
    # them, so the permutation test can ask for exact equality.
    caller_slot: torch.Tensor  # (B, N)
    keys: list[list[str]]


def collate(examples: list[Packed], pad_id: int) -> Batch:
    length = max(len(e.ids) for e in examples)
    width = max(len(e.keys) for e in examples)
    batch = len(examples)
    ids = torch.full((batch, length), pad_id, dtype=torch.long)
    positions = torch.zeros((batch, length), dtype=torch.long)
    segments = torch.full((batch, length), PAD, dtype=torch.long)
    option_segments = torch.zeros((batch, width), dtype=torch.long)
    option_valid = torch.zeros((batch, width), dtype=torch.bool)
    caller_slot = torch.zeros((batch, width), dtype=torch.long)
    for row, example in enumerate(examples):
        n = len(example.ids)
        ids[row, :n] = torch.tensor(example.ids)
        positions[row, :n] = torch.tensor(example.positions)
        segments[row, :n] = torch.tensor(example.segments)
        count = len(example.keys)
        option_segments[row, :count] = torch.arange(1, count + 1)
        option_valid[row, :count] = True
        for slot, key in enumerate(example.keys):
            caller_slot[row, slot] = example.segment_of[key] - 1
        # A padded caller slot points at a padded canonical slot, which is -inf.
        caller_slot[row, count:] = torch.arange(count, width)
    keys = [e.keys for e in examples]
    return Batch(ids, positions, segments, option_segments, option_valid, caller_slot, keys)
