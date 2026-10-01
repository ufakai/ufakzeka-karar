"""The masked-language objective the conversion trains against.

The backbone is a causal model and its tokenizer has no mask token, but its
vocabulary is padded to 40960 with twelve reserved ids that were never trained
on real text. Taking one of those as [MASK] means the embedding matrix keeps
its shape, the tied output head keeps its shape, and no pretrained row is
disturbed; the reserved row is untrained, which is exactly what a new mask
token should be.

The masking rate comes from the plan: 30 percent while stable, 10 percent in
the decay phase. 30 is current practice rather than the original 15, which
"has since been shown to be sub-optimal" for encoders of this generation, and
higher rates help larger models (arXiv 2202.08005, ModernBERT, read
2026-09-21).

The 80/10/10 replacement is kept. Replacing every selected token with [MASK]
"had a destructive effect on some model's downstream performance", and the
reason bites harder here than usual: the mask embedding starts untrained, so a
model that only ever sees [MASK] at predicted positions can learn to read the
position rather than the language.
"""

from __future__ import annotations

import numpy as np

# The reserved id taken as [MASK]. Untrained in the backbone, so it carries no
# meaning to unlearn, and inside the existing vocabulary, so nothing resizes.
MASK_TOKEN = "<|reserved_0|>"

# Cross-entropy ignores this label, so unselected positions cost nothing.
IGNORE = -100

# Of the selected positions: this share becomes [MASK], then half the rest
# becomes a random token and the remainder is left as it was.
MASK_SHARE = 0.8
RANDOM_SHARE = 0.1


def mask_tokens(
    ids: np.ndarray,
    *,
    rate: float,
    mask_id: int,
    vocab_size: int,
    rng: np.random.Generator,
    protected: frozenset[int] = frozenset(),
) -> tuple[np.ndarray, np.ndarray]:
    """Corrupt a batch of token ids and return what to predict where.

    Returns `(inputs, labels)`, both the shape of `ids`. `labels` holds the
    original id at every selected position and IGNORE everywhere else, so the
    loss is taken only where something was hidden.

    `protected` ids are never selected, which is how padding stays out of the
    objective. At least one position is always selected when any is eligible:
    a batch with nothing to predict produces a loss over zero elements, which
    is NaN, and one NaN ends a run that has been paying for hours.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be between 0 and 1, got {rate}")
    if mask_id >= vocab_size:
        raise ValueError(f"mask id {mask_id} is outside a vocabulary of {vocab_size}")

    eligible = np.ones(ids.shape, dtype=bool)
    for token in protected:
        eligible &= ids != token

    selected = (rng.random(ids.shape) < rate) & eligible
    if rate > 0 and eligible.any() and not selected.any():
        flat = np.flatnonzero(eligible.reshape(-1))
        selected.reshape(-1)[rng.choice(flat)] = True

    labels = np.where(selected, ids, IGNORE).astype(np.int64)

    inputs = ids.copy()
    draw = rng.random(ids.shape)
    to_mask = selected & (draw < MASK_SHARE)
    to_randomise = selected & (draw >= MASK_SHARE) & (draw < MASK_SHARE + RANDOM_SHARE)
    inputs[to_mask] = mask_id
    if to_randomise.any():
        inputs[to_randomise] = rng.integers(0, vocab_size, size=int(to_randomise.sum()))
    return inputs, labels


def masking_rate(
    step: int, total_steps: int, *, stable: float, decay: float, decay_frac: float
) -> float:
    """The rate at `step`: `stable` through the run, `decay` over the tail.

    The plan stages the masking rather than annealing it continuously, and the
    tail is where the inference mask is trained, so the change is a step and
    not a ramp. Anything past the end of the run stays at the decay rate.
    """
    if not 0.0 <= decay_frac <= 1.0:
        raise ValueError(f"decay_frac must be between 0 and 1, got {decay_frac}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    first_decay_step = int(total_steps * (1.0 - decay_frac))
    return decay if step >= first_decay_step else stable
