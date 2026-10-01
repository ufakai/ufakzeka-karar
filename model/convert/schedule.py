"""The learning-rate schedule for the conversion: warmup, stable, decay.

WSD is the right family here because the conversion is a continued run whose
length we may extend: the plan takes checkpoints at 5B, 14B and 27B tokens and
continues "only while the instrument improves". A cosine schedule has to know
its end in advance, so extending it means either a wrong shape or a restart.
WSD holds a constant rate and only anneals at the end, so a stopped run is a
finished run.

The decay is 1 minus the square root of progress through the tail, not linear.
The backbone's own pretraining used linear over the last 15 percent, but the
ablation since found "with a 1-sqrt decay profile, WSD with 20% decay matched
cosine performance, while shorter decay fractions underperformed" (read
2026-09-21), so the default here is that shape over 20 percent. The linear
form is kept so the backbone's schedule can be reproduced exactly when a
comparison needs it.

The same tail carries the masking step-down and the inference-mask
training the plan puts in the decay phase, so one fraction governs all three
and they cannot drift apart.
"""

from __future__ import annotations

import math

SHAPES = ("one_minus_sqrt", "linear")
DEFAULT_DECAY_FRAC = 0.20


def wsd_scale(
    step: int,
    total_steps: int,
    *,
    warmup_steps: int,
    decay_frac: float = DEFAULT_DECAY_FRAC,
    shape: str = "one_minus_sqrt",
    final_frac: float = 0.0,
) -> float:
    """The multiplier on the peak learning rate at `step`, between 0 and 1.

    `final_frac` is where the decay lands, as a fraction of the peak. Zero
    means the run ends at no learning rate at all, which is what a final
    checkpoint wants; a small non-zero value is useful when a run will be
    continued from its own end.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not 0.0 <= decay_frac <= 1.0:
        raise ValueError(f"decay_frac must be between 0 and 1, got {decay_frac}")
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}, got {shape!r}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps cannot be negative, got {warmup_steps}")

    if step < warmup_steps:
        return (step + 1) / warmup_steps

    decay_start = int(total_steps * (1.0 - decay_frac))
    if step < decay_start:
        return 1.0

    span = max(1, total_steps - decay_start)
    through = min(1.0, (step - decay_start) / span)
    remaining = 1.0 - math.sqrt(through) if shape == "one_minus_sqrt" else 1.0 - through
    return final_frac + (1.0 - final_frac) * max(0.0, remaining)


def steps_for_tokens(tokens: int, *, batch_tokens: int) -> int:
    """How many optimizer steps a token budget buys at a fixed batch size.

    The conversion holds the batch constant rather than ramping it, unlike the
    backbone's pretraining: a ramp exists to stabilise a model starting from
    noise, and this one starts from trained weights.
    """
    if tokens <= 0:
        raise ValueError(f"tokens must be positive, got {tokens}")
    if batch_tokens <= 0:
        raise ValueError(f"batch_tokens must be positive, got {batch_tokens}")
    return max(1, tokens // batch_tokens)


def decay_branch_steps(rung_step: int, decay_frac: float) -> int:
    """Total steps for a run whose decay begins exactly at `rung_step`.

    A trunk trained at the stable rate can be stopped at any rung and annealed
    from there, which is what the warmup-stable-decay shape is for: the cost of
    finding out where to stop is one short branch per rung rather than one full
    run per budget. The branch is an ordinary run with a shorter total, chosen
    so that the learning rate, the masking rate and the mixture all step at the
    rung and not a few steps either side of it.
    """
    if rung_step <= 0:
        raise ValueError(f"rung_step must be positive, got {rung_step}")
    if not 0.0 < decay_frac < 1.0:
        raise ValueError(f"decay_frac must be between 0 and 1, got {decay_frac}")
    guess = round(rung_step / (1.0 - decay_frac))
    for total in (guess, guess + 1, guess - 1, guess + 2, guess - 2):
        if total > rung_step and int(total * (1.0 - decay_frac)) == rung_step:
            return total
    raise ValueError(f"no total puts the decay boundary at step {rung_step}")


def steps_to_reach(tokens: int, batch_tokens: int) -> int:
    """The first step count at which `tokens` have been seen. Rounded up.

    A launch that stops "at 1B tokens" has to stop at or after the 1B rung, and
    a rung is kept at the step where the count first reaches it. 1B is not a
    multiple of the batch, so rounding down stops one step short: the run ends
    with 999.8M tokens, no rung is written, and the decay branch that expects
    one finds nothing after the run has been paid for.
    """
    if tokens <= 0 or batch_tokens <= 0:
        raise ValueError("tokens and batch_tokens must be positive")
    return -(-tokens // batch_tokens)
