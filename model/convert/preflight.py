"""Checking a run's assumptions against the real thing before it costs anything.

Three faults in this package were found by reading it, none by the tests that
passed while they were there: a resume that restarted the data, a concatenate
that needed 76 GB, and a reader that could not read the format the corpus is
written in. All three were invisible to a unit test because a unit test runs
against a fake, and every one of them lives in the gap between the fake and the
real model, tokenizer, file format, volume path or GPU.

So the gap gets its own checks. Each returns a `Check` rather than raising, so
one call reports everything that is wrong instead of the first thing, and the
expensive function runs the same gate at startup so it cannot be skipped by
running the wrong command.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def report(checks: Iterable[Check]) -> str:
    lines = [f"  {'ok ' if c.ok else 'FAIL'} {c.name}: {c.detail}" for c in checks]
    return "\n".join(lines)


def raise_on_failure(checks: list[Check]) -> None:
    """Refuse to continue while anything is wrong, naming everything that is."""
    bad = [c for c in checks if not c.ok]
    if bad:
        raise RuntimeError("preflight failed:\n" + report(bad))


def check_shards(tier: str, paths: list[Path], context: int, itemsize: int = 2) -> Check:
    """The shards exist, are whole numbers of tokens, and hold enough windows."""
    if not paths:
        return Check(f"tier {tier} shards", False, "no .bin shards found")
    total = 0
    for path in paths:
        if not path.is_file():
            return Check(f"tier {tier} shards", False, f"{path} is missing")
        size = path.stat().st_size
        if size % itemsize:
            return Check(f"tier {tier} shards", False, f"{path.name} is {size} bytes, not whole")
        total += size // itemsize
    windows = total // context
    if windows == 0:
        return Check(f"tier {tier} shards", False, f"{total} tokens is under one window")
    return Check(
        f"tier {tier} shards", True, f"{len(paths)} files, {total:,} tokens, {windows:,} windows"
    )


def check_tokens_cover_budget(available: int, budget: int, tier: str) -> Check:
    """Reading a tier more than once is allowed, but never by accident."""
    passes = budget / available if available else float("inf")
    ok = passes <= 1.0
    return Check(
        f"tier {tier} coverage",
        ok,
        f"{passes:.2f} passes over the tier for this budget"
        + ("" if ok else "; text would be repeated"),
    )


def check_mask_token(tokenizer, mask_token: str, vocab_size: int) -> Check:
    """The mask token resolves, is inside the matrix, and is not the unknown id."""
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    unknown = getattr(tokenizer, "unk_token_id", None)
    if mask_id is None or mask_id < 0:
        return Check("mask token", False, f"{mask_token} is not in the vocabulary")
    if unknown is not None and mask_id == unknown:
        return Check("mask token", False, f"{mask_token} resolves to the unknown id")
    if mask_id >= vocab_size:
        return Check("mask token", False, f"id {mask_id} is outside a vocabulary of {vocab_size}")
    return Check("mask token", True, f"{mask_token} is id {mask_id}")


def check_separator(tokenizer) -> Check:
    if tokenizer.eos_token_id is None:
        return Check("separator", False, "the tokenizer has no end-of-text id to pack with")
    return Check("separator", True, f"end of text is id {tokenizer.eos_token_id}")


def check_model_fits_context(config, context: int) -> Check:
    limit = getattr(config, "max_position_embeddings", None)
    if limit is None:
        return Check("context", False, "the model declares no position limit")
    if context > limit:
        return Check("context", False, f"context {context} is beyond the model's {limit}")
    return Check("context", True, f"context {context} inside the model's {limit}")


def check_vocab_matches(config, tokenizer) -> Check:
    """The embedding matrix must be at least as wide as the tokenizer's ids.

    A tokenizer wider than the matrix produces an index error thousands of
    steps in, when a rare id first appears.
    """
    matrix = getattr(config, "vocab_size", None)
    spoken = len(tokenizer)
    if matrix is None:
        return Check("vocabulary", False, "the model declares no vocabulary size")
    if spoken > matrix:
        return Check("vocabulary", False, f"tokenizer has {spoken} ids, matrix holds {matrix}")
    return Check("vocabulary", True, f"{spoken} ids in a matrix of {matrix}")


def check_attention_is_uniform(config) -> Check:
    """Every layer full attention, so the bidirectional swap has one path.

    A sliding-window layer needs its own bidirectional mask, and getting that
    wrong is silent: the model trains, and reads less than it should.
    """
    types = getattr(config, "layer_types", None)
    if types and set(types) != {"full_attention"}:
        return Check("attention", False, f"mixed layer types: {sorted(set(types))}")
    if getattr(config, "use_sliding_window", False):
        return Check("attention", False, "sliding window is enabled")
    return Check("attention", True, "every layer is full attention")


def check_batch_fits(
    free_bytes: int,
    parameters: int,
    *,
    micro_batch: int,
    context: int,
    layers: int,
    hidden: int,
    vocab_size: int,
    masked_share: float = 0.3,
) -> Check:
    """Whether one micro-batch fits, which is what actually has to.

    Gradient accumulation means only `micro_batch` rows are resident at once,
    so the full batch is the wrong number to check; an earlier version of this
    used it and was cheerful about a run that needed 176 GB.

    Optimizer states are four copies of the parameters in float32: the weights,
    the gradients and Adam's two moments. Activations are the part that scales,
    and a transformer layer saves roughly seven hidden-sized tensors per token
    for the backward pass, in bfloat16. Attention itself is taken as linear in
    the sequence, which holds for the scaled-dot-product path and not for the
    quadratic eager one, so the attention implementation is checked separately.
    """
    states = parameters * 4 * 4
    per_token = layers * hidden * 7 * 2
    activations = micro_batch * context * per_token
    # The vocabulary projection, which an earlier version of this forgot and
    # was three times under because of it. At 40960 classes and 1024 context a
    # single float32 copy of the logits is 1.34 GB for eight rows, and between
    # the logits, the float cast and the backward there are about three. The
    # loss now projects only the masked positions, so the share is the masking
    # rate rather than all of them.
    logits = micro_batch * context * vocab_size * 4 * 3 * masked_share
    needed = states + activations + logits
    # Four fifths, not nine tenths. The allocator fragments and transient peaks
    # run well above the steady state, so a run approved at ninety percent of
    # the card will sometimes fail anyway, which is the failure this exists to
    # prevent. Refusing a batch that would have just fitted costs one smaller
    # batch; approving one that does not costs the run.
    ok = needed < free_bytes * 0.8
    return Check(
        "memory",
        ok,
        f"about {needed / 1e9:.1f} GB for {micro_batch} rows x {context} "
        f"against {free_bytes / 1e9:.1f} GB free" + ("" if ok else "; reduce micro_batch"),
    )


def check_attention_implementation(model) -> Check:
    """Attention must be sub-quadratic, or the memory estimate is a fiction.

    The eager path materialises a sequence-by-sequence matrix per head, which
    at 1024 context is a different order of memory from the scaled-dot-product
    path and turns a comfortable batch into an out-of-memory error.
    """
    if hasattr(model, "body"):
        # The backbone's own network: a block mask through FlexAttention on a
        # GPU, the fused kernel when there is no mask. Neither is quadratic.
        return Check("attention kernel", True, "native network, block mask or fused kernel")
    found = getattr(getattr(model, "config", None), "_attn_implementation", None)
    if found in {"sdpa", "flash_attention_2", "flash_attention_3", "kernels-community/flash-attn"}:
        return Check("attention kernel", True, f"{found}")
    return Check("attention kernel", False, f"{found!r} is quadratic in the sequence")


def describe_plan(total_tokens: int, batch_tokens: int, context: int, mixture: dict) -> Check:
    steps = max(1, total_tokens // batch_tokens)
    rows = max(1, batch_tokens // context)
    share = ", ".join(f"{t} {w:.0%}" for t, w in sorted(mixture.items()) if w > 0)
    return Check(
        "plan",
        True,
        f"{total_tokens:,} tokens, {steps:,} steps of {rows} rows x {context} ({share})",
    )


def sample_windows_are_plausible(windows, tries: int = 8, seed: int = 0) -> Check:
    """Read a few windows and look at them, rather than trusting the index.

    A wrong dtype or offset still returns an array of the right shape. What it
    does not return is a spread of ids that looks like text: it returns zeros,
    or one value, or something far outside the vocabulary.
    """
    if len(windows) == 0:
        return Check("sample", False, "the tier holds no windows")
    rng = np.random.default_rng(seed)
    seen = []
    for index in rng.integers(0, len(windows), size=min(tries, len(windows))):
        seen.append(windows.window(int(index)))
    stacked = np.concatenate(seen)
    distinct = len(np.unique(stacked))
    if distinct < max(4, len(stacked) // 100):
        return Check("sample", False, f"only {distinct} distinct ids in {len(stacked)} tokens")
    return Check(
        "sample",
        True,
        f"{len(seen)} windows, {distinct:,} distinct ids, range {stacked.min()}-{stacked.max()}",
    )


def check_separator_in_corpus(windows, separator_id: int, tier: str, tries: int = 32) -> Check:
    """The id the document mask is built from really does appear in the shards.

    The tokenizer answering with an end-of-text id says nothing about which id
    the corpus was packed with, and the backbone's config lists two. If they
    differ, every window is one document, the mask hides nothing, and the run
    still pays the mask's 11 percent for it without any error.
    """
    if len(windows) == 0:
        return Check(f"tier {tier} separators", False, "the tier holds no windows")
    rng = np.random.default_rng(1)
    picks = rng.integers(0, len(windows), size=min(tries, len(windows)))
    found = sum(int((windows.window(int(i)) == separator_id).sum()) for i in picks)
    per_window = found / len(picks)
    if found == 0:
        return Check(
            f"tier {tier} separators",
            False,
            f"id {separator_id} never appears in {len(picks)} sampled windows",
        )
    return Check(
        f"tier {tier} separators", True, f"{per_window:.1f} per window of id {separator_id}"
    )
