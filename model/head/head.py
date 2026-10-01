"""The decision head: one score per option, one abstain logit per question.

The backbone reads the packed sequence (model/head/pack.py) with the head's own
mask and positions. Each option is read from its own tokens by pooling, scored
by one shared linear map, and the scores meet only in the softmax over the
question's options, which is where the comparison between options happens.
The prefix is pooled on its own for the abstain output, which is trained
later from held-out errors (PLAN.md); until then its weight in the loss
is zero.

Pooling is the mean over an option's tokens, with the last token as the
alternative for a causal backbone, whose last position is the only one that has
read the whole option. A fresh marker token per option failed to learn in the
open replication that tried it, while span pooling did, so markers
are not used.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from model.convert.native import NativeModel
from model.head.pack import PAD, PREFIX, Batch, attention_mask

Pooling = Literal["mean", "last"]


def pool(hidden: torch.Tensor, segments: torch.Tensor, which: torch.Tensor, how: Pooling):
    """For each (example, slot), pool the tokens whose segment equals `which`.

    hidden (B, T, D), segments (B, T), which (B, N) -> (B, N, D). Slots whose
    segment appears nowhere pool to zero and are masked by the caller.
    """
    member = segments[:, None, :] == which[:, :, None]  # (B, N, T)
    if how == "mean":
        weights = member.to(hidden.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1.0)
        return weights @ hidden
    if how == "last":
        length = segments.shape[1]
        index = torch.arange(length, device=segments.device)
        last = torch.where(member, index, torch.full_like(index, -1)).amax(-1)  # (B, N)
        gathered = hidden.gather(1, last.clamp(min=0)[..., None].expand(-1, -1, hidden.shape[-1]))
        return gathered * (last >= 0)[..., None].to(hidden.dtype)
    raise ValueError(f"unknown pooling {how!r}")


class DecisionHead(nn.Module):
    # The token id padding is filled with; set from the tokenizer by the loop.
    pad_id: int = 0

    def __init__(self, backbone: NativeModel, *, causal: bool, pooling: Pooling = "mean",
                 layout: str = "blind"):  # fmt: skip
        super().__init__()
        self.backbone = backbone
        self.causal = causal
        self.pooling = pooling
        # How the batch was laid out (pack.arrange); the mask must agree with it.
        self.layout = layout
        width = backbone.cfg.d_model
        self.score = nn.Linear(width, 1)
        self.abstain = nn.Linear(width, 1)
        nn.init.normal_(self.score.weight, std=0.02)
        nn.init.zeros_(self.score.bias)
        nn.init.normal_(self.abstain.weight, std=0.02)
        nn.init.zeros_(self.abstain.bias)

    def forward(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Option logits (B, N), -inf where a slot is empty, and abstain logits (B,)."""
        mask = attention_mask(batch.segments, causal=self.causal, layout=self.layout)
        hidden = self.backbone.hidden(batch.ids, mask=mask, positions=batch.positions)
        options = pool(hidden, batch.segments, batch.option_segments, self.pooling)
        logits = self.score(options).squeeze(-1).float()
        logits = logits.masked_fill(~batch.option_valid, float("-inf"))
        # Back to the caller's order by indexing only (pack.Batch.caller_slot).
        logits = logits.gather(1, batch.caller_slot)
        prefix_slot = torch.full_like(batch.option_segments[:, :1], PREFIX)
        prefix = pool(hidden, batch.segments, prefix_slot, self.pooling).squeeze(1)
        abstain = self.abstain(prefix).squeeze(-1).float()
        return logits, abstain

    def to_device(self, batch: Batch, device) -> Batch:
        return Batch(
            ids=batch.ids.to(device),
            positions=batch.positions.to(device),
            segments=batch.segments.to(device),
            option_segments=batch.option_segments.to(device),
            option_valid=batch.option_valid.to(device),
            caller_slot=batch.caller_slot.to(device),
            keys=batch.keys,
        )


def soft_targets(batch: Batch, targets: list[dict[str, float]]) -> torch.Tensor:
    """Targets aligned to the batch's option slots, (B, N), zero in empty slots."""
    out = torch.zeros(batch.option_valid.shape, dtype=torch.float32)
    for row, (keys, target) in enumerate(zip(batch.keys, targets, strict=True)):
        for slot, key in enumerate(keys):
            out[row, slot] = target[key]
    return out


def objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    kind: str = "ce",
    weights: torch.Tensor | None = None,
    ordinal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy or Brier against soft targets, a weighted mean over questions.

    Both are strictly proper, so the minimiser is the target distribution
    itself (PLAN.md: direct minimisation of Brier or cross-entropy on vote
    distributions, no label smoothing, no focal loss). Score questions, marked
    by `ordinal`, add the ranked probability score: the squared distance
    between the predicted and the target cumulative distributions over the
    levels, in their order, divided by the number of gaps. It is also strictly
    proper, so the sum keeps the target as the minimiser, and it charges mass
    two levels off more than mass one level off, which a plain proper score
    over unordered options cannot (PLAN.md's ordinal targets).
    `weights` give each question's share of the mean (back to the
    natural prior); None weighs them equally.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    if kind == "ce":
        per_row = -(targets * log_probs.masked_fill(targets == 0, 0.0)).sum(-1)
    elif kind == "brier":
        per_row = ((log_probs.exp() - targets) ** 2).sum(-1)
    else:
        raise ValueError(f"unknown objective {kind!r}")
    if ordinal is not None and bool(ordinal.any()):
        # Levels sit in their natural order in the caller's slots; empty slots
        # carry zero probability and zero target, so both cumulatives are flat
        # (at one) past the last level and add nothing.
        gap = (torch.isfinite(logits).sum(-1) - 1).clamp(min=1).to(per_row.dtype)
        cdf = torch.cumsum(log_probs.exp(), -1) - torch.cumsum(targets, -1)
        rps = (cdf**2).sum(-1) / gap
        per_row = per_row + torch.where(ordinal, rps, torch.zeros_like(rps))
    if weights is None:
        return per_row.mean()
    return (per_row * weights).sum() / weights.sum()


def policy_gradient(
    logits: torch.Tensor,
    targets: torch.Tensor,
    samples: int = 8,
    weights: torch.Tensor | None = None,
    ordinal: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """REINFORCE on the head's own distribution, step 6's claim 4 arm.

    The policy is the softmax over a question's options; `samples` answers are
    drawn from it per question. A sampled answer's reward is the target's mass on
    it (its vote share; correctness on one-hot rows). On score questions, marked
    by `ordinal`, the reward is distance-aware: the target's mass weighted by
    1 - |a - k| / gaps. The baseline for each sample is the mean reward of the
    other samples (RLOO, arXiv 2402.14740), with no standard-deviation scaling
    (arXiv 2503.20783). The surrogate's gradient is an unbiased estimate of the
    gradient of the expected reward, sum_a p(a) r(a), which is linear in p and so
    maximised by putting all mass on the best answer, whatever the target is:
    unlike cross-entropy or Brier, not a proper score. That cost is what the arm
    measures. `weights` weigh questions as in `objective`.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp().detach()
    actions = torch.multinomial(probs, samples, replacement=True, generator=generator)
    reward = targets.gather(1, actions)
    if ordinal is not None and bool(ordinal.any()):
        gaps = (torch.isfinite(logits).sum(-1) - 1).clamp(min=1).to(targets.dtype)
        levels = torch.arange(logits.shape[-1], device=logits.device)
        distance = (actions[:, :, None] - levels[None, None, :]).abs().to(targets.dtype)
        credit = (1 - distance / gaps[:, None, None]).clamp(min=0)
        ranked = (targets[:, None, :] * credit).sum(-1)
        reward = torch.where(ordinal[:, None], ranked, reward)
    baseline = (reward.sum(1, keepdim=True) - reward) / max(1, samples - 1)
    per_row = -((reward - baseline) * log_probs.gather(1, actions)).mean(1)
    if weights is None:
        return per_row.mean()
    return (per_row * weights).sum() / weights.sum()


__all__ = ["PAD", "DecisionHead", "objective", "policy_gradient", "pool", "soft_targets"]
