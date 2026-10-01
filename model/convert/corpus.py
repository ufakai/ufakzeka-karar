"""Which text the conversion is allowed to read, and in what proportion.

The backbone's own pretraining corpus is already tokenised with the same
tokenizer and sits on the ufakzeka-data volume, so the conversion needs no new
collection. Two things stand between it and being usable.

First, the licence. The tokenised shards are grouped by quality tier, not by
source, and tier A contains FineWiki, which is CC-BY-SA. The plan excludes
ShareAlike by name, so tier A cannot be read as it stands. It does not have to
be rebuilt from scratch either: `stage2/train/A` is parquet that still carries
a `source` column, so tier A is retokenised from there with FineWiki filtered
out, which keeps the original tier assignment exactly. That matters because
tier A is not simply a list of sources: a tier B document was promoted to A
when its quality score passed a threshold, so deriving tiers again from
`stage1` would mean reproducing that threshold and risking a different corpus.
Tiers B and C are wholly ODC-By and are usable as they are.

Second, the mixture. The conversion changes the objective, not the data, so it
keeps the proportions the backbone was pretrained on: the stable phase mixes
tiers the way the backbone's stable phase did, and the decay phase the way its
anneal phase did. Holding the distribution fixed is also what BidirLM's
multi-domain mixture asks for as a guard against forgetting, and it means
a change in the instrument can be attributed to the objective rather than to a
change of diet.
"""

from __future__ import annotations

from dataclasses import dataclass

from data.licenses import ALLOWED_LICENSE_IDS

# SPDX ids as verified for the backbone's release, from its ATTRIBUTION.md.
SOURCE_LICENSES: dict[str, str] = {
    "bilge_math": "Apache-2.0",
    "bilge_stories": "Apache-2.0",
    "bilge_web": "Apache-2.0",
    "cosmos_syn": "Apache-2.0",
    "finepdfs_edu": "ODC-By-1.0",
    "finemath": "ODC-By-1.0",
    "fw2hq": "ODC-By-1.0",
    "mogan": "ODC-By-1.0",
    # CC-BY-SA 4.0 and GFDL. Excluded: the plan rules out ShareAlike.
    "finewiki": "CC-BY-SA-4.0",
}

# Which tiers each source contributes to, from the corpus build's own stats.
# fw2hq spans two: its high-quality head is tier A and its bulk is tier B.
SOURCE_TIERS: dict[str, tuple[str, ...]] = {
    "bilge_math": ("A",),
    "bilge_stories": ("A",),
    "bilge_web": ("A",),
    "cosmos_syn": ("A",),
    "finepdfs_edu": ("A",),
    "finewiki": ("A",),
    "fw2hq": ("A", "B"),
    "mogan": ("B",),
    "finemath": ("C",),
}

TIERS: tuple[str, ...] = ("A", "B", "C")

# The backbone's own tier weights, read from its training config. The stable
# phase leans on bulk web, the anneal phase on the curated tier.
MIX_STABLE: dict[str, float] = {"A": 0.25, "B": 0.71, "C": 0.04}
MIX_DECAY: dict[str, float] = {"A": 0.60, "B": 0.37, "C": 0.03}


def allowed_sources() -> set[str]:
    """Sources whose licence the allow-list admits (data/licenses.py)."""
    return {name for name, spdx in SOURCE_LICENSES.items() if spdx in ALLOWED_LICENSE_IDS}


def excluded_sources() -> dict[str, str]:
    """The ones left out, with the licence that leaves them out."""
    return {name: spdx for name, spdx in SOURCE_LICENSES.items() if spdx not in ALLOWED_LICENSE_IDS}


def tiers_needing_rebuild() -> dict[str, list[str]]:
    """Tiers holding an excluded source, with the sources that spoil them.

    A tokenised tier shard is shuffled across its sources, so one disallowed
    source makes the whole tier unreadable and the tier has to be tokenised
    again from the per-source text without it.
    """
    out: dict[str, list[str]] = {}
    for name in sorted(excluded_sources()):
        for tier in SOURCE_TIERS[name]:
            out.setdefault(tier, []).append(name)
    return out


@dataclass(frozen=True)
class Allocation:
    """How many tokens to draw from each tier, and whether any must repeat."""

    per_tier: dict[str, int]
    total: int
    # Tokens asked of a tier divided by tokens it holds. Above 1.0 means the
    # tier is read more than once, which is worth knowing before it happens.
    passes: dict[str, float]

    @property
    def repeats(self) -> bool:
        return any(value > 1.0 for value in self.passes.values())


def allocate(budget: int, available: dict[str, int], mix: dict[str, float]) -> Allocation:
    """Split a token budget across tiers by the backbone's own weights.

    A tier that cannot cover its share is capped at what it holds and the
    shortfall is offered to the others in proportion to their remaining room,
    so the budget is met without silently reading one tier many times over.
    Whatever cannot be placed that way is left on the tiers in proportion to
    the mix, and `passes` then reports the repetition instead of hiding it.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    weights = {tier: mix.get(tier, 0.0) for tier in available}
    if not any(weights.values()):
        raise ValueError(f"no tier in {sorted(available)} carries weight in the mix {mix}")

    scale = sum(weights.values())
    wanted = {tier: budget * weight / scale for tier, weight in weights.items()}
    taken = {tier: min(int(wanted[tier]), available[tier]) for tier in available}

    shortfall = budget - sum(taken.values())
    room = {tier: available[tier] - taken[tier] for tier in available}
    if shortfall > 0 and sum(room.values()) > 0:
        spare = sum(room.values())
        for tier in sorted(room, key=lambda t: -room[t]):
            if shortfall <= 0:
                break
            give = min(room[tier], -(-shortfall * room[tier] // spare))
            taken[tier] += give
            shortfall -= give

    if shortfall > 0:
        # Every tier is exhausted, so the rest is repetition, spread by the mix.
        for tier in sorted(weights, key=lambda t: -weights[t]):
            if shortfall <= 0:
                break
            give = min(shortfall, max(1, int(budget * weights[tier] / scale)))
            taken[tier] += give
            shortfall -= give

    return Allocation(
        per_tier=dict(sorted(taken.items())),
        total=sum(taken.values()),
        passes={
            tier: (taken[tier] / available[tier] if available[tier] else float("inf"))
            for tier in sorted(taken)
        },
    )
