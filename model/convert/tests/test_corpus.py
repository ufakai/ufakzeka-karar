"""The conversion corpus: what the licence admits, and how the budget is split."""

import pytest

from model.convert.corpus import (
    MIX_DECAY,
    MIX_STABLE,
    SOURCE_LICENSES,
    SOURCE_TIERS,
    TIERS,
    allocate,
    allowed_sources,
    excluded_sources,
    tiers_needing_rebuild,
)


def test_every_source_has_a_licence_and_a_tier_and_they_agree():
    assert set(SOURCE_LICENSES) == set(SOURCE_TIERS)
    for name, tiers in SOURCE_TIERS.items():
        assert tiers, name
        assert set(tiers) <= set(TIERS), name


def test_the_share_alike_source_is_the_only_one_excluded():
    assert excluded_sources() == {"finewiki": "CC-BY-SA-4.0"}
    assert "finewiki" not in allowed_sources()
    # The admitted ODC-By sources are in.
    assert {"fw2hq", "mogan", "finemath", "finepdfs_edu"} <= allowed_sources()


def test_only_the_tier_holding_the_excluded_source_needs_rebuilding():
    assert tiers_needing_rebuild() == {"A": ["finewiki"]}
    # B and C are wholly allowed, so their tokenised shards are usable as they are.
    assert "B" not in tiers_needing_rebuild()
    assert "C" not in tiers_needing_rebuild()


def test_the_mixes_are_proportions_and_cover_every_tier():
    for mix in (MIX_STABLE, MIX_DECAY):
        assert set(mix) == set(TIERS)
        assert sum(mix.values()) == pytest.approx(1.0)
        assert all(value > 0 for value in mix.values())
    # The backbone leant on bulk web while stable and on the curated tier while
    # annealing; the conversion inherits both, so the two must differ.
    assert MIX_DECAY["A"] > MIX_STABLE["A"]


def test_a_budget_is_split_by_the_mix_when_every_tier_can_cover_its_share():
    available = {"A": 10_000, "B": 10_000, "C": 10_000}
    got = allocate(1_000, available, {"A": 0.25, "B": 0.71, "C": 0.04})
    assert got.total == 1_000
    assert got.per_tier == {"A": 250, "B": 710, "C": 40}
    assert got.repeats is False


def test_a_tier_too_small_for_its_share_is_capped_and_the_rest_take_the_slack():
    # C holds 10 tokens but the mix asks for 40.
    available = {"A": 10_000, "B": 10_000, "C": 10}
    got = allocate(1_000, available, {"A": 0.25, "B": 0.71, "C": 0.04})
    assert got.per_tier["C"] == 10
    assert got.total == 1_000
    assert got.repeats is False
    # The slack went to the tiers with room, not to the exhausted one.
    assert got.per_tier["A"] + got.per_tier["B"] == 990


def test_asking_for_more_than_exists_is_reported_as_repetition_not_hidden():
    available = {"A": 100, "B": 100, "C": 100}
    got = allocate(600, available, MIX_STABLE)
    assert got.total == 600
    assert got.repeats is True
    assert max(got.passes.values()) > 1.0


def test_the_real_shape_of_the_pilot_needs_no_repetition():
    # Measured availability, in tokens, with tier A rebuilt without FineWiki.
    available = {"A": 5_750_000_000, "B": 9_500_000_000, "C": 820_000_000}
    got = allocate(5_000_000_000, available, MIX_STABLE)
    assert got.total == 5_000_000_000
    assert got.repeats is False
    # Tier B carries most of a stable-phase budget, as it did for the backbone.
    assert got.per_tier["B"] > got.per_tier["A"] > got.per_tier["C"]


def test_a_budget_that_is_not_positive_is_refused():
    with pytest.raises(ValueError, match="budget must be positive"):
        allocate(0, {"A": 10}, MIX_STABLE)


def test_a_mix_that_covers_none_of_the_tiers_on_hand_is_refused():
    with pytest.raises(ValueError, match="carries weight"):
        allocate(10, {"D": 100}, MIX_STABLE)
