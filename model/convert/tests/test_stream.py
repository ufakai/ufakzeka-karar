"""The data stream: the mixture holds, and a resume reads exactly what it would have."""

import numpy as np
import pytest

from model.convert.corpus import MIX_STABLE
from model.convert.stream import (
    ArrayWindows,
    MixedStream,
    observed_mixture,
    tier_for_step,
)

CONTEXT = 8


def sources(sizes=(40, 40, 40)):
    out = {}
    for tier, size in zip(("A", "B", "C"), sizes, strict=True):
        base = {"A": 1000, "B": 2000, "C": 3000}[tier]
        tokens = np.arange(base, base + size * CONTEXT, dtype=np.int64)
        out[tier] = ArrayWindows(tokens, CONTEXT)
    return out


def test_the_mixture_holds_over_a_run():
    counts = {"A": 0, "B": 0, "C": 0}
    for step in range(60_000):
        counts[tier_for_step(step, MIX_STABLE, seed=7)] += 1
    for tier, weight in MIX_STABLE.items():
        assert counts[tier] / 60_000 == pytest.approx(weight, abs=0.01)


def test_the_tier_for_a_step_needs_no_state_and_never_changes():
    for step in (0, 1, 999, 1_000_000):
        first = tier_for_step(step, MIX_STABLE, seed=3)
        assert tier_for_step(step, MIX_STABLE, seed=3) == first
    # A different seed gives a different sequence.
    a = [tier_for_step(s, MIX_STABLE, seed=3) for s in range(200)]
    b = [tier_for_step(s, MIX_STABLE, seed=4) for s in range(200)]
    assert a != b


def test_a_tier_with_no_weight_is_never_drawn():
    mixture = {"A": 0.5, "B": 0.5, "C": 0.0}
    drawn = {tier_for_step(s, mixture, seed=1) for s in range(5_000)}
    assert drawn == {"A", "B"}


def test_a_resumed_stream_reads_exactly_what_the_unbroken_one_would():
    full = MixedStream(sources(), MIX_STABLE, seed=11)
    expected = [full.take(step)[1].copy() for step in range(50)]

    part = MixedStream(sources(), MIX_STABLE, seed=11)
    got = [part.take(step)[1].copy() for step in range(20)]
    saved_cursors, saved_step = part.state(), 20

    # A fresh process, as after a preemption.
    resumed = MixedStream(sources(), MIX_STABLE, seed=11)
    resumed.load_state(saved_cursors)
    got += [resumed.take(step)[1].copy() for step in range(saved_step, 50)]

    assert len(got) == len(expected)
    assert all(np.array_equal(a, b) for a, b in zip(got, expected, strict=True))


def test_every_window_is_the_context_length_and_comes_from_its_own_tier():
    stream = MixedStream(sources(), MIX_STABLE, seed=5)
    for step in range(100):
        tier, window = stream.take(step)
        assert window.shape == (CONTEXT,)
        base = {"A": 1000, "B": 2000, "C": 3000}[tier]
        assert base <= int(window[0]) < base + 40 * CONTEXT


def test_a_tier_that_runs_out_wraps_rather_than_stopping():
    # C is tiny and heavily drawn, so it must repeat rather than raise.
    stream = MixedStream(sources(sizes=(40, 40, 2)), {"C": 1.0}, seed=2)
    seen = [stream.take(step)[1][0] for step in range(10)]
    assert len(set(seen)) == 2
    assert stream.state()["C"] == 10


def test_a_batch_reports_where_its_rows_came_from():
    stream = MixedStream(sources(), MIX_STABLE, seed=9)
    rows, counts = stream.batch(step=0, size=32)
    assert rows.shape == (32, CONTEXT)
    assert sum(counts.values()) == 32


def test_the_observed_mixture_matches_the_plan_over_many_batches():
    stream = MixedStream(sources(), MIX_STABLE, seed=13)
    counts = [stream.batch(step, 64)[1] for step in range(200)]
    seen = observed_mixture(counts)
    for tier, weight in MIX_STABLE.items():
        assert seen[tier] == pytest.approx(weight, abs=0.02)
    assert observed_mixture([]) == {}


def test_a_mixture_naming_a_tier_with_no_source_is_refused():
    with pytest.raises(ValueError, match="no source"):
        MixedStream({"A": ArrayWindows(np.arange(80), CONTEXT)}, MIX_STABLE, seed=1)


def test_saved_cursors_for_an_unknown_tier_are_refused():
    stream = MixedStream(sources(), MIX_STABLE, seed=1)
    with pytest.raises(ValueError, match="unknown tiers"):
        stream.load_state({"D": 5})


def test_impossible_arguments_are_refused():
    with pytest.raises(ValueError, match="step cannot be negative"):
        tier_for_step(-1, MIX_STABLE, seed=1)
    with pytest.raises(ValueError, match="no positive weight"):
        tier_for_step(0, {"A": 0.0}, seed=1)
    with pytest.raises(IndexError):
        ArrayWindows(np.arange(80), CONTEXT).window(10)


def _shards(tmp_path, sizes=(50, 30, 44), dtype=np.uint16):
    """Raw uint16 streams, the format the backbone's corpus is written in."""
    paths = []
    start = 0
    for i, n in enumerate(sizes):
        path = tmp_path / f"part_{i}.bin"
        np.arange(start, start + n, dtype=dtype).tofile(path)
        paths.append(path)
        start += n
    return paths


def test_sharded_windows_never_load_a_whole_tier(tmp_path):
    from model.convert.stream import ShardedWindows

    paths = _shards(tmp_path, sizes=(50, 30, 44))
    windows = ShardedWindows(paths, context=10)
    # 5 + 3 + 4 windows; the 4, 0 and 4 leftover tokens are dropped.
    assert len(windows) == 12
    # Nothing is resident until a window is asked for.
    assert windows._maps == {}
    first = windows.window(0)
    assert first.tolist() == list(range(10))
    assert len(windows._maps) == 1


def test_a_window_never_straddles_two_shards(tmp_path):
    from model.convert.stream import ShardedWindows

    windows = ShardedWindows(_shards(tmp_path, sizes=(50, 30, 44)), context=10)
    # Window 5 is the first of the second shard, which starts at token 50.
    assert windows.window(5).tolist() == list(range(50, 60))
    # Window 8 is the first of the third shard, which starts at token 80.
    assert windows.window(8).tolist() == list(range(80, 90))
    # Every window is contiguous and ascending, so none spans a boundary.
    for i in range(len(windows)):
        w = windows.window(i)
        assert (np.diff(w) == 1).all()


def test_sharded_windows_widen_to_int64_for_the_model(tmp_path):
    from model.convert.stream import ShardedWindows

    windows = ShardedWindows(_shards(tmp_path, dtype=np.uint16), context=10)
    assert windows.window(0).dtype == np.int64


def test_sharded_windows_drive_a_mixed_stream(tmp_path):
    from model.convert.stream import ShardedWindows

    a = tmp_path / "A"
    a.mkdir()
    stream = MixedStream({"A": ShardedWindows(_shards(a), context=10)}, {"A": 1.0}, seed=1)
    rows, counts = stream.batch(0, 4)
    assert rows.shape == (4, 10)
    assert counts == {"A": 4}


def test_impossible_shard_sets_are_refused(tmp_path):
    from model.convert.stream import ShardedWindows

    with pytest.raises(ValueError, match="no token shards"):
        ShardedWindows([], context=10)
    with pytest.raises(ValueError, match="context must be positive"):
        ShardedWindows(_shards(tmp_path), context=0)
    with pytest.raises(IndexError):
        ShardedWindows(_shards(tmp_path), context=10).window(999)


def test_a_shard_that_is_not_a_whole_number_of_tokens_is_refused(tmp_path):
    from model.convert.stream import ShardedWindows

    odd = tmp_path / "odd.bin"
    odd.write_bytes(b"\x01\x02\x03")  # three bytes is not a whole uint16 count
    with pytest.raises(ValueError, match="not a whole number of tokens"):
        ShardedWindows([odd], context=1)


def test_the_mixture_changes_for_the_tail_of_the_run():
    """Bug 12: MIX_DECAY was declared and never reached, so the tail of the
    run would have trained on the stable diet with nothing saying so."""
    from model.convert.corpus import MIX_DECAY, MIX_STABLE
    from model.convert.stream import staged_mixture

    at = staged_mixture(MIX_STABLE, MIX_DECAY, total_steps=1000, decay_frac=0.2)
    assert at(0) == MIX_STABLE
    assert at(799) == MIX_STABLE
    assert at(800) == MIX_DECAY
    assert at(5000) == MIX_DECAY

    stream = MixedStream(sources(), at, seed=3)
    early = [stream.mixture_at(s) for s in (0, 500)]
    late = [stream.mixture_at(s) for s in (800, 999)]
    assert all(m == MIX_STABLE for m in early)
    assert all(m == MIX_DECAY for m in late)


def test_the_tail_really_draws_more_of_the_curated_tier():
    from model.convert.corpus import MIX_DECAY, MIX_STABLE
    from model.convert.stream import staged_mixture

    at = staged_mixture(MIX_STABLE, MIX_DECAY, total_steps=1000, decay_frac=0.2)
    stream = MixedStream(sources(sizes=(400, 400, 400)), at, seed=5)
    stable = observed_mixture([stream.batch(s, 64)[1] for s in range(100)])
    decay = observed_mixture([stream.batch(s, 64)[1] for s in range(800, 900)])
    assert stable["A"] == pytest.approx(MIX_STABLE["A"], abs=0.03)
    assert decay["A"] == pytest.approx(MIX_DECAY["A"], abs=0.03)
    assert decay["A"] > stable["A"]


def test_every_row_of_one_batch_uses_that_step_s_mixture():
    # The window counter is not the training step; resolving the mixture from
    # the wrong one would straddle the boundary inside a single batch.
    from model.convert.stream import staged_mixture

    at = staged_mixture({"A": 1.0, "B": 0.0, "C": 0.0}, {"A": 0.0, "B": 1.0, "C": 0.0}, 10, 0.2)
    stream = MixedStream(sources(), at, seed=1)
    assert stream.batch(0, 8)[1] == {"A": 8, "B": 0, "C": 0}
    assert stream.batch(9, 8)[1] == {"A": 0, "B": 8, "C": 0}


def test_a_staged_mixture_missing_a_tier_is_refused_at_construction():
    from model.convert.stream import staged_mixture

    at = staged_mixture({"A": 1.0}, {"D": 1.0}, total_steps=10, decay_frac=0.2)
    with pytest.raises(ValueError, match="no source"):
        MixedStream(sources(), at, seed=1)
    with pytest.raises(ValueError, match="total_steps must be positive"):
        staged_mixture({"A": 1.0}, {"A": 1.0}, total_steps=0, decay_frac=0.2)


def test_reads_are_scattered_over_the_tier_and_still_visit_everything_once():
    """File order was an unchosen curriculum, and a third of a tier was a prefix."""
    from model.convert.stream import scattered_index

    for size in (1, 2, 7, 1000, 7_487_123):
        sample = range(size) if size <= 1000 else range(2000)
        seen = [scattered_index(cursor, size, seed=1) for cursor in sample]
        assert len(set(seen)) == len(seen), f"a window repeats early at size {size}"
        assert all(0 <= index < size for index in seen)
    assert sorted(scattered_index(c, 1000, seed=1) for c in range(1000)) == list(range(1000))

    # The first third of the reads reaches every part of the tier, not its head.
    early = [scattered_index(cursor, 9000, seed=1) for cursor in range(3000)]
    assert min(early) < 100 and max(early) > 8900
    # Consecutive reads are far apart, so a batch is not one stretch of one file.
    assert abs(early[1] - early[0]) > 1000
    # A different seed is a different order; the same seed is the same order.
    assert scattered_index(5, 9000, seed=2) != scattered_index(5, 9000, seed=1)
    assert scattered_index(9005, 9000, seed=1) == scattered_index(5, 9000, seed=1)
