"""Packing: full windows, separators where documents meet, nothing invented."""

import numpy as np
import pytest

from model.convert.packing import pack, packed_token_count

SEP = 999


def test_every_window_is_exactly_the_context_length():
    docs = [list(range(i, i + 37)) for i in range(20)]
    windows = list(pack(docs, context=64, separator_id=SEP))
    assert windows
    assert all(window.shape == (64,) for window in windows)


def test_tokens_come_out_in_the_order_they_went_in_with_separators_between():
    docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    # 3 + 1 + 2 + 1 + 4 + 1 = 12 tokens, so two windows of six.
    windows = list(pack(docs, context=6, separator_id=SEP))
    assert [list(w) for w in windows] == [[1, 2, 3, SEP, 4, 5], [SEP, 6, 7, 8, 9, SEP]]


def test_the_tail_that_cannot_fill_a_window_is_dropped_not_padded():
    windows = list(pack([[1, 2, 3, 4, 5]], context=4, separator_id=SEP))
    # 5 tokens plus a separator is 6, so one window of four and two dropped.
    assert [list(w) for w in windows] == [[1, 2, 3, 4]]
    # Nothing is invented to fill the gap: no padding id appears anywhere.
    assert 0 not in windows[0]
    assert len(windows) * 4 < 5 + 1


def test_a_document_longer_than_the_window_spans_several_and_loses_nothing():
    long_doc = list(range(1, 201))
    windows = list(pack([long_doc], context=50, separator_id=SEP))
    assert len(windows) == 4
    joined = [int(t) for w in windows for t in w]
    assert joined == long_doc  # 200 tokens exactly fill four windows
    # A continuation carries no separator, because it is the same document.
    assert SEP not in joined


def test_no_documents_yields_no_windows():
    assert list(pack([], context=16, separator_id=SEP)) == []
    # Nor does a single document too short to fill one.
    assert list(pack([[1, 2]], context=16, separator_id=SEP)) == []


def test_the_dtype_is_honoured_so_shards_can_stay_small():
    windows = list(pack([list(range(100))], context=32, separator_id=SEP, dtype=np.uint16))
    assert windows[0].dtype == np.uint16


def test_a_context_that_is_not_positive_is_refused():
    with pytest.raises(ValueError, match="context must be positive"):
        list(pack([[1, 2]], context=0, separator_id=SEP))
    with pytest.raises(ValueError, match="context must be positive"):
        packed_token_count(100, 1, 0)


def test_the_counted_total_matches_what_packing_actually_yields():
    rng = np.random.default_rng(0)
    docs = [list(rng.integers(0, 500, size=int(n))) for n in rng.integers(5, 80, size=60)]
    total = sum(len(d) for d in docs)
    windows = list(pack(docs, context=64, separator_id=SEP))
    assert packed_token_count(total, len(docs), 64) == 64 * len(windows)
