"""Large option sets: scored in passes, every option kept, independent of the listing."""

import pytest

torch = pytest.importorskip("torch")

from model.head.infer import chunked_choice  # noqa: E402
from model.head.tests.test_head import choice, encode, head  # noqa: E402


def test_a_large_set_keeps_every_option_and_the_listing_order_does_not_matter():
    model = head(causal=False)
    options = [f"seçenek {i:02d}" for i in range(27)]
    first = chunked_choice(model, "Bir metin.", choice(options), encode)
    again = chunked_choice(model, "Bir metin.", choice(options[::-1]), encode)
    assert list(first) == options
    assert first == pytest.approx(again, abs=1e-6)
    assert sum(first.values()) == pytest.approx(1.0, abs=1e-9)


def test_how_the_set_is_split_does_not_change_the_answer():
    model = head(causal=False)
    options = [f"seçenek {i:02d}" for i in range(12)]
    by_ten = chunked_choice(model, "Bir metin.", choice(options), encode)
    by_three = chunked_choice(model, "Bir metin.", choice(options), encode, per_pass=3)
    assert by_ten == pytest.approx(by_three, abs=1e-5)


def test_a_set_that_fits_one_pass_is_scored_in_one():
    model = head(causal=False)
    out = chunked_choice(model, "Metin.", choice(["a", "b", "c"]), encode)
    assert list(out) == ["a", "b", "c"]
