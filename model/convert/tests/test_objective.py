"""The masked-language objective: what it hides, what it asks for, what it never touches."""

import numpy as np
import pytest

from model.convert.objective import (
    IGNORE,
    MASK_SHARE,
    RANDOM_SHARE,
    mask_tokens,
    masking_rate,
)

VOCAB = 40_960
MASK_ID = 40_948  # <|reserved_0|>
PAD = 40_947


def ids(shape=(8, 256), seed=0, high=40_000):
    return np.random.default_rng(seed).integers(0, high, size=shape).astype(np.int64)


def test_labels_are_the_originals_where_selected_and_ignored_everywhere_else():
    original = ids()
    rng = np.random.default_rng(1)
    inputs, labels = mask_tokens(original, rate=0.3, mask_id=MASK_ID, vocab_size=VOCAB, rng=rng)
    selected = labels != IGNORE
    assert np.array_equal(labels[selected], original[selected])
    assert (labels[~selected] == IGNORE).all()
    # Nothing outside the selection is altered, so the model sees the real text there.
    assert np.array_equal(inputs[~selected], original[~selected])


def test_the_selected_share_is_close_to_the_requested_rate():
    original = ids(shape=(64, 512))
    rng = np.random.default_rng(2)
    _, labels = mask_tokens(original, rate=0.3, mask_id=MASK_ID, vocab_size=VOCAB, rng=rng)
    assert (labels != IGNORE).mean() == pytest.approx(0.3, abs=0.01)


def test_the_replacement_split_is_eighty_ten_ten():
    original = ids(shape=(64, 512))
    rng = np.random.default_rng(3)
    inputs, labels = mask_tokens(original, rate=0.3, mask_id=MASK_ID, vocab_size=VOCAB, rng=rng)
    selected = labels != IGNORE
    masked = (inputs[selected] == MASK_ID).mean()
    unchanged = (inputs[selected] == original[selected]).mean()
    assert masked == pytest.approx(MASK_SHARE, abs=0.02)
    # A random draw can land on the original id, so this is a lower bound plus noise.
    assert unchanged == pytest.approx(1 - MASK_SHARE - RANDOM_SHARE, abs=0.02)


def test_padding_is_never_selected_and_never_corrupted():
    original = ids()
    original[:, 200:] = PAD
    rng = np.random.default_rng(4)
    inputs, labels = mask_tokens(
        original, rate=0.9, mask_id=MASK_ID, vocab_size=VOCAB, rng=rng, protected=frozenset({PAD})
    )
    assert (labels[:, 200:] == IGNORE).all()
    assert (inputs[:, 200:] == PAD).all()


def test_a_batch_always_has_something_to_predict():
    # A tiny batch at a low rate can select nothing by chance, and a loss over
    # zero elements is NaN, which would end a paid run.
    for seed in range(40):
        rng = np.random.default_rng(seed)
        _, labels = mask_tokens(
            ids(shape=(1, 4), seed=seed), rate=0.01, mask_id=MASK_ID, vocab_size=VOCAB, rng=rng
        )
        assert (labels != IGNORE).any()


def test_a_rate_of_zero_hides_nothing_and_a_rate_of_one_hides_everything_eligible():
    original = ids(shape=(4, 32))
    _, labels = mask_tokens(
        original, rate=0.0, mask_id=MASK_ID, vocab_size=VOCAB, rng=np.random.default_rng(5)
    )
    assert (labels == IGNORE).all()
    _, labels = mask_tokens(
        original, rate=1.0, mask_id=MASK_ID, vocab_size=VOCAB, rng=np.random.default_rng(5)
    )
    assert (labels != IGNORE).all()


def test_the_same_seed_gives_the_same_corruption():
    original = ids()
    a = mask_tokens(
        original, rate=0.3, mask_id=MASK_ID, vocab_size=VOCAB, rng=np.random.default_rng(7)
    )
    b = mask_tokens(
        original, rate=0.3, mask_id=MASK_ID, vocab_size=VOCAB, rng=np.random.default_rng(7)
    )
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])  # fmt: skip


def test_random_replacements_stay_inside_the_vocabulary():
    original = ids(shape=(32, 128))
    inputs, _ = mask_tokens(
        original, rate=1.0, mask_id=MASK_ID, vocab_size=VOCAB, rng=np.random.default_rng(8)
    )
    assert inputs.min() >= 0 and inputs.max() < VOCAB


@pytest.mark.parametrize(("rate", "mask_id"), [(1.5, MASK_ID), (-0.1, MASK_ID), (0.3, VOCAB)])
def test_impossible_arguments_are_refused(rate, mask_id):
    with pytest.raises(ValueError):
        mask_tokens(
            ids(shape=(2, 8)), rate=rate, mask_id=mask_id, vocab_size=VOCAB,
            rng=np.random.default_rng(9),
        )  # fmt: skip


def test_the_rate_steps_down_for_the_tail_of_the_run():
    kwargs = {"stable": 0.3, "decay": 0.1, "decay_frac": 0.15}
    assert masking_rate(0, 1000, **kwargs) == 0.3
    assert masking_rate(849, 1000, **kwargs) == 0.3
    assert masking_rate(850, 1000, **kwargs) == 0.1
    assert masking_rate(999, 1000, **kwargs) == 0.1
    # Past the end it stays in the decay phase rather than jumping back.
    assert masking_rate(5000, 1000, **kwargs) == 0.1


def test_a_run_with_no_steps_or_an_impossible_tail_is_refused():
    with pytest.raises(ValueError, match="total_steps must be positive"):
        masking_rate(0, 0, stable=0.3, decay=0.1, decay_frac=0.15)
    with pytest.raises(ValueError, match="decay_frac"):
        masking_rate(0, 100, stable=0.3, decay=0.1, decay_frac=1.5)
