"""The paired bootstrap: agrees with sklearn, finds real gaps, calls small ones ties."""

import numpy as np
import pytest

from calib.compare import TIE_THRESHOLD, compare, macro_f1

K = 4


def make(n, separation, seed, n_classes=K):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, n).astype(np.int64)
    logits = (np.eye(n_classes)[labels] * separation + rng.normal(size=(n, n_classes))).astype(
        np.float32
    )
    return labels, logits


def system(labels, separation, seeds, base=0):
    """One logit array per seed, all scored against the same labels."""
    out = []
    for offset in range(seeds):
        rng = np.random.default_rng(1000 + base + offset)
        noise = rng.normal(size=(len(labels), K))
        out.append((np.eye(K)[labels] * separation + noise).astype(np.float32))
    return out


def test_macro_f1_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    for seed in range(4):
        labels, logits = make(600, 1.2, seed)
        expected = sklearn.f1_score(labels, logits.argmax(1), average="macro", zero_division=0)
        assert macro_f1(labels, logits) == pytest.approx(expected, abs=1e-9)


def test_macro_f1_ignores_a_class_absent_from_both_truth_and_prediction():
    sklearn = pytest.importorskip("sklearn.metrics")
    # Five columns but the fifth class never occurs and is never predicted.
    labels, logits = make(400, 1.5, 0, n_classes=5)
    keep = labels != 4
    labels, logits = labels[keep], logits[keep]
    logits[:, 4] = -50.0
    expected = sklearn.f1_score(labels, logits.argmax(1), average="macro", zero_division=0)
    # sklearn averages over the labels it sees; the absent class is in neither.
    assert macro_f1(labels, logits) == pytest.approx(expected, abs=1e-9)


def test_a_real_gap_is_found_and_the_interval_excludes_zero():
    labels, _ = make(1500, 1.0, 0)
    systems = {
        "strong": system(labels, 2.2, 5, base=0),
        "weak": system(labels, 0.7, 5, base=50),
    }
    result = {item.system: item for item in compare(systems, labels, "weak", n_bootstrap=400)}
    assert result["strong"].mean > result["weak"].mean
    assert result["strong"].low > 0 and result["strong"].beats_reference is True
    assert result["strong"].tie is False
    assert result["weak"].difference is None and result["weak"].low is None


def test_two_systems_of_the_same_quality_are_called_a_tie():
    labels, _ = make(2000, 1.0, 1)
    systems = {"a": system(labels, 1.4, 5, base=0), "b": system(labels, 1.4, 5, base=500)}
    result = {item.system: item for item in compare(systems, labels, "a", n_bootstrap=400)}
    assert abs(result["b"].difference) < TIE_THRESHOLD
    assert result["b"].low < 0 < result["b"].high
    assert result["b"].beats_reference is False


def test_the_interval_is_wider_than_the_spread_over_seeds_alone():
    # Test-set noise is the point of resampling examples: ignoring it would
    # report an interval that is too narrow to be honest.
    labels, _ = make(800, 1.0, 2)
    systems = {"a": system(labels, 1.6, 5, base=0), "b": system(labels, 1.5, 5, base=300)}
    result = {item.system: item for item in compare(systems, labels, "a", n_bootstrap=600)}
    width = result["b"].high - result["b"].low
    seeds_only = np.std(result["b"].per_seed, ddof=1) + np.std(result["a"].per_seed, ddof=1)
    assert width > seeds_only


def test_results_are_reproducible_and_ordered_best_first():
    labels, _ = make(700, 1.0, 3)
    systems = {"a": system(labels, 1.8, 4, base=0), "b": system(labels, 1.0, 4, base=200)}
    first = compare(systems, labels, "a", n_bootstrap=200, seed=7)
    again = compare(systems, labels, "a", n_bootstrap=200, seed=7)
    assert [item.system for item in first] == ["a", "b"]
    assert first[1].low == again[1].low and first[1].high == again[1].high
    other = compare(systems, labels, "a", n_bootstrap=200, seed=8)
    assert other[1].low != first[1].low


def test_per_seed_scores_are_the_real_ones():
    sklearn = pytest.importorskip("sklearn.metrics")
    labels, _ = make(500, 1.0, 4)
    runs = system(labels, 1.5, 3, base=0)
    result = {item.system: item for item in compare({"a": runs}, labels, "a", n_bootstrap=50)}
    expected = [
        sklearn.f1_score(labels, run.argmax(1), average="macro", zero_division=0) for run in runs
    ]
    assert result["a"].per_seed == pytest.approx(expected, abs=1e-9)
    assert result["a"].mean == pytest.approx(float(np.mean(expected)), abs=1e-9)


@pytest.mark.parametrize(
    ("systems_in", "message"),
    [
        ({"a": 3, "b": 2}, "same non-zero number of seeds"),
        ({"a": 0, "b": 0}, "same non-zero number of seeds"),
    ],
)
def test_mismatched_seed_counts_are_refused(systems_in, message):
    labels, _ = make(200, 1.0, 5)
    systems = {name: system(labels, 1.0, count) for name, count in systems_in.items()}
    with pytest.raises(ValueError, match=message):
        compare(systems, labels, "a", n_bootstrap=10)


def test_an_unknown_reference_is_refused():
    labels, _ = make(200, 1.0, 6)
    with pytest.raises(ValueError, match="reference"):
        compare({"a": system(labels, 1.0, 2)}, labels, "missing", n_bootstrap=10)


def test_logits_that_do_not_match_the_labels_are_refused():
    labels, _ = make(200, 1.0, 7)
    systems = {"a": system(labels, 1.0, 2)}
    systems["a"][1] = systems["a"][1][:100]
    with pytest.raises(ValueError, match="expected"):
        compare(systems, labels, "a", n_bootstrap=10)
