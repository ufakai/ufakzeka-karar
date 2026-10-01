"""Offline selection and the run report, on synthetic logits with known properties.

Needs the instrument dependency group. Run with `just test-instrument`.
"""

import numpy as np
import pytest

pytest.importorskip("probmetrics")
pytest.importorskip("sklearn")

from calib.metrics import constant_predictor_brier, report  # noqa: E402
from calib.selection import (  # noqa: E402
    SMALL_VALIDATION,
    refinement_estimator,
    select_learning_rate,
    select_point,
)

K = 4


def make_logits(n, separation, scale, seed):
    """Gaussian class evidence. `separation` sets how well classes are told
    apart, `scale` multiplies the logits: above 1 is overconfident."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, K, n)
    evidence = np.eye(K)[labels] * separation + rng.normal(size=(n, K))
    # For this generator the calibrated logits are separation * evidence.
    return (evidence * separation * scale).astype(np.float32), labels.astype(np.int64)


def write_run(path, points, n=2400, seed=0):
    """points: {step: (separation, scale)}. Same labels and noise at every step."""
    path.mkdir(parents=True)
    labels = None
    for step, (separation, scale) in points.items():
        logits, labels = make_logits(n, separation, scale, seed)
        np.save(path / f"logits_validation_step{step}.npy", logits)
    np.save(path / "labels_validation.npy", labels)
    return path


def test_the_rule_prefers_a_sharper_model_even_when_it_is_overconfident(tmp_path):
    # Step 200 separates the classes better but is badly overconfident, so
    # its raw validation loss is worse. Refinement sees through the scale.
    run = write_run(tmp_path / "run", {100: (1.0, 1.0), 200: (1.6, 6.0)})
    selection = select_point(run)
    assert selection.step == 200
    assert selection.audit["raw_logloss"] == 100
    assert selection.audit["macro_f1"] == 200
    assert selection.estimator == "refinement_logloss_ts-mix_all"
    assert set(selection.scores) == {100, 200}
    assert selection.score == selection.scores[200] < selection.scores[100]


def test_the_score_does_not_depend_on_the_logit_scale(tmp_path):
    run = write_run(tmp_path / "run", {1: (1.2, 1.0), 2: (1.2, 5.0), 3: (1.2, 0.3)})
    scores = select_point(run).scores
    assert scores[1] == pytest.approx(scores[2], abs=2e-3)
    assert scores[1] == pytest.approx(scores[3], abs=2e-3)


def test_ties_go_to_the_earlier_point(tmp_path):
    run = write_run(tmp_path / "run", {10: (1.0, 1.0), 20: (1.0, 1.0), 30: (1.0, 1.0)})
    assert select_point(run).step == 10


def test_a_diverged_point_is_never_selected_and_stays_in_the_record(tmp_path):
    run = write_run(tmp_path / "run", {1: (1.0, 1.0), 2: (2.0, 1.0)})
    broken = np.load(run / "logits_validation_step2.npy")
    broken[0, 0] = np.nan
    np.save(run / "logits_validation_step2.npy", broken)
    selection = select_point(run)
    assert selection.step == 1
    assert selection.scores[2] == float("inf")
    assert selection.audit["last"] == 1


def test_a_run_where_everything_diverged_is_an_error(tmp_path):
    run = write_run(tmp_path / "run", {1: (1.0, 1.0)})
    np.save(run / "logits_validation_step1.npy", np.full((2400, K), np.inf, dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        select_point(run)


def test_small_validation_sets_use_the_five_fold_estimate(tmp_path):
    assert refinement_estimator(SMALL_VALIDATION - 1).endswith("cv-5")
    assert refinement_estimator(SMALL_VALIDATION).endswith("all")
    run = write_run(tmp_path / "run", {1: (1.0, 1.0), 2: (1.5, 3.0)}, n=400)
    selection = select_point(run)
    assert selection.estimator == "refinement_logloss_ts-mix_cv-5"
    assert selection.step == 2


def test_selection_never_opens_a_test_file(tmp_path):
    run = write_run(tmp_path / "run", {1: (1.0, 1.0), 2: (1.5, 1.0)})
    # Files that would raise if np.load touched them.
    (run / "logits_test_step1.npy").write_text("not an array", encoding="utf-8")
    (run / "logits_test_step2.npy").write_text("not an array", encoding="utf-8")
    (run / "labels_test.npy").write_text("not an array", encoding="utf-8")
    assert select_point(run).step == 2


def test_mismatched_rows_are_refused(tmp_path):
    run = write_run(tmp_path / "run", {1: (1.0, 1.0)})
    np.save(run / "labels_validation.npy", np.zeros(10, dtype=np.int64))
    with pytest.raises(ValueError, match="rows"):
        select_point(run)


def test_no_logits_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        select_point(tmp_path / "empty")


def test_learning_rate_selection_uses_each_rates_best_point(tmp_path):
    runs = {
        1e-5: write_run(tmp_path / "a", {1: (0.8, 1.0), 2: (1.0, 1.0)}),
        3e-5: write_run(tmp_path / "b", {1: (1.2, 1.0), 2: (1.7, 4.0)}),
        1e-4: write_run(tmp_path / "c", {1: (1.4, 1.0), 2: (0.2, 1.0)}),
    }
    best, selections = select_learning_rate(runs)
    assert best == 3e-5
    assert selections[3e-5].step == 2
    assert selections[1e-4].step == 1


def test_report_recovers_the_temperature_and_leaves_decisions_alone():
    val_logits, val_labels = make_logits(3000, 1.3, 4.0, seed=1)
    test_logits, test_labels = make_logits(3000, 1.3, 4.0, seed=2)
    got = report(test_logits, test_labels, val_logits, val_labels)

    # Logits were scaled by 4, so the fitted inverse temperature is near 1/4.
    assert got["inverse_temperature"] == pytest.approx(0.25, rel=0.15)
    # Temperature scaling cannot change an argmax.
    for key in ("macro_f1", "accuracy", "mcc"):
        assert got["calibrated"][key] == pytest.approx(got["raw"][key])
    # It does repair the overconfidence.
    assert got["calibrated"]["brier"] < got["raw"]["brier"]
    assert got["calibrated"]["logloss"] < got["raw"]["logloss"]
    assert got["calibrated"]["smooth_ece"] < got["raw"]["smooth_ece"]
    assert got["brier_improvement"] == pytest.approx(
        got["raw"]["brier"] - got["calibrated"]["brier"]
    )
    # Refinement is a property of the ranking, so scaling barely moves it.
    assert got["calibrated"]["brier_refinement"] == pytest.approx(
        got["raw"]["brier_refinement"], abs=5e-3
    )
    assert got["raw"]["brier_calibration_error"] > got["calibrated"]["brier_calibration_error"]
    assert (got["n_test"], got["n_validation"], got["n_classes"]) == (3000, 3000, K)


def test_brier_conventions():
    labels = np.array([0, 0, 0, 1], dtype=np.int64)
    # Gini index of (0.75, 0.25).
    assert constant_predictor_brier(labels, 2) == pytest.approx(1 - (0.75**2 + 0.25**2))

    val_logits, val_labels = make_logits(2000, 1.0, 1.0, seed=3)
    test_logits, test_labels = make_logits(2000, 1.0, 1.0, seed=4)
    raw = report(test_logits, test_labels, val_logits, val_labels)["raw"]
    probs = np.exp(test_logits - test_logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    by_hand = float(((probs - np.eye(K)[test_labels]) ** 2).sum(1).mean())
    assert raw["brier"] == pytest.approx(by_hand, rel=1e-4)  # sum over classes
    assert raw["root_brier"] == pytest.approx(np.sqrt(by_hand / 2))
    assert 0 < raw["brier_normalised"] < 1  # better than the constant predictor


def test_a_model_with_no_skill_scores_about_one_when_normalised():
    rng = np.random.default_rng(5)
    labels = rng.integers(0, K, 4000).astype(np.int64)
    flat = np.zeros((4000, K), dtype=np.float32)
    got = report(flat, labels, flat, labels)
    assert got["raw"]["brier_normalised"] == pytest.approx(1.0, abs=0.01)


def test_non_finite_logits_are_refused():
    logits, labels = make_logits(100, 1.0, 1.0, seed=6)
    bad = logits.copy()
    bad[3, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        report(bad, labels, logits, labels)
