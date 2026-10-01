"""The instrument's test-set report for one run.

Everything is reported before and after temperature scaling. The temperature
is fitted on validation logits only and applied to test logits. Conventions,
stated once because they differ between libraries:

  Brier        sum over classes of squared error, range 0 to 2 (the
               convention of arXiv 2501.19195 and of probmetrics; twice
               scikit-learn's binary value).
  root Brier   square root of Brier / 2, so it lies in 0 to 1. Recommended
               next to binned or kernel metrics when samples are few
               (arXiv 2504.18278), because those have high variance.
  smooth ECE   on top-label confidence (relplot, through probmetrics).
  ECE-15       15 equal-width bins, top label. Biased and bin-dependent;
               kept only so older numbers can be compared.
  calibration / refinement
               the variational split of arXiv 2501.19195, five-fold on the
               test set so the calibrator is not scored on rows it was
               fitted on.
  normalised   Brier divided by the Brier of the constant predictor that
               always outputs the test label frequencies. Comparable across
               sets with different class counts; 1.0 means no skill.
"""

from __future__ import annotations

import numpy as np

_BASE = ["logloss", "brier", "accuracy", "ece-15", "smece"]
_SPLIT = ["calib-err_brier_ts-mix_cv-5", "refinement_brier_ts-mix_cv-5"]


def _probmetrics(names: list[str], labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    import torch
    from probmetrics.metrics import Metrics

    result = Metrics.from_names(names).compute_all_from_labels_logits(
        torch.from_numpy(labels.astype(np.int64)), torch.from_numpy(logits.astype(np.float32))
    )
    return {name: float(value) for name, value in result.items()}


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray):
    """Temperature scaling by bisection with Laplace smoothing ("ts-mix"), fitted on validation."""
    import torch
    from probmetrics.calibrators import get_calibrator
    from probmetrics.distributions import CategoricalLogits

    calibrator = get_calibrator("ts-mix")
    calibrator.fit_torch(
        CategoricalLogits(torch.from_numpy(val_logits.astype(np.float32))),
        torch.from_numpy(val_labels.astype(np.int64)),
    )
    return calibrator


def apply_temperature(calibrator, logits: np.ndarray) -> np.ndarray:
    import torch
    from probmetrics.distributions import CategoricalLogits

    dist = calibrator.predict_proba_torch(
        CategoricalLogits(torch.from_numpy(logits.astype(np.float32)))
    )
    return dist.get_probs().numpy().astype(np.float64)


def inverse_temperature(calibrator) -> float | None:
    """The fitted 1/T, found wherever the library keeps it."""

    def find(obj, depth: int = 0):
        if hasattr(obj, "invtemp_"):
            return float(obj.invtemp_)
        if depth < 3 and hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                found = find(value, depth + 1)
                if found is not None:
                    return found
        return None

    return find(calibrator)


def constant_predictor_brier(labels: np.ndarray, n_classes: int) -> float:
    """Brier (sum convention) of always predicting the label frequencies: the Gini index."""
    freq = np.bincount(labels, minlength=n_classes) / len(labels)
    return float(1.0 - np.sum(freq**2))


def _block(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import f1_score, matthews_corrcoef

    n_classes = logits.shape[1]
    out = _probmetrics(_BASE + _SPLIT, labels, logits)
    predicted = logits.argmax(axis=1)
    reference = constant_predictor_brier(labels, n_classes)
    return {
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "accuracy": out["accuracy"],
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "logloss": out["logloss"],
        "brier": out["brier"],
        "root_brier": float(np.sqrt(out["brier"] / 2.0)),
        "brier_normalised": out["brier"] / reference if reference > 0 else float("nan"),
        "smooth_ece": out["smece"],
        "ece_15": out["ece-15"],
        "brier_calibration_error": out["calib-err_brier_ts-mix_cv-5"],
        "brier_refinement": out["refinement_brier_ts-mix_cv-5"],
    }


def report(
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    val_logits: np.ndarray,
    val_labels: np.ndarray,
) -> dict:
    """Metrics for one selected evaluation point, raw and temperature-scaled."""
    for name, array in (("test", test_logits), ("validation", val_logits)):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} logits contain non-finite values")
    calibrator = fit_temperature(val_logits, val_labels)
    scaled = apply_temperature(calibrator, test_logits)
    # The metric code takes logits; log-probabilities are logits of the same distribution.
    scaled_logits = np.log(np.clip(scaled, 1e-12, 1.0))

    raw = _block(test_labels, test_logits)
    calibrated = _block(test_labels, scaled_logits)
    return {
        "n_test": int(len(test_labels)),
        "n_validation": int(len(val_labels)),
        "n_classes": int(test_logits.shape[1]),
        "inverse_temperature": inverse_temperature(calibrator),
        "raw": raw,
        "calibrated": calibrated,
        # Post-hoc improvement in a proper score (arXiv 2605.30188).
        "brier_improvement": raw["brier"] - calibrated["brier"],
        "logloss_improvement": raw["logloss"] - calibrated["logloss"],
    }
