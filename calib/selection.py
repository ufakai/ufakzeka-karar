"""Offline checkpoint selection: refine, then calibrate.

A training run writes validation and test logits at every evaluation point
and keeps no checkpoints. Selection happens afterwards, here, and follows
arXiv 2501.19195: pick the point with the lowest validation log loss AFTER
temperature scaling, which estimates the refinement error, and leave
calibration to a post-hoc step. Picking on raw validation loss would stop
too early, because raw loss rises as soon as the model turns overconfident
even while it is still getting better at separating the classes.

This module reads validation files only. It never opens a test file: the
caller asks for the test logits of the selected point by name, and that is
the single place test data enters.

A run folder:
    run.json                      metadata written by the training code
    labels_validation.npy         int64 [n]
    logits_validation_step<N>.npy float32 [n, k], one per evaluation point
    labels_test.npy, logits_test_step<N>.npy
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The paper sees a clear benefit from a validation size of about 1,600. Below
# that, fitting and scoring the temperature on the same rows overfits, so the
# five-fold estimate is used.
SMALL_VALIDATION = 1600

_STEP = re.compile(r"^logits_validation_step(\d+)\.npy$")


@dataclass(frozen=True)
class Selection:
    step: int
    score: float  # validation log loss after temperature scaling at the selected step
    estimator: str  # the probmetrics metric name that produced the scores
    scores: dict[int, float]  # every evaluation point, for the record
    # What other rules would have picked, kept so the effect of this rule can be audited.
    audit: dict[str, int]


def refinement_estimator(n_validation: int) -> str:
    split = "cv-5" if n_validation < SMALL_VALIDATION else "all"
    return f"refinement_logloss_ts-mix_{split}"


def _metrics(names: list[str], labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    # Imported here: probmetrics pulls in torch and numba, and the module
    # should import without them for the unit tests of the file handling.
    import torch
    from probmetrics.metrics import Metrics

    result = Metrics.from_names(names).compute_all_from_labels_logits(
        torch.from_numpy(labels.astype(np.int64)), torch.from_numpy(logits.astype(np.float32))
    )
    return {name: float(value) for name, value in result.items()}


def macro_f1(labels: np.ndarray, logits: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(labels, logits.argmax(axis=1), average="macro"))


def validation_steps(run_dir: Path) -> list[int]:
    steps = sorted(int(m.group(1)) for p in run_dir.iterdir() if (m := _STEP.match(p.name)))
    if not steps:
        raise FileNotFoundError(f"no validation logits in {run_dir}")
    return steps


def select_point(run_dir: Path) -> Selection:
    labels = np.load(run_dir / "labels_validation.npy")
    estimator = refinement_estimator(len(labels))
    scores: dict[int, float] = {}
    raw: dict[str, dict[int, float]] = {"logloss": {}, "brier": {}, "macro_f1": {}}
    for step in validation_steps(run_dir):
        logits = np.load(run_dir / f"logits_validation_step{step}.npy")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{run_dir}: step {step} has {logits.shape[0]} rows, labels {len(labels)}"
            )
        if not np.isfinite(logits).all():
            # A diverged point is never selected, and is visible in the record.
            scores[step] = float("inf")
            for values in raw.values():
                values[step] = float("nan")
            continue
        got = _metrics([estimator, "logloss", "brier"], labels, logits)
        scores[step] = got[estimator]
        raw["logloss"][step] = got["logloss"]
        raw["brier"][step] = got["brier"]
        raw["macro_f1"][step] = macro_f1(labels, logits)
    if not np.isfinite(min(scores.values())):
        raise ValueError(f"{run_dir}: every evaluation point has non-finite logits")

    # min() returns the first minimum, and steps are ascending: ties go to the earlier point.
    best = min(scores, key=scores.__getitem__)
    finite = [s for s in scores if np.isfinite(scores[s])]
    audit = {
        "raw_logloss": min(finite, key=raw["logloss"].__getitem__),
        "raw_brier": min(finite, key=raw["brier"].__getitem__),
        "macro_f1": max(finite, key=raw["macro_f1"].__getitem__),
        "last": max(finite),
    }
    return Selection(best, scores[best], estimator, scores, audit)


def select_learning_rate(run_dirs: dict[float, Path]) -> tuple[float, dict[float, Selection]]:
    """Pick the learning rate whose best point has the lowest refinement score.

    Returns the rate and every rate's selection. A rate at either end of the
    grid is the caller's cue to extend the grid.
    """
    selections = {rate: select_point(path) for rate, path in sorted(run_dirs.items())}
    best = min(selections, key=lambda rate: selections[rate].score)
    return best, selections
