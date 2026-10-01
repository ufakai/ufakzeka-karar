"""Comparing backbones with an interval that covers both sources of noise.

Two things make a difference between two backbones uncertain: which seeds
were drawn, and which examples the test set happens to contain. Reporting
only the spread over five seeds understates the second, and on test sets of
a few thousand rows the two are of the same order. So the interval here
resamples seeds and test examples together, which is the multi-bootstrap
idea (MultiBERTs, arXiv 2106.16163) applied to a paired design.

Paired means every backbone is scored on the same resampled examples and
the same resampled seeds inside one bootstrap draw, so the shared noise
cancels in the difference instead of widening it.

Macro-F1 is recomputed from a confusion matrix on each draw. Resampling is
done with multinomial weights rather than index lists, which lets one
bincount score a whole draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_BOOTSTRAP = 2000
# A difference under one point is a tie.
TIE_THRESHOLD = 0.01


@dataclass(frozen=True)
class Comparison:
    system: str
    mean: float  # macro-F1 averaged over seeds
    per_seed: list[float]
    difference: float | None  # against the reference, None for the reference itself
    low: float | None  # 95 percent interval of that difference
    high: float | None
    beats_reference: bool | None  # interval strictly above the tie threshold
    tie: bool | None  # interval inside plus or minus the tie threshold


def _cells(labels: np.ndarray, logits: np.ndarray, n_classes: int) -> np.ndarray:
    """One confusion-matrix cell index per test example."""
    return labels * n_classes + logits.argmax(axis=1)


def _macro_f1_from_counts(flat: np.ndarray, n_classes: int) -> float:
    """Macro-F1 from a flattened confusion matrix of weighted counts.

    A class with no true and no predicted rows in this draw contributes
    nothing and is left out of the average, which is what scikit-learn does
    with zero_division=0 and keeps a draw that misses a rare class from
    scoring it as zero.
    """
    matrix = flat.reshape(n_classes, n_classes)
    true_positive = np.diag(matrix)
    actual = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    denominator = actual + predicted
    present = denominator > 0
    if not present.any():
        return float("nan")
    scores = np.zeros(n_classes)
    scores[present] = 2.0 * true_positive[present] / denominator[present]
    return float(scores[present].mean())


def macro_f1(labels: np.ndarray, logits: np.ndarray) -> float:
    n_classes = logits.shape[1]
    counts = np.bincount(_cells(labels, logits, n_classes), minlength=n_classes * n_classes)
    return _macro_f1_from_counts(counts.astype(np.float64), n_classes)


def compare(
    systems: dict[str, list[np.ndarray]],
    labels: np.ndarray,
    reference: str,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 0,
) -> list[Comparison]:
    """Mean macro-F1 per system, and the interval of each difference from `reference`.

    `systems` maps a name to one test-logit array per seed, every array in
    the same example order as `labels`.
    """
    if reference not in systems:
        raise ValueError(f"reference {reference!r} is not one of {sorted(systems)}")
    counts = {name: len(runs) for name, runs in systems.items()}
    if len(set(counts.values())) != 1 or not all(counts.values()):
        raise ValueError(f"every system needs the same non-zero number of seeds: {counts}")
    n_seeds = next(iter(counts.values()))
    n_examples = len(labels)
    n_classes = next(iter(systems.values()))[0].shape[1]
    for name, runs in systems.items():
        for run in runs:
            if run.shape != (n_examples, n_classes):
                raise ValueError(
                    f"{name}: logits are {run.shape}, expected {(n_examples, n_classes)}"
                )

    # One cell index per (system, seed, example), so a draw is one bincount.
    cells = {
        name: [_cells(labels, run, n_classes) for run in runs] for name, runs in systems.items()
    }
    observed = {
        name: [
            _macro_f1_from_counts(
                np.bincount(cell, minlength=n_classes * n_classes).astype(np.float64), n_classes
            )
            for cell in per_seed
        ]
        for name, per_seed in cells.items()
    }

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in systems}
    for _ in range(n_bootstrap):
        # The same examples and the same seeds for every system in this draw.
        weights = rng.multinomial(n_examples, np.full(n_examples, 1.0 / n_examples)).astype(
            np.float64
        )
        chosen_seeds = rng.integers(0, n_seeds, n_seeds)
        for name, per_seed in cells.items():
            scores = [
                _macro_f1_from_counts(
                    np.bincount(per_seed[index], weights=weights, minlength=n_classes * n_classes),
                    n_classes,
                )
                for index in chosen_seeds
            ]
            draws[name].append(float(np.nanmean(scores)))

    reference_draws = np.asarray(draws[reference])
    out = []
    for name in sorted(systems):
        mean = float(np.mean(observed[name]))
        if name == reference:
            out.append(Comparison(name, mean, observed[name], None, None, None, None, None))
            continue
        difference = np.asarray(draws[name]) - reference_draws
        low, high = (float(v) for v in np.percentile(difference, [2.5, 97.5]))
        out.append(
            Comparison(
                system=name,
                mean=mean,
                per_seed=observed[name],
                difference=mean - float(np.mean(observed[reference])),
                low=low,
                high=high,
                beats_reference=low > TIE_THRESHOLD,
                tie=low > -TIE_THRESHOLD and high < TIE_THRESHOLD,
            )
        )
    # Best first, with the reference wherever its score puts it.
    return sorted(out, key=lambda item: -item.mean)
