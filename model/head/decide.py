"""The backbone decision, computed from the runs' saved predictions.

The statistic is the mean over tasks of macro F1, and the decision reads the
interval of its difference between two systems (converted minus causal). The
interval comes from a paired bootstrap that resamples, in every draw, the
validation items within each task and the seeds of each system, and scores both
systems on the same resampled items, so the noise they share cancels.

A system is a list of runs (one per seed), each a predictions file of per-row
records {row_id, task, keys, probs, target}. Rows are matched across runs by
row_id, which is a content hash and the same in every run. Macro F1 is over
option names, so a MASSIVE row that carries its own ten intents is scored on
the intent it predicts, which is the same thing the 59-way question asks.

The rule, fixed before any number: keep the converted backbone only if
the whole interval lies above zero. Anything else, a tie inside one point or an
unclear interval, keeps the causal backbone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TIE = 0.01


def load(path: Path) -> dict[str, dict]:
    records = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["row_id"]] = record
    return records


def _encode(records_by_run: list[dict[str, dict]], row_ids: list[str]):
    """Gold and predicted labels as integer arrays per run, over a shared vocabulary."""
    names: dict[str, int] = {}

    def index(name: str) -> int:
        return names.setdefault(name, len(names))

    gold = np.array(
        [index(max(records_by_run[0][r]["target"], key=records_by_run[0][r]["target"].get))
         for r in row_ids]
    )  # fmt: skip
    predicted = []
    for records in records_by_run:
        predicted.append(
            np.array([index(records[r]["keys"][int(np.argmax(records[r]["probs"]))])
                      for r in row_ids])
        )  # fmt: skip
    return gold, np.stack(predicted), len(names)


def _macro_f1(gold: np.ndarray, predicted: np.ndarray, weights: np.ndarray, classes: int) -> float:
    """Macro F1 with rows weighted by how often the bootstrap drew them."""
    true_positive = np.bincount(gold[gold == predicted], weights[gold == predicted], classes)
    actual = np.bincount(gold, weights, classes)
    guessed = np.bincount(predicted, weights, classes)
    denominator = actual + guessed
    present = denominator > 0
    return float(np.mean(2 * true_positive[present] / denominator[present]))


def decide(
    converted: list[Path],
    causal: list[Path],
    *,
    draws: int = 2000,
    seed: int = 0,
    exclude: frozenset[str] = frozenset(),
    only: frozenset[str] | None = None,
) -> dict:
    """The verdict from two lists of predictions files, one per seed.

    `exclude` leaves tasks out of the statistic and `only` keeps just those, so
    a rule that names which tasks decide is applied as written.
    """
    runs_a = [load(p) for p in converted]
    runs_b = [load(p) for p in causal]
    shared = set.intersection(*(set(r) for r in runs_a + runs_b))
    tasks = sorted(
        t
        for t in {runs_a[0][r]["task"] for r in shared}
        if t not in exclude and (only is None or t in only)
    )
    if not tasks:
        raise ValueError("no task left to decide on")
    shared = {r for r in shared if runs_a[0][r]["task"] in tasks}
    rng = np.random.default_rng(seed)

    per_task = {}
    for task in tasks:
        ids = sorted(r for r in shared if runs_a[0][r]["task"] == task)
        gold, predicted, classes = _encode(runs_a + runs_b, ids)
        per_task[task] = (gold, predicted[: len(runs_a)], predicted[len(runs_a) :], classes)

    def statistic(pick_a, pick_b, weights_by_task):
        scores = []
        for task in tasks:
            gold, a, b, classes = per_task[task]
            w = weights_by_task[task]
            f_a = np.mean([_macro_f1(gold, a[i], w, classes) for i in pick_a])
            f_b = np.mean([_macro_f1(gold, b[i], w, classes) for i in pick_b])
            scores.append(f_a - f_b)
        return float(np.mean(scores))

    ones = {task: np.ones(len(per_task[task][0])) for task in tasks}
    observed = statistic(range(len(runs_a)), range(len(runs_b)), ones)
    differences = []
    for _ in range(draws):
        weights = {
            task: rng.multinomial(len(g), np.full(len(g), 1 / len(g))).astype(float)
            for task, (g, *_rest) in per_task.items()
        }
        pick_a = rng.integers(0, len(runs_a), len(runs_a))
        pick_b = rng.integers(0, len(runs_b), len(runs_b))
        differences.append(statistic(pick_a, pick_b, weights))
    low, high = (float(v) for v in np.percentile(differences, [2.5, 97.5]))
    if low > 0:
        verdict = "keep the converted backbone"
    elif low > -TIE and high < TIE:
        verdict = "tie: keep the causal backbone"
    else:
        verdict = "not above zero: keep the causal backbone"
    return {
        "tasks": tasks,
        "rows_per_task": {t: int(len(per_task[t][0])) for t in tasks},
        "seeds": {"converted": len(runs_a), "causal": len(runs_b)},
        "difference": observed,
        "interval": [low, high],
        "verdict": verdict,
    }
