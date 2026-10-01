"""From run folders to committed results rows.

Two jobs, both offline and both cheap, so they never touch a GPU:

  chosen_rate   phase one: read a backbone's sweep runs on one dataset and
                pick the learning rate, by the same refinement rule that
                picks a checkpoint.
  row_for_run   phase two: pick the evaluation point from validation, then
                report the test metrics of that one point.

The order matters and is enforced by the module boundary: calib/selection.py
reads validation files only, and the single line below is the one place a
test file is opened, after the point has already been chosen.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from calib.metrics import report
from calib.selection import Selection, select_point
from model.instrument.plan import LEARNING_RATES, RunSpec, sweep_specs, treatment_mismatch

# Facts about the run that a reader needs next to the numbers.
RUN_FACTS = (
    "backbone",
    "revision",
    "dataset",
    "dataset_source_revision",
    "max_length",
    "truncated_share",
    "parameters_total",
    "parameters_non_embedding",
    "precision",
    "gpu_name",
    "attn_implementation",
    "git_commit",
    "total_steps",
    "eval_steps",
    "wall_seconds",
    "diverged_at_step",
)


def chosen_rate(runs_root: Path, dataset: str, backbone: str, attention: str = "causal") -> dict:
    """Pick the learning rate for one backbone on one dataset.

    Returns the rate, every rate's refinement score, and whether the winner
    sits at an end of the grid, which is the cue to extend it.
    """
    folders = {
        spec.learning_rate: runs_root / spec.path
        for spec in sweep_specs([dataset], attention=attention)
        if spec.backbone == backbone and (runs_root / spec.path / "run.json").is_file()
    }
    missing = [f"{rate:.0e}" for rate in sorted(set(LEARNING_RATES) - set(folders))]
    if missing:
        raise FileNotFoundError(f"{dataset}/{backbone}: no sweep run for rates {missing}")

    selections: dict[float, Selection] = {}
    unusable: dict[str, str] = {}
    by_rate = {
        spec.learning_rate: spec
        for spec in sweep_specs([dataset], attention=attention)
        if spec.backbone == backbone
    }
    for rate, folder in sorted(folders.items()):
        recorded = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        problem = treatment_mismatch(recorded, by_rate[rate])
        if problem:
            # Not "unusable", which would let selection carry on with the rest:
            # a sweep that was given the wrong treatment selects nothing.
            raise ValueError(problem)
        try:
            selections[rate] = select_point(folder)
        except (FileNotFoundError, ValueError) as reason:
            # A rate high enough to diverge before the first evaluation point
            # leaves nothing to score. That is a result about the rate, not a
            # reason to abandon the whole backbone.
            unusable[f"{rate:.0e}"] = str(reason).split("\n")[0]
    if not selections:
        raise RuntimeError(f"{dataset}/{backbone}: every learning rate was unusable: {unusable}")

    best = min(selections, key=lambda rate: selections[rate].score)
    usable = sorted(selections)
    return {
        "dataset": dataset,
        "backbone": backbone,
        "attention": attention,
        "learning_rate": best,
        # An edge of the rates that could be scored, which is what a wider grid
        # would have to reach past.
        "at_grid_edge": best in (usable[0], usable[-1])
        and best in (min(LEARNING_RATES), max(LEARNING_RATES)),
        "scores": {f"{rate:.0e}": round(sel.score, 6) for rate, sel in selections.items()},
        "unusable": unusable,
        "estimator": selections[best].estimator,
        "selected_step": selections[best].step,
    }


def _facts(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return {key: meta.get(key) for key in RUN_FACTS}


def row_for_run(run_dir: Path, spec: RunSpec) -> dict:
    """The results row for one final run: one selected point, reported on test."""
    recorded = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    problem = treatment_mismatch(recorded, spec)
    if problem:
        # The row would carry the spec's label over another treatment's numbers.
        raise ValueError(problem)
    selection: Selection = select_point(run_dir)
    validation = np.load(run_dir / f"logits_validation_step{selection.step}.npy")
    # The only place a test file is opened, and only for the chosen point.
    test = np.load(run_dir / f"logits_test_step{selection.step}.npy")
    measured = report(
        test,
        np.load(run_dir / "labels_test.npy"),
        validation,
        np.load(run_dir / "labels_validation.npy"),
    )
    return {
        "dataset": spec.dataset,
        "backbone": spec.backbone,
        "pooling": spec.pooling,
        "attention": spec.attention,
        "learning_rate": spec.learning_rate,
        "seed": spec.seed,
        "run_path": spec.path,
        "selection": {
            "step": selection.step,
            "score": round(selection.score, 6),
            "estimator": selection.estimator,
            # What the other rules would have picked, so this one can be audited.
            "audit": selection.audit,
            "points": len(selection.scores),
        },
        **measured,
        "run": _facts(run_dir),
    }


def rows_for_specs(runs_root: Path, specs: list[RunSpec]) -> tuple[list[dict], list[str]]:
    """Rows for every finished run in `specs`, and the paths that are not there yet."""
    rows, absent = [], []
    for spec in specs:
        run_dir = runs_root / spec.path
        if not (run_dir / "run.json").is_file():
            absent.append(spec.path)
            continue
        rows.append(row_for_run(run_dir, spec))
    return rows, absent
