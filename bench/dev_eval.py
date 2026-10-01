"""A model's predictions scored on HakemBench-dev v0, beside the judges' panel.

    python -m bench.dev_eval --runs r1-main-s1,r1-main-s2,r1-main-s3

Dev items are validation and held-out rows of the build, so each run's
predictions on them already exist (predictions_step*.jsonl at the last point,
and predictions_heldout.jsonl). The seeds' probabilities are averaged per item;
accuracy, Brier and top-label ECE against the owner's answer are reported with
95 percent bootstrap intervals, and the paired difference from the panel's
target on the same items. Writes results/step6/dev_eval/<first run>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from bench.dev import PRIVATE
from data.label.owner_check import BUILD, brier, ece, top
from model.head.accept import last_step
from model.head.compare import fetch

OUT = Path("results/step6/dev_eval")


def macro_f1_over_tasks(rows: list[dict], source: dict[str, dict]) -> float:
    """Mean over tasks of macro F1 against the owner's answers (the statistic)."""
    by_task: dict[str, list[tuple[str, str]]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append((top(r["target"]), top(source[r["id"]])))
    scores = []
    for pairs in by_task.values():
        labels = {x for pair in pairs for x in pair}
        f1 = []
        for k in labels:
            tp = sum(g == k and p == k for g, p in pairs)
            n = sum(g == k for g, _ in pairs) + sum(p == k for _, p in pairs)
            f1.append(2 * tp / n if n else 0.0)
        scores.append(sum(f1) / len(f1))
    return sum(scores) / len(scores)


def score(dev: list[dict], model: dict[str, dict], panel: dict[str, dict], draws: int = 2000,
          seed: int = 0) -> dict:  # fmt: skip
    rows = [r for r in dev if r["id"] in model]
    rng = np.random.default_rng(seed)
    out: dict = {"rows": len(rows), "missing": len(dev) - len(rows)}
    per: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, source in (("model", model), ("panel", panel)):
        answer = [top(r["target"]) for r in rows]
        correct = np.array([top(source[r["id"]]) == a for r, a in zip(rows, answer, strict=True)],
                           dtype=float)  # fmt: skip
        briers = np.array([brier(source[r["id"]], a) for r, a in zip(rows, answer, strict=True)])
        confidence = np.array([max(source[r["id"]].values()) for r in rows])
        per[name] = (correct, briers)
        picks = rng.integers(0, len(rows), (draws, len(rows)))
        bounds = np.percentile(correct[picks].mean(1), [2.5, 97.5]).tolist()
        out[name] = {"accuracy": float(correct.mean()), "accuracy_interval": bounds,
                     "macro_f1_mean_over_tasks": macro_f1_over_tasks(rows, source),
                     "brier": float(briers.mean()), "ece_10": ece(confidence, correct),
                     "mean_confidence": float(confidence.mean())}  # fmt: skip
    picks = rng.integers(0, len(rows), (draws, len(rows)))
    for i, metric in enumerate(("accuracy", "brier")):
        diff = per["model"][i] - per["panel"][i]
        out[f"model_minus_panel_{metric}"] = {
            "difference": float(diff.mean()),
            "interval": np.percentile(diff[picks].mean(1), [2.5, 97.5]).tolist()}  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.dev_eval")
    parser.add_argument("--runs", required=True)
    # The last validation point by default, read from each run's summary.
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--cache", type=Path, default=Path(".cache/karar/predictions"))
    args = parser.parse_args(argv)
    dev = [json.loads(x) for x in PRIVATE.read_text(encoding="utf-8").splitlines() if x.strip()]
    ids = {r["id"] for r in dev}
    sums: dict[str, list[dict]] = {}
    for run in args.runs.split(","):
        step = f"predictions_step{args.step}.jsonl" if args.step else last_step(run, args.cache)
        for name in (step, "predictions_heldout.jsonl"):
            for line in fetch(run, args.cache, name).read_text(encoding="utf-8").splitlines():
                record = json.loads(line) if line.strip() else None
                if record and record["row_id"] in ids:
                    sums.setdefault(record["row_id"], []).append(
                        dict(zip(record["keys"], record["probs"], strict=True)))  # fmt: skip
    model = {i: {k: float(np.mean([p[k] for p in ps])) for k in ps[0]} for i, ps in sums.items()}
    panel = {}
    for line in BUILD.read_text(encoding="utf-8").splitlines():
        row = json.loads(line) if line.strip() else None
        if row and row["row_id"] in ids:
            panel[row["row_id"]] = row["target"]
    result = {"runs": args.runs.split(","), "dev": str(PRIVATE), **score(dev, model, panel)}
    OUT.mkdir(parents=True, exist_ok=True)
    runs = args.runs.split(",")
    # One file per set of runs: a seed alone and the seeds together must not
    # overwrite each other.
    name = runs[0] if len(runs) == 1 else runs[0].rsplit("-s", 1)[0] + "-seeds"
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
