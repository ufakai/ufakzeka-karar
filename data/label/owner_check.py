"""The judges measured against the owner's labels on the generated families.

    python -m data.label.owner_check --labels <folder of label documents>

The owner labelled a stratified sample of the build's evaluation rows blind
(data/label/owner_sample.py, the label desk). Each answered row is compared with
judge A, judge C and the panel's target (their mean):

- accuracy: the source's most probable answer is the owner's;
- Brier against the owner's answer as a one-hot target;
- top-label ECE over 10 equal-width bins;
- 95 percent bootstrap intervals over rows, and the paired difference A minus C.

Rows the owner marked "Doesn't fit" or "Can't tell" are counted and left out,
as are rows the current text filters drop. Writes
results/step3/owner_labels/judge_check.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from data.label.texts import still_allowed

ITEMS = Path("results/step3/owner_labels/items.jsonl")
BUILD = Path("results/step3/build/rows.jsonl")
OUT = Path("results/step3/owner_labels/judge_check.json")
SOURCES = ("A", "C", "panel")


def load_labels(folder: Path) -> dict[str, dict]:
    """The label desk's documents, one JSON file per item, keyed by item id."""
    labels = {}
    for path in sorted(Path(folder).glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        labels[doc.get("item") or path.stem] = doc
    return labels


def top(distribution: dict[str, float]) -> str:
    return max(distribution, key=distribution.get)


def brier(distribution: dict[str, float], answer: str) -> float:
    return sum((p - float(k == answer)) ** 2 for k, p in distribution.items())


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            total += inside.mean() * abs(confidence[inside].mean() - correct[inside].mean())
    return float(total)


def interval(values: np.ndarray, draws: int, rng: np.random.Generator) -> list[float]:
    picks = rng.integers(0, len(values), (draws, len(values)))
    means = values[picks].mean(axis=1)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def compare(items: list[dict], build: dict[str, dict], labels: dict[str, dict],
            draws: int = 2000, seed: int = 0) -> dict:  # fmt: skip
    counts = defaultdict(int)
    kept = []
    for item in items:
        label = labels.get(item["id"])
        if label is None:
            counts["unlabelled"] += 1
        elif not still_allowed({"text": item["text"]}):
            counts["dropped_by_filters"] += 1
        elif label.get("flag") in ("no_fit", "cant_tell"):
            counts[label["flag"]] += 1
        elif label.get("answer") is None:
            counts["no_answer"] += 1
        else:
            row = build[item["id"]]
            judges = {j["judge"]: j["distribution"] for j in row["judges"]}
            kept.append({"cell": item["cell"], "answer": str(label["answer"]),
                         "A": judges.get("A"), "C": judges.get("C"),
                         "panel": row["target"]})  # fmt: skip
    counts["answered"] = len(kept)
    rng = np.random.default_rng(seed)
    result: dict = {"counts": dict(counts), "sources": {}}
    correct_by: dict[str, np.ndarray] = {}
    for source in SOURCES:
        rows = [r for r in kept if r[source] is not None]
        if not rows:
            continue
        correct = np.array([top(r[source]) == r["answer"] for r in rows], dtype=float)
        confidence = np.array([max(r[source].values()) for r in rows])
        briers = np.array([brier(r[source], r["answer"]) for r in rows])
        correct_by[source] = correct
        result["sources"][source] = {
            "rows": len(rows),
            "accuracy": float(correct.mean()),
            "accuracy_interval": interval(correct, draws, rng),
            "brier": float(briers.mean()),
            "brier_interval": interval(briers, draws, rng),
            "ece_10": ece(confidence, correct),
            "mean_confidence": float(confidence.mean()),
        }
    both = [r for r in kept if r["A"] is not None and r["C"] is not None]
    if both:
        diff = np.array([float(top(r["A"]) == r["answer"]) - float(top(r["C"]) == r["answer"])
                         for r in both])  # fmt: skip
        result["accuracy_a_minus_c"] = {"difference": float(diff.mean()),
                                        "interval": interval(diff, draws, rng)}  # fmt: skip
    per_cell: dict[str, list[float]] = defaultdict(list)
    for r in kept:
        per_cell[r["cell"]].append(float(top(r["panel"]) == r["answer"]))
    result["panel_accuracy_per_cell"] = {
        cell: {"rows": len(v), "accuracy": round(sum(v) / len(v), 3)}
        for cell, v in sorted(per_cell.items())
    }  # fmt: skip
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data.label.owner_check")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    items = [json.loads(line) for line in ITEMS.read_text(encoding="utf-8").splitlines()
             if line.strip()]  # fmt: skip
    wanted = {i["id"] for i in items}
    build = {}
    for line in BUILD.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["row_id"] in wanted:
                build[row["row_id"]] = row
    result = compare(items, build, load_labels(args.labels))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "panel_accuracy_per_cell"},
                     indent=2))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
