"""The rows the owner labels to measure the judges on the generated families.

    python -m data.label.owner_sample --out results/step3/owner_labels/items.jsonl

About 300 rows from the evaluation splits of the build (validation and held
out; never train, so the labels can also seed HakemBench-dev), spread evenly
over the family and type cells and, inside a cell, over templates. What the
page shows is the text, the question and its options: the judges' votes and
the target stay out, so the owner labels blind.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from schema.rows import TrainingRow, outcomes, text_of

FILES = ("data/built/sss/validation.jsonl", "data/built/sss/heldout_task.jsonl")


def options_of(row: TrainingRow) -> list[dict]:
    q = row.question
    if q.type == "choice":
        return [{"key": k, "label": k, "detail": q.criteria[k] or ""} for k in outcomes(q)]
    if q.type == "noul":
        c = q.criteria
        return [
            {"key": "true", "label": "Evet", "detail": (c.true if c else "") or ""},
            {"key": "false", "label": "Hayır", "detail": (c.false if c else "") or ""},
        ]
    return [{"key": str(i), "label": f"Düzey {i}", "detail": level}
            for i, level in enumerate(q.criteria)]  # fmt: skip


def sample(total: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    cells: dict[str, dict[str, list[TrainingRow]]] = defaultdict(lambda: defaultdict(list))
    for path in FILES:
        for line in Path(path).read_text("utf-8").splitlines():
            if line.strip():
                row = TrainingRow.model_validate_json(line)
                family, kind, tid = row.task.rsplit("-", 2)
                cells[f"{family}-{kind}"][tid].append(row)
    per_cell = total // len(cells)
    picked = []
    for cell, templates in sorted(cells.items()):
        pools = [rng.sample(rows, len(rows)) for _, rows in sorted(templates.items())]
        rows, i = [], 0
        while len(rows) < per_cell and any(pools):
            pool = pools[i % len(pools)]
            if pool:
                rows.append(pool.pop())
            i += 1
        picked += [(cell, r) for r in rows]
    rng.shuffle(picked)
    return [{"id": r.row_id, "order": n, "cell": cell, "task": r.task, "split": r.split,
             "type": r.question.type, "text": text_of(r.state),
             "question": text_of(r.question.instructions), "options": options_of(r)}
            for n, (cell, r) in enumerate(picked)]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data.label.owner_sample")
    parser.add_argument("--out", type=Path, default=Path("results/step3/owner_labels/items.jsonl"))
    parser.add_argument("--total", type=int, default=300)
    args = parser.parse_args(argv)
    items = sample(args.total)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), "utf-8")
    cells = defaultdict(int)
    for i in items:
        cells[i["cell"]] += 1
    print(len(items), "items,", len(cells), "cells:", dict(sorted(cells.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
