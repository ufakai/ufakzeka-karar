"""Which files round 1 trains and validates on, and how the tasks are mixed.

PLAN.md: examples-proportional mixing with a cap. Every task keeps its rows up
to `cap`, drawn with a fixed seed, so the mix is the same in every run and a
large converted set cannot drown the small generated tasks.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

MIX_CAP = 3000


def train_files(root: Path) -> list[Path]:
    return sorted([*root.glob("typed/*/train.jsonl"), *root.glob("sss/train.jsonl"),
                   *root.glob("synth/train.jsonl"),
                   *root.glob("r4/train.jsonl")])  # fmt: skip


def validation_files(root: Path) -> list[Path]:
    return sorted([*root.glob("typed/*/validation.jsonl"), *root.glob("sss/validation.jsonl"),
                   *root.glob("synth/validation.jsonl"),
                   *root.glob("r4/validation.jsonl")])  # fmt: skip


def mix(files: list[Path], cap: int = MIX_CAP, seed: int = 1) -> list[str]:
    """Every task's rows up to `cap`, drawn with a fixed seed; tasks in name order."""
    by_task = defaultdict(list)
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                by_task[json.loads(line)["task"]].append(line)
    rng = random.Random(seed)
    out = []
    for task in sorted(by_task):
        lines = by_task[task]
        out += rng.sample(lines, cap) if len(lines) > cap else lines
    return out


# Rows of a capped task the mix never drew, kept for round 2's memorisation
# check: scored with the trained rows, never trained on.
UNSEEN_PER_TASK = 2000


def unseen(files: list[Path], cap: int = MIX_CAP, seed: int = 1,
           per_task: int = UNSEEN_PER_TASK) -> list[str]:  # fmt: skip
    """Up to `per_task` rows of every capped task that `mix` leaves out."""
    drawn = set(mix(files, cap, seed))
    by_task = defaultdict(list)
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and line not in drawn:
                by_task[json.loads(line)["task"]].append(line)
    rng = random.Random(seed + 1)
    out = []
    for task in sorted(by_task):
        lines = by_task[task]
        out += rng.sample(lines, per_task) if len(lines) > per_task else lines
    return out


FOLDS = 5


def fold_of(line: str, folds: int = FOLDS) -> int:
    """A row's cross-fit fold, from a hash of its text, so every template of one
    text lands in the same fold."""
    import hashlib

    state = json.loads(line)["state"]
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % folds


# Task-level filters for a run: {"task": fraction} keeps that share of the
# task's mixed rows; 0.0 drops the task. The cut is deterministic and stratified.
FRACTION_SEED = 133
# Where a task's strata come from: records with the row's text and the fields to
# stratify by. r4's benign look-alikes carry the writer's kind and product context.
STRATA_SOURCES = {
    "prompt_injection-r4": (Path("results/private/step9/r4/labels/guvenlik/kept.json"),
                            ("kind", "context")),
}  # fmt: skip


def strata_from_records(path: Path, fields: tuple[str, ...]) -> dict[str, str]:
    """A row text to its stratum label, such as "K3|iş takip ve proje yönetimi botu"."""
    records = json.loads(path.read_text(encoding="utf-8"))
    return {r["text"]: "|".join(str(r[f]) for f in fields) for r in records}


def load_strata(folder: Path) -> dict[str, dict[str, str]]:
    """Every <task>.json under `folder`, as task to (text to stratum)."""
    if not folder.is_dir():
        return {}
    paths = sorted(folder.glob("*.json"))
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in paths}


def _state_text(row: dict) -> str:
    state = row["state"]
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def take_fractions(lines: list[str], fractions: dict[str, float],
                   strata: dict[str, dict[str, str]] | None = None,
                   seed: int = FRACTION_SEED) -> list[str]:  # fmt: skip
    """The mix with each named task cut to its fraction; other tasks and the order untouched.

    Inside a task the rows are ordered by stratum (rows without one share the empty
    stratum), then by a seeded hash of the row id, and systematic sampling keeps
    floor(n * fraction) of them: every stratum keeps its share to within one row, and
    with a nested label such as kind|context both levels stay balanced.
    """
    import hashlib

    for task, fraction in fractions.items():
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"{task}: fraction {fraction} is outside [0, 1]")
    strata = strata or {}
    by_task: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for line in lines:
        row = json.loads(line)
        if row["task"] in fractions:
            label = strata.get(row["task"], {}).get(_state_text(row), "")
            key = hashlib.sha256(f"{seed}:{row['row_id']}".encode()).hexdigest()
            by_task[row["task"]].append((label, key, line))
    keep: set[str] = set()
    for task, rows in by_task.items():
        rows.sort()
        fraction = fractions[task]
        keep.update(line for i, (_, _, line) in enumerate(rows)
                    if int((i + 1) * fraction) > int(i * fraction))  # fmt: skip
    return [x for x in lines if json.loads(x)["task"] not in fractions or x in keep]
