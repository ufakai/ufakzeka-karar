"""The public board recomputed from public files only.

    python -m release.public_board export          # archive only: writes the public inputs
    python -m release.public_board args            # the bench.board arguments, one per line
    python -m release.public_board check NEW.json  # NEW against results/step9/board_public.json

results/step9/board_public.json was computed from each model's part A rows (public files)
and its part B rows, which sat in the private archive beside the answers on the 574 left-out
items. `export` writes what the board read of them into public files:

- results/step9/runs/<model>-open-partb.jsonl: each board model's part B rows on open items,
  the lines copied unchanged, rows on left-out items dropped;
- results/step9/runs/surface-baseline-open.jsonl: the surface baseline's rows on open items;
- results/step9/owner_check_open_answers.json: the founder's own blind answers on the open
  support questions of the owner check, the same answers the board scored.

`args` gives the board the same rows in the same order, so the models come out in the same
order, and `just board-public` runs it and then `check`, which compares every number of the
two boards. Only the run's own record differs (when and at which commit it ran, and the
input paths and hashes it lists). The board draws are seeded, so the numbers are the same
on any machine up to floating-point rounding: a machine of another architecture can move
the last digits of a float. `check` reports exact identity and the largest relative
difference, and passes when every difference is a float within 1e-12 of it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOARD = Path("results/step9/board_public.json")
RUNS = Path("results/step9/runs")
SPLITS = Path("bench/hakembench/v1.0/open/splits.json")
PROVENANCE = Path("bench/hakembench/v1.0/open/provenance.jsonl")
WEIGHTS = Path("bench/board_weights_step9.json")
OWNER = Path("results/step9/owner_check_open_answers.json")
# The board's row files in its own order: (part A file, part B file) per model, the surface
# baseline's one file of both parts third, as results/step9/board_public.json lists them.
MODELS = (
    "ufakzeka-karar-r5",
    "surface-baseline",
    "jev-1.13",
    "decider-2b",
    "kev-4b",
    "kev-9b",
    "simple-jev-qwen3.5-4b",
    "laya",
    "laya-full",
    "laya-multilingual",
    "laya-multilingual-full",
    "open-jev-deberta-v3-large",
    "gpt-5.6-sol",
    "gemini-3.8-flash",
    "glm-5.3",
    "deepseek-v4-pro-0813",
)
SURFACE = "surface-baseline"
# The run's own record, which a recomputation changes by design.
RECORD = ("created_utc", "git_commit", "inputs")
# Floating-point rounding across machines, far below any digit the board is read at.
REL_TOL = 1e-12


def rows_files(model: str) -> list[Path]:
    if model == SURFACE:
        return [RUNS / f"{SURFACE}-open.jsonl"]
    return [RUNS / f"{model}-public.jsonl", RUNS / f"{model}-open-partb.jsonl"]


def board_args(out: Path) -> list[str]:
    rows = [str(p) for m in MODELS for p in rows_files(m)]
    return [
        "--rows",
        *rows,
        "--splits",
        str(SPLITS),
        "--weights",
        str(WEIGHTS),
        "--provenance",
        str(PROVENANCE),
        "--owner-answers",
        str(OWNER),
        "--only-listed",
        "--out",
        str(out),
    ]


def _diff(a, b, where: str, out: list, worst: list) -> None:
    """Every difference between two parsed JSON values; float differences are measured."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{where}/{key}: only in {'new' if key in a else 'committed'}")
            else:
                _diff(a[key], b[key], f"{where}/{key}", out, worst)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{where}: {len(a)} items against {len(b)}")
        for n, (x, y) in enumerate(zip(a, b, strict=False)):
            _diff(x, y, f"{where}[{n}]", out, worst)
    elif isinstance(a, float) and isinstance(b, float):
        if a != b and not (math.isnan(a) and math.isnan(b)):
            worst.append((abs(a - b), abs(a - b) / max(abs(b), 1e-300)))
            out.append(f"{where}: {a!r} against {b!r}")
    elif a != b:
        out.append(f"{where}: {a!r} against {b!r}")


def check(new: Path, committed: Path = BOARD, rel_tol: float = REL_TOL) -> dict:
    """Every number of two boards compared. Only floats may differ, and only by rounding: a
    machine of another architecture sums in another order (a Mac against the Linux container
    the committed board ran on), which moves the last digits and nothing else."""
    a = json.loads(new.read_text(encoding="utf-8"))
    b = json.loads(committed.read_text(encoding="utf-8"))
    out: list[str] = []
    worst: list[float] = []
    for key in RECORD:
        a.pop(key, None)
        b.pop(key, None)
    _diff(a, b, "", out, worst)
    floats_only = len(worst) == len(out)
    largest = max((r for _, r in worst), default=0.0)
    return {
        "identical": not out,
        "differences": len(out),
        "all_differences_are_floats": floats_only,
        "largest_float_difference": max((d for d, _ in worst), default=0.0),
        "largest_relative_difference": largest,
        f"same_within_relative_{rel_tol:g}": floats_only and largest <= rel_tol,
        "first": out[:20],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="release.public_board")
    sub = parser.add_subparsers(dest="command", required=True)
    args_cmd = sub.add_parser("args")
    args_cmd.add_argument("--out", type=Path, default=Path("results/step9/board_public_check.json"))
    check_cmd = sub.add_parser("check")
    check_cmd.add_argument("new", type=Path)
    check_cmd.add_argument("--committed", type=Path, default=BOARD)
    args = parser.parse_args(argv)
    if args.command == "args":
        print("\n".join(board_args(args.out)))
        return 0
    if args.command == "check":
        result = check(args.new, args.committed)
        print(json.dumps(result, indent=1))
        return 0 if result["identical"] or result[f"same_within_relative_{REL_TOL:g}"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
