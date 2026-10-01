"""Remove the rows a decontamination report flagged from a training file.

    python -m data.decontam.apply --report results/step3/decontam/mide22-train.json \
        --rows data/built/typed/mide22/train.jsonl

Rule 2: nothing that overlaps an evaluation set enters training. The check
(data/decontam/cli.py) writes a report with one record per row; this rewrites
the rows file without the flagged ones, matched by row id, so it applies to
any file holding those rows, such as the split files promoted from a build.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Every row id any report has flagged. A report made on an already cleaned
# file no longer names the rows removed before it, so a converter rerun from
# its raw data would bring them back; this list keeps them out for good.
LEDGER = Path("results/step3/decontam/flagged_ids.txt")


def flagged_ids(report: Path) -> set[str]:
    data = json.loads(report.read_text(encoding="utf-8"))
    return {r["row_id"] for r in data["per_row"] if r["flagged"]}


def remember(ids: set[str], ledger: Path = LEDGER) -> set[str]:
    """Add ids to the ledger and return everything it holds."""
    known = set(ledger.read_text(encoding="utf-8").split()) if ledger.exists() else set()
    if ids - known:
        known |= ids
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text("".join(i + "\n" for i in sorted(known)), encoding="utf-8")
    return known


def apply(report: Path, rows: Path, ledger: Path = LEDGER) -> dict[str, int]:
    drop = remember(flagged_ids(report), ledger)
    lines = [x for x in rows.read_text(encoding="utf-8").splitlines() if x.strip()]
    kept = [x for x in lines if json.loads(x)["row_id"] not in drop]
    rows.write_text("".join(x + "\n" for x in kept), encoding="utf-8")
    return {"rows": len(lines), "removed": len(lines) - len(kept), "kept": len(kept)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data.decontam.apply")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rows", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    for rows in args.rows:
        print(rows, json.dumps(apply(args.report, rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
