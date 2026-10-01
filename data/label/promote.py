"""The build's rows, split into the files training reads (data/built/sss/).

    python -m data.label.promote --rows results/step3/build/rows.jsonl

Each row is validated again as a TrainingRow, so a row whose target or votes
do not add up never reaches training, checked against the current text
filters, and written to the file of its split.
A text may not appear both in train and in an evaluation file; the build
guarantees it and this refuses to write if it does not hold.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from data.label.texts import still_allowed
from schema.rows import TrainingRow, text_of

OUT = Path("data/built/sss")


def promote(rows_path: Path, out: Path = OUT, filter_texts: bool = True) -> dict[str, int]:
    by_split: dict[str, list[TrainingRow]] = defaultdict(list)
    dropped = 0
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = TrainingRow.model_validate_json(line)
            # The text filters may have grown since the rows were labelled.
            if filter_texts and not still_allowed({"text": text_of(row.state)}):
                dropped += 1
                continue
            by_split[row.split].append(row)
    train_texts = {text_of(r.state) for r in by_split.get("train", [])}
    shared = [r.row_id for s in ("validation", "heldout_task") for r in by_split.get(s, [])
              if text_of(r.state) in train_texts]  # fmt: skip
    if shared:
        raise ValueError(f"{len(shared)} evaluation rows share a text with train, e.g. {shared[0]}")
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in by_split.items():
        (out / f"{split}.jsonl").write_text(
            "".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8"
        )
    return {**{split: len(rows) for split, rows in sorted(by_split.items())},
            "dropped_by_current_filters": dropped}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data.label.promote")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    # Generated texts are not web pages: the web filters (gambling, markup,
    # language) would drop spam written to look like spam.
    parser.add_argument("--keep-all", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(promote(args.rows, args.out, filter_texts=not args.keep_all)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
