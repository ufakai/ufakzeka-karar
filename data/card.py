"""The counts behind the dataset card (docs/DATA_CARD.md), from the built files.

    python -m data.card --out results/step3/data_card.json

Reads every training file under data/built/ and reports, per source and
split, the rows, the question types, the label kind and, for judge-labelled
rows, how often the two judges agreed on the top answer and how many rows
were mined; for r4's folder, which holds several tasks, the rows of each
task too. The card quotes this file and nothing else (rule 3).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SOURCES = {path.name: path for path in sorted(Path("data/built/typed").iterdir()) if path.is_dir()}
SOURCES["sss"] = Path("data/built/sss")
SOURCES["synth"] = Path("data/built/synth")
SOURCES["r4"] = Path("data/built/r4")


def top(distribution: dict[str, float]) -> str:
    return max(distribution, key=distribution.get)


def summarise(folder: Path) -> dict:
    out = {}
    for path in sorted(folder.glob("*.jsonl")) if folder.exists() else []:
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
        judged = [r for r in rows if r["judges"]]
        families = (
            Counter(r["task"].rsplit("-", 1)[0] for r in rows) if folder.name == "sss" else {}
        )
        out[path.stem] = {
            "rows": len(rows),
            "types": dict(Counter(r["question"]["type"] for r in rows)),
            "label_kind": dict(Counter(r["label_kind"] for r in rows)),
            "tasks": len({r["task"] for r in rows}),
            "distinct_texts": len({json.dumps(r["state"], ensure_ascii=False) for r in rows}),
            "judges_agree_on_top": (
                round(
                    sum(len({top(v["distribution"]) for v in r["judges"]}) == 1 for r in judged)
                    / len(judged),
                    4,
                )
                if judged
                else None
            ),  # fmt: skip
            "mined": sum(r["recipe"].endswith("-mined") for r in rows),
            "family_type": dict(sorted(families.items())),
        }
        if folder.name == "r4":
            out[path.stem]["task_rows"] = dict(sorted(Counter(r["task"] for r in rows).items()))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data.card")
    parser.add_argument("--out", type=Path, default=Path("results/step3/data_card.json"))
    args = parser.parse_args(argv)
    report = defaultdict(dict)
    for name, folder in SOURCES.items():
        report[name] = summarise(folder)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for name, splits in report.items():
        print(name, {split: v["rows"] for split, v in splits.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
