"""Rows per task in a run's training mix, recomputed from the local training files.

    uv run python -m model.head.mix_counts --run r5b-base-s1

Repeats what model/head/round1.py does before training: mix(train_files(data/built),
MIX_CAP, seed=1), then take_fractions with the run's task_fraction and the strata under
data/built/strata. The mix itself stays on the GPU volume; this writes only the counts.
It refuses unless every training file's sha256 equals the run record's data_hashes, every
task fraction's row count before the cut equals the record's, and the total equals the
record's mixed_rows, so the counts are the run's own. A cross-fit fold run is refused: its
mix also depends on the fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from model.head.mixing import MIX_CAP, load_strata, mix, take_fractions, train_files

DATA = Path("data/built")
RECORDS = Path("results/step4/round1")
OUT = Path("results/step9/r5/mix_counts.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(record_path: Path, data: Path = DATA) -> dict:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    spec = record["spec"]
    if spec.get("fold") is not None:
        raise SystemExit(f"{spec['name']} is a fold run; its mix depends on the fold")
    files = train_files(data)
    names = [f.relative_to(data).as_posix() for f in files]
    expected = {k: v for k, v in record["data_hashes"].items() if k.endswith("train.jsonl")}
    if sorted(names) != sorted(expected):
        raise SystemExit(f"training files differ from the record: {names} against {expected}")
    for name, path in zip(names, files, strict=True):
        if sha256(path) != expected[name]:
            raise SystemExit(f"{name} is not the file the run trained on")
    task_file = {}
    for name, path in zip(names, files, strict=True):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                task_file.setdefault(json.loads(line)["task"], name)
    lines = mix(files, MIX_CAP, seed=1)
    fractions = spec.get("task_fraction") or {}
    before = {t: sum(json.loads(x)["task"] == t for x in lines) for t in fractions}
    for task, n in before.items():
        if n != record["task_fraction_rows"][task]["before"]:
            raise SystemExit(f"{task}: {n} rows before the cut, the record says otherwise")
    if fractions:
        lines = take_fractions(lines, fractions, load_strata(data / "strata"))
    if len(lines) != record["mixed_rows"]:
        raise SystemExit(f"{len(lines)} mixed rows against the record's {record['mixed_rows']}")
    per_task = Counter(json.loads(x)["task"] for x in lines)
    per_file = Counter()
    for task, n in per_task.items():
        per_file[task_file[task]] += n
    return {
        "script": "model/head/mix_counts.py",
        "run": spec["name"],
        "record": {"path": record_path.as_posix(), "sha256": sha256(record_path)},
        "mix_cap": MIX_CAP,
        "task_fraction": fractions,
        "checked": {
            "training_files_match_data_hashes": len(files),
            "rows_before_fraction": before,
            "mixed_rows": len(lines),
        },
        "mixed_rows": len(lines),
        "tasks": len(per_task),
        "rows_per_task": dict(sorted(per_task.items())),
        "rows_per_file": dict(sorted(per_file.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="r5b-base-s1")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = counts(RECORDS / f"{args.run}.json", args.data)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {out['mixed_rows']} rows over {out['tasks']} tasks")


if __name__ == "__main__":
    main()
