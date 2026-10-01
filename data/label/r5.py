"""r5 data: the conversation prompt-injection set, checked for overlap and built.

    python -m data.label.r5 overlap   # writes results/step9/r5/overlap.json
    python -m data.label.r5 build     # data/built/typed/prompt_injection_conv/, minus the drops

r4 learned that natural product register means benign, because its 800 benign
look-alikes were the only training text in that register. The
conversation set brings attacks in the same register. Its train split trains as
prompt_injection-conv; its validation and test splits are the guardrail dev set.

`overlap` drops, by the rules in both directions:
- every text, either split, that touches a HakemBench v1.0 item of either half, a
  probe or a HakemBench-dev text (data.label.r4.touching);
- every train text that touches a dev-set text, so the dev set is unseen;
- every dev-set text that touches an existing training text (the training files the
  round 1 mix reads), so no dev text was trained on under another task.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from bench.hakembench.common import overlapping
from data.label.r4 import touching
from data.typed.guardrails import CONV_OUT, build_prompt_injection_conv, conv_candidates, dedupe
from model.head.mixing import train_files
from schema.rows import text_of

REPORT = Path("results/step9/r5/overlap.json")
BUILT = Path("data/built")
# The guardrail track's training files, also checked in the reverse direction
# (a short training text inside a longer dev text).
GUARD_TRAIN = ("typed/prompt_injection/train.jsonl", "r4/train.jsonl")


def pair_touches(refs: dict[str, str], others: dict[str, str]) -> dict[str, list[str]]:
    """Keys of `others` that touch a text of `refs`, by the dropping rules, both directions."""
    out: dict[str, set[str]] = {}
    forward = overlapping(others, ((f"ref:{k}", v) for k, v in refs.items()))
    for key, entry in forward.items():
        if entry["dropped"]:
            out.setdefault(key, set()).update(entry["rules"])
    backward = overlapping(refs, ((k, v) for k, v in others.items()))
    for entry in backward.values():
        if entry["dropped"]:
            for rule, where in entry["rows"].items():
                if where in others:
                    out.setdefault(where, set()).add(f"{rule}-back")
    return {k: sorted(v) for k, v in out.items()}


def existing_training(root: Path = BUILT) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Every training text the mix reads, and the guardrail track's by row id."""
    every, guard = [], {}
    for path in train_files(root):
        if path.parent.name == CONV_OUT:
            continue
        rel = path.relative_to(root).as_posix()
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            text = text_of(row["state"])
            every.append((f"{rel}:{row['row_id']}", text))
            if rel in GUARD_TRAIN and row["track"] == "guvenlik":
                guard[f"{rel}:{row['row_id']}"] = text
    return every, guard


def overlap(report: Path = REPORT, root: Path = BUILT) -> dict:
    candidates, hashes = conv_candidates()
    kept, deduped = dedupe(candidates)
    texts = {c.source_id: c.text for c in kept}

    hakem = touching(texts)
    left = {c.source_id: c for c in kept if c.source_id not in hakem}
    train = {k: c.text for k, c in left.items() if c.split == "train"}
    dev = {k: c.text for k, c in left.items() if c.split == "validation"}

    # A train text near a dev text leaves train; the dev set stays whole here.
    train_vs_dev = pair_touches(dev, train)
    # A dev text near an existing training text leaves the dev set.
    every, guard = existing_training(root)
    forward = overlapping(dev, every)
    dev_vs_existing: dict[str, set[str]] = {}
    for key, entry in forward.items():
        if entry["dropped"]:
            dev_vs_existing.setdefault(key, set()).update(entry["rules"])
    backward = overlapping(guard, ((k, v) for k, v in dev.items()))
    for entry in backward.values():
        if entry["dropped"]:
            for rule, where in entry["rows"].items():
                if where in dev:
                    dev_vs_existing.setdefault(where, set()).add(f"{rule}-back")

    def labels(ids) -> dict[str, int]:
        by_id = {c.source_id: c for c in kept}
        return dict(Counter(f"{by_id[i].source_id.split(':')[1]}:{by_id[i].label}" for i in ids))

    drops = sorted({*hakem, *train_vs_dev, *dev_vs_existing})
    result = {
        "files_sha256": hashes,
        "candidates": len(candidates),
        "after_dedupe": len(kept),
        "dedupe": dict(deduped),
        "touch_hakembench": {"count": len(hakem), "by_split_label": labels(hakem), "ids": hakem},
        "train_touches_dev": {
            "count": len(train_vs_dev),
            "by_split_label": labels(train_vs_dev),
            "ids": train_vs_dev,
        },  # fmt: skip
        "dev_touches_training": {
            "count": len(dev_vs_existing),
            "by_split_label": labels(dev_vs_existing),
            "ids": {k: sorted(v) for k, v in sorted(dev_vs_existing.items())},
        },
        "existing_training_texts": len(every),
        "drops": drops,
        "kept_by_split_label": labels(k for k in texts if k not in set(drops)),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def build(report: Path = REPORT) -> dict:
    drops = json.loads(report.read_text(encoding="utf-8"))["drops"]
    return build_prompt_injection_conv(drops)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data.label.r5")
    parser.add_argument("command", choices=("overlap", "build"))
    args = parser.parse_args(argv)
    if args.command == "overlap":
        result = overlap()
        for key in ("touch_hakembench", "train_touches_dev", "dev_touches_training"):
            print(key, result[key]["count"], result[key]["by_split_label"])
        print("kept", result["kept_by_split_label"])
    else:
        print(json.dumps(build(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
