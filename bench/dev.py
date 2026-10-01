"""HakemBench-dev v0: the owner's labelled support items, frozen.

    python -m bench.dev --labels results/private/step3/owner_labels/labels

The rows the owner answered on the label desk (not "Doesn't fit", not "Can't
tell", not dropped by the current text filters) become the development set for
the support track, with the owner's answer as a one-hot target. The items
themselves stay under results/private/ (out of the public build); what is
committed is a manifest: the file's hash, counts per cell, and the hash of
every text, so any later change to the set is visible.

The freeze also checks every training file against the dev texts by the
decontamination's rules (a whole text, more than half a row's tokens in shared
8-grams, or a dev text inside a row). Matching training rows are written to a
decontamination report and removed through its ledger (data/decontam/apply.py),
so a rebuild keeps them out; the freeze then checks again and refuses to write
if any match remains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from data.decontam.apply import apply
from data.decontam.ngrams import NgramIndex, gram_hashes, tokens
from data.label.owner_check import ITEMS, load_labels
from data.label.texts import still_allowed
from model.head.mixing import train_files

PRIVATE = Path("results/private/hakembench/dev_v0.jsonl")
MANIFEST = Path("results/step6/hakembench_dev_v0.json")
REPORT = Path("results/step3/decontam/hakembench-dev.json")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dev_rows(items: list[dict], labels: dict[str, dict]) -> list[dict]:
    out = []
    for item in items:
        label = labels.get(item["id"])
        if not label or label.get("flag") in ("no_fit", "cant_tell"):
            continue
        if label.get("answer") is None or not still_allowed({"text": item["text"]}):
            continue
        keys = [o["key"] for o in item["options"]]
        answer = str(label["answer"])
        out.append({"id": item["id"], "cell": item["cell"], "task": item["task"],
                    "type": item["type"], "text": item["text"], "question": item["question"],
                    "options": item["options"],
                    "target": {k: float(k == answer) for k in keys}})  # fmt: skip
    return out


def training_overlap(rows: list[dict], files: list[Path]) -> list[dict]:
    """Training rows that carry a dev text, by the decontamination's rules.

    Each match names the dev rows it involves under "refs"; for the n-gram rule
    that is every dev row sharing an 8-gram with the training row.
    """
    index = NgramIndex.build((f"dev:{r['id']}", r["text"]) for r in rows)
    wholes: dict[str, list[str]] = {}
    for r in rows:
        wholes.setdefault(text_hash(r["text"]), []).append(r["id"])
    found = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            state = row["state"]
            text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            where = {"file": str(path), "row_id": row["row_id"]}
            if (same := wholes.get(text_hash(text))) is not None:
                found.append({**where, "rule": "whole", "refs": sorted(same)})
                continue
            fraction, refs = index.overlap(text)
            if fraction > 0.5:
                found.append({**where, "rule": "ngram", "refs": _ids(refs)})
                continue
            if not refs:
                continue
            grams = set(gram_hashes(tokens(text), index.n))
            if covered := index.covered_references(grams, refs):
                found.append({**where, "rule": "covers", "refs": _ids(covered)})
    return found


def _ids(refs: list[str]) -> list[str]:
    return [ref.split(":", 1)[1] for ref in refs]


def _removed() -> int:
    if not REPORT.is_file():
        return 0
    return len(json.loads(REPORT.read_text(encoding="utf-8"))["per_row"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.dev")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/built"))
    args = parser.parse_args(argv)
    items = [json.loads(line) for line in ITEMS.read_text(encoding="utf-8").splitlines()
             if line.strip()]  # fmt: skip
    rows = dev_rows(items, load_labels(args.labels))
    overlap = training_overlap(rows, train_files(args.data))
    if overlap:
        REPORT.write_text(json.dumps({"reference": "hakembench-dev-v0", "per_row": [
            {**o, "flagged": True} for o in overlap]}, indent=2), encoding="utf-8")  # fmt: skip
        for path in sorted({Path(o["file"]) for o in overlap}):
            print(path, json.dumps(apply(REPORT, path)))
        overlap = training_overlap(rows, train_files(args.data))
    if overlap:
        print(f"{len(overlap)} training rows still carry a dev text; not frozen")
        return 1
    PRIVATE.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    PRIVATE.write_text(body, encoding="utf-8")
    manifest = {
        "file": str(PRIVATE),
        "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "rows": len(rows),
        "per_cell": dict(sorted(Counter(r["cell"] for r in rows).items())),
        "per_type": dict(sorted(Counter(r["type"] for r in rows).items())),
        "training_files_checked": [str(p) for p in train_files(args.data)],
        "training_rows_removed": _removed(),
        "training_rows_matching": 0,
        "text_sha256": sorted({text_hash(r["text"]) for r in rows}),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "text_sha256"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
