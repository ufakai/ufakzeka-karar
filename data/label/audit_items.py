"""The owner's blind audit of rows confident learning flags.

    python -m data.label.audit_items --audit results/step6/round2/audit_b.json

The flagged training rows (model.head.scoring's audit sample) are laid out for
the label desk like the first 291 items: text, question and options, no target
and no model answer, so the owner answers blind. The items continue the desk's
order after the last one it holds. Writes the items and the chunk document the
desk reads (results/step6/round2/audit_b_items.jsonl, audit_b_chunk.json).

After the owner answers, `--score` compares each answer with the row's training
target and with the model's answer: if the owner agrees with the model on at
least half, the flag weight applies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.label.owner_sample import options_of
from model.head.mixing import train_files
from schema.rows import TrainingRow, text_of

FIRST_ORDER = 291


def items(row_ids: list[str], data: Path = Path("data/built")) -> list[dict]:
    wanted = set(row_ids)
    found: dict[str, TrainingRow] = {}
    for path in train_files(data):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = TrainingRow.model_validate_json(line)
                if row.row_id in wanted:
                    found[row.row_id] = row
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"{len(missing)} audit rows are not in the training files")
    return [{"id": r.row_id, "order": FIRST_ORDER + n, "cell": f"denetim: {r.task}",
             "task": r.task, "split": r.split, "type": r.question.type,
             "text": text_of(r.state), "question": text_of(r.question.instructions),
             "options": options_of(r)}
            for n, r in enumerate(found[i] for i in row_ids)]  # fmt: skip


def verdict(audit: list[dict], labels: dict[str, dict], scores: dict[str, dict]) -> dict:
    """How often the owner sides with the model and with the training target."""
    with_model = with_target = answered = 0
    for item in audit:
        label = labels.get(item["id"])
        if not label or label.get("answer") is None:
            continue
        answered += 1
        row = scores[item["id"]]
        answer = str(label["answer"])
        with_model += answer == max(row["probs"], key=row["probs"].get)
        with_target += answer == max(row["target"], key=row["target"].get)
    share = with_model / answered if answered else 0.0
    return {"answered": answered, "owner_with_model": with_model,
            "owner_with_target": with_target, "flags_apply": share >= 0.5}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data.label.audit_items")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/step6/round2"))
    args = parser.parse_args(argv)
    laid = items(json.loads(args.audit.read_text(encoding="utf-8")))
    stem = args.audit.stem
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{stem}_items.jsonl").write_text(
        "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in laid), encoding="utf-8"
    )
    (args.out / f"{stem}_chunk.json").write_text(
        json.dumps({"items": laid}, ensure_ascii=False), encoding="utf-8")  # fmt: skip
    print(len(laid), "audit items written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
