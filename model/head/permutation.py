"""The permutation check on the real backbone, as a results file.

    python -m model.head.permutation --out results/step4/permutation_real_weights.json

The unit test in model/head/tests/test_head.py demands bit-for-bit equality
under reordered options on a small random network. This runs the same demand
on ufakzeka-1-base itself, on CPU in fp32, over real questions drawn from the
validation files, each asked in its given order and in several shuffled
orders. The head's own weights are fresh: the claim is about the network's
arithmetic under the mask, positions and canonical packing, which the head's
values do not enter.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REVISION = "f9e11eea28cbb2ba953a5628f972d416fe0c3cfe"
FILES = (
    "data/built/sss/validation.jsonl",
    "data/built/typed/massive_tr/validation.jsonl",
    "data/built/typed/mide22/validation.jsonl",
)


def run(questions: int, orders: int, seed: int = 1) -> dict:
    import torch
    from pydantic import TypeAdapter
    from transformers import AutoTokenizer

    from model.convert.native import from_pretrained
    from model.head.head import DecisionHead
    from model.head.pack import collate, pack
    from schema.questions import Question
    from schema.rows import TrainingRow, outcomes

    torch.manual_seed(0)
    rng = random.Random(seed)
    adapter = TypeAdapter(Question)
    tokenizer = AutoTokenizer.from_pretrained("ufakai/ufakzeka-1-base", revision=REVISION)
    backbone, _ = from_pretrained("ufakai/ufakzeka-1-base", REVISION)
    model = DecisionHead(backbone, causal=True, pooling="mean").eval()
    model.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False).input_ids

    rows = []
    for path in FILES:
        lines = [x for x in Path(path).read_text("utf-8").splitlines() if x.strip()]
        picked = [TrainingRow.model_validate_json(x) for x in rng.sample(lines, questions * 3)]
        rows += [r for r in picked if r.question.type == "choice"
                 and len(outcomes(r.question)) >= 3][: questions // len(FILES)]  # fmt: skip

    def logits_of(row, question) -> dict[str, float]:
        batch = collate([pack(row.state, question, encode)], model.pad_id)
        with torch.no_grad():
            out, _ = model(batch)
        return dict(zip(batch.keys[0], out[0].tolist(), strict=True))

    exact, checked, worst = 0, 0, 0.0
    for row in rows:
        reference = logits_of(row, row.question)
        criteria = list(row.question.criteria.items())
        for _ in range(orders):
            rng.shuffle(criteria)
            shuffled = adapter.validate_python({"type": "choice",
                                                "instructions": row.question.instructions,
                                                "criteria": dict(criteria)})  # fmt: skip
            got = logits_of(row, shuffled)
            checked += 1
            exact += got == reference
            worst = max(worst, max(abs(got[k] - reference[k]) for k in reference))
    return {
        "backbone": f"ufakai/ufakzeka-1-base@{REVISION}",
        "attention": "causal inside each part, options blind to each other",
        "dtype": "float32",
        "device": "cpu",
        "questions": len(rows),
        "shuffled_orders_per_question": orders,
        "comparisons": checked,
        "bit_exact": exact,
        "max_abs_logit_difference": worst,
        "torch": torch.__version__,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model.head.permutation")
    parser.add_argument(
        "--out", type=Path, default=Path("results/step4/permutation_real_weights.json")
    )
    parser.add_argument("--questions", type=int, default=60)
    parser.add_argument("--orders", type=int, default=5)
    args = parser.parse_args(argv)
    result = run(args.questions, args.orders)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["bit_exact"] == result["comparisons"] else 1


if __name__ == "__main__":
    sys.exit(main())
