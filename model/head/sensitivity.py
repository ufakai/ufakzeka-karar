"""How much a head's answer depends on option order (step 6's claim 5).

On a fixed sample of validation choice questions with at least three options,
each question is scored in its caller's order and in `orders` random orders.
Reported: the flip rate (the answer differs from the caller-order answer) and
the mean total variation between the probability vectors, compared by option
name, each with a bootstrap interval over questions. The blind head is exactly
zero by construction (model/head/permutation.py proves it bit for bit); the
sequential head is measured.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from model.head.pack import arrange, collate, parts


def sample(rows: list, count: int = 300, seed: int = 1) -> list:
    eligible = [r for r in rows if r.question.type == "choice" and len(r.target) >= 3]
    return random.Random(seed).sample(eligible, min(count, len(eligible)))


def order_sensitivity(model, rows: list, encode, device: str, orders: int = 8, seed: int = 1,
                      draws: int = 2000) -> dict:  # fmt: skip
    rng = random.Random(seed)
    flips, variation = [], []
    model.eval()
    with torch.no_grad():
        for row in rows:
            p = parts(row.state, row.question, encode)

            def probs(order, p=p):
                batch = model.to_device(collate([arrange(p, model.layout, order)], model.pad_id),
                                        device)  # fmt: skip
                logits, _ = model(batch)
                values = torch.softmax(logits[0].float(), -1).tolist()
                return dict(zip(batch.keys[0], values, strict=False))

            base = probs(list(p.keys))
            answer = max(base, key=base.get)
            row_flips, row_tv = [], []
            for _ in range(orders):
                other = probs(rng.sample(p.keys, len(p.keys)))
                row_flips.append(float(max(other, key=other.get) != answer))
                row_tv.append(0.5 * sum(abs(base[k] - other[k]) for k in p.keys))
            flips.append(np.mean(row_flips))
            variation.append(np.mean(row_tv))
    boot = np.random.default_rng(seed)
    out = {"questions": len(rows), "orders": orders}
    for name, values in (("flip_rate", np.array(flips)), ("total_variation", np.array(variation))):
        picks = boot.integers(0, len(values), (draws, len(values)))
        bounds = np.percentile(values[picks].mean(1), [2.5, 97.5]).tolist()
        out[name] = {"mean": float(values.mean()), "interval": bounds}
    return out
