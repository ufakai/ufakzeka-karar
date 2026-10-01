"""Training the decision head on typed rows, and scoring it where it trains.

The loop is deliberately small: rows in, packed questions batched by length,
one optimizer with a lower rate for the backbone than for the head, soft
cross-entropy or Brier against each row's target, and an evaluation on the
validation rows at fixed points. Every evaluation point writes its per-row
probabilities, so selection (refine, then calibrate: arXiv 2501.19195) and
the paired bootstrap run afterwards on files, not on a model that has
to be kept.

Metrics per task: macro F1 over option names (correct when a recast MASSIVE row
carries its own ten), accuracy, Brier as the sum over options, log loss, and
the calibration numbers calib/metrics.py computes on the per-row arrays.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from model.head.head import DecisionHead, objective, policy_gradient, soft_targets
from model.head.pack import Packed, arrange, collate, pack, parts
from schema.rows import TrainingRow


@dataclass(frozen=True)
class HeadConfig:
    backbone: str
    revision: str | None
    causal: bool
    train_files: tuple[str, ...]
    validation_files: tuple[str, ...]
    out_dir: str
    pooling: str = "mean"
    loss: str = "ce"
    backbone_lr: float = 3e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: float = 2.0
    batch_questions: int = 32
    warmup_frac: float = 0.06
    eval_points: int = 6
    max_prefix: int = 448
    max_option: int = 48
    max_train_rows: int | None = None
    # Rows times padded length per forward pass; a batch longer than this is
    # split into micro-batches whose gradients add up to the batch's.
    max_micro_tokens: int = 12_000
    seed: int = 1
    device: str = "cpu"
    # Round 1: the ranked probability term on score questions,
    # weights that restore each task's natural answer mix, the mined-row
    # ablation, and the trained weights saved for export and calibration.
    ordinal: bool = True
    prior_weights: bool = False
    drop_mined: bool = False
    save_model: bool = False
    # Round 2: files scored at every evaluation point (training rows,
    # and rows of capped tasks the mix never drew), and a JSON file mapping
    # row ids to weight multipliers, applied inside each task.
    score_files: tuple[str, ...] = ()
    row_weights: str | None = None
    # Step 6's ablations: the conventional sequential layout with the
    # option order shuffled every step (claim 5), and the policy-gradient loss
    # with `rl_samples` answers drawn per question (claim 4, loss="rl").
    layout: str = "blind"
    shuffle_options: bool = False
    rl_samples: int = 8
    # r5: tasks whose validation rows are a dev set; they are evaluated like
    # any validation row but never shape training, not even through the prior weights.
    prior_exclude: tuple[str, ...] = ()


def read_rows(paths: tuple[str, ...], limit: int | None = None, seed: int = 1) -> list[TrainingRow]:
    rows = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(TrainingRow.model_validate_json(line))
    if limit is not None and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
    return rows


WEIGHT_CLIP = (0.25, 4.0)
# A task's natural shares come from its validation rows; with few of them the
# ratio is mostly noise, so its logarithm is shrunk toward zero by n / (n + 20).
SHRINK_ROWS = 20


def prior_weights(
    train: list[TrainingRow],
    validation: list[TrainingRow],
    multipliers: dict[str, float] | None = None,
) -> dict[str, float]:
    """A weight per training row that moves each task's answer mix back to its natural one.

    The build caps a template's most common answer in training and mines rare
    answers, and leaves the validation split uncapped, so training
    sees a flatter prior than evaluation, which post-hoc temperature cannot
    undo. Each row weighs natural share over training share of its top answer,
    both from counts with one added per option, raised to n / (n + 20) for n
    validation rows so a small task's weights stay near one, clipped to
    WEIGHT_CLIP and scaled to a mean of one inside the task. A task with no validation rows
    keeps weight one.

    With `multipliers` (round 2) the training shares are counted on the
    multiplied rows, each row's weight is its prior weight times its
    multiplier, and the task is scaled back to a mean of one, so reweighting
    moves weight between a task's rows and never between tasks.
    """
    multipliers = multipliers or {}

    def top(row: TrainingRow) -> str:
        return max(row.target, key=row.target.get)

    def mult(row: TrainingRow) -> float:
        return multipliers.get(row.row_id, 1.0)

    by_task_train, by_task_val = defaultdict(list), defaultdict(list)
    for row in train:
        by_task_train[row.task].append(row)
    for row in validation:
        by_task_val[row.task].append(row)
    weights = {}
    for task, rows in by_task_train.items():
        natural = by_task_val.get(task)
        if not natural:
            raw = [mult(r) for r in rows]
        else:
            keys = sorted({k for r in rows + natural for k in r.target})
            tr: Counter[str] = Counter()
            for r in rows:
                tr[top(r)] += mult(r)
            weighted = sum(mult(r) for r in rows)
            nat = Counter(top(r) for r in natural)
            share = {k: ((nat[k] + 1) / (len(natural) + len(keys)))
                     / ((tr[k] + 1) / (weighted + len(keys))) for k in keys}  # fmt: skip
            keep = len(natural) / (len(natural) + SHRINK_ROWS)
            raw = [min(max(share[top(r)] ** keep, WEIGHT_CLIP[0]), WEIGHT_CLIP[1]) * mult(r)
                   for r in rows]  # fmt: skip
        mean = sum(raw) / len(raw)
        weights.update({r.row_id: w / mean for r, w in zip(rows, raw, strict=True)})
    return weights


def row_multipliers(path: str | None) -> dict[str, float]:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}


def micro_batches(group: list, budget: int) -> list[list]:
    """Consecutive slices of a batch whose rows times longest row stay within `budget`."""
    group = sorted(group, key=lambda item: len(item[0].ids))
    out, current = [], []
    for item in group:
        longest = max([len(item[0].ids)] + [len(p.ids) for p, _ in current])
        if current and (len(current) + 1) * longest > budget:
            out.append(current)
            current = []
        current.append(item)
    if current:
        out.append(current)
    return out


def accumulate(model, group: list, config, weight_of: dict, autocast,
               generator: torch.Generator | None = None) -> torch.Tensor:  # fmt: skip
    """One batch's loss, back-propagated micro-batch by micro-batch.

    The batch's loss is a weighted mean over its rows; each micro-batch's mean is
    scaled by its share of the batch's total weight, so the gradients add up to
    those of the whole batch at once.
    """
    total = sum(weight_of[r.row_id] for _, r in group) if weight_of else float(len(group))
    loss_sum = 0.0
    for micro in micro_batches(group, config.max_micro_tokens):
        batch = model.to_device(collate([p for p, _ in micro], model.pad_id), config.device)
        targets = soft_targets(batch, [r.target for _, r in micro]).to(config.device)
        weights = (
            torch.tensor([weight_of[r.row_id] for _, r in micro], device=config.device)
            if weight_of else None
        )  # fmt: skip
        ordinal = (
            torch.tensor([r.question.type == "score" for _, r in micro], device=config.device)
            if config.ordinal else None
        )  # fmt: skip
        with autocast:
            logits, _ = model(batch)
        share = (float(weights.sum()) if weights is not None else float(len(micro))) / total
        if config.loss == "rl":
            loss = policy_gradient(logits.float(), targets, config.rl_samples, weights,
                                   ordinal, generator) * share  # fmt: skip
        else:
            loss = objective(logits.float(), targets, config.loss, weights, ordinal) * share
        loss.backward()
        loss_sum += float(loss.detach())
    return torch.tensor(loss_sum)


def batches_by_length(
    packed: list[tuple[Packed, TrainingRow]], size: int, rng: random.Random, shuffle: bool
) -> list[list[tuple[Packed, TrainingRow]]]:
    """Batches of similar length, so padding is small, in a shuffled order.

    Sorting within windows of fifty batches keeps most of the padding saving
    while leaving the order random enough that a batch is not all one task.
    """
    items = list(packed)
    if shuffle:
        rng.shuffle(items)
    window = size * 50
    out = []
    for start in range(0, len(items), window):
        chunk = sorted(items[start : start + window], key=lambda item: len(item[0].ids))
        out += [chunk[i : i + size] for i in range(0, len(chunk), size)]
    if shuffle:
        rng.shuffle(out)
    return out


def macro_f1(gold: list[str], predicted: list[str]) -> float:
    labels = sorted(set(gold) | set(predicted))
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted, strict=True))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted, strict=True))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted, strict=True))
        if tp + fp + fn:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else float("nan")


def task_metrics(records: list[dict]) -> dict[str, float]:
    """Metrics for one task from per-row records {keys, probs, target}."""
    gold = [max(r["target"], key=r["target"].get) for r in records]
    predicted = [r["keys"][int(np.argmax(r["probs"]))] for r in records]
    brier = np.mean(
        [sum((p - r["target"][k]) ** 2 for k, p in zip(r["keys"], r["probs"], strict=True))
         for r in records]
    )  # fmt: skip
    logloss = -np.mean(
        [sum(r["target"][k] * math.log(max(p, 1e-12)) for k, p in zip(r["keys"], r["probs"],
         strict=True)) for r in records]
    )  # fmt: skip
    confidence = np.array([max(r["probs"]) for r in records])
    correct = np.array([g == p for g, p in zip(gold, predicted, strict=True)], dtype=float)
    bins = np.minimum((confidence * 15).astype(int), 14)
    ece = sum(
        abs(confidence[bins == b].mean() - correct[bins == b].mean()) * (bins == b).mean()
        for b in range(15)
        if (bins == b).any()
    )
    return {
        "rows": len(records),
        "macro_f1": macro_f1(gold, predicted),
        "accuracy": float(correct.mean()),
        "brier": float(brier),
        "logloss": float(logloss),
        "ece_15": float(ece),
    }


def evaluate(model, batches, device) -> tuple[dict[str, dict], list[dict]]:
    model.eval()
    records = []
    with torch.no_grad():
        for group in batches:
            batch = model.to_device(collate([p for p, _ in group], model.pad_id), device)
            logits, abstain = model(batch)
            probs = torch.softmax(logits.float(), dim=-1).cpu()
            for index, (_, row) in enumerate(group):
                count = len(batch.keys[index])
                records.append(
                    {
                        "row_id": row.row_id,
                        "task": row.task,
                        "keys": batch.keys[index],
                        "probs": probs[index, :count].tolist(),
                        "logits": logits[index, :count].float().cpu().tolist(),
                        "abstain_logit": float(abstain[index]),
                        "target": row.target,
                    }
                )
    model.train()
    by_task = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)
    return {task: task_metrics(rs) for task, rs in sorted(by_task.items())}, records


@dataclass
class HeadReport:
    steps: int = 0
    seconds: float = 0.0
    losses: list[float] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)
    # The trained head, for evaluations the loop does not run itself. Not saved.
    model: object = None


def train_head(config: HeadConfig, backbone, tokenizer) -> HeadReport:
    """Train and evaluate; writes run.json and one predictions file per point."""
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False).input_ids

    def packed_rows(rows):
        return [
            (pack(r.state, r.question, encode, max_prefix=config.max_prefix,
                  max_option=config.max_option, layout=config.layout), r)
            for r in rows
        ]  # fmt: skip

    train_rows = read_rows(config.train_files, config.max_train_rows, config.seed)
    if config.drop_mined:
        train_rows = [r for r in train_rows if not r.recipe.endswith("-mined")]
    validation_rows = read_rows(config.validation_files)
    multipliers = row_multipliers(config.row_weights)
    if config.prior_weights:
        natural = [r for r in validation_rows if r.task not in config.prior_exclude]
        weight_of = prior_weights(train_rows, natural, multipliers)
    elif multipliers:
        weight_of = prior_weights(train_rows, [], multipliers)
    else:
        weight_of = {}
    train_packed, validation_packed = packed_rows(train_rows), packed_rows(validation_rows)
    score_batches = (
        batches_by_length(packed_rows(read_rows(config.score_files)), 64, rng, shuffle=False)
        if config.score_files else []
    )  # fmt: skip

    model = DecisionHead(backbone, causal=config.causal, pooling=config.pooling,
                         layout=config.layout)  # fmt: skip
    # The shuffle draws from its own stream, so the batches come in the same
    # order as in a run without it, and the RL arm samples from its own
    # generator.
    shuffle_rng = random.Random(config.seed + 1000)
    option_parts = (
        {r.row_id: parts(r.state, r.question, encode, max_prefix=config.max_prefix,
                         max_option=config.max_option) for r in train_rows}
        if config.shuffle_options else {}
    )  # fmt: skip
    generator = torch.Generator(device=config.device)
    generator.manual_seed(config.seed)

    def shuffled(group):
        out = []
        for packed, row in group:
            if row.question.type == "score":  # levels keep their natural order
                out.append((packed, row))
                continue
            p = option_parts[row.row_id]
            order = shuffle_rng.sample(p.keys, len(p.keys))
            out.append((arrange(p, config.layout, order), row))
        return out

    model.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    model.to(config.device)
    head_params = [*model.score.parameters(), *model.abstain.parameters()]
    head_ids = {id(p) for p in head_params}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in body if p.ndim >= 2], "lr": config.backbone_lr,
             "weight_decay": config.weight_decay},
            {"params": [p for p in body if p.ndim < 2], "lr": config.backbone_lr,
             "weight_decay": 0.0},
            {"params": head_params, "lr": config.head_lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.98),
        eps=1e-6,
    )  # fmt: skip
    for group in optimizer.param_groups:
        group["base_lr"] = group["lr"]

    per_epoch = math.ceil(len(train_packed) / config.batch_questions)
    total = max(1, int(per_epoch * config.epochs))
    warmup = max(1, int(total * config.warmup_frac))
    eval_at = sorted({max(1, round(total * (i + 1) / config.eval_points)) for i in range(
        config.eval_points)})  # fmt: skip
    validation_batches = batches_by_length(validation_packed, 64, rng, shuffle=False)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(config.device).startswith("cuda")
        else torch.autocast("cpu", enabled=False)
    )

    report = HeadReport()
    began = time.perf_counter()
    step = 0
    while step < total:
        for group in batches_by_length(train_packed, config.batch_questions, rng, shuffle=True):
            if step >= total:
                break
            # Linear warm-up, then linear decay to zero at the last step.
            if step < warmup:
                scale = (step + 1) / warmup
            else:
                scale = max(0.0, (total - step) / max(1, total - warmup))
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["base_lr"] * scale
            optimizer.zero_grad(set_to_none=True)
            if config.shuffle_options:
                group = shuffled(group)
            loss = accumulate(model, group, config, weight_of, autocast, generator)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            report.losses.append(loss.item())
            if step in eval_at:
                metrics, records = evaluate(model, validation_batches, config.device)
                path = out / f"predictions_step{step}.jsonl"
                path.write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                    encoding="utf-8",
                )
                report.evaluations.append({"step": step, "metrics": metrics})
                if score_batches:
                    # Only the records: these rows are scored for weights, not reported.
                    _, scored = evaluate(model, score_batches, config.device)
                    (out / f"scores_step{step}.jsonl").write_text(
                        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scored),
                        encoding="utf-8",
                    )
                summary = ", ".join(
                    f"{task} F1 {m['macro_f1']:.3f} Brier {m['brier']:.3f}"
                    for task, m in metrics.items()
                )
                print(f"  step {step}/{total} loss {np.mean(report.losses[-50:]):.4f} {summary}",
                      flush=True)  # fmt: skip
    report.steps = step
    report.seconds = time.perf_counter() - began
    report.model = model
    if config.save_model:
        # The whole network: the backbone is fine-tuned with the head, and the
        # int8 export and step 7's calibration need both.
        torch.save(model.state_dict(), out / "model.pt")
    (out / "run.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "steps": report.steps,
                "seconds": round(report.seconds, 1),
                "evaluations": report.evaluations,
                "final_loss": float(np.mean(report.losses[-50:])) if report.losses else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report
