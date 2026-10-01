"""One fine-tuning run of the instrument.

The instrument compares backbones, so every backbone goes through this same
code: same optimiser, same schedule, same inputs, same evaluation points.
The run keeps no checkpoints. It writes validation and test logits at every
evaluation point and a run.json that records the conditions, and selection
happens afterwards in calib/selection.py. Writing logits instead of weights
is what makes the comparison auditable at a few megabytes per run.

    uv run --group instrument python -m model.instrument.train \\
        --backbone dbmdz/bert-base-turkish-cased --dataset-dir data/raw/instrument/mide22 \\
        --out-dir results/instrument/mide22/berturk-lr3e-5-s1 --lr 3e-5 --seed 1 --epochs 5
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import subprocess
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import accelerate
import numpy as np
import torch
import transformers
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)

from data.instrument_format import SPLITS, Row, read_split

POOLINGS = ("stock", "appended_end", "mean")
# The backbone as pretrained, and the arm measured against it.
ATTENTIONS = ("causal", "bidirectional")

# TabiBench's AdamW settings, kept fixed across backbones so the comparison
# is about the backbone and not about the optimiser.
ADAM_BETAS = (0.9, 0.98)
ADAM_EPS = 1e-6
GRAD_CLIP = 1.0

ModelLoader = Callable[[int], tuple[PreTrainedTokenizerBase, PreTrainedModel]]


@dataclass
class RunConfig:
    backbone: str
    revision: str | None
    dataset_dir: Path
    out_dir: Path
    learning_rate: float
    seed: int
    epochs: float
    batch_size: int = 32
    # Micro-batches accumulate into batch_size, so the optimiser sees exactly the
    # same batch and the same number of steps while activations fit in memory.
    # A 512-token dataset does not fit at batch 32 on the GPU we use.
    micro_batch_size: int | None = None
    weight_decay: float = 1e-5
    warmup: float = 0.06
    eval_points: int = 10
    max_length_cap: int = 512
    pooling: str = "stock"
    # "causal" keeps the backbone as it was trained; "bidirectional" is the
    # arm that comparison buys. Nothing else differs between them.
    attention: str = "causal"
    # The learning-rate sweep selects on validation alone, so it does not
    # pay to evaluate test. Runs that feed the results table write both.
    eval_splits: tuple[str, ...] = ("validation", "test")
    device: str | None = None
    # Tests and smoke runs pass a tiny local model, so the protocol can be
    # exercised without downloading a backbone.
    model_loader: ModelLoader | None = None


def _conversion_fingerprint(backbone: str) -> str | None:
    """The run fingerprint a converted checkpoint folder carries, or None.

    A folder has no hub revision, so without this its runs recorded null and a
    different export into the same folder would have been scored with no error.
    """
    note = Path(backbone) / "conversion.json"
    if not note.is_file():
        return None
    return json.loads(note.read_text(encoding="utf-8")).get("fingerprint")


def config_from_payload(spec: dict, *, data_root: Path, runs_root: Path, device: str) -> RunConfig:
    """The run a dispatched payload describes. The other half of `plan.payload`.

    Every field that changes what is trained is read with `spec[...]`, never
    `.get(...)` with a default: a payload that lacks one is refused here rather
    than trained as something else. That default is how a whole arm ran under
    the wrong attention mask without an error anywhere.
    """
    return RunConfig(
        backbone=spec["model_id"],
        revision=spec["revision"],
        dataset_dir=data_root / spec["dataset"],
        out_dir=runs_root / spec["path"],
        learning_rate=spec["learning_rate"],
        seed=spec["seed"],
        epochs=spec["epochs"],
        batch_size=spec["batch_size"],
        micro_batch_size=spec["micro_batch_size"],
        pooling=spec["pooling"],
        attention=spec["attention"],
        eval_splits=tuple(spec["eval_splits"]),
        device=device,
    )


class MeanPooledClassifier(nn.Module):
    """Mask-weighted mean of the base model's final states, then the model's own head.

    One arm of the pooling ablation for the causal backbone. The published
    evidence for last-token pooling under a causal mask is all at 1.3B
    parameters or more, so at 150M we measure it ourselves.
    """

    def __init__(self, model: PreTrainedModel, bidirectional: bool = False) -> None:
        super().__init__()
        if not (hasattr(model, "model") and hasattr(model, "score")):
            raise ValueError(
                f"mean pooling needs a model with .model and .score, "
                f"{type(model).__name__} has neither"
            )
        self.base = model.model
        self.score = model.score
        # A checkpoint converted with a masked-language objective has no
        # position that was trained to summarise the sequence: each one learnt
        # to recover itself. Last-token pooling, which suits a causal model,
        # then reads one token's state and calls it the text, so a converted
        # checkpoint is also read the way encoders are, by the mean.
        self.bidirectional = bidirectional

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.bidirectional:
            from model.convert.bidirectional import bidirectional_masks

            embeds = self.base.embed_tokens(input_ids)
            masks = bidirectional_masks(self.base, embeds, attention_mask)
            states = self.base(inputs_embeds=embeds, attention_mask=masks).last_hidden_state
        else:
            states = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        weights = attention_mask.unsqueeze(-1).to(states.dtype)
        pooled = (states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.score(pooled)


class BidirectionalClassifier(nn.Module):
    """The same classifier, reading in both directions and nothing else changed.

    Gemma Encoder (arXiv 2503.02656) reports that a decoder fine-tuned with
    bidirectional masking and no conversion training at all matches T5-large on
    GLUE, at 2B and 9B parameters. BidirLM reports the opposite, that the mask
    flip alone leaves performance "mixed" until a masked-prediction phase is
    added. Nobody has published which holds at 150M, which is where we are, and
    the answer decides whether step 2 is worth buying.

    So this arm changes exactly one thing against the committed causal numbers:
    the mask. Same protocol, same seeds, same selection rule, same calibration,
    same last-token pooling, which is also the pooling that paper recommends.

    The classification head finds its token from `input_ids != pad_token_id`
    rather than from the attention mask (transformers 5.17.0, read 2026-09-21),
    so replacing the mask with a prepared mapping does not disturb pooling.
    """

    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        if not hasattr(model, "model"):
            raise ValueError(
                f"bidirectional attention needs a model with .model, "
                f"{type(model).__name__} has none"
            )
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        from model.convert.bidirectional import bidirectional_masks

        embeds = self.model.get_input_embeddings()(input_ids)
        masks = bidirectional_masks(self.model.model, embeds, attention_mask)
        return self.model(input_ids=input_ids, attention_mask=masks).logits


def choose_precision(device_type: str, capability: tuple[int, int] | None) -> dict[str, bool]:
    """Pick autocast and TF32 from the compute capability, never from a support query.

    `torch.cuda.is_bf16_supported()` answers True on a T4, which emulates
    bf16 slowly, so asking it would silently halve our throughput.
    """
    if device_type == "cuda" and capability is not None and capability[0] >= 8:
        return {"bf16": True, "fp16": False, "tf32": True}
    if device_type == "cuda":
        return {"bf16": False, "fp16": True, "tf32": False}
    return {"bf16": False, "fp16": False, "tf32": False}


def token_lengths(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    text_pairs: Sequence[str] | None,
) -> list[int]:
    encoded = tokenizer(
        list(texts),
        text_pair=list(text_pairs) if text_pairs is not None else None,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    return [len(ids) for ids in encoded["input_ids"]]


def max_length_for(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    text_pairs: Sequence[str] | None,
    cap: int,
    model_limit: int,
) -> int:
    """The shortest length that holds 95 percent of the rows, under the cap.

    The quantile is taken with the inverted-CDF rule, so a handful of very
    long rows cannot pull the budget up: with 95 percent of rows at one
    token, the answer is one token, and those rows get truncated.
    """
    lengths = token_lengths(tokenizer, texts, text_pairs)
    percentile = math.ceil(float(np.quantile(lengths, 0.95, method="inverted_cdf")))
    return max(1, min(percentile, cap, model_limit))


def default_loader(
    config: RunConfig, num_labels: int
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(config.backbone, revision=config.revision)
    model_config = AutoConfig.from_pretrained(
        config.backbone, revision=config.revision, num_labels=num_labels
    )
    extra: dict[str, str] = {}
    # The ModernBERT family pools the mean, and both TabiBERT and MoganBERT-TR
    # are evaluated that way by their authors. Pin it rather than inherit it.
    if getattr(model_config, "classifier_pooling", None) is not None:
        extra["classifier_pooling"] = "mean"
    model = AutoModelForSequenceClassification.from_pretrained(
        config.backbone,
        revision=config.revision,
        num_labels=num_labels,
        # The backbone's config says bfloat16 and v5 would honour it; fine-tuning
        # needs float32 master weights.
        dtype=torch.float32,
        attn_implementation="sdpa",
        **extra,
    )
    return tokenizer, model


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_split(
    tokenizer: PreTrainedTokenizerBase, rows: list[Row], max_length: int
) -> tuple[list[dict[str, list[int]]], float]:
    """Tokenize one split and report the share of rows that lost text to truncation."""
    texts = [row.text for row in rows]
    pairs = [row.text_pair for row in rows]
    text_pairs = pairs if any(pair is not None for pair in pairs) else None
    lengths = token_lengths(tokenizer, texts, text_pairs)
    encoded = tokenizer(
        texts,
        text_pair=text_pairs,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    features: list[dict[str, list[int]]] = []
    for index in range(len(rows)):
        feature = {"input_ids": list(encoded["input_ids"][index])}
        if "token_type_ids" in encoded:
            feature["token_type_ids"] = list(encoded["token_type_ids"][index])
        features.append(feature)
    truncated = sum(1 for length in lengths if length > max_length)
    return features, truncated / len(rows) if rows else 0.0


def collate(
    tokenizer: PreTrainedTokenizerBase,
    features: Sequence[dict[str, list[int]]],
    appended_token_id: int | None,
) -> dict[str, torch.Tensor]:
    items: list[dict[str, list[int]]] = []
    for feature in features:
        item = {key: list(value) for key, value in feature.items()}
        if appended_token_id is not None:
            # Appended after truncation, so the pooled position is always this
            # token and never a truncated tail (arXiv 2506.05176).
            item["input_ids"].append(appended_token_id)
            if "token_type_ids" in item:
                item["token_type_ids"].append(item["token_type_ids"][-1])
        items.append(item)
    return tokenizer.pad(items, padding=True, return_attention_mask=True, return_tensors="pt")


def forward_logits(module: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    output = module(**batch)
    return output if isinstance(output, torch.Tensor) else output.logits


def batch_order(rng: np.random.Generator, rows: int, batch_size: int) -> Iterator[np.ndarray]:
    """Shuffled row indices, reshuffled every epoch, so data order follows the seed."""
    while True:
        order = rng.permutation(rows)
        for start in range(0, rows, batch_size):
            yield order[start : start + batch_size]


def evaluation_steps(total_steps: int, points: int) -> list[int]:
    """Evenly spaced points over training, the last one at the final step."""
    steps = {max(1, round(total_steps * (index + 1) / points)) for index in range(points)}
    steps.add(total_steps)
    return sorted(steps)


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    """The usual grouping: no decay on biases and on norm weights."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim <= 1 or name.endswith(".bias") else decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def linear_schedule(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        remaining = total_steps - step
        return max(0.0, remaining / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def versions() -> dict[str, str]:
    found = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "python": platform.python_version(),
    }
    try:
        import datasets
    except ImportError:
        return found
    found["datasets"] = datasets.__version__
    return found


def git_commit() -> str | None:
    """The commit of the code that ran, so every run traces to committed code.

    On a rented GPU the code arrives as an archive without a .git folder. The
    launcher then passes the commit it archived in KARAR_GIT_COMMIT.
    """
    from_launcher = os.environ.get("KARAR_GIT_COMMIT")
    if from_launcher:
        return from_launcher
    root = Path(__file__).resolve().parents[2]
    try:
        done = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


def run(config: RunConfig) -> Path:
    """Fine-tune one backbone on one dataset and write the run folder."""
    started = time.monotonic()
    out_dir = Path(config.out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; a measured run is never overwritten")
    if config.pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}, got {config.pooling!r}")
    if config.attention not in ATTENTIONS:
        raise ValueError(f"attention must be one of {ATTENTIONS}, got {config.attention!r}")

    dataset_dir = Path(config.dataset_dir)
    meta = json.loads((dataset_dir / "meta.json").read_text(encoding="utf-8"))
    labels = meta["labels"]
    rows = {split: read_split(dataset_dir / f"{split}.jsonl") for split in SPLITS}

    device = torch.device(config.device or pick_device())
    precision = choose_precision(
        device.type, torch.cuda.get_device_capability() if device.type == "cuda" else None
    )
    torch.backends.cuda.matmul.allow_tf32 = precision["tf32"]
    torch.backends.cudnn.allow_tf32 = precision["tf32"]

    # Before the model is built, so the new head's initialisation follows the seed.
    set_seed(config.seed)
    loader = config.model_loader or (lambda num_labels: default_loader(config, num_labels))
    tokenizer, model = loader(len(labels))

    wrong_dtypes = sorted({str(p.dtype) for p in model.parameters()} - {"torch.float32"})
    if wrong_dtypes:
        raise ValueError(f"weights must be float32, found {wrong_dtypes}")
    config_pad = getattr(model.config, "pad_token_id", None)
    if config_pad is not None and config_pad != tokenizer.pad_token_id:
        # The causal head finds its position by comparing input ids with
        # config.pad_token_id, so a disagreement pools the wrong token.
        raise ValueError(
            f"config pad_token_id {config_pad} is not the tokenizer's pad token "
            f"{tokenizer.pad_token_id}"
        )

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    # BERTurk's legacy tokenizer config leaves model_max_length unset, so the
    # real limit is read from the model, for every backbone alike.
    model_limit = getattr(model.config, "max_position_embeddings", None) or config.max_length_cap
    train_pairs = [row.text_pair for row in rows["train"]]
    max_length = max_length_for(
        tokenizer,
        [row.text for row in rows["train"]],
        train_pairs if any(pair is not None for pair in train_pairs) else None,
        cap=config.max_length_cap,
        model_limit=model_limit,
    )

    appended_token_id = None
    if config.pooling == "appended_end":
        appended_token_id = tokenizer.eos_token_id
        if appended_token_id is None:
            raise ValueError("appended_end pooling needs a tokenizer with an eos token")
        # The end token is added after truncation, so leave it a position. This
        # only bites when the text budget already equals the model's limit.
        max_length = min(max_length, model_limit - 1)

    features, truncated_share = {}, {}
    for split in SPLITS:
        features[split], truncated_share[split] = encode_split(tokenizer, rows[split], max_length)
    split_labels = {
        split: np.array([row.label for row in rows[split]], dtype=np.int64) for split in SPLITS
    }

    model.to(device)
    module: nn.Module = model
    if config.pooling == "mean":
        module = MeanPooledClassifier(model, bidirectional=config.attention == "bidirectional")
    elif config.attention == "bidirectional":
        module = BidirectionalClassifier(model)
    module.to(device)

    micro = config.micro_batch_size or config.batch_size
    if micro > config.batch_size or config.batch_size % micro:
        raise ValueError(f"micro_batch_size {micro} must divide batch_size {config.batch_size}")
    accumulation = config.batch_size // micro

    train_rows = len(rows["train"])
    steps_per_epoch = max(1, math.ceil(train_rows / config.batch_size))
    total_steps = max(1, round(steps_per_epoch * config.epochs))
    eval_steps = evaluation_steps(total_steps, config.eval_points)
    warmup_steps = int(config.warmup * total_steps)

    optimizer = torch.optim.AdamW(
        parameter_groups(model, config.weight_decay),
        lr=config.learning_rate,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
    )
    scheduler = linear_schedule(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler(device.type) if precision["fp16"] else None
    amp_dtype = None
    if precision["bf16"]:
        amp_dtype = torch.bfloat16
    elif precision["fp16"]:
        amp_dtype = torch.float16

    def autocast():
        if amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=amp_dtype)

    def logits_for(split: str) -> np.ndarray:
        module.eval()
        pieces = []
        with torch.inference_mode():
            for start in range(0, len(features[split]), micro):
                batch = collate(
                    tokenizer,
                    features[split][start : start + micro],
                    appended_token_id,
                ).to(device)
                with autocast():
                    pieces.append(forward_logits(module, batch).float().cpu().numpy())
        return np.concatenate(pieces, axis=0).astype(np.float32)

    eval_splits = tuple(config.eval_splits)
    unknown = [split for split in eval_splits if split not in ("validation", "test")]
    if unknown or not eval_splits:
        raise ValueError(f"eval_splits must be a non-empty subset of validation, test: {unknown}")

    out_dir.mkdir(parents=True)
    for split in eval_splits:
        np.save(out_dir / f"labels_{split}.npy", split_labels[split])

    batches = batch_order(np.random.default_rng(config.seed), train_rows, config.batch_size)
    train_labels = torch.from_numpy(split_labels["train"])
    train_loss: list[list[float]] = []
    losses_since_point: list[float] = []
    diverged_at_step: int | None = None

    for step in range(1, total_steps + 1):
        indices = next(batches)
        module.train()
        optimizer.zero_grad(set_to_none=True)
        # One optimiser step over `accumulation` micro-batches. Each micro-batch
        # contributes its share of the mean, weighted by its size, so a short
        # last micro-batch cannot count for as much as a full one.
        value = 0.0
        for piece in np.array_split(indices, accumulation):
            if not len(piece):
                continue
            batch = collate(tokenizer, [features["train"][i] for i in piece], appended_token_id).to(
                device
            )
            targets = train_labels[piece].to(device)
            with autocast():
                logits = forward_logits(module, batch)
            # The loss is taken in float32 even under autocast: it is the number we log.
            share = len(piece) / len(indices)
            loss = nn.functional.cross_entropy(logits.float(), targets) * share
            value += float(loss.detach())
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        if not math.isfinite(value):
            diverged_at_step = step
            break
        losses_since_point.append(value)
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        scheduler.step()

        if step in eval_steps:
            for split in eval_splits:
                np.save(out_dir / f"logits_{split}_step{step}.npy", logits_for(split))
            train_loss.append([step, float(np.mean(losses_since_point))])
            losses_since_point = []

    report = {
        "backbone": config.backbone,
        "revision": getattr(model.config, "_commit_hash", None)
        or config.revision
        or _conversion_fingerprint(config.backbone),
        "dataset": meta["name"],
        "dataset_source_revision": meta["source_revision"],
        "labels": labels,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "micro_batch_size": micro,
        "weight_decay": config.weight_decay,
        "warmup": config.warmup,
        "max_length": max_length,
        "pooling": config.pooling,
        "attention": config.attention,
        "eval_splits": list(eval_splits),
        "truncated_share": truncated_share,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": precision,
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "parameters_non_embedding": sum(p.numel() for p in model.parameters())
        - model.get_input_embeddings().weight.numel(),
        "weights_dtype": "float32",
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "versions": versions(),
        "git_commit": git_commit(),
        "total_steps": total_steps,
        "eval_steps": eval_steps,
        "train_rows": train_rows,
        "wall_seconds": round(time.monotonic() - started, 3),
        "train_loss": train_loss,
    }
    if appended_token_id is not None:
        report["appended_token_id"] = appended_token_id
    if diverged_at_step is not None:
        # The logits written before the divergence stay: selection skips
        # non-finite points and the record shows where the run went.
        report["diverged_at_step"] = diverged_at_step
    (out_dir / "run.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="one instrument fine-tuning run")
    parser.add_argument("--backbone", required=True, help="Hub model id")
    parser.add_argument("--revision", default=None, help="Hub revision to pin")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--lr", required=True, type=float, dest="learning_rate")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--micro-batch-size", default=None, type=int)
    parser.add_argument("--eval-points", default=10, type=int)
    parser.add_argument("--pooling", default="stock", choices=POOLINGS)
    parser.add_argument(
        "--eval-splits", default="validation,test", help="comma separated, for the sweep"
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    fields = vars(args)
    fields["eval_splits"] = tuple(fields["eval_splits"].split(","))
    out_dir = run(RunConfig(**fields))
    print(out_dir)


if __name__ == "__main__":
    main()
