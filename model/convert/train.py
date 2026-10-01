"""The conversion loop: same weights, same text, a different way of reading.

Nothing here trains a model from scratch. The run starts from
ufakzeka-1-base's weights, which already hold 13.5B tokens of Turkish, and
changes one thing: attention stops hiding the future. The weights have never
met that. Every attention head was fitted under a mask that made the right-hand
context unavailable, and relative offsets were never negative. So the model has
to learn to use what it already knows under a new arrangement, which is what
these steps buy.

That is also why the text is the backbone's own. Reading new material
would confound the measurement the step 5 gate depends on: a change in the
instrument could then be the new data rather than the new attention. Holding
the diet fixed leaves the objective as the only variable, and it keeps the
model from drifting off the distribution it was fitted to, which is the
forgetting BidirLM spends a weight merge to undo.

The loss is masked-language, not the causal head's shifted loss, so the logits
are scored against the labels at the same position rather than the next one.
Passing `labels=` to a causal model here would silently train the wrong thing
and still produce a falling curve.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn

# Looked up on this module by the library engine, so a test can watch it here.
from model.convert.bidirectional import document_ids_for  # noqa: F401
from model.convert.checkpoint import Progress, fingerprint, latest, load, save
from model.convert.objective import IGNORE, mask_tokens, masking_rate
from model.convert.schedule import steps_for_tokens, wsd_scale


@dataclass(frozen=True)
class ConvertConfig:
    """Everything the run needs, and everything a resume must agree on."""

    backbone: str
    revision: str
    mask_id: int
    separator_id: int
    pad_id: int
    vocab_size: int
    total_tokens: int = 5_000_000_000
    context: int = 1024
    batch_tokens: int = 524_288
    micro_batch: int = 8
    peak_lr: float = 1e-4
    warmup_steps: int = 500
    decay_frac: float = 0.20
    mask_stable: float = 0.30
    mask_decay: float = 0.10
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0
    seed: int = 1
    checkpoint_every: int = 500
    log_every: int = 20
    mask_documents: bool = True
    # The fused optimizer step is what the throughput probe measured. It
    # needs CUDA, so the loop turns it off by itself anywhere else.
    fused: bool = True
    # Token counts at which a checkpoint is kept for evaluation. Not part of
    # the fingerprint: adding a rung to a run in progress changes nothing it
    # has already done.
    rungs: tuple[int, ...] = ()
    # Which network and optimizers run the loop: "library" is the published
    # class with AdamW, "native" is the backbone's own network with the Muon
    # and AdamW split it was pretrained with (model/convert/engine.py). The
    # native rates are the backbone's continued-pretraining ones.
    engine: str = "library"
    muon_lr: float = 0.01
    muon_momentum: float = 0.95
    embed_lr: float = 3e-3
    scalar_lr: float = 3e-4
    z_loss: float = 1e-5
    # The native engine's document mask as a dense matrix instead of a block
    # mask. The same attention either way; which is faster depends on the
    # context, and at 1024 it is a thing to measure.
    dense_mask: bool = False
    # "mlm" is the conversion. "clm" is continued causal pretraining on new
    # text (PLAN.md backbone item 4): the backbone's own objective, next
    # token under its causal mask inside each document, native engine only.
    objective: str = "mlm"

    @property
    def total_steps(self) -> int:
        return steps_for_tokens(self.total_tokens, batch_tokens=self.batch_tokens)

    @property
    def batch_rows(self) -> int:
        return max(1, self.batch_tokens // self.context)

    @property
    def accumulation(self) -> int:
        """Forward passes per step, none of them over `micro_batch` rows.

        Rounded up. Rounding down made 512 rows at a micro-batch of 48 into ten
        passes of 52, which is 8 percent more memory than the preflight had
        approved, on a card the probe measured at 70 of 80 GB.
        """
        return max(1, -(-self.batch_rows // self.micro_batch))

    def fingerprint(self) -> str:
        """The settings a resume must match. Excludes what can change safely."""
        fixed = {
            "backbone": self.backbone,
            "revision": self.revision,
            "total_tokens": self.total_tokens,
            "context": self.context,
            "batch_tokens": self.batch_tokens,
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "decay_frac": self.decay_frac,
            "mask_stable": self.mask_stable,
            "mask_decay": self.mask_decay,
            "seed": self.seed,
        }
        # Settings added to the fingerprint later are included only when they
        # differ from their default, so every checkpoint written before still
        # resumes, and a resume can no longer change them silently.
        for name, default in (
            ("mask_documents", True), ("weight_decay", 0.01), ("z_loss", 1e-5),
            ("objective", "mlm"),
        ):  # fmt: skip
            if getattr(self, name) != default:
                fixed[name] = getattr(self, name)
        if self.engine != "library":
            # Added only off the default, so checkpoints written before the
            # engines existed still resume.
            fixed |= {
                "engine": self.engine,
                "muon_lr": self.muon_lr,
                "embed_lr": self.embed_lr,
                "scalar_lr": self.scalar_lr,
            }
        return fingerprint(fixed)


def mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy at the masked positions, with no shift.

    A causal model's own loss compares position i against token i+1. Masked
    language modelling compares position i against token i, so the loss is
    taken here rather than by passing `labels=` to the model.
    """
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        labels.reshape(-1),
        ignore_index=IGNORE,
    )


def masked_logits(model: nn.Module, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Project only the positions the loss will read.

    Projecting every position to a 40960-wide vocabulary is the largest single
    allocation in the step: 8 rows of 1024 is 1.34 GB per float32 copy, and
    between the logits, the float cast and the backward there are several. It
    was the whole reason a micro-batch of 32 ran out of memory on a card with
    22 GB while the model itself needs two.

    Only 30 percent of positions are masked, so only 30 percent need logits.
    Gathering first cuts that allocation by three and a third and removes the
    same share of the head's matmul, which is a fifth of the forward pass.
    """
    selected = labels != IGNORE
    return model.lm_head(hidden[selected])


def masked_loss(model: nn.Module, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """The same number `mlm_loss` gives, without materialising every logit."""
    selected = labels != IGNORE
    if not selected.any():
        # cross_entropy over nothing is NaN; the objective guarantees at least
        # one selected position, and this keeps the guarantee local too.
        # Tied to the graph, so `.backward()` on it is legal and adds nothing.
        return hidden.sum().float() * 0.0
    return nn.functional.cross_entropy(model.lm_head(hidden[selected]).float(), labels[selected])


def next_token_labels(batch: np.ndarray, config: ConvertConfig) -> np.ndarray:
    """Each position's label is the next token of its own document.

    The last position has no next token in the window, and a separator closes
    its document, so the token after it starts another one and is not what the
    separator's position should predict; both are ignored, as are pads.
    """
    labels = np.full(batch.shape, IGNORE, dtype=np.int64)
    labels[:, :-1] = batch[:, 1:]
    closing = batch == config.separator_id
    labels[closing] = IGNORE
    labels[batch == config.pad_id] = IGNORE
    labels[:, :-1][batch[:, 1:] == config.pad_id] = IGNORE
    return labels


def corrupt(batch: np.ndarray, config: ConvertConfig, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Mask one batch at the rate its step calls for, reproducibly.

    Under the causal objective nothing is masked: the inputs are the text and
    the labels its next tokens.

    The generator is derived from the seed and the step rather than carried
    along, so a resumed run corrupts the same batch the same way without
    saving any generator state.
    """
    if config.objective == "clm":
        return batch, next_token_labels(batch, config)
    rate = masking_rate(
        step,
        config.total_steps,
        stable=config.mask_stable,
        decay=config.mask_decay,
        decay_frac=config.decay_frac,
    )
    return mask_tokens(
        batch,
        rate=rate,
        mask_id=config.mask_id,
        vocab_size=config.vocab_size,
        rng=np.random.default_rng([config.seed, step]),
        # The separator is protected as well as the padding. The document mask
        # is read off the separators, so hiding one joins two documents, and at
        # 30 percent masking that was a fifth of all boundaries on every step.
        protected=frozenset({config.pad_id, config.separator_id}),
    )


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Decay the matrices, not the norms and biases.

    Low layers are never excluded. CLM's supervision shapes layers 0 to 7 far
    more than MLM does, and freezing them removed the benefit entirely in the
    study that measured it (arXiv 2605.12438), so every layer trains.
    """
    decay, plain = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (plain if parameter.ndim < 2 else decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]


class BatchSource(Protocol):
    """A stream that can say where it is and be put back there.

    `train` takes this rather than a bare callable on purpose. Cursors travel
    in the checkpoint, and a callable has nowhere to put them back, so a
    resumed run would silently reread windows it had already trained on.
    """

    def batch(self, step: int, size: int) -> tuple[np.ndarray, dict[str, int]]: ...

    def state(self) -> dict[str, int]: ...

    def load_state(self, cursors: dict[str, int]) -> None: ...


def precision_context(device: str | torch.device):
    """bfloat16 autocast on hardware that supports it, plain float32 otherwise.

    Support is read from the compute capability rather than from
    `is_bf16_supported()`, which returns true on cards that emulate it slowly,
    and which the instrument already learned not to trust. Ampere is 8.0, and
    every GPU this project runs on is at or above it. No gradient scaler: bf16
    has float32's exponent range, so the underflow that fp16 needs a scaler for
    does not arise.
    """
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        return contextlib.nullcontext()
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 0):
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16)


@dataclass
class RunReport:
    """What the run did, for the ledger and the results file."""

    steps: int = 0
    tokens: int = 0
    wall_seconds: float = 0.0
    losses: list[float] = field(default_factory=list)
    # Wall time between one step's loss being read and the next, which is the
    # one point per step where the GPU is synchronised. The probe reads its
    # speed from the later entries, so it times this loop and not a copy of it.
    step_seconds: list[float] = field(default_factory=list)
    # Set when the loss stopped being a number. A diverged run is over, and
    # saying so is what stops the next hour being paid for.
    diverged_at_step: int | None = None
    # Tokens a resume or a branch arrived with. `tokens` counts from the start
    # of the run, which is what a checkpoint needs; speed must not.
    tokens_at_start: int = 0

    @property
    def tokens_per_second(self) -> float:
        """Speed of this launch alone. Dividing every token the run has ever
        seen by this launch's seconds reported a resumed trunk 12 percent fast
        and a decay branch at 966,000 tok/s."""
        done = self.tokens - self.tokens_at_start
        return done / self.wall_seconds if self.wall_seconds else 0.0


def train(
    model: nn.Module,
    config: ConvertConfig,
    source: BatchSource,
    *,
    out_dir: Path,
    device: str = "cpu",
    on_checkpoint: Callable[[], None] | None = None,
    max_steps: int | None = None,
    branch_from: Path | None = None,
) -> RunReport:
    """Run the conversion, resuming from `out_dir` if a checkpoint is there.

    A resume restores the weights, the optimizer, the step and the stream's
    position, so the continued run reads the windows it had not reached rather
    than starting the data again. `on_checkpoint` is called after each write,
    which is where a Modal volume commit belongs.
    """
    from model.convert.engine import ENGINES

    engine = ENGINES[config.engine]
    model.to(device)
    model.train()
    optimizer = engine.optimizers(model, config, device)

    run_id = config.fingerprint()
    start = Progress(step=0, tokens_seen=0, cursors=source.state())
    resume_from = latest(out_dir)
    # A branch starts from another run's rung, once. After its own first
    # checkpoint exists it resumes from that like any other run, so a preempted
    # branch does not go back to the rung and repeat itself.
    branching = resume_from is None and branch_from is not None
    if branching:
        resume_from = branch_from
    if resume_from is not None:
        start = load(
            resume_from,
            model=model,
            optimizer=optimizer,
            run_fingerprint=run_id,
            map_location=device,
            branch=branching,
        )
        # The line this whole class exists for: put the stream back where it
        # was, or the resumed run retrains on text it has already seen.
        source.load_state(start.cursors)

    report = RunReport(tokens=start.tokens_seen, tokens_at_start=start.tokens_seen)
    stop_at = config.total_steps if max_steps is None else min(config.total_steps, max_steps)
    autocast = precision_context(device)
    began = time.perf_counter()
    synced = began

    for step in range(start.step, stop_at):
        scale = wsd_scale(
            step,
            config.total_steps,
            warmup_steps=config.warmup_steps,
            decay_frac=config.decay_frac,
        )
        optimizer.scale_lr(scale)

        rows, _ = source.batch(step, config.batch_rows)
        inputs, labels = corrupt(rows, config, step)

        optimizer.zero_grad(set_to_none=True)
        running = torch.zeros((), device=device, dtype=torch.float32)
        bounds = range(0, len(rows), config.micro_batch)
        splits = len(bounds)
        for low in bounds:
            high = low + config.micro_batch
            ids = torch.from_numpy(np.ascontiguousarray(inputs[low:high])).to(device)
            targets = torch.from_numpy(np.ascontiguousarray(labels[low:high])).to(device)
            # Boundaries come from the text as it was, never from the corrupted
            # copy: a random replacement can also invent a separator.
            clean = torch.from_numpy(np.ascontiguousarray(rows[low:high])).to(device)
            with autocast:
                # The head is applied only where the loss will read, inside
                # autocast; the softmax itself is float32 because that is the
                # quantity being optimised and bf16 loses accuracy there.
                loss = engine.loss(model, ids, clean, targets, config) / splits
            loss.backward()
            running += loss.detach()

        # One synchronisation per step rather than one per forward pass.
        total = float(running)
        now = time.perf_counter()
        report.step_seconds.append(now - synced)
        synced = now
        if not math.isfinite(total):
            # Checked before the optimizer moves. Stepping first writes the NaN
            # into the weights, and a checkpoint taken then is the newest one,
            # so a resume would load the wreck. Nothing is saved here: the last
            # good checkpoint stays the latest.
            report.diverged_at_step = step
            print(f"  step {step}: loss is {total}, stopping before the update", flush=True)
            break

        if config.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        report.steps = step + 1 - start.step
        report.tokens += int(rows.size)
        report.losses.append(total)

        if config.log_every and (step + 1) % config.log_every == 0:
            seconds = time.perf_counter() - began
            done = report.tokens - start.tokens_seen
            print(
                f"  step {step + 1}/{config.total_steps} loss {total:.4f} "
                f"lr x{scale:.3f} "
                f"{done / seconds:,.0f} tok/s",
                flush=True,
            )

        crossed = [
            rung for rung in config.rungs if report.tokens - int(rows.size) < rung <= report.tokens
        ]
        if crossed:
            save(
                out_dir,
                model=model,
                optimizer=optimizer,
                progress=Progress(step + 1, report.tokens, source.state()),
                run_fingerprint=run_id,
                rung=True,
            )
            print(f"  rung kept at {report.tokens:,} tokens, step {step + 1}", flush=True)
            if on_checkpoint is not None:
                on_checkpoint()

        last = step + 1 == stop_at
        if last or (step + 1) % config.checkpoint_every == 0:
            save(
                out_dir,
                model=model,
                optimizer=optimizer,
                progress=Progress(step + 1, report.tokens, source.state()),
                run_fingerprint=run_id,
            )
            if on_checkpoint is not None:
                on_checkpoint()

    report.wall_seconds = time.perf_counter() - began
    return report
