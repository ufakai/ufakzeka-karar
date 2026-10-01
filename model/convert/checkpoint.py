"""Saving a run so a preempted one continues instead of restarting.

Modal caps a Function call at 24 hours and cannot make a GPU container
non-preemptible, so a long conversion is not one run but a series of them. The
documentation is blunt about the consequence: checkpoint and resume are
mandatory, and skipping them is the usual way teams lose GPU hours
(modal.com/docs/guide/preemption and /examples/long-training, read
2026-09-20).

Three things make the difference between a checkpoint and a false economy.

It must be complete. Weights alone restart the optimizer from nothing, which
loses the moments Adam spent thousands of steps estimating and puts a
discontinuity in the middle of a run that is meant to be one curve. So the
optimizer state, the step, the tokens seen and the stream cursors all travel
together.

It must be atomic. A container preempted halfway through a write leaves a file
that loads as a plausible model with corrupt tensors. Writing to a temporary
name and renaming means a reader sees either the old checkpoint or the new one
and never a half of either.

It must refuse a mismatch. Resuming a 1024-context run into a 512-context
config, or a different backbone, produces numbers rather than an error, and
numbers are what we publish. Every checkpoint carries a fingerprint of the run
it belongs to and a load against a different one raises.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

STEP_PATTERN = re.compile(r"^step_(\d+)\.pt$")
# A rung is a checkpoint kept on purpose, at a token count worth evaluating. It
# has its own name so that pruning never removes it and `latest` never resumes
# from it by accident.
RUNG_PATTERN = re.compile(r"^rung_(\d+)\.pt$")


@dataclass(frozen=True)
class Progress:
    """Everything a resume needs that is not a tensor."""

    step: int
    tokens_seen: int
    cursors: dict[str, int] = field(default_factory=dict)


def fingerprint(config: dict) -> str:
    """A short stable hash of the settings a resume must not differ on."""
    import hashlib

    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def save(
    directory: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    progress: Progress,
    run_fingerprint: str,
    keep: int = 2,
    rung: bool = False,
) -> Path:
    """Write a checkpoint atomically and prune all but the newest `keep`.

    With `rung`, the file is named for the tokens seen and is never pruned.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"rung_{progress.tokens_seen}" if rung else f"step_{progress.step}"
    path = directory / f"{stem}.pt"
    temporary = directory / f".{stem}.pt.partial"

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "progress": asdict(progress),
        "fingerprint": run_fingerprint,
        "torch_rng": torch.get_rng_state(),
    }
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    if not rung:
        _prune(directory, keep)
    return path


def rungs(directory: Path) -> dict[int, Path]:
    """The rungs a run has kept, by tokens seen."""
    if not directory.is_dir():
        return {}
    return {
        int(m.group(1)): p for p in sorted(directory.iterdir()) if (m := RUNG_PATTERN.match(p.name))
    }


def latest(directory: Path) -> Path | None:
    """The newest complete checkpoint, or None. Partial writes are invisible."""
    if not directory.is_dir():
        return None
    found = [(int(m.group(1)), p) for p in directory.iterdir() if (m := STEP_PATTERN.match(p.name))]
    return max(found)[1] if found else None


def load(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    run_fingerprint: str,
    map_location: str | torch.device = "cpu",
    branch: bool = False,
) -> Progress:
    """Restore a run in place and return where it had got to.

    `branch` is the one case where a different run's checkpoint is wanted: a
    decay branch continues a trunk's weights, optimizer and data position under
    a schedule that ends sooner. It has to be asked for by name, so a resume
    can never turn into one by accident.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    found = payload.get("fingerprint")
    if found != run_fingerprint and not branch:
        raise ValueError(
            f"{path.name} belongs to run {found}, not {run_fingerprint}. Resuming across a "
            "configuration change would produce numbers rather than an error, so it is refused."
        )
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if "torch_rng" in payload:
        torch.set_rng_state(payload["torch_rng"].to(torch.uint8).cpu())
    return Progress(**payload["progress"])


def _prune(directory: Path, keep: int) -> None:
    if keep < 1:
        raise ValueError(f"keep must be at least 1, got {keep}")
    found = sorted(
        ((int(m.group(1)), p) for p in directory.iterdir() if (m := STEP_PATTERN.match(p.name))),
        reverse=True,
    )
    for _, path in found[keep:]:
        path.unlink(missing_ok=True)
