"""The two ways the conversion can run: on the library class, or on our own network.

The loop in train.py is the same for both. What differs is how a batch becomes
a loss and which optimizer moves the weights, and those two things are what an
engine supplies. Keeping them behind one seam means the schedule, the
corruption, the rungs, the checkpoints and the divergence guard are written
once and tested once, and a finding on one engine is a finding about the
engine and not about a second copy of the loop.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn

from model.convert.objective import IGNORE


class OptimizerSet:
    """Several optimizers that save, load, step and schedule as one."""

    def __init__(self, *optimizers: torch.optim.Optimizer):
        self.optimizers = optimizers
        for group in self.param_groups:
            group.setdefault("base_lr", group["lr"])

    @property
    def param_groups(self) -> list[dict]:
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def scale_lr(self, scale: float) -> None:
        for group in self.param_groups:
            group["lr"] = group["base_lr"] * scale

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state: dict) -> None:
        saved = state["optimizers"]
        if len(saved) != len(self.optimizers):
            raise ValueError(
                f"checkpoint holds {len(saved)} optimizers, this run has {len(self.optimizers)}"
            )
        for optimizer, item in zip(self.optimizers, saved, strict=True):
            optimizer.load_state_dict(item)
        for group in self.param_groups:
            group.setdefault("base_lr", group["lr"])


class Engine(Protocol):
    name: str

    def optimizers(self, model: nn.Module, config, device: str) -> OptimizerSet: ...

    def loss(self, model: nn.Module, ids, clean, targets, config) -> torch.Tensor: ...


def _masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits.float(), targets)


class LibraryEngine:
    """The published checkpoint in the library's own class, with AdamW."""

    name = "library"

    def optimizers(self, model, config, device):
        from model.convert.train import parameter_groups

        return OptimizerSet(
            torch.optim.AdamW(
                parameter_groups(model, config.weight_decay),
                lr=config.peak_lr,
                betas=config.betas,
                eps=config.eps,
                fused=config.fused and torch.device(device).type == "cuda",
            )
        )

    def loss(self, model, ids, clean, targets, config):
        from model.convert import train as loop
        from model.convert.bidirectional import bidirectional_masks

        if config.objective != "mlm":
            raise ValueError("the library engine runs the conversion only; use the native engine")

        documents = (
            loop.document_ids_for(clean, config.separator_id) if config.mask_documents else None
        )
        embeds = model.get_input_embeddings()(ids)
        masks = bidirectional_masks(model, embeds, document_ids=documents)
        hidden = model.model(
            inputs_embeds=embeds, attention_mask=masks, use_cache=False
        ).last_hidden_state
        return loop.masked_loss(model, hidden, targets)


class NativeEngine:
    """The backbone's own network, with the optimizers it was pretrained with."""

    name = "native"

    def optimizers(self, model, config, device):
        from model.convert.muon import Muon

        matrices = [
            p for n, p in model.body.named_parameters() if p.ndim == 2 and n.startswith("layers.")
        ]
        scalars = [p for p in model.body.parameters() if p.ndim < 2]
        muon = Muon(matrices, lr=config.muon_lr, momentum=config.muon_momentum)
        adam = torch.optim.AdamW(
            [
                {
                    "params": [model.body.embed_tokens.weight],
                    "lr": config.embed_lr,
                    "weight_decay": 0.0,
                },
                {"params": scalars, "lr": config.scalar_lr, "weight_decay": 0.0},
            ],
            betas=config.betas,
            eps=config.eps,
            fused=config.fused and torch.device(device).type == "cuda",
        )
        return OptimizerSet(muon, adam)

    def loss(self, model, ids, clean, targets, config):
        from model.convert.native import (
            causal_document_block_mask,
            dense_causal_document_mask,
            dense_document_mask,
            document_block_mask,
        )

        mask = None
        causal = config.objective == "clm"
        if config.mask_documents:
            blocks = ids.is_cuda and not config.dense_mask
            if causal:
                build = causal_document_block_mask if blocks else dense_causal_document_mask
            else:
                build = document_block_mask if blocks else dense_document_mask
            mask = build(clean, config.separator_id)
        if causal and mask is None:
            hidden = model.hidden(ids, None, causal=True)
            selected = targets != IGNORE
            if not selected.any():
                return hidden.sum().float() * 0.0
            logits = model.logits(hidden[selected])
            loss = _masked_cross_entropy(logits, targets[selected])
            if config.z_loss > 0:
                loss = loss + config.z_loss * (torch.logsumexp(logits.float(), dim=-1) ** 2).mean()
            return loss
        hidden = model.hidden(ids, mask)
        selected = targets != IGNORE
        if not selected.any():
            return hidden.sum().float() * 0.0
        logits = model.logits(hidden[selected])
        loss = _masked_cross_entropy(logits, targets[selected])
        if config.z_loss > 0:
            # The backbone's own guard on the size of its logits, kept at the
            # weight it was pretrained with.
            loss = loss + config.z_loss * (torch.logsumexp(logits.float(), dim=-1) ** 2).mean()
        return loss


ENGINES: dict[str, Engine] = {"library": LibraryEngine(), "native": NativeEngine()}
