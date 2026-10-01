"""The decision head on a Turkish encoder, for the encoder comparison.

The owner's call: train the same head on the three encoders the
instrument compared against (BERTurk, TabiBERT, MoganBERT-TR) so the comparison
with our backbone is measured with one head, one data mix and one protocol.
They stay baseline rows, never the shipped base.

An encoder is wrapped to look like the native backbone to the head: `cfg.d_model`
and `hidden(ids, mask, positions)`, where `mask` is the head's (batch, 1, T, T)
boolean mask (True where attention is allowed) and `positions` its shared
positions. Whether a library model honours a custom four-dimensional mask and
explicit positions is not assumed: `check_blindness` runs the head's two
guarantees on the wrapped model, that adding an option leaves the other
options' scores alone and that option order changes nothing, and a model that
fails is not trained.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

ENCODERS = {
    "berturk": ("dbmdz/bert-base-turkish-cased", "b6e1de16c983e0f2c70664591ea3f22810072608"),
    "tabibert": ("boun-tabilab/TabiBERT", "36d24bfae8d67f7b1f7d0827fc43076e4a1c3e9f"),
    "moganbert": ("moganai/MoganBERT-TR", "2614f39447ba72a3b2917b59c920a4844ff96cdd"),
}


class EncoderBackbone(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.cfg = SimpleNamespace(d_model=model.config.hidden_size)

    def additive(self, mask: torch.Tensor) -> torch.Tensor:
        """Zero where attention is allowed, the dtype's lowest value where it is not."""
        dtype = self.model.dtype
        out = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
        return out.masked_fill(~mask, torch.finfo(dtype).min)

    def hidden(self, ids: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor):
        config = self.model.config
        attention = self.additive(mask)
        if "sliding_attention" in (getattr(config, "layer_types", None) or ()):
            # ModernBERT's local layers were pretrained to see a window of
            # config.sliding_window positions each side; a custom mask replaces
            # its own, so the window is laid on the head's mask here, on the
            # shared positions.
            band = (positions[:, :, None] - positions[:, None, :]).abs() <= config.sliding_window
            attention = {"full_attention": attention,
                         "sliding_attention": self.additive(mask & band[:, None])}  # fmt: skip
        out = self.model(input_ids=ids, attention_mask=attention, position_ids=positions)
        return out.last_hidden_state


def load_encoder(name: str):
    from transformers import AutoModel, AutoTokenizer

    repo, revision = ENCODERS[name]
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    # SDPA takes the custom four-dimensional float mask on both families and,
    # unlike eager attention, does not keep a (batch, heads, T, T) score matrix
    # per layer, which at the round's longest rows would not fit an L4.
    model = AutoModel.from_pretrained(repo, revision=revision, attn_implementation="sdpa",
                                      dtype=torch.float32)  # fmt: skip
    return EncoderBackbone(model), tokenizer


def check_blindness(head, tokenizer, atol: float = 1e-4) -> dict:
    """The head's guarantees on a wrapped encoder: order-free and blind options."""
    from pydantic import TypeAdapter

    from model.head.pack import collate, pack
    from schema.questions import Question

    adapter = TypeAdapter(Question)

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False).input_ids

    def scores(options: list[str]) -> dict[str, float]:
        q = adapter.validate_python({"type": "choice", "instructions": "Konu nedir?",
                                     "criteria": dict.fromkeys(options)})  # fmt: skip
        batch = collate([pack("Faturam iki kez kesildi, iade istiyorum.", q, encode)], 0)
        with torch.no_grad():
            logits, _ = head(batch)
        return dict(zip(batch.keys[0], logits[0].tolist(), strict=True))

    head.eval()
    # Canonical packing makes reordering trivially exact, so positions are
    # checked directly: the options' outputs under the shared positions must
    # differ from their outputs under plain sequential positions, or the model
    # ignores the positions it is given.
    q = adapter.validate_python(
        {
            "type": "choice",
            "instructions": "Konu nedir?",
            "criteria": dict.fromkeys(["fatura", "kargo", "iade"]),
        }
    )
    batch = collate([pack("Faturam iki kez kesildi, iade istiyorum.", q, encode)], 0)
    from model.head.pack import attention_mask

    mask = attention_mask(batch.segments, causal=False)
    options = batch.segments[0] > 0
    with torch.no_grad():
        shared = head.backbone.hidden(batch.ids, mask, batch.positions)[0][options]
        sequential = torch.arange(batch.ids.shape[1])[None]
        plain = head.backbone.hidden(batch.ids, mask, sequential)[0][options]
    positions_honoured = float((shared - plain).abs().max()) > 1e-3
    three = scores(["fatura", "kargo", "iade"])
    four = scores(["fatura", "kargo", "iade", "teknik destek"])
    reordered = scores(["iade", "fatura", "kargo"])
    added = max(abs(three[k] - four[k]) for k in three)
    moved = max(abs(three[k] - reordered[k]) for k in three)
    return {"adding_an_option_moves_others_by": added, "reordering_moves_scores_by": moved,
            "positions_honoured": positions_honoured,
            "passes": added <= atol and moved <= atol and positions_honoured}  # fmt: skip
