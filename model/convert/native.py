"""The backbone's own network, for the conversion to run on.

ufakzeka-1-base was pretrained with a small hand-written transformer, not with
the library class its published weights load into. That network trained at
224,000 tokens a second on one H100 where the library path measured 115,000 on
the same card, because its forward compiles as one graph and its packed
document mask is a block mask the attention kernel skips through instead of a
dense matrix. Running the conversion on it costs about half as much per token.

It is also the more faithful host. The backbone was trained with its logits soft-capped at 30, which
the published config does not carry and the library path therefore never applied, and with Muon on
the block matrices, which is the optimizer the conversion was asked to match.

This file is that network, taken from the ufakzeka repository (Apache-2.0, the
lab's own), with the training-time extras removed and one change: attention is
not causal. The parameter names are the published checkpoint's without the
`model.` prefix, so weights move in either direction by renaming keys, and
`tests/test_native.py` holds the two implementations to the same outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

PREFIX = "model."


@dataclass(frozen=True)
class NativeConfig:
    vocab_size: int = 40960
    n_layer: int = 24
    d_model: int = 768
    n_head: int = 12
    n_kv_head: int = 4
    d_ff: int = 2048
    head_dim: int = 64
    max_positions: int = 4096
    rope_theta: float = 100_000.0
    norm_eps: float = 1e-5
    # Applied in pretraining, absent from the published config.
    softcap: float = 30.0

    @classmethod
    def from_hf(cls, config, softcap: float = 30.0) -> NativeConfig:
        """Read the shape from the published config, refusing what this file lacks."""
        if not getattr(config, "tie_word_embeddings", False):
            raise ValueError("this network ties its embeddings and the config does not")
        if getattr(config, "attention_bias", False):
            raise ValueError("this network has no attention bias and the config asks for one")
        if getattr(config, "use_sliding_window", False):
            raise ValueError("this network has no sliding-window layers")
        rope = getattr(config, "rope_parameters", None) or {}
        theta = rope.get("rope_theta") or getattr(config, "rope_theta", None)
        if theta is None:
            raise ValueError("the config gives no rope theta")
        if rope.get("rope_type", "default") != "default":
            raise ValueError(f"unsupported rope type {rope.get('rope_type')}")
        return cls(
            vocab_size=config.vocab_size,
            n_layer=config.num_hidden_layers,
            d_model=config.hidden_size,
            n_head=config.num_attention_heads,
            n_kv_head=config.num_key_value_heads,
            d_ff=config.intermediate_size,
            head_dim=getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads,
            max_positions=config.max_position_embeddings,
            rope_theta=float(theta),
            norm_eps=config.rms_norm_eps,
            softcap=softcap,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).type_as(x)


def rope_cache(length: int, head_dim: int, theta: float, device) -> tuple[torch.Tensor, ...]:
    inverse = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(length, device=device).float(), inverse)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return (x * cos + rotated * sin).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: NativeConfig):
        super().__init__()
        self.cfg = cfg
        hd = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * hd, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * hd, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * hd, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * hd, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(hd, cfg.norm_eps)
        self.k_norm = RMSNorm(hd, cfg.norm_eps)

    def forward(self, x, cos, sin, mask=None, causal: bool = False):
        batch, length, _ = x.shape
        hd, heads, kv = self.cfg.head_dim, self.cfg.n_head, self.cfg.n_kv_head
        q = self.q_norm(self.q_proj(x).view(batch, length, heads, hd).transpose(1, 2))
        k = self.k_norm(self.k_proj(x).view(batch, length, kv, hd).transpose(1, 2))
        v = self.v_proj(x).view(batch, length, kv, hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if mask is None or isinstance(mask, torch.Tensor):
            # No mask is every position seeing every other, which is the fast
            # kernel. A dense boolean mask is for tests and small inputs.
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=causal and mask is None, enable_gqa=True
            )
        else:
            from torch.nn.attention.flex_attention import flex_attention

            y = flex_attention(q, k, v, block_mask=mask, enable_gqa=True)
        return self.o_proj(y.transpose(1, 2).reshape(batch, length, heads * hd))


class MLP(nn.Module):
    def __init__(self, cfg: NativeConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: NativeConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, mask=None, causal: bool = False):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask, causal)
        return x + self.mlp(self.post_attention_layernorm(x))


class Body(nn.Module):
    """Embeddings to final norm. The part that compiles as one graph."""

    def __init__(self, cfg: NativeConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)

    def _rope(self, length: int, device):
        stale = self.rope_cos is None or self.rope_cos.device != device
        if stale or self.rope_cos.shape[0] < length:
            reach = max(length, self.cfg.max_positions)
            self.rope_cos, self.rope_sin = rope_cache(
                reach, self.cfg.head_dim, self.cfg.rope_theta, device
            )
        return self.rope_cos[:length], self.rope_sin[:length]

    def forward(
        self,
        ids: torch.Tensor,
        mask=None,
        causal: bool = False,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`positions`, shape (batch, length), gives each token its rotary position.

        Left out, token i sits at position i, which is how the backbone was
        trained. The decision head passes its own, because every option of a
        question starts at the same position: an option's encoding must
        not depend on where in the sequence it happened to be packed.
        """
        if positions is None:
            cos, sin = self._rope(ids.shape[1], ids.device)
        else:
            full_cos, full_sin = self._rope(int(positions.max()) + 1, ids.device)
            # (batch, 1, length, head_dim), broadcast over the heads.
            cos, sin = full_cos[positions].unsqueeze(1), full_sin[positions].unsqueeze(1)
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = layer(x, cos, sin, mask, causal)
        return self.norm(x)


class NativeModel(nn.Module):
    """The body and its tied head, kept apart so the head runs on few positions."""

    def __init__(self, cfg: NativeConfig):
        super().__init__()
        self.cfg = cfg
        self.body = Body(cfg)

    def hidden(
        self, ids: torch.Tensor, mask=None, causal: bool = False, positions=None
    ) -> torch.Tensor:
        return self.body(ids, mask, causal, positions)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Tied projection, soft-capped the way the backbone was trained."""
        out = F.linear(hidden, self.body.embed_tokens.weight)
        if self.cfg.softcap > 0:
            out = self.cfg.softcap * torch.tanh(out.float() / self.cfg.softcap)
        return out

    def load_hf_state(self, state: dict[str, torch.Tensor]) -> None:
        """Take the published checkpoint's weights. Every key must be used."""
        wanted = self.body.state_dict()
        renamed = {
            key[len(PREFIX) :]: value for key, value in state.items() if key.startswith(PREFIX)
        }
        extra = set(state) - {PREFIX + key for key in renamed} - {"lm_head.weight"}
        missing = set(wanted) - set(renamed)
        if extra or missing:
            raise ValueError(
                f"weights do not fit: missing {sorted(missing)}, extra {sorted(extra)}"
            )
        self.body.load_state_dict({key: renamed[key] for key in wanted})

    def hf_state(self) -> dict[str, torch.Tensor]:
        """The weights under the published names, for the library class to load."""
        return {PREFIX + key: value for key, value in self.body.state_dict().items()}


def document_block_mask(clean_ids: torch.Tensor, separator_id: int):
    """Both directions inside a document, nothing across one.

    The backbone's own mask with the causal condition removed. A separator
    belongs to the document it closes. Built outside the compiled forward, once
    per batch, from the uncorrupted ids.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    batch, length = clean_ids.shape
    document = document_index(clean_ids, separator_id)

    def same_document(b, h, q_idx, kv_idx):
        return document[b, q_idx] == document[b, kv_idx]

    return create_block_mask(
        same_document,
        batch,
        None,
        length,
        length,
        device=str(clean_ids.device),
        _compile=clean_ids.is_cuda,
    )


def document_index(clean_ids: torch.Tensor, separator_id: int) -> torch.Tensor:
    is_separator = (clean_ids == separator_id).to(torch.int32)
    return torch.cumsum(is_separator, dim=1) - is_separator


def dense_document_mask(clean_ids: torch.Tensor, separator_id: int) -> torch.Tensor:
    """The same mask as a boolean matrix, for tests and for CPU."""
    document = document_index(clean_ids, separator_id)
    return (document[:, :, None] == document[:, None, :])[:, None, :, :]


def causal_document_block_mask(clean_ids: torch.Tensor, separator_id: int):
    """The backbone's own pretraining mask: causal, and nothing across a document."""
    from torch.nn.attention.flex_attention import create_block_mask

    batch, length = clean_ids.shape
    document = document_index(clean_ids, separator_id)

    def causal_same_document(b, h, q_idx, kv_idx):
        return (document[b, q_idx] == document[b, kv_idx]) & (q_idx >= kv_idx)

    return create_block_mask(
        causal_same_document,
        batch,
        None,
        length,
        length,
        device=str(clean_ids.device),
        _compile=clean_ids.is_cuda,
    )


def dense_causal_document_mask(clean_ids: torch.Tensor, separator_id: int) -> torch.Tensor:
    """The causal document mask as a boolean matrix, for tests and for CPU."""
    length = clean_ids.shape[1]
    order = torch.arange(length, device=clean_ids.device)
    causal = order[:, None] >= order[None, :]
    return dense_document_mask(clean_ids, separator_id) & causal[None, None]


def from_pretrained(backbone: str, revision: str | None) -> tuple[NativeModel, object]:
    """The published weights in this network, in float32, and the published config."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(backbone, revision=revision)
    library = AutoModelForCausalLM.from_pretrained(backbone, revision=revision, dtype=torch.float32)
    model = NativeModel(NativeConfig.from_hf(config))
    model.load_hf_state(library.state_dict())
    return model, config


def load_checkpoint_into_library(state: dict[str, torch.Tensor], library) -> None:
    """Put a training checkpoint's weights into the library class, whichever engine wrote it.

    The native engine saves under `body.` names and the library engine under
    the published ones. Either way every weight must land and none may be left
    over, because a model that loads with some layers still at their published
    values evaluates without an error and measures the wrong thing.
    """
    if any(key.startswith("body.") for key in state):
        native = NativeModel(NativeConfig.from_hf(library.config))
        native.load_state_dict(state)
        state = native.hf_state()
    missing, unexpected = library.load_state_dict(state, strict=False)
    missing = [key for key in missing if key != "lm_head.weight"]
    if missing or unexpected:
        raise ValueError(f"checkpoint does not fit: missing {missing}, unexpected {unexpected}")
