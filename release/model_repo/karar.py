"""ufakzeka-karar on CPU: Turkish text and typed questions in, calibrated answers out.

    import json

    from karar import Karar

    karar = Karar.from_pretrained("path/to/this/folder")  # or a Hugging Face repo id
    result = karar.decide(
        "Acil! Paketiniz adres hatası yüzünden merkezimizde bekliyor. "
        "24 saat içinde güncelleme yapmazsanız iade edilecek. [bağlantı]",
        {
            "mesaj_turu": {
                "type": "choice",
                "instructions": "Bu mesaj ne tür bir mesaj?",
                "criteria": {
                    "işlem bildirimi": "Bir şirketin müşterisine yaptığı işlemle ilgili bildirim.",
                    "izinli pazarlama": "Müşterisi olunan bir yerden gelen kampanya ya da duyuru.",
                    "istenmeyen reklam": "İlişki kurulmamış bir yerden gelen izinsiz reklam.",
                    "dolandırıcılık": "Kişiyi para, şifre ya da bilgi vermeye kandırma girişimi.",
                    "kişisel yazışma": "İki kişi arasındaki gündelik mesaj.",
                },
            },
            "oltalama": {
                "type": "noul",
                "instructions": "Bu mesaj, kişiyi para, şifre, doğrulama kodu ya da kişisel bilgi "
                "vermeye kandırmaya çalışan bir dolandırıcılık girişimi mi?",
                "criteria": {
                    "true": "Evet, mesaj bir dolandırıcılık girişimi.",
                    "false": "Hayır, mesaj bir dolandırıcılık girişimi değil.",
                },
            },
        },
    )


    def rounded(value):
        if isinstance(value, float):
            return round(value, 3)
        if isinstance(value, dict):
            return {key: rounded(v) for key, v in value.items()}
        return value


    print(json.dumps(rounded(result), ensure_ascii=False, indent=2))

The example is a HakemBench open-set item (spam-9155fdca2238bef7), one of the
karar demo's examples; its gold answers are dolandırıcılık and yes. It prints
(on CPU, from the released folder):

    {
      "model": "ufakzeka-karar",
      "answers": {
        "mesaj_turu": {
          "type": "choice",
          "choice": "dolandırıcılık",
          "probabilities": {
            "işlem bildirimi": 0.175,
            "izinli pazarlama": 0.037,
            "istenmeyen reklam": 0.016,
            "dolandırıcılık": 0.763,
            "kişisel yazışma": 0.01
          },
          "confidence": 0.748,
          "abstain": 0.252
        },
        "oltalama": {
          "type": "noul",
          "noul": 0.955,
          "abstain": 0.055
        }
      }
    }

A request is a state (a string, or any JSON object or array) and a map of named
questions in the typed-decision format: choice (one option out of a set, 2 to
255 options), score (a level on an ordered rubric, 2 to 10 levels) and noul
(yes or no). The answer map uses the same names. Each question is read in one
forward pass and nothing is generated; a choice question with more than ten
options takes one pass per ten, and one softmax runs over all of their logits.

What an answer holds:

- probabilities: the option logits divided by the question type's temperature
  from calibrator.json, then the softmax;
- abstain: the expected error given the raw maximum probability (the one before
  the temperature), read from the calibrator's isotonic map;
- confidence, on choice and score answers: 1 minus that expected error. A noul
  answer carries the probability of yes and the abstain value only.

The calibrator belongs to one set of weights, so loading refuses a
model.safetensors whose sha256 differs from the one calibrator.json names.

This file holds the network and the input packing of the ufakzeka-karar
repository (the backbone, the decision head, the packing, the chunked choice
path and the calibrated answer), copied so that it runs without that
repository; the tests there hold the copy to the same outputs as the original.

Requirements (requirements.txt): torch, numpy, safetensors, tokenizers, and
huggingface_hub for a repo id. License: Apache-2.0.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["Karar", "Calibration"]

FORMAT_VERSION = 1
MAX_CHOICE_OPTIONS = 255
MIN_CHOICE_OPTIONS = 2
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

# Segment ids: 0 for the prefix, i for option i (from 1), -1 for padding.
PREFIX = 0
PAD = -1


# ---------------------------------------------------------------------------
# Questions: the checks of the typed-decision contract, on plain dicts.
# ---------------------------------------------------------------------------


def _check_text(text: str, what: str) -> None:
    """A string the tokenizer can read: UTF-8, so no lone surrogate."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{what} is not valid Unicode text (it holds a lone surrogate)") from None


def _check_json(value: Any, what: str) -> None:
    if not isinstance(value, str | dict | list):
        raise ValueError(f"{what} must be a string, a JSON object or a JSON array")
    try:
        text = text_of(value)
    except (TypeError, ValueError) as exc:
        # A set, a key that is not a string beside one that is, a circular reference.
        raise ValueError(f"{what} is not JSON: {exc}") from None
    _check_text(text, what)


def _check_question(qid: str, q: Any) -> None:
    if not isinstance(q, dict):
        raise ValueError(f"{qid}: a question is a JSON object")
    kind = q.get("type")
    allowed = {"type", "instructions", "criteria"}
    if set(q) - allowed:
        raise ValueError(f"{qid}: unknown fields {sorted(set(q) - allowed)}")
    if "instructions" not in q:
        raise ValueError(f"{qid}: instructions are required")
    _check_json(q["instructions"], f"{qid}: instructions")
    criteria = q.get("criteria")
    if kind == "noul":
        if criteria is None:
            return
        if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
            raise ValueError(f"{qid}: noul criteria take only 'true' and 'false'")
        if any(v is not None and not isinstance(v, str) for v in criteria.values()):
            raise ValueError(f"{qid}: noul criteria are strings or null")
        for key, text in criteria.items():
            if text is not None:
                _check_text(text, f"{qid}: the {key} description")
    elif kind == "choice":
        if not isinstance(criteria, dict):
            raise ValueError(f"{qid}: choice criteria map each option to a description or null")
        if not MIN_CHOICE_OPTIONS <= len(criteria) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"{qid}: a choice question takes {MIN_CHOICE_OPTIONS} to "
                             f"{MAX_CHOICE_OPTIONS} options, got {len(criteria)}")  # fmt: skip
        if any(not isinstance(option, str) for option in criteria):
            raise ValueError(f"{qid}: option names must be strings")
        if any(not option.strip() for option in criteria):
            raise ValueError(f"{qid}: option names must not be empty")
        if any(v is not None and not isinstance(v, str) for v in criteria.values()):
            raise ValueError(f"{qid}: option descriptions are strings or null")
        for option, text in criteria.items():
            _check_text(option, f"{qid}: option {option!r}")
            if text is not None:
                _check_text(text, f"{qid}: the description of option {option!r}")
    elif kind == "score":
        if not isinstance(criteria, list) or not all(isinstance(v, str) for v in criteria):
            raise ValueError(f"{qid}: score criteria are a list of level descriptions")
        if not MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS:
            raise ValueError(f"{qid}: a score question takes {MIN_SCORE_LEVELS} to "
                             f"{MAX_SCORE_LEVELS} levels, got {len(criteria)}")  # fmt: skip
        if any(not level.strip() for level in criteria):
            raise ValueError(f"{qid}: level descriptions must not be empty")
        for level, text in enumerate(criteria):
            _check_text(text, f"{qid}: level {level}")
    else:
        raise ValueError(f"{qid}: unknown question type {kind!r}")


def text_of(value: Any) -> str:
    """A string as it is; JSON as one canonical line, keys sorted."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def outcomes(q: dict) -> list[str]:
    """The answer keys of a question, in the question's own order."""
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "score":
        return [str(level) for level in range(len(q["criteria"]))]
    return ["true", "false"]


def expected_level(probabilities: dict[str, float]) -> float:
    return math.fsum(int(level) * p for level, p in probabilities.items())


# ---------------------------------------------------------------------------
# Packing: one question as one sequence.
#
#     [state and question: the prefix][option 1][option 2] ... [option n]
#
# Every option starts at the position after the prefix, attends to the prefix
# and to itself only, and options are packed in the order of their token ids,
# so the answer does not depend on the order the caller lists them in.
# ---------------------------------------------------------------------------


def option_texts(q: dict) -> dict[str, str]:
    if q["type"] == "choice":
        return {name: f"Seçenek: {name}" + (f". {text}" if text else "")
                for name, text in q["criteria"].items()}  # fmt: skip
    if q["type"] == "noul":
        criteria = q.get("criteria") or {}
        yes = criteria.get("true") or ""
        no = criteria.get("false") or ""
        return {"true": "Cevap: evet" + (f". {yes}" if yes else ""),
                "false": "Cevap: hayır" + (f". {no}" if no else "")}  # fmt: skip
    return {str(k): f"Düzey {k}: {text}" for k, text in enumerate(q["criteria"])}


@dataclass(frozen=True)
class Packed:
    ids: list[int]
    positions: list[int]
    segments: list[int]
    keys: list[str]
    segment_of: dict[str, int]


def pack(state: Any, q: dict, encode: Callable[[str], list[int]],
         max_prefix: int = 448, max_option: int = 48) -> Packed:  # fmt: skip
    """A long prefix loses the end of its state first; it keeps the question, up to half of
    the prefix (a longer question loses its own end)."""
    instructions = text_of(q["instructions"])
    prefix = encode(f"{text_of(state)}\n\nSoru: {instructions}")
    if len(prefix) > max_prefix:
        asked = encode(f"\n\nSoru: {instructions}")[: max_prefix // 2]
        prefix = encode(text_of(state))[: max_prefix - len(asked)] + asked
    texts = option_texts(q)
    keys = outcomes(q)
    encoded = {key: encode("\n" + texts[key])[:max_option] for key in keys}
    for key, tokens in encoded.items():
        if not tokens:
            raise ValueError(f"option {key!r} encodes to no tokens")
    if len({tuple(encoded[k]) for k in keys}) != len(keys):
        raise ValueError("two options encode to the same tokens and could not be told apart")
    laid = sorted(keys, key=lambda key: (encoded[key], key))
    n = len(prefix)
    ids, positions, segments = list(prefix), list(range(n)), [PREFIX] * n
    segment_of = {}
    for index, key in enumerate(laid, start=1):
        tokens = encoded[key]
        ids += tokens
        positions += list(range(n, n + len(tokens)))
        segments += [index] * len(tokens)
        segment_of[key] = index
    return Packed(ids, positions, segments, keys, segment_of)


@dataclass
class Batch:
    ids: torch.Tensor  # (1, T)
    positions: torch.Tensor  # (1, T)
    segments: torch.Tensor  # (1, T)
    option_segments: torch.Tensor  # (1, N): slot j pools segment j + 1
    option_valid: torch.Tensor  # (1, N)
    caller_slot: torch.Tensor  # (1, N): the slot that scores the caller's k-th outcome
    keys: list[str]


def collate(p: Packed, device) -> Batch:
    count = len(p.keys)
    caller = [p.segment_of[key] - 1 for key in p.keys]
    return Batch(
        ids=torch.tensor([p.ids], dtype=torch.long, device=device),
        positions=torch.tensor([p.positions], dtype=torch.long, device=device),
        segments=torch.tensor([p.segments], dtype=torch.long, device=device),
        option_segments=torch.arange(1, count + 1, device=device)[None],
        option_valid=torch.ones((1, count), dtype=torch.bool, device=device),
        caller_slot=torch.tensor([caller], dtype=torch.long, device=device),
        keys=list(p.keys),
    )


def attention_mask(segments: torch.Tensor, causal: bool) -> torch.Tensor:
    """(batch, 1, T, T), True where allowed: the prefix sees itself, an option the prefix and
    itself, nothing sees another option; with `causal` only forward inside each part."""
    q = segments[:, :, None]
    k = segments[:, None, :]
    real_q = q != PAD
    allowed = real_q & ((k == PREFIX) | (k == q)) & (k != PAD)
    if causal:
        index = torch.arange(segments.shape[1], device=segments.device)
        allowed = allowed & (index[None, :, None] >= index[None, None, :])
    eye = torch.eye(segments.shape[1], dtype=torch.bool, device=segments.device)[None]
    allowed = allowed | (eye & ~real_q)
    return allowed[:, None, :, :]


# ---------------------------------------------------------------------------
# The backbone (ufakzeka-1-base's own network) and the decision head.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackboneConfig:
    vocab_size: int
    n_layer: int
    d_model: int
    n_head: int
    n_kv_head: int
    d_ff: int
    head_dim: int
    max_positions: int
    rope_theta: float
    norm_eps: float
    softcap: float


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
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        hd = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * hd, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * hd, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * hd, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * hd, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(hd, cfg.norm_eps)
        self.k_norm = RMSNorm(hd, cfg.norm_eps)

    def forward(self, x, cos, sin, mask):
        batch, length, _ = x.shape
        hd, heads, kv = self.cfg.head_dim, self.cfg.n_head, self.cfg.n_kv_head
        q = self.q_norm(self.q_proj(x).view(batch, length, heads, hd).transpose(1, 2))
        k = self.k_norm(self.k_proj(x).view(batch, length, kv, hd).transpose(1, 2))
        v = self.v_proj(x).view(batch, length, kv, hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False,
                                           enable_gqa=True)  # fmt: skip
        return self.o_proj(y.transpose(1, 2).reshape(batch, length, heads * hd))


class MLP(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class Body(nn.Module):
    def __init__(self, cfg: BackboneConfig):
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

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor):
        full_cos, full_sin = self._rope(int(positions.max()) + 1, ids.device)
        cos, sin = full_cos[positions].unsqueeze(1), full_sin[positions].unsqueeze(1)
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.norm(x)


class Backbone(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.body = Body(cfg)


def pool(hidden: torch.Tensor, segments: torch.Tensor, which: torch.Tensor, how: str):
    """For each (example, slot), pool the tokens whose segment equals `which`."""
    member = segments[:, None, :] == which[:, :, None]  # (B, N, T)
    if how == "mean":
        weights = member.to(hidden.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1.0)
        return weights @ hidden
    if how == "last":
        index = torch.arange(segments.shape[1], device=segments.device)
        last = torch.where(member, index, torch.full_like(index, -1)).amax(-1)
        gathered = hidden.gather(1, last.clamp(min=0)[..., None].expand(-1, -1, hidden.shape[-1]))
        return gathered * (last >= 0)[..., None].to(hidden.dtype)
    raise ValueError(f"unknown pooling {how!r}")


class DecisionHead(nn.Module):
    """One score per option from its pooled tokens; the scores meet only in the softmax."""

    def __init__(self, cfg: BackboneConfig, *, causal: bool, pooling: str):
        super().__init__()
        self.backbone = Backbone(cfg)
        self.causal = causal
        self.pooling = pooling
        self.score = nn.Linear(cfg.d_model, 1)
        self.abstain = nn.Linear(cfg.d_model, 1)

    def forward(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        mask = attention_mask(batch.segments, causal=self.causal)
        hidden = self.backbone.body(batch.ids, mask, batch.positions)
        options = pool(hidden, batch.segments, batch.option_segments, self.pooling)
        logits = self.score(options).squeeze(-1).float()
        logits = logits.masked_fill(~batch.option_valid, float("-inf"))
        logits = logits.gather(1, batch.caller_slot)
        prefix_slot = torch.full_like(batch.option_segments[:, :1], PREFIX)
        prefix = pool(hidden, batch.segments, prefix_slot, self.pooling).squeeze(1)
        abstain = self.abstain(prefix).squeeze(-1).float()
        return logits, abstain


# ---------------------------------------------------------------------------
# Calibration: per-type temperatures and the abstain map.
# ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _softmax(values: np.ndarray) -> np.ndarray:
    e = np.exp(values - values.max())
    return e / e.sum()


@dataclass(frozen=True)
class Calibration:
    temperature: float
    map_x: tuple[float, ...]
    map_y: tuple[float, ...]
    by_type: tuple[tuple[str, float], ...] = ()

    @classmethod
    def from_dict(cls, data: dict, weights_sha256: str | None = None) -> Calibration:
        if data.get("engine") != "fp32":
            raise ValueError(f"the calibrator was fitted on {data.get('engine')!r}, not fp32")
        if weights_sha256 is not None and data.get("weights_sha256") != weights_sha256:
            raise ValueError("the calibrator was fitted on other weights: its weights_sha256 "
                             "differs from the sha256 of model.safetensors")  # fmt: skip
        temperatures = data["temperatures"]
        form = temperatures.get("form")
        if form not in ("global", "type"):
            raise ValueError("only the global and per-type temperature forms are supported")
        by_type = tuple(sorted((k, float(v)) for k, v in temperatures.get("type", {}).items())
                        ) if form == "type" else ()  # fmt: skip
        out = cls(temperature=float(temperatures["global"]), by_type=by_type,
                  map_x=tuple(float(v) for v in data["abstain_map"]["x"]),
                  map_y=tuple(float(v) for v in data["abstain_map"]["y"]))  # fmt: skip
        for t in (out.temperature, *(v for _, v in out.by_type)):
            if not (math.isfinite(t) and t > 0):
                raise ValueError(f"the temperature must be positive, got {t!r}")
        if len(out.map_x) != len(out.map_y) or not out.map_x:
            raise ValueError("the abstain map needs as many y values as x values, at least one")
        if any(b < a for a, b in zip(out.map_x, out.map_x[1:], strict=False)):
            raise ValueError("the abstain map's x must be ascending")
        if any(not 0.0 <= y <= 1.0 for y in out.map_y):
            raise ValueError("the abstain map's y must lie in 0 to 1")
        return out

    def expected_error(self, raw_max_probability: float) -> float:
        return float(np.interp(raw_max_probability, self.map_x, self.map_y))

    def temperature_for(self, question_type: str) -> float:
        return dict(self.by_type).get(question_type, self.temperature)

    def probabilities(self, logits: list[float], question_type: str) -> np.ndarray:
        x = np.asarray(logits, dtype=np.float64) / self.temperature_for(question_type)
        return _softmax(x)


# ---------------------------------------------------------------------------
# The model.
# ---------------------------------------------------------------------------


def _local_folder(path_or_repo: str | Path, revision: str | None) -> Path:
    path = Path(path_or_repo)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(path_or_repo), revision=revision))


class Karar:
    """ufakzeka-karar: one forward pass per question (per ten options), no generation."""

    def __init__(self, head: DecisionHead, encode: Callable[[str], list[int]],
                 calibration: Calibration, config: dict, device: str = "cpu") -> None:  # fmt: skip
        self.head = head.to(device).eval()
        self.encode = encode
        self.calibration = calibration
        self.config = config
        self.device = torch.device(device)
        self.model_id = config.get("model_id", "ufakzeka-karar")
        self.max_prefix = int(config.get("max_prefix_tokens", 448))
        self.max_option = int(config.get("max_option_tokens", 48))
        self.per_pass = int(config.get("options_per_pass", 10))

    @classmethod
    def from_pretrained(cls, path_or_repo: str | Path, device: str = "cpu",
                        revision: str | None = None) -> Karar:  # fmt: skip
        from safetensors.torch import load_file
        from tokenizers import Tokenizer

        folder = _local_folder(path_or_repo, revision)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if config.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"config.json has format {config.get('format_version')!r}, "
                             f"this file reads {FORMAT_VERSION}")  # fmt: skip
        if config.get("layout", "blind") != "blind":
            raise ValueError("only the blind option layout is supported")
        weights = folder / "model.safetensors"
        calibration = Calibration.from_dict(
            json.loads((folder / "calibrator.json").read_text(encoding="utf-8")),
            weights_sha256=sha256(weights),
        )
        with torch.device("meta"):
            head = DecisionHead(
                BackboneConfig(**config["backbone_config"]),
                causal=bool(config["causal"]),
                pooling=config["pooling"],
            )
        # A strict load: weights that do not fit fail here, none left at an initial value.
        head.load_state_dict(load_file(weights, device="cpu"), strict=True, assign=True)
        head.to(torch.float32)
        tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))

        def encode(text: str) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=False).ids

        return cls(head, encode, calibration, config, device=device)

    def _run(self, state: Any, q: dict) -> list[float]:
        batch = collate(pack(state, q, self.encode, self.max_prefix, self.max_option), self.device)
        with torch.no_grad():
            out, _ = self.head(batch)
        return out[0].float().tolist()

    def logits(self, state: Any, q: dict) -> list[float]:
        """The outcome logits in the answer's key order, before any temperature."""
        if q["type"] != "choice":
            by_key = dict(zip(outcomes(q), self._run(state, q), strict=True))
            if q["type"] == "noul":
                return [by_key["true"], by_key["false"]]
            return [by_key[str(k)] for k in range(len(q["criteria"]))]
        # Passes of at most ten in the canonical order; an option's logit is the same in any
        # pass that holds it, so one softmax runs over all of them.
        options = list(q["criteria"])
        keys = sorted(options, key=lambda key: (self.encode("\n" + key), key))
        passes = [keys[i : i + self.per_pass] for i in range(0, len(keys), self.per_pass)]
        if len(passes) > 1:
            # Two options that encode alike are refused in one pass (pack); check them all
            # together first, so a pair split across passes is refused as well.
            texts = option_texts(q)
            encoded = [tuple(self.encode("\n" + texts[key])[: self.max_option]) for key in keys]
            if not all(encoded):
                raise ValueError("an option encodes to no tokens")
            if len(set(encoded)) != len(encoded):
                raise ValueError(
                    "two options encode to the same tokens and could not be told apart"
                )
        if len(passes) > 1 and len(passes[-1]) == 1:
            passes[-1] = [passes[-2][-1], *passes[-1]]
        found: dict[str, float] = {}
        for keys_in_pass in passes:
            sub = {"type": "choice", "instructions": q["instructions"],
                   "criteria": {key: q["criteria"][key] for key in keys_in_pass}}  # fmt: skip
            for key, value in zip(keys_in_pass, self._run(state, sub), strict=True):
                found.setdefault(key, value)
        return [found[key] for key in options]

    def _answer(self, q: dict, logits: list[float]) -> dict:
        cal = self.calibration
        raw = _softmax(np.asarray(logits, dtype=np.float64))
        probs = cal.probabilities(logits, q["type"])
        error = cal.expected_error(float(raw.max()))
        if q["type"] == "noul":
            return {"type": "noul", "noul": float(probs[0]), "abstain": error}
        if q["type"] == "choice":
            keys = list(q["criteria"])
            return {"type": "choice", "choice": keys[int(np.argmax(probs))],
                    "probabilities": dict(zip(keys, probs.tolist(), strict=True)),
                    "confidence": 1.0 - error, "abstain": error}  # fmt: skip
        distribution = {str(k): float(p) for k, p in enumerate(probs)}
        level = min(max(expected_level(distribution), 0.0), len(q["criteria"]) - 1.0)
        return {"type": "score", "score": level,
                "legend": {str(k): text for k, text in enumerate(q["criteria"])},
                "probabilities": distribution, "confidence": 1.0 - error,
                "abstain": error}  # fmt: skip

    def decide(self, state: Any, questions: dict[str, dict]) -> dict:
        """Answer every named question about `state`; returns {"model", "answers"}."""
        _check_json(state, "state")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions is a non-empty map of question id to question")
        for qid, q in questions.items():
            if not isinstance(qid, str):
                raise ValueError(f"question ids must be strings, got {type(qid).__name__}")
            if not qid.strip():
                raise ValueError("question ids must not be empty")
            _check_text(qid, "a question id")
            _check_question(qid, q)
        answers = {}
        for qid, q in questions.items():
            logits = self.logits(state, q)
            if not all(math.isfinite(v) for v in logits):
                raise ValueError(f"{qid}: the model returned a non-finite logit")
            answers[qid] = self._answer(q, logits)
        return {"model": self.model_id, "answers": answers}
