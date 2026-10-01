"""Our decision model: the decision head on the backbone, fp32 torch on CPU.

Each question is read in one forward pass and nothing is generated. A choice
question with more than ten options takes one pass per ten
(model/head/infer.py), and one softmax runs over all of their logits.

What the served answer is, as shipped in step 7:

- Probabilities: the option logits divided by one global temperature
  (results/step7/calibrator.json, 1.27 on fp32), then the softmax.
- Abstain: the expected error given the raw maximum probability, the one
  before the temperature. It is an isotonic map stored as its breakpoints x
  (ascending) and y, read with np.interp, which holds the end values outside x
  as the fitted map (out_of_bounds="clip") does.
- Confidence, on choice and score answers: 1 minus that expected error. A
  noul answer has no confidence field and carries the expected error as
  abstain only.

torch and transformers are imported inside the loading path, so this module
imports without them. Tests inject a small network and an encode function.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bench.adapters.base import AdapterError, AdapterResult, ModelInfo
from schema.api import Request, Response
from schema.questions import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion, expected_level

DEFAULT_MODEL = "ufakzeka-karar"
# The backbone every head run was trained on (model/head/modal_app.py CAUSAL).
BACKBONE = "ufakai/ufakzeka-1-base"
BACKBONE_REVISION = "f9e11eea28cbb2ba953a5628f972d416fe0c3cfe"
CALIBRATOR = Path("results/step7/calibrator.json")
# The confidence is the model's own abstain output, not a derived maximum probability.
CONFIDENCE_SOURCE = "model"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Calibration:
    """The shipped temperature and abstain map (calib/step7.py writes them)."""

    temperature: float
    map_x: tuple[float, ...]
    map_y: tuple[float, ...]
    # What the numbers were read from, for ModelInfo.
    source: str = "given"
    # One temperature per question type (noul, choice, score) when step 7 chose that form;
    # `temperature` is then the global one, used for any type the file does not name.
    by_type: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        for t in (self.temperature, *(v for _, v in self.by_type)):
            if not (math.isfinite(t) and t > 0):
                raise AdapterError(f"the temperature must be positive, got {t!r}")
        if len(self.map_x) != len(self.map_y) or not self.map_x:
            raise AdapterError("the abstain map needs as many y values as x values, at least one")
        if any(b < a for a, b in zip(self.map_x, self.map_x[1:], strict=False)):
            raise AdapterError("the abstain map's x must be ascending for np.interp")
        if any(not 0.0 <= y <= 1.0 for y in self.map_y):
            raise AdapterError("the abstain map's y must lie in 0 to 1")

    @classmethod
    def from_file(cls, path: Path, engine: str = "fp32",
                  weights_sha256: str | None = None) -> Calibration:  # fmt: skip
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("engine") != engine:
            raise AdapterError(f"{path} was fitted on {data.get('engine')!r}, not {engine!r}")
        # A temperature and abstain map belong to one checkpoint: other weights would
        # silently get the wrong ones.
        if weights_sha256 is not None and data.get("weights_sha256") != weights_sha256:
            raise AdapterError(f"{path} was fitted on {data.get('weights') or 'unrecorded'} "
                               "weights, not on the weights given")  # fmt: skip
        temperatures = data["temperatures"]
        # The global and the per-type forms are applied as fitted; a per-bucket file would be
        # applied wrongly by one number per type, so it is refused rather than guessed.
        form = temperatures.get("form")
        if form not in ("global", "type"):
            raise AdapterError(
                f"{path}: only the global and per-type temperature forms are supported"
            )
        by_type = tuple(sorted((k, float(v)) for k, v in temperatures.get("type", {}).items())
                        ) if form == "type" else ()  # fmt: skip
        return cls(
            temperature=float(temperatures["global"]),
            by_type=by_type,
            map_x=tuple(float(v) for v in data["abstain_map"]["x"]),
            map_y=tuple(float(v) for v in data["abstain_map"]["y"]),
            source=f"{path.name} sha256 {sha256(path)[:12]}",
        )

    def expected_error(self, raw_max_probability: float) -> float:
        return float(np.interp(raw_max_probability, self.map_x, self.map_y))

    def temperature_for(self, question_type: str | None) -> float:
        return dict(self.by_type).get(question_type, self.temperature)

    def probabilities(self, logits: list[float], question_type: str | None = None) -> np.ndarray:
        """softmax(logits / T), in float64, T the question type's when the file has one."""
        x = np.asarray(logits, dtype=np.float64) / self.temperature_for(question_type)
        e = np.exp(x - x.max())
        return e / e.sum()


def _softmax(logits: list[float]) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def load_model(weights: Path, backbone: str = BACKBONE, revision: str = BACKBONE_REVISION):
    """The decision head with the trained weights, fp32 on CPU, and its encode function.

    The backbone's shape and tokenizer come from the published config; every
    weight, the backbone's included, comes from `weights` (a DecisionHead
    state dict, as model/head/train.py saves it). The load is strict, so a
    file that does not fit fails here instead of answering with some layers
    left at their initial values.
    """
    import torch
    from transformers import AutoConfig, AutoTokenizer

    from model.convert.native import NativeConfig, NativeModel
    from model.head.head import DecisionHead

    config = AutoConfig.from_pretrained(backbone, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(backbone, revision=revision)
    # The shipped head (r2-base): causal backbone, mean pooling, blind layout.
    model = DecisionHead(NativeModel(NativeConfig.from_hf(config)), causal=True, pooling="mean")
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    model.to(torch.float32).eval()

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False).input_ids

    return model, encode


class KararAdapter:
    """Answers typed questions with ufakzeka-karar, one pass per question (per ten options)."""

    def __init__(
        self,
        weights: Path | None = None,
        calibrator: Path = CALIBRATOR,
        model_id: str = DEFAULT_MODEL,
        *,
        model: Any | None = None,
        encode: Callable[[str], list[int]] | None = None,
        calibration: Calibration | None = None,
        revision: str | None = None,
    ) -> None:
        """Give `weights`, or inject `model` and `encode` (tests) with a `revision` label."""
        if model is None and weights is None:
            raise AdapterError("the karar adapter needs a weights file or an injected model")
        if (model is None) != (encode is None):
            raise AdapterError("an injected model needs its encode function, and the reverse")
        self.weights = weights
        self.model_id = model_id
        self.device = "cpu"
        self._model = model
        self._encode = encode
        self._calibration = calibration
        self._calibrator_path = calibrator
        self._revision = revision

    def _ready(self) -> tuple[Any, Callable[[str], list[int]], Calibration]:
        """Model, encode and calibration, loaded on first use, before any clock starts."""
        if self._model is None:
            self._model, self._encode = load_model(self.weights)
        if self._calibration is None:
            given = sha256(self.weights) if self.weights is not None else None
            self._calibration = Calibration.from_file(self._calibrator_path,
                                                      weights_sha256=given)  # fmt: skip
        self._model.eval()
        return self._model, self._encode, self._calibration

    def _logits(self, model: Any, encode: Callable, state: Any, question: Question) -> list[float]:
        """The outcome logits in the answer's key order (schema.rows.outcomes)."""
        import torch

        from model.head.infer import chunked_logits
        from model.head.pack import collate, pack

        if isinstance(question, ChoiceQuestion):
            return list(chunked_logits(model, state, question, encode, self.device).values())
        layout = getattr(model, "layout", "blind")
        batch = model.to_device(
            collate([pack(state, question, encode, layout=layout)], model.pad_id), self.device
        )
        with torch.no_grad():
            out, _ = model(batch)
        by_key = dict(zip(batch.keys[0], out[0].float().tolist(), strict=True))
        if isinstance(question, NoulQuestion):
            return [by_key["true"], by_key["false"]]
        return [by_key[str(k)] for k in range(len(question.criteria))]

    def _answer(
        self, question: Question, logits: list[float], cal: Calibration
    ) -> tuple[dict, dict]:
        raw = _softmax(logits)
        probs = cal.probabilities(logits, question.type)
        raw_max = float(raw.max())
        error = cal.expected_error(raw_max)
        diagnostics = {"temperature": cal.temperature_for(question.type),
                       "raw_max_probability": raw_max,
                       "expected_error": error, "raw_probabilities": raw.tolist()}  # fmt: skip
        if isinstance(question, NoulQuestion):
            return {"type": "noul", "noul": float(probs[0]), "abstain": error}, diagnostics
        if isinstance(question, ChoiceQuestion):
            keys = question.options
            distribution = dict(zip(keys, probs.tolist(), strict=True))
            return {"type": "choice", "choice": keys[int(np.argmax(probs))],
                    "probabilities": distribution, "confidence": 1.0 - error,
                    "abstain": error}, diagnostics  # fmt: skip
        if isinstance(question, ScoreQuestion):
            distribution = {str(k): float(p) for k, p in enumerate(probs)}
            # The expected level can overshoot the last level by a rounding error.
            level = min(max(expected_level(distribution), 0.0), len(question.criteria) - 1.0)
            return {"type": "score", "score": level,
                    "legend": {str(k): text for k, text in enumerate(question.criteria)},
                    "probabilities": distribution, "confidence": 1.0 - error,
                    "abstain": error}, diagnostics  # fmt: skip
        raise AdapterError(f"unknown question type {type(question).__name__}")

    def answer(self, request: Request) -> AdapterResult:
        model, encode, cal = self._ready()
        answers: dict[str, Any] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        per_question_ms: dict[str, float] = {}
        confidence_source: dict[str, str] = {}

        started = time.perf_counter()
        for qid, question in request.questions.items():
            question_started = time.perf_counter()
            logits = self._logits(model, encode, request.state, question)
            if not all(math.isfinite(v) for v in logits):
                raise AdapterError(f"{qid}: the model returned a non-finite logit")
            answers[qid], diagnostics[qid] = self._answer(question, logits, cal)
            per_question_ms[qid] = (time.perf_counter() - question_started) * 1000
            # A noul answer has no confidence field, so it has no source.
            if not isinstance(question, NoulQuestion):
                confidence_source[qid] = CONFIDENCE_SOURCE
        latency_ms = (time.perf_counter() - started) * 1000

        # No usage: the model reads tokens but produces none, and its cost is CPU time.
        response = Response.model_validate({"model": self.model_id, "answers": answers})
        return AdapterResult(
            response=response,
            latency_ms=latency_ms,
            per_question_ms=per_question_ms,
            confidence_source=confidence_source,
            diagnostics=diagnostics,
        )

    def info(self) -> ModelInfo:
        _, _, cal = self._ready()
        if self._revision is not None:
            revision = self._revision
        else:
            revision = f"model.pt sha256 {sha256(self.weights)}"
        return ModelInfo(
            adapter="karar",
            model=self.model_id,
            revision=revision,
            path_used=(
                "karar adapter, decision head on the backbone, fp32 torch, "
                f"temperature {cal.temperature:.4f}"
                + (
                    " (per type: " + ", ".join(f"{k} {v:.4f}" for k, v in cal.by_type) + ")"
                    if cal.by_type
                    else ""
                )
                + f" and abstain map from {cal.source}"
            ),
            device=self.device,
            prompt_version=None,
        )
