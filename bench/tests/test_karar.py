"""The karar adapter on a small network with the real head, packing and chunking code."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bench.adapters.base import AdapterError  # noqa: E402
from bench.adapters.karar import Calibration, KararAdapter  # noqa: E402
from bench.harness.items import Item  # noqa: E402
from bench.harness.results import ResultsWriter  # noqa: E402
from bench.harness.runner import run  # noqa: E402
from model.head.infer import chunked_choice  # noqa: E402
from model.head.pack import collate, pack  # noqa: E402
from model.head.tests.test_head import encode, head  # noqa: E402
from schema.api import Request, check_response  # noqa: E402

CALIBRATOR = Path(__file__).resolve().parents[2] / "results" / "step7" / "calibrator.json"
MANY = [f"seçenek {i:02d}" for i in range(12)]
CAL = Calibration(temperature=2.0, map_x=(0.2, 0.9), map_y=(0.8, 0.1))
QUESTIONS = {
    "birim": {"type": "choice", "instructions": "Hangi birim?",
              "criteria": {"teknik destek": "Arızalar", "fatura": None, "satış": None}},
    "çok": {"type": "choice", "instructions": "Hangisi?", "criteria": dict.fromkeys(MANY)},
    "öfke": {"type": "score", "instructions": "Ne kadar öfkeli?",
             "criteria": ["Sakin", "Rahatsız", "Öfkeli"]},
    "acil": {"type": "noul", "instructions": "Acil mi?"},
}  # fmt: skip
STATE = "Üç gündür internetim yok ve kimse aramadı."


def request() -> Request:
    return Request.model_validate({"state": STATE, "model": "x", "questions": QUESTIONS})


def adapter(model=None) -> KararAdapter:
    return KararAdapter(model=model or head(causal=True), encode=encode, calibration=CAL,
                        revision="test weights")  # fmt: skip


def direct_logits(model, question) -> dict[str, float]:
    batch = collate([pack(STATE, question, encode)], model.pad_id)
    with torch.no_grad():
        out, _ = model(batch)
    return dict(zip(batch.keys[0], out[0].tolist(), strict=True))


def tempered(logits: list[float], t: float) -> np.ndarray:
    x = np.asarray(logits) / t
    e = np.exp(x - x.max())
    return e / e.sum()


def test_answers_follow_the_contract_and_the_shipped_calibration():
    model = head(causal=True)
    a = adapter(model)
    req = request()
    result = a.answer(req)
    check_response(req, result.response)
    answers = result.response.answers

    # Choice with three options: softmax(logits / T), confidence from the raw maximum.
    logits = direct_logits(model, req.questions["birim"])
    keys = list(QUESTIONS["birim"]["criteria"])
    expected = tempered([logits[k] for k in keys], 2.0)
    got = answers["birim"]
    assert [got.probabilities[k] for k in keys] == pytest.approx(expected.tolist(), abs=1e-6)
    raw_max = tempered([logits[k] for k in keys], 1.0).max()
    error = float(np.interp(raw_max, CAL.map_x, CAL.map_y))
    assert got.confidence == pytest.approx(1 - error)
    assert got.abstain == pytest.approx(error)
    assert got.choice == keys[int(np.argmax(expected))]

    # The temperature flattens, and the raw distribution is kept beside the answer.
    raw = result.diagnostics["birim"]["raw_probabilities"]
    assert max(raw) >= max(got.probabilities.values())
    assert result.diagnostics["birim"]["temperature"] == 2.0

    # Score: levels in order, the expected level, the same confidence rule.
    logits = direct_logits(model, req.questions["öfke"])
    expected = tempered([logits[str(k)] for k in range(3)], 2.0)
    got = answers["öfke"]
    assert [got.probabilities[str(k)] for k in range(3)] == pytest.approx(expected.tolist())
    assert got.score == pytest.approx(float(np.dot(np.arange(3), expected)))
    assert got.legend == {"0": "Sakin", "1": "Rahatsız", "2": "Öfkeli"}

    # Noul: the probability of yes, the expected error as abstain, no confidence.
    logits = direct_logits(model, req.questions["acil"])
    expected = tempered([logits["true"], logits["false"]], 2.0)
    assert answers["acil"].noul == pytest.approx(expected[0])
    raw_max = tempered([logits["true"], logits["false"]], 1.0).max()
    assert answers["acil"].abstain == pytest.approx(np.interp(raw_max, CAL.map_x, CAL.map_y))

    assert result.confidence_source == {"birim": "model", "çok": "model", "öfke": "model"}
    assert set(result.per_question_ms) == set(QUESTIONS)


def test_more_than_ten_options_go_through_the_chunked_passes():
    model = head(causal=True)
    req = request()
    result = adapter(model).answer(req)
    got = result.response.answers["çok"]
    assert list(got.probabilities) == MANY
    raw = chunked_choice(model, STATE, req.questions["çok"], encode)
    assert result.diagnostics["çok"]["raw_probabilities"] == pytest.approx(list(raw.values()))
    expected = tempered(np.log(list(raw.values())), 2.0)
    assert list(got.probabilities.values()) == pytest.approx(expected.tolist(), abs=1e-9)


def test_the_committed_calibrator_is_read():
    cal = Calibration.from_file(CALIBRATOR)
    data = json.loads(CALIBRATOR.read_text(encoding="utf-8"))
    assert cal.temperature == data["temperatures"]["global"]
    # The map holds its end values outside its range, as the fitted isotonic map clips.
    assert cal.expected_error(0.0) == data["abstain_map"]["y"][0]
    assert cal.expected_error(1.0) == data["abstain_map"]["y"][-1]
    assert cal.expected_error(0.6) > cal.expected_error(0.95)
    assert "sha256" in cal.source


def test_a_calibrator_of_another_form_or_engine_is_refused(tmp_path):
    data = json.loads(CALIBRATOR.read_text(encoding="utf-8"))
    bucket = {**data, "temperatures": {"form": "bucket", "global": 1.2, "bucket": {"a": 1.1}}}
    (tmp_path / "bucket.json").write_text(json.dumps(bucket))
    with pytest.raises(AdapterError, match="global"):
        Calibration.from_file(tmp_path / "bucket.json")
    (tmp_path / "int8.json").write_text(json.dumps({**data, "engine": "int8-dynamic"}))
    with pytest.raises(AdapterError, match="fp32"):
        Calibration.from_file(tmp_path / "int8.json")
    with pytest.raises(AdapterError, match="ascending"):
        Calibration(temperature=1.0, map_x=(0.9, 0.2), map_y=(0.1, 0.8))
    with pytest.raises(AdapterError, match="positive"):
        Calibration(temperature=0.0, map_x=(0.2,), map_y=(0.5,))


def test_an_adapter_needs_weights_or_a_model_with_its_encoder():
    with pytest.raises(AdapterError):
        KararAdapter()
    with pytest.raises(AdapterError):
        KararAdapter(model=head(causal=True))


def test_info_is_honest():
    info = adapter().info()
    assert info.adapter == "karar" and info.model == "ufakzeka-karar"
    assert info.revision == "test weights" and info.device == "cpu"
    assert info.prompt_version is None
    assert "temperature 2.0000" in info.path_used


def test_info_hashes_the_weights_file(tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"not really weights")
    a = KararAdapter(weights, model=head(causal=True), encode=encode, calibration=CAL)
    assert a.info().revision.startswith("model.pt sha256 ")


def test_the_runner_writes_one_row_per_question(tmp_path):
    gold = {"birim": "fatura", "acil": True}
    item = Item.model_validate({"id": "t1", "track": "destek", "state": STATE,
                                "questions": QUESTIONS, "gold": gold})  # fmt: skip
    out = tmp_path / "rows.jsonl"
    with ResultsWriter(out) as writer:
        run([item], adapter(), writer, repo_root=tmp_path, script="test", git_commit="abc")
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    by_id = {r["question_id"]: r for r in rows}
    assert by_id["birim"]["adapter"] == "karar" and by_id["birim"]["gold"] == "fatura"
    assert by_id["birim"]["latency_scope"] == "question"
    assert by_id["acil"]["confidence_source"] is None
    assert by_id["acil"]["answer"]["abstain"] is not None


def test_a_calibrator_is_refused_for_other_weights(tmp_path):
    data = json.loads(CALIBRATOR.read_text(encoding="utf-8"))
    ok = Calibration.from_file(CALIBRATOR, weights_sha256=data["weights_sha256"])
    assert ok.temperature == data["temperatures"]["global"]
    with pytest.raises(AdapterError, match="weights"):
        Calibration.from_file(CALIBRATOR, weights_sha256="0" * 64)
    (tmp_path / "old.json").write_text(
        json.dumps({k: v for k, v in data.items() if k not in ("weights", "weights_sha256")})
    )
    with pytest.raises(AdapterError, match="unrecorded"):
        Calibration.from_file(tmp_path / "old.json", weights_sha256=data["weights_sha256"])


def test_a_per_type_calibrator_applies_each_type_its_own_temperature(tmp_path):
    data = json.loads(CALIBRATOR.read_text(encoding="utf-8"))
    typed = {**data, "temperatures": {"form": "type", "global": 1.2,
                                      "type": {"choice": 2.0, "noul": 1.0}}}  # fmt: skip
    (tmp_path / "typed.json").write_text(json.dumps(typed))
    cal = Calibration.from_file(tmp_path / "typed.json")
    assert cal.temperature_for("choice") == 2.0 and cal.temperature_for("noul") == 1.0
    assert cal.temperature_for("score") == 1.2
    logits = [2.0, 0.0]
    assert cal.probabilities(logits, "choice")[0] < cal.probabilities(logits, "noul")[0]
