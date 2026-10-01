"""The Hugging Face model repo answers as the karar adapter does.

The first tests run everywhere torch is installed: the copied network, packing
and calibration in release/model_repo/karar.py against the repository's own code
on a small random network, and the two sha256 refusals. The last ones build the
repo from the run the adapter's default calibrator was fitted on (r3-base-s2 as
shipped) and are skipped when those weights are absent. The build goes to
$KARAR_BUILD_DIR when it is set, else to a pytest temporary folder, and is
deleted afterwards either way.
"""

import importlib.util
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")

from bench.adapters.karar import CALIBRATOR as ADAPTER_CALIBRATOR  # noqa: E402
from bench.adapters.karar import Calibration, KararAdapter  # noqa: E402
from model.head.tests.test_head import encode, head  # noqa: E402
from release import convert  # noqa: E402
from schema.api import Request, Response, check_response  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "release" / "model_repo" / "karar.py"
CALIBRATOR = ROOT / ADAPTER_CALIBRATOR
PUBLIC = ROOT / "bench" / "hakembench" / "v1.0" / "public.jsonl"
ITEMS = 20
TOLERANCE = 1e-5

MANY = [f"seçenek {i:02d}" for i in range(13)]
QUESTIONS = {
    "birim": {"type": "choice", "instructions": "Hangi birim?",
              "criteria": {"teknik destek": "Arızalar", "fatura": None, "satış": None}},
    "çok": {"type": "choice", "instructions": "Hangisi?", "criteria": dict.fromkeys(MANY)},
    "öfke": {"type": "score", "instructions": "Ne kadar öfkeli?",
             "criteria": ["Sakin", "Rahatsız", "Öfkeli"]},
    "acil": {"type": "noul", "instructions": "Acil mi?", "criteria": {"true": "Hemen bakılmalı"}},
}  # fmt: skip
STATE = "Üç gündür internetim yok ve kimse aramadı."
FITTED = {
    "engine": "fp32", "weights": "test", "weights_sha256": "0" * 64,
    "temperatures": {"form": "type", "global": 1.3,
                     "type": {"choice": 1.2, "noul": 1.6, "score": 0.9}},
    "abstain_map": {"x": [0.2, 0.5, 0.9], "y": [0.7, 0.4, 0.05]},
}  # fmt: skip


def load_karar(path: Path):
    """karar.py as a user loads it, from its own file and nothing else."""
    name = f"karar_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up in sys.modules while the class is built.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_same(expected: dict, got: dict, where: str) -> None:
    """One answer of the adapter (as JSON) against one of karar.py."""
    assert got["type"] == expected["type"], where
    assert set(got) == set(expected), where
    for field in ("noul", "score", "confidence", "abstain"):
        if field in expected:
            assert got[field] == pytest.approx(expected[field], abs=TOLERANCE), (where, field)
    if "probabilities" in expected:
        assert list(got["probabilities"]) == list(expected["probabilities"]), where
        for key, value in expected["probabilities"].items():
            assert got["probabilities"][key] == pytest.approx(value, abs=TOLERANCE), (where, key)
    for field in ("choice", "legend"):
        if field in expected:
            assert got[field] == expected[field], (where, field)


def adapter_answers(adapter: KararAdapter, state, questions: dict) -> dict:
    request = Request.model_validate({"state": state, "model": "x", "questions": questions})
    response = adapter.answer(request).response
    return {qid: answer.model_dump(mode="json", exclude_none=True)
            for qid, answer in response.answers.items()}  # fmt: skip


def karar_answers(model, state, questions: dict) -> dict:
    out = model.decide(state, questions)
    request = Request.model_validate({"state": state, "model": "x", "questions": questions})
    check_response(request, Response.model_validate(out))
    return out["answers"]


def test_the_copied_code_answers_like_the_repository_code(tmp_path):
    karar = load_karar(SOURCE)
    original = head(causal=True)
    copy = karar.DecisionHead(karar.BackboneConfig(**asdict(original.backbone.cfg)),
                              causal=True, pooling="mean")  # fmt: skip
    copy.load_state_dict(original.state_dict(), strict=True)
    calibrator = tmp_path / "calibrator.json"
    calibrator.write_text(json.dumps(FITTED), encoding="utf-8")
    adapter = KararAdapter(model=original, encode=encode, revision="test",
                           calibration=Calibration.from_file(calibrator))  # fmt: skip
    model = karar.Karar(copy, encode, karar.Calibration.from_dict(FITTED), {"model_id": "x"})

    expected = adapter_answers(adapter, STATE, QUESTIONS)
    got = karar_answers(model, STATE, QUESTIONS)
    for qid in QUESTIONS:
        assert_same(expected[qid], got[qid], qid)
    # The listing order changes nothing, over more options than one pass holds.
    reordered = {**QUESTIONS["çok"], "criteria": dict.fromkeys(MANY[::-1])}
    again = karar_answers(model, STATE, {"çok": reordered})["çok"]
    for key, value in got["çok"]["probabilities"].items():
        assert again["probabilities"][key] == pytest.approx(value, abs=1e-6)


def test_a_calibrator_for_other_weights_is_refused():
    karar = load_karar(SOURCE)
    with pytest.raises(ValueError, match="other weights"):
        karar.Calibration.from_dict(FITTED, weights_sha256="1" * 64)
    assert karar.Calibration.from_dict(FITTED, weights_sha256="0" * 64).temperature == 1.3


def test_convert_refuses_a_calibrator_fitted_on_other_weights(tmp_path):
    (tmp_path / "models" / "run").mkdir(parents=True)
    (tmp_path / "models" / "run" / "model.pt").write_bytes(b"not these weights")
    calibrator = tmp_path / "calibrator.json"
    calibrator.write_text(json.dumps(FITTED), encoding="utf-8")
    with pytest.raises(ValueError, match="fitted on test weights"):
        convert.build("run", calibrator, tmp_path / "out", models=tmp_path / "models")
    assert not (tmp_path / "out").exists()


def shipped_run() -> str | None:
    """The run the adapter's default calibrator belongs to, if its weights are here."""
    if not CALIBRATOR.is_file():
        return None
    run = json.loads(CALIBRATOR.read_text(encoding="utf-8")).get("weights")
    return run if run and (convert.MODELS / run / "model.pt").is_file() else None


needs_weights = pytest.mark.skipif(shipped_run() is None, reason="the shipped weights are absent")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    base = Path(os.environ.get("KARAR_BUILD_DIR") or tmp_path_factory.mktemp("model_repo"))
    out = base / "ufakzeka-karar"
    shutil.rmtree(out, ignore_errors=True)
    try:
        sums = convert.build(shipped_run(), CALIBRATOR, out)
        yield out, sums
    finally:
        shutil.rmtree(out, ignore_errors=True)


def public_items() -> list[dict]:
    """ITEMS public items, taken in turn from each track in file order."""
    by_track: dict[str, list[dict]] = {}
    for line in PUBLIC.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            by_track.setdefault(item["track"], []).append(item)
    picked, depth = [], 0
    while len(picked) < ITEMS:
        for track in sorted(by_track):
            if depth < len(by_track[track]) and len(picked) < ITEMS:
                picked.append(by_track[track][depth])
        depth += 1
    return picked


@pytest.mark.local_model
@needs_weights
def test_the_built_repo_lists_its_files_and_their_sha256(built):
    out, sums = built
    files = {"model.safetensors", "config.json", "calibrator.json", "tokenizer.json",
             "tokenizer_config.json", "karar.py", "requirements.txt", "LICENSE"}  # fmt: skip
    assert set(sums) == files
    assert {p.name for p in out.iterdir()} == files | {convert.SUMS}
    listed = dict(reversed(line.split("  ")) for line in
                  (out / convert.SUMS).read_text(encoding="utf-8").splitlines())  # fmt: skip
    assert listed == sums
    assert (out / "karar.py").read_bytes() == SOURCE.read_bytes()
    assert (out / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    shipped = json.loads((out / "calibrator.json").read_text(encoding="utf-8"))
    assert shipped["weights_sha256"] == sums["model.safetensors"]
    assert shipped["source_weights_sha256"] == convert.sha256(
        convert.MODELS / shipped_run() / "model.pt"
    )


@pytest.mark.local_model
@needs_weights
def test_the_built_repo_refuses_a_calibrator_for_other_weights(built, tmp_path):
    out, _ = built
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(out / name, tmp_path / name)
    (tmp_path / "model.safetensors").symlink_to(out / "model.safetensors")
    shipped = json.loads((out / "calibrator.json").read_text(encoding="utf-8"))
    shipped["weights_sha256"] = shipped["source_weights_sha256"]
    (tmp_path / "calibrator.json").write_text(json.dumps(shipped), encoding="utf-8")
    karar = load_karar(out / "karar.py")
    with pytest.raises(ValueError, match="other weights"):
        karar.Karar.from_pretrained(tmp_path)


@pytest.mark.local_model
@needs_weights
def test_the_built_repo_answers_20_public_items_as_the_adapter_does(built):
    out, _ = built
    karar = load_karar(out / "karar.py")
    model = karar.Karar.from_pretrained(out, device="cpu")
    adapter = KararAdapter(weights=convert.MODELS / shipped_run() / "model.pt",
                           calibrator=CALIBRATOR)  # fmt: skip
    _, adapter_encode, _ = adapter._ready()

    items = public_items()
    kinds = {q["type"] for item in items for q in item["questions"].values()}
    assert kinds == {"choice", "score", "noul"}
    compared = 0
    for item in items:
        # The same tokens, from tokenizer.json alone.
        assert model.encode(item["state"]) == adapter_encode(item["state"]), item["id"]
        expected = adapter_answers(adapter, item["state"], item["questions"])
        got = karar_answers(model, item["state"], item["questions"])
        for qid in item["questions"]:
            assert_same(expected[qid], got[qid], f"{item['id']} {qid}")
            compared += 1
    assert compared >= ITEMS


def char_encode(text: str) -> list[int]:
    """One token per character: option names that share their first characters collide."""
    return [ord(c) for c in text]


# Twelve options take two passes, [0:10] and [10:12] in token order; the look-alike pair sorts
# ninth and tenth, one in each pass, and is the same once cut to 48 tokens.
LONG = "m" + "x" * 40
SPLIT = [f"a{i}" for i in range(9)] + [LONG + "1", LONG + "2", "z"]


def toy_karar():
    karar = load_karar(SOURCE)
    original = head(causal=True)
    copy = karar.DecisionHead(karar.BackboneConfig(**asdict(original.backbone.cfg)),
                              causal=True, pooling="mean")  # fmt: skip
    copy.load_state_dict(original.state_dict(), strict=True)
    return karar.Karar(copy, char_encode, karar.Calibration.from_dict(FITTED), {"model_id": "x"})


@pytest.mark.parametrize(
    ("state", "questions", "message"),
    [
        (STATE, {"q": {"type": "choice", "instructions": "?", "criteria": {1: None, "b": None}}},
         "option names must be strings"),
        (STATE, {7: QUESTIONS["acil"]}, "question ids must be strings, got int"),
        (STATE, {" ": QUESTIONS["acil"]}, "question ids must not be empty"),
        ({"etiketler": {"a", "b"}}, {"acil": QUESTIONS["acil"]}, "state is not JSON"),
        ({1: "bir", "iki": 2}, {"acil": QUESTIONS["acil"]}, "state is not JSON"),
        ("yarım \ud800 metin", {"acil": QUESTIONS["acil"]}, "state is not valid Unicode"),
        (STATE, {"q": {"type": "noul", "instructions": {"s": {1, 2}}}}, "instructions is not JSON"),
        (STATE, {"q": {"type": "choice", "instructions": "?", "criteria": dict.fromkeys(SPLIT)}},
         "two options encode to the same tokens"),
    ],
)  # fmt: skip
def test_bad_input_is_refused_with_a_plain_message(state, questions, message):
    with pytest.raises(ValueError, match=message):
        toy_karar().decide(state, questions)


def test_a_look_alike_pair_split_across_passes_is_refused_by_the_adapter_path_too():
    from model.head.infer import chunked_logits
    from schema.questions import ChoiceQuestion

    keys = sorted(SPLIT, key=lambda key: (char_encode("\n" + key), key))
    assert keys.index(LONG + "1") == 9 and keys.index(LONG + "2") == 10
    question = ChoiceQuestion(type="choice", instructions="?", criteria=dict.fromkeys(SPLIT))
    with pytest.raises(ValueError, match="encode to the same tokens"):
        chunked_logits(None, STATE, question, char_encode)
