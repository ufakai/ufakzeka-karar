"""The head's training loop runs end to end on rows and leaves files to select on."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from model.convert.native import NativeConfig, NativeModel  # noqa: E402
from model.head.train import HeadConfig, macro_f1, task_metrics, train_head  # noqa: E402
from schema.rows import TrainingRow  # noqa: E402


class Tokens:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        class Out:
            input_ids = [4 + (ord(c) % 90) for c in text]

        return Out()


FRUITS = ["elma", "armut", "kiraz"]


def rows(count, split, seed=0):
    out = []
    for i in range(count):
        answer = FRUITS[(i + seed) % 3]
        question = {"type": "choice", "instructions": "Hangi meyve?",
                    "criteria": {f: None for f in FRUITS}}  # fmt: skip
        row = TrainingRow(
            track="deneme", task="meyve", split=split, origin="authored", label_kind="human",
            source="authored", state=f"{i}. gün {answer} yedim.", question=question,
            target={f: float(f == answer) for f in FRUITS}, recipe="test",
        )  # fmt: skip
        out.append(row.model_dump_json())
    return "\n".join(out) + "\n"


def test_training_writes_evaluations_and_learns(tmp_path):
    (tmp_path / "train.jsonl").write_text(rows(90, "train"), encoding="utf-8")
    (tmp_path / "validation.jsonl").write_text(rows(30, "validation", seed=1), encoding="utf-8")
    torch.manual_seed(0)
    cfg = NativeConfig(vocab_size=97, n_layer=2, d_model=32, n_head=4, n_kv_head=2, d_ff=64,
                       head_dim=8, max_positions=256)  # fmt: skip
    config = HeadConfig(
        backbone="tiny", revision=None, causal=False,
        train_files=(str(tmp_path / "train.jsonl"),),
        validation_files=(str(tmp_path / "validation.jsonl"),),
        out_dir=str(tmp_path / "run"), epochs=6, batch_questions=8, eval_points=3,
        backbone_lr=3e-3, head_lr=3e-3,
    )  # fmt: skip
    report = train_head(config, NativeModel(cfg), Tokens())
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["train_rows"] == 90 and run["validation_rows"] == 30
    assert len(run["evaluations"]) == 3
    last = run["evaluations"][-1]["metrics"]["meyve"]
    assert last["accuracy"] > 0.9, last
    predictions = sorted((tmp_path / "run").glob("predictions_step*.jsonl"))
    assert len(predictions) == 3
    record = json.loads(predictions[-1].read_text().splitlines()[0])
    assert set(record) >= {"row_id", "keys", "probs", "target", "abstain_logit"}
    assert abs(sum(record["probs"]) - 1) < 1e-5
    assert report.losses[-1] < report.losses[0]


def test_macro_f1_uses_option_names_not_slots():
    assert macro_f1(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    assert macro_f1(["a", "b"], ["b", "a"]) == 0.0
    records = [
        {"keys": ["x", "y"], "probs": [0.9, 0.1], "target": {"x": 1.0, "y": 0.0}},
        {"keys": ["y", "x"], "probs": [0.2, 0.8], "target": {"x": 1.0, "y": 0.0}},
    ]
    metrics = task_metrics(records)
    assert metrics["accuracy"] == 1.0 and metrics["macro_f1"] == 1.0
    assert metrics["brier"] == pytest.approx(((0.1**2) * 2 + (0.2**2) * 2) / 2)


def test_prior_weights_move_a_flattened_mix_back_to_the_natural_one():
    from model.head.train import prior_weights

    def row(answer, i, split):
        question = {"type": "choice", "instructions": "Hangi meyve?",
                    "criteria": {f: None for f in FRUITS}}  # fmt: skip
        return TrainingRow(
            track="deneme", task="meyve", split=split, origin="authored", label_kind="human",
            source="authored", state=f"{split} {i} {answer}", question=question,
            target={f: float(f == answer) for f in FRUITS}, recipe="test",
        )  # fmt: skip

    # Training was capped to an even mix; the natural mix is mostly "elma".
    train = [row(FRUITS[i % 3], i, "train") for i in range(30)]
    natural = [row("elma" if i < 16 else FRUITS[1 + i % 2], i, "validation") for i in range(20)]
    weights = prior_weights(train, natural)
    by_answer = {f: [weights[r.row_id] for r in train if r.target[f] == 1.0] for f in FRUITS}
    assert min(by_answer["elma"]) > max(by_answer["armut"] + by_answer["kiraz"])
    assert sum(weights.values()) / len(weights) == pytest.approx(1.0)


def test_round_one_options_train_and_save_the_network(tmp_path):
    (tmp_path / "train.jsonl").write_text(rows(30, "train"), encoding="utf-8")
    (tmp_path / "validation.jsonl").write_text(rows(12, "validation", seed=1), encoding="utf-8")
    torch.manual_seed(0)
    cfg = NativeConfig(vocab_size=97, n_layer=2, d_model=32, n_head=4, n_kv_head=2, d_ff=64,
                       head_dim=8, max_positions=256)  # fmt: skip
    config = HeadConfig(
        backbone="tiny", revision=None, causal=True,
        train_files=(str(tmp_path / "train.jsonl"),),
        validation_files=(str(tmp_path / "validation.jsonl"),),
        out_dir=str(tmp_path / "run"), epochs=1, batch_questions=8, eval_points=1,
        prior_weights=True, drop_mined=True, save_model=True,
    )  # fmt: skip
    train_head(config, NativeModel(cfg), Tokens())
    state = torch.load(tmp_path / "run" / "model.pt")
    assert any(key.startswith("score.") for key in state)


def test_a_task_with_few_validation_rows_keeps_weights_near_one():
    from model.head.train import prior_weights

    def row(answer, i, split):
        question = {"type": "choice", "instructions": "Hangi meyve?",
                    "criteria": {f: None for f in FRUITS}}  # fmt: skip
        return TrainingRow(
            track="deneme", task="az", split=split, origin="authored", label_kind="human",
            source="authored", state=f"{split} {i} {answer}", question=question,
            target={f: float(f == answer) for f in FRUITS}, recipe="test",
        )  # fmt: skip

    train = [row(FRUITS[i % 3], i, "train") for i in range(30)]
    natural = [row("elma", i, "validation") for i in range(3)]
    weights = prior_weights(train, natural)
    assert max(weights.values()) / min(weights.values()) < 1.5


def test_micro_batches_give_the_whole_batch_s_gradient():
    from model.head.head import DecisionHead, objective, soft_targets
    from model.head.pack import collate, pack
    from model.head.train import accumulate, micro_batches

    torch.manual_seed(0)
    cfg = NativeConfig(vocab_size=97, n_layer=2, d_model=32, n_head=4, n_kv_head=2, d_ff=64,
                       head_dim=8, max_positions=256)  # fmt: skip
    model = DecisionHead(NativeModel(cfg), causal=True)
    lines = rows(9, "train").splitlines()
    group = []
    for i, line in enumerate(lines):
        row = TrainingRow.model_validate_json(line)
        state = row.state + " uzun" * (i * 3)
        group.append((pack(state, row.question, lambda t: Tokens()(t).input_ids), row))
    weight_of = {r.row_id: 1.0 + (i % 3) for i, (_, r) in enumerate(group)}
    assert len(micro_batches(group, 120)) > 1
    config = HeadConfig(backbone="t", revision=None, causal=True, train_files=(),
                        validation_files=(), out_dir="x", max_micro_tokens=120)  # fmt: skip
    model.zero_grad()
    accumulate(model, group, config, weight_of, torch.autocast("cpu", enabled=False))
    split = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    model.zero_grad()
    batch = collate([p for p, _ in group], 0)
    logits, _ = model(batch)
    weights = torch.tensor([weight_of[r.row_id] for _, r in group])
    objective(logits, soft_targets(batch, [r.target for _, r in group]), "ce", weights).backward()
    whole = [p.grad for p in model.parameters() if p.grad is not None]
    for a, b in zip(split, whole, strict=True):
        assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def _fruit_row(answer, i, split, task="meyve"):
    question = {"type": "choice", "instructions": "Hangi meyve?",
                "criteria": {f: None for f in FRUITS}}  # fmt: skip
    return TrainingRow(
        track="deneme", task=task, split=split, origin="authored", label_kind="human",
        source="authored", state=f"{task} {split} {i} {answer}", question=question,
        target={f: float(f == answer) for f in FRUITS}, recipe="test",
    )  # fmt: skip


def test_multipliers_move_weight_inside_a_task_and_never_between_tasks():
    from model.head.train import prior_weights

    train = [_fruit_row(FRUITS[i % 3], i, "train") for i in range(30)]
    other = [_fruit_row(FRUITS[i % 3], i, "train", task="sebze") for i in range(12)]
    natural = [_fruit_row(FRUITS[i % 3], i, "validation") for i in range(30)]
    plain = prior_weights(train + other, natural)
    assert prior_weights(train + other, natural, {}) == plain
    down = {r.row_id: 0.25 for r in train[:6]}
    weights = prior_weights(train + other, natural, down)
    for task_rows in (train, other):
        assert sum(weights[r.row_id] for r in task_rows) == pytest.approx(len(task_rows))
    assert weights[train[0].row_id] < plain[train[0].row_id]
    assert weights[train[29].row_id] > plain[train[29].row_id]
    assert all(weights[r.row_id] == pytest.approx(1.0) for r in other)


def test_scored_files_are_written_at_every_evaluation_point(tmp_path):
    (tmp_path / "train.jsonl").write_text(rows(30, "train"), encoding="utf-8")
    (tmp_path / "validation.jsonl").write_text(rows(12, "validation", seed=1), encoding="utf-8")
    torch.manual_seed(0)
    cfg = NativeConfig(vocab_size=97, n_layer=2, d_model=32, n_head=4, n_kv_head=2, d_ff=64,
                       head_dim=8, max_positions=256)  # fmt: skip
    config = HeadConfig(
        backbone="tiny", revision=None, causal=True,
        train_files=(str(tmp_path / "train.jsonl"),),
        validation_files=(str(tmp_path / "validation.jsonl"),),
        out_dir=str(tmp_path / "run"), epochs=1, batch_questions=8, eval_points=2,
        score_files=(str(tmp_path / "train.jsonl"),),
    )  # fmt: skip
    train_head(config, NativeModel(cfg), Tokens())
    scored = sorted((tmp_path / "run").glob("scores_step*.jsonl"))
    assert len(scored) == 2
    records = [json.loads(line) for line in scored[-1].read_text().splitlines()]
    assert len(records) == 30 and {"row_id", "probs", "target", "keys"} <= set(records[0])


@pytest.mark.parametrize("arm", [{"loss": "rl"}, {"layout": "sequential", "shuffle_options": True}])
def test_the_step_6_arms_train_and_write_predictions(tmp_path, arm):
    (tmp_path / "train.jsonl").write_text(rows(24, "train"), encoding="utf-8")
    (tmp_path / "validation.jsonl").write_text(rows(8, "validation", seed=1), encoding="utf-8")
    torch.manual_seed(0)
    cfg = NativeConfig(vocab_size=97, n_layer=2, d_model=32, n_head=4, n_kv_head=2, d_ff=64,
                       head_dim=8, max_positions=512)  # fmt: skip
    config = HeadConfig(
        backbone="tiny", revision=None, causal=True,
        train_files=(str(tmp_path / "train.jsonl"),),
        validation_files=(str(tmp_path / "validation.jsonl"),),
        out_dir=str(tmp_path / "run"), epochs=1, batch_questions=8, eval_points=1,
        prior_weights=True, **arm,
    )  # fmt: skip
    report = train_head(config, NativeModel(cfg), Tokens())
    assert report.model.layout == arm.get("layout", "blind")
    assert all(np.isfinite(report.losses))
    assert list((tmp_path / "run").glob("predictions_step*.jsonl"))


def test_a_dev_task_is_evaluated_but_kept_out_of_the_prior_weights(tmp_path, monkeypatch):
    import model.head.train as train_module

    seen = {}

    def spy(train, validation, multipliers=None):
        seen["tasks"] = {r.task for r in validation}
        return {}

    monkeypatch.setattr(train_module, "prior_weights", spy)
    (tmp_path / "train.jsonl").write_text(rows(16, "train"), encoding="utf-8")
    dev = rows(6, "validation", seed=1).replace('"task":"meyve"', '"task":"dev"')
    (tmp_path / "validation.jsonl").write_text(rows(6, "validation") + dev, encoding="utf-8")
    cfg = NativeConfig(vocab_size=97, n_layer=1, d_model=16, n_head=2, n_kv_head=1, d_ff=32,
                       head_dim=8, max_positions=256)  # fmt: skip
    config = HeadConfig(
        backbone="tiny", revision=None, causal=False,
        train_files=(str(tmp_path / "train.jsonl"),),
        validation_files=(str(tmp_path / "validation.jsonl"),),
        out_dir=str(tmp_path / "run"), epochs=1, batch_questions=8, eval_points=1,
        prior_weights=True, prior_exclude=("dev",),
    )  # fmt: skip
    train_head(config, NativeModel(cfg), Tokens())
    assert seen["tasks"] == {"meyve"}
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert set(run["evaluations"][-1]["metrics"]) == {"meyve", "dev"}
