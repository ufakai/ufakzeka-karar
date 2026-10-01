"""The instrument's training runner, on tiny random models. No download, CPU only.

These tests are the specification of model/instrument/train.py. Needs the
instrument dependency group: `just test-instrument`.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import (  # noqa: E402
    BertConfig,
    BertForSequenceClassification,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForSequenceClassification,
)

from data.instrument_format import Row, write_split  # noqa: E402
from model.instrument.train import (  # noqa: E402
    RunConfig,
    choose_precision,
    max_length_for,
    run,
)

WORDS = ["iyi", "güzel", "harika", "sevdim", "kötü", "berbat", "rezalet", "sevmedim", "film", "bu"]
PAD, END = 1, 2


def tokenizer():
    vocab = {"[UNK]": 0, "<|pad|>": PAD, "<|endoftext|>": END}
    vocab.update({w: i + 3 for i, w in enumerate(WORDS)})
    core = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=core, pad_token="<|pad|>", eos_token="<|endoftext|>", unk_token="[UNK]"
    )


def causal_loader(num_labels):
    config = Qwen3Config(
        vocab_size=13 + 3, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
        pad_token_id=PAD, num_labels=num_labels, tie_word_embeddings=True,
    )  # fmt: skip
    return tokenizer(), Qwen3ForSequenceClassification(config)


def encoder_loader(num_labels):
    config = BertConfig(
        vocab_size=13 + 3, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, max_position_embeddings=64, pad_token_id=PAD, num_labels=num_labels,
    )  # fmt: skip
    return tokenizer(), BertForSequenceClassification(config)


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(0)
    folder = tmp_path / "data" / "toy"

    def make(prefix, n):
        out = []
        for i in range(n):
            label = i % 2
            pool = WORDS[:4] if label else WORDS[4:8]
            words = ["bu", "film"] + list(rng.choice(pool, size=int(rng.integers(1, 5))))
            out.append(
                Row(f"{prefix}-{i}", " ".join(words) + f" {'bu ' * (i % 7)}".rstrip(), None, label)
            )
        return out

    splits = {"train": make("tr", 96), "validation": make("va", 32), "test": make("te", 40)}
    for name, split_rows in splits.items():
        write_split(folder / f"{name}.jsonl", split_rows)
    meta = {
        "name": "toy", "task": "toy sentiment", "source": "https://example.invalid/toy",
        "source_revision": "rev0", "license_id": "MIT", "labels": ["olumsuz", "olumlu"],
        "splits": {k: len(v) for k, v in splits.items()},
        "split_origin": {k: "source" for k in splits}, "dropped": {}, "notes": "",
    }  # fmt: skip
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return folder


def config(dataset, out, loader=causal_loader, **changes):
    fields = dict(
        backbone="tiny/causal", revision=None, dataset_dir=dataset, out_dir=out,
        learning_rate=5e-3, seed=1, epochs=3, batch_size=16, eval_points=6,
        device="cpu", model_loader=loader,
    )  # fmt: skip
    return RunConfig(**{**fields, **changes})


def test_a_run_writes_logits_for_every_evaluation_point(dataset, tmp_path):
    out = run(config(dataset, tmp_path / "run"))

    val_labels = np.load(out / "labels_validation.npy")
    test_labels = np.load(out / "labels_test.npy")
    assert val_labels.dtype == np.int64 and len(val_labels) == 32 and len(test_labels) == 40

    val_files = sorted(out.glob("logits_validation_step*.npy"))
    test_files = sorted(out.glob("logits_test_step*.npy"))
    assert len(val_files) == 6 and len(test_files) == 6
    assert {f.name.replace("validation", "test") for f in val_files} == {f.name for f in test_files}
    for path in val_files + test_files:
        logits = np.load(path)
        assert logits.dtype == np.float32 and logits.shape[1] == 2
        assert np.isfinite(logits).all()
    assert np.load(val_files[0]).shape[0] == 32

    # Logits are in dataset order: the labels file matches the split file.
    rows = [json.loads(line) for line in (dataset / "test.jsonl").read_text("utf-8").splitlines()]
    assert test_labels.tolist() == [r["label"] for r in rows]
    # The last evaluation point is the end of training.
    steps = sorted(int(f.stem.split("step")[1]) for f in val_files)
    meta = json.loads((out / "run.json").read_text("utf-8"))
    assert steps[-1] == meta["total_steps"] and steps == meta["eval_steps"]
    # No checkpoints are kept.
    assert not list(out.glob("checkpoint-*")) and not list(out.glob("*.safetensors"))


def test_the_toy_task_is_learned(dataset, tmp_path):
    out = run(config(dataset, tmp_path / "run", epochs=8))
    last = sorted(out.glob("logits_test_step*.npy"), key=lambda p: int(p.stem.split("step")[1]))[-1]
    accuracy = (np.load(last).argmax(1) == np.load(out / "labels_test.npy")).mean()
    assert accuracy > 0.9


def test_run_metadata_describes_the_run(dataset, tmp_path):
    out = run(config(dataset, tmp_path / "run"))
    meta = json.loads((out / "run.json").read_text("utf-8"))
    # Every field the treatment check compares is written, and written as a
    # value. The check skips a missing field so that old records stay readable;
    # this is what stops a new record from being missing one.
    from model.instrument.plan import TREATMENT

    for key in ("dataset", "backbone", *TREATMENT):
        assert meta.get(key) is not None, f"run.json does not record {key}"
    for key in (
        "backbone", "revision", "dataset", "dataset_source_revision", "labels", "learning_rate",
        "seed", "epochs", "batch_size", "weight_decay", "warmup", "max_length", "pooling",
        "truncated_share", "device", "gpu_name", "precision", "parameters_total",
        "parameters_non_embedding", "weights_dtype", "attn_implementation", "versions",
        "git_commit", "total_steps", "eval_steps", "train_rows", "wall_seconds", "train_loss",
    ):  # fmt: skip
        assert key in meta, key
    assert meta["dataset"] == "toy" and meta["dataset_source_revision"] == "rev0"
    assert meta["weights_dtype"] == "float32"
    assert meta["precision"] == {"bf16": False, "fp16": False, "tf32": False}
    assert meta["pooling"] == "stock"
    assert meta["parameters_total"] > meta["parameters_non_embedding"] > 0
    assert {"torch", "transformers", "accelerate"} <= set(meta["versions"])
    assert set(meta["truncated_share"]) == {"train", "validation", "test"}


def test_the_same_seed_gives_the_same_logits_and_another_seed_does_not(dataset, tmp_path):
    a = run(config(dataset, tmp_path / "a", seed=4))
    b = run(config(dataset, tmp_path / "b", seed=4))
    c = run(config(dataset, tmp_path / "c", seed=21))
    name = sorted(p.name for p in a.glob("logits_validation_step*.npy"))[-1]
    assert np.allclose(np.load(a / name), np.load(b / name), atol=1e-5)
    assert not np.allclose(np.load(a / name), np.load(c / name), atol=1e-3)


def test_the_new_head_follows_the_seed(dataset, tmp_path):
    # The seed must be set BEFORE the model is built, or the head's
    # initialisation is left to whatever the process did earlier.
    seen = {}

    def loader(num_labels):
        tok, model = causal_loader(num_labels)
        seen["head"] = model.score.weight.detach().clone()
        return tok, model

    torch.manual_seed(999)
    run(config(dataset, tmp_path / "a", loader=loader, seed=40))
    first = seen["head"]
    torch.manual_seed(123)
    run(config(dataset, tmp_path / "b", loader=loader, seed=40))
    assert torch.equal(first, seen["head"])


def test_an_encoder_runs_through_the_same_code(dataset, tmp_path):
    out = run(config(dataset, tmp_path / "run", loader=encoder_loader, backbone="tiny/encoder"))
    assert len(list(out.glob("logits_test_step*.npy"))) == 6


def test_weights_that_are_not_float32_are_refused(dataset, tmp_path):
    def half_loader(num_labels):
        tok, model = causal_loader(num_labels)
        return tok, model.to(torch.bfloat16)

    with pytest.raises(ValueError, match="float32"):
        run(config(dataset, tmp_path / "run", loader=half_loader))


def test_a_pad_token_mismatch_is_refused(dataset, tmp_path):
    # The causal head finds its position by comparing input ids with
    # config.pad_token_id, so the two must agree.
    def wrong_pad(num_labels):
        tok, model = causal_loader(num_labels)
        model.config.pad_token_id = END
        return tok, model

    with pytest.raises(ValueError, match="pad"):
        run(config(dataset, tmp_path / "run", loader=wrong_pad))


def test_an_existing_run_folder_is_not_overwritten(dataset, tmp_path):
    out = run(config(dataset, tmp_path / "run"))
    with pytest.raises(FileExistsError):
        run(config(dataset, out))


def test_appended_end_pooling_puts_the_end_token_last(dataset, tmp_path):
    out = run(config(dataset, tmp_path / "run", pooling="appended_end"))
    meta = json.loads((out / "run.json").read_text("utf-8"))
    assert meta["pooling"] == "appended_end"
    assert meta["appended_token_id"] == END


def test_the_commit_can_come_from_the_launcher(dataset, tmp_path, monkeypatch):
    # On a rented GPU the code is an archive with no .git folder.
    monkeypatch.setenv("KARAR_GIT_COMMIT", "abc123")
    out = run(config(dataset, tmp_path / "run"))
    assert json.loads((out / "run.json").read_text("utf-8"))["git_commit"] == "abc123"


def test_appended_end_leaves_room_at_the_position_limit(dataset, tmp_path):
    # A cap of 4 tokens on a model whose limit is 4: the end token needs a position.
    def short_limit(num_labels):
        tok, model = causal_loader(num_labels)
        model.config.max_position_embeddings = 4
        return tok, model

    out = run(config(dataset, tmp_path / "run", loader=short_limit, pooling="appended_end"))
    assert json.loads((out / "run.json").read_text("utf-8"))["max_length"] == 3


def test_mean_pooling_runs_and_ignores_padding(dataset, tmp_path):
    out = run(config(dataset, tmp_path / "run", pooling="mean"))
    assert json.loads((out / "run.json").read_text("utf-8"))["pooling"] == "mean"
    logits = np.load(sorted(out.glob("logits_test_step*.npy"))[-1])
    assert np.isfinite(logits).all()


def test_accumulation_matches_one_big_batch(dataset, tmp_path):
    # Four micro-batches of 4 must give the same optimiser step as one batch of
    # 16: the same effective batch, the same number of steps, the same result
    # up to the order floating point adds things in.
    whole = run(config(dataset, tmp_path / "whole", batch_size=16))
    split = run(config(dataset, tmp_path / "split", batch_size=16, micro_batch_size=4))

    both = (
        json.loads((whole / "run.json").read_text("utf-8")),
        json.loads((split / "run.json").read_text("utf-8")),
    )
    assert both[0]["total_steps"] == both[1]["total_steps"]
    assert both[0]["eval_steps"] == both[1]["eval_steps"]
    assert (both[0]["micro_batch_size"], both[1]["micro_batch_size"]) == (16, 4)
    for a, b in zip(both[0]["train_loss"], both[1]["train_loss"], strict=True):
        assert a[0] == b[0]
        assert a[1] == pytest.approx(b[1], abs=2e-4)
    name = sorted(p.name for p in whole.glob("logits_validation_step*.npy"))[-1]
    assert np.allclose(np.load(whole / name), np.load(split / name), atol=2e-3)


@pytest.mark.parametrize("micro", [5, 32])
def test_a_micro_batch_that_does_not_divide_the_batch_is_refused(dataset, tmp_path, micro):
    with pytest.raises(ValueError, match="micro_batch_size"):
        run(config(dataset, tmp_path / "run", batch_size=16, micro_batch_size=micro))


def test_the_sweep_can_skip_test_evaluation(dataset, tmp_path):
    out = run(config(dataset, tmp_path / "run", eval_splits=("validation",)))
    assert len(list(out.glob("logits_validation_step*.npy"))) == 6
    assert not list(out.glob("logits_test_step*.npy"))
    assert not (out / "labels_test.npy").exists()
    assert (out / "labels_validation.npy").exists()
    assert json.loads((out / "run.json").read_text("utf-8"))["eval_splits"] == ["validation"]


@pytest.mark.parametrize("splits", [(), ("train",), ("validation", "nope")])
def test_bad_eval_splits_are_refused(dataset, tmp_path, splits):
    with pytest.raises(ValueError, match="eval_splits"):
        run(config(dataset, tmp_path / "run", eval_splits=splits))


def test_unknown_pooling_is_refused(dataset, tmp_path):
    with pytest.raises(ValueError, match="pooling"):
        run(config(dataset, tmp_path / "run", pooling="first"))


@pytest.mark.parametrize(
    ("device_type", "capability", "expected"),
    [
        ("cuda", (8, 0), {"bf16": True, "fp16": False, "tf32": True}),  # A100
        ("cuda", (8, 9), {"bf16": True, "fp16": False, "tf32": True}),  # L4
        # A T4 reports bf16 support through emulation; it must get fp16.
        ("cuda", (7, 5), {"bf16": False, "fp16": True, "tf32": False}),
        ("mps", None, {"bf16": False, "fp16": False, "tf32": False}),
        ("cpu", None, {"bf16": False, "fp16": False, "tf32": False}),
    ],
)
def test_precision_follows_the_compute_capability(device_type, capability, expected):
    assert choose_precision(device_type, capability) == expected


def test_max_length_is_the_95th_percentile_capped():
    tok = tokenizer()
    texts = ["bu"] * 95 + ["bu " * 40] * 5
    # 95 percent of rows have one token; the long tail does not set the length.
    assert max_length_for(tok, texts, None, cap=512, model_limit=64) <= 2
    long_texts = ["bu " * 50] * 100
    assert max_length_for(tok, long_texts, None, cap=512, model_limit=64) == 50
    assert max_length_for(tok, long_texts, None, cap=32, model_limit=64) == 32
    assert max_length_for(tok, long_texts, None, cap=512, model_limit=40) == 40
    pairs = ["bu " * 10] * 100
    assert max_length_for(tok, pairs, pairs, cap=512, model_limit=64) == 20


def test_the_bidirectional_arm_really_reads_both_ways():
    """The point of the bidirectional arm: a position must see the tokens after it.

    Checking the wrapper was constructed proves nothing. This runs the real
    classifier both ways and asserts the two arms disagree, which they cannot
    do unless the mask actually changed what each position may read.
    """
    from model.instrument.train import BidirectionalClassifier

    torch.manual_seed(0)
    _, model = causal_loader(3)
    model.eval()
    ids = torch.randint(3, 13, (1, 8))
    mask = torch.ones_like(ids)
    later = ids.clone()
    later[0, -1] = 3 + (int(ids[0, -1]) + 5) % 10

    with torch.no_grad():
        causal_a = model(input_ids=ids, attention_mask=mask).logits
        both = BidirectionalClassifier(model)
        bidi_a = both(input_ids=ids, attention_mask=mask)
        bidi_b = both(input_ids=later, attention_mask=mask)

    assert not torch.allclose(causal_a, bidi_a, atol=1e-5)
    assert not torch.allclose(bidi_a, bidi_b, atol=1e-5)


def test_padding_does_not_change_the_bidirectional_answer():
    """The property that matters: padding a sequence must not move its logits.

    An earlier version of this test changed the padded positions to an ordinary
    token, which legitimately moves the head's pooling point because the head
    finds its token from `input_ids != pad_token_id`. That tested the head, not
    the mask.
    """
    from model.instrument.train import BidirectionalClassifier

    torch.manual_seed(1)
    _, model = causal_loader(3)
    model.eval()
    both = BidirectionalClassifier(model)

    ids = torch.randint(3, 13, (1, 6))
    padded = torch.cat([ids, torch.full((1, 4), PAD)], dim=1)
    with torch.no_grad():
        short = both(input_ids=ids, attention_mask=torch.ones_like(ids))
        long = both(
            input_ids=padded,
            attention_mask=torch.cat([torch.ones(1, 6), torch.zeros(1, 4)], dim=1).long(),
        )
    assert torch.allclose(short, long, atol=1e-5)


def test_the_bidirectional_arm_trains_through_the_same_protocol(dataset, tmp_path):
    out = run(config(dataset, tmp_path / "bidi", attention="bidirectional"))
    written = json.loads((out / "run.json").read_text(encoding="utf-8"))
    # Recorded, so the paper can tell the two arms apart from the files alone.
    assert written["attention"] == "bidirectional"
    assert written["pooling"] == "stock"


def test_the_two_arms_cannot_overwrite_each_other():
    from model.instrument.plan import RunSpec

    causal = RunSpec("massive_tr", "ufakzeka", 3e-5, 1)
    bidi = RunSpec("massive_tr", "ufakzeka", 3e-5, 1, attention="bidirectional")
    assert causal.path != bidi.path
    # The committed runs keep the paths they were written with, so the resume
    # rule does not report the new arm as already done.
    assert causal.path.endswith("stock-lr3e-5-s1")
    assert "bidirectional" in bidi.path


def test_an_unknown_attention_is_refused(dataset, tmp_path):
    with pytest.raises(ValueError, match="attention must be one of"):
        run(config(dataset, tmp_path / "a", attention="sideways"))


def test_mean_pooling_under_bidirectional_attention_really_is_both():
    """The wrapper used to bypass the bidirectional one, so the pair was refused.

    It now builds the same mask mapping itself. Held here to a direct
    computation: every position sees every real token, in both directions, and
    padding neither attends nor is averaged.
    """
    from transformers import Qwen3Config, Qwen3ForSequenceClassification

    from model.convert.bidirectional import bidirectional_masks
    from model.instrument.train import MeanPooledClassifier

    torch.manual_seed(0)
    model = Qwen3ForSequenceClassification(
        Qwen3Config(
            vocab_size=50,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_labels=3,
            pad_token_id=0,
            max_position_embeddings=32,
        )  # fmt: skip
    ).eval()
    ids = torch.randint(1, 50, (2, 10))
    mask = torch.ones_like(ids)
    ids[1, 7:], mask[1, 7:] = 0, 0

    with torch.no_grad():
        got = MeanPooledClassifier(model, bidirectional=True)(ids, mask)
        embeds = model.model.embed_tokens(ids)
        states = model.model(
            inputs_embeds=embeds, attention_mask=bidirectional_masks(model.model, embeds, mask)
        ).last_hidden_state
        weights = mask.unsqueeze(-1).float()
        expected = model.score((states * weights).sum(1) / weights.sum(1))
        causal = MeanPooledClassifier(model)(ids, mask)
    assert torch.allclose(got, expected, atol=1e-6)
    assert not torch.allclose(got, causal, atol=1e-3), "it is still reading causally"

    # Changing a later token changes an earlier position's view, so the first
    # row's logits move; under a causal mask with last-token pooling excluded,
    # the mean would move too, so the check is on the first position's state.
    changed = ids.clone()
    changed[0, 9] = (changed[0, 9] % 49) + 1
    with torch.no_grad():
        e1, e2 = model.model.embed_tokens(ids), model.model.embed_tokens(changed)
        s1 = model.model(
            inputs_embeds=e1, attention_mask=bidirectional_masks(model.model, e1, mask)
        ).last_hidden_state
        s2 = model.model(
            inputs_embeds=e2, attention_mask=bidirectional_masks(model.model, e2, mask)
        ).last_hidden_state
    assert not torch.allclose(s1[0, 0], s2[0, 0], atol=1e-5)
