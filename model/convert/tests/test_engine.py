"""The backbone's own network and optimizers, driven by the same loop."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from model.convert.engine import ENGINES, OptimizerSet  # noqa: E402
from model.convert.muon import Muon, newton_schulz  # noqa: E402
from model.convert.native import NativeConfig, NativeModel  # noqa: E402
from model.convert.train import ConvertConfig, train  # noqa: E402

CONTEXT, VOCAB, MASK, SEP, PAD = 16, 64, 60, 3, 61


def network():
    torch.manual_seed(0)
    cfg = NativeConfig(
        vocab_size=VOCAB, n_layer=2, d_model=32, n_head=4, n_kv_head=2, d_ff=64, head_dim=8,
        max_positions=64,
    )  # fmt: skip
    return NativeModel(cfg)


def config(**overrides):
    base = dict(
        backbone="tiny", revision="abc", mask_id=MASK, separator_id=SEP, pad_id=PAD,
        vocab_size=VOCAB, total_tokens=4 * CONTEXT * 30, context=CONTEXT,
        batch_tokens=4 * CONTEXT, micro_batch=2, warmup_steps=2, checkpoint_every=5,
        engine="native", muon_lr=0.02, embed_lr=3e-2, scalar_lr=3e-3,
    )  # fmt: skip
    return ConvertConfig(**{**base, **overrides})


class Repeating:
    """Text with structure to learn: each row is one short pattern repeated."""

    def __init__(self):
        self.reads = 0

    def batch(self, step, size):
        rng = np.random.default_rng(step)
        rows = np.stack(
            [np.tile(rng.integers(4, 50, size=4), CONTEXT // 4) for _ in range(size)]
        ).astype(np.int64)
        rows[:, 7] = SEP
        self.reads += size
        return rows, {}

    def state(self):
        return {"reads": self.reads}

    def load_state(self, cursors):
        self.reads = cursors["reads"]


def test_the_native_engine_learns_under_the_shared_loop(tmp_path):
    report = train(network(), config(), Repeating(), out_dir=tmp_path, max_steps=30)
    assert report.diverged_at_step is None
    assert np.mean(report.losses[-5:]) < np.mean(report.losses[:5]) - 0.3


def test_a_native_run_resumes_with_both_optimizers_and_the_held_norms(tmp_path):
    first = network()
    train(first, config(), Repeating(), out_dir=tmp_path, max_steps=5)
    resumed_net = network()
    source = Repeating()
    report = train(resumed_net, config(), source, out_dir=tmp_path, max_steps=8)
    assert report.steps == 3
    assert source.reads == 5 * 4 + 3 * 4

    # The same eight steps in one go land on the same weights.
    straight = network()
    train(straight, config(), Repeating(), out_dir=tmp_path / "straight", max_steps=8)
    for a, b in zip(resumed_net.parameters(), straight.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_muon_holds_each_matrix_at_the_norm_it_arrived_with():
    net = network()
    matrix = net.body.layers[0].mlp.up_proj.weight
    before = matrix.norm().item()
    optimizer = ENGINES["native"].optimizers(net, config(), "cpu")
    assert isinstance(optimizer, OptimizerSet) and isinstance(optimizer.optimizers[0], Muon)
    for _ in range(5):
        ids = torch.randint(4, 50, (2, CONTEXT))
        net.logits(net.hidden(ids)).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()
    assert matrix.norm().item() == pytest.approx(before, rel=1e-5)
    # Embeddings and norms are AdamW's, never Muon's.
    muon_params = {id(p) for g in optimizer.optimizers[0].param_groups for p in g["params"]}
    assert id(net.body.embed_tokens.weight) not in muon_params
    assert all(p.ndim == 2 for g in optimizer.optimizers[0].param_groups for p in g["params"])
    every = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert every == {id(p) for p in net.parameters()}, "a parameter has no optimizer"


def test_the_schedule_scales_every_group_from_its_own_base_rate():
    optimizer = ENGINES["native"].optimizers(network(), config(), "cpu")
    optimizer.scale_lr(0.5)
    rates = sorted({round(group["lr"], 6) for group in optimizer.param_groups})
    assert rates == [0.0015, 0.01, 0.015]


def test_newton_schulz_returns_something_close_to_orthogonal():
    torch.manual_seed(0)
    out = newton_schulz(torch.randn(16, 48)).float()
    singular = torch.linalg.svdvals(out)
    assert singular.min() > 0.5 and singular.max() < 1.5


def test_a_native_checkpoint_exports_into_the_class_the_instrument_loads(tmp_path):
    """The path a rung takes: trained natively, saved, exported, scored bidirectionally."""
    pytest.importorskip("transformers")
    from transformers import AutoModelForSequenceClassification, Qwen3Config, Qwen3ForCausalLM

    from model.convert.checkpoint import latest
    from model.convert.native import load_checkpoint_into_library

    library_config = Qwen3Config(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
        rms_norm_eps=1e-5, rope_parameters={"rope_theta": 100000.0, "rope_type": "default"},
        tie_word_embeddings=True, pad_token_id=PAD,
    )  # fmt: skip
    net = network()
    train(net, config(), Repeating(), out_dir=tmp_path / "run", max_steps=5)
    payload = torch.load(latest(tmp_path / "run"), map_location="cpu", weights_only=False)

    library = Qwen3ForCausalLM(library_config)
    load_checkpoint_into_library(payload["model"], library)
    library.save_pretrained(tmp_path / "export")

    # What the instrument does with the folder.
    classifier = AutoModelForSequenceClassification.from_pretrained(
        tmp_path / "export", num_labels=3, dtype=torch.float32
    ).eval()
    trained = net.body.layers[1].mlp.down_proj.weight
    assert torch.equal(classifier.model.layers[1].mlp.down_proj.weight, trained)
    ids = torch.randint(4, 50, (2, CONTEXT))
    with torch.no_grad():
        expected = net.eval().hidden(ids)
        got = classifier.model(input_ids=ids, is_causal=False).last_hidden_state
    assert torch.allclose(got, expected, atol=1e-5)

    # A checkpoint missing a layer is refused, not half loaded.
    broken = dict(payload["model"])
    broken.pop("body.layers.0.mlp.up_proj.weight")
    with pytest.raises((ValueError, RuntimeError)):
        load_checkpoint_into_library(broken, Qwen3ForCausalLM(library_config))


def test_the_causal_objective_predicts_each_next_token_inside_its_document():
    from model.convert.train import corrupt, next_token_labels

    cfg = config(objective="clm")
    rows = np.array([[10, 11, SEP, 20, 21, PAD]])
    labels = next_token_labels(rows, cfg)
    # 10 -> 11, 11 -> SEP (a document ends), SEP predicts nothing, 20 -> 21,
    # 21 -> PAD is ignored, the last position has no next token.
    assert labels.tolist() == [[11, SEP, -100, 21, -100, -100]]
    inputs, again = corrupt(rows, cfg, 0)
    assert (inputs == rows).all() and (again == labels).all()


def test_the_causal_document_mask_hides_the_future_and_other_documents():
    from model.convert.native import dense_causal_document_mask

    ids = torch.tensor([[5, 6, SEP, 7, 8]])
    mask = dense_causal_document_mask(ids, SEP)[0, 0]
    assert mask[4, 3] and mask[4, 4] and not mask[4, 1]  # own past only, not document one
    assert mask[1, 0] and not mask[0, 1]  # no future
    assert mask[2, 0]  # the separator belongs to the document it closes


def test_the_native_causal_loss_is_next_token_cross_entropy():
    from model.convert.native import dense_causal_document_mask
    from model.convert.train import next_token_labels

    model = network()
    cfg = config(objective="clm")
    rows = np.array([[10, 11, 12, SEP, 20, 21, 22, 23]])
    ids = torch.from_numpy(rows)
    targets = torch.from_numpy(next_token_labels(rows, cfg))
    got = ENGINES["native"].loss(model, ids, ids, targets, cfg)
    hidden = model.hidden(ids, dense_causal_document_mask(ids, SEP))
    keep = targets != -100
    logits = model.logits(hidden[keep]).float()
    expected = torch.nn.functional.cross_entropy(logits, targets[keep])
    expected = expected + cfg.z_loss * (torch.logsumexp(logits, dim=-1) ** 2).mean()
    assert torch.allclose(got, expected, atol=1e-6)


def test_the_causal_objective_changes_the_fingerprint_and_trains(tmp_path):
    assert config(objective="clm").fingerprint() != config().fingerprint()
    report = train(network(), config(objective="clm", total_tokens=4 * CONTEXT * 8),
                   Repeating(), out_dir=tmp_path,
                   max_steps=8)  # fmt: skip
    assert report.losses[-1] < report.losses[0]
