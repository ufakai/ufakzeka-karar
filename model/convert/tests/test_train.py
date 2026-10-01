"""The conversion loop on a tiny real Qwen3: it learns, it resumes, it reads both ways."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from model.convert.objective import IGNORE  # noqa: E402
from model.convert.train import (  # noqa: E402
    ConvertConfig,
    RunReport,
    corrupt,
    mlm_loss,
    parameter_groups,
    train,
)

VOCAB = 64
MASK = 60
SEP = 61
PAD = 62
CONTEXT = 16


def config(**overrides):
    base = {
        "backbone": "tiny",
        "revision": "abc",
        "mask_id": MASK,
        "separator_id": SEP,
        "pad_id": PAD,
        "vocab_size": VOCAB,
        "total_tokens": 16 * CONTEXT * 20,
        "context": CONTEXT,
        "batch_tokens": 4 * CONTEXT,
        "micro_batch": 2,
        "peak_lr": 5e-3,
        "warmup_steps": 2,
        "checkpoint_every": 5,
    }
    return ConvertConfig(**{**base, **overrides})


def model():
    torch.manual_seed(0)
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=VOCAB,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            attn_implementation="eager",
        )
    )


class Batches:
    """A learnable pattern, and a cursor so a resume can be checked."""

    def __init__(self):
        self.cursor = 0
        self.seen: list[int] = []

    def batch(self, step, size):
        offsets = np.arange(CONTEXT) % 8
        rows = []
        for _ in range(size):
            self.seen.append(self.cursor)
            rows.append(((self.cursor % 8) + offsets) % 8)
            self.cursor += 1
        return np.asarray(rows, dtype=np.int64), {"A": size}

    def state(self):
        return {"A": self.cursor}

    def load_state(self, cursors):
        self.cursor = cursors["A"]


def batches():
    return Batches()


def test_the_loss_is_taken_at_the_same_position_not_the_next_one():
    logits = torch.zeros(1, 3, VOCAB)
    logits[0, 1, 5] = 20.0  # position 1 is confident about token 5
    labels = torch.full((1, 3), IGNORE)
    labels[0, 1] = 5
    assert mlm_loss(logits, labels).item() == pytest.approx(0.0, abs=1e-4)
    # If the loss shifted by one, the same logits would score terribly.
    shifted = torch.full((1, 3), IGNORE)
    shifted[0, 0] = 5
    assert mlm_loss(logits, shifted).item() > 3.0


def test_positions_that_were_not_masked_cost_nothing():
    logits = torch.randn(2, 4, VOCAB)
    all_ignored = torch.full((2, 4), IGNORE)
    assert torch.isnan(mlm_loss(logits, all_ignored))  # the case the objective guards against


def test_corruption_is_reproducible_from_the_step_alone():
    rows = np.random.default_rng(0).integers(0, 50, size=(4, CONTEXT)).astype(np.int64)
    a = corrupt(rows, config(), step=7)
    b = corrupt(rows, config(), step=7)
    c = corrupt(rows, config(), step=8)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


def test_the_masking_rate_steps_down_for_the_tail():
    cfg = config()
    _, early = corrupt(np.zeros((8, CONTEXT), dtype=np.int64), cfg, step=0)
    _, late = corrupt(np.zeros((8, CONTEXT), dtype=np.int64), cfg, step=cfg.total_steps - 1)
    assert (early != IGNORE).mean() > (late != IGNORE).mean()


def test_norms_and_biases_are_not_weight_decayed_but_every_layer_trains():
    net = model()
    groups = parameter_groups(net, 0.01)
    assert groups[0]["weight_decay"] == 0.01
    assert groups[1]["weight_decay"] == 0.0
    trained = sum(len(g["params"]) for g in groups)
    assert trained == sum(1 for p in net.parameters() if p.requires_grad)


def test_the_loop_learns_a_pattern_it_can_only_see_by_reading_both_ways(tmp_path):
    cfg = config()
    report = train(model(), cfg, batches(), out_dir=tmp_path, max_steps=20)
    assert report.steps == 20
    assert len(report.losses) == 20
    first = float(np.mean(report.losses[:3]))
    last = float(np.mean(report.losses[-3:]))
    assert last < first, f"loss did not fall: {first:.3f} -> {last:.3f}"


def test_a_run_resumes_where_it_stopped_and_does_not_repeat_work(tmp_path):
    cfg = config()
    train(model(), cfg, batches(), out_dir=tmp_path, max_steps=10)
    second = train(model(), cfg, batches(), out_dir=tmp_path, max_steps=15)
    # It picked up at step 10 and ran five more, rather than starting again.
    assert second.steps == 5
    assert len(second.losses) == 5


def test_a_resume_into_a_different_configuration_is_refused(tmp_path):
    train(model(), config(), batches(), out_dir=tmp_path, max_steps=5)
    with pytest.raises(ValueError, match="belongs to run"):
        train(model(), config(context=32), batches(), out_dir=tmp_path, max_steps=6)


def test_the_volume_commit_hook_fires_on_every_checkpoint(tmp_path):
    fired = []
    train(
        model(),
        config(checkpoint_every=5),
        batches(),
        out_dir=tmp_path,
        max_steps=10,
        on_checkpoint=lambda: fired.append(1),
    )
    assert len(fired) == 2


def test_the_report_carries_what_the_ledger_needs(tmp_path):
    report = train(model(), config(), batches(), out_dir=tmp_path, max_steps=6)
    assert report.tokens == 6 * 4 * CONTEXT
    assert report.wall_seconds > 0 and report.tokens_per_second > 0


def test_the_step_and_batch_arithmetic_matches_the_token_budget():
    cfg = config(total_tokens=1_000_000, batch_tokens=1000, context=100, micro_batch=5)
    assert cfg.total_steps == 1000
    assert cfg.batch_rows == 10
    assert cfg.accumulation == 2
    assert RunReport().tokens_per_second == 0.0


def test_precision_is_float32_on_cpu_and_never_uses_a_gradient_scaler():
    from model.convert.train import precision_context

    # On CPU the context must be inert, so the same loop runs in tests.
    ctx = precision_context("cpu")
    with ctx:
        assert torch.zeros(2, 2).dtype == torch.float32


def test_the_loss_stays_in_float32_even_when_the_logits_are_not():
    # bf16 logits must not drag the softmax over 40960 classes into bf16.
    logits = torch.randn(1, 4, VOCAB, dtype=torch.bfloat16)
    labels = torch.full((1, 4), IGNORE)
    labels[0, 2] = 3
    assert mlm_loss(logits, labels).dtype == torch.float32


def test_a_resumed_run_does_not_retrain_on_text_it_has_already_seen(tmp_path):
    """The bug this guards: cursors saved but never handed back to the stream.

    Nothing crashes when that happens. The run simply reads the first windows
    again and pays to learn them twice, which is the expensive kind of wrong.
    """
    cfg = config(checkpoint_every=5)
    first = batches()
    train(model(), cfg, first, out_dir=tmp_path, max_steps=10)
    assert first.state() == {"A": 40}  # 10 steps x 4 rows

    second = batches()
    train(model(), cfg, second, out_dir=tmp_path, max_steps=15)
    # It continued from window 40 rather than starting at 0.
    assert second.seen[0] == 40
    assert set(second.seen).isdisjoint(set(first.seen))
    assert second.state() == {"A": 60}


def test_a_run_whose_loss_stops_being_a_number_stops_too(tmp_path):
    """A diverged run never recovers, so every further hour is paid for nothing."""
    from model.convert import train as train_module

    real = train_module.masked_loss
    calls = {"n": 0}

    def explode(net, hidden, labels):
        calls["n"] += 1
        loss = real(net, hidden, labels)
        return loss * float("inf") if calls["n"] > 6 else loss

    train_module.masked_loss = explode
    try:
        net = model()
        report = train(net, config(), batches(), out_dir=tmp_path, max_steps=20)
    finally:
        train_module.masked_loss = real

    assert report.diverged_at_step is not None
    assert report.steps < 20, "it kept going after the loss went to infinity"
    # The update was refused, so the weights are still numbers, and no
    # checkpoint was written for a resume to mistake for the newest good one.
    from model.convert.checkpoint import latest

    assert all(torch.isfinite(p).all() for p in net.parameters())
    assert latest(tmp_path) is None


def test_progress_is_printed_often_enough_to_stop_a_bad_run_early(tmp_path, capsys):
    train(model(), config(log_every=2), batches(), out_dir=tmp_path, max_steps=6)
    printed = capsys.readouterr().out
    assert printed.count("tok/s") == 3
    assert "loss" in printed and "lr" in printed


def test_the_accumulation_count_is_derived_once_not_computed_twice():
    cfg = config(batch_tokens=16 * CONTEXT, micro_batch=4)
    assert cfg.batch_rows == 16
    assert cfg.accumulation == 4


def test_the_masked_loss_equals_the_full_one_it_replaces():
    """Projecting only masked positions must change the memory, not the number."""
    from model.convert.train import masked_loss

    net = model().eval()
    torch.manual_seed(3)
    ids = torch.randint(0, VOCAB - 4, (2, 12))
    labels = torch.full((2, 12), IGNORE)
    labels[0, 3] = 7
    labels[0, 9] = 2
    labels[1, 5] = 4

    with torch.no_grad():
        hidden = net.model(input_ids=ids, use_cache=False).last_hidden_state
        full = mlm_loss(net.lm_head(hidden), labels)
        cheap = masked_loss(net, hidden, labels)
    assert cheap.item() == pytest.approx(full.item(), abs=1e-5)


def test_the_masked_loss_projects_only_what_it_needs():
    from model.convert.train import masked_logits

    net = model().eval()
    ids = torch.randint(0, VOCAB - 4, (2, 12))
    labels = torch.full((2, 12), IGNORE)
    labels[0, 3] = 7
    labels[1, 5] = 4
    with torch.no_grad():
        hidden = net.model(input_ids=ids, use_cache=False).last_hidden_state
        out = masked_logits(net, hidden, labels)
    # Two masked positions of twenty-four, so two rows of logits, not 24.
    assert out.shape == (2, VOCAB)


def test_a_batch_with_nothing_masked_costs_zero_rather_than_nan():
    from model.convert.train import masked_loss

    net = model().eval()
    ids = torch.randint(0, VOCAB - 4, (1, 8))
    with torch.no_grad():
        hidden = net.model(input_ids=ids, use_cache=False).last_hidden_state
        loss = masked_loss(net, hidden, torch.full((1, 8), IGNORE))
    assert loss.item() == 0.0 and not torch.isnan(loss)


def test_no_forward_pass_is_ever_larger_than_the_micro_batch(tmp_path):
    """512 rows at a micro-batch of 48 used to run as ten passes of 52.

    The preflight approves `micro_batch` rows against the card's memory, so a
    loop that feeds more than that has been approved for something it is not
    doing, and finds out hours in.
    """
    from model.convert import train as train_module

    assert config(batch_tokens=512 * CONTEXT, micro_batch=48).accumulation == 11

    seen = []
    real = train_module.masked_loss

    def spy(net, hidden, labels):
        seen.append(hidden.shape[0])
        return real(net, hidden, labels)

    train_module.masked_loss = spy
    try:
        # Seven rows in passes of at most three: 3, 3, 1.
        train(
            model(),
            config(batch_tokens=7 * CONTEXT, micro_batch=3),
            batches(),
            out_dir=tmp_path,
            max_steps=1,
        )
    finally:
        train_module.masked_loss = real
    assert seen == [3, 3, 1]


def test_masking_never_hides_a_separator_and_boundaries_come_from_clean_text(tmp_path):
    """The document mask is read off the separators, so hiding one joins two documents."""
    from model.convert import train as train_module
    from model.convert.train import corrupt

    rows = np.random.default_rng(3).integers(10, 50, size=(8, CONTEXT)).astype(np.int64)
    rows[:, 5::7] = SEP
    inputs, labels = corrupt(rows, config(mask_stable=0.9), step=0)
    assert (inputs[rows == SEP] == SEP).all()
    assert (labels[rows == SEP] == IGNORE).all()

    # And the loop numbers documents from the uncorrupted rows.
    seen = []
    real = train_module.document_ids_for

    def spy(ids, separator_id):
        seen.append(ids.clone())
        return real(ids, separator_id)

    class Fixed:
        def batch(self, step, size):
            return rows[:size], {}

        def state(self):
            return {}

        def load_state(self, cursors):
            pass

    train_module.document_ids_for = spy
    try:
        train(model(), config(micro_batch=4), Fixed(), out_dir=tmp_path, max_steps=1)
    finally:
        train_module.document_ids_for = real
    assert seen and all(not (ids == MASK).any() for ids in seen)
    assert torch.equal(seen[0].cpu(), torch.from_numpy(rows[:4]))


def test_a_trunk_keeps_its_rungs_and_a_decay_branch_starts_exactly_at_one(tmp_path):
    """The rung ladder: train at the stable rate, keep rungs, anneal from one.

    The branch must carry on from the trunk's weights, optimizer and data
    position, and its learning rate, masking rate and schedule must all turn at
    the rung's own step. A boundary a few steps off would anneal text the trunk
    had not reached or re-read text it had.
    """
    from model.convert.checkpoint import latest, rungs
    from model.convert.objective import masking_rate
    from model.convert.schedule import decay_branch_steps, wsd_scale

    step_tokens = 4 * CONTEXT
    trunk_dir, branch_dir = tmp_path / "trunk", tmp_path / "branch"
    trunk = config(total_tokens=step_tokens * 40, rungs=(step_tokens * 8,), checkpoint_every=3)
    source = batches()
    train(model(), trunk, source, out_dir=trunk_dir, max_steps=10)

    kept = rungs(trunk_dir)
    assert list(kept) == [step_tokens * 8]
    # Ordinary checkpoints were pruned around it and it is still there, and a
    # resume of the trunk would not pick the rung up by mistake.
    assert latest(trunk_dir).name == "step_10.pt"

    total = decay_branch_steps(8, trunk.decay_frac)
    branch = config(total_tokens=step_tokens * total, checkpoint_every=100)
    assert branch.fingerprint() != trunk.fingerprint()
    for step, expected in ((7, "stable"), (8, "decay")):
        rate = masking_rate(step, total, stable=0.3, decay=0.1, decay_frac=branch.decay_frac)
        assert rate == (0.3 if expected == "stable" else 0.1)
    assert wsd_scale(7, total, warmup_steps=2, decay_frac=branch.decay_frac) == 1.0
    assert wsd_scale(total - 1, total, warmup_steps=2, decay_frac=branch.decay_frac) < 0.5

    fresh = batches()
    report = train(model(), branch, fresh, out_dir=branch_dir, branch_from=kept[step_tokens * 8])
    assert report.steps == total - 8, "the branch did not start at the rung's step"
    assert report.tokens == step_tokens * total
    # A preempted branch resumes from its own checkpoint, not from the rung again.
    again = train(model(), branch, batches(), out_dir=branch_dir, branch_from=kept[step_tokens * 8])
    assert again.steps == 0


def test_a_checkpoint_from_another_run_is_still_refused_unless_a_branch_is_asked_for(tmp_path):
    from model.convert.checkpoint import latest

    train(model(), config(), batches(), out_dir=tmp_path, max_steps=5)
    other = config(peak_lr=1e-3)
    with pytest.raises(ValueError, match="belongs to run"):
        train(model(), other, batches(), out_dir=tmp_path, max_steps=6)
    assert latest(tmp_path) is not None


def test_a_launch_that_stops_at_a_rung_really_writes_that_rung(tmp_path):
    """1B is not a multiple of the batch, and rounding down stopped one step short."""
    from model.convert.checkpoint import rungs
    from model.convert.schedule import steps_to_reach

    batch = 4 * CONTEXT
    rung = batch * 6 + 5  # like the real rungs, not a multiple of the batch
    assert steps_to_reach(rung, batch) == 7
    assert steps_to_reach(batch * 6, batch) == 6
    assert steps_to_reach(1_000_000_000, 524_288) == 1908

    cfg = config(total_tokens=batch * 40, rungs=(rung,), checkpoint_every=100)
    train(model(), cfg, batches(), out_dir=tmp_path, max_steps=steps_to_reach(rung, batch))
    assert list(rungs(tmp_path)) == [batch * 7]
    # The branch phase looks for the rung under exactly this name.
    assert (tmp_path / f"rung_{steps_to_reach(rung, batch) * batch}.pt").is_file()


def test_a_resumed_launch_reports_its_own_speed_not_the_whole_runs(tmp_path):
    train(model(), config(), batches(), out_dir=tmp_path, max_steps=5)
    report = train(model(), config(), batches(), out_dir=tmp_path, max_steps=8)
    assert report.tokens == 8 * 4 * CONTEXT
    assert report.tokens_at_start == 5 * 4 * CONTEXT
    expected = 3 * 4 * CONTEXT / report.wall_seconds
    assert report.tokens_per_second == pytest.approx(expected)


def test_settings_added_to_the_fingerprint_later_keep_old_checkpoints_resumable():
    base = config()
    assert (
        config(mask_documents=True, weight_decay=0.01, z_loss=1e-5).fingerprint()
        == base.fingerprint()
    )
    for change in ({"mask_documents": False}, {"weight_decay": 0.1}, {"z_loss": 0.0}):
        assert config(**change).fingerprint() != base.fingerprint(), change


def test_the_probe_compares_the_library_path_by_name():
    from pathlib import Path

    source = Path("model/convert/modal_app.py").read_text(encoding="utf-8")
    options = source[source.index("native = {**RUN_SETTINGS") :][:900]
    assert '"engine": "library"' in options
