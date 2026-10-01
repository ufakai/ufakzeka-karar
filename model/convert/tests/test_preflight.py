"""Preflight: every check catches the real failure it was written for."""

import numpy as np
import pytest

from model.convert.preflight import (
    Check,
    check_attention_is_uniform,
    check_batch_fits,
    check_mask_token,
    check_model_fits_context,
    check_separator,
    check_shards,
    check_tokens_cover_budget,
    check_vocab_matches,
    describe_plan,
    raise_on_failure,
    report,
    sample_windows_are_plausible,
)


class FakeTokenizer:
    def __init__(self, ids, eos=99, unk=None, size=None):
        self.ids = ids
        self.eos_token_id = eos
        self.unk_token_id = unk
        self._size = size if size is not None else len(ids)

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, -1)

    def __len__(self):
        return self._size


class FakeConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def shards(tmp_path, sizes=(100, 100), dtype=np.uint16):
    out = []
    for i, n in enumerate(sizes):
        path = tmp_path / f"part_{i}.bin"
        np.arange(n, dtype=dtype).tofile(path)
        out.append(path)
    return out


def test_missing_or_ragged_shards_are_caught(tmp_path):
    assert not check_shards("B", [], context=10).ok
    assert not check_shards("B", [tmp_path / "gone.bin"], context=10).ok
    odd = tmp_path / "odd.bin"
    odd.write_bytes(b"\x01\x02\x03")
    assert not check_shards("B", [odd], context=1).ok
    # The real bug: a reader that could not read this format at all.
    good = check_shards("B", shards(tmp_path), context=10)
    assert good.ok and "20 windows" in good.detail


def test_a_tier_too_small_for_its_budget_is_caught(tmp_path):
    assert check_tokens_cover_budget(1_000, 500, "A").ok
    repeated = check_tokens_cover_budget(1_000, 5_000, "A")
    assert not repeated.ok and "repeated" in repeated.detail
    assert not check_tokens_cover_budget(0, 10, "A").ok


def test_the_mask_token_must_resolve_to_a_real_row():
    good = FakeTokenizer({"<|reserved_0|>": 40948})
    assert check_mask_token(good, "<|reserved_0|>", 40960).ok
    # Absent from the vocabulary.
    assert not check_mask_token(good, "[MASK]", 40960).ok
    # Silently the unknown id, which would train the model to predict nothing.
    sneaky = FakeTokenizer({"[MASK]": 3}, unk=3)
    assert not check_mask_token(sneaky, "[MASK]", 40960).ok
    # Outside the embedding matrix.
    assert not check_mask_token(FakeTokenizer({"x": 50_000}), "x", 40960).ok


def test_a_tokenizer_with_no_end_of_text_cannot_pack():
    assert check_separator(FakeTokenizer({}, eos=40944)).ok
    assert not check_separator(FakeTokenizer({}, eos=None)).ok


def test_a_context_beyond_the_model_is_caught():
    assert check_model_fits_context(FakeConfig(max_position_embeddings=4096), 1024).ok
    assert not check_model_fits_context(FakeConfig(max_position_embeddings=512), 1024).ok
    assert not check_model_fits_context(FakeConfig(), 1024).ok


def test_a_tokenizer_wider_than_the_matrix_is_caught():
    # This fails thousands of steps in, when a rare id first appears.
    assert check_vocab_matches(FakeConfig(vocab_size=40960), FakeTokenizer({}, size=40944)).ok
    assert not check_vocab_matches(FakeConfig(vocab_size=1000), FakeTokenizer({}, size=40944)).ok
    assert not check_vocab_matches(FakeConfig(), FakeTokenizer({})).ok


def test_mixed_attention_layers_are_caught():
    full = FakeConfig(layer_types=["full_attention"] * 24, use_sliding_window=False)
    assert check_attention_is_uniform(full).ok
    mixed = FakeConfig(layer_types=["full_attention", "sliding_attention"])
    assert not check_attention_is_uniform(mixed).ok
    assert not check_attention_is_uniform(FakeConfig(use_sliding_window=True)).ok


def test_a_batch_that_cannot_fit_is_refused_before_the_gpu_proves_it():
    shape = {"context": 1024, "layers": 24, "hidden": 768, "vocab_size": 40960}
    # The backbone at 64 rows on a 24 GB card: 17 GB of activations alone.
    tight = check_batch_fits(24e9, 150_000_000, micro_batch=64, **shape)
    assert not tight.ok and "micro_batch" in tight.detail
    # Eight rows fits, and the estimate now includes the logits that made the
    # measured peak 16.55 GB where the old formula predicted 5.0.
    roomy = check_batch_fits(24e9, 150_000_000, micro_batch=8, **shape)
    assert roomy.ok and "8 rows x 1024" in roomy.detail
    without_logits = 150_000_000 * 16 + 8 * 1024 * 24 * 768 * 7 * 2
    assert "GB" in roomy.detail and without_logits < 8e9
    # The earlier version of this check passed the full batch and was cheerful
    # about a run needing 176 GB, so the micro-batch is what it takes now.
    assert not check_batch_fits(24e9, 150_000_000, micro_batch=512, **shape).ok


def test_a_quadratic_attention_kernel_is_caught():
    from model.convert.preflight import check_attention_implementation

    class Model:
        def __init__(self, impl):
            self.config = FakeConfig(_attn_implementation=impl)

    assert check_attention_implementation(Model("sdpa")).ok
    assert check_attention_implementation(Model("flash_attention_2")).ok
    bad = check_attention_implementation(Model("eager"))
    assert not bad.ok and "quadratic" in bad.detail


def test_the_plan_is_printed_in_the_terms_the_ledger_uses():
    plan = describe_plan(5_000_000_000, 524_288, 1024, {"A": 0.25, "B": 0.71, "C": 0.04})
    assert plan.ok
    assert "9,536 steps" in plan.detail and "512 rows x 1024" in plan.detail
    assert "B 71%" in plan.detail


def test_windows_that_are_not_really_text_are_caught(tmp_path):
    from model.convert.stream import ShardedWindows

    # A wrong dtype or offset still yields the right shape; what it does not
    # yield is a spread of ids.
    flat = tmp_path / "flat.bin"
    np.zeros(1000, dtype=np.uint16).tofile(flat)
    assert not sample_windows_are_plausible(ShardedWindows([flat], context=100)).ok

    real = tmp_path / "real.bin"
    np.random.default_rng(0).integers(0, 40000, size=10_000).astype(np.uint16).tofile(real)
    good = sample_windows_are_plausible(ShardedWindows([real], context=100))
    assert good.ok and "distinct ids" in good.detail


def test_failures_are_all_reported_at_once_not_one_at_a_time():
    checks = [Check("a", True, "fine"), Check("b", False, "broken"), Check("c", False, "also")]
    text = report(checks)
    assert "ok  a" in text and "FAIL b" in text
    with pytest.raises(RuntimeError, match="preflight failed") as caught:
        raise_on_failure(checks)
    # Both failures named, so one launch fixes both rather than finding them in turn.
    assert "b" in str(caught.value) and "c" in str(caught.value)
    raise_on_failure([Check("a", True, "fine")])


def test_a_corpus_packed_with_a_different_separator_is_caught(tmp_path):
    from model.convert.preflight import check_separator_in_corpus
    from model.convert.stream import ShardedWindows

    tokens = np.random.default_rng(0).integers(10, 40000, size=20_000).astype(np.uint16)
    tokens[::500] = 40944
    path = tmp_path / "part.bin"
    tokens.tofile(path)
    windows = ShardedWindows([path], context=1000)
    assert check_separator_in_corpus(windows, 40944, "B").ok
    # The config's other end-of-text id: the mask would silently hide nothing.
    wrong = check_separator_in_corpus(windows, 40946, "B")
    assert not wrong.ok and "never appears" in wrong.detail


def test_the_launch_configuration_passes_the_gate_it_will_meet():
    """Read as text: modal is absent here. The run once defaulted to a kernel
    its own preflight refuses, inside a function that retries ten times."""
    import ast
    from pathlib import Path

    from model.convert.preflight import check_attention_implementation

    source = Path("model/convert/modal_app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "RUN_SETTINGS"
    }
    run = found["RUN_SETTINGS"]

    class Model:
        class config:  # noqa: N801
            _attn_implementation = run["attn_implementation"]

    assert check_attention_implementation(Model).ok
    assert f'"attn_implementation", "{run["attn_implementation"]}"' in source
    assert "raise_on_failure(checks)" not in source[source.index("def convert(") :][:4000]


def test_the_causal_objective_is_charged_for_every_position():
    """Under the causal loss every position is projected onto the vocabulary,
    so a micro-batch that fits the masked objective can fail this one."""
    from pathlib import Path

    sizes = {"parameters": 170_000_000, "micro_batch": 48, "context": 1024, "layers": 16,
             "hidden": 768, "vocab_size": 40960}  # fmt: skip
    free = 80 * 2**30
    masked = check_batch_fits(free, sizes["parameters"], **{k: v for k, v in sizes.items()
                                                            if k != "parameters"})  # fmt: skip
    causal = check_batch_fits(free, sizes["parameters"], masked_share=1.0,
                              **{k: v for k, v in sizes.items() if k != "parameters"})  # fmt: skip
    assert float(causal.detail.split()[1]) > float(masked.detail.split()[1]) + 15
    source = Path("model/convert/modal_app.py").read_text(encoding="utf-8")
    assert 'masked_share=1.0 if settings.get("objective") == "clm" else 0.3' in source


def test_the_continuation_launches_with_the_largest_probed_batch_that_fits():
    import ast
    from pathlib import Path

    source = Path("model/convert/modal_app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "continue_batch")  # fmt: skip
    scope: dict = {}
    exec(compile(ast.Module([node], []), "continue_batch", "exec"), {"H100_GIB": 79.6}, scope)
    pick = scope["continue_batch"]
    rows = [{"micro_batch": 16, "peak_gib": 30.0, "tokens_per_second": 190_000},
            {"micro_batch": 32, "peak_gib": 60.0, "tokens_per_second": 200_000},
            {"micro_batch": 48, "error": "CUDA out of memory"}]  # fmt: skip
    assert pick(rows)["micro_batch"] == 32
    rows[1]["peak_gib"] = 70.0
    assert pick(rows)["micro_batch"] == 16
    with pytest.raises(RuntimeError):
        pick([{"micro_batch": 16, "error": "x"}])
