"""The head's guarantees, tested on a small network with the real code paths."""

import pytest

torch = pytest.importorskip("torch")

from pydantic import TypeAdapter  # noqa: E402

from model.convert.native import NativeConfig, NativeModel  # noqa: E402
from model.head.head import DecisionHead, objective, soft_targets  # noqa: E402
from model.head.pack import PAD, PREFIX, attention_mask, collate, pack  # noqa: E402
from schema.questions import Question  # noqa: E402

PAD_ID = 0
QUESTION = TypeAdapter(Question)


def encode(text: str) -> list[int]:
    return [4 + (ord(c) % 90) for c in text]


def choice(options: list[str]):
    return QUESTION.validate_python(
        {"type": "choice", "instructions": "Hangisi?", "criteria": {o: None for o in options}}
    )


def head(causal: bool, pooling="mean"):
    torch.manual_seed(0)
    cfg = NativeConfig(
        vocab_size=97, n_layer=2, d_model=32, n_head=4, n_kv_head=2, d_ff=64, head_dim=8,
        max_positions=512,
    )  # fmt: skip
    return DecisionHead(NativeModel(cfg), causal=causal, pooling=pooling).eval()


def scores(model, state, question):
    packed = pack(state, question, encode)
    batch = collate([packed], PAD_ID)
    with torch.no_grad():
        logits, _ = model(batch)
    return dict(zip(batch.keys[0], logits[0].tolist(), strict=True))


def test_options_share_positions_and_follow_the_prefix():
    packed = pack("Metin burada.", choice(["kırmızı", "mavi", "yeşil"]), encode)
    prefix_length = packed.segments.count(PREFIX)
    for segment in (1, 2, 3):
        first = packed.segments.index(segment)
        assert packed.positions[first] == prefix_length


@pytest.mark.parametrize("causal", [False, True])
def test_the_prefix_never_sees_an_option_and_options_never_see_each_other(causal):
    a = pack("Kısa metin.", choice(["a seçeneği", "b"]), encode)
    b = pack("Daha uzun bir metin burada.", choice(["x", "y", "z"]), encode)
    segments = collate([a, b], PAD_ID).segments
    allowed = attention_mask(segments, causal=causal)[:, 0]
    for row in range(2):
        seg = segments[row]
        for q in range(len(seg)):
            for k in range(len(seg)):
                sq, sk = int(seg[q]), int(seg[k])
                if sq == PAD:
                    assert bool(allowed[row, q, k]) == (q == k)
                    continue
                if sk == PAD or (sk != PREFIX and sk != sq):
                    assert not allowed[row, q, k], (row, q, k)
                if sq == PREFIX and sk != PREFIX:
                    assert not allowed[row, q, k]
                if sk == PREFIX and not causal:
                    assert allowed[row, q, k]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("pooling", ["mean", "last"])
def test_option_order_cannot_change_a_single_bit(causal, pooling):
    model = head(causal, pooling)
    options = ["fatura", "teknik destek", "iptal", "adres değişikliği", "diğer"]
    reference = scores(model, "Faturam iki kez kesildi.", choice(options))
    for order in (options[::-1], options[2:] + options[:2], sorted(options)):
        got = scores(model, "Faturam iki kez kesildi.", choice(order))
        assert got == reference, "a permutation changed the output"


def test_adding_an_option_leaves_the_others_scores_alone():
    model = head(causal=False)
    three = scores(model, "Kargo gelmedi.", choice(["kargo", "iade", "fatura"]))
    four = scores(model, "Kargo gelmedi.", choice(["kargo", "iade", "fatura", "başka"]))
    for key, value in three.items():
        assert four[key] == pytest.approx(value, abs=1e-5)


def test_an_example_is_scored_the_same_alone_or_in_a_padded_batch():
    model = head(causal=False)
    short = pack("Kısa.", choice(["a", "b"]), encode)
    long = pack(
        "Bu çok daha uzun bir metin, dolgu gerektirir.", choice(["x", "y", "z", "w"]), encode
    )
    with torch.no_grad():
        alone, _ = model(collate([short], PAD_ID))
        together, _ = model(collate([short, long], PAD_ID))
    assert torch.allclose(alone[0, :2], together[0, :2], atol=1e-5)
    assert torch.isinf(together[0, 2:]).all(), "empty slots must score -inf"


def test_both_objectives_are_minimised_at_the_target():
    target = torch.tensor([[0.7, 0.2, 0.1]])
    at_target = torch.log(target)
    elsewhere = torch.tensor([[2.0, 0.0, -1.0]])
    for kind in ("ce", "brier"):
        assert objective(at_target, target, kind) < objective(elsewhere, target, kind)
    padded_logits = torch.tensor([[0.0, 0.0, float("-inf")]])
    padded_target = torch.tensor([[0.5, 0.5, 0.0]])
    assert torch.isfinite(objective(padded_logits, padded_target, "ce"))


def test_the_head_learns_a_toy_decision():
    """Which option names the fruit in the text: a task only reading can solve."""
    torch.manual_seed(1)
    model = head(causal=False).train()
    fruits = ["elma", "armut", "kiraz", "üzüm"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for step in range(120):
        answer = fruits[step % 4]
        question = choice(fruits[step % 3 :] + fruits[: step % 3])
        packed = pack(f"Bugün {answer} yedim.", question, encode)
        batch = collate([packed], PAD_ID)
        target = soft_targets(batch, [{f: float(f == answer) for f in fruits}])
        logits, _ = model(batch)
        loss = objective(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert sum(losses[-20:]) / 20 < 0.5 * sum(losses[:20]) / 20


def test_no_module_in_the_head_package_shadows_a_top_level_package():
    """Modal puts the entrypoint's folder on the import path, so a file named
    model.py there made `import model` find it instead of the project package."""
    from pathlib import Path

    names = {p.stem for p in Path("model/head").glob("*.py")}
    assert not names & {"model", "data", "schema", "calib", "bench"}


def test_a_long_text_is_cut_and_the_question_is_kept():
    from model.head.pack import pack

    question = choice(["kargo", "iade"])
    long = pack("kelime " * 2000, question, encode, max_prefix=64)
    short = pack("kısa bir metin", question, encode, max_prefix=64)
    asked = encode("\n\nSoru: " + question.instructions)
    prefix = [t for t, s in zip(long.ids, long.segments, strict=True) if s == 0]
    assert len(prefix) == 64
    assert prefix[-len(asked) :] == asked
    whole = encode("kısa bir metin\n\nSoru: " + question.instructions)
    assert [t for t, s in zip(short.ids, short.segments, strict=True) if s == 0] == whole


def test_the_ordinal_term_charges_far_levels_more_and_keeps_the_target_optimal():
    from model.head.head import objective

    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    near = torch.tensor([[0.0, 2.0, -9.0, -9.0]])
    far = torch.tensor([[0.0, -9.0, -9.0, 2.0]])
    flag = torch.tensor([True])
    assert objective(near, target) == pytest.approx(objective(far, target).item(), rel=1e-4)
    assert objective(near, target, ordinal=flag) < objective(far, target, ordinal=flag)
    soft = torch.tensor([[0.1, 0.6, 0.3, 0.0]])
    at = torch.log(soft.clamp(min=1e-9))
    for other in (torch.tensor([[0.0, 1.0, 0.0, -2.0]]), torch.tensor([[-1.0, 0.5, 0.2, -5.0]])):
        assert objective(at, soft, ordinal=flag) < objective(other, soft, ordinal=flag)


def test_weights_set_each_question_s_share_of_the_mean():
    from model.head.head import objective

    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    good, bad = objective(logits[:1], target[:1]), objective(logits[1:], target[1:])
    mixed = objective(logits, target, weights=torch.tensor([3.0, 1.0]))
    assert mixed == pytest.approx(((3 * good + bad) / 4).item(), rel=1e-5)
