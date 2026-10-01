"""Step 6's ablation arms: the sequential layout and the policy-gradient loss."""

import pytest

torch = pytest.importorskip("torch")

from model.head.head import DecisionHead, objective, policy_gradient  # noqa: E402
from model.head.pack import PAD, PREFIX, arrange, attention_mask, collate, pack, parts  # noqa: E402
from model.head.tests.test_head import PAD_ID, choice, encode, head  # noqa: E402

OPTIONS = ["kırmızı", "mavi renk", "yeşil"]


def test_sequential_positions_run_on_in_the_given_order():
    p = parts("Metin.", choice(OPTIONS), encode)
    laid = arrange(p, "sequential", ["yeşil", "kırmızı", "mavi renk"])
    assert laid.positions == list(range(len(laid.ids)))
    assert laid.segment_of == {"yeşil": 1, "kırmızı": 2, "mavi renk": 3}
    assert laid.keys == p.keys  # scores still come back in the caller's order


def test_blind_arrange_is_the_shipped_pack():
    p = parts("Metin.", choice(OPTIONS), encode)
    assert arrange(p, "blind") == pack("Metin.", choice(OPTIONS), encode)


@pytest.mark.parametrize("causal", [True, False])
def test_the_sequential_mask_lets_an_option_see_only_earlier_options(causal):
    laid = arrange(parts("Metin.", choice(OPTIONS), encode), "sequential")
    segments = collate([laid], PAD_ID).segments
    mask = attention_mask(segments, causal=causal, layout="sequential")[0, 0]
    seg = segments[0]
    first = int((seg == 1).nonzero()[0])
    second = int((seg == 2).nonzero()[-1])
    prefix = int((seg == PREFIX).nonzero()[0])
    assert mask[second, first] and not mask[first, second]
    assert not mask[prefix, first]
    if causal:
        real = seg != PAD
        plain = torch.tril(torch.ones(len(seg), len(seg), dtype=torch.bool))
        assert torch.equal(mask[real][:, real], plain[real][:, real])


def test_order_moves_the_sequential_head_and_never_the_blind_one():
    torch.manual_seed(0)
    blind, seq = head(causal=True), head(causal=True)
    seq.layout = "sequential"
    p = parts("Bugün hava güzel.", choice(OPTIONS), encode)

    def logits(model, order):
        batch = collate([arrange(p, model.layout, order)], PAD_ID)
        with torch.no_grad():
            out, _ = model(batch)
        return dict(zip(batch.keys[0], out[0].tolist(), strict=True))

    a, b = list(p.keys), list(reversed(p.keys))
    assert logits(blind, a) == logits(blind, b)
    assert logits(seq, a) != logits(seq, b)


def test_the_policy_gradient_is_unbiased_for_the_expected_reward():
    torch.manual_seed(0)
    targets = torch.tensor([[0.6, 0.3, 0.1]])
    logits = torch.tensor([[0.2, -0.1, 0.4]], requires_grad=True)
    exact = torch.autograd.grad((torch.softmax(logits, -1) * targets).sum(), logits)[0]
    generator = torch.Generator().manual_seed(0)
    estimate = torch.zeros_like(logits)
    rounds = 4000
    for _ in range(rounds):
        loss = policy_gradient(logits, targets, samples=8, generator=generator)
        estimate -= torch.autograd.grad(loss, logits)[0] / rounds
    assert torch.allclose(estimate, exact, atol=0.01)


def test_rl_goes_one_hot_where_cross_entropy_settles_on_the_target():
    targets = torch.tensor([[0.6, 0.4]])
    rl = torch.zeros(1, 2, requires_grad=True)
    ce = torch.zeros(1, 2, requires_grad=True)
    optimiser = torch.optim.SGD([rl, ce], lr=0.5)
    generator = torch.Generator().manual_seed(1)
    for _ in range(3000):
        optimiser.zero_grad()
        (policy_gradient(rl, targets, samples=8, generator=generator)
         + objective(ce, targets)).backward()  # fmt: skip
        optimiser.step()
    assert float(torch.softmax(rl.detach(), -1)[0, 0]) > 0.95
    assert float(torch.softmax(ce.detach(), -1)[0, 0]) == pytest.approx(0.6, abs=0.02)


def test_score_rewards_credit_near_levels():
    targets = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    logits = torch.tensor([[-50.0, -50.0, -50.0, 50.0]])  # always samples level 3
    generator = torch.Generator().manual_seed(0)
    probs = torch.softmax(logits, -1).detach()
    actions = torch.multinomial(probs, 4, replacement=True, generator=generator)
    assert (actions == 3).all()  # reward is 1 - |3 - 1| / 3 = 1/3 on each draw
    loss = policy_gradient(logits, targets, 4, ordinal=torch.tensor([True]),
                           generator=torch.Generator().manual_seed(0))  # fmt: skip
    assert loss.item() == pytest.approx(0.0, abs=1e-6)  # equal rewards, zero advantage


def test_the_layout_is_not_part_of_the_saved_weights():
    model = head(causal=True)
    other = DecisionHead(model.backbone, causal=True, layout="sequential")
    assert set(model.state_dict()) == set(other.state_dict())
