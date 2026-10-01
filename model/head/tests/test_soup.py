"""The seed soup averages floating weights and keeps the rest from the first run."""

import pytest

torch = pytest.importorskip("torch")

from model.head.soup import average_states  # noqa: E402


def test_floating_weights_are_averaged_and_integer_buffers_kept():
    a = {"w": torch.tensor([1.0, 3.0]), "step": torch.tensor(5)}
    b = {"w": torch.tensor([3.0, 5.0]), "step": torch.tensor(7)}
    out = average_states([a, b])
    assert torch.equal(out["w"], torch.tensor([2.0, 4.0]))
    assert int(out["step"]) == 5
