"""Checkpoints: complete, atomic, and unwilling to resume the wrong run."""

import pytest

torch = pytest.importorskip("torch")

from model.convert.checkpoint import (  # noqa: E402
    Progress,
    fingerprint,
    latest,
    load,
    save,
)

FINGERPRINT = "abc123"


def make():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def train_a_little(model, optimizer, steps=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x = torch.randn(8, 4, generator=generator)
        loss = (model(x) ** 2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


def test_a_resumed_run_continues_identically_to_one_that_was_never_broken(tmp_path):
    unbroken, unbroken_opt = make()
    train_a_little(unbroken, unbroken_opt, steps=5, seed=1)
    train_a_little(unbroken, unbroken_opt, steps=5, seed=2)

    broken, broken_opt = make()
    train_a_little(broken, broken_opt, steps=5, seed=1)
    save(
        tmp_path,
        model=broken,
        optimizer=broken_opt,
        progress=Progress(step=5, tokens_seen=5 * 1024, cursors={"A": 3, "B": 2}),
        run_fingerprint=FINGERPRINT,
    )

    # A fresh process, as after a preemption: new objects, nothing shared.
    resumed, resumed_opt = make()
    progress = load(
        latest(tmp_path), model=resumed, optimizer=resumed_opt, run_fingerprint=FINGERPRINT
    )
    assert progress == Progress(step=5, tokens_seen=5120, cursors={"A": 3, "B": 2})
    train_a_little(resumed, resumed_opt, steps=5, seed=2)

    for a, b in zip(unbroken.parameters(), resumed.parameters(), strict=True):
        assert torch.equal(a, b)


def test_the_optimizer_moments_survive_so_the_curve_has_no_step_in_it(tmp_path):
    model, optimizer = make()
    train_a_little(model, optimizer, steps=5)
    before = optimizer.state_dict()["state"][0]["exp_avg_sq"].clone()

    save(
        tmp_path,
        model=model,
        optimizer=optimizer,
        progress=Progress(step=5, tokens_seen=0),
        run_fingerprint=FINGERPRINT,
    )
    fresh, fresh_opt = make()
    load(latest(tmp_path), model=fresh, optimizer=fresh_opt, run_fingerprint=FINGERPRINT)
    after = fresh_opt.state_dict()["state"][0]["exp_avg_sq"]
    assert torch.equal(before, after)
    assert after.abs().sum() > 0  # the moments are real, not a fresh zero


def test_a_checkpoint_from_another_run_is_refused_rather_than_loaded(tmp_path):
    model, optimizer = make()
    save(
        tmp_path,
        model=model,
        optimizer=optimizer,
        progress=Progress(step=1, tokens_seen=0),
        run_fingerprint=FINGERPRINT,
    )
    fresh, fresh_opt = make()
    with pytest.raises(ValueError, match="belongs to run"):
        load(latest(tmp_path), model=fresh, optimizer=fresh_opt, run_fingerprint="different")


def test_a_fingerprint_changes_when_anything_that_matters_changes():
    base = {"backbone": "ufakzeka", "context": 1024, "batch_tokens": 524288}
    assert fingerprint(base) == fingerprint(dict(reversed(list(base.items()))))
    assert fingerprint(base) != fingerprint({**base, "context": 512})
    assert len(fingerprint(base)) == 32


def test_a_half_written_checkpoint_is_never_returned(tmp_path):
    model, optimizer = make()
    save(
        tmp_path,
        model=model,
        optimizer=optimizer,
        progress=Progress(step=10, tokens_seen=0),
        run_fingerprint=FINGERPRINT,
    )
    # What a container preempted mid-write leaves behind.
    (tmp_path / ".step_20.pt.partial").write_bytes(b"not a checkpoint")
    assert latest(tmp_path).name == "step_10.pt"


def test_only_the_newest_checkpoints_are_kept(tmp_path):
    model, optimizer = make()
    for step in (1, 2, 3, 4):
        save(
            tmp_path,
            model=model,
            optimizer=optimizer,
            progress=Progress(step=step, tokens_seen=0),
            run_fingerprint=FINGERPRINT,
            keep=2,
        )
    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert remaining == ["step_3.pt", "step_4.pt"]
    assert latest(tmp_path).name == "step_4.pt"


def test_the_newest_is_by_step_number_not_by_name(tmp_path):
    model, optimizer = make()
    for step in (9, 100):
        save(
            tmp_path,
            model=model,
            optimizer=optimizer,
            progress=Progress(step=step, tokens_seen=0),
            run_fingerprint=FINGERPRINT,
        )
    # Sorted as text, "step_9" would win over "step_100".
    assert latest(tmp_path).name == "step_100.pt"


def test_an_empty_or_missing_directory_has_no_checkpoint(tmp_path):
    assert latest(tmp_path) is None
    assert latest(tmp_path / "never_created") is None
