"""The WSD schedule: warm up, hold, anneal, and never run past the ends."""

import pytest

from model.convert.schedule import DEFAULT_DECAY_FRAC, steps_for_tokens, wsd_scale

BASE = {"warmup_steps": 100, "decay_frac": 0.2}


def test_warmup_climbs_from_near_zero_to_the_peak():
    assert wsd_scale(0, 1000, **BASE) == pytest.approx(0.01)
    assert wsd_scale(49, 1000, **BASE) == pytest.approx(0.5)
    assert wsd_scale(99, 1000, **BASE) == pytest.approx(1.0)


def test_the_stable_phase_holds_the_peak_exactly():
    for step in (100, 300, 700, 799):
        assert wsd_scale(step, 1000, **BASE) == 1.0


def test_the_tail_anneals_to_zero_and_stops_there():
    assert wsd_scale(800, 1000, **BASE) == pytest.approx(1.0)
    assert wsd_scale(1000, 1000, **BASE) == pytest.approx(0.0)
    # Past the end it stays at zero rather than going negative.
    assert wsd_scale(5000, 1000, **BASE) == 0.0


def test_the_default_decay_falls_faster_than_linear_early_in_the_tail():
    # 1-sqrt drops quickly then flattens; linear is a straight line. At the
    # quarter point of the tail the square-root shape is already at half.
    quarter = 850
    assert wsd_scale(quarter, 1000, **BASE) == pytest.approx(0.5, abs=1e-9)
    linear = wsd_scale(quarter, 1000, **BASE, shape="linear")
    assert linear == pytest.approx(0.75, abs=1e-9)
    assert wsd_scale(quarter, 1000, **BASE) < linear


def test_the_backbone_s_own_linear_schedule_can_be_reproduced():
    # Its pretraining used linear decay over the last 15 percent.
    kwargs = {"warmup_steps": 1000, "decay_frac": 0.15, "shape": "linear"}
    assert wsd_scale(0, 10_000, **kwargs) == pytest.approx(0.001)
    assert wsd_scale(5_000, 10_000, **kwargs) == 1.0
    assert wsd_scale(9_250, 10_000, **kwargs) == pytest.approx(0.5)
    assert wsd_scale(10_000, 10_000, **kwargs) == pytest.approx(0.0)


def test_a_run_can_be_left_with_a_floor_so_it_can_be_continued():
    end = wsd_scale(1000, 1000, **BASE, final_frac=0.1)
    assert end == pytest.approx(0.1)
    # The floor does not disturb the stable phase.
    assert wsd_scale(500, 1000, **BASE, final_frac=0.1) == 1.0


def test_the_default_tail_is_the_one_the_ablation_supports():
    assert DEFAULT_DECAY_FRAC == 0.20


def test_no_warmup_means_the_peak_from_the_first_step():
    assert wsd_scale(0, 100, warmup_steps=0, decay_frac=0.2) == 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"total_steps": 0, "warmup_steps": 10, "decay_frac": 0.2}, "total_steps"),
        ({"total_steps": 10, "warmup_steps": 10, "decay_frac": 1.5}, "decay_frac"),
        ({"total_steps": 10, "warmup_steps": -1, "decay_frac": 0.2}, "warmup_steps"),
        ({"total_steps": 10, "warmup_steps": 1, "decay_frac": 0.2, "shape": "cosine"}, "shape"),
    ],
)
def test_impossible_schedules_are_refused(kwargs, message):
    total = kwargs.pop("total_steps")
    with pytest.raises(ValueError, match=message):
        wsd_scale(0, total, **kwargs)


def test_a_token_budget_becomes_a_step_count():
    assert steps_for_tokens(5_000_000_000, batch_tokens=524_288) == 9536
    # Never zero steps, however small the budget.
    assert steps_for_tokens(1, batch_tokens=524_288) == 1
    with pytest.raises(ValueError, match="tokens must be positive"):
        steps_for_tokens(0, batch_tokens=1)
    with pytest.raises(ValueError, match="batch_tokens must be positive"):
        steps_for_tokens(10, batch_tokens=0)


def test_a_decay_branch_turns_at_the_rung_for_every_rung_we_might_pick():
    from model.convert.schedule import decay_branch_steps

    for decay_frac in (0.1, 0.2, 0.25):
        for rung_step in (1, 7, 8, 1907, 4768, 9536, 13351):
            total = decay_branch_steps(rung_step, decay_frac)
            assert int(total * (1.0 - decay_frac)) == rung_step
            assert total > rung_step
