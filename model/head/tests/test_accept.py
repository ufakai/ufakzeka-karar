"""Round 2's acceptance: the primary must win and no secondary may be worse after Holm."""

import numpy as np

from model.head.accept import accept, holm, paired, secondaries


def test_holm_steps_down():
    assert holm({"a": 0.01, "b": 0.02, "c": 0.5}) == {"a": True, "b": True, "c": False}
    assert holm({"a": 0.03, "b": 0.04}) == {"a": False, "b": False}


def test_a_primary_win_with_a_significantly_worse_secondary_is_not_accepted():
    win = {"interval": [0.01, 0.03]}
    fine = {"brier_choice": {"difference": 0.01, "p": 0.5}}
    worse = {"brier_choice": {"difference": 0.02, "p": 0.001}}
    better = {"brier_choice": {"difference": -0.02, "p": 0.001}}
    assert accept(win, fine)["accepted"]
    assert not accept(win, worse)["accepted"]
    assert accept(win, better)["accepted"]
    assert not accept({"interval": [-0.01, 0.03]}, fine)["accepted"]


def test_paired_bootstrap_sees_a_clear_difference():
    rng = np.random.default_rng(0)
    out = paired(np.full(200, 0.3), np.full(200, 0.1), np.mean, 200, rng)
    assert abs(out["difference"] - 0.2) < 1e-9 and out["p"] == 0.0


def test_secondaries_are_computed_per_type_and_on_dev():
    def rows(p):
        return {
            f"r{i}": {"task": "t", "target": {"a": 1.0, "b": 0.0}, "probs": {"a": p, "b": 1 - p}}
            for i in range(40)
        }

    types = {f"r{i}": "noul" for i in range(40)}
    dev = {f"r{i}": "a" for i in range(10)}
    out = secondaries(rows(0.6), rows(0.9), types, dev, draws=40, seed=0)
    assert out["brier_noul"]["difference"] > 0  # the arm is worse
    assert set(out) == {"brier_noul", "smooth_ece_noul", "dev_macro_f1_loss"}
