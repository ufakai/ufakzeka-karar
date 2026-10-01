"""Step 7's decisions on synthetic outputs whose answers are known."""

import numpy as np

from calib.step7 import abstain_map, choose_form, gate, scored


def rows(n, sharpen, seed=0, part_cycle=("fit", "fit", "fit", "selection", "heldout")):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        logits = rng.normal(size=3) * 2
        true = np.exp(logits) / np.exp(logits).sum()
        shown = logits * sharpen
        part = part_cycle[i % len(part_cycle)]
        out.append({"logprobs": (shown - np.log(np.exp(shown).sum())).tolist(),
                    "target": true.tolist(), "type": ["choice", "noul"][i % 2], "n_options": 3,
                    "row_id": f"r{i}", "task": "t", "part": part})  # fmt: skip
    return out


def test_an_overconfident_engine_gets_a_temperature_and_a_calibrated_one_does_not():
    calibrator, losses = choose_form(rows(1500, 2.0))
    assert calibrator["form"] != "none" and losses["global"] < losses["none"]
    calibrator, _ = choose_form(rows(1500, 1.0))
    assert calibrator["form"] == "none"


def test_identical_engines_pass_the_gate_with_no_change():
    a = rows(600, 1.5)
    cal, _ = choose_form(a)
    out = gate(a, a, cal, cal)
    assert out["passes"] and out["answers_changed"] == 0
    assert out["calibrated_probability_change"]["max"] == 0.0


def test_scores_and_the_abstain_map_are_produced():
    a = rows(900, 1.5)
    cal, _ = choose_form(a)
    out = scored([r for r in a if r["part"] == "heldout"], cal)
    assert set(out) == {"all", "choice", "noul"}
    assert out["all"]["calibrated"]["soft_ce"] <= out["all"]["raw"]["soft_ce"] + 1e-6
    m = abstain_map(a)
    assert len(m["map"]["x"]) == len(m["map"]["y"]) and m["heldout_augrc"] > 0


def test_rows_of_the_selection_task_are_never_fitted_on(tmp_path):
    import json

    from calib.step7 import load

    lines = [{"row_id": "a", "split": "validation", "task": "prompt_injection-conv", "type": "noul",
              "keys": ["true", "false"], "target": {"true": 1.0, "false": 0.0}},
             {"row_id": "b", "split": "validation", "task": "spam", "type": "noul",
              "keys": ["true", "false"], "target": {"true": 0.0, "false": 1.0}}]  # fmt: skip
    path = tmp_path / "outputs.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    parts = {r["row_id"]: r["part"] for r in load(path, {"a": "x", "b": "y"}, set(), set())}
    assert parts["a"] == "dev_only"
    assert parts["b"] in {"fit", "selection"}
