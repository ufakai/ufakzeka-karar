"""Learning-rate choice and results rows, on synthetic run folders."""

import json

import numpy as np
import pytest

pytest.importorskip("probmetrics")
pytest.importorskip("sklearn")

from model.instrument.aggregate import (  # noqa: E402
    RUN_FACTS,
    chosen_rate,
    row_for_run,
    rows_for_specs,
)
from model.instrument.plan import LEARNING_RATES, RunSpec, sweep_specs  # noqa: E402

K = 3


def logits(n, separation, scale, seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, K, n).astype(np.int64)
    evidence = np.eye(K)[labels] * separation + rng.normal(size=(n, K))
    return (evidence * separation * scale).astype(np.float32), labels


def write_run(folder, points, *, n=2000, with_test=True, seed=0, extra=None, spec=None):
    """points: {step: (separation, scale)}. `spec` records the treatment a real run would."""
    folder.mkdir(parents=True)
    labels = None
    for step, (separation, scale) in points.items():
        values, labels = logits(n, separation, scale, seed)
        np.save(folder / f"logits_validation_step{step}.npy", values)
        if with_test:
            test_values, test_labels = logits(n, separation, scale, seed + 1000)
            np.save(folder / f"logits_test_step{step}.npy", test_values)
            np.save(folder / "labels_test.npy", test_labels)
    np.save(folder / "labels_validation.npy", labels)
    meta = {key: None for key in RUN_FACTS}
    meta.update(
        {
            # Identity is left unrecorded unless a spec is given, the way a
            # record from before these fields were checked reads.
            "backbone": None,
            "revision": None,
            "dataset": None,
            "total_steps": max(points),
            "eval_steps": sorted(points),
            "wall_seconds": 1.0,
            "max_length": 32,
            "truncated_share": {"train": 0.0, "validation": 0.0, "test": 0.0},
        }
    )
    if spec is not None:
        from model.instrument.plan import BY_KEY

        meta.update(
            dataset=spec.dataset,
            backbone=BY_KEY[spec.backbone].model_id,
            revision=BY_KEY[spec.backbone].revision,
            attention=spec.attention,
            pooling=spec.pooling,
            learning_rate=spec.learning_rate,
            seed=spec.seed,
        )
    meta.update(extra or {})
    (folder / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    return folder


def test_the_learning_rate_with_the_best_refinement_wins(tmp_path):
    # 3e-5 separates the classes best. 1e-4 looks calm but learns least.
    quality = {
        # The two rates added at the bottom: too small to learn much in the
        # steps available, which is what put a floor under the grid.
        2e-6: 0.3,
        5e-6: 0.5,
        1e-5: 0.7,
        2e-5: 1.0,
        3e-5: 1.8,
        5e-5: 1.4,
        8e-5: 1.0,
        1e-4: 0.4,
        2e-4: 0.3,
        3e-4: 0.2,
        5e-4: 0.15,
        1e-3: 0.1,
    }
    for spec in sweep_specs(["massive_tr"]):
        if spec.backbone != "ufakzeka":
            continue
        separation = quality[spec.learning_rate]
        # A high rate is also overconfident, which the rule must see past.
        scale = 1.0 if spec.learning_rate < 5e-5 else 5.0
        write_run(tmp_path / spec.path, {50: (separation * 0.8, scale), 100: (separation, scale)})

    got = chosen_rate(tmp_path, "massive_tr", "ufakzeka")
    assert got["learning_rate"] == 3e-5
    assert got["at_grid_edge"] is False
    assert set(got["scores"]) == {f"{rate:.0e}" for rate in LEARNING_RATES}
    assert got["estimator"].startswith("refinement_logloss")


def test_a_winner_at_the_end_of_the_grid_is_flagged(tmp_path):
    for spec in sweep_specs(["massive_tr"]):
        if spec.backbone != "berturk":
            continue
        # Monotone in the rate, so the best is the largest, at the grid's edge.
        separation = 0.5 + 3000 * spec.learning_rate
        write_run(tmp_path / spec.path, {100: (separation, 1.0)})
    got = chosen_rate(tmp_path, "massive_tr", "berturk")
    assert got["learning_rate"] == max(LEARNING_RATES)
    assert got["at_grid_edge"] is True


def test_a_diverged_rate_is_recorded_and_does_not_stop_the_choice(tmp_path):
    # The highest rate blew up before its first evaluation point, so it has no
    # logits at all. Selection must still answer, and must say what happened.
    for spec in sweep_specs(["massive_tr"]):
        if spec.backbone != "ufakzeka":
            continue
        folder = tmp_path / spec.path
        if spec.learning_rate == max(LEARNING_RATES):
            folder.mkdir(parents=True)
            (folder / "run.json").write_text(json.dumps({"diverged_at_step": 3}), encoding="utf-8")
            continue
        write_run(folder, {100: (0.4 + 4000 * spec.learning_rate, 1.0)})

    got = chosen_rate(tmp_path, "massive_tr", "ufakzeka")
    assert f"{max(LEARNING_RATES):.0e}" in got["unusable"]
    assert f"{max(LEARNING_RATES):.0e}" not in got["scores"]
    assert got["learning_rate"] in LEARNING_RATES
    # The best usable rate is the largest one that produced logits, but that is
    # not the grid's ceiling, so it is not a reason to widen the grid.
    assert got["at_grid_edge"] is False


def test_every_rate_unusable_is_an_error(tmp_path):
    for spec in sweep_specs(["massive_tr"]):
        if spec.backbone != "ufakzeka":
            continue
        folder = tmp_path / spec.path
        folder.mkdir(parents=True)
        (folder / "run.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="every learning rate was unusable"):
        chosen_rate(tmp_path, "massive_tr", "ufakzeka")


def test_a_missing_sweep_run_is_an_error_not_a_quiet_winner(tmp_path):
    for spec in sweep_specs(["massive_tr"]):
        if spec.backbone == "ufakzeka" and spec.learning_rate != 1e-4:
            write_run(tmp_path / spec.path, {100: (1.0, 1.0)})
    with pytest.raises(FileNotFoundError, match="1e-04"):
        chosen_rate(tmp_path, "massive_tr", "ufakzeka")


def test_a_results_row_reports_the_selected_point_and_carries_its_conditions(tmp_path):
    spec = RunSpec("massive_tr", "ufakzeka", 3e-5, 4)
    folder = write_run(
        tmp_path / spec.path,
        {40: (0.8, 1.0), 80: (1.6, 6.0), 120: (1.5, 6.0)},
        extra={"git_commit": "c0ffee", "gpu_name": "NVIDIA L4", "parameters_total": 182495232},
    )
    row = row_for_run(folder, spec)

    assert (row["dataset"], row["backbone"], row["seed"]) == ("massive_tr", "ufakzeka", 4)
    assert row["learning_rate"] == 3e-5 and row["pooling"] == "stock"
    # The sharpest point, not the one raw loss would pick.
    assert row["selection"]["step"] == 80
    assert row["selection"]["audit"]["raw_logloss"] == 40
    assert row["selection"]["points"] == 3
    # Metrics come in both states, and scaling cannot move an argmax.
    assert row["raw"]["macro_f1"] == pytest.approx(row["calibrated"]["macro_f1"])
    assert row["calibrated"]["brier"] < row["raw"]["brier"]
    assert row["brier_improvement"] > 0
    assert 0 < row["inverse_temperature"] < 1
    assert row["run"]["git_commit"] == "c0ffee" and row["run"]["gpu_name"] == "NVIDIA L4"
    assert set(row["run"]) == set(RUN_FACTS)


def test_rows_are_built_for_finished_runs_and_the_rest_are_named(tmp_path):
    specs = [RunSpec("massive_tr", "ufakzeka", 3e-5, seed) for seed in (1, 4, 21)]
    write_run(tmp_path / specs[0].path, {100: (1.2, 1.0)})
    write_run(tmp_path / specs[2].path, {100: (1.2, 1.0)}, seed=7)
    rows, absent = rows_for_specs(tmp_path, specs)
    assert [row["seed"] for row in rows] == [1, 21]
    assert absent == [specs[1].path]


def test_a_run_without_test_logits_cannot_produce_a_row(tmp_path):
    # This is what a phase-one folder looks like. It must fail loudly rather
    # than silently reporting validation numbers as test numbers.
    spec = RunSpec("massive_tr", "ufakzeka", 3e-5, 1)
    folder = write_run(tmp_path / spec.path, {100: (1.2, 1.0)}, with_test=False)
    with pytest.raises(FileNotFoundError):
        row_for_run(folder, spec)


def test_selection_and_rows_keep_the_two_arms_apart(tmp_path):
    """The whole chain on synthetic runs: sweep, select, row, table.

    If any link loses the arm, the bidirectional runs either overwrite the
    committed causal ones or get averaged into them, and the comparison the
    arm exists to make disappears without an error.
    """
    from model.instrument.plan import sweep_specs
    from model.instrument.table import gate_numbers, summarise

    for arm, quality in (("causal", 1.0), ("bidirectional", 1.6)):
        for spec in sweep_specs(["massive_tr"], attention=arm):
            if spec.backbone != "ufakzeka":
                continue
            write_run(tmp_path / spec.path, {100: (quality, 1.0)}, spec=spec)

    causal = chosen_rate(tmp_path, "massive_tr", "ufakzeka", "causal")
    bidi = chosen_rate(tmp_path, "massive_tr", "ufakzeka", "bidirectional")
    assert causal["attention"] == "causal"
    assert bidi["attention"] == "bidirectional"
    assert len(causal["scores"]) == len(bidi["scores"]) == len(LEARNING_RATES)

    # A results row carries the arm, and the table keeps them as separate rows.
    spec = RunSpec("massive_tr", "ufakzeka", 3e-5, 1, attention="bidirectional")
    folder = write_run(tmp_path / spec.path, {80: (1.5, 1.0)}, spec=spec)
    row = row_for_run(folder, spec)
    assert row["attention"] == "bidirectional"

    causal_row = dict(row, attention="causal", macro_f1=0.1)
    summary = summarise([row, causal_row])
    assert len(summary) == 2, "the two arms were averaged into one row"
    # The gate stays the causal arm alone.
    gate = gate_numbers(summary)["ufakzeka"]
    assert gate["datasets"] == {"massive_tr": causal_row["calibrated"]["macro_f1"]}


def test_the_table_labels_each_arm_and_does_not_print_one_name_twice(tmp_path):
    """Two arms of one backbone rendered as two rows both called `ufakzeka`.

    The summary keyed them apart correctly, so the numbers were right; the
    label dropped the arm, which made the committed artefact unreadable and
    the comparison block impossible to match against its own table.
    """
    from model.instrument.table import render, summarise

    spec = RunSpec("massive_tr", "ufakzeka", 3e-5, 1, attention="bidirectional")
    row = row_for_run(write_run(tmp_path / spec.path, {80: (1.5, 1.0)}, spec=spec), spec)
    causal = dict(row, attention="causal", learning_rate=8e-5)
    text = render(summarise([row, causal]), {}, {}, "c0ffee" * 7)

    per_dataset = text[: text.index("## The gate")]
    # Each arm has its own label, once in each of the dataset's two tables, so
    # neither arm is ever printed under the other's name.
    assert per_dataset.count("| ufakzeka-bidirectional |") == 2
    assert per_dataset.count("| ufakzeka |") == 2, "the two arms render under one name"


def test_the_gate_picks_pooling_on_validation_and_lists_every_system():
    from model.instrument.table import decision_gate

    def entry(dataset, backbone, pooling, f1, score, attention="causal", sd=0.01):
        return {
            "dataset": dataset,
            "backbone": backbone,
            "pooling": pooling,
            "attention": attention,
            "calibrated_macro_f1": f1,
            "calibrated_macro_f1_sd": sd,
            "selection_score": score,
        }

    summary = {
        "a": entry("massive_tr", "ufakzeka", "stock", 0.81, 0.60),
        # Better on validation and on test: chosen.
        "b": entry("massive_tr", "ufakzeka", "appended_end", 0.83, 0.55),
        # Best on test, worst on validation: must not be chosen, that is peeking.
        "c": entry("massive_tr", "ufakzeka", "mean", 0.90, 0.70),
        "d": entry("massive_tr", "berturk", "stock", 0.84, 0.50),
        "e": entry("massive_tr", "ufakzeka", "stock", 0.80, 0.61, attention="bidirectional"),
        "f": entry("trcola", "ufakzeka", "stock", 0.59, 0.60),
        "g": entry("trcola", "berturk", "stock", 0.69, 0.50),
        # Saturated: the spread between backbones is inside one backbone's seeds.
        "h": entry("legal_nli_tr", "ufakzeka", "stock", 0.997, 0.01, sd=0.002),
        "i": entry("legal_nli_tr", "berturk", "stock", 0.998, 0.01, sd=0.002),
    }
    gate = decision_gate(summary)
    assert "legal_nli_tr" in gate["saturated"]
    assert "legal_nli_tr" not in gate["ranking_datasets"]

    ours = gate["systems"]["ufakzeka"]["datasets"]["massive_tr"]
    assert (ours["pooling"], ours["macro_f1"], ours["poolings"]) == ("appended_end", 0.83, 3)
    # The arm is its own system, and it is shown as incomplete, not left out.
    arm = gate["systems"]["ufakzeka-bidirectional"]
    assert arm["mean_macro_f1"] is None and "trcola" in arm["missing"]
    assert set(gate["systems"]) == {"ufakzeka", "ufakzeka-bidirectional", "berturk"}


def test_a_causal_run_filed_under_a_bidirectional_name_is_refused_everywhere(tmp_path):
    """What the volume held for a day: the right folder name, a causal run inside.

    Selection must not pick a rate from it and no results row may be built
    from it, because both would carry the arm's label over the baseline's
    numbers, which is how a finding was once reported about a treatment nobody ran.
    """
    for spec in sweep_specs(["massive_tr"], attention="bidirectional"):
        if spec.backbone == "ufakzeka":
            write_run(tmp_path / spec.path, {100: (1.0, 1.0)}, extra={"attention": "causal"})
    with pytest.raises(ValueError, match="recorded attention 'causal'"):
        chosen_rate(tmp_path, "massive_tr", "ufakzeka", "bidirectional")

    final = RunSpec("massive_tr", "ufakzeka", 3e-5, 1, attention="bidirectional")
    folder = write_run(tmp_path / final.path, {80: (1.5, 1.0)}, extra={"attention": "causal"})
    with pytest.raises(ValueError, match="the spec says 'bidirectional'"):
        row_for_run(folder, final)
    rows, _ = None, None
    with pytest.raises(ValueError):
        rows, _ = rows_for_specs(tmp_path, [final])
    assert rows is None


def test_a_folder_holding_another_dataset_or_backbone_is_refused(tmp_path):
    """Identity is checked as well as treatment: dataset, model id and revision."""
    from model.instrument.plan import treatment_mismatch

    spec = RunSpec("trcola", "ufakzeka", 1e-5, 1)
    folder = write_run(tmp_path / spec.path, {80: (1.5, 1.0)}, spec=spec)
    recorded = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert treatment_mismatch(recorded, spec) is None
    assert "recorded dataset" in treatment_mismatch(recorded | {"dataset": "mide22"}, spec)
    assert "recorded backbone" in treatment_mismatch(recorded | {"backbone": "dbmdz/x"}, spec)
    assert "recorded revision" in treatment_mismatch(recorded | {"revision": "0" * 40}, spec)
    # A converted checkpoint's record carries its conversion fingerprint, so a
    # different export in the same folder is caught.
    rung = RunSpec("trcola", "ufakzeka-mlm-1b", 2e-5, 1, attention="bidirectional")
    good = {
        "dataset": "trcola",
        "attention": "bidirectional",
        "revision": "78df50fc0ea8af8b6daaab276595e60c",
    }
    assert treatment_mismatch(good, rung) is None
    assert treatment_mismatch(good | {"revision": "ffff"}, rung) is not None
