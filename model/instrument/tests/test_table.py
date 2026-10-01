"""The results table: means and spreads over seeds, the gate number, and honesty about gaps."""

import json

import pytest

from model.instrument.plan import DATASETS
from model.instrument.table import build, gate_numbers, render, summarise


def row(dataset, backbone, seed, f1, *, pooling="stock", rate=5e-5, brier=0.5):
    block = {
        "macro_f1": f1,
        "accuracy": f1,
        "mcc": f1,
        "logloss": 1.0,
        "brier": brier,
        "root_brier": brier**0.5,
        "brier_normalised": 0.6,
        "smooth_ece": 0.02,
        "ece_15": 0.03,
        "brier_calibration_error": 0.01,
        "brier_refinement": 0.4,
    }
    return {
        "dataset": dataset,
        "backbone": backbone,
        "pooling": pooling,
        "learning_rate": rate,
        "seed": seed,
        "run_path": f"final/{dataset}/{backbone}/{pooling}-lr-s{seed}",
        "selection": {
            "step": 100 + seed,
            "score": 0.5,
            "estimator": "x",
            "audit": {},
            "points": 10,
        },
        "n_test": 1000,
        "n_validation": 500,
        "n_classes": 3,
        "inverse_temperature": 0.5,
        "raw": {**block, "brier": brier + 0.1},
        "calibrated": block,
        "brier_improvement": 0.1,
        "logloss_improvement": 0.05,
        "run": {
            "max_length": 128,
            "truncated_share": {"train": 0.05, "validation": 0.04, "test": 0.03},
            "parameters_total": 182495232,
        },
    }


def test_a_summary_is_a_mean_and_a_sample_spread_over_seeds():
    rows = [
        row("massive_tr", "ufakzeka", seed, f1)
        for seed, f1 in zip((1, 4, 21), (0.80, 0.82, 0.84), strict=True)
    ]
    entry = summarise(rows)["massive_tr/ufakzeka/stock"]
    assert entry["calibrated_macro_f1"] == pytest.approx(0.82)
    assert entry["calibrated_macro_f1_sd"] == pytest.approx(0.02)
    assert entry["seeds"] == [1, 4, 21]
    assert entry["selected_steps"] == [101, 104, 121]
    # The raw state is kept beside the calibrated one, not replaced by it.
    assert entry["raw_brier"] > entry["calibrated_brier"]


def test_one_seed_has_a_spread_of_zero_rather_than_an_error():
    entry = summarise([row("mide22", "berturk", 1, 0.7)])["mide22/berturk/stock"]
    assert entry["calibrated_macro_f1_sd"] == 0.0


def test_the_gate_number_is_the_unweighted_mean_over_datasets():
    rows = []
    # A large dataset must not be able to carry a backbone on its own, so the
    # mean is over datasets and not over rows.
    for dataset, f1 in zip([d.key for d in DATASETS], (0.9, 0.5, 0.5, 0.5, 0.5), strict=True):
        rows += [row(dataset, "ufakzeka", seed, f1) for seed in (1, 4)]
    gate = gate_numbers(summarise(rows))["ufakzeka"]
    assert gate["mean_macro_f1"] == pytest.approx((0.9 + 0.5 * 4) / 5)
    assert gate["complete"] is True and gate["missing"] == []


def test_an_incomplete_backbone_is_named_not_quietly_averaged():
    rows = [row("massive_tr", "tabibert", seed, 0.8) for seed in (1, 4)]
    gate = gate_numbers(summarise(rows))["tabibert"]
    assert gate["complete"] is False
    assert set(gate["missing"]) == {d.key for d in DATASETS} - {"massive_tr"}
    assert gate["mean_macro_f1"] == pytest.approx(0.8)
    # A backbone with no runs at all reports nothing rather than zero.
    assert gate_numbers(summarise(rows))["berturk"]["mean_macro_f1"] is None


def test_the_ablation_is_reported_apart_from_the_gate():
    rows = [row("massive_tr", "ufakzeka", seed, 0.8) for seed in (1, 4)]
    rows += [row("massive_tr", "ufakzeka", seed, 0.6, pooling="mean") for seed in (1, 4)]
    summary = summarise(rows)
    assert set(summary) == {"massive_tr/ufakzeka/stock", "massive_tr/ufakzeka/mean"}
    # Only the stock arm feeds the gate.
    assert gate_numbers(summary)["ufakzeka"]["mean_macro_f1"] == pytest.approx(0.8)
    text = render(summary, gate_numbers(summary), {}, "c0ffee")
    assert "Pooling ablation" in text and "mean" in text


def test_the_table_reports_intervals_and_reads_them():
    rows = [row("massive_tr", "ufakzeka", seed, 0.80) for seed in (1, 4)]
    rows += [row("massive_tr", "tabibert", seed, 0.85) for seed in (1, 4)]
    comparisons = {
        "massive_tr": [
            {"system": "ufakzeka", "mean": 0.80, "difference": None, "low": None, "high": None,
             "tie": None, "beats_reference": None, "per_seed": [0.8, 0.8]},
            {"system": "tabibert", "mean": 0.85, "difference": 0.05, "low": 0.02, "high": 0.08,
             "tie": False, "beats_reference": True, "per_seed": [0.85, 0.85]},
        ]
    }  # fmt: skip
    summary = summarise(rows)
    text = render(summary, gate_numbers(summary), comparisons, "abcdef1234")
    assert "+0.0500" in text and "[+0.0200, +0.0800]" in text and "better" in text
    assert "Reference for the interval below: ufakzeka" in text
    # The reader is told what the input budget was, because truncation shapes the score.
    assert "128 tokens" in text and "3% of test rows truncated" in text
    assert "abcdef1234" in text


def test_a_worse_result_is_called_worse_and_a_straddling_interval_unclear():
    rows = [row("mide22", "ufakzeka", 1, 0.8), row("mide22", "berturk", 1, 0.7)]
    comparisons = {
        "mide22": [
            {"system": "ufakzeka", "mean": 0.8, "difference": None, "low": None, "high": None,
             "tie": None, "beats_reference": None, "per_seed": [0.8]},
            {"system": "berturk", "mean": 0.7, "difference": -0.1, "low": -0.15, "high": -0.05,
             "tie": False, "beats_reference": False, "per_seed": [0.7]},
        ]
    }  # fmt: skip
    summary = summarise(rows)
    assert "worse" in render(summary, gate_numbers(summary), comparisons, "c")
    comparisons["mide22"][1].update(low=-0.06, high=0.04, difference=-0.01)
    assert "unclear" in render(summary, gate_numbers(summary), comparisons, "c")


def test_build_writes_both_files_and_they_reload(tmp_path):
    rows = [row("massive_tr", "ufakzeka", seed, 0.8) for seed in (1, 4, 21)]
    (tmp_path / "rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    table_path, summary_path = build(tmp_path, "c0ffee1234")
    assert table_path.is_file() and summary_path.is_file()
    reloaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert reloaded["summary"]["massive_tr/ufakzeka/stock"]["calibrated_macro_f1"] == pytest.approx(
        0.8
    )
    assert reloaded["gate"]["ufakzeka"]["complete"] is False
    assert table_path.read_text(encoding="utf-8").startswith("# The instrument")


def test_a_dataset_whose_spread_is_smaller_than_its_seed_noise_is_named_saturated():
    from model.instrument.table import saturated_datasets

    rows = []
    # legal_nli_tr: four backbones within 0.001 of each other, seeds moving 0.003.
    for i, backbone in enumerate(["ufakzeka", "tabibert", "moganbert", "berturk"]):
        for seed, jitter in zip((1, 4, 21), (-0.003, 0.0, 0.003), strict=True):
            rows.append(row("legal_nli_tr", backbone, seed, 0.997 + i * 0.0003 + jitter))
    # massive_tr: backbones far apart, seeds steady.
    for i, backbone in enumerate(["ufakzeka", "tabibert", "moganbert", "berturk"]):
        for seed in (1, 4, 21):
            rows.append(row("massive_tr", backbone, seed, 0.80 + i * 0.02))
    summary = summarise(rows)
    assert saturated_datasets(summary) == ["legal_nli_tr"]


def test_the_gate_can_be_taken_without_a_named_dataset_and_the_plan_number_is_unchanged():
    rows = []
    for dataset, f1 in zip([d.key for d in DATASETS], (0.9, 0.5, 0.5, 0.5, 0.5), strict=True):
        rows += [row(dataset, "ufakzeka", seed, f1) for seed in (1, 4)]
    summary = summarise(rows)
    full = gate_numbers(summary)["ufakzeka"]
    without = gate_numbers(summary, exclude=("massive_tr",))["ufakzeka"]
    assert full["mean_macro_f1"] == pytest.approx((0.9 + 0.5 * 4) / 5)
    assert without["mean_macro_f1"] == pytest.approx(0.5)
    # Leaving one out must not be reported as an incomplete backbone.
    assert without["complete"] is True and without["missing"] == []


def test_the_table_shows_the_dilution_when_a_dataset_cannot_rank():
    rows = []
    for i, backbone in enumerate(["ufakzeka", "tabibert"]):
        for seed, jitter in zip((1, 4, 21), (-0.003, 0.0, 0.003), strict=True):
            rows.append(row("legal_nli_tr", backbone, seed, 0.997 + i * 0.0003 + jitter))
        for seed in (1, 4, 21):
            rows.append(row("massive_tr", backbone, seed, 0.80 + i * 0.05))
    summary = summarise(rows)
    text = render(summary, gate_numbers(summary), {}, "c0ffee")
    assert "cannot rank anything" in text and "legal_nli_tr" in text


def test_the_protocol_section_reports_the_cost_and_says_when_the_ranking_flips():
    from model.instrument.table import protocol_section

    rows = [row("massive_tr", "moganbert", s, 0.8465) for s in (1, 4)]
    rows += [row("massive_tr", "berturk", s, 0.8428) for s in (1, 4)]
    summary = summarise(rows)
    protocol = {
        "massive_tr": {
            "moganbert": {"status": "compared", "tuned_rate": 3e-4, "fixed_rate": 3e-5,
                          "systems": [{"system": "fixed", "mean": 0.8159, "difference": -0.0306,
                                       "low": -0.0459, "high": -0.0119, "tie": False}]},
            "berturk": {"status": "compared", "tuned_rate": 8e-5, "fixed_rate": 3e-5,
                        "systems": [{"system": "fixed", "mean": 0.8273, "difference": -0.0154,
                                     "low": -0.0296, "high": 0.0076, "tie": False}]},
        }
    }  # fmt: skip
    text = "\n".join(protocol_section(protocol, summary))
    assert "-0.0306" in text and "[-0.0459, -0.0119]" in text
    # Tuned puts moganbert first, fixed puts berturk first, so the flip is stated.
    assert "Ranked by the tuned protocol: moganbert > berturk" in text
    assert "Ranked by the fixed protocol: berturk > moganbert" in text
    assert "disagree about which backbone is best" in text


def test_the_protocol_section_stays_quiet_when_nothing_was_compared():
    from model.instrument.table import protocol_section

    summary = summarise([row("massive_tr", "ufakzeka", 1, 0.8)])
    assert protocol_section({"massive_tr": {"ufakzeka": {"status": "incomplete"}}}, summary) == []
    assert protocol_section({}, summary) == []
