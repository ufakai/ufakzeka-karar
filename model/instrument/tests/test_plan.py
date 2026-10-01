"""The grid: stable run paths, the right phases, no accidental overlaps."""

import pytest

from model.instrument.plan import (
    ABLATION_BACKBONE,
    BACKBONES,
    BY_KEY,
    DATASETS,
    LEARNING_RATES,
    POOLINGS,
    SEEDS,
    SELECTION_SEED,
    RunSpec,
    ablation_specs,
    seed_specs,
    sweep_specs,
)


def test_the_registries_are_consistent():
    from model.instrument.plan import CONVERTED

    assert len(BACKBONES) == 4 and len(BY_KEY) == 4 + len(CONVERTED)
    for rung in CONVERTED:
        # A folder on the project volume, on the bidirectional arm only.
        assert rung.model_id.startswith("/vol/convert/") and rung.family == "converted"
    assert {backbone.family for backbone in BACKBONES} == {"causal", "modernbert", "bert"}
    assert ABLATION_BACKBONE in BY_KEY
    for backbone in BACKBONES:
        # A pinned revision is a full commit hash, never a branch name.
        assert len(backbone.revision) == 40
        int(backbone.revision, 16)
        assert backbone.license_id in ("Apache-2.0", "MIT")
    assert len({dataset.key for dataset in DATASETS}) == len(DATASETS) == 5
    assert SELECTION_SEED in SEEDS and len(set(SEEDS)) == 5
    assert POOLINGS[0] == "stock"
    # The grid is ascending and spans more than one order of magnitude, so a
    # winner at either end is a real signal about the backbone, not a gap.
    assert list(LEARNING_RATES) == sorted(LEARNING_RATES)
    assert max(LEARNING_RATES) / min(LEARNING_RATES) >= 10


def test_run_paths_are_stable_readable_and_unique():
    spec = RunSpec("massive_tr", "ufakzeka", 5e-5, 1)
    assert spec.path == "final/massive_tr/ufakzeka/stock-lr5e-5-s1"
    assert RunSpec("mide22", "berturk", 1e-4, 124).path == "final/mide22/berturk/stock-lr1e-4-s124"
    assert RunSpec("trcola", "ufakzeka", 3e-5, 4, pooling="mean").path.endswith("mean-lr3e-5-s4")

    everything = sweep_specs() + seed_specs({("massive_tr", "ufakzeka"): 5e-5})
    everything += ablation_specs("massive_tr", 5e-5)
    assert len({spec.path for spec in everything}) == len(everything)
    assert all(" " not in spec.path for spec in everything)


def test_every_learning_rate_is_distinct_in_a_path():
    paths = {RunSpec("d", "b", rate, 1).path for rate in LEARNING_RATES}
    assert len(paths) == len(LEARNING_RATES)


def test_the_sweep_is_one_seed_and_skips_test_evaluation():
    specs = sweep_specs()
    assert len(specs) == len(DATASETS) * len(BACKBONES) * len(LEARNING_RATES)
    assert {spec.seed for spec in specs} == {SELECTION_SEED}
    assert {spec.eval_splits for spec in specs} == {("validation",)}
    assert {spec.pooling for spec in specs} == {"stock"}
    # One dataset at a time is the unit a launch is sized by.
    one = sweep_specs(["massive_tr"])
    assert len(one) == len(BACKBONES) * len(LEARNING_RATES)
    assert {spec.dataset for spec in one} == {"massive_tr"}


def test_the_seed_phase_covers_every_seed_and_writes_both_splits():
    chosen = {("massive_tr", "ufakzeka"): 5e-5, ("mide22", "berturk"): 2e-5}
    specs = seed_specs(chosen)
    assert len(specs) == 2 * len(SEEDS)
    assert {spec.eval_splits for spec in specs} == {("validation", "test")}
    for (dataset, backbone), rate in chosen.items():
        seeds = {s.seed for s in specs if s.dataset == dataset and s.backbone == backbone}
        rates = {s.learning_rate for s in specs if s.dataset == dataset and s.backbone == backbone}
        assert seeds == set(SEEDS) and rates == {rate}


def test_the_selection_seed_is_rerun_in_its_own_folder():
    # Phase one wrote no test logits, so its run cannot produce a results row.
    # If the two phases shared a folder, phase two would skip it as done and
    # the row would never exist. The phase prefix is what prevents that.
    specs = seed_specs({("massive_tr", "ufakzeka"): 5e-5})
    selection = [spec for spec in specs if spec.seed == SELECTION_SEED]
    assert len(selection) == 1 and selection[0].eval_splits == ("validation", "test")

    phase_one = [s for s in sweep_specs(["massive_tr"]) if s.learning_rate == 5e-5]
    phase_one = [s for s in phase_one if s.backbone == "ufakzeka"]
    assert len(phase_one) == 1
    assert phase_one[0].seed == selection[0].seed
    assert phase_one[0].learning_rate == selection[0].learning_rate
    assert phase_one[0].path != selection[0].path
    assert phase_one[0].path.startswith("sweep/") and selection[0].path.startswith("final/")


def test_the_ablation_is_the_causal_backbone_and_the_other_two_poolings():
    specs = ablation_specs("massive_tr", 5e-5)
    assert {spec.backbone for spec in specs} == {ABLATION_BACKBONE}
    assert {spec.pooling for spec in specs} == {"appended_end", "mean"}
    assert len(specs) == 2 * len(SEEDS)


@pytest.mark.parametrize("dataset", [d.key for d in DATASETS])
def test_every_dataset_has_usable_training_settings(dataset):
    spec = RunSpec(dataset, "ufakzeka", 3e-5, 1)
    assert 1 <= spec.epochs <= 10
    assert spec.batch_size == 32


def test_a_resume_sends_only_the_runs_it_still_needs():
    from model.instrument.plan import split_finished

    specs = sweep_specs(["massive_tr"])
    finished = {spec.path for spec in specs[:20]}
    todo, already = split_finished(specs, finished)
    assert len(todo) + len(already) == len(specs)
    assert len(already) == 20
    # Order is kept, so a dispatch still runs the grid in grid order.
    assert [spec.path for spec in todo] == [s.path for s in specs if s.path not in finished]
    # Nothing finished, nothing filtered.
    assert split_finished(specs, set())[0] == specs
    assert split_finished(specs, set())[1] == []


def test_the_fixed_rate_protocol_covers_every_backbone_and_stays_out_of_the_results():
    from model.instrument.plan import BACKBONES, SEEDS, protocol_specs

    specs = protocol_specs(["massive_tr"], 3e-5)
    assert len(specs) == len(BACKBONES) * len(SEEDS)
    assert {s.backbone for s in specs} == {b.key for b in BACKBONES}
    assert {s.learning_rate for s in specs} == {3e-5}
    # A separate tree, so no results row or gate number can ever read them.
    assert all(s.path.startswith("fixed/") for s in specs)
    assert all("final/" not in s.path and "sweep/" not in s.path for s in specs)


def test_the_bidirectional_arm_applies_only_to_the_causal_backbone():
    """An encoder already reads both ways, so the arm would be a costly no-op."""
    from model.instrument.plan import BACKBONES, backbones_for, sweep_specs

    assert backbones_for("causal") == BACKBONES
    from model.instrument.plan import CONVERTED

    only = backbones_for("bidirectional")
    # Our own backbone and the converted checkpoints of it, never an encoder.
    assert [b.key for b in only] == ["ufakzeka", *(b.key for b in CONVERTED)]
    assert all(b.family in ("causal", "converted") for b in only)

    four_datasets = ["massive_tr", "offenseval_tr", "trcola", "mide22"]
    causal = sweep_specs(four_datasets)
    bidi = sweep_specs(four_datasets, attention="bidirectional")
    assert len(causal) == 4 * len(BACKBONES) * len(LEARNING_RATES)
    # One backbone at a time, not four: 48 runs each, which is what the arm
    # and every rung after it were costed at.
    assert len(bidi) == 48 * (1 + len(CONVERTED))
    alone = sweep_specs(four_datasets, attention="bidirectional", only=("ufakzeka",))
    assert len(alone) == 4 * len(LEARNING_RATES) == 48


def test_every_arm_aware_phase_filters_its_backbones():
    """The sweep was filtered and selection was not, which failed at the first
    encoder it reached. This pins the rule at the function that takes an arm,
    rather than banning the backbone tuple outright: warming the model cache
    quite properly touches every backbone.

    The app is read as text, not imported: modal is deliberately absent from
    the environment the unit tests run in.
    """
    from pathlib import Path

    source = Path("model/instrument/modal_app.py").read_text(encoding="utf-8")
    body = source[source.index("def select_rates(") : source.index("def build_rows(")]
    assert "backbones_for" in body, "select_rates does not filter by arm"
    assert "for backbone in BACKBONES" not in body, "select_rates walks every backbone"


def test_a_comparison_keeps_the_two_arms_as_two_systems():
    """The bug this pins cost nothing only because it was read before it ran.

    The comparison named a system by `key.split("/")[1]`, which turned
    `massive_tr/ufakzeka/bidirectional` into `ufakzeka` and built a causal
    spec from the bidirectional arm's rate. On massive_tr and mide22, where
    the arms chose different rates, that read a folder which does not exist.
    On offenseval_tr and trcola, where they chose the same rate, it read the
    causal folders and overwrote the causal system with a copy of itself, so
    the bootstrap would have reported a difference of exactly zero and no
    error at all.
    """
    from model.instrument.plan import comparison_systems

    entries = [
        {"dataset": "massive_tr", "backbone": "ufakzeka", "learning_rate": 8e-5},
        {
            "dataset": "massive_tr",
            "backbone": "ufakzeka",
            "learning_rate": 5e-5,
            "attention": "bidirectional",
        },
        {
            "dataset": "massive_tr",
            "backbone": "berturk",
            "learning_rate": 8e-5,
            "attention": "causal",
        },
    ]
    systems = comparison_systems(entries)
    assert set(systems) == {"ufakzeka", "ufakzeka-bidirectional", "berturk"}
    assert all(len(specs) == len(SEEDS) for specs in systems.values())

    # Each system points at its own folders, and no two systems share one.
    paths = {name: {spec.path for spec in specs} for name, specs in systems.items()}
    assert not paths["ufakzeka"] & paths["ufakzeka-bidirectional"]
    assert all("-bidirectional-" in path for path in paths["ufakzeka-bidirectional"])
    assert all("-bidirectional-" not in path for path in paths["ufakzeka"])

    # The case that hid the bug: same rate on both arms, still two systems.
    same_rate = comparison_systems(
        [
            {"dataset": "trcola", "backbone": "ufakzeka", "learning_rate": 1e-5},
            {
                "dataset": "trcola",
                "backbone": "ufakzeka",
                "learning_rate": 1e-5,
                "attention": "bidirectional",
            },
        ]
    )
    assert len(same_rate) == 2
    assert len({spec.path for specs in same_rate.values() for spec in specs}) == 2 * len(SEEDS)


def test_the_comparison_dispatcher_passes_whole_entries_not_bare_rates():
    """Read as text: modal is absent from the unit-test environment."""
    from pathlib import Path

    source = Path("model/instrument/modal_app.py").read_text(encoding="utf-8")
    body = source[source.index("def compare_backbones(") : source.index("GPU_PRICES")]
    assert "comparison_systems" in body, "compare_backbones builds its own names again"
    assert 'split("/")' not in body, "the arm is being parsed out of the key again"


def test_a_converted_checkpoint_is_swept_alone_and_only_on_the_bidirectional_arm(monkeypatch):
    """Scoring one rung must not re-dispatch or re-select the rest of the grid."""
    from model.instrument import plan

    rung = plan.Backbone(
        "ufakzeka-mlm-1b", "/vol/convert/x/export/step_1", "abc123", "converted", "Apache-2.0"
    )
    monkeypatch.setattr(plan, "CONVERTED", (rung,))

    assert rung not in plan.backbones_for("causal")
    arm = plan.backbones_for("bidirectional")
    assert [b.key for b in arm] == ["ufakzeka", "ufakzeka-mlm-1b"]

    specs = plan.sweep_specs(["trcola"], attention="bidirectional", only=("ufakzeka-mlm-1b",))
    assert len(specs) == len(LEARNING_RATES)
    assert {spec.backbone for spec in specs} == {"ufakzeka-mlm-1b"}
    assert all("-bidirectional-" in spec.path for spec in specs)

    # An encoder, or a typo, is refused instead of silently sweeping nothing.
    with pytest.raises(ValueError, match="not on the bidirectional arm"):
        plan.sweep_specs(["trcola"], attention="bidirectional", only=("berturk",))


def test_the_ablation_can_be_narrowed_to_the_pooling_worth_paying_for():
    narrowed = ablation_specs("trcola", 1e-5, poolings=("appended_end",))
    assert len(narrowed) == len(SEEDS)
    assert {spec.pooling for spec in narrowed} == {"appended_end"}
    assert {spec.backbone for spec in narrowed} == {ABLATION_BACKBONE}
    # The default is still both, so the committed massive_tr ablation is unchanged.
    assert len(ablation_specs("massive_tr", 5e-5)) == 2 * len(SEEDS)
    with pytest.raises(ValueError, match="unknown poolings"):
        ablation_specs("trcola", 1e-5, poolings=("median",))


def test_the_final_phase_can_read_a_backbone_with_another_pooling():
    specs = seed_specs(
        {("trcola", "ufakzeka-mlm-1b"): 5e-5}, attention="bidirectional", pooling="mean"
    )
    assert len(specs) == len(SEEDS)
    assert all(spec.path.startswith("final/trcola/ufakzeka-mlm-1b/mean-bidirectional-lr5e-5-s")
               for spec in specs)  # fmt: skip
    stock = seed_specs({("trcola", "ufakzeka-mlm-1b"): 5e-5}, attention="bidirectional")
    assert not {s.path for s in specs} & {s.path for s in stock}
    with pytest.raises(ValueError, match="pooling must be one of"):
        seed_specs({("trcola", "ufakzeka"): 1e-5}, pooling="median")


def test_every_field_of_a_spec_that_changes_training_reaches_the_run(tmp_path):
    """The hand-off that lost `attention` and turned an arm into a rerun of the baseline.

    A spec goes to a payload and the payload to the run configuration the
    container trains with. Every treatment field must survive both steps, for
    every value the grid uses, and a payload missing one must be refused.
    """
    from model.instrument.plan import TREATMENT, payload
    from model.instrument.train import config_from_payload

    for attention in ("causal", "bidirectional"):
        for pooling in POOLINGS:
            spec = RunSpec(
                "trcola", "ufakzeka", 3e-5, 21, pooling=pooling, attention=attention,
                eval_splits=("validation",), phase="sweep",
            )  # fmt: skip
            sent = payload(spec, "c0ffee")
            config = config_from_payload(
                sent, data_root=tmp_path / "data", runs_root=tmp_path / "runs", device="cpu"
            )
            for field in TREATMENT:
                assert getattr(config, field) == getattr(spec, field), field
            assert config.eval_splits == spec.eval_splits
            assert config.out_dir == tmp_path / "runs" / spec.path
            assert config.dataset_dir == tmp_path / "data" / "trcola"

    for field in ("attention", "pooling", "learning_rate", "seed"):
        broken = {k: v for k, v in payload(spec, "c0ffee").items() if k != field}
        with pytest.raises(KeyError):
            config_from_payload(broken, data_root=tmp_path, runs_root=tmp_path, device="cpu")


def test_a_folder_is_the_run_its_record_says_not_the_one_its_name_says():
    from model.instrument.plan import treatment_mismatch

    spec = RunSpec("trcola", "ufakzeka", 1e-5, 1, attention="bidirectional")
    good = {"attention": "bidirectional", "pooling": "stock", "learning_rate": 1e-5, "seed": 1}
    assert treatment_mismatch(good, spec) is None
    # What the volume actually held: the right name, a causal run inside.
    assert "recorded attention 'causal'" in treatment_mismatch(good | {"attention": "causal"}, spec)
    # A record from before the arm existed has no attention field and was causal.
    assert treatment_mismatch({"pooling": "stock"}, spec) is not None
    assert treatment_mismatch({"pooling": "stock"}, RunSpec("trcola", "ufakzeka", 1e-5, 1)) is None
    assert treatment_mismatch(good | {"pooling": "mean"}, spec) is not None
    assert treatment_mismatch(good | {"seed": 4}, spec) is not None
    assert treatment_mismatch(good | {"learning_rate": 1.0000000000000001e-5}, spec) is None


def test_the_dispatcher_no_longer_lists_run_fields_by_hand():
    """Read as text: modal is absent from the unit-test environment."""
    from pathlib import Path

    source = Path("model/instrument/modal_app.py").read_text(encoding="utf-8")
    body = source[source.index("def train_one(") : source.index("def select_rates(")]
    assert "config_from_payload(" in body and "RunConfig(" not in body
    assert "treatment_mismatch" in body
    payload_body = source[source.index("def _payload(") : source.index("def _identity(")]
    assert "payload(spec, git_commit)" in payload_body and '"pooling"' not in payload_body


def test_a_comparison_can_hold_one_backbone_under_two_poolings():
    from model.instrument.plan import comparison_systems

    base = {"dataset": "trcola", "backbone": "ufakzeka", "learning_rate": 1e-5}
    systems = comparison_systems([base, {**base, "pooling": "appended_end"}])
    assert set(systems) == {"ufakzeka", "ufakzeka+appended_end"}
    assert all("appended_end-" in s.path for s in systems["ufakzeka+appended_end"])
    assert all(s.path.split("/")[-1].startswith("stock-") for s in systems["ufakzeka"])


def test_the_fixed_rate_phase_reads_the_causal_arm_only_and_comparisons_check_records():
    """Read as text: modal is absent here."""
    from pathlib import Path

    source = Path("model/instrument/modal_app.py").read_text(encoding="utf-8")
    protocol = source[source.index('if phase == "protocol"') :][:1500]
    assert 'entry.get("attention", "causal") == "causal"' in protocol
    for name in ("def compare_protocol(", "def compare_backbones("):
        body = source[source.index(name) :][:3000]
        assert "_refuse_mislabelled(" in body, f"{name} reads folders without checking their record"
    train_one = source[source.index("def train_one(") - 600 : source.index("def train_one(")]
    assert "scaledown_window=2" in train_one


def test_a_converted_folder_records_its_fingerprint_as_its_revision(tmp_path):
    import json

    from model.instrument.train import _conversion_fingerprint

    assert _conversion_fingerprint(str(tmp_path)) is None
    (tmp_path / "conversion.json").write_text(json.dumps({"fingerprint": "abc123"}))
    assert _conversion_fingerprint(str(tmp_path)) == "abc123"
    assert _conversion_fingerprint("ufakai/ufakzeka-1-base") is None
