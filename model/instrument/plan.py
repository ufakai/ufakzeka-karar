"""The instrument's grid: which backbones, which datasets, which runs.

One committed file so the grid cannot drift between a launch and the table
that reports it. Every run in the instrument is one entry from the product
of these registries, and its folder name is derived here.
"""

from __future__ import annotations

from dataclasses import dataclass

# TrGLUE's published seed values, so our spread is comparable with theirs.
SEEDS: tuple[int, ...] = (1, 4, 21, 40, 124)
# The learning rate is selected on this seed, then the winner is run on all of SEEDS.
SELECTION_SEED = SEEDS[0]

# Contains TabiBench's upper three values and the ModernBERT and Ettin ranges,
# so no baseline is held back by a grid that stops too low. The top
# grid grew three times, each time from a measurement showing a backbone still
# descending at an end of it. It now spans a factor of 500,
# because the four backbones do not share an optimum and the spread differs by
# dataset. 1e-3 is the hard stop upward, where MoganBERT-TR's own authors
# report that model collapsing; 2e-6 is below TabiBench's published floor.
LEARNING_RATES: tuple[float, ...] = (
    2e-6,
    5e-6,
    1e-5,
    2e-5,
    3e-5,
    5e-5,
    8e-5,
    1e-4,
    2e-4,
    3e-4,
    5e-4,
    1e-3,
)

POOLINGS: tuple[str, ...] = ("stock", "appended_end", "mean")
# The backbone as pretrained, and the arm measured against it.
ATTENTIONS: tuple[str, ...] = ("causal", "bidirectional")


@dataclass(frozen=True)
class Backbone:
    key: str  # short name, used in run folders
    model_id: str
    revision: str  # pinned on 2026-09-20
    family: str  # causal, modernbert or bert
    license_id: str


BACKBONES: tuple[Backbone, ...] = (
    Backbone(
        "ufakzeka",
        "ufakai/ufakzeka-1-base",
        "f9e11eea28cbb2ba953a5628f972d416fe0c3cfe",
        "causal",
        "Apache-2.0",
    ),
    Backbone(
        "tabibert",
        "boun-tabilab/TabiBERT",
        "36d24bfae8d67f7b1f7d0827fc43076e4a1c3e9f",
        "modernbert",
        "Apache-2.0",
    ),
    Backbone(
        "moganbert",
        "moganai/MoganBERT-TR",
        "2614f39447ba72a3b2917b59c920a4844ff96cdd",
        "modernbert",
        "Apache-2.0",
    ),
    Backbone(
        "berturk",
        "dbmdz/bert-base-turkish-cased",
        "b6e1de16c983e0f2c70664591ea3f22810072608",
        "bert",
        "MIT",
    ),
)

# Only the causal backbone has no published pooling precedent at this size,
# so only it gets the ablation.
ABLATION_BACKBONE = "ufakzeka"


@dataclass(frozen=True)
class Dataset:
    key: str
    epochs: float
    batch_size: int
    # None means one micro-batch per step. Only the 512-token set needs to be
    # split to fit the GPU; the optimiser still sees batch_size.
    micro_batch_size: int | None = None


# Epochs are fixed here, before any run, and follow the published protocols
# for sets of this size: five for the mid-sized sets as TrGLUE uses, three for
# the two largest, ten for the smallest, which is TabiBench's ceiling.
DATASETS: tuple[Dataset, ...] = (
    Dataset("massive_tr", epochs=5, batch_size=32),
    Dataset("offenseval_tr", epochs=3, batch_size=32),
    Dataset("trcola", epochs=5, batch_size=32),
    Dataset("mide22", epochs=10, batch_size=32),
    Dataset("legal_nli_tr", epochs=3, batch_size=32, micro_batch_size=8),
)

# Converted checkpoints of our own backbone, scored as they are produced.
# Each is a folder on the project volume written by the conversion's export
# phase, registered here and committed before it is run, so a result can be
# traced to one checkpoint the way the hub models trace to one revision. The
# revision field holds the conversion run's fingerprint. They are read on the
# bidirectional arm only, which is what they were trained for.
CONVERTED: tuple[Backbone, ...] = (
    # The 1B rung: trunk at Muon 0.01 to 1,000,341,504 tokens, annealed
    # over 477 more steps to 1,250,426,880. results/step2/runs/ holds both
    # reports, and the folder carries its own conversion.json.
    Backbone(
        "ufakzeka-mlm-1b",
        "/vol/convert/trunk-v1-muon0.01-decay-1000000000/export/step_2385",
        "78df50fc0ea8af8b6daaab276595e60c",
        "converted",
        "Apache-2.0",
    ),
)

BY_KEY = {backbone.key: backbone for backbone in (*BACKBONES, *CONVERTED)}
DATASET_BY_KEY = {dataset.key: dataset for dataset in DATASETS}


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    backbone: str
    learning_rate: float
    seed: int
    pooling: str = "stock"
    attention: str = "causal"
    # The learning-rate sweep selects on validation alone, so those runs do not
    # pay for test evaluation. Runs that feed the results table write both.
    eval_splits: tuple[str, ...] = ("validation", "test")
    # Phase one and phase two share a (dataset, backbone, rate, seed) at the
    # selection seed but differ in what they evaluate. Separate roots keep the
    # cheaper phase-one run from being mistaken for a finished measurement.
    phase: str = "final"

    @property
    def path(self) -> str:
        """Where the run folder lives, under the runs root."""
        rate = f"{self.learning_rate:.0e}".replace("e-0", "e-")
        # The attention arm is in the path, or the bidirectional runs would
        # land on top of the committed causal ones and the resume rule would
        # report them as already done.
        arm = "" if self.attention == "causal" else f"-{self.attention}"
        return (
            f"{self.phase}/{self.dataset}/{self.backbone}/{self.pooling}{arm}-lr{rate}-s{self.seed}"
        )

    @property
    def epochs(self) -> float:
        return DATASET_BY_KEY[self.dataset].epochs

    @property
    def batch_size(self) -> int:
        return DATASET_BY_KEY[self.dataset].batch_size

    @property
    def micro_batch_size(self) -> int | None:
        return DATASET_BY_KEY[self.dataset].micro_batch_size


def payload(spec: RunSpec, git_commit: str) -> dict:
    """Everything a training container needs to run `spec`, and nothing it must guess.

    Built here and turned back into a run configuration by
    `train.config_from_payload`, with a test that takes a spec through both.
    The first version of this lived beside the dispatcher, listed the fields by
    hand, and left out `attention`. The container then used its default, which
    is causal, so every run of the "bidirectional arm" was a causal run in a
    folder named bidirectional: 116 runs, a withdrawn finding
    and a converted checkpoint scored under the wrong mask.
    """
    backbone = BY_KEY[spec.backbone]
    return {
        "path": spec.path,
        "dataset": spec.dataset,
        "backbone": spec.backbone,
        "model_id": backbone.model_id,
        # A converted checkpoint is a folder on the volume, and its revision
        # field holds the conversion run's fingerprint, which is not something
        # the hub can be asked for.
        "revision": None if backbone.model_id.startswith("/") else backbone.revision,
        "learning_rate": spec.learning_rate,
        "seed": spec.seed,
        "epochs": spec.epochs,
        "batch_size": spec.batch_size,
        "micro_batch_size": spec.micro_batch_size,
        "pooling": spec.pooling,
        "attention": spec.attention,
        "eval_splits": list(spec.eval_splits),
        "git_commit": git_commit,
    }


# What a run records about how it was treated, and the spec field it must equal.
TREATMENT = ("attention", "pooling", "learning_rate", "seed")


def treatment_mismatch(recorded: dict, spec: RunSpec) -> str | None:
    """Why a run folder is not the run `spec` describes, or None if it is.

    A folder's name is a claim. Its run.json is what happened. Anything that
    counts a run as finished, selects on it or builds a row from it asks this
    first, so a run that was given the wrong treatment cannot be reused,
    reported or compared under the name it was filed under. A record written
    before the attention arm existed has no such field and was causal.
    """
    # Identity: the dataset, the backbone's model id and its revision. A field
    # a record does not carry is not compared, which is how records written
    # before a field existed stay readable; tests/test_train.py holds the
    # training code to writing every one of them now.
    backbone = BY_KEY.get(spec.backbone)
    identity = {"dataset": spec.dataset}
    if backbone is not None:
        identity["backbone"] = backbone.model_id
        identity["revision"] = backbone.revision
    for field, expected in identity.items():
        found = recorded.get(field)
        if found is not None and found != expected:
            return f"{spec.path}: recorded {field} {found!r}, the spec says {expected!r}"
    for field in TREATMENT:
        expected = getattr(spec, field)
        found = recorded.get(field, "causal" if field == "attention" else None)
        if found is None:
            continue
        same = (
            abs(float(found) - float(expected)) <= 1e-12 * max(1.0, abs(float(expected)))
            if field == "learning_rate"
            else found == expected
        )
        if not same:
            return f"{spec.path}: recorded {field} {found!r}, the spec says {expected!r}"
    return None


def backbones_for(attention: str, only: tuple[str, ...] = ()) -> tuple[Backbone, ...]:
    """Which backbones an attention arm applies to.

    Only a causal model can be read bidirectionally as a change: the three
    encoders already attend in both directions, so a "bidirectional arm" on
    them would re-run the causal arm under a different name at four times the
    cost. Caught by a dry run that printed 192 specs where 48 were intended.
    """
    if attention == "causal":
        found = BACKBONES
    else:
        found = (*(b for b in BACKBONES if b.family == "causal"), *CONVERTED)
    if only:
        unknown = set(only) - {backbone.key for backbone in found}
        if unknown:
            raise ValueError(f"not on the {attention} arm: {sorted(unknown)}")
        found = tuple(backbone for backbone in found if backbone.key in only)
    return tuple(found)


def sweep_specs(
    datasets: list[str] | None = None, attention: str = "causal", only: tuple[str, ...] = ()
) -> list[RunSpec]:
    """Phase one: every learning rate on the selection seed, validation only."""
    keys = datasets or [dataset.key for dataset in DATASETS]
    return [
        RunSpec(
            dataset=key,
            backbone=backbone.key,
            learning_rate=rate,
            seed=SELECTION_SEED,
            eval_splits=("validation",),
            phase="sweep",
            attention=attention,
        )
        for key in keys
        for backbone in backbones_for(attention, only)
        for rate in LEARNING_RATES
    ]


def seed_specs(
    chosen: dict[tuple[str, str], float], attention: str = "causal", pooling: str = "stock"
) -> list[RunSpec]:
    """Phase two: the selected learning rate on every seed.

    `chosen` maps (dataset, backbone) to the learning rate phase one picked.
    The selection seed is included again because phase one wrote no test
    logits, so its run cannot produce a results row.
    """
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
    return [
        RunSpec(
            dataset=dataset,
            backbone=backbone,
            learning_rate=rate,
            seed=seed,
            attention=attention,
            pooling=pooling,
        )
        for (dataset, backbone), rate in sorted(chosen.items())
        for seed in SEEDS
    ]


def system_name(backbone: str, attention: str = "causal", pooling: str = "stock") -> str:
    """What a backbone on one arm is called wherever it is reported.

    One function so the comparison block and the results table agree: a reader
    matching a row against its interval has to be reading the same system, and
    two places building the name separately is how they stop agreeing.
    """
    name = backbone if attention == "causal" else f"{backbone}-{attention}"
    # The pooling is part of the system too once a backbone has been read more
    # than one way: the bar a converted checkpoint has to clear is the causal
    # backbone under its best pooling, not under the stock one.
    return name if pooling == "stock" else f"{name}+{pooling}"


def comparison_systems(entries: list[dict]) -> dict[str, list[RunSpec]]:
    """The runs behind each system in one dataset's comparison, named by arm.

    `entries` are the selection records for one dataset, as chosen_rates.json
    holds them. A name taken from the backbone alone collapses the two arms:
    the bidirectional entry lands on the key the causal one already has and
    replaces it, so the bootstrap compares one arm against a copy of itself
    and reports a difference of zero. Where the two arms happen to choose the
    same learning rate, as offenseval_tr and trcola did, the substitution
    reads as a correct run and nothing raises, so the name carries the arm.
    """
    systems: dict[str, list[RunSpec]] = {}
    for entry in entries:
        attention = entry.get("attention", "causal")
        backbone = entry["backbone"]
        pooling = entry.get("pooling", "stock")
        name = system_name(backbone, attention, pooling)
        if name in systems:
            raise ValueError(f"two selected entries name the same system {name!r}")
        systems[name] = [
            RunSpec(
                dataset=entry["dataset"],
                backbone=backbone,
                learning_rate=entry["learning_rate"],
                seed=seed,
                attention=attention,
                pooling=pooling,
            )
            for seed in SEEDS
        ]
    return systems


def ablation_specs(
    dataset: str, learning_rate: float, poolings: tuple[str, ...] = ()
) -> list[RunSpec]:
    """The pooling ablation for the causal backbone, on one dataset.

    `poolings` narrows it. On massive_tr mean pooling lost to the stock one by
    2.9 macro F1, as it should on a causal model whose early positions have
    seen almost nothing, so the datasets added later compare the appended end
    token alone rather than pay again for an answer already in hand.
    """
    unknown = set(poolings) - set(POOLINGS)
    if unknown:
        raise ValueError(f"unknown poolings: {sorted(unknown)}")
    return [
        RunSpec(
            dataset=dataset,
            backbone=ABLATION_BACKBONE,
            learning_rate=learning_rate,
            seed=seed,
            pooling=pooling,
        )
        for pooling in POOLINGS
        if pooling != "stock" and (not poolings or pooling in poolings)
        for seed in SEEDS
    ]


def split_finished(specs: list[RunSpec], finished: set[str]) -> tuple[list[RunSpec], list[RunSpec]]:
    """Specs still to run, and the ones a resumed sweep already has.

    The finished ones are filtered here rather than skipped inside a container.
    A resumed sweep can begin with dozens of specs that return in
    milliseconds, and every dispatch that behaved that way ran serially while
    its pool sat idle, where every dispatch of uniform work used the whole
    pool. Sending only real work also means a resume costs no container time
    at all for the runs it is reusing.
    """
    todo = [spec for spec in specs if spec.path not in finished]
    already = [spec for spec in specs if spec.path in finished]
    return todo, already


def protocol_specs(datasets: list[str], learning_rate: float) -> list[RunSpec]:
    """Every backbone at one fixed learning rate, on every seed.

    Published Turkish encoder comparisons commonly fix a single learning rate
    for every model. These runs are that protocol, so the cost of it can be
    measured against tuning per backbone rather than asserted. They carry
    their own phase so they cannot reach the results table or the gate.
    """
    return [
        RunSpec(
            dataset=dataset,
            backbone=backbone.key,
            learning_rate=learning_rate,
            seed=seed,
            phase="fixed",
        )
        for dataset in datasets
        for backbone in BACKBONES
        for seed in SEEDS
    ]
