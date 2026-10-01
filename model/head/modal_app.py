"""The week-1 slice on the GPU service: the decision head on both backbones.

    modal volume put ufakzeka-karar data/built/typed typed          # once, free
    modal run model/head/modal_app.py::main --phase grid --dry-run  # prints the plan
    modal run model/head/modal_app.py::main --phase grid
    modal run model/head/modal_app.py::main --phase seeds

The grid gives each backbone the same small search, two backbone learning
rates by two poolings on seed 1, because one fixed rate biases a comparison
and the converted rung preferred higher rates than the causal backbone
on the instrument. Each backbone's best configuration, chosen on validation log
loss, then runs on two more seeds, and the comparison is made on the
three seeds of each. After training, every run also answers MASSIVE's full
59-intent question through chunked inference, which is how the server will.

Every run writes its folder under /vol/head/slice on the project volume and
returns its summary, which the entrypoint keeps under results/step4/slice.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "ufakzeka-karar-head"
VOLUME_NAME = "ufakzeka-karar"
GPU = os.environ.get("KARAR_GPU", "L4")
MAX_CONTAINERS = 8
VOL = Path("/vol")
HF_HOME = VOL / "hf"
SLICE_ROOT = VOL / "head" / "slice"

CAUSAL = ("ufakai/ufakzeka-1-base", "f9e11eea28cbb2ba953a5628f972d416fe0c3cfe")
CONVERTED = ("/vol/convert/trunk-v1-muon0.01-decay-1000000000/export/step_2385", None)
TASKS = ("massive_tr", "offenseval_tr", "mide22")
# Rows per task in training. OffensEval alone has 28,570; a cap keeps one task
# from dominating a slice whose job is to test the design, not to finish it.
MAX_PER_TASK = 8000
GRID_RATES = (3e-5, 1e-4)
GRID_POOLINGS = ("mean", "last")
MORE_SEEDS = (2, 3)
# Measured on the instrument's L4 runs and scaled for longer packed sequences;
# used only to cap a run. A run that reaches this is wrong.
RUN_TIMEOUT = 3600

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.0",
        "transformers==5.17.0",
        "numpy>=2",
        "pydantic>=2",
        "pyyaml>=6",
        "huggingface-hub>=1.0",
        "hf-xet>=1.6",
    )
    .env(
        {"HF_HOME": str(HF_HOME), "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"}
    )
    .add_local_python_source("data", "model", "schema")
)
app = modal.App(APP_NAME, image=image)


def run_name(kind: str, rate: float, pooling: str, seed: int) -> str:
    return f"{kind}-{pooling}-lr{rate:.0e}-s{seed}".replace("e-0", "e-")


@app.function(
    gpu=GPU,
    volumes={VOL.as_posix(): volume},
    timeout=RUN_TIMEOUT,
    max_containers=MAX_CONTAINERS,
    # No retry: a failure after training would pay for the training twice.
    retries=0,
    scaledown_window=2,
)
def train_slice(spec: dict) -> dict:
    import random
    from dataclasses import asdict

    import torch
    from transformers import AutoTokenizer

    from model.convert.native import from_pretrained
    from model.head.infer import chunked_choice
    from model.head.train import HeadConfig, train_head
    from schema.questions import ChoiceQuestion

    os.environ["KARAR_GIT_COMMIT"] = spec["git_commit"]
    out = SLICE_ROOT / spec["name"]
    if (out / "summary.json").is_file():
        return json.loads((out / "summary.json").read_text(encoding="utf-8")) | {
            "status": "skipped"
        }

    if spec["kind"] not in ("causal", "converted"):
        raise ValueError(f"unknown backbone kind {spec['kind']!r}")
    backbone_id, revision = CAUSAL if spec["kind"] == "causal" else CONVERTED
    tokenizer = AutoTokenizer.from_pretrained(backbone_id, revision=revision)
    backbone, _ = from_pretrained(backbone_id, revision)
    import hashlib

    # Which weights were loaded, taken before training touches them, so the
    # record can be checked against the backbone it claims.
    weights = backbone.body.embed_tokens.weight.detach().cpu().numpy().tobytes()
    weight_fingerprint = hashlib.blake2b(weights, digest_size=12).hexdigest()

    # A fixed cap per task with a fixed seed, so every run trains on the same rows.
    capped = out / "train_rows"
    capped.mkdir(parents=True, exist_ok=True)
    train_files = []
    for task in TASKS:
        lines = (VOL / "typed" / task / "train.jsonl").read_text(encoding="utf-8").splitlines()
        lines = [line for line in lines if line.strip()]
        if len(lines) > MAX_PER_TASK:
            lines = random.Random(1).sample(lines, MAX_PER_TASK)
        path = capped / f"{task}.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        train_files.append(str(path))

    config = HeadConfig(
        backbone=backbone_id,
        revision=revision,
        causal=spec["kind"] == "causal",
        train_files=tuple(train_files),
        validation_files=tuple(str(VOL / "typed" / t / "validation.jsonl") for t in TASKS),
        out_dir=str(out),
        pooling=spec["pooling"],
        backbone_lr=spec["rate"],
        epochs=spec["epochs"],
        seed=spec["seed"],
        device="cuda",
    )
    started = time.perf_counter()
    report = train_head(config, backbone, tokenizer)
    model = report.model

    # MASSIVE's full question, 59 options, answered the way the server will.
    from data.instrument_format import Row
    from data.typed.instrument import MASSIVE_OPTIONS, _question
    from model.head.train import macro_f1

    meta = json.loads((VOL / "instrument/massive_tr/meta.json").read_text(encoding="utf-8"))
    question_dict, keys = _question("massive_tr", meta["labels"])
    question = ChoiceQuestion(**question_dict)
    encode = lambda text: tokenizer(text, add_special_tokens=False).input_ids  # noqa: E731
    gold, predicted = [], []
    for line in (
        (VOL / "instrument/massive_tr/validation.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        if not line.strip():
            continue
        row = Row(**json.loads(line))
        probs = chunked_choice(model, row.text, question, encode, device="cuda")
        gold.append(keys[row.label])
        predicted.append(max(probs, key=probs.get))
    massive59 = {
        "rows": len(gold),
        "macro_f1": macro_f1(gold, predicted),
        "accuracy": sum(g == p for g, p in zip(gold, predicted, strict=True)) / len(gold),
    }
    assert len(MASSIVE_OPTIONS) == 59

    summary = {
        "name": spec["name"],
        "spec": spec,
        "config": asdict(config),
        "weight_fingerprint": weight_fingerprint,
        "train_rows": sum(
            1
            for f in train_files
            for line in Path(f).read_text("utf-8").splitlines()
            if line.strip()
        ),
        "validation_rows": sum(m["rows"] for m in report.evaluations[-1]["metrics"].values()),
        "steps": report.steps,
        "train_seconds": round(report.seconds, 1),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "evaluations": report.evaluations,
        "massive59": massive59,
        "gpu": torch.cuda.get_device_name(0),
        "status": "done",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    return summary


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--", "model", "data", "schema"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    if dirty:
        raise SystemExit(
            "uncommitted changes; commit first so every run traces to committed code:\n" + dirty
        )
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def grid_specs(git_commit: str, epochs: float) -> list[dict]:
    return [
        {"name": run_name(kind, rate, pooling, 1), "kind": kind, "rate": rate,
         "pooling": pooling, "seed": 1, "epochs": epochs, "git_commit": git_commit}
        for kind in ("causal", "converted")
        for rate in GRID_RATES
        for pooling in GRID_POOLINGS
    ]  # fmt: skip


def best_config(summaries: list[dict], kind: str) -> dict:
    """The grid configuration with the lowest mean validation log loss over tasks.

    The minimum over evaluation points for each run, so a run is judged at its
    best point, as refine-then-calibrate selection does.
    """

    def score(summary):
        points = [
            sum(m["logloss"] for m in e["metrics"].values()) / len(e["metrics"])
            for e in summary["evaluations"]
        ]
        return min(points)

    runs = [s for s in summaries if s["spec"]["kind"] == kind and s["spec"]["seed"] == 1]
    if len(runs) != len(GRID_RATES) * len(GRID_POOLINGS):
        raise SystemExit(f"{kind}: the grid is incomplete, {len(runs)} runs")
    return min(runs, key=score)["spec"]


@app.local_entrypoint()
def main(phase: str = "grid", epochs: float = 2.0, dry_run: bool = False, only: str = "") -> None:
    root = Path(__file__).resolve().parents[2]
    results = root / "results/step4/slice"
    results.mkdir(parents=True, exist_ok=True)
    git_commit = "dry-run" if dry_run else _git_commit()

    if phase == "grid":
        specs = grid_specs(git_commit, epochs)
    elif phase == "seeds":
        summaries = [json.loads(p.read_text(encoding="utf-8")) for p in results.glob("*.json")]
        specs = []
        for kind in ("causal", "converted"):
            chosen = best_config(summaries, kind)
            print(f"  {kind}: selected {chosen['pooling']} at {chosen['rate']:.0e}")
            for seed in MORE_SEEDS:
                name = run_name(kind, chosen["rate"], chosen["pooling"], seed)
                specs.append({**chosen, "seed": seed, "git_commit": git_commit, "name": name})
    else:
        raise SystemExit(f"unknown phase {phase!r}")

    todo = [s for s in specs if not (results / f"{s['name']}.json").is_file()]
    if only:
        # One named run, the first-run gate: its record is read before the rest.
        todo = [s for s in todo if s["name"] == only]
        if not todo:
            raise SystemExit(f"{only!r} is not a pending run of phase {phase}")
    done = sum((results / f"{s['name']}.json").is_file() for s in specs)
    print(f"{len(todo)} runs on {GPU} ({done} of {len(specs)} already in results)")
    for spec in todo:
        print(f"  {spec['name']}")
    if dry_run or not todo:
        return
    started = time.perf_counter()
    for summary in train_slice.map(todo, return_exceptions=True, order_outputs=False):
        if isinstance(summary, Exception):
            print(f"  FAILED {summary}", flush=True)
            continue
        (results / f"{summary['name']}.json").write_text(json.dumps(summary, indent=2) + "\n")
        last = summary["evaluations"][-1]["metrics"]
        print(f"  {summary['name']} {summary['wall_seconds']}s "
              + " ".join(f"{t} {m['macro_f1']:.3f}" for t, m in last.items())
              + f" massive59 {summary['massive59']['macro_f1']:.3f}", flush=True)  # fmt: skip
    print(f"done in {(time.perf_counter() - started) / 60:.1f} min")
