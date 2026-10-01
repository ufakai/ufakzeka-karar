"""The instrument on Modal: one container per run, many at a time.

Every run in the grid is independent, so they are dispatched in parallel and
billed by the second. The container image pins the
same library versions the tests run against, so "one protocol" holds for the
environment as well as the code.

    modal run model/instrument/modal_app.py::main --phase upload
    modal run model/instrument/modal_app.py::main --phase prepare
    modal run model/instrument/modal_app.py::main --phase smoke
    modal run model/instrument/modal_app.py::main --phase sweep --datasets massive_tr
    KARAR_GPU=L40S modal run model/instrument/modal_app.py::main --phase sweep

A run whose folder already holds a run.json is skipped, so a resumed sweep
never pays twice for the same measurement. A folder without one is a crashed
run and is cleared before the retry.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "ufakzeka-karar-instrument"
VOLUME_NAME = "ufakzeka-karar"

# The GPU every training container asks for, fixed when the app is defined.
# It is chosen here rather than per call on purpose. Function.with_options
# returns a variant that runs in "a distinct container pool" (Modal 1.5.4
# docstring, read 2026-09-20), and a sweep dispatched into one of those pools
# ran one run at a time while eight containers sat idle. A statically
# declared pool parallelises, which is what every earlier sweep measured.
# An L4 holds a 150M model at 512 tokens with room to spare and is the
# cheapest GPU Modal offers above a T4 (modal.com/pricing, read 2026-09-20).
GPU = os.environ.get("KARAR_GPU", "L4")
# There is deliberately no second constant naming the GPU that reported runs
# use. There was one, it said L40S, and the instrument later moved to
# L4 without it: the final phase then asked for a GPU the app was not defined
# for and the dispatcher refused to launch. One name, set above, is the whole
# point. Every run of one dispatch is on one GPU because the app is defined
# for one, and each run records the device it actually saw in its run.json.

# No single run of this grid is near an hour. A run that hits this is wrong,
# and the cap stops it from quietly spending.
RUN_TIMEOUT_SECONDS = 3600
# The account's GPU concurrency allowance.
MAX_CONTAINERS = 8

VOL = Path("/vol")
DATA_ROOT = VOL / "instrument"
RUNS_ROOT = VOL / "runs"
HF_HOME = VOL / "hf"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.0",
        "transformers==5.17.0",
        "accelerate==1.15.0",
        "numpy>=2",
        "huggingface-hub>=1.0",
        # The Hub is served by the Xet backend now and hf_transfer cannot be
        # used any more (huggingface_hub environment-variable reference, read
        # 2026-09-20). Pinned explicitly rather than left to a transitive
        # dependency, so the transfer path is part of the recorded environment.
        "hf-xet>=1.6",
    )
    .env(
        {
            "HF_HOME": str(HF_HOME),
            # The documented replacement for HF_HUB_ENABLE_HF_TRANSFER, which
            # this image set until later and which had become a no-op that only
            # printed a deprecation warning.
            "HF_XET_HIGH_PERFORMANCE": "1",
            # Tokenizer threads fight the dataloader for two vCPUs and change nothing.
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

# Python files only, so the converted datasets under data/raw are not shipped
# with the code. They live on the volume instead.
image = base_image.add_local_python_source("data", "model")

# Selection and metrics need the calibration stack, which the GPU runs do not.
# Keeping it in its own image means a change here never rebuilds the training
# image, and a training container never carries libraries it will not import.
analysis_image = base_image.uv_pip_install(
    "probmetrics==1.3.0",
    # probmetrics 1.3.0 imports numba at load time without declaring it.
    "numba>=0.61",
    "scikit-learn>=1.7",
    # Smooth ECE. Its publisher's license is not an OSI one, so it stays an
    # evaluation dependency and none of its code is copied into the repo.
    "relplot>=1.0.3",
).add_local_python_source("data", "model", "calib")

app = modal.App(APP_NAME, image=image)


@app.function(volumes={VOL.as_posix(): volume}, timeout=1800, cpu=2.0)
def prepare(backbones: list[dict]) -> list[dict]:
    """Warm the model cache on the volume, once, before any GPU starts.

    Without this every container in the first wave would download the same
    weights at the same time.
    """
    from huggingface_hub import snapshot_download

    HF_HOME.mkdir(parents=True, exist_ok=True)
    report = []
    for backbone in backbones:
        started = time.perf_counter()
        path = snapshot_download(
            backbone["model_id"],
            revision=backbone["revision"],
            # The tokenizer and the weights, not the optional extras.
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        )
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        report.append(
            {
                "key": backbone["key"],
                "megabytes": round(size / 1e6, 1),
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
    volume.commit()
    return report


@app.function(volumes={VOL.as_posix(): volume}, timeout=600)
def finished(specs: list[dict]) -> list[str]:
    """Which of these runs the volume already holds a result for.

    One cheap CPU call, so a resumed sweep can send only the work it still
    needs and the GPU pool never sees an input that returns in milliseconds.
    A folder counts only if its own record says it was given the
    treatment the spec names; one that was not is sent again, and the
    training container sets the old folder aside before it runs.
    """
    from model.instrument.plan import RunSpec, treatment_mismatch

    volume.reload()
    done = []
    for item in specs:
        record = RUNS_ROOT / item["path"] / "run.json"
        if not record.is_file():
            continue
        fields = {k: v for k, v in item.items() if k != "path"}
        spec = RunSpec(**fields, phase=item["path"].split("/")[0])
        if treatment_mismatch(json.loads(record.read_text(encoding="utf-8")), spec) is None:
            done.append(item["path"])
    return done


@app.function(
    gpu=GPU,
    volumes={VOL.as_posix(): volume},
    timeout=RUN_TIMEOUT_SECONDS,
    max_containers=MAX_CONTAINERS,
    retries=modal.Retries(max_retries=2, initial_delay=5.0),
    # Two seconds, not the default sixty. Every container billed a minute of
    # idle GPU after its last run, which is most of the gap once misattributed to CPU and memory.
    scaledown_window=2,
)
def train_one(spec: dict) -> dict:
    """One fine-tuning run of the instrument. Writes its folder to the volume."""
    import torch

    from model.instrument.plan import RunSpec, treatment_mismatch
    from model.instrument.train import config_from_payload, run

    out_dir = RUNS_ROOT / spec["path"]
    record = out_dir / "run.json"
    if record.is_file():
        wanted = RunSpec(
            dataset=spec["dataset"],
            backbone=spec["backbone"],
            learning_rate=spec["learning_rate"],
            seed=spec["seed"],
            pooling=spec["pooling"],
            attention=spec["attention"],
            phase=spec["path"].split("/")[0],
        )
        problem = treatment_mismatch(json.loads(record.read_text(encoding="utf-8")), wanted)
        if problem is None:
            return {"path": spec["path"], "status": "skipped"}
        # A finished run that was given another treatment than its folder's
        # name claims. Kept, out of the way, because it is evidence of what
        # happened; never reused.
        aside = RUNS_ROOT / "_mislabelled" / spec["path"]
        aside.parent.mkdir(parents=True, exist_ok=True)
        if aside.exists():
            shutil.rmtree(aside)
        shutil.move(out_dir.as_posix(), aside.as_posix())
        print(f"  set aside: {problem}", flush=True)
    if out_dir.exists():
        # A folder without run.json is a crashed or retried run, never a result.
        shutil.rmtree(out_dir)

    os.environ["KARAR_GIT_COMMIT"] = spec["git_commit"]
    started = time.perf_counter()
    config = config_from_payload(spec, data_root=DATA_ROOT, runs_root=RUNS_ROOT, device="cuda")
    try:
        run(config)
    except torch.cuda.OutOfMemoryError as error:
        # Deterministic: the same run on the same GPU will fail the same way, so
        # a retry only spends. Reported instead of raised, which is what stops it.
        if out_dir.exists():
            shutil.rmtree(out_dir)
        volume.commit()
        return {"path": spec["path"], "status": "out of memory", "detail": str(error)[:200]}
    except Exception as error:
        # The volume keeps nothing from a failed run, so a retry starts clean.
        if out_dir.exists():
            shutil.rmtree(out_dir)
        volume.commit()
        raise RuntimeError(f"{spec['path']}: {type(error).__name__}: {error}") from error
    volume.commit()

    report = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    return {
        "path": spec["path"],
        "status": "done",
        "seconds": round(time.perf_counter() - started, 1),
        "gpu": torch.cuda.get_device_name(0),
        "steps": report["total_steps"],
        "final_train_loss": report["train_loss"][-1][1] if report["train_loss"] else None,
        "diverged_at_step": report.get("diverged_at_step"),
    }


@app.function(image=analysis_image, volumes={VOL.as_posix(): volume}, timeout=3600, cpu=4.0)
def select_rates(dataset: str, attention: str = "causal", only: tuple[str, ...] = ()) -> list[dict]:
    """Phase one's answer: the learning rate for each backbone on one dataset."""
    from model.instrument.aggregate import chosen_rate
    from model.instrument.plan import backbones_for

    # Only the backbones the arm applies to. An encoder is already
    # bidirectional, so no such sweep exists to select from, and asking for one
    # raises rather than inventing a winner.
    return [
        chosen_rate(RUNS_ROOT, dataset, backbone.key, attention)
        for backbone in backbones_for(attention, tuple(only))
    ]


@app.function(image=analysis_image, volumes={VOL.as_posix(): volume}, timeout=3600, cpu=4.0)
def build_rows(specs: list[dict]) -> dict:
    """Results rows for finished runs. Reads the volume, returns only numbers."""
    from model.instrument.aggregate import rows_for_specs
    from model.instrument.plan import RunSpec

    rebuilt = [RunSpec(**{**spec, "eval_splits": tuple(spec["eval_splits"])}) for spec in specs]
    rows, absent = rows_for_specs(RUNS_ROOT, rebuilt)
    return {"rows": rows, "absent": absent}


@app.function(image=analysis_image, volumes={VOL.as_posix(): volume}, timeout=3600, cpu=8.0)
def _refuse_mislabelled(folder: Path, spec) -> None:
    """A comparison reads a folder only if its record is the run the spec names."""
    from model.instrument.plan import treatment_mismatch

    problem = treatment_mismatch(
        json.loads((folder / "run.json").read_text(encoding="utf-8")), spec
    )
    if problem:
        raise RuntimeError(problem)


def compare_protocol(dataset: str, tuned: dict, fixed_rate: float) -> dict:
    """What fixing one learning rate for every backbone costs.

    For each backbone, its tuned runs and its fixed-rate runs are the two
    systems, compared on the same test examples and the same seeds with the
    paired bootstrap. The question is per backbone, so each gets its own
    interval, and the reference is always the tuned arm.
    """
    from dataclasses import asdict

    import numpy as np

    from calib.compare import compare
    from calib.selection import select_point
    from model.instrument.plan import SEEDS, RunSpec

    def runs_for(backbone: str, rate: float, phase: str):
        out, labels = [], None
        for seed in SEEDS:
            spec = RunSpec(
                dataset=dataset, backbone=backbone, learning_rate=rate, seed=seed, phase=phase
            )
            folder = RUNS_ROOT / spec.path
            if not (folder / "run.json").is_file():
                return None, None
            _refuse_mislabelled(folder, spec)
            step = select_point(folder).step
            out.append(np.load(folder / f"logits_test_step{step}.npy"))
            labels = np.load(folder / "labels_test.npy")
        return out, labels

    report = {}
    for backbone, rate in sorted(tuned.items()):
        tuned_runs, labels = runs_for(backbone, rate, "final")
        fixed_runs, _ = runs_for(backbone, fixed_rate, "fixed")
        if tuned_runs is None or fixed_runs is None:
            report[backbone] = {"status": "incomplete"}
            continue
        if rate == fixed_rate:
            # Tuning picked the conventional rate here, so there is nothing to
            # compare and saying so is more useful than an interval of zero.
            report[backbone] = {"status": "tuning chose the fixed rate", "tuned_rate": rate}
            continue
        result = [asdict(item) for item in compare(
            {"tuned": tuned_runs, "fixed": fixed_runs}, labels, "tuned"
        )]  # fmt: skip
        report[backbone] = {
            "status": "compared",
            "tuned_rate": rate,
            "fixed_rate": fixed_rate,
            "systems": result,
        }
    return report


@app.function(image=analysis_image, volumes={VOL.as_posix(): volume}, timeout=3600, cpu=8.0)
def compare_backbones(dataset: str, entries: list[dict], reference: str) -> list[dict]:
    """Paired bootstrap between systems on one dataset, where the logits are.

    Each system's selected point per seed is read, then seeds and test
    examples are resampled together (calib/compare.py). A system is a
    backbone on one attention arm, so the causal and bidirectional runs of
    the same backbone compare against each other here rather than one
    quietly standing in for the other.

    An incomplete system is an error, not a system left out. Dropping it
    silently is how the arm bug stayed invisible: the comparison still
    printed a full-looking table, with a row missing that no one counted.
    """
    from dataclasses import asdict

    import numpy as np

    from calib.compare import compare
    from calib.selection import select_point
    from model.instrument.plan import comparison_systems

    systems, labels = {}, None
    for name, specs in comparison_systems(entries).items():
        runs = []
        for spec in specs:
            folder = RUNS_ROOT / spec.path
            if not (folder / "run.json").is_file():
                raise RuntimeError(f"{dataset}: {name} is missing {spec.path}")
            _refuse_mislabelled(folder, spec)
            step = select_point(folder).step
            runs.append(np.load(folder / f"logits_test_step{step}.npy"))
            labels = np.load(folder / "labels_test.npy")
        systems[name] = runs
    if reference not in systems:
        raise RuntimeError(f"{dataset}: no complete set of seeds for {reference}")
    return [asdict(item) for item in compare(systems, labels, reference)]


# Per-second prices, read from modal.com/pricing on 2026-09-20. Used only to
# turn a measured speed into a cost per run, never to guess a speed.
# Billed dollars over GPU seconds times price, measured on this project's own
# bills: the container's start, model load and CPU and memory on top.
BILLED_RATIO = {"H100": 1.07, "L4": 1.25}

GPU_PRICES = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
    "H200": 0.001261,
}


@app.function(volumes={VOL.as_posix(): volume}, timeout=1200, max_containers=1)
def bench_gpu(model_id: str, revision: str, shapes: list[dict], steps: int = 40) -> dict:
    """Time real training steps of the real model at the shapes the grid uses.

    Synthetic tokens, because the question is how fast the hardware runs this
    model at this shape, not what it learns. Same precision policy as a real
    run, so the number transfers.
    """
    import time as clock

    import torch
    from torch import nn
    from transformers import AutoModelForSequenceClassification

    from model.instrument.train import ADAM_BETAS, ADAM_EPS, choose_precision, parameter_groups

    precision = choose_precision("cuda", torch.cuda.get_device_capability())
    torch.backends.cuda.matmul.allow_tf32 = precision["tf32"]
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision, num_labels=3, dtype=torch.float32, attn_implementation="sdpa"
    ).to("cuda")
    model.train()
    optimizer = torch.optim.AdamW(
        parameter_groups(model, 1e-5), lr=1e-5, betas=ADAM_BETAS, eps=ADAM_EPS
    )
    amp = torch.bfloat16 if precision["bf16"] else (torch.float16 if precision["fp16"] else None)
    scaler = torch.amp.GradScaler("cuda") if precision["fp16"] else None
    vocab = model.config.vocab_size

    out = {"gpu": torch.cuda.get_device_name(0), "precision": precision, "shapes": []}
    for shape in shapes:
        batch, length = shape["batch"], shape["length"]
        ids = torch.randint(0, vocab - 16, (batch, length), device="cuda")
        mask = torch.ones_like(ids)
        targets = torch.randint(0, 3, (batch,), device="cuda")
        torch.cuda.reset_peak_memory_stats()

        def one_step(ids=ids, mask=mask, targets=targets):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp) if amp else torch.enable_grad():
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = nn.functional.cross_entropy(logits.float(), targets)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        try:
            # Warm up first: the first steps pay for allocation and kernel choice.
            for _ in range(5):
                one_step()
            torch.cuda.synchronize()
            started = clock.perf_counter()
            for _ in range(steps):
                one_step()
            torch.cuda.synchronize()
            elapsed = clock.perf_counter() - started
            out["shapes"].append(
                {
                    "batch": batch,
                    "length": length,
                    "steps_per_second": steps / elapsed,
                    "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            )
        except torch.cuda.OutOfMemoryError:
            out["shapes"].append({"batch": batch, "length": length, "steps_per_second": None})
            torch.cuda.empty_cache()
    return out


def _git_commit() -> str:
    """The commit the code was archived at. A dirty tree is refused."""
    root = Path(__file__).resolve().parents[2]
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--", ".", ":(exclude)results"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            "uncommitted changes; commit first so every run traces to committed code:\n" + dirty
        )
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _payload(spec, git_commit: str) -> dict:
    from model.instrument.plan import payload

    return payload(spec, git_commit)


def _identity(spec) -> dict:
    """What `finished` needs to tell a run from a folder that only has its name."""
    return {
        "path": spec.path,
        "dataset": spec.dataset,
        "backbone": spec.backbone,
        "learning_rate": spec.learning_rate,
        "seed": spec.seed,
        "pooling": spec.pooling,
        "attention": spec.attention,
    }


def _dispatch(specs, git_commit: str, gpu: str = GPU) -> None:
    """Run every spec, printing each as it lands so a long sweep is watchable."""
    from model.instrument.plan import split_finished
    from model.instrument.throughput import stop_message

    if gpu != GPU:
        raise SystemExit(
            f"this app is defined for {GPU}, not {gpu}. Run it again as "
            f"KARAR_GPU={gpu} modal run ... so every run shares one static "
            "container pool and the sweep actually runs in parallel."
        )
    todo, already = split_finished(list(specs), set(finished.remote([_identity(s) for s in specs])))
    if not todo:
        print(f"nothing to do: all {len(already)} runs are already on the volume")
        return
    payloads = [_payload(spec, git_commit) for spec in todo]
    runner = train_one
    print(
        f"{len(payloads)} runs on {gpu}, up to {MAX_CONTAINERS} at a time"
        f" ({len(already)} already there, not dispatched)",
        flush=True,
    )
    started = time.perf_counter()
    done, skipped, failed, seconds = 0, len(already), 0, []
    # Completion order, not input order. With the default ordering a result is
    # held back until every earlier input has landed, so the arrival times are
    # serialised by whichever run is slowest and the concurrency measured from
    # them reads far too low: a finished sweep of 27 runs at 7.4 containers
    # busy yielded three results in 96 minutes and tripped the guard.
    # Every result carries its own path, so the order was never needed.
    for result in runner.map(payloads, return_exceptions=True, order_outputs=False):
        if isinstance(result, Exception):
            failed += 1
            print(f"  FAILED {result}", flush=True)
            continue
        if result["status"] == "skipped":
            skipped += 1
            continue
        if result["status"] != "done":
            failed += 1
            print(f"  {result['status'].upper()} {result['path']}", flush=True)
            continue
        done += 1
        seconds.append(result["seconds"])
        print(
            f"  {result['path']} {result['seconds']}s loss={result['final_train_loss']}", flush=True
        )
        stop = stop_message(seconds, time.perf_counter() - started, gpu, MAX_CONTAINERS)
        if stop:
            print(stop, flush=True)
            raise SystemExit(2)
    elapsed = time.perf_counter() - started
    print(f"{done} run, {skipped} already there, {failed} failed, {elapsed / 60:.1f} min wall")
    if seconds:
        total = sum(seconds)
        print(
            f"GPU seconds {total:.0f}, mean per run {total / len(seconds):.0f}s, "
            f"about ${total * GPU_PRICES[gpu] * BILLED_RATIO.get(gpu, 1.25):.2f} billed on {gpu}"
        )


@app.local_entrypoint()
def main(
    phase: str = "smoke",
    datasets: str = "",
    limit: int = 0,
    ablation: str = "",
    gpu: str = "",
    rate: float = 3e-5,
    attention: str = "causal",
    backbones: str = "",
    poolings: str = "",
    pooling: str = "",
) -> None:
    from model.instrument.plan import (
        ABLATION_BACKBONE,
        BACKBONES,
        BY_KEY,
        DATASETS,
        RunSpec,
        ablation_specs,
        protocol_specs,
        seed_specs,
        sweep_specs,
    )

    root = Path(__file__).resolve().parents[2]
    # Restricts a phase to some backbones, which is how one converted checkpoint
    # is swept, selected and scored without touching the rest of the grid.
    only = tuple(key for key in backbones.split(",") if key)
    # The pooling ablation, on one dataset or several, over all the alternative
    # poolings or the ones named.
    ablated = [key for key in ablation.split(",") if key]
    pooled = tuple(key for key in poolings.split(",") if key)

    if phase == "upload":
        local = root / "data/raw/instrument"
        folders = sorted(p for p in local.iterdir() if (p / "meta.json").is_file())
        if not folders:
            raise SystemExit(f"no converted datasets in {local}; run the converters first")
        with volume.batch_upload(force=True) as batch:
            for folder in folders:
                batch.put_directory(folder, f"/instrument/{folder.name}")
        print("uploaded:", ", ".join(folder.name for folder in folders))
        return

    if phase == "prepare":
        for line in prepare.remote([vars(backbone) for backbone in BACKBONES]):
            print(line)
        return

    git_commit = _git_commit()
    if phase == "smoke":
        # The cheapest end to end check on a real GPU: the smallest dataset,
        # the causal backbone, a fraction of an epoch.
        spec = RunSpec(
            dataset="mide22",
            backbone="ufakzeka",
            learning_rate=3e-5,
            seed=SMOKE_SEED,
            eval_splits=("validation",),
        )
        payload = _payload(spec, git_commit)
        payload["epochs"] = 0.2
        payload["path"] = "smoke/" + payload["path"]
        print(train_one.remote(payload))
        return

    if phase == "sweep":
        keys = [key for key in datasets.split(",") if key] or None
        specs = sweep_specs(keys, attention=attention, only=only)
        if limit:
            specs = specs[:limit]
        _dispatch(specs, git_commit, gpu or GPU)
        return

    if phase == "bench-gpu":
        # The shapes the grid actually runs: the short sets, and the 512-token
        # set at the micro-batch it is split into.
        shapes = [{"batch": 32, "length": 64}, {"batch": 8, "length": 512}]
        wanted = [g for g in datasets.split(",") if g] or ["L4", "A10", "L40S", "A100-40GB", "H100"]
        causal = BY_KEY[ABLATION_BACKBONE]
        results = []
        for name in wanted:
            print(f"  {name} ...", flush=True)
            try:
                result = bench_gpu.with_options(gpu=name).remote(
                    causal.model_id, causal.revision, shapes
                )
            except Exception as error:
                print(f"    unavailable: {str(error)[:120]}")
                continue
            result["requested"] = name
            result["price_per_second"] = GPU_PRICES[name]
            results.append(result)
            for shape in result["shapes"]:
                rate = shape["steps_per_second"]
                if rate is None:
                    print(f"    {shape['batch']}x{shape['length']}: out of memory")
                    continue
                print(
                    f"    {shape['batch']}x{shape['length']}: {rate:.2f} steps/s, "
                    f"{shape['peak_gib']:.1f} GiB, "
                    f"${GPU_PRICES[name] / rate * 1000:.3f} per 1000 steps"
                )
        path = root / "results/step1/gpu_bench.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Merge, so benchmarking a second set of GPUs does not drop the first.
        kept = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
        by_gpu = {entry["requested"]: entry for entry in kept}
        by_gpu.update({entry["requested"]: entry for entry in results})
        merged = sorted(by_gpu.values(), key=lambda e: e["price_per_second"])
        path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
        return

    if phase == "select":
        keys = [key for key in datasets.split(",") if key] or [d.key for d in DATASETS]
        chosen = {}
        for key in keys:
            for entry in select_rates.remote(key, attention, only):
                edge = "  AT GRID EDGE" if entry["at_grid_edge"] else ""
                print(f"  {key:14} {entry['backbone']:10} {entry['learning_rate']:.0e}{edge}")
                arm = "" if entry["attention"] == "causal" else f"/{entry['attention']}"
                chosen[f"{key}/{entry['backbone']}{arm}"] = entry
        path = root / "results/step1/chosen_rates.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        merged.update(chosen)
        path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path}")
        return

    if phase == "final":
        chosen_path = root / "results/step1/chosen_rates.json"
        chosen = json.loads(chosen_path.read_text(encoding="utf-8"))
        keys = [key for key in datasets.split(",") if key] or [d.key for d in DATASETS]
        rates = {
            (entry["dataset"], entry["backbone"]): entry["learning_rate"]
            for entry in chosen.values()
            if entry["dataset"] in keys
            and entry.get("attention", "causal") == attention
            and (not only or entry["backbone"] in only)
        }
        if not rates:
            raise SystemExit(
                f"no selected rates for {keys} on the {attention} arm in {chosen_path}; "
                "run --phase select for that arm first"
            )
        specs = seed_specs(rates, attention=attention, pooling=pooling or "stock")
        if ablation:
            for name in ablated:
                if (name, ABLATION_BACKBONE) not in rates:
                    raise SystemExit(f"no selected rate for {ABLATION_BACKBONE} on {name}")
                specs += ablation_specs(name, rates[(name, ABLATION_BACKBONE)], pooled)
        if limit:
            specs = specs[:limit]
        _dispatch(specs, git_commit, GPU)
        return

    if phase == "protocol":
        keys = [key for key in datasets.split(",") if key] or ["massive_tr"]
        chosen = json.loads((root / "results/step1/chosen_rates.json").read_text(encoding="utf-8"))
        _dispatch(protocol_specs(keys, rate), git_commit, GPU)
        report = {}
        for key in keys:
            # The causal arm only. A bidirectional entry for the same backbone
            # would otherwise overwrite the causal rate and be read from causal
            # folders, which is the shape of the fault behind the withdrawn runs.
            tuned = {
                entry["backbone"]: entry["learning_rate"]
                for entry in chosen.values()
                if entry["dataset"] == key and entry.get("attention", "causal") == "causal"
            }
            report[key] = compare_protocol.remote(key, tuned, rate)
            for backbone, item in sorted(report[key].items()):
                if item["status"] != "compared":
                    print(f"  {key:14} {backbone:10} {item['status']}")
                    continue
                fixed = next(s for s in item["systems"] if s["system"] == "fixed")
                print(
                    f"  {key:14} {backbone:10} tuned {item['tuned_rate']:.0e} "
                    f"fixed {rate:.0e}  {fixed['difference']:+.4f} "
                    f"[{fixed['low']:+.4f}, {fixed['high']:+.4f}]"
                    + ("  tie" if fixed["tie"] else "")
                )
        out_path = root / "results/step1/fixed_rate.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {out_path}")
        return

    if phase == "compare":
        keys = [key for key in datasets.split(",") if key] or [d.key for d in DATASETS]
        chosen = json.loads((root / "results/step1/chosen_rates.json").read_text(encoding="utf-8"))
        out_path = root / "results/step1/comparisons.json"
        merged = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else {}
        for key in keys:
            # Every arm of every backbone on this dataset, in one bootstrap,
            # so the causal and bidirectional differences are paired over the
            # same resampled examples rather than measured apart.
            entries = [entry for entry in chosen.values() if entry["dataset"] == key]
            # The causal backbone under each extra pooling named, as systems of
            # their own, so a converted checkpoint gets an interval against the
            # bar it actually has to clear.
            for extra in pooled:
                base = chosen[f"{key}/{ABLATION_BACKBONE}"]
                entries.append({**base, "pooling": extra})
            merged[key] = compare_backbones.remote(key, entries, ABLATION_BACKBONE)
            for item in merged[key]:
                verdict = (
                    ""
                    if item["difference"] is None
                    else (
                        f" {item['difference']:+.4f} [{item['low']:+.4f}, {item['high']:+.4f}]"
                        + ("  tie" if item["tie"] else "")
                    )
                )
                print(f"  {key:14} {item['system']:10} {item['mean']:.4f}{verdict}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {out_path}")
        return

    if phase == "report":
        keys = [key for key in datasets.split(",") if key] or [d.key for d in DATASETS]
        chosen = json.loads((root / "results/step1/chosen_rates.json").read_text(encoding="utf-8"))
        rates = {
            (entry["dataset"], entry["backbone"]): entry["learning_rate"]
            for entry in chosen.values()
            if entry["dataset"] in keys
            and entry.get("attention", "causal") == attention
            and (not only or entry["backbone"] in only)
        }
        runs = seed_specs(rates, attention=attention, pooling=pooling or "stock")
        if ablation:
            # The ablation runs are real measurements and need rows like any
            # other. Without this they are paid for and never reported.
            for name in ablated:
                if (name, ABLATION_BACKBONE) not in rates:
                    raise SystemExit(f"no selected rate for {ABLATION_BACKBONE} on {name}")
                runs += ablation_specs(name, rates[(name, ABLATION_BACKBONE)], pooled)
        specs = [vars(spec) | {"eval_splits": list(spec.eval_splits)} for spec in runs]
        result = build_rows.remote(specs)
        out = root / "results/step1/rows.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if out.is_file():
            for line in out.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    existing[row["run_path"]] = row
        for row in result["rows"]:
            existing[row["run_path"]] = row
        with out.open("w", encoding="utf-8") as handle:
            for key in sorted(existing):
                handle.write(json.dumps(existing[key], ensure_ascii=False) + "\n")
        print(f"{len(result['rows'])} rows, {len(result['absent'])} runs not finished")
        print(f"wrote {out} with {len(existing)} rows in total")
        return

    raise SystemExit(f"unknown phase {phase!r}")


# Kept out of the grid's seeds so a smoke run can never be mistaken for a result.
SMOKE_SEED = 999
