"""Step 9: open-weight decision models on HakemBench v1.0, one Modal GPU each.

    # 20 items of one file; the rows stay on the volume and a copy goes to STEP9_SMOKE_DIR
    modal run bench/baselines/modal_gpu.py --model decider-2b --smoke --files public
    # every item file; the rows are copied into results/
    modal run bench/baselines/modal_gpu.py --model decider-2b

Each model runs through its own project's published inference code, installed
at a pinned commit in its own image (bench/adapters/decider.py, kev.py,
simple_jev.py say which call and which settings). Each item goes to the model
as the benchmark holds it; the harness writes one row per question
(bench/harness/runner.py, bench/harness/results.py) and resumes a file from the
rows it already has (bench/harness/cli.py answered and still_to_run). The item
files, public and private, go to the project's own volume and nowhere else.

An item the model's own code refuses is not answered and not guessed: it gets
no row, and its id and the model's message go to <out>.unanswered.jsonl next to
the rows (bench/adapters/systemone.py Unanswerable).

Rows are written on the volume under step9/, then copied to
results/step9/runs/<model>-<file>.jsonl (public files) and
results/private/step9/runs/<model>-<file>.jsonl (private files). A local file
that already exists is never overwritten.

A full run needs a clean tree (rows trace to committed code). A smoke run may
start from uncommitted code; its rows then say so in git_commit.

Every function has a timeout and no retry. `modal run` makes an ephemeral app
that stops when this program exits; check with `modal app list` afterwards.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "ufakzeka-karar-step9-gpu"
VOLUME_NAME = "ufakzeka-karar"
VOL = Path("/vol")
HF_HOME = VOL / "hf"
STEP = VOL / "step9"
GPU = os.environ.get("STEP9_GPU", "L4")
# A full model run is measured at the smoke run first; this only caps a run that went wrong.
RUN_TIMEOUT = 3 * 3600
# In the container this file is /root/modal_gpu.py and only the functions run, which never use it.
_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent
SCRIPT = "bench/baselines/modal_gpu.py"
SMOKE_ITEMS = 20

# Pinned code of each project, read before use (step 9).
DECIDER_COMMIT = "a5120cce45b9ff70964fac54ea6e8c1ac5b08c7f"
KEV_COMMIT = "5920c5fe4ca8e0970ed4209ac2c9b8e18bea5109"
SIMPLE_JEV_COMMIT = "dae340e30b2e2a2b27dfa1678a7ec5d66e28feca"

# name -> family, weights repo, weights revision, weights licence, code commit.
MODELS: dict[str, dict[str, str]] = {
    "decider-2b": {"family": "decider", "repo": "Mapika/decider-2b",
                   "revision": "533964dae8be954c5b5e19fa4948e48408094c1e",
                   "licence": "Apache-2.0", "code": DECIDER_COMMIT},
    "kev-4b": {"family": "kev", "repo": "jaredpalmer/kev-4b",
               "revision": "139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
               "licence": "Apache-2.0", "code": KEV_COMMIT},
    # With CUDA graphs and fused kernels on, Kev-9B ran out of memory on the L4 at the smoke run.
    "kev-9b": {"family": "kev", "repo": "jaredpalmer/kev-9b",
               "revision": "2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c",
               "licence": "Apache-2.0", "code": KEV_COMMIT, "eager": "yes"},
    # The Laya checkpoints again with the package's per-call budget override, so no text or
    # option list is cut (the shipped 512/192 cut hukuk options and long states).
    "laya-full": {"family": "laya", "checkpoint": "laya", "repo": "convaiinnovations/laya",
                  "revision": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
                  "licence": "Apache-2.0", "code": "laya==0.3.20",
                  "max_len": "2048", "head_max_len": "1024"},
    "laya-multilingual-full": {"family": "laya", "checkpoint": "laya-multilingual",
                               "repo": "convaiinnovations/laya-multilingual",
                               "revision": "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
                               "licence": "Apache-2.0", "code": "laya==0.3.20",
                               "max_len": "2048", "head_max_len": "1024"},
    "simple-jev-qwen3.5-4b": {"family": "simple-jev", "repo": "Qwen/Qwen3.5-4B",
                              "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                              "licence": "Apache-2.0", "code": SIMPLE_JEV_COMMIT},
}  # fmt: skip

# name -> the item file in this repo. slots.private.jsonl is empty and has no entry.
PUBLIC_V1 = "bench/hakembench/v1.0"
PRIVATE_V1 = "results/private/step8/v1.0"
FILES: dict[str, str] = {
    "public": f"{PUBLIC_V1}/public.jsonl",
    "private": f"{PRIVATE_V1}/private.jsonl",
    "english-public": f"{PUBLIC_V1}/probes/english.public.jsonl",
    "paraphrase-public": f"{PUBLIC_V1}/probes/paraphrase.public.jsonl",
    "permutations-public": f"{PUBLIC_V1}/probes/permutations.public.jsonl",
    "slots-public": f"{PUBLIC_V1}/probes/slots.public.jsonl",
    "english-private": f"{PRIVATE_V1}/probes/english.private.jsonl",
    "paraphrase-private": f"{PRIVATE_V1}/probes/paraphrase.private.jsonl",
    "permutations-private": f"{PRIVATE_V1}/probes/permutations.private.jsonl",
}


# The external sets (bench/external.py), as `python -m bench.external build` caches them
# outside the repo; never part of "all". Every model that runs them is compared on the same
# kept items (overlap-checked against our training files).
EXTERNAL_FILES: dict[str, str] = {
    "external-mmlu_pro_tr": ".cache/karar/external/mmlu_pro_tr/items.jsonl",
    "external-turkish_mmlu": ".cache/karar/external/turkish_mmlu/items.jsonl",
}
ALL_FILES = {**FILES, **EXTERNAL_FILES}


def is_private(file: str) -> bool:
    return file == "private" or file.endswith("-private")


def local_out(model: str, file: str) -> Path:
    side = "results/private/step9/runs" if is_private(file) else "results/step9/runs"
    return REPO_ROOT / side / f"{model}-{file}.jsonl"


def smoke_dest(model: str) -> Path:
    return Path(os.environ.get("STEP9_SMOKE_DIR", "/tmp")) / "step9-smoke"


def run_log(model: str) -> Path:
    return REPO_ROOT / "results/step9/gpu_runs.jsonl"


def remote_out(model: str, file: str, smoke: bool) -> Path:
    return STEP / ("smoke" if smoke else "runs") / f"{model}-{file}.jsonl"


ENV = {"HF_HOME": str(HF_HOME), "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false",
       "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1"}  # fmt: skip

# decider-ai's own requirements (torch, transformers>=5, flash-linear-attention, numpy<2,
# huggingface_hub, jinja2), pinned to the versions its changelog reports testing.
decider_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("torch==2.14.0", "transformers==5.17.0", "flash-linear-attention==0.5.2",
                    "numpy<2", "huggingface-hub>=1.0", "hf-xet>=1.6", "jinja2", "pydantic>=2.9")
    .uv_pip_install(f"decider-ai @ git+https://github.com/Mapika/decider@{DECIDER_COMMIT}")
    .env(ENV)
    .add_local_python_source("bench", "schema")
)  # fmt: skip
# Kev's own endpoint image (skills/kev-deploy/scripts/kev_serve.py): kev[serve] from git, then
# the fla and triton its fused kernels need; torch, transformers, numpy and peft at the
# versions of its uv.lock.
kev_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(f"kev[serve] @ git+https://github.com/jaredpalmer/kev.git@{KEV_COMMIT}",
                    "torch==2.8.0", "transformers==5.17.0", "numpy==2.5.3", "peft==0.21.0")
    .uv_pip_install("flash-linear-attention==0.5.2", "triton>=3.7.1")
    .env({**ENV, "TRITON_CACHE_DIR": "/root/triton-cache"})
    .add_local_python_source("bench", "schema")
)  # fmt: skip
# simple-jev's README install: a clone and `pip install -e ./hf-server`, after a pinned CUDA torch.
# flash-linear-attention is added: without it transformers runs the Qwen3.5 DeltaNet layers in
# its reference PyTorch code (0.74 items a second on an L4 at the smoke run). It is the kernel
# transformers itself asks for and the one decider and Kev install; it changes no prompt or score.
simple_jev_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install("torch==2.14.0", "transformers==5.17.0", "flash-linear-attention==0.5.2",
                    "hf-xet>=1.6")
    .run_commands(
        "git clone https://github.com/featherless-ai/simple-jev.git /opt/simple-jev",
        f"cd /opt/simple-jev && git checkout {SIMPLE_JEV_COMMIT}",
        "cd /opt/simple-jev && python -m pip install -e ./hf-server",
    )
    .env(ENV)
    .add_local_python_source("bench", "schema")
)  # fmt: skip

# The laya package at the version the CPU rows used, with the torch and transformers of that run.
laya_image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("torch==2.14.0", "transformers==5.17.0", "laya==0.3.20", "safetensors>=0.4",
                    "huggingface-hub==1.32.0", "hf-xet>=1.6", "pydantic>=2.9", "numpy")
    .env(ENV)
    .add_local_python_source("bench", "schema")
)  # fmt: skip

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
FUNCTION = dict(gpu=GPU, volumes={VOL.as_posix(): volume}, timeout=RUN_TIMEOUT,
                startup_timeout=1800, retries=0, max_containers=1, scaledown_window=2)  # fmt: skip


def _versions(names: list[str]) -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = "absent"
    return found


def _gpu() -> str:
    import torch

    return torch.cuda.get_device_name(0)


class _Counting:
    """Passes calls through and counts them, so a refused item can be found by its position."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls = 0

    def info(self):
        return self.inner.info()

    def answer(self, request):
        self.calls += 1
        return self.inner.answer(request)


def _run_file(adapter: Any, spec: dict, file: str) -> dict:
    """Every item of one file not yet answered, rows appended; refused items to the sidecar."""
    from bench.adapters.systemone import Unanswerable
    from bench.harness.cli import answered, still_to_run
    from bench.harness.items import load_items
    from bench.harness.results import ResultsWriter
    from bench.harness.runner import run

    items = load_items(STEP / "items" / f"{file}.jsonl")
    out = remote_out(spec["name"], file, spec["smoke"])
    refused_path = out.with_suffix(".unanswered.jsonl")
    info = adapter.info()
    done = answered(out, info.adapter, info.model, info.revision, info.prompt_version)
    refused = set()
    if refused_path.exists():
        refused = {json.loads(line)["item_id"] for line in refused_path.read_text().splitlines()}
    left = [item for item in still_to_run(items, done) if item.id not in refused]
    if spec["smoke"]:
        left = left[:SMOKE_ITEMS]
    started = time.perf_counter()
    todo, unanswered = len(left), 0
    while left:
        counting = _Counting(adapter)
        try:
            with ResultsWriter(out, append=out.exists()) as writer:
                run(left, counting, writer, repo_root=Path("/"), script=SCRIPT,
                    git_commit=spec["git_commit"])  # fmt: skip
            left = []
        except Unanswerable as error:
            item = left[counting.calls - 1]
            out.parent.mkdir(parents=True, exist_ok=True)
            with refused_path.open("a", encoding="utf-8") as handle:
                record = {"item_id": item.id, "questions": sorted(item.questions),
                          "error": str(error)}  # fmt: skip
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            unanswered += 1
            left = left[counting.calls :]
        volume.commit()
    seconds = time.perf_counter() - started
    rows = sum(1 for _ in out.open()) if out.exists() else 0
    return {"file": file, "items_in_file": len(items), "items_run": todo, "rows_in_out": rows,
            "unanswered_this_call": unanswered, "unanswered_total": unanswered + len(refused),
            "seconds": round(seconds, 1),
            "items_per_second": round(todo / seconds, 3) if seconds and todo else None}  # fmt: skip


def _evaluate(spec: dict, build) -> dict:
    """Load the model, then every file in spec["files"]; returns timings and counts."""
    started = time.perf_counter()
    adapter = build(spec)
    load_seconds = time.perf_counter() - started
    files = [_run_file(adapter, spec, file) for file in spec["files"]]
    info = adapter.info()
    return {"model": spec["name"], "smoke": spec["smoke"], "gpu": _gpu(),
            "revision": info.revision, "path_used": info.path_used,
            "prompt_version": info.prompt_version, "load_seconds": round(load_seconds, 1),
            "total_seconds": round(time.perf_counter() - started, 1), "files": files}  # fmt: skip


def _snapshot(repo: str, revision: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=revision)


@app.function(image=decider_image, **FUNCTION)
def run_decider(spec: dict) -> dict:
    def build(spec):
        from bench.adapters.decider import DeciderAdapter, load

        decider, settings = load(_snapshot(spec["repo"], spec["revision"]))
        runtime = {"code": f"github.com/Mapika/decider@{spec['code']}", **settings,
                   "gpu": _gpu(), "weights licence": spec["licence"],
                   **_versions(["decider-ai", "torch", "transformers", "flash-linear-attention",
                                "fla-core", "triton", "numpy"])}  # fmt: skip
        return DeciderAdapter(decider, model_id=spec["repo"], revision=spec["revision"],
                              runtime=runtime, device=f"cuda {_gpu()}")  # fmt: skip

    return _evaluate(spec, build)


@app.function(image=kev_image, **FUNCTION)
def run_kev(spec: dict) -> dict:
    def build(spec):
        from bench.adapters.kev import KevAdapter, load

        eager = spec.get("eager") == "yes"
        call, settings = load(f"{spec['repo']}@{spec['revision']}", eager=eager)
        runtime = {"code": f"github.com/jaredpalmer/kev@{spec['code']}", **settings,
                   "gpu": _gpu(), "weights licence": spec["licence"],
                   **_versions(["kev", "torch", "transformers", "peft", "flash-linear-attention",
                                "fla-core", "triton", "numpy"])}  # fmt: skip
        return KevAdapter(call, model_id=spec["repo"], revision=spec["revision"],
                          runtime=runtime, device=f"cuda {_gpu()}")  # fmt: skip

    return _evaluate(spec, build)


@app.function(image=simple_jev_image, **FUNCTION)
def run_simple_jev(spec: dict) -> dict:
    def build(spec):
        from bench.adapters.simple_jev import SimpleJevAdapter, load

        service, settings = load(spec["repo"], spec["revision"])
        runtime = {"code": f"github.com/featherless-ai/simple-jev@{spec['code']}", **settings,
                   "gpu": _gpu(), "weights licence": spec["licence"],
                   **_versions(["simple-jev", "torch", "transformers", "accelerate",
                                "flash-linear-attention", "fla-core", "triton",
                                "numpy"])}  # fmt: skip
        return SimpleJevAdapter(service, model_id=spec["repo"], revision=spec["revision"],
                                runtime=runtime, device=f"cuda {_gpu()}")  # fmt: skip

    return _evaluate(spec, build)


# Two containers, so the two Laya checkpoints run side by side at the same cost.
@app.function(image=laya_image, **{**FUNCTION, "max_containers": 2})
def run_laya(spec: dict) -> dict:
    def build(spec):
        from bench.adapters.laya import LayaAdapter

        return LayaAdapter(spec["checkpoint"], device="cuda", max_len=int(spec["max_len"]),
                           head_max_len=int(spec["head_max_len"]))  # fmt: skip

    return _evaluate(spec, build)


RUNNERS = {"decider": run_decider, "kev": run_kev, "simple-jev": run_simple_jev,
           "laya": run_laya}  # fmt: skip


def upload_items(files: list[str]) -> None:
    """The item files to the project's own volume (never to a hosted API)."""
    with volume.batch_upload(force=True) as batch:
        for file in files:
            batch.put_file(str(REPO_ROOT / ALL_FILES[file]), f"step9/items/{file}.jsonl")


def download(model: str, file: str, smoke: bool, dest: Path) -> str:
    """Copy one rows file (and its unanswered sidecar, if any) from the volume; never overwrite."""
    copied = []
    source = remote_out(model, file, smoke).relative_to(VOL)
    for remote, local in ((source, dest),
                          (source.with_suffix(".unanswered.jsonl"),
                           dest.with_suffix(".unanswered.jsonl"))):  # fmt: skip
        try:
            data = b"".join(volume.read_file(remote.as_posix()))
        except (FileNotFoundError, modal.exception.NotFoundError):
            continue
        if local.exists():
            if local.read_bytes() != data:
                raise FileExistsError(
                    f"{local} exists and differs from the volume copy; not touched"
                )
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        copied.append(str(local))
    return ", ".join(copied) or "nothing new"


@app.local_entrypoint()
def main(model: str, smoke: bool = False, files: str = "all", fetch_only: bool = False) -> None:
    import subprocess

    from bench.harness.results import DirtyTreeError, committed_code_version

    if model not in MODELS:
        raise SystemExit(f"--model must be one of {', '.join(MODELS)}")
    chosen = list(FILES) if files == "all" else files.split(",")
    unknown = [file for file in chosen if file not in ALL_FILES]
    if unknown:
        raise SystemExit(f"unknown files {unknown}; known: {', '.join(ALL_FILES)}")
    # Checked before any GPU starts: rows must trace to committed code. A smoke run's rows are
    # not a measurement; from an uncommitted tree they carry HEAD and say it was not clean.
    try:
        commit = committed_code_version(REPO_ROOT)
    except DirtyTreeError:
        if not smoke:
            raise
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()  # fmt: skip
        commit = f"{head} plus uncommitted changes (smoke)"
    spec = {**MODELS[model], "name": model, "smoke": smoke, "files": chosen, "git_commit": commit}
    if not fetch_only:
        upload_items(chosen)
        started = time.time()
        summary = RUNNERS[spec["family"]].remote(spec)
        summary["wall_seconds_local"] = round(time.time() - started, 1)
        summary["git_commit"] = commit
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        log = run_log(model)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    if smoke:
        dest = smoke_dest(model)
        for file in chosen:
            print(
                file, download(model, file, True, dest / f"{model}-{file}.jsonl"), file=sys.stderr
            )
        return
    for file in chosen:
        print(file, download(model, file, False, local_out(model, file)), file=sys.stderr)
