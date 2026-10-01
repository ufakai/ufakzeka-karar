"""The external evaluation (bench/external.py) on one Modal CPU container.

    # r3-base-s2 with its own calibrator
    uv run --with modal==1.5.5 modal run bench/modal_external.py --run r3-base-s2 \
        --calibrator results/step7/r3-base-s2/calibrator.json
    # r4-base-s2, once results/step7/calibrator.json is refitted on its weights
    uv run --with modal==1.5.5 modal run bench/modal_external.py --run r4-base-s2 \
        --calibrator results/step7/calibrator.json

Every set built by `python -m bench.external build` (items under
.cache/karar/external/, never committed) is uploaded to the project volume
under external/<stamp>/ with the calibrator, answered in one container with the
weights at round1/runs/<run>/model.pt, and removed from the volume afterwards.
The calibrator must name the run's weights: it is checked here against the
hashes below before anything starts, and again in the container by the
adapter's own check (bench/adapters/karar.py Calibration.from_file).

Cost: CPU and memory are billed at the higher of request and use (modal.com/docs/guide/resources,
read 2026-09-27), so the request is also the limit. The price a second of 32 cores and 32 GiB
follows from the listed rates for a physical core second and a GiB second (modal.com/pricing, read
2026-09-27). After 3 percent of the padded tokens the container projects the whole run and stops,
writing nothing, if it would pass --max-usd.

Scoring runs here on the returned rows (bench/external.py summarise), and the
summary and rows go to results/step9/external/<set>-<run>.json and .rows.jsonl.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
VOL = Path("/vol")
CPU = 32.0
MEMORY_MIB = 32768
PRICE_CORE_SECOND = 0.0000131
PRICE_GIB_SECOND = 0.00000222
RATE_USD_SECOND = CPU * PRICE_CORE_SECOND + MEMORY_MIB / 1024 * PRICE_GIB_SECOND
GUARD_AFTER = 0.03
# The weights each run's calibrator must be fitted on (sha256 of round1/runs/<run>/model.pt).
WEIGHTS_SHA256 = {
    "r3-base-s2": "aab36f09d3b92f3c64635fe3e8bb410bff79a1e0d305e8f61026ca986975c6d8",
    "r4-base-s2": "43459f581b1db1bd9c73e7b47d2d95755b4dec200e7f159dab921fcaa42a43f5",
    "r5b-base-s1": "25fb92963ba1f658524de699884e3658a5ae5e582d75c55936047b5c178305bc",
}
SETS = ("mmlu_pro_tr", "turkish_mmlu", "offenseval_tr", "massive_tr", "mide22", "xcopa_tr",
        "xfact_tr")  # fmt: skip

# The head image's pinned torch and transformers (model/head/modal_app.py), torch from the
# CPU index so no CUDA wheel is pulled for a CPU container.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cpu")
    .uv_pip_install("transformers==5.17.0", "numpy>=2", "pydantic>=2.9", "pyyaml>=6",
                    "huggingface-hub>=1.0", "hf-xet>=1.6")
    .env({"HF_HOME": str(VOL / "hf"), "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("bench", "calib", "data", "model", "schema")
)  # fmt: skip
app = modal.App("ufakzeka-karar-external", image=image)
volume = modal.Volume.from_name("ufakzeka-karar", create_if_missing=False)


class OverBudget(RuntimeError):
    pass


@app.function(volumes={VOL.as_posix(): volume}, cpu=(CPU, CPU), memory=MEMORY_MIB,
              timeout=3 * 3600, retries=0, max_containers=1)  # fmt: skip
def evaluate(run: str, sets: list[str], stamp: str, max_usd: float) -> dict:
    import platform

    import torch

    from bench.adapters.karar import Calibration, load_model, sha256
    from bench.external import Progress, execute, finish, prepare
    from bench.harness.items import load_items

    started = time.perf_counter()
    torch.set_num_threads(int(CPU))
    root = VOL / "external" / stamp
    weights = VOL / "round1" / "runs" / run / "model.pt"
    weights_sha256 = sha256(weights)
    calibration = Calibration.from_file(root / "calibrator.json", weights_sha256=weights_sha256)
    model, encode = load_model(weights)
    load_seconds = time.perf_counter() - started
    layout = getattr(model, "layout", "blind")
    prepared = {name: prepare(load_items(root / name / "items.jsonl"), encode, layout=layout)
                for name in sets}  # fmt: skip
    prepare_seconds = time.perf_counter() - started - load_seconds
    progress = Progress(total_tokens=sum(p.tokens for p in prepared.values()))
    checked = {"done": False}

    def guard(p: Progress) -> None:
        if checked["done"] or p.done_tokens < GUARD_AFTER * p.total_tokens:
            return
        checked["done"] = True
        projected = (time.perf_counter() - started) + p.projected_seconds() - p.elapsed()
        cost = projected * RATE_USD_SECOND
        print(json.dumps({"guard": "checked", "projected_seconds": round(projected),
                          "projected_usd": round(cost, 3)}), flush=True)  # fmt: skip
        if cost > max_usd:
            raise OverBudget(f"projected {projected:.0f} s, ${cost:.2f}, over ${max_usd:.2f}")

    out = {}
    for name in sets:
        set_started = time.perf_counter()
        outputs = execute(model, prepared[name], progress, guard)
        rows, refused = finish(prepared[name], outputs, calibration)
        out[name] = {"rows": rows, "refused": refused, "tokens": prepared[name].tokens,
                     "seconds": round(time.perf_counter() - set_started, 1)}  # fmt: skip
        print(json.dumps({"set": name, "questions": len(rows), "seconds": out[name]["seconds"]}),
              flush=True)  # fmt: skip
    total = time.perf_counter() - started
    return {"sets": out, "weights_sha256": weights_sha256,
            "runtime": {"container": f"Modal CPU, {CPU:g} physical cores (request and limit), "
                                     f"{MEMORY_MIB} MiB",
                        "host": f"{platform.system()} {platform.machine()} {platform.processor()}",
                        "threads": torch.get_num_threads(), "load_seconds": round(load_seconds, 1),
                        "prepare_seconds": round(prepare_seconds, 1),
                        "function_seconds": round(total, 1), "padded_tokens": progress.total_tokens,
                        "estimated_usd": round(total * RATE_USD_SECOND, 3)}}  # fmt: skip


@app.local_entrypoint()
def main(run: str, calibrator: str, sets: str = "", max_usd: float = 2.0,
         out_dir: str = "results/step9/external") -> None:  # fmt: skip
    from bench.external import CACHE, cache_dir, load_built, model_info, run_record, write_run

    if run not in WEIGHTS_SHA256:
        raise SystemExit(f"--run must be one of {', '.join(WEIGHTS_SHA256)}")
    calibrator_path = REPO_ROOT / calibrator
    fitted_on = json.loads(calibrator_path.read_text(encoding="utf-8")).get("weights_sha256")
    if fitted_on != WEIGHTS_SHA256[run]:
        raise SystemExit(f"{calibrator} was fitted on weights {str(fitted_on)[:8]}..., not on "
                         f"{run} ({WEIGHTS_SHA256[run][:8]}...); refit it first")  # fmt: skip
    chosen = sets.split(",") if sets else [s for s in SETS
                                           if (cache_dir(s) / "items.jsonl").exists()]  # fmt: skip
    missing = [s for s in chosen if not (cache_dir(s) / "items.jsonl").exists()]
    if missing:
        raise SystemExit(f"not built: {missing}; run python -m bench.external build first")
    outs = {s: REPO_ROOT / out_dir / f"{s}-{run}.json" for s in chosen}
    taken = [str(p) for p in outs.values() if p.exists()]
    if taken:
        raise SystemExit(f"results exist and are never overwritten: {taken}")
    stamp = f"{run}-{int(time.time())}"
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(calibrator_path), f"external/{stamp}/calibrator.json")
        for s in chosen:
            batch.put_file(str(cache_dir(s) / "items.jsonl"), f"external/{stamp}/{s}/items.jsonl")
    print(json.dumps({"run": run, "sets": chosen, "stamp": stamp, "cache": CACHE.as_posix()}))
    try:
        result = evaluate.remote(run, chosen, stamp, max_usd)
    finally:
        # The items are other people's test sets: they leave the volume with the run.
        volume.remove_file(f"external/{stamp}", recursive=True)
    info = model_info(REPO_ROOT / "models" / run / "model.pt", calibrator_path, run) if (
        REPO_ROOT / "models" / run / "model.pt").exists() else {
        "run": run, "weights": f"volume round1/runs/{run}/model.pt",
        "weights_sha256": result["weights_sha256"], "calibrator": calibrator}  # fmt: skip
    info["weights_sha256_in_container"] = result["weights_sha256"]
    for s in chosen:
        got = result["sets"][s]
        _, groups = load_built(s)
        runtime = {**result["runtime"], "set_seconds": got["seconds"],
                   "set_padded_tokens": got["tokens"]}  # fmt: skip
        record = run_record(s, got["rows"], got["refused"], groups, info, runtime)
        write_run(outs[s], record, got["rows"])
        metrics = record["results"]["metrics"]
        print(s, {k: round(v["value"], 4) for k, v in metrics.items()})
    print(json.dumps(result["runtime"], indent=2))
