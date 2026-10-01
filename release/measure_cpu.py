"""Milliseconds per question and peak memory of the release model on CPU, one torch thread.

    uv run --group local python -m release.measure_cpu --model-dir ../karar-model [--write]

Loads the model folder release/convert.py wrote with its own karar.py, answers the demo's
examples (the seven open HakemBench items the karar demo shows, 13 questions) one question
per call, and times every call. One warm-up round, then REPEATS timed rounds; the median is
over every timed call. Peak memory is the process's peak resident set size after the rounds
(ru_maxrss). The load averages before and after are recorded, since other work on the
machine slows the calls; the numbers describe this one machine, not a benchmark.

With --write the result replaces the "cpu" section of results/step9/model_facts.json.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "bench/hakembench/v1.0/open/items.jsonl"
FACTS = ROOT / "results/step9/model_facts.json"
SECTION = "cpu_measure_2026-09-28"
# The karar demo's examples (its src/data/presets.json), one open item per category.
DEMO_ITEMS = (
    "tbmm-b44f489d12ea4baa",
    "egitim-38236c441c4c99cd",
    "guvenlik-24ba7d78d5535eaf",
    "hukuk-03c66d2b3fbe17f3",
    "moderasyon-0072519d7fcba5e5",
    "spam-9155fdca2238bef7",
    "sss-5acb0ad9d86b20e7",
)
REPEATS = 5


def machine() -> str:
    """The machine in one line: the CPU model where the system names it, cores, OS."""
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        with contextlib.suppress(OSError):
            brand = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            cpu = brand.stdout.strip() or cpu
    return f"{cpu}, {os.cpu_count()} cores, {platform.system()} {platform.release()}"


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return int(peak if sys.platform == "darwin" else peak * 1024)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_karar(model_dir: Path):
    """The folder's own karar.py, imported without writing bytecode into the folder."""
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("karar", model_dir / "karar.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["karar"] = module
    spec.loader.exec_module(module)
    return module


def measure(model_dir: Path, repeats: int = REPEATS) -> dict:
    import torch

    torch.set_num_threads(1)
    before = os.getloadavg()
    karar = load_karar(model_dir).Karar.from_pretrained(model_dir)
    items = {}
    for line in ITEMS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["id"] in DEMO_ITEMS:
                items[row["id"]] = row
    calls = [
        (items[i]["state"], {q: s}) for i in DEMO_ITEMS for q, s in items[i]["questions"].items()
    ]
    for state, question in calls:
        karar.decide(state, question)
    timed = []
    for _ in range(repeats):
        for state, question in calls:
            start = time.perf_counter()
            karar.decide(state, question)
            timed.append((time.perf_counter() - start) * 1000)
    after = os.getloadavg()
    peak = peak_rss_bytes()
    return {
        "machine": machine(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "measured_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        # Read after the peak, in blocks, so hashing adds nothing to it.
        "model_safetensors_sha256": file_sha256(model_dir / "model.safetensors"),
        "questions": len(calls),
        "timed_calls": len(timed),
        "median_ms_per_question_on_demo_examples": round(statistics.median(timed), 1),
        "min_ms_per_question": round(min(timed), 1),
        "max_ms_per_question": round(max(timed), 1),
        "peak_rss_bytes": peak,
        "load_average_before": [round(x, 2) for x in before],
        "load_average_after": [round(x, 2) for x in after],
        "script": "release/measure_cpu.py",
        "note": (
            "one machine, one torch thread, the demo's 13 questions one per call, median over "
            f"{repeats} timed rounds after one warm-up; the load averages (1, 5, 15 minutes) "
            "show what else ran; not a benchmark"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="release.measure_cpu")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--write", action="store_true", help=f"replace {SECTION} in {FACTS}")
    args = parser.parse_args(argv)
    result = measure(args.model_dir, args.repeats)
    print(json.dumps(result, indent=1))
    if args.write:
        facts = json.loads(FACTS.read_text(encoding="utf-8"))
        if result["model_safetensors_sha256"] != facts["model_safetensors_sha256"]:
            raise SystemExit("the folder's weights are not the release weights model_facts names")
        facts[SECTION] = result
        FACTS.write_text(json.dumps(facts, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
