"""The decontamination index and checks on a Modal CPU container.

With TabiBench and Cetvel's remaining tasks the references hold 328,398 items
and about 13.6 million distinct 8-grams; the index needs 5 to 6 GB of memory
to build, which neither the laptop nor the serving box can spare. This uploads
the reference cache and the training files, builds the index there, checks
every file with the committed CLI, and brings back the reports:

    modal run data/decontam/modal_check.py

Reports land in results/step3/decontam/; the index stays on the volume.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

VOL = Path("/vol")
ROOT = "decontam"
# Resolved on the laptop, where the built files are; the container gets the names.
FILES = {
    "sss-build": "results/step3/build/rows.jsonl",
    "synth": "results/step3/synth/rows.jsonl",
} | {
    f"{path.parent.name}-{path.stem}": str(path)
    for path in sorted(Path("data/built/typed").glob("*/*.jsonl"))
    if path.stem in ("train", "validation")
}
FILES = {name: path for name, path in FILES.items() if Path(path).exists()}
volume = modal.Volume.from_name("ufakzeka-karar", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy>=2", "pydantic>=2", "pyyaml>=6")
    .add_local_python_source("data", "schema")
)
app = modal.App("ufakzeka-karar-decontam", image=image)


@app.function(volumes={VOL.as_posix(): volume}, cpu=4, memory=32768, timeout=3600)
def run(names: list[str]) -> dict:
    base = VOL / ROOT
    index, lsh = base / "index.pkl", base / "lsh.pkl"
    cli = [sys.executable, "-m", "data.decontam.cli"]
    subprocess.run([*cli, "build", "--cache", str(base / "eval_refs"), "--index", str(index),
                    "--lsh", str(lsh)], check=True)  # fmt: skip
    reports = {}
    for name in names:
        out = base / "reports" / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        # The CLI exits 1 when a set is over its limit; the report is written either way.
        subprocess.run([*cli, "check", "--rows", str(base / "in" / f"{name}.jsonl"), "--index",
                        str(index), "--lsh", str(lsh), "--out", str(out)], check=False)  # fmt: skip
        reports[name] = json.loads(out.read_text())
    volume.commit()
    return reports


@app.local_entrypoint()
def main(only: str = "") -> None:
    files = {n: p for n, p in FILES.items() if not only or n in only.split(",")}
    cache = Path(".cache/karar/eval_refs")
    with volume.batch_upload(force=True) as batch:
        for path in sorted(cache.glob("*.json*")):
            batch.put_file(str(path), f"/{ROOT}/eval_refs/{path.name}")
        for name, path in files.items():
            batch.put_file(path, f"/{ROOT}/in/{name}.jsonl")
    reports = run.remote(list(files))
    out = Path("results/step3/decontam")
    out.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        report["rows_file"] = FILES[name]
        (out / f"{name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(name, report.get("clean_rows"), "clean;", "over limit:",
              [s for s, v in report["sets"].items() if v["over_limit"]])  # fmt: skip
