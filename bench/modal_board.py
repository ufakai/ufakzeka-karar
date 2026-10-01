"""The board and the probe statistics on one large Modal CPU container.

    uv run --with modal==1.5.5 modal run bench/modal_board.py --tool board --args-file ARGS

ARGS holds the arguments of `python -m bench.board` (or bench.probes), one per line, as they
would be given locally. Every argument that names an existing file is uploaded to the project
volume under the same relative path, the unchanged board or probe code runs there with 60
workers, and the output file comes back to the --out path. The local tree must be committed:
its commit is checked here and stamped on the output, as a local run would stamp it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

VOL = Path("/vol")
ROOT = VOL / "board"
REPO_ROOT = Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("numpy==2.5.3", "relplot==1.0.3", "scipy==1.18.1", "pydantic==2.13.5",
                    "pyyaml")
    .add_local_python_source("bench", "schema", "calib")
)  # fmt: skip
app = modal.App("ufakzeka-karar-board", image=image)
volume = modal.Volume.from_name("ufakzeka-karar", create_if_missing=False)


@app.function(volumes={VOL.as_posix(): volume}, cpu=64, memory=65536, timeout=2 * 3600,
              retries=0, max_containers=1)  # fmt: skip
def compute(tool: str, argv: list[str], run: str, commit: str) -> str:
    os.environ["KARAR_GIT_COMMIT"] = commit
    os.chdir(ROOT / run)
    if tool == "board":
        from bench.board import main
    else:
        from bench.probes import main
    code = main(argv)
    if code:
        raise RuntimeError(f"{tool} exited with {code}")
    out = argv[argv.index("--out") + 1]
    return Path(out).read_text(encoding="utf-8")


@app.local_entrypoint()
def main(tool: str, args_file: str) -> None:
    from bench.harness.results import committed_code_version

    if tool not in ("board", "probes"):
        raise SystemExit("--tool is board or probes")
    commit = committed_code_version(REPO_ROOT)
    argv = [a for a in Path(args_file).read_text().split() if a]
    out = argv[argv.index("--out") + 1]
    if (REPO_ROOT / out).exists():
        raise SystemExit(f"{out} exists; results are never overwritten")
    run = f"{tool}-{int(time.time())}"
    files = [a for a in argv if (REPO_ROOT / a).is_file()]
    with volume.batch_upload(force=True) as batch:
        for f in files:
            batch.put_file(str(REPO_ROOT / f), f"/board/{run}/{f}")
    print(json.dumps({"run": run, "uploaded": len(files), "commit": commit}))
    extra = ["--workers", "60"] if tool == "board" else []
    text = compute.remote(tool, argv + extra, run, commit)
    (REPO_ROOT / out).parent.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / out).write_text(text, encoding="utf-8")
    print(f"wrote {out}")
