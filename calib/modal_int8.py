"""Step 7's int8 check on a trained round 1 network, on a Modal CPU container.

    modal run calib/modal_int8.py --run r1-main-s1

PLAN.md ships int8 only if it is faster and within 0.5 of fp32 on F1 and ECE.
Speed was measured on the serving box; this measures the rest on the
trained network: export with its weights, dynamic int8, and accuracy,
macro F1, Brier and ECE of the int8 file beside torch fp32 on 1,000 questions
drawn from every validation file. The laptop has no room for the 700 MB fp32
file and the box no memory to quantize it beside the chat demo, so every step
runs here.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from model.head.modal_app import HF_HOME, VOL, volume

# The head image's packages plus the export ones, with the local source last,
# as Modal requires.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.0", "transformers==5.17.0", "numpy>=2", "pydantic>=2", "pyyaml>=6",
        "huggingface-hub>=1.0", "hf-xet>=1.6", "onnx==1.23.0", "onnxscript==0.7.2",
        "onnxruntime==1.30.0",
    )
    .env({"HF_HOME": str(HF_HOME), "HF_XET_HIGH_PERFORMANCE": "1",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("calib", "data", "model", "schema")
)  # fmt: skip
app = modal.App("ufakzeka-karar-int8", image=image)


@app.function(volumes={VOL.as_posix(): volume}, cpu=8, memory=32768, timeout=3600)
def int8_check(run: str, sample: int = 4000) -> dict:
    from calib.export import build, quality, quantize

    out = VOL / "int8" / run
    out.mkdir(parents=True, exist_ok=True)
    head = VOL / "round1" / "runs" / run / "model.pt"
    built = build(out, head, sample, 100, root=VOL / "round1" / "data")
    quantize(out)
    result = {
        "run": run,
        "build": built,
        "fp32_onnx": quality(out, "fp32"),
        "int8_dynamic": quality(out, "int8-dynamic"),
        "int8_dynamic_per_channel": quality(out, "int8-dynamic-per-channel"),
    }
    (out / "int8_check.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(volumes={VOL.as_posix(): volume}, cpu=8, memory=32768, timeout=4 * 3600)
def int8_outputs(run: str) -> dict:
    """Step 7: every validation and held-out row through fp32 and each int8 file."""
    from calib.export import build, outputs, quantize_backbone
    from model.head.mixing import validation_files

    out = VOL / "int8" / f"{run}-step7"
    out.mkdir(parents=True, exist_ok=True)
    data = VOL / "round1" / "data"
    built = build(out, VOL / "round1" / "runs" / run / "model.pt", 50, 0, root=data)
    backbone = quantize_backbone(out)
    engines = ["fp32", "int8-dynamic", "int8-matmul", "int8-matmul-per-channel"]
    files = [*validation_files(data), data / "sss" / "heldout_task.jsonl"]
    ran = outputs(out, engines, files, out / "outputs.jsonl")
    result = {"run": run, "build": built, "quantized": backbone, "outputs": ran}
    (out / "step7.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


# 32 cores for the release's step 7 (the 8-core run took about 85 minutes, 2026-09-27).
@app.function(volumes={VOL.as_posix(): volume}, cpu=32, memory=32768, timeout=2 * 3600)
def fp32_outputs(run: str) -> dict:
    """Step 7 for a new checkpoint once fp32 ships: the fp32 file's outputs only."""
    from calib.export import build, outputs
    from model.head.mixing import validation_files

    out = VOL / "int8" / f"{run}-fp32"
    out.mkdir(parents=True, exist_ok=True)
    data = VOL / "round1" / "data"
    built = build(out, VOL / "round1" / "runs" / run / "model.pt", 50, 0, root=data)
    files = [*validation_files(data), data / "sss" / "heldout_task.jsonl"]
    ran = outputs(out, ["fp32"], files, out / "outputs.jsonl", threads=32)
    result = {"run": run, "build": built, "outputs": ran}
    (out / "step7.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(volumes={VOL.as_posix(): volume}, cpu=8, memory=32768, timeout=3 * 3600)
def no_down_outputs(run: str) -> dict:
    """The int8 file with the down projections in fp32, over every row."""
    from calib.export import outputs, quantize_no_down
    from model.head.mixing import validation_files

    out = VOL / "int8" / f"{run}-step7"
    data = VOL / "round1" / "data"
    built = quantize_no_down(out)
    files = [*validation_files(data), data / "sss" / "heldout_task.jsonl"]
    ran = outputs(out, ["int8-no-down"], files, out / "outputs_no_down.jsonl")
    volume.commit()
    return {"run": run, "quantized": built, "outputs": ran}


@app.local_entrypoint()
def main(
    run: str = "r1-main-s1", step7: bool = False, no_down: bool = False, fp32_only: bool = False
) -> None:
    if fp32_only:
        result = fp32_outputs.remote(run)
        path = Path(__file__).resolve().parents[1] / "results/step7" / f"fp32_outputs_{run}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    if no_down:
        result = no_down_outputs.remote(run)
        path = Path(__file__).resolve().parents[1] / "results/step7" / f"no_down_{run}.json"
        path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    if step7:
        result = int8_outputs.remote(run)
        path = Path(__file__).resolve().parents[1] / "results/step7" / f"int8_outputs_{run}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    result = int8_check.remote(run)
    path = Path(__file__).resolve().parents[1] / "results/step4/round1" / f"int8_{run}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
