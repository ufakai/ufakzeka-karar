"""The int8 quantization of calib/export.py on a Modal CPU container.

The serving box has too little memory to quantize the fp32 file beside the chat
demo (a 1.5 GB cap was reached), and the laptop too little disk, so this one
step runs here, on files put on the project volume:

    modal volume put ufakzeka-karar .cache/karar/export/model.onnx export/model.onnx
    modal volume put ufakzeka-karar .cache/karar/export/calibration.npz export/calibration.npz
    modal run calib/modal_export.py
    modal volume get ufakzeka-karar export/model-int8-dynamic.onnx .   # on the box
"""

from __future__ import annotations

from pathlib import Path

import modal

VOL = Path("/vol")
volume = modal.Volume.from_name("ufakzeka-karar", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy>=2", "onnx==1.23.0", "onnxruntime==1.30.0")
    .add_local_python_source("calib")
)
app = modal.App("ufakzeka-karar-export", image=image)


@app.function(volumes={VOL.as_posix(): volume}, cpu=4, memory=16384, timeout=1800)
def quantize() -> dict:
    from calib.export import quantize as run

    sizes = run(VOL / "export")
    volume.commit()
    return sizes


@app.local_entrypoint()
def main() -> None:
    print(quantize.remote())
