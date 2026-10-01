"""Build the Hugging Face model repo of ufakzeka-karar from one trained run.

    uv run --group local python -m release.convert --run <run> \\
        --calibrator results/step7/<run>/calibrator.json --out /path/to/ufakzeka-karar

with <run> the release run under models/ (the one the selection rule chose) and its step 7
calibrator.

Input: models/<run>/model.pt (a DecisionHead state dict, as model/head/train.py
saves it) and the step 7 calibrator fitted on exactly those weights. Output, in
`--out`, which must be empty or absent:

- model.safetensors: every weight, float32, under the state dict's own names;
- config.json: the backbone's shape and the head's settings (causal, pooling,
  layout, truncation lengths, options per pass), read from the loaded model and
  the packing code, not restated by hand;
- calibrator.json: the calibrator as fitted, with weights_sha256 moved to the
  sha256 of model.safetensors (the file karar.py checks) and the sha256 of the
  source model.pt kept as source_weights_sha256;
- tokenizer.json and tokenizer_config.json of the backbone, at its pinned revision;
- karar.py and requirements.txt from release/model_repo/;
- LICENSE: the repository's own, the Apache License 2.0 the model is released under;
- SHA256SUMS: the sha256 of every other file, in the format `shasum -c` reads.

The model is loaded with the karar adapter's own loader (bench/adapters/karar.py),
strictly, so a model.pt that does not fit the network fails here and the saved
weights are the ones the adapter would answer with.
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from bench.adapters.karar import BACKBONE, BACKBONE_REVISION, Calibration, load_model, sha256
from schema.questions import HEAD_OPTIONS_PER_PASS

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
TEMPLATE = ROOT / "release" / "model_repo"
# Read by karar.py; bump both together when the layout changes.
FORMAT_VERSION = 1
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
SUMS = "SHA256SUMS"


def build(run: str, calibrator: Path, out: Path, models: Path = MODELS) -> dict[str, str]:
    """Write the repo into `out`; returns {file name: sha256}."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file

    from model.head.pack import pack

    weights = models / run / "model.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"no weights at {weights}")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    # The calibrator must belong to these weights; checked before anything is loaded.
    source_sha = sha256(weights)
    fitted = json.loads(calibrator.read_text(encoding="utf-8"))
    if fitted.get("weights_sha256") != source_sha:
        raise ValueError(f"{calibrator} was fitted on {fitted.get('weights') or 'unrecorded'} "
                         f"weights, not on {weights}")  # fmt: skip
    Calibration.from_file(calibrator, weights_sha256=source_sha)  # the adapter's own checks

    model, _ = load_model(weights)
    out.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().to(torch.float32).contiguous()
             for key, value in model.state_dict().items()}  # fmt: skip
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "model.safetensors").chmod(0o644)  # written owner-only, read like the others

    defaults = inspect.signature(pack).parameters
    config = {
        "format_version": FORMAT_VERSION,
        "model_id": "ufakzeka-karar",
        "source_run": run,
        "backbone": {"repo": BACKBONE, "revision": BACKBONE_REVISION},
        "backbone_config": asdict(model.backbone.cfg),
        "causal": model.causal,
        "pooling": model.pooling,
        "layout": getattr(model, "layout", "blind"),
        "max_prefix_tokens": defaults["max_prefix"].default,
        "max_option_tokens": defaults["max_option"].default,
        "options_per_pass": HEAD_OPTIONS_PER_PASS,
        "dtype": "float32",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    shipped = dict(fitted)
    # The shipped file names the model, not the internal run it was fitted on.
    shipped["weights"] = "ufakzeka-karar"
    shipped["weights_file"] = "model.safetensors"
    shipped["weights_sha256"] = sha256(out / "model.safetensors")
    shipped["source_weights_sha256"] = source_sha
    (out / "calibrator.json").write_text(json.dumps(shipped, indent=2) + "\n", encoding="utf-8")

    for name in TOKENIZER_FILES:
        shutil.copyfile(hf_hub_download(BACKBONE, name, revision=BACKBONE_REVISION), out / name)
    for name in ("karar.py", "requirements.txt"):
        shutil.copyfile(TEMPLATE / name, out / name)
    shutil.copyfile(ROOT / "LICENSE", out / "LICENSE")

    sums = {path.name: sha256(path) for path in sorted(out.iterdir())
            if path.is_file() and path.name != SUMS}  # fmt: skip
    (out / SUMS).write_text("".join(f"{digest}  {name}\n" for name, digest in sums.items()),
                            encoding="utf-8")  # fmt: skip
    return sums


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="a run name under models/")
    parser.add_argument("--calibrator", type=Path, required=True,
                        help="the step 7 calibrator fitted on that run's model.pt")  # fmt: skip
    parser.add_argument("--out", type=Path, required=True, help="an empty or absent folder")
    parser.add_argument("--models", type=Path, default=MODELS)
    args = parser.parse_args()
    sums = build(args.run, args.calibrator, args.out, args.models)
    for name, digest in sums.items():
        size = (args.out / name).stat().st_size
        print(f"{digest}  {size:>12,}  {name}")


if __name__ == "__main__":
    main()
