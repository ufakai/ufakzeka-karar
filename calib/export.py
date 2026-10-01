"""The decision model as an ONNX file, its int8 forms, and what each costs on a CPU.

The week-1 slice asks for an int8 export and a latency and memory check on the serving
box before the data build; this is that check. Three steps; only the
first needs torch, so the box needs ONNX Runtime and numpy alone:

    python -m calib.export build --out .cache/karar/export      # torch, the laptop
    python -m calib.export quantize --dir .cache/karar/export   # ONNX Runtime
    python -m calib.export bench --dir .cache/karar/export --engine int8-dynamic \\
        --threads 1 --out results/step4/serve/box-int8-dynamic-t1.json

`build` loads the backbone, puts the decision head on it (trained weights from
--head, or a fresh head: latency and memory do not depend on the values),
exports one question per call with the sequence length and the number of
options free, and writes three files: fp32, int8 by dynamic quantization, which
ONNX Runtime's guide recommends for transformer models ("dynamic quantization
for RNNs and transformer-based models, and static quantization for CNN
models", onnxruntime.ai quantization page, read 2026-09-23), and int8 by static
quantization in the QDQ S8S8 format the guide calls the first choice, which
had been planned. It also writes a sample of real packed questions, drawn from
the validation files, and the fp32 torch outputs on them.

`bench` runs one engine in a fresh process on that sample, one question at a
time as a server would, and reports latency percentiles, the process's peak
memory, the file size, and how far the probabilities move from fp32 torch.
With a fresh head the last number measures the arithmetic of the backbone, not
a trained model's accuracy; that comparison waits for trained weights.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np

INPUTS = ("ids", "positions", "segments", "option_segments", "option_valid", "caller_slot")
BACKBONE = "ufakai/ufakzeka-1-base"
# The revision every head run uses (model/head/modal_app.py CAUSAL).
REVISION = "f9e11eea28cbb2ba953a5628f972d416fe0c3cfe"
SAMPLE_FILES = (
    "data/built/sss/validation.jsonl",
    "data/built/typed/massive_tr/validation.jsonl",
    "data/built/typed/offenseval_tr/validation.jsonl",
    "data/built/typed/mide22/validation.jsonl",
)
ENGINES = {"fp32": "model.onnx", "int8-dynamic": "model-int8-dynamic.onnx",
           "int8-dynamic-per-channel": "model-int8-dynamic-pc.onnx",
           "int8-static": "model-int8-static.onnx",
           # Step 7: the backbone's matrix products only, the embeddings
           # and both heads kept in fp32.
           "int8-matmul": "model-int8-matmul.onnx",
           "int8-matmul-per-channel": "model-int8-matmul-pc.onnx",
           # The down projections kept in fp32 too, chosen on fit rows.
           "int8-no-down": "model-int8-no-down.onnx"}  # fmt: skip


def head_matmuls(model) -> list[str]:
    """The head's MatMul nodes: products with a constant (width, 1) weight.

    Node names from the exporter are not stable across exports, and the module
    path in a node's metadata does not survive every graph pass, so the heads
    are found by what they are. The option scorer, applied to a (B, N, width)
    tensor, exports as such a MatMul; the abstain output, on a (B, width)
    tensor, exports as a Gemm, which dynamic quantization never touches
    (it is not in ONNX Runtime's integer-ops registry).
    """
    weights = {i.name: list(i.dims) for i in model.graph.initializer}
    return [n.name for n in model.graph.node
            if n.op_type == "MatMul" and len(n.input) > 1
            and weights.get(n.input[1], [0, 0])[-1] == 1]  # fmt: skip


def head_gemms(model) -> list[str]:
    """Gemm nodes with a (1, width) weight: the abstain output."""
    weights = {i.name: list(i.dims) for i in model.graph.initializer}
    return [
        n.name
        for n in model.graph.node
        if n.op_type == "Gemm" and any(weights.get(i, [0, 0])[0] == 1 for i in n.input[1:])
    ]


def down_projections(model) -> list[str]:
    """MatMuls whose weight contracts the feed-forward width back to the model width.

    The feed-forward width is the largest input width of any product (2048
    here, against 768 for every other weight), and the down projection is the
    only product that takes it in. The fit-row ranking found these the
    source of most of int8's error.
    """
    weights = {i.name: list(i.dims) for i in model.graph.initializer}
    products = [(n.name, weights[n.input[1]]) for n in model.graph.node
                if n.op_type == "MatMul" and len(n.input) > 1 and n.input[1] in weights
                and len(weights[n.input[1]]) == 2]  # fmt: skip
    widest = max(dims[0] for _, dims in products)
    return [name for name, dims in products if dims[0] == widest and dims[1] < widest]


def quantize_no_down(folder: Path) -> dict:
    """int8 of the backbone's products except the down projections."""
    import onnx
    from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic

    model = onnx.load(str(folder / ENGINES["fp32"]))
    del model.graph.value_info[:]
    stripped, prepared = folder / "model-stripped.onnx", folder / "model-prep.onnx"
    onnx.save(model, str(stripped))
    quant_pre_process(str(stripped), str(prepared), skip_symbolic_shape=True)
    stripped.unlink()
    graph = onnx.load(str(prepared))
    heads, down = head_matmuls(graph), down_projections(graph)
    if len(heads) != 1 or len(down) != 24:
        raise RuntimeError(f"expected 1 scorer and 24 down projections, found {heads} {len(down)}")
    quantize_dynamic(str(prepared), str(folder / ENGINES["int8-no-down"]),
                     weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                     nodes_to_exclude=heads + down)  # fmt: skip
    prepared.unlink()
    kinds = [n.op_type for n in onnx.load(str(folder / ENGINES["int8-no-down"])).graph.node]
    return {
        "excluded": len(heads) + len(down),
        "bytes": (folder / ENGINES["int8-no-down"]).stat().st_size,
        "integer_products": kinds.count("MatMulInteger"),
        "fp32_products_left": kinds.count("MatMul"),
    }


def quantize_backbone(folder: Path) -> dict:
    """Dynamic int8 of the backbone's matrix products, heads and embeddings kept in fp32.

    ONNX Runtime's dynamic mode also quantizes Gather (the embeddings) unless
    told otherwise, so only MatMul is named; the serving box has AVX-512 VNNI,
    so reduce_range stays off (onnxruntime quantization docs, read 2026-09-24).
    """
    import onnx
    from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic

    model = onnx.load(str(folder / ENGINES["fp32"]))
    del model.graph.value_info[:]
    stripped = folder / "model-stripped.onnx"
    onnx.save(model, str(stripped))
    prepared = folder / "model-prep.onnx"
    quant_pre_process(str(stripped), str(prepared), skip_symbolic_shape=True)
    stripped.unlink()
    graph = onnx.load(str(prepared))
    heads, gemms = head_matmuls(graph), head_gemms(graph)
    if len(heads) != 1 or len(gemms) != 1:
        raise RuntimeError(
            f"expected one scorer MatMul and one abstain Gemm, found {heads} {gemms}"
        )
    report = {"heads_excluded": heads, "abstain_gemm_left_fp32": gemms}
    # The earlier default beside the two new files, for comparison:
    # every MatMul and Gather, the heads included.
    quantize_dynamic(str(prepared), str(folder / ENGINES["int8-dynamic"]),
                     weight_type=QuantType.QInt8)  # fmt: skip
    for engine, per_channel in (("int8-matmul", False), ("int8-matmul-per-channel", True)):
        quantize_dynamic(str(prepared), str(folder / ENGINES[engine]),
                         weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                         nodes_to_exclude=heads, per_channel=per_channel)  # fmt: skip
        quantized = onnx.load(str(folder / ENGINES[engine]))
        kinds = [n.op_type for n in quantized.graph.node]
        report[engine] = {
            "bytes": (folder / ENGINES[engine]).stat().st_size,
            "integer_products": sum(k in ("MatMulInteger", "DynamicQuantizeMatMul") for k in kinds),
            "fp32_products_left": kinds.count("MatMul"),
            # The token embeddings stay a float32 initializer only if Gather was left alone.
            "embeddings_fp32": any(
                i.name.endswith("embed_tokens.weight") and i.data_type == 1
                for i in quantized.graph.initializer
            ),
        }
    prepared.unlink()
    return report


def sample_rows(count: int, seed: int = 1, files: tuple = SAMPLE_FILES) -> list:
    from schema.rows import TrainingRow

    rng = random.Random(seed)
    rows = []
    per_file = count // len(files)
    for path in files:
        lines = [x for x in Path(path).read_text("utf-8").splitlines() if x.strip()]
        take = rng.sample(lines, min(per_file, len(lines)))
        rows += [TrainingRow.model_validate_json(x) for x in take]
    # Shuffled, so the scored rows and the calibration rows both span every file.
    rng.shuffle(rows)
    return rows


def build(
    out: Path, head_path: Path | None, count: int, calibrate: int, root: Path | None = None
) -> dict:
    """The ONNX file and its sample; `root` samples every validation file under it."""
    import torch
    from transformers import AutoTokenizer

    from model.convert.native import from_pretrained
    from model.head.head import DecisionHead
    from model.head.pack import Batch, collate, pack

    torch.manual_seed(0)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, revision=REVISION)
    backbone, _ = from_pretrained(BACKBONE, REVISION)
    model = DecisionHead(backbone, causal=True, pooling="mean").eval()
    if head_path is not None:
        model.load_state_dict(torch.load(head_path, map_location="cpu"))
    model.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    class Graph(torch.nn.Module):
        """The head with plain tensor inputs, which an exporter can trace."""

        def __init__(self, head):
            super().__init__()
            self.head = head

        def forward(self, ids, positions, segments, option_segments, option_valid, caller_slot):
            batch = Batch(ids, positions, segments, option_segments, option_valid, caller_slot,
                          keys=[])  # fmt: skip
            logits, abstain = self.head(batch)
            return torch.softmax(logits, -1), abstain

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False).input_ids

    if root is not None:
        from model.head.mixing import validation_files

        files = tuple(str(p) for p in validation_files(root))
        rows = sample_rows(count + calibrate, files=files)
    else:
        rows = sample_rows(count + calibrate)
    # The files may hold fewer rows than asked; the calibration rows are kept,
    # and the scored sample is what remains.
    count = min(count, len(rows) - calibrate)
    packed = [pack(r.state, r.question, encode) for r in rows]
    tensors = []
    for p in packed:
        b = collate([p], model.pad_id)
        tensors.append((b.ids, b.positions, b.segments, b.option_segments, b.option_valid,
                        b.caller_slot))  # fmt: skip
    graph = Graph(model).eval()
    length = torch.export.Dim("length", min=2, max=1024)
    options = torch.export.Dim("options", min=2, max=10)
    shapes = {"ids": {1: length}, "positions": {1: length}, "segments": {1: length},
              "option_segments": {1: options}, "option_valid": {1: options},
              "caller_slot": {1: options}}  # fmt: skip
    example = max(tensors[:count], key=lambda t: t[3].shape[1])
    fp32 = out / ENGINES["fp32"]
    torch.onnx.export(graph, example, str(fp32), input_names=list(INPUTS),
                      output_names=["probs", "abstain"], dynamic_shapes=shapes,
                      external_data=False)  # fmt: skip

    reference = []
    with torch.no_grad():
        for t in tensors[:count]:
            reference.append(graph(*t)[0].numpy())
    arrays = {}
    for i, t in enumerate(tensors[:count]):
        for name, value in zip(INPUTS, t, strict=True):
            arrays[f"{i}:{name}"] = value.numpy()
        arrays[f"{i}:reference"] = reference[i]
    np.savez_compressed(out / "sample.npz", **arrays)
    # What each sampled question asks and its target, keyed like the probabilities
    # (the caller's order), for the quality comparison of step 7.
    meta = [{"task": r.task, "keys": p.keys, "target": r.target}
            for r, p in zip(rows[:count], packed[:count], strict=True)]  # fmt: skip
    (out / "sample_meta.json").write_text(json.dumps(meta, ensure_ascii=False), "utf-8")
    calibration = {f"{i}:{name}": value.numpy() for i, t in enumerate(tensors[count:])
                   for name, value in zip(INPUTS, t, strict=True)}  # fmt: skip
    np.savez_compressed(out / "calibration.npz", **calibration)
    return {
        "sample": count,
        "calibration": calibrate,
        "head": str(head_path or "fresh"),
        "fp32_bytes": fp32.stat().st_size,
    }


def quantize(folder: Path) -> dict:
    """The two int8 files from the fp32 one, with ONNX Runtime only."""
    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quant_pre_process,
        quantize_dynamic,
        quantize_static,
    )

    fp32 = folder / ENGINES["fp32"]
    # The exporter stores intermediate shapes, and one of them disagrees with
    # what ONNX infers (768 against 1), which stops every shape pass the
    # quantizer runs. Stored intermediate shapes are hints only, so they are
    # dropped and inferred again.
    model = onnx.load(str(fp32))
    del model.graph.value_info[:]
    stripped = folder / "model-stripped.onnx"
    onnx.save(model, str(stripped))
    prepared = folder / "model-prep.onnx"
    # Symbolic shape inference fails on the mask's Range node; ONNX shape
    # inference and the graph optimisation still run.
    quant_pre_process(str(stripped), str(prepared), skip_symbolic_shape=True)
    stripped.unlink()
    quantize_dynamic(str(prepared), str(folder / ENGINES["int8-dynamic"]),
                     weight_type=QuantType.QInt8)  # fmt: skip
    # One scale per output channel, which the ONNX Runtime guide suggests when
    # "the accuracy loss is large".
    quantize_dynamic(str(prepared), str(folder / ENGINES["int8-dynamic-per-channel"]),
                     weight_type=QuantType.QInt8, per_channel=True)  # fmt: skip
    data = np.load(folder / "calibration.npz")
    count = len({k.split(":")[0] for k in data.files})

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items = iter(range(count))

        def get_next(self):
            i = next(self.items, None)
            return None if i is None else {n: data[f"{i}:{n}"] for n in INPUTS}

    quantize_static(str(prepared), str(folder / ENGINES["int8-static"]), Reader(),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, per_channel=True)  # fmt: skip
    prepared.unlink()
    return {name: (folder / file).stat().st_size for name, file in ENGINES.items()}


def peak_memory_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 1024  # bytes on macOS, KB on Linux


def bench(folder: Path, engine: str, threads: int, repeats: int) -> dict:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    base = peak_memory_mb()
    session = ort.InferenceSession(str(folder / ENGINES[engine]), options,
                                   providers=["CPUExecutionProvider"])  # fmt: skip
    loaded = peak_memory_mb()
    data = np.load(folder / "sample.npz")
    count = len({k.split(":")[0] for k in data.files})
    items = [({n: data[f"{i}:{n}"] for n in INPUTS}, data[f"{i}:reference"]) for i in range(count)]
    for feed, _ in items[:5]:
        session.run(["probs"], feed)
    times, lengths, diffs, agree = [], [], [], []
    for _ in range(repeats):
        for feed, _reference in items:
            start = time.perf_counter()
            session.run(["probs"], feed)
            times.append(time.perf_counter() - start)
    for feed, reference in items:
        probs = session.run(["probs"], feed)[0]
        valid = feed["option_valid"][0]
        diffs.append(float(np.abs(probs[0][valid] - reference[0][valid]).max()))
        agree.append(int(np.argmax(probs[0]) == np.argmax(reference[0])))
        lengths.append(int(feed["ids"].shape[1]))
    ms = np.array(times) * 1000
    return {
        "engine": engine,
        "threads": threads,
        "machine": platform.machine(),
        "processor": platform.processor() or platform.platform(),
        "onnxruntime": ort.__version__,
        "questions": count,
        "tokens_p50": int(np.percentile(lengths, 50)),
        "tokens_max": max(lengths),
        "latency_ms_p50": round(float(np.percentile(ms, 50)), 2),
        "latency_ms_p95": round(float(np.percentile(ms, 95)), 2),
        "latency_ms_mean": round(float(ms.mean()), 2),
        "peak_memory_mb": round(peak_memory_mb(), 1),
        "memory_for_session_mb": round(loaded - base, 1),
        "file_mb": round((folder / ENGINES[engine]).stat().st_size / 2**20, 1),
        "max_prob_diff_vs_torch_fp32": round(max(diffs), 5),
        "mean_prob_diff_vs_torch_fp32": round(float(np.mean(diffs)), 5),
        "top1_agreement_vs_torch_fp32": round(sum(agree) / len(agree), 4),
    }


def scores(probs: list, meta: list[dict]) -> dict:
    """Accuracy, Brier, top-label ECE (15 bins) and mean-over-tasks macro F1."""
    from collections import defaultdict

    correct, brier, confidence, by_task = [], [], [], defaultdict(list)
    for p, m in zip(probs, meta, strict=True):
        keys = m["keys"]
        target = np.array([m["target"][k] for k in keys])
        pred = np.asarray(p[: len(keys)], dtype=float)
        gold, guess = int(np.argmax(target)), int(np.argmax(pred))
        correct.append(gold == guess)
        confidence.append(float(pred[guess]))
        brier.append(float(((pred - target) ** 2).sum()))
        by_task[m["task"]].append((keys[gold], keys[guess]))
    correct_a, confidence_a = np.array(correct, dtype=float), np.array(confidence)
    edges = np.linspace(0, 1, 16)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        inside = (confidence_a > lo) & (confidence_a <= hi)
        if inside.any():
            ece += inside.mean() * abs(correct_a[inside].mean() - confidence_a[inside].mean())
    f1s = []
    for pairs in by_task.values():
        labels = {g for g, _ in pairs} | {q for _, q in pairs}
        per = []
        for label in labels:
            tp = sum(g == q == label for g, q in pairs)
            fp = sum(q == label != g for g, q in pairs)
            fn = sum(g == label != q for g, q in pairs)
            per.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
        f1s.append(sum(per) / len(per))
    return {"questions": len(correct), "accuracy": round(float(correct_a.mean()), 4),
            "brier": round(float(np.mean(brier)), 4), "ece": round(float(ece), 4),
            "macro_f1_mean_over_tasks": round(float(np.mean(f1s)), 4)}  # fmt: skip


def quality(folder: Path, engine: str, threads: int = 4) -> dict:
    """Step 7's comparison: the engine's scores beside torch fp32's on the same sample."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(folder / ENGINES[engine]), options,
                                   providers=["CPUExecutionProvider"])  # fmt: skip
    data = np.load(folder / "sample.npz")
    meta = json.loads((folder / "sample_meta.json").read_text("utf-8"))
    engine_probs, reference = [], []
    for i in range(len(meta)):
        feed = {n: data[f"{i}:{n}"] for n in INPUTS}
        engine_probs.append(session.run(["probs"], feed)[0][0])
        reference.append(data[f"{i}:reference"][0])
    ours, torch_fp32 = scores(engine_probs, meta), scores(reference, meta)
    flips = sum(int(np.argmax(e[: len(m["keys"])]) != np.argmax(r[: len(m["keys"])]))
                for e, r, m in zip(engine_probs, reference, meta, strict=True))  # fmt: skip
    return {"engine": engine, "engine_scores": ours, "torch_fp32_scores": torch_fp32,
            "answers_changed_vs_fp32": flips,
            "points_lost": {k: round(100 * (torch_fp32[k] - ours[k]), 3)
                            for k in ("accuracy", "macro_f1_mean_over_tasks")},
            "ece_points_added": round(100 * (ours["ece"] - torch_fp32["ece"]), 3)}  # fmt: skip


def outputs(folder: Path, engines: list[str], files: list[Path], out: Path,
            threads: int = 8) -> dict:  # fmt: skip
    """Every row of `files` through each engine, one question at a time (step 7).

    Writes one JSON line per row: its id, task, question type, option keys in the
    caller's order, target, and per engine the log-probabilities and the abstain
    logit. Calibration and the int8 gate are fitted from this file, so they read
    exactly what the shipped file computes.
    """
    import onnxruntime as ort
    from transformers import AutoTokenizer

    from model.head.pack import collate, pack
    from model.head.train import read_rows

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, revision=REVISION)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False).input_ids

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    sessions = {e: ort.InferenceSession(str(folder / ENGINES[e]), options,
                                        providers=["CPUExecutionProvider"])
                for e in engines}  # fmt: skip
    rows = read_rows(tuple(str(f) for f in files))
    started = time.perf_counter()
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            packed = pack(row.state, row.question, encode)
            b = collate([packed], pad)
            feed = dict(zip(INPUTS, (b.ids.numpy(), b.positions.numpy(), b.segments.numpy(),
                                     b.option_segments.numpy(), b.option_valid.numpy(),
                                     b.caller_slot.numpy()), strict=True))  # fmt: skip
            record = {"row_id": row.row_id, "task": row.task, "type": row.question.type,
                      "split": row.split, "keys": packed.keys, "target": row.target}  # fmt: skip
            for engine, session in sessions.items():
                probs, abstain = session.run(["probs", "abstain"], feed)
                p = np.clip(probs[0][: len(packed.keys)].astype(float), 1e-12, 1.0)
                record[engine] = {"logprobs": np.log(p).round(6).tolist(),
                                  "abstain": float(abstain[0])}  # fmt: skip
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"rows": len(rows), "engines": engines,
            "seconds": round(time.perf_counter() - started, 1)}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="calib.export")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, default=Path(".cache/karar/export"))
    b.add_argument("--head", type=Path)
    b.add_argument("--sample", type=int, default=200)
    b.add_argument("--calibrate", type=int, default=100)
    q = sub.add_parser("quantize")
    q.add_argument("--dir", type=Path, default=Path(".cache/karar/export"))
    r = sub.add_parser("bench")
    r.add_argument("--dir", type=Path, default=Path(".cache/karar/export"))
    r.add_argument("--engine", choices=list(ENGINES), required=True)
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build(args.out, args.head, args.sample, args.calibrate)
    elif args.command == "quantize":
        result = quantize(args.dir)
    else:
        result = bench(args.dir, args.engine, args.threads, args.repeats)
    text = json.dumps(result, indent=2)
    if getattr(args, "out", None) and args.command == "bench":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
