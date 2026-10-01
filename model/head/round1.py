"""Round 1 on the GPU service: the decision head on every track, and the arms of the round's plan.

    modal run model/head/round1.py --dry-run     # mixes locally and prints the plan
    modal run model/head/round1.py --only r1-main-s1
    modal run model/head/round1.py

Training data is every train file under data/built (typed/*, sss, synth; r4),
uploaded to the project volume with a hash of each file in the run's record.
Tasks are mixed in proportion to their size up to a cap of MIX_CAP rows a task
(PLAN.md: examples-proportional mixing with a cap), with a fixed seed, so a
28,000-row set cannot drown eighty templates of a hundred rows each.
Validation is every validation file; the held-out tasks (data/built/sss/
heldout_task.jsonl) are scored once, at the end, and never enter a choice.

Arms on the causal backbone with the slice's setting (mean pooling, backbone
rate 1e-4, two epochs), the ordinal term on score questions:

- r1-main-s{1,2,3}: weights back to each task's natural answer mix;
- r1-noweights-s1: the same without the weights;
- r1-nomined-s1: the same as main without the mined rows;

and the same head and data on the three Turkish encoders, each at
two rates (ENCODER_RATES), seed 1.

Every run saves its trained network (model.pt) for the int8 export and step 7,
answers MASSIVE's full 59-intent question through chunked inference, and writes
its summary under results/step4/round1/. Choosing among the runs waits for
HakemBench-dev, which waits for the owner's labels.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import modal

from model.head.mixing import (
    FOLDS,
    MIX_CAP,
    STRATA_SOURCES,
    fold_of,
    load_strata,
    mix,
    strata_from_records,
    take_fractions,
    train_files,
    unseen,
    validation_files,
)
from model.head.modal_app import CAUSAL, VOL, _git_commit, image, volume

APP_NAME = "ufakzeka-karar-round1"
GPU = "L4"
EPOCHS = 2.0
RATE = 1e-4
RUN_TIMEOUT = 3 * 3600
# r5 runs on H200 by the owner's leave (2026-09-27), with a tighter timeout
# so a stuck run cannot pass the ledger's cap: r4 took 36 minutes on L4, and H200
# measured 2.6 to 3.5 times L4's steps a second on this model.
R5_GPU = "H200"
R5_TIMEOUT = 30 * 60
DATA = VOL / "round1" / "data"
RUNS = VOL / "round1" / "runs"
ROUND2 = VOL / "round2"
BUILT = Path("data/built")

app = modal.App(APP_NAME, image=image)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# The encoders' own rates: each one's median of the rates the instrument chose
# for it (results/step1/chosen_rates.json), and 1e-4, the rate the slice's grid
# chose for our backbone under this head. Each encoder's better rate is read by
# mean validation log loss, the slice's rule, fixed before any number.
ENCODER_RATES = {"berturk": (8e-5, 1e-4), "tabibert": (3e-5, 1e-4), "moganbert": (1e-4,)}
# Two more seeds of each competitive encoder at the rate round 1 chose,
# so the comparison rests on three seeds a side; TabiBERT is left at one.
MORE_ENCODER_SEEDS = {"berturk": 1e-4, "moganbert": 1e-4}
# The continued backbone, exported by model/convert/modal_app.py.
CONTINUED = VOL / "convert" / "continue-v1" / "export"


def specs(git_commit: str) -> list[dict]:
    base = {"git_commit": git_commit, "prior_weights": True, "drop_mined": False,
            "backbone": "ufakzeka", "rate": RATE}  # fmt: skip
    ours = [
        *({**base, "name": f"r1-main-s{s}", "seed": s} for s in (1, 2, 3)),
        {**base, "name": "r1-noweights-s1", "seed": 1, "prior_weights": False},
        {**base, "name": "r1-nomined-s1", "seed": 1, "drop_mined": True},
    ]
    encoders = [{**base, "name": f"r1-{name}-lr{rate:.0e}-s1".replace("e-0", "e-"), "seed": 1,
                 "backbone": name, "rate": rate}
                for name, rates in ENCODER_RATES.items() for rate in rates]  # fmt: skip
    more = [{**base, "name": f"r1-{name}-lr{rate:.0e}-s{seed}".replace("e-0", "e-"),
             "seed": seed, "backbone": name, "rate": rate}
            for name, rate in MORE_ENCODER_SEEDS.items() for seed in (2, 3)]  # fmt: skip
    continued = [{**base, "name": f"r1-continued-s{seed}", "seed": seed,
                  "backbone": "continued"} for seed in (1, 2, 3)]  # fmt: skip
    # Round 2: the baseline on the filtered data, scoring its training rows
    # and the rows the mix left out; the weighted rounds read their weights file.
    round2 = [{**base, "name": f"r2-base-s{seed}", "seed": seed, "score": True}
              for seed in (1, 2, 3)]  # fmt: skip
    # The weighted rounds do not score their training rows: in-sample scores
    # were judged memorised, so round C rescores with a cross-fit.
    round2 += [{**base, "name": f"r2-{r}-s{seed}", "seed": seed,
                "row_weights": str(ROUND2 / f"weights_{r}.json")}
               for r in ("b", "c") for seed in (1, 2, 3)]  # fmt: skip
    # The cross-fit the round 2 plan falls back to when in-sample scores look
    # memorised: each fold trains on four fifths of the mix and scores the rest.
    round2 += [{**base, "name": f"r2-fold{f}", "seed": 1, "fold": f} for f in range(FOLDS)]
    # Step 6's ablations on the accepted data: the conventional sequential
    # head with shuffled option order (claim 5), and REINFORCE in place of the
    # proper score (claim 4). Everything else is r2-base's.
    round2 += [{**base, "name": f"r2-seq-s{seed}", "seed": seed, "layout": "sequential",
                "shuffle_options": True} for seed in (1, 2, 3)]  # fmt: skip
    round2 += [{**base, "name": f"r2-rl-s{seed}", "seed": seed, "loss": "rl"}
               for seed in (1, 2, 3)]  # fmt: skip
    # The retrain after the HakemBench v1.0 freeze: r2-base's recipe and the two
    # ablations on the training files cleaned against the frozen benchmark. No row
    # scoring: the confident-learning pass it fed was rejected.
    round3 = [{**base, "name": f"r3-base-s{seed}", "seed": seed} for seed in (1, 2, 3)]
    round3 += [{**base, "name": f"r3-seq-s{seed}", "seed": seed, "layout": "sequential",
                "shuffle_options": True} for seed in (1, 2, 3)]  # fmt: skip
    round3 += [{**base, "name": f"r3-rl-s{seed}", "seed": seed, "loss": "rl"}
               for seed in (1, 2, 3)]  # fmt: skip
    # The release model: r3-base's recipe with data/built/r4 added (the four support
    # questions labelled, new guardrail and moderation texts). r3 stays the held-out result.
    round4 = [{**base, "name": f"r4-base-s{seed}", "seed": seed} for seed in (1, 2, 3)]
    # r5: r4-base's recipe plus the conversation set's train split
    # (prompt_injection-conv, on disk under typed/), with r4's benign look-alikes halved
    # (r5a) or left out (r5b). The conversation set's validation rows are the guardrail
    # dev set: evaluated, never trained on, kept out of the prior weights.
    r5 = {"prior_exclude": ["prompt_injection-conv"], "gpu": R5_GPU, "timeout": R5_TIMEOUT}
    round5 = [{**base, **r5, "name": f"r5{v}-base-s{seed}", "seed": seed,
               "task_fraction": {"prompt_injection-r4": fraction}}
              for v, fraction in (("a", 0.5), ("b", 0.0)) for seed in (1, 2, 3)]  # fmt: skip
    return ours + encoders + more + continued + round2 + round3 + round4 + round5


@app.function(gpu=GPU, volumes={VOL.as_posix(): volume}, timeout=RUN_TIMEOUT, retries=0,
              max_containers=10, scaledown_window=2)  # fmt: skip
def train_round1(spec: dict) -> dict:
    import os
    from dataclasses import asdict

    import torch
    from transformers import AutoTokenizer

    from model.convert.native import from_pretrained
    from model.head.train import (
        HeadConfig,
        train_head,
    )

    os.environ["KARAR_GIT_COMMIT"] = spec["git_commit"]
    out = RUNS / spec["name"]
    if (out / "summary.json").is_file():
        return json.loads((out / "summary.json").read_text(encoding="utf-8")) | {
            "status": "skipped"
        }
    out.mkdir(parents=True, exist_ok=True)
    mixed = out / "train_mixed.jsonl"
    lines = mix(train_files(DATA), MIX_CAP, seed=1)
    scored = ()
    fold = spec.get("fold")
    if fold is not None:
        held = [x for x in lines if fold_of(x) == fold]
        lines = [x for x in lines if fold_of(x) != fold]
        (out / "fold.jsonl").write_text("\n".join(held) + "\n", encoding="utf-8")
        scored = (str(out / "fold.jsonl"),)
    fractions = spec.get("task_fraction") or {}
    before = {task: sum(json.loads(x)["task"] == task for x in lines) for task in fractions}
    if fractions:
        lines = take_fractions(lines, fractions, load_strata(DATA / "strata"))
    mixed.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if spec.get("score"):
        left_out = out / "unseen.jsonl"
        left_out.write_text("\n".join(unseen(train_files(DATA), MIX_CAP, seed=1)) + "\n",
                            encoding="utf-8")  # fmt: skip
        scored = (str(mixed), str(left_out))
    if spec.get("row_weights") and not Path(spec["row_weights"]).is_file():
        raise RuntimeError(f"no weights file at {spec['row_weights']}")

    if spec["backbone"] == "continued":
        exports = sorted(CONTINUED.glob("step_*"))
        if len(exports) != 1:
            raise RuntimeError(f"expected one continued export, found {exports}")
        backbone_id, revision = str(exports[0]), None
        tokenizer = AutoTokenizer.from_pretrained(backbone_id)
        backbone, _ = from_pretrained(backbone_id, None)
        embeddings = backbone.body.embed_tokens.weight
        causal = True
    elif spec["backbone"] == "ufakzeka":
        backbone_id, revision = CAUSAL
        tokenizer = AutoTokenizer.from_pretrained(backbone_id, revision=revision)
        backbone, _ = from_pretrained(backbone_id, revision)
        embeddings = backbone.body.embed_tokens.weight
        causal = True
    else:
        from model.head.encoder import ENCODERS, load_encoder

        backbone_id, revision = ENCODERS[spec["backbone"]]
        backbone, tokenizer = load_encoder(spec["backbone"])
        embeddings = backbone.model.get_input_embeddings().weight
        causal = False
    weights = embeddings.detach().cpu().numpy().tobytes()
    fingerprint = hashlib.blake2b(weights, digest_size=12).hexdigest()

    config = HeadConfig(
        backbone=backbone_id, revision=revision, causal=causal,
        train_files=(str(mixed),),
        validation_files=tuple(str(p) for p in validation_files(DATA)),
        out_dir=str(out), pooling="mean", backbone_lr=spec["rate"], epochs=EPOCHS,
        seed=spec["seed"], device="cuda", eval_points=1 if fold is not None else 4,
        ordinal=True, prior_weights=spec["prior_weights"], drop_mined=spec["drop_mined"],
        prior_exclude=tuple(spec.get("prior_exclude", ())),
        save_model=fold is None,
        score_files=scored, row_weights=spec.get("row_weights"),
        layout=spec.get("layout", "blind"), shuffle_options=spec.get("shuffle_options", False),
        loss=spec.get("loss", "ce"),
    )  # fmt: skip
    started = time.perf_counter()
    report = train_head(config, backbone, tokenizer)
    model = report.model
    # The trained network and the evaluations are on the volume before anything
    # after training can fail.
    volume.commit()
    summary = {
        "name": spec["name"], "spec": spec, "config": asdict(config),
        "weight_fingerprint": fingerprint,
        "data_hashes": {str(p.relative_to(DATA)): sha256(p)
                        for p in [*train_files(DATA), *validation_files(DATA),
                                  DATA / "sss" / "heldout_task.jsonl"]},
        "mixed_rows": len(lines), "steps": report.steps,
        "task_fraction_rows": {task: {"before": n, "after": sum(json.loads(x)["task"] == task
                                                               for x in lines)}
                               for task, n in before.items()},
        "strata_hashes": {p.name: sha256(p) for p in sorted((DATA / "strata").glob("*.json"))},
        "train_seconds": round(report.seconds, 1),
        "evaluations": report.evaluations,
        "gpu": torch.cuda.get_device_name(0),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "status": "trained",
    }  # fmt: skip
    try:
        if fold is None:  # a fold run is read for its scores alone
            summary |= after_training(model, tokenizer, out)
        summary["status"] = "done"
    except Exception as error:  # the paid training is kept; the record says what failed
        summary["after_training_error"] = repr(error)[:500]
    summary["wall_seconds"] = round(time.perf_counter() - started, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    return summary


@app.function(gpu=GPU, volumes={VOL.as_posix(): volume}, timeout=3600, retries=0)
def soup(runs: list[str], name: str) -> dict:
    """Average the runs' trained networks and evaluate the average like a run.

    Writes the same files a run does (predictions at the runs' last step, the
    held-out predictions, MASSIVE's 59 intents, summary.json), so every reader
    of a run reads the soup unchanged.
    """
    import torch
    from transformers import AutoTokenizer

    from model.convert.native import from_pretrained
    from model.head.head import DecisionHead
    from model.head.pack import pack
    from model.head.train import batches_by_length, evaluate, read_rows

    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    backbone_id, revision = CAUSAL
    tokenizer = AutoTokenizer.from_pretrained(backbone_id, revision=revision)
    backbone, _ = from_pretrained(backbone_id, revision)
    model = DecisionHead(backbone, causal=True, pooling="mean")
    model.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    states = [torch.load(RUNS / r / "model.pt", map_location="cpu") for r in runs]
    from model.head.soup import average_states

    model.load_state_dict(average_states(states))
    model.cuda().train(False)
    steps = json.loads((RUNS / runs[0] / "summary.json").read_text(encoding="utf-8"))["steps"]

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False).input_ids

    rows = read_rows(tuple(str(p) for p in validation_files(DATA)))
    packed = [(pack(r.state, r.question, encode, layout=model.layout), r) for r in rows]
    with torch.no_grad():
        metrics, records = evaluate(
            model, batches_by_length(packed, 64, random.Random(0), shuffle=False), "cuda"
        )
    (out / f"predictions_step{steps}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    summary = {"name": name, "soup_of": runs, "steps": steps,
               "evaluations": [{"step": steps, "metrics": metrics}]}  # fmt: skip
    summary |= after_training(model, tokenizer, out)
    summary["status"] = "done"
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    volume.commit()
    return summary


def after_training(model, tokenizer, run_dir: Path) -> dict:
    """The held-out tasks and MASSIVE's 59 intents, on the trained network."""
    from data.instrument_format import Row
    from data.typed.instrument import _question
    from model.head.infer import chunked_choice
    from model.head.pack import pack
    from model.head.train import batches_by_length, evaluate, macro_f1, read_rows
    from schema.questions import ChoiceQuestion

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False).input_ids

    # The held-out tasks, once, at the end (never used to choose).
    heldout = read_rows((str(DATA / "sss" / "heldout_task.jsonl"),))
    packed = [(pack(r.state, r.question, encode, layout=model.layout), r) for r in heldout]
    heldout_metrics, records = evaluate(
        model, batches_by_length(packed, 64, random.Random(0), shuffle=False), "cuda"
    )
    (run_dir / "predictions_heldout.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )

    # MASSIVE's full question, 59 options, with one softmax over all of them.
    # The instrument's validation split: 1,981 rows, three more than the
    # typed validation file, which decontamination trimmed for training only.
    meta = json.loads((VOL / "instrument/massive_tr/meta.json").read_text(encoding="utf-8"))
    question_dict, keys = _question("massive_tr", meta["labels"])
    question = ChoiceQuestion(**question_dict)
    gold, predicted = [], []
    for line in (VOL / "instrument/massive_tr/validation.jsonl").read_text("utf-8").splitlines():
        if line.strip():
            row = Row(**json.loads(line))
            probs = chunked_choice(model, row.text, question, encode, device="cuda")
            gold.append(keys[row.label])
            predicted.append(max(probs, key=probs.get))
    massive59 = {
        "rows": len(gold),
        "macro_f1": macro_f1(gold, predicted),
        "accuracy": sum(g == p for g, p in zip(gold, predicted, strict=True)) / len(gold),
    }
    result = {"heldout": heldout_metrics, "massive59": massive59}
    if model.layout == "sequential":
        # Claim 5: how far the answer moves with option order.
        from model.head.sensitivity import order_sensitivity, sample

        rows = read_rows(tuple(str(p) for p in validation_files(DATA)))
        result["order_sensitivity"] = order_sensitivity(model, sample(rows), encode, "cuda")
    return result


@app.function(volumes={VOL.as_posix(): volume}, cpu=4, memory=8192, timeout=3600)
def prefetch() -> dict:
    """The encoders' weights on the volume, committed, before any GPU waits for them.

    The first encoder run spent about 55 minutes of an idle L4 downloading its
    weights, because the probe's download was never committed.
    """
    from huggingface_hub import snapshot_download

    from model.head.encoder import ENCODERS

    out = {}
    for name, (repo, revision) in ENCODERS.items():
        start = time.perf_counter()
        path = snapshot_download(repo, revision=revision)
        out[name] = {"path": path, "seconds": round(time.perf_counter() - start, 1)}
        volume.commit()
    return out


@app.function(volumes={VOL.as_posix(): volume}, cpu=4, memory=16384, timeout=1800)
def probe_encoders() -> dict:
    """The head's guarantees on each wrapped encoder, before any is trained."""
    import torch

    from model.head.encoder import ENCODERS, check_blindness, load_encoder
    from model.head.head import DecisionHead

    torch.manual_seed(0)
    out = {}
    for name in ENCODERS:
        try:
            backbone, tokenizer = load_encoder(name)
            head = DecisionHead(backbone, causal=False, pooling="mean")
            out[name] = check_blindness(head, tokenizer)
        except Exception as error:  # a model that cannot take the mask is reported, not trained
            out[name] = {"passes": False, "error": repr(error)[:400]}
    return out


@app.function(gpu=GPU, volumes={VOL.as_posix(): volume}, timeout=1800)
def probe_memory() -> dict:
    """Peak memory and step time at the round's worst shape, on each backbone.

    Thirty-two questions, each a 448-token text and ten 48-token options (928
    tokens), forward and backward under the training's bf16 autocast, through
    the training's own micro-batching at two token budgets.
    """
    import torch
    from transformers import AutoTokenizer

    from model.convert.native import from_pretrained
    from model.head.encoder import ENCODERS, load_encoder
    from model.head.head import DecisionHead
    from model.head.pack import pack
    from model.head.train import HeadConfig, accumulate
    from schema.questions import ChoiceQuestion
    from schema.rows import TrainingRow

    out = {}
    for name in ("ufakzeka", *ENCODERS):
        try:
            if name == "ufakzeka":
                tokenizer = AutoTokenizer.from_pretrained(CAUSAL[0], revision=CAUSAL[1])
                backbone, _ = from_pretrained(*CAUSAL)
                causal = True
            else:
                backbone, tokenizer = load_encoder(name)
                causal = False
            model = DecisionHead(backbone, causal=causal, pooling="mean").cuda().train()

            def encode(text: str, tokenizer=tokenizer) -> list[int]:
                return tokenizer(text, add_special_tokens=False).input_ids

            words = "karar metin mahkeme başvuru hak ödeme fatura kargo iade süre " * 80
            options = {f"seçenek {i} " + words[i * 7 :]: None for i in range(10)}
            question = ChoiceQuestion(type="choice", instructions="Hangisi?", criteria=options)
            packed = pack(words, question, encode)
            row = TrainingRow(track="t", task="t", split="train", origin="authored",
                              label_kind="rule", source="authored", state=words,
                              question=question, target=dict.fromkeys(options, 0.1),
                              recipe="probe")  # fmt: skip
            group = [(packed, row)] * 32
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            autocast = torch.autocast("cuda", dtype=torch.bfloat16)
            result = {"tokens": len(packed.ids)}
            for budget in (12_000, 8_000):
                config = HeadConfig(backbone="probe", revision=None, causal=causal,
                                    train_files=(), validation_files=(), out_dir="/tmp",
                                    device="cuda", max_micro_tokens=budget)  # fmt: skip
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                times = []
                try:
                    for _ in range(3):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        optimizer.zero_grad(set_to_none=True)
                        accumulate(model, group, config, {}, autocast)
                        optimizer.step()
                        torch.cuda.synchronize()
                        times.append(time.perf_counter() - start)
                    result[str(budget)] = {
                        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                        "step_seconds": round(min(times), 3)}  # fmt: skip
                except torch.OutOfMemoryError:
                    result[str(budget)] = "out of memory"
            out[name] = result
            del model, optimizer, backbone
            torch.cuda.empty_cache()
        except Exception as error:
            out[name] = {"error": repr(error)[:400]}
    out["gpu"] = torch.cuda.get_device_name(0)
    out["total_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    return out


@app.local_entrypoint()
def main(dry_run: bool = False, only: str = "", probe: bool = False, fetch: bool = False,
         soup_of: str = "", soup_name: str = "") -> None:  # fmt: skip
    root = Path(__file__).resolve().parents[2]
    if soup_of:
        # A seed soup of finished runs: --soup-of r2-base-s1... --soup-name r2-base-soup
        summary = soup.remote(soup_of.split(","), soup_name)
        (root / "results/step4/round1").mkdir(parents=True, exist_ok=True)
        (root / "results/step4/round1" / f"{soup_name}.json").write_text(
            json.dumps(summary, indent=2) + "\n")  # fmt: skip
        print(json.dumps({k: summary[k] for k in ("name", "steps", "massive59")}, indent=2))
        return
    if fetch:
        print(json.dumps(prefetch.remote(), indent=2))
        return
    if probe:
        (root / "results/step4/round1").mkdir(parents=True, exist_ok=True)
        result = probe_encoders.remote()
        (root / "results/step4/round1/encoder_probe.json").write_text(json.dumps(result, indent=2))
        memory = probe_memory.remote()
        (root / "results/step4/round1/memory_probe.json").write_text(json.dumps(memory, indent=2))
        print(json.dumps({"guarantees": result, "memory": memory}, indent=2))
        return
    results = root / "results/step4/round1"
    results.mkdir(parents=True, exist_ok=True)
    lines = mix(train_files(root / BUILT))
    tasks = defaultdict(int)
    for line in lines:
        tasks[json.loads(line)["task"]] += 1
    print(f"mixed training rows {len(lines)} over {len(tasks)} tasks, "
          f"{round(len(lines) * EPOCHS / 32)} steps at 32 a batch")  # fmt: skip
    git_commit = "dry-run" if dry_run else _git_commit()
    todo = [s for s in specs(git_commit) if not (results / f"{s['name']}.json").is_file()]
    if only:
        # One or more named runs, the gate: their records are read before the rest.
        names = set(only.split(","))
        todo = [s for s in todo if s["name"] in names]
    else:
        # The continued arms wait for their export and round 2 waits for the
        # owner's labels, so they run only when named.
        todo = [s for s in todo if s["backbone"] != "continued"
                and not s["name"].startswith(("r2-", "r3-", "r4-", "r5"))]  # fmt: skip
    for spec in todo:
        print(f"  {spec['name']}{' on ' + spec['gpu'] if spec.get('gpu') else ''}")
    local_strata = {task: strata_from_records(root / path, fields)
                    for task, (path, fields) in STRATA_SOURCES.items()}  # fmt: skip
    read_fraction_mixes(todo, lines, local_strata)
    if dry_run or not todo:
        return
    with volume.batch_upload(force=True) as batch:
        for path in [*train_files(root / BUILT), *validation_files(root / BUILT),
                     root / BUILT / "sss" / "heldout_task.jsonl"]:  # fmt: skip
            batch.put_file(str(path), "/round1/data/" + str(path.relative_to(root / BUILT)))
        # Round 2's weights, written by model.head.scoring from the baseline's scores.
        for path in sorted((root / "results/step6/round2").glob("weights_*.json")):
            batch.put_file(str(path), f"/round2/{path.name}")
        # The strata a task_fraction cut reads, as text to stratum label.
        for task, strata in local_strata.items():
            body = json.dumps(strata, ensure_ascii=False, sort_keys=True).encode("utf-8")
            batch.put_file(io.BytesIO(body), f"/round1/data/strata/{task}.json")
    # Specs with their own GPU (r5) run in their own container pool; the rest keep
    # the round's GPU and timeout.
    groups = defaultdict(list)
    for spec in todo:
        groups[(spec.get("gpu"), spec.get("timeout"))].append(spec)
    for (gpu, timeout), group in groups.items():
        function = train_round1
        if gpu is not None:
            function = train_round1.with_options(
                gpu=gpu, timeout=timeout or RUN_TIMEOUT, volumes={VOL.as_posix(): volume}
            )
            print(f"  {len(group)} runs on {gpu}, timeout {timeout or RUN_TIMEOUT} s", flush=True)
        for summary in function.map(group, return_exceptions=True, order_outputs=False):
            if isinstance(summary, Exception):
                print(f"  FAILED {summary}", flush=True)
                continue
            (results / f"{summary['name']}.json").write_text(json.dumps(summary, indent=2) + "\n")
            massive = summary.get("massive59", {}).get("macro_f1")
            print(f"  {summary['name']} {summary.get('gpu')} {summary['wall_seconds']}s massive59 "
                  f"{'-' if massive is None else f'{massive:.3f}'}", flush=True)  # fmt: skip


def read_fraction_mixes(todo: list[dict], lines: list[str],
                        strata: dict[str, dict[str, str]]) -> None:  # fmt: skip
    """For each task_fraction among the runs, the mix it trains on, printed for reading."""
    seen = set()
    for spec in todo:
        fractions = spec.get("task_fraction")
        key = json.dumps(fractions, sort_keys=True)
        if not fractions or key in seen:
            continue
        seen.add(key)
        cut = take_fractions(lines, fractions, strata)
        tasks = defaultdict(int)
        for line in cut:
            tasks[json.loads(line)["task"]] += 1
        print(f"  mix for {key}: {len(cut)} rows over {len(tasks)} tasks, "
              f"{round(len(cut) * EPOCHS / 32)} steps at 32 a batch")  # fmt: skip
        for task in sorted(tasks):
            print(f"    {task} {tasks[task]}")
        for task in fractions:
            labels = defaultdict(int)
            for line in cut:
                row = json.loads(line)
                if row["task"] == task:
                    labels[strata.get(task, {}).get(row["state"], "?").split("|")[0]] += 1
            if labels:
                print(f"    {task} by stratum: {dict(sorted(labels.items()))}")
