"""The conversion on Modal: one long run that survives being interrupted.

This is shaped differently from the instrument's app. That one dispatched a
hundred short independent runs and its risk was a pool that would not fan out.
This is a single run of many hours, and its risk is losing it: a Function call
is capped at 24 hours and a GPU container cannot be made non-preemptible, so
the run is built to be stopped and continued rather than to finish in one go
(modal.com/docs/guide/preemption and /examples/long-training, read
2026-09-20).

Three settings carry that. `retries` restarts immediately after a preemption
rather than backing off, `single_use_containers` makes each restart a clean
one, and the entrypoint spawns rather than calls so the handle does not expire
at 24 hours while the work continues. The loop itself checkpoints to the volume
and resumes from it, with the stream's position travelling in the checkpoint.

    modal run model/convert/modal_app.py::main --phase rebuild
    modal run model/convert/modal_app.py::main --phase smoke
    modal run model/convert/modal_app.py::main --phase pilot
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

from model.convert.corpus import MIX_DECAY, MIX_STABLE

APP_NAME = "ufakzeka-karar-convert"
KARAR_VOLUME = "ufakzeka-karar"
# The backbone's own corpus, read only. Nothing here ever writes to it.
DATA_VOLUME = "ufakzeka-data"

# H100 is what the run was costed on. The default used to be L4, which the
# same entry measured at more than twice the price for the same tokens.
GPU = os.environ.get("KARAR_GPU", "H100")
MAX_CONTAINERS = 1  # one run, not a pool
HOUR = 3600

VOL = Path("/vol")
DATA = Path("/corpus")
CONVERT_ROOT = VOL / "convert"
TOKENS_ROOT = VOL / "convert_tokens"
HF_HOME = VOL / "hf"

karar_volume = modal.Volume.from_name(KARAR_VOLUME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.0",
        "transformers==5.17.0",
        "numpy>=2",
        "huggingface-hub>=1.0",
        "hf-xet>=1.6",
        "pyarrow>=17",
    )
    # The fast tokenizer parallelises across a batch only when this is on. It is
    # normally turned off because a forked dataloader worker deadlocks with it;
    # nothing here forks, and with it off the rebuild used one core of sixteen
    # and took sixteen minutes a shard.
    .env(
        {
            "HF_HOME": str(HF_HOME),
            "TOKENIZERS_PARALLELISM": "true",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .add_local_python_source("data", "model")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={VOL.as_posix(): karar_volume, DATA.as_posix(): data_volume},
    timeout=6 * HOUR,
    cpu=16.0,
    memory=32768,
)
def rebuild_tier_a(backbone: str, revision: str, limit_shards: int = 0) -> dict:
    """Retokenise tier A from stage2, without the ShareAlike source.

    stage2 already carries the tier decision and the source column, so this
    filters rather than reclassifies. Tiers B and C need none of this: they
    are wholly ODC-By and their tokenised shards are used as they stand.
    """
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from model.convert.corpus import excluded_sources

    banned = set(excluded_sources())
    tokenizer = AutoTokenizer.from_pretrained(backbone, revision=revision)
    separator = tokenizer.eos_token_id

    source_dir = DATA / "stage2/train/A"
    shards = sorted(source_dir.glob("part_*.parquet"))
    if limit_shards:
        shards = shards[:limit_shards]
    if not shards:
        raise RuntimeError(f"no parquet shards under {source_dir}")

    out_dir = TOKENS_ROOT / "A"
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = dropped = tokens = 0
    started = time.perf_counter()

    # A preempted container restarts with the same input and would otherwise
    # retokenise every shard it had already finished. Observed happening on the
    # very first run of this function, so it is not hypothetical. A shard is
    # finished only once its marker exists, which is written after the data, so
    # a container killed mid-write leaves no marker and the shard is redone.
    def finished_marker(index: int) -> Path:
        return out_dir / f"part_{index}.done"

    # Documents are tokenised in batches and appended as they go. A shard holds
    # hundreds of thousands of documents, and the obvious version builds one
    # Python list of every token id first, which at 28 bytes an integer is
    # gigabytes for a file that is megabytes on disk.
    batch_size = 2_000

    for index, shard in enumerate(shards):
        marker = finished_marker(index)
        if marker.is_file():
            done = json.loads(marker.read_text(encoding="utf-8"))
            kept += done["kept"]
            dropped += done["dropped"]
            tokens += done["tokens"]
            print(f"  shard {index}: already done, {done['tokens']:,} tokens", flush=True)
            continue
        table = pq.read_table(shard, columns=["source", "text"])
        sources = table.column("source").to_pylist()
        texts = table.column("text").to_pylist()
        keep = [t for s, t in zip(sources, texts, strict=True) if s not in banned]
        shard_kept, shard_dropped = len(keep), len(texts) - len(keep)
        del sources, texts, table
        if not keep:
            continue

        written = 0
        # Raw uint16, the same format the rest of the corpus is in, so one
        # reader serves every tier.
        with (out_dir / f"part_{index}.bin").open("wb") as handle:
            for start in range(0, len(keep), batch_size):
                encoded = tokenizer(keep[start : start + batch_size], add_special_tokens=False)
                flat: list[int] = []
                for ids in encoded["input_ids"]:
                    flat.extend(ids)
                    flat.append(separator)
                array = np.asarray(flat, dtype=np.uint16)
                array.tofile(handle)
                written += int(array.size)
        del keep

        kept += shard_kept
        dropped += shard_dropped
        tokens += written
        # Written after the data, so a container killed mid-write leaves no
        # marker and the shard is redone rather than half counted.
        marker.write_text(
            json.dumps({"kept": shard_kept, "dropped": shard_dropped, "tokens": written}),
            encoding="utf-8",
        )
        karar_volume.commit()
        print(f"  shard {index}: {written:,} tokens from {shard_kept:,} documents", flush=True)

    report = {
        "shards": len(shards),
        "documents_kept": kept,
        "documents_dropped": dropped,
        "dropped_sources": sorted(banned),
        "tokens": tokens,
        "seconds": round(time.perf_counter() - started, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    karar_volume.commit()
    return report


@app.function(
    gpu=GPU,
    volumes={VOL.as_posix(): karar_volume, DATA.as_posix(): data_volume},
    timeout=24 * HOUR,
    max_containers=MAX_CONTAINERS,
    # Straight back after a preemption, and clean each time. Ten restarts is
    # the long-training example's figure and covers a run of many hours.
    retries=modal.Retries(max_retries=10, initial_delay=0.0),
    single_use_containers=True,
)
def convert(settings: dict) -> dict:
    """One conversion run, resuming from its own checkpoints if there are any."""
    import torch
    from transformers import AutoTokenizer

    from model.convert.checkpoint import rungs
    from model.convert.schedule import steps_to_reach
    from model.convert.stream import MixedStream, ShardedWindows, staged_mixture
    from model.convert.train import ConvertConfig, train

    name = settings.pop("name")
    tier_paths = settings.pop("tiers")
    mixture = settings.pop("mixture")
    decay_mixture = settings.pop("decay_mixture", mixture)
    out_dir = CONVERT_ROOT / name

    tokenizer = AutoTokenizer.from_pretrained(settings["backbone"], revision=settings["revision"])
    mask_token = settings.pop("mask_token")
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    if mask_id is None or mask_id < 0:
        raise RuntimeError("the mask token is not in this tokenizer's vocabulary")

    # Where this launch stops, which for a trunk is the next rung and not the
    # end of its schedule, and the rung a decay branch starts from.
    stop_tokens = settings.pop("stop_tokens", 0)
    branch_from = settings.pop("branch_from", None)
    if branch_from and not Path(branch_from).is_file():
        return {"preflight_failed": f"no rung at {branch_from}"}
    settings["rungs"] = tuple(settings.get("rungs", ()))
    attention = settings.pop("attn_implementation", "sdpa")
    compiled = settings.pop("compile", False)
    model, model_config = _build_model(
        settings["backbone"],
        settings["revision"],
        settings.get("engine", "library"),
        attention,
        compiled,
    )
    config = ConvertConfig(
        mask_id=int(mask_id),
        separator_id=int(tokenizer.eos_token_id),
        pad_id=int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1),
        vocab_size=int(model_config.vocab_size),
        **settings,
    )

    free_bytes, _total = torch.cuda.mem_get_info()
    checks = _gate(
        settings,
        tier_paths,
        mixture,
        tokenizer,
        model_config,
        free_bytes,
        mask_token,
        model=model,
        decay_mixture=decay_mixture,
    )
    from model.convert.preflight import report

    print(report(checks), flush=True)
    failed = [check for check in checks if not check.ok]
    if failed:
        # Returned, not raised. An exception here is retried ten times, and a
        # failed preflight fails identically every time, on a GPU each time.
        return {"preflight_failed": report(failed)}

    sources = {}
    for tier, folder in tier_paths.items():
        files = sorted(Path(folder).glob("*.bin"))
        if not files:
            raise RuntimeError(f"tier {tier}: no token shards under {folder}")
        sources[tier] = ShardedWindows(files, config.context)
    stream = MixedStream(
        sources,
        staged_mixture(mixture, decay_mixture, config.total_steps, config.decay_frac),
        seed=config.seed,
    )

    report = train(
        model,
        config,
        stream,
        out_dir=out_dir,
        device="cuda",
        on_checkpoint=karar_volume.commit,
        max_steps=steps_to_reach(stop_tokens, config.batch_tokens) if stop_tokens else None,
        branch_from=Path(branch_from) if branch_from else None,
    )
    out = {
        "name": name,
        "steps": report.steps,
        "tokens": report.tokens,
        "rungs": sorted(rungs(out_dir)),
        "wall_seconds": round(report.wall_seconds, 1),
        "tokens_per_second": round(report.tokens_per_second, 1),
        "first_losses": [round(x, 4) for x in report.losses[:5]],
        "last_losses": [round(x, 4) for x in report.losses[-5:]],
        "diverged_at_step": report.diverged_at_step,
        # Every step's loss. With no text repeated and every arm reading the
        # same windows under the same masks, this is a held-out loss and a
        # paired one, which is what the learning-rate arms are compared on.
        "losses": [round(x, 5) for x in report.losses],
        "muon_lr": config.muon_lr,
        "engine": config.engine,
        "warmup_steps": config.warmup_steps,
        "gpu": torch.cuda.get_device_name(0),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    (out_dir / "report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    karar_volume.commit()
    return out


def _build_model(backbone: str, revision: str, engine: str, attention: str, compiled: bool):
    """The model an engine trains, and the published config beside it."""
    import torch
    from transformers import AutoModelForCausalLM

    if engine == "native":
        from model.convert.native import from_pretrained

        model, config = from_pretrained(backbone, revision)
        if compiled:
            # In place, so the state_dict keeps its key names; torch.compile's
            # wrapper would prefix every key and the checkpoint would not load
            # back into an uncompiled model. The body is the part that is
            # called, so it is the part that is compiled.
            model.body.compile()
        return model, config
    model = AutoModelForCausalLM.from_pretrained(
        backbone, revision=revision, dtype=torch.float32, attn_implementation=attention
    )
    if compiled:
        # The inner model, because the loop calls `model.model(...)` and the
        # head separately. Compiling the wrapper set a compiled `__call__` that
        # nothing ever called, so the earlier finding that compilation changed
        # nothing measured nothing.
        model.model.compile()
    return model, model.config


# The continuation experiment (PLAN.md backbone item 4): continued causal
# pretraining on court decisions and question-and-answer text, with a fifth of
# the backbone's own web tier replayed so a loss on the general tasks can be
# told from the new text's effect.
CONTINUE_ROOT = VOL / "continue"
CONTINUE_NAME = "continue-v1"
CONTINUE_TOKENS = 300_000_000
CONTINUE_MIX = {"legal": 0.6, "questions": 0.2, "B": 0.2}
# A tier holds more than its share of the run, so nothing is read twice.
CONTINUE_TARGETS = {"legal": 200_000_000, "questions": 70_000_000}
# The causal objective scores every position, so the micro-batch the masked
# probe measured (48 rows, 48.5 GiB) is not known to fit. The continue-probe
# phase measures these under the causal loss, and the run takes the largest that
# peaked under four fifths of the card, at the speed measured for it.
CONTINUE_PROBE_BATCHES = (16, 32, 48)
CONTINUE_PROBE = Path("results/step4/continue/probe.json")
H100_GIB = 79.6


def continue_batch(probe_rows: list[dict], card_gib: float = H100_GIB) -> dict:
    """The probe row the continuation launches with: the largest micro-batch that fits."""
    fits = [r for r in probe_rows if "error" not in r and r["peak_gib"] < 0.8 * card_gib]
    if not fits:
        raise RuntimeError("no probed micro-batch fits the causal objective")
    return max(fits, key=lambda r: r["micro_batch"])


# The overlap check is the build's cost, so it runs in one process per core the
# container reserves, less one for the reader and the tokenizer. Each worker
# shares the parent's indexes by fork; about 5 GB resident each in the worst case
# where every page is copied, hence 96 GiB.
CONTINUE_WORKERS = 15


@app.function(volumes={VOL.as_posix(): karar_volume}, timeout=4 * HOUR, cpu=16.0,
              memory=98304)  # fmt: skip
def build_continuation(seed: int = 1) -> dict:
    """Both tiers of the continuation corpus, masked and cleaned of evaluation text.

    A tier that finished keeps its report beside its shards and is not rebuilt,
    so a restart after a timeout redoes only the tier that was cut off.
    """
    from transformers import AutoTokenizer

    from data import hub
    from data.continuation import (
        LEGAL_FILES,
        LEGAL_REPO,
        LEGAL_REVISION,
        evaluation_texts,
        legal_documents,
        legal_text,
        question_documents,
        question_text,
        write_tier,
    )
    from data.decontam.ngrams import MinHashLSH, NgramIndex, minhash_signature

    raw = CONTINUE_ROOT / "raw"
    started = time.perf_counter()
    legal_paths = []
    for name in LEGAL_FILES:
        target = raw / "legal" / name.replace("/", "__")
        if not target.is_file():
            url = f"https://huggingface.co/datasets/{LEGAL_REPO}/resolve/{LEGAL_REVISION}/{name}"
            hub.download(url, target)
        legal_paths.append(target)
    question_paths = [
        *hub.parquet_files("clips/mqa", "tr-faq-question", "train", raw / "mqa"),
        *hub.parquet_files("clips/mqa", "tr-cqa-question", "train", raw / "mqa"),
    ]
    karar_volume.commit()

    references = NgramIndex.load(VOL / "decontam" / "index.pkl")
    reference_lsh = MinHashLSH.load(VOL / "decontam" / "lsh.pkl")
    ours = list(evaluation_texts(VOL / "round1" / "data"))
    ours_index = NgramIndex.build(ours)
    ours_lsh = MinHashLSH()
    for key, text in ours:
        signature = minhash_signature(text)
        if signature is not None:
            ours_lsh.add(key, signature)
    checks = [("reference", references, reference_lsh), ("ours", ours_index, ours_lsh)]

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, revision=REVISION)

    def encode_batch(texts):
        return tokenizer(texts, add_special_tokens=False).input_ids

    report = {"evaluation_texts": len(ours), "seed": seed}
    for tier, documents, prepare in (
        ("legal", legal_documents(legal_paths, seed), legal_text),
        ("questions", question_documents(question_paths, seed), question_text),
    ):
        out = CONTINUE_ROOT / "tokens" / tier
        done = out / "tier.json"
        if done.is_file():
            report[tier] = json.loads(done.read_text(encoding="utf-8"))
            continue
        for old in out.glob("*.bin"):
            old.unlink()
        tier_started = time.perf_counter()
        report[tier] = write_tier(documents, encode_batch, int(tokenizer.eos_token_id), out,
                                  CONTINUE_TARGETS[tier], checks, prepare=prepare,
                                  workers=CONTINUE_WORKERS)  # fmt: skip
        report[tier]["seconds"] = round(time.perf_counter() - tier_started, 1)
        done.write_text(json.dumps(report[tier], indent=2), encoding="utf-8")
        karar_volume.commit()
    report["seconds"] = round(time.perf_counter() - started, 1)
    (CONTINUE_ROOT / "corpus.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    karar_volume.commit()
    return report


@app.function(volumes={VOL.as_posix(): karar_volume}, timeout=HOUR, cpu=4.0, memory=16384)
def export_checkpoint(name: str, filename: str) -> dict:
    """Turn a training checkpoint into a model folder the instrument can load.

    A checkpoint holds the optimizer as well as the weights and is three times
    the size of the model; what an evaluation needs is the weights in the
    library's own layout, with the tokenizer beside them and a note saying
    which run and which token count they came from.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = CONVERT_ROOT / name / filename
    if not source.is_file():
        raise FileNotFoundError(f"{source} is not on the volume")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    model = AutoModelForCausalLM.from_pretrained(BACKBONE, revision=REVISION, dtype=torch.float32)
    from model.convert.native import load_checkpoint_into_library

    load_checkpoint_into_library(payload["model"], model)
    out = CONVERT_ROOT / name / "export" / source.stem
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(BACKBONE, revision=REVISION).save_pretrained(out)
    note = {
        "backbone": BACKBONE,
        "revision": REVISION,
        "run": name,
        "checkpoint": filename,
        "fingerprint": payload.get("fingerprint"),
        "progress": payload.get("progress"),
        "mask_token": MASK_TOKEN,
    }
    (out / "conversion.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    karar_volume.commit()
    return {"path": out.as_posix(), **note}


def _gate(
    settings: dict,
    tier_paths: dict,
    mixture: dict,
    tokenizer,
    config,
    free_bytes: int,
    mask_token: str,
    model=None,
    decay_mixture: dict | None = None,
):
    """Every check, run before a step is taken. Shared by the phase and the run."""
    from model.convert import preflight as pf
    from model.convert.stream import ShardedWindows

    context = settings.get("context", 1024)
    budget = settings.get("total_tokens", 5_000_000_000)
    checks = [
        pf.check_mask_token(tokenizer, mask_token, config.vocab_size),
        pf.check_separator(tokenizer),
        pf.check_vocab_matches(config, tokenizer),
        pf.check_model_fits_context(config, context),
        pf.check_attention_is_uniform(config),
        pf.describe_plan(budget, settings.get("batch_tokens", 524_288), context, mixture),
    ]
    if model is not None:
        checks.append(pf.check_attention_implementation(model))
    for tier, folder in sorted(tier_paths.items()):
        files = sorted(Path(folder).glob("*.bin"))
        shard_check = pf.check_shards(tier, files, context)
        checks.append(shard_check)
        if not shard_check.ok:
            continue
        windows = ShardedWindows(files, context)
        checks.append(pf.sample_windows_are_plausible(windows))
        checks.append(pf.check_separator_in_corpus(windows, tokenizer.eos_token_id, tier))
        # What the tier is asked for over the whole schedule: its stable share
        # of the stable tokens plus its decay share of the decay tokens. The
        # curated tier's share more than doubles in the decay, so checking the
        # stable share alone passed a run that would have read it twice.
        tail = settings.get("decay_frac", 0.20)
        late = (decay_mixture or mixture).get(tier, 0.0)
        wanted = budget * ((1.0 - tail) * mixture.get(tier, 0.0) + tail * late)
        checks.append(pf.check_tokens_cover_budget(len(windows) * context, int(wanted), tier))
    if free_bytes:
        parameters = (
            sum(p.numel() for p in model.parameters())
            if model is not None
            else estimate_parameters(config)
        )
        checks.append(
            pf.check_batch_fits(
                free_bytes,
                parameters,
                micro_batch=settings.get("micro_batch", 8),
                context=context,
                layers=config.num_hidden_layers,
                hidden=config.hidden_size,
                vocab_size=config.vocab_size,
                # The causal objective scores every position, not the masked
                # thirty percent, so the vocabulary projection is 3.3 times larger.
                masked_share=1.0 if settings.get("objective") == "clm" else 0.3,
            )
        )
    return checks


def estimate_parameters(config) -> int:
    """Parameter count from the config alone, for checking memory before loading.

    Deliberately an overestimate: the attention projections are counted as four
    square matrices when grouped-query attention makes two of them narrower.
    Overstating the model makes the memory check stricter, which is the safe
    direction. Where the model is already loaded its real count is used instead.
    """
    hidden, layers, vocab = config.hidden_size, config.num_hidden_layers, config.vocab_size
    per_layer = 4 * hidden * hidden + 3 * hidden * config.intermediate_size
    return vocab * hidden + layers * per_layer


@app.function(
    volumes={VOL.as_posix(): karar_volume, DATA.as_posix(): data_volume},
    timeout=HOUR,
    cpu=2.0,
)
def preflight_only(settings: dict) -> list[dict]:
    """Every assumption checked against the real artefacts, on CPU, for cents."""
    from dataclasses import asdict

    from transformers import AutoConfig, AutoTokenizer

    from model.convert.preflight import report

    tier_paths = settings.pop("tiers")
    mixture = settings.pop("mixture")
    late = settings.pop("decay_mixture", None)
    mask_token = settings.pop("mask_token", MASK_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(settings["backbone"], revision=settings["revision"])
    config = AutoConfig.from_pretrained(settings["backbone"], revision=settings["revision"])
    checks = _gate(
        settings, tier_paths, mixture, tokenizer, config, 0, mask_token, decay_mixture=late
    )
    print(report(checks))
    return [asdict(c) for c in checks]


# Measured on the instrument's own runs, per second, from modal.com/pricing
# read 2026-09-20. Used to turn a measured speed into a cost, never to guess one.
# Per second, read from modal.com/pricing on 2026-09-21. The same table as the
# instrument's, so a card priced there cannot be a KeyError here.
GPU_PRICES = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
    "H200": 0.001261,
    "B200": 0.001736,
}

BACKBONE = "ufakai/ufakzeka-1-base"
# Pinned to the commit the instrument measured, so the converted model and its
# causal baseline start from the same weights.
REVISION = "f9e11eea28cbb2ba953a5628f972d416fe0c3cfe"
MASK_TOKEN = "<|reserved_0|>"


@app.function(
    gpu=GPU,
    volumes={VOL.as_posix(): karar_volume, DATA.as_posix(): data_volume},
    # Twenty minutes, as the ledger row says. A probe that runs longer is wrong.
    timeout=1200,
    max_containers=1,
)
def probe(settings: dict, options: list[dict], steps: int = 12) -> list[dict]:
    """Measure tokens per second for several configurations, on the real model.

    The smoke run measured one configuration and it was the wrong one: 8.4
    percent of the card against the 20.4 percent the backbone's own pretraining
    achieved. Rather than reason about which of micro-batch, compilation and
    the attention kernel matters, each is varied and timed.
    """

    import torch
    from transformers import AutoTokenizer

    from model.convert.stream import MixedStream, ShardedWindows
    from model.convert.train import ConvertConfig

    tier_paths = settings.pop("tiers")
    mixture = settings.pop("mixture")
    settings.pop("decay_mixture", None)
    settings.pop("name", None)
    mask_token = settings.pop("mask_token")
    tokenizer = AutoTokenizer.from_pretrained(settings["backbone"], revision=settings["revision"])

    import statistics
    import tempfile

    from model.convert.train import train

    results = []
    for option in options:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model, model_config = _build_model(
            settings["backbone"],
            settings["revision"],
            option.get("engine", "library"),
            option["attn_implementation"],
            bool(option.get("compile")),
        )

        loop = {k: v for k, v in {**settings, **option}.items() if k not in NOT_CONFIG}
        config = ConvertConfig(
            mask_id=int(tokenizer.convert_tokens_to_ids(mask_token)),
            separator_id=int(tokenizer.eos_token_id),
            pad_id=int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1),
            vocab_size=int(model_config.vocab_size),
            # The run's own batch, eleven passes a step. The first pass of this
            # probe used two, which charges the optimizer to a fifth of the
            # tokens it is really spread over, and Muon's step is the heavier
            # one, so it understated the native engine more than the library.
            **loop,
        )
        sources = {
            tier: ShardedWindows(sorted(Path(f).glob("*.bin")), config.context)
            for tier, f in tier_paths.items()
        }
        stream = MixedStream(sources, mixture, seed=1)
        try:
            # The real loop, not a copy of it. An earlier probe had its own
            # forward and backward, and so could not see what the run's
            # accumulation, clipping and synchronisation cost.
            with tempfile.TemporaryDirectory() as scratch:
                report = train(
                    model, config, stream, out_dir=Path(scratch), device="cuda", max_steps=steps
                )
            settled = report.step_seconds[steps // 3 :]
            per_step = statistics.median(settled)
            results.append(
                {
                    **option,
                    "tokens_per_second": round(config.batch_tokens / per_step),
                    "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    "final_loss": report.losses[-1] if report.losses else None,
                }
            )
        except Exception as error:  # an out-of-memory here is a result, not a crash
            results.append({**option, "error": str(error)[:200]})
        del model
    return results


def tier_paths(include_a: bool = True) -> dict:
    """Where each tier's tokens actually live.

    B and C are read in place from the backbone's own corpus volume, mounted
    read only: they contain no excluded source, so copying them to our volume
    would duplicate nineteen gigabytes to change nothing. Only tier A is ours,
    because only tier A had to be rebuilt without the ShareAlike source.
    """
    paths = {
        "B": (DATA / "stage3/train/B").as_posix(),
        "C": (DATA / "stage3/train/C").as_posix(),
    }
    if include_a:
        paths["A"] = (TOKENS_ROOT / "A").as_posix()
    return paths


# The configuration a paid run launches with, in one place, and the first thing
# the probe measures. The probe once measured one configuration while the run
# defaulted to another, so the cost estimate described a run nobody would launch.
# The document mask stays on: dropping it was worth 11 percent of throughput and
# is the one lever with a quality price, and quality is what the run is for.
# Keys that choose how the model is built rather than how the loop runs.
NOT_CONFIG = {"attn_implementation", "compile", "gradient_checkpointing"}

# The native engine, compiled, is what results/step2/probe.json measured at
# 227,454 tok/s on H100 at the run's own batch: the backbone's own network with
# the optimizers it was pretrained with, 2.4 percent behind the library path and
# the more faithful of the two.
RUN_SETTINGS = {
    "engine": "native",
    "micro_batch": 48,
    "attn_implementation": "sdpa",
    "compile": True,
    "mask_documents": True,
    "fused": True,
}


# The ladder. The trunk's schedule is longer than any rung so it never anneals
# by itself; the literature's budget is about half the base model's, 7B for us,
# and one pass over the corpus is the ceiling.
TRUNK = "trunk-v1"
# The Muon rates a trunk may be launched at. 0.01 is what the backbone's
# own continued pretraining used; nothing above it is allowed, because 0.02 was
# its from-scratch peak. The embedding and norm rates are not varied.
MUON_RATES = (0.01, 0.005, 0.0025)
# A short re-warm-up, as continued pretraining uses: 100 steps is 52M tokens,
# five percent of the first rung. The 500 the loop defaults to is 262M.
WARMUP_STEPS = 100


# What the second probe measured for the launch configuration on H100, used
# only to cap a launch: 1.3 times the expected time plus a quarter of an hour for
# start-up and compilation. A run that reaches its cap is wrong, and the cap is
# what stops it spending; the function's own 24 hours would not.
MEASURED_TOKENS_PER_SECOND = 227_454


def capped(tokens_to_train: int, tokens_per_second: float = MEASURED_TOKENS_PER_SECOND):
    seconds = int(1.3 * tokens_to_train / tokens_per_second) + 900
    # Two retries, not the function's ten. A timed-out call is a failure, and
    # failures are retried (modal/_functions.py: "User errors including
    # timeouts are managed by the user specified retry policy"). For a slow run
    # that is the right thing, it resumes from its checkpoint; for a stuck one
    # it would pay for the cap ten times over, so the cap alone capped nothing.
    return convert.with_options(
        timeout=seconds, retries=modal.Retries(max_retries=2, initial_delay=0.0)
    )


def trunk_name(lr: float) -> str:
    return f"{TRUNK}-muon{lr:g}"


def _finish(report: dict, label: str, folder: str = "results/step2/runs") -> None:
    """Print a run's report, keep it under results/, and say what it cost."""
    print(json.dumps({k: v for k, v in report.items() if k != "losses"}, indent=2))
    if not report.get("preflight_failed"):
        out = Path(__file__).resolve().parents[2] / folder / f"{label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    _cost(report)


# The curated tier sets the top rung. It holds 2.14B tokens, a rung of R tokens
# reads 0.25 R of it at the stable mixture and its decay branch another
# 0.60 x 0.25 R, so one pass runs out at R = 5.35B. Going past 5B means reading
# the curated tier more than once, which is a decision and not a default.
TRUNK_SCHEDULE_TOKENS = 8_000_000_000
RUNGS = (1_000_000_000, 2_500_000_000, 5_000_000_000)
DECAY_FRAC = 0.20


def _settings(name: str, tiers: dict, mixture: dict, **overrides) -> dict:
    base = {
        **RUN_SETTINGS,
        "name": name,
        "tiers": tiers,
        "mixture": mixture,
        "backbone": BACKBONE,
        "revision": REVISION,
        "mask_token": MASK_TOKEN,
    }
    return {**base, **overrides}


@app.local_entrypoint()
def main(
    phase: str = "smoke",
    tokens: int = 0,
    shards: int = 0,
    name: str = "",
    file: str = "",
    lr: float = 0.0,
) -> None:
    """rebuild | smoke | pilot. Nothing costly runs without a printed estimate."""
    if phase == "preflight":
        settings = _settings(
            "preflight",
            tier_paths(),
            MIX_STABLE,
            decay_mixture=MIX_DECAY,
            total_tokens=tokens or 5_000_000_000,
        )
        preflight_only.remote(settings)
        return

    if phase == "probe":
        # The three levers the backbone's own training used and the smoke run
        # did not: a bigger micro-batch, compilation, and a compiled block mask
        # instead of a materialised one.
        # One factor at a time. The first probe varied batch size and kernel
        # together and so could not say which caused the out-of-memory. These
        # hold the batch at 8 until the kernel question is settled.
        # The kernel is settled: it changes little. Memory caps the batch at 8
        # rows on a 22 GB card, so the remaining levers are recomputation and a
        # bigger card. This runs on whatever KARAR_GPU names.
        # 64 rows with recomputation used 17.6 GB of 80, so the untested middle
        # is storing activations at a batch that still fits: it skips the third
        # of compute recomputation costs. Compile is retried here because it
        # was measured on an L4, where the kernels are the bottleneck and the
        # Python overhead it removes is not.
        # Four levers left that cost nothing but a probe: a bigger batch in the
        # 32 GB still free, dropping document boundaries (worth 15 percent on
        # L4), a fused optimizer step, and both together.
        # The winning configuration only, so a second card can be compared on
        # tokens per dollar without paying for five runs to find that out.
        # The run's own configuration first, then one change at a time from it:
        # compilation, which has never been measured while actually active, and
        # a smaller batch in case 48 rows does not fit beside the document mask.
        # Second pass, at the run's own batch. The first found compilation
        # worth 2.3x on the library path now that it is really on, and the
        # library's dense mask faster than the native block mask at this
        # context, so the native network is also tried with the dense mask.
        native = {**RUN_SETTINGS, "engine": "native", "compile": True}
        options = [
            native,
            {**native, "dense_mask": True},
            # The library path, named explicitly: RUN_SETTINGS now selects the
            # native engine, so inheriting it would time the native one twice.
            {**RUN_SETTINGS, "engine": "library", "compile": True},
        ]
        settings = _settings(
            "probe",
            {"B": tier_paths(include_a=False)["B"]},
            {"B": 1.0},
            total_tokens=20_000_000,
        )
        results = probe.remote(settings, options)
        # Committed, so every cost quoted from a probe traces to a file (rule 3).
        # Kept per card, and appended to, so a repeat measurement sits beside
        # the first rather than replacing it.
        record = Path(__file__).resolve().parents[2] / "results/step2/probe.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        history = json.loads(record.read_text(encoding="utf-8")) if record.is_file() else []
        history.append(
            {
                "gpu": GPU,
                "price_per_second": GPU_PRICES.get(GPU),
                "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "steps": 12,
                "results": results,
            }
        )
        record.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print()
        for row in results:
            if "error" in row:
                print(f"  {row['micro_batch']:>3} rows compile={row['compile']:<5} "
                      f"{row['attn_implementation']:<14} FAILED {row['error'][:60]}")  # fmt: skip
                continue
            rate = row["tokens_per_second"]
            hours = 5e9 / rate / 3600
            docs = ("no-docs" if not row.get("mask_documents", True) else "docmask") + (
                "+fused" if row.get("fused") else ""
            )
            print(f"  {row['micro_batch']:>3} rows compile={row['compile']:<5} "
                  f"{row['attn_implementation']:<14} {docs:<8} {rate:>8,} tok/s  "
                  f"{row['peak_gib']:>5.1f} GiB  5B in {hours:>5.1f} h "
                  f"= ${5e9/rate*GPU_PRICES[GPU]:.2f} on {GPU}")  # fmt: skip
        return

    if phase == "rebuild":
        report = rebuild_tier_a.remote(BACKBONE, REVISION, shards)
        print(json.dumps(report, indent=2))
        return

    if phase == "smoke":
        # Tier B only, so this needs no rebuild and proves the expensive things
        # a CPU test cannot: the real weights load, the mask token resolves,
        # the volume paths are right, the batch fits in memory, and how fast it
        # actually goes.
        settings = _settings(
            "smoke",
            {"B": tier_paths(include_a=False)["B"]},
            {"B": 1.0},
            total_tokens=tokens or 20_000_000,
            checkpoint_every=25,
            log_every=10,
            warmup_steps=5,
        )
        report = convert.remote(settings)
        print(json.dumps(report, indent=2))
        _cost(report)
        return

    if phase == "trunk":
        # The rung ladder. One run at the stable rate with a schedule long
        # enough never to reach its own decay, stopped at `tokens`, keeping a
        # rung at each budget worth evaluating. Launching again with a larger
        # `tokens` continues it from its last checkpoint.
        if not tokens:
            raise SystemExit("--tokens says where this launch stops, for example 1000000000")
        if lr not in MUON_RATES:
            raise SystemExit(f"--lr must be one of {MUON_RATES}, the allowed rates")
        settings = _settings(
            trunk_name(lr),
            tier_paths(),
            MIX_STABLE,
            total_tokens=TRUNK_SCHEDULE_TOKENS,
            decay_frac=0.0,
            warmup_steps=WARMUP_STEPS,
            muon_lr=lr,
            rungs=list(RUNGS),
            stop_tokens=tokens,
            # About ten minutes of work between checkpoints, so that is the
            # most a preemption or a cap can cost. Not part of the fingerprint.
            checkpoint_every=250,
        )
        # Capped for the whole distance from zero, which over-allows a
        # continuation but never under-allows a resume after a preemption.
        handle = capped(tokens).spawn(settings)
        print(f"{trunk_name(lr)} to {tokens:,} tokens spawned as {handle.object_id}")
        _finish(handle.get(), f"{trunk_name(lr)}-to-{tokens}")
        return

    if phase == "branch":
        # Anneal from one rung. The branch is an ordinary run whose schedule
        # ends a quarter of the rung later, so its decay begins at the rung.
        from model.convert.schedule import decay_branch_steps, steps_to_reach

        if tokens not in RUNGS:
            raise SystemExit(f"--tokens must be one of the rungs {RUNGS}")
        if lr not in MUON_RATES:
            raise SystemExit(f"--lr names the trunk to branch from, one of {MUON_RATES}")
        batch_tokens = 524_288
        rung_step = steps_to_reach(tokens, batch_tokens)
        total_steps = decay_branch_steps(rung_step, DECAY_FRAC)
        kept = rung_step * batch_tokens
        settings = _settings(
            f"{trunk_name(lr)}-decay-{tokens}",
            tier_paths(),
            MIX_STABLE,
            decay_mixture=MIX_DECAY,
            total_tokens=total_steps * batch_tokens,
            decay_frac=DECAY_FRAC,
            warmup_steps=WARMUP_STEPS,
            muon_lr=lr,
            branch_from=(CONVERT_ROOT / trunk_name(lr) / f"rung_{kept}.pt").as_posix(),
        )
        handle = capped((total_steps - rung_step) * batch_tokens).spawn(settings)
        print(f"decay branch from {kept:,} tokens spawned as {handle.object_id}")
        _finish(handle.get(), f"{trunk_name(lr)}-decay-{tokens}")
        return

    if phase == "continue-build":
        report = build_continuation.remote()
        out = Path(__file__).resolve().parents[2] / "results/step4/continue/corpus.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    if phase == "continue-probe":
        settings = _settings(
            "continue-probe",
            {"B": tier_paths(include_a=False)["B"]},
            {"B": 1.0},
            objective="clm",
            total_tokens=20_000_000,
            muon_lr=0.01,
        )
        options = [{**RUN_SETTINGS, "micro_batch": m} for m in CONTINUE_PROBE_BATCHES]
        results = probe.remote(settings, options)
        record = Path(__file__).resolve().parents[2] / CONTINUE_PROBE
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({
            "gpu": GPU,
            "objective": "clm",
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "steps": 12,
            "results": results,
        }, indent=2) + "\n", encoding="utf-8")  # fmt: skip
        print(json.dumps(results, indent=2))
        return

    if phase == "continue":
        from model.convert.schedule import steps_for_tokens

        # Nothing launches on a guess: the micro-batch and the speed the cap is
        # computed from are the causal probe's own.
        measured = json.loads(
            (Path(__file__).resolve().parents[2] / CONTINUE_PROBE).read_text(encoding="utf-8")
        )
        chosen = continue_batch(measured["results"])
        print(f"micro-batch {chosen['micro_batch']}: {chosen['peak_gib']} GiB, "
              f"{chosen['tokens_per_second']:,} tok/s")  # fmt: skip

        tiers = {
            "legal": (CONTINUE_ROOT / "tokens" / "legal").as_posix(),
            "questions": (CONTINUE_ROOT / "tokens" / "questions").as_posix(),
            "B": tier_paths(include_a=False)["B"],
        }
        settings = _settings(
            CONTINUE_NAME,
            tiers,
            CONTINUE_MIX,
            objective="clm",
            total_tokens=CONTINUE_TOKENS,
            decay_frac=DECAY_FRAC,
            warmup_steps=30,
            muon_lr=0.01,
            checkpoint_every=200,
            micro_batch=chosen["micro_batch"],
        )
        handle = capped(CONTINUE_TOKENS, chosen["tokens_per_second"]).spawn(settings)
        print(f"{CONTINUE_NAME} spawned as {handle.object_id}")
        report = handle.get()
        _finish(report, CONTINUE_NAME, "results/step4/continue")
        if report.get("preflight_failed") or report.get("diverged_at_step") is not None:
            return
        final = steps_for_tokens(CONTINUE_TOKENS, batch_tokens=524_288)
        print(json.dumps(export_checkpoint.remote(CONTINUE_NAME, f"step_{final}.pt"), indent=2))
        return

    if phase == "export":
        # --name is the run folder, --file the checkpoint inside it.
        print(json.dumps(export_checkpoint.remote(name, file), indent=2))
        return

    if phase == "pilot":
        settings = _settings(
            "pilot-5b",
            tier_paths(),
            MIX_STABLE,
            decay_mixture=MIX_DECAY,
            total_tokens=tokens or 5_000_000_000,
        )
        # Spawned, not called, so the run does not depend on this client
        # staying connected; a local network drop has killed three clients in
        # this project while their containers carried on.
        handle = convert.spawn(settings)
        print(f"pilot spawned as {handle.object_id}; it resumes itself if preempted")
        report = handle.get()
        print(json.dumps(report, indent=2))
        _cost(report)
        return

    raise SystemExit(f"unknown phase {phase!r}")


def _cost(report: dict) -> None:
    """What this run cost, and what the 5B run would cost at the same speed."""
    if report.get("preflight_failed"):
        raise SystemExit("preflight failed, nothing was trained:\n" + report["preflight_failed"])
    gpu = GPU
    price = GPU_PRICES.get(gpu)
    if not price or not report.get("wall_seconds"):
        return
    spent = report["wall_seconds"] * price
    rate = report.get("tokens_per_second") or 0
    print(f"  {report['wall_seconds']:.0f} s on {gpu}, about ${spent:.2f}")
    if rate:
        hours = 5_000_000_000 / rate / 3600
        print(f"  at {rate:,.0f} tok/s, 5B tokens is {hours:.1f} h and about "
              f"${5_000_000_000 / rate * price:.2f}")  # fmt: skip
