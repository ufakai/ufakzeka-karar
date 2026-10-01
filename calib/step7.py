"""Step 7's decisions from the shipped files' own outputs, fixed before the outputs exist.

    python -m calib.step7 --outputs .cache/karar/int8/<run>/outputs.jsonl \
        --run <run> --weights models/<run>/model.pt

Rows: every validation and held-out row through fp32 and each int8 file
(calib/modal_int8.py int8_outputs). Validation rows are split 80/20 by a hash of
the text into fit and selection (calib/abstain_preflight.py), with every
HakemBench-dev item and dev text removed from both.

For each engine:
1. Calibration form (calib/fit.py): each form fitted on fit rows; the form with
   the lowest soft cross-entropy on selection rows is chosen, and a simpler form
   within 0.001 of it is preferred.
2. Scores on the held-out cells and on dev (the owner's answers), per question
   type and overall, before and after calibration: accuracy, mean-over-tasks
   macro F1, Brier and soft cross-entropy against the target, smooth ECE of the
   top answer against the target's top answer.
3. The gate of PLAN.md against fp32, both calibrated, on selection plus
   held-out rows: macro F1 and smooth ECE within 0.5 points. The gate beside
   it: how far the calibrated probabilities move from fp32's, as mean, 99th
   percentile and maximum absolute change, and how many answers change.

The shipped file: the first of int8-matmul, int8-matmul-per-channel and
int8-dynamic that passes the gate, in that order (heads and embeddings in fp32
first); fp32 if none passes.

The abstain map is refitted on the shipped file: isotonic, expected
soft error given the raw maximum probability, fitted on selection, judged on
the held-out cells (AUGRC, coverage at 5 and 1 percent risk).

Writes results/step7/decision.json and results/step7/calibrator.json, the
temperatures and the abstain map the server applies.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from calib import fit as calfit
from calib.abstain_preflight import split
from calib.selective import at_threshold, augrc, threshold_at_risk

ENGINES = ("fp32", "int8-matmul", "int8-matmul-per-channel", "int8-dynamic", "int8-no-down")
# The down projections kept in fp32 was chosen on fit rows after the
# first three failed the gate; it is judged by the same gate, first in line.
PREFERENCE = ("int8-no-down", "int8-matmul", "int8-matmul-per-channel", "int8-dynamic")
DEV = Path("results/private/hakembench/dev_v0.jsonl")
OUT = Path("results/step7")
GATE_POINTS = 0.5
SIMPLER_BY = 0.001
# Tasks whose validation rows chose the checkpoint (the guardrail dev set): scored like
# HakemBench-dev, never fitted on, so selection and calibration do not share rows.
SELECTION_TASKS = {"prompt_injection-conv"}


def load(path: Path, texts: dict[str, str], dev_ids: set[str], dev_texts: set[str]) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        text = texts.get(r["row_id"], "")
        if r["split"] == "heldout_task":
            part = "heldout"
        elif r["row_id"] in dev_ids or text in dev_texts or r.get("task") in SELECTION_TASKS:
            part = "dev_only"
        else:
            part = split(text)
        rows.append({**r, "part": part, "n_options": len(r["keys"]),
                     "target_array": [float(r["target"][k]) for k in r["keys"]]})  # fmt: skip
    return rows


def view(rows: list[dict], engine: str) -> list[dict]:
    """The rows as calib/fit.py reads them, for one engine."""
    return [{"logprobs": r[engine]["logprobs"], "target": r["target_array"], "type": r["type"],
             "n_options": r["n_options"], "row_id": r["row_id"], "task": r["task"],
             "part": r["part"]} for r in rows]  # fmt: skip


def choose_form(rows: list[dict]) -> tuple[dict, dict]:
    fitting = [r for r in rows if r["part"] == "fit"]
    selection = [r for r in rows if r["part"] == "selection"]
    fitted = {form: calfit.fit(fitting, form) for form in calfit.FORMS}
    losses = {}
    for form, calibrator in fitted.items():
        losses[form] = float(np.mean([
            -np.dot(r["target"], np.log(np.clip(calfit.apply(calibrator, r), 1e-12, None)))
            for r in selection]))  # fmt: skip
    best = min(losses, key=losses.get)
    for form in calfit.FORMS:  # simplest first
        if losses[form] <= losses[best] + SIMPLER_BY:
            best = form
            break
    return fitted[best], losses


def metrics(rows: list[dict], probs: list[np.ndarray], answers: dict[str, str] | None = None,
            keys: dict[str, list[str]] | None = None) -> dict:  # fmt: skip
    """Scores over `rows`; with `answers`, against the owner's answer instead of the target."""
    import relplot

    gold, guess, conf, brier, ce, tasks = [], [], [], [], [], []
    for r, p in zip(rows, probs, strict=True):
        target = np.asarray(r["target"], dtype=float)
        if answers is not None:
            target = np.array([float(k == answers[r["row_id"]]) for k in keys[r["row_id"]]])
        g, q = int(np.argmax(target)), int(np.argmax(p))
        gold.append(g), guess.append(q), conf.append(float(p[q])), tasks.append(r["task"])
        brier.append(float(((p - target) ** 2).sum()))
        ce.append(float(-np.dot(target, np.log(np.clip(p, 1e-12, None)))))
    correct = np.array([g == q for g, q in zip(gold, guess, strict=True)], dtype=float)
    per_task = defaultdict(list)
    for t, g, q in zip(tasks, gold, guess, strict=True):
        per_task[t].append((g, q))
    f1s = []
    for pairs in per_task.values():
        labels = {x for pair in pairs for x in pair}
        f1s.append(np.mean([2 * sum(g == q == k for g, q in pairs)
                            / max(1, sum(g == k for g, _ in pairs) + sum(q == k for _, q in pairs))
                            for k in labels]))  # fmt: skip
    return {"rows": len(rows), "accuracy": float(correct.mean()),
            "macro_f1_mean_over_tasks": float(np.mean(f1s)), "brier": float(np.mean(brier)),
            "soft_ce": float(np.mean(ce)),
            "smooth_ece": float(relplot.smECE(np.array(conf), correct))}  # fmt: skip


def scored(rows, calibrator, answers=None, keys=None) -> dict:
    out = {}
    for label, subset in (
        ("all", rows),
        *((k, [r for r in rows if r["type"] == k]) for k in sorted({r["type"] for r in rows})),
    ):
        if not subset:
            continue
        raw = [np.exp(np.asarray(r["logprobs"])) / np.exp(np.asarray(r["logprobs"])).sum()
               for r in subset]  # fmt: skip
        cal = [calfit.apply(calibrator, r) for r in subset]
        out[label] = {"raw": metrics(subset, raw, answers, keys),
                      "calibrated": metrics(subset, cal, answers, keys)}  # fmt: skip
    return out


def gate(engine_rows, fp32_rows, engine_cal, fp32_cal) -> dict:
    judged = [i for i, r in enumerate(engine_rows) if r["part"] in ("selection", "heldout")]
    a = [calfit.apply(engine_cal, engine_rows[i]) for i in judged]
    b = [calfit.apply(fp32_cal, fp32_rows[i]) for i in judged]
    ma = metrics([engine_rows[i] for i in judged], a)
    mb = metrics([fp32_rows[i] for i in judged], b)
    change = np.concatenate([np.abs(x - y) for x, y in zip(a, b, strict=True)])
    f1_lost = 100 * (mb["macro_f1_mean_over_tasks"] - ma["macro_f1_mean_over_tasks"])
    ece_added = 100 * (ma["smooth_ece"] - mb["smooth_ece"])
    return {"rows": len(judged), "macro_f1_points_lost": f1_lost,
            "smooth_ece_points_added": ece_added,
            "passes": f1_lost <= GATE_POINTS and ece_added <= GATE_POINTS,
            "calibrated_probability_change": {
                "mean": float(change.mean()), "p99": float(np.percentile(change, 99)),
                "max": float(change.max())},
            "answers_changed": int(sum(np.argmax(x) != np.argmax(y)
                                       for x, y in zip(a, b, strict=True)))}  # fmt: skip


def abstain_map(rows: list[dict]) -> dict:
    from sklearn.isotonic import IsotonicRegression

    def arrays(part):
        subset = [r for r in rows if r["part"] == part]
        conf = np.array([float(np.exp(max(r["logprobs"]))) for r in subset])
        loss = np.array([1.0 - r["target"][int(np.argmax(r["logprobs"]))] for r in subset])
        return conf, loss

    sel_conf, sel_loss = arrays("selection")
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=False)
    iso.fit(sel_conf, sel_loss)
    conf, loss = arrays("heldout")
    at_risk = {}
    for r in (0.05, 0.01):
        t = threshold_at_risk(sel_conf, sel_loss, r)
        at_risk[f"heldout_at_risk_{r}"] = at_threshold(conf, loss, t)
    x, y = iso.X_thresholds_.round(6).tolist(), iso.y_thresholds_.round(6).tolist()
    return {"map": {"x": x, "y": y}, "heldout_augrc": augrc(conf, loss), **at_risk}


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="calib.step7")
    parser.add_argument("--outputs", type=Path, required=True)
    # Further engines' outputs over the same rows, merged by row id.
    parser.add_argument("--extra", type=Path, action="append", default=[])
    # The checkpoint the outputs came from: the calibrator names it, and the adapter
    # refuses to apply it to any other weights.
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    texts = {}
    for path in [*sorted(Path("data/built").glob("**/validation.jsonl")),
                 Path("data/built/sss/heldout_task.jsonl")]:  # fmt: skip
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                s = row["state"]
                texts[row["row_id"]] = s if isinstance(s, str) else json.dumps(s)
    dev = [json.loads(x) for x in DEV.read_text(encoding="utf-8").splitlines() if x.strip()]
    dev_ids, dev_texts = {r["id"] for r in dev}, {r["text"] for r in dev}
    answers = {r["id"]: max(r["target"], key=r["target"].get) for r in dev}
    rows = load(args.outputs, texts, dev_ids, dev_texts)
    for extra in args.extra:
        more = {}
        for line in extra.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                more[r["row_id"]] = {
                    k: v for k, v in r.items() if isinstance(v, dict) and "logprobs" in v
                }
        if set(more) != {r["row_id"] for r in rows}:
            raise RuntimeError(f"{extra} does not cover the same rows")
        for r in rows:
            r.update(more[r["row_id"]])
    keys = {r["row_id"]: r["keys"] for r in rows}
    result: dict = {"rows": {p: sum(r["part"] == p for r in rows)
                             for p in ("fit", "selection", "heldout", "dev_only")}}  # fmt: skip
    calibrators, views = {}, {}
    # Once fp32 ships, a new checkpoint's outputs hold fp32 alone.
    present = [e for e in ENGINES if all(e in r for r in rows)]
    preference = [e for e in PREFERENCE if e in present]
    for engine in present:
        views[engine] = view(rows, engine)
        calibrators[engine], losses = choose_form(views[engine])
        dev_rows = [r for r in views[engine] if r["row_id"] in dev_ids]
        result[engine] = {
            "calibrator": calibrators[engine], "selection_soft_ce": losses,
            "heldout": scored([r for r in views[engine] if r["part"] == "heldout"],
                              calibrators[engine]),
            "dev_owner": scored(dev_rows, calibrators[engine], answers, keys),
        }  # fmt: skip
    for engine in preference:
        result[engine]["gate"] = gate(views[engine], views["fp32"], calibrators[engine],
                                      calibrators["fp32"])  # fmt: skip
    shipped = next((e for e in preference if result[e]["gate"]["passes"]), "fp32")
    result["shipped"] = shipped
    result["abstain"] = abstain_map(views[shipped])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "decision.json").write_text(json.dumps(result, indent=2))
    (OUT / "calibrator.json").write_text(json.dumps(
        {"engine": shipped, "weights": args.run, "weights_sha256": file_sha256(args.weights),
         "temperatures": calibrators[shipped], "abstain_map": result["abstain"]["map"]},
        indent=2))  # fmt: skip
    print(json.dumps({"shipped": shipped, "gates": {e: result[e]["gate"] for e in preference},
                      "calibrator": calibrators[shipped]}, indent=2))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
