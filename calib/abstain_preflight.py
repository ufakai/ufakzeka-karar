"""The free preflight for step 4's abstain output: how well the logits alone rank errors.

    python -m calib.abstain_preflight --run r3-base-s2

From the checkpoint's saved predictions only, no new pass:
- validation rows split 80/20 by a hash of the text into fit and selection,
  with every HakemBench-dev item and dev text removed from both;
- the held-out cells as the evaluation set;
- the loss is the soft error, 1 minus the target's mass on the model's answer;
  a 0/1 error on rows whose target puts at least 0.6 on one answer is the
  secondary;
- every baseline confidence (calib/selective.py) and a logistic regression on
  the three distribution statistics (top probability, top-two gap, normalised
  entropy) fitted on the fit split;
- AUGRC on the held-out cells with a paired bootstrap against maximum
  probability, the pre-registered comparator, Holm over the rest;
- thresholds for 5 and 1 percent risk read on the selection split and applied
  to the held-out cells;
- the oracle AUGRC, the headroom any trained abstain could at most close.

Writes results/step6/abstain_preflight.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from bench.dev import PRIVATE
from calib.selective import at_threshold, augrc, oracle_augrc, scores, threshold_at_risk
from model.head.accept import holm
from model.head.compare import fetch

OUT = Path("results/step6/abstain_preflight.json")
STATS = ("max_probability", "top_two_gap", "neg_entropy")


def rows_of(path: Path, texts: dict[str, str]) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        answer = record["keys"][int(np.argmax(record["probs"]))]
        target = record["target"]
        out.append({"row_id": record["row_id"], "task": record["task"],
                    "text": texts.get(record["row_id"], ""),
                    "scores": scores(record["logits"]),
                    "loss": 1.0 - float(target.get(answer, 0.0)),
                    "sure_target": max(target.values()) >= 0.6,
                    "hard_error": float(answer != max(target, key=target.get))})  # fmt: skip
    return out


def split(text: str) -> str:
    return "selection" if int(hashlib.sha256(text.encode()).hexdigest(), 16) % 5 == 0 else "fit"


def main(argv: list[str] | None = None) -> int:
    from sklearn.linear_model import LogisticRegression

    parser = argparse.ArgumentParser(prog="calib.abstain_preflight")
    parser.add_argument("--run", default="r3-base-s2")
    parser.add_argument("--step", type=int, default=2688)
    parser.add_argument("--cache", type=Path, default=Path(".cache/karar/predictions"))
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args(argv)

    texts = {}
    for path in sorted(Path("data/built").glob("**/validation.jsonl")) + [
        Path("data/built/sss/heldout_task.jsonl")
    ]:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                state = row["state"]
                texts[row["row_id"]] = state if isinstance(state, str) else json.dumps(state)
    dev = [json.loads(x) for x in PRIVATE.read_text(encoding="utf-8").splitlines() if x.strip()]
    dev_ids, dev_texts = {r["id"] for r in dev}, {r["text"] for r in dev}

    validation = rows_of(fetch(args.run, args.cache, f"predictions_step{args.step}.jsonl"), texts)
    validation = [
        r for r in validation if r["row_id"] not in dev_ids and r["text"] not in dev_texts
    ]
    fit = [r for r in validation if split(r["text"]) == "fit"]
    selection = [r for r in validation if split(r["text"]) == "selection"]
    heldout = rows_of(fetch(args.run, args.cache, "predictions_heldout.jsonl"), texts)
    heldout = [r for r in heldout if r["row_id"] not in dev_ids]

    def features(rows):
        return np.array([[r["scores"][s] for s in STATS] for r in rows])

    model = LogisticRegression(max_iter=1000).fit(
        features(fit), np.array([r["loss"] > 0.5 for r in fit], dtype=int)
    )

    def confidence(rows, name):
        if name == "stats_logistic":
            return -model.predict_proba(features(rows))[:, 1]
        return np.array([r["scores"][name] for r in rows])

    names = [*scores([0.0, 1.0]).keys(), "stats_logistic"]
    loss = np.array([r["loss"] for r in heldout])
    rng = np.random.default_rng(0)
    picks = rng.integers(0, len(loss), (args.draws, len(loss)))
    base = confidence(heldout, "max_probability")
    base_draws = np.array([augrc(base[p], loss[p]) for p in picks])
    result: dict = {"run": args.run, "rows": {"fit": len(fit), "selection": len(selection),
                    "heldout": len(heldout), "dev_removed": len(dev_ids)},
                    "heldout_error_rate": float(loss.mean()),
                    "oracle_augrc": oracle_augrc(loss), "scores": {}}  # fmt: skip
    sel_loss = np.array([r["loss"] for r in selection])
    for name in names:
        conf = confidence(heldout, name)
        entry = {"augrc": augrc(conf, loss)}
        if name != "max_probability":
            draws = np.array([augrc(conf[p], loss[p]) for p in picks]) - base_draws
            p_value = float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())))
            entry["minus_max_probability"] = {
                "difference": entry["augrc"] - augrc(base, loss),
                "interval": np.percentile(draws, [2.5, 97.5]).tolist(),
                "p": p_value}  # fmt: skip
        sel_conf = confidence(selection, name)
        for risk in (0.05, 0.01):
            t = threshold_at_risk(sel_conf, sel_loss, risk)
            entry[f"at_risk_{risk}"] = at_threshold(conf, loss, t)
        sure = np.array([r["sure_target"] for r in heldout])
        hard = np.array([r["hard_error"] for r in heldout])
        entry["augrc_hard_error_sure_rows"] = augrc(conf[sure], hard[sure])
        result["scores"][name] = entry
    # The shipped abstain probability: the expected error given the
    # maximum probability, an isotonic map fitted on the selection split, and
    # its reliability on the held-out cells.
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=False)
    iso.fit(confidence(selection, "max_probability"), sel_loss)
    predicted = iso.predict(base)
    edges = np.linspace(0, 1, 11)
    bins = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = (predicted >= low) & (predicted < high if high < 1 else predicted <= high)
        if inside.any():
            bins.append({"bin": [float(low), float(high)], "rows": int(inside.sum()),
                         "predicted": float(predicted[inside].mean()),
                         "realised": float(loss[inside].mean())})  # fmt: skip
    result["shipped_abstain"] = {
        "map": "isotonic, expected soft error given maximum probability, fitted on selection",
        "ece_10": float(sum(b["rows"] * abs(b["predicted"] - b["realised"]) for b in bins)
                        / len(loss)),
        "reliability": bins,
    }  # fmt: skip
    sure = np.array([r["sure_target"] for r in heldout])
    hard = np.array([r["hard_error"] for r in heldout])
    sel_sure = np.array([r["sure_target"] for r in selection])
    sel_hard = np.array([r["hard_error"] for r in selection])
    sel_base = confidence(selection, "max_probability")
    decisive = {}
    for r in (0.05, 0.01):
        t = threshold_at_risk(sel_base[sel_sure], sel_hard[sel_sure], r)
        decisive[f"at_risk_{r}"] = at_threshold(base[sure], hard[sure], t)
    result["decisive_rows_hard_error"] = {
        "heldout_rows": int(sure.sum()),
        "error_rate": float(hard[sure].mean()),
        **decisive,
    }
    rejected = holm({n: e["minus_max_probability"]["p"] for n, e in result["scores"].items()
                     if n != "max_probability"})  # fmt: skip
    result["better_than_max_probability_after_holm"] = sorted(
        n
        for n, ok in rejected.items()
        if ok and result["scores"][n]["minus_max_probability"]["difference"] < 0
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
