"""Round 2's acceptance rule, fixed before any round 2 number.

    python -m model.head.accept --arm r2-b-s1,r2-b-s2,r2-b-s3 \
        --against r2-base-s1,r2-base-s2,r2-base-s3

- Primary: macro F1 on the held-out cells, arm minus the set it is judged
  against, by model/head/decide.py's paired bootstrap over items within each
  task and over seeds. The arm wins only if the whole 95 percent interval lies
  above zero.
- Secondaries, under Holm at 0.05: HakemBench-dev macro F1 against the owner's
  answers, and Brier and smooth ECE per question type on validation. Each gets
  a paired bootstrap over items (seeds averaged per item) and a two-sided
  bootstrap p-value.
- The arm is accepted if the primary wins and no secondary is significantly
  worse after Holm. Anything else keeps the previous set.

Writes results/step6/accept/<arm>.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from model.head.compare import fetch
from model.head.decide import decide, load

OUT = Path("results/step6/accept")
DEV = Path("results/private/hakembench/dev_v0.jsonl")


def last_step(run: str, cache: Path) -> str:
    """The run's last validation predictions file name, from its summary."""
    summary = json.loads(fetch(run, cache, "summary.json").read_text(encoding="utf-8"))
    return f"predictions_step{summary['steps']}.jsonl"


def seed_mean(paths: list[Path]) -> dict[str, dict]:
    """Per row: task, target, and the seeds' probabilities averaged by option name."""
    runs = [load(p) for p in paths]
    shared = set.intersection(*(set(r) for r in runs))
    out = {}
    for row_id in shared:
        first = runs[0][row_id]
        probs: dict[str, float] = defaultdict(float)
        for run in runs:
            for key, p in zip(run[row_id]["keys"], run[row_id]["probs"], strict=True):
                probs[key] += p / len(runs)
        out[row_id] = {"task": first["task"], "target": first["target"], "probs": dict(probs)}
    return out


def _top(d: dict[str, float]) -> str:
    return max(d, key=d.get)


def paired(a: np.ndarray, b: np.ndarray, statistic, draws: int, rng) -> dict:
    """Observed a minus b, its 95 percent interval and a two-sided bootstrap p-value."""
    n = len(a)
    observed = statistic(a) - statistic(b)
    diffs = np.empty(draws)
    for i in range(draws):
        pick = rng.integers(0, n, n)
        diffs[i] = statistic(a[pick]) - statistic(b[pick])
    low, high = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"difference": float(observed), "interval": [float(low), float(high)],
            "p": float(min(1.0, p))}  # fmt: skip


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Which hypotheses Holm's step-down procedure rejects."""
    order = sorted(pvalues, key=pvalues.get)
    rejected, stop = {}, False
    for i, name in enumerate(order):
        stop = stop or pvalues[name] > alpha / (len(order) - i)
        rejected[name] = not stop
    return rejected


def secondaries(arm: dict[str, dict], base: dict[str, dict], types: dict[str, str],
                dev: dict[str, str], draws: int, seed: int) -> dict:  # fmt: skip
    import relplot

    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    shared = sorted(set(arm) & set(base))
    for kind in sorted({types[r] for r in shared if r in types}):
        ids = [r for r in shared if types.get(r) == kind]

        def brier(rows, i=ids):
            return np.array([sum((rows[r]["probs"].get(k, 0.0) - rows[r]["target"].get(k, 0.0)) ** 2
                                 for k in set(rows[r]["probs"]) | set(rows[r]["target"]))
                             for r in i])  # fmt: skip

        # Lower is better for both, so "worse" is a positive difference.
        out[f"brier_{kind}"] = paired(brier(arm), brier(base), np.mean, draws, rng)
        conf = {n: np.array([max(rows[r]["probs"].values()) for r in ids])
                for n, rows in (("arm", arm), ("base", base))}  # fmt: skip
        hit = {n: np.array([_top(rows[r]["probs"]) == _top(rows[r]["target"]) for r in ids],
                           dtype=float) for n, rows in (("arm", arm), ("base", base))}  # fmt: skip
        stacked_a = np.stack([conf["arm"], hit["arm"]], axis=1)
        stacked_b = np.stack([conf["base"], hit["base"]], axis=1)
        out[f"smooth_ece_{kind}"] = paired(
            stacked_a, stacked_b, lambda x: float(relplot.smECE(x[:, 0], x[:, 1])), draws // 4, rng
        )
    dev_ids = sorted(r for r in dev if r in arm and r in base)
    if dev_ids:
        gold = np.array([dev[r] for r in dev_ids])
        pred_a = np.array([_top(arm[r]["probs"]) for r in dev_ids])
        pred_b = np.array([_top(base[r]["probs"]) for r in dev_ids])
        tasks = np.array([arm[r]["task"] for r in dev_ids])

        def dev_f1(rows):
            # Rows carry (gold, predicted, task); mean over tasks of macro F1.
            scores = []
            for task in np.unique(rows[:, 2]):
                g, p = rows[rows[:, 2] == task, 0], rows[rows[:, 2] == task, 1]
                labels = np.unique(np.concatenate([g, p]))
                f1 = [2 * ((g == k) & (p == k)).sum() / max(1, (g == k).sum() + (p == k).sum())
                      for k in labels]  # fmt: skip
                scores.append(np.mean(f1))
            return float(np.mean(scores))

        a = np.stack([gold, pred_a, tasks], axis=1)
        b = np.stack([gold, pred_b, tasks], axis=1)
        # Higher is better: stored as base minus arm so "worse" is positive here too.
        result = paired(b, a, dev_f1, draws, rng)
        out["dev_macro_f1_loss"] = result
    return out


def accept(primary: dict, second: dict[str, dict]) -> dict:
    rejected = holm({k: v["p"] for k, v in second.items()})
    worse = sorted(k for k, v in second.items() if rejected[k] and v["difference"] > 0)
    wins = primary["interval"][0] > 0
    return {"primary_wins": wins, "secondaries_worse_after_holm": worse,
            "accepted": wins and not worse}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model.head.accept")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--against", required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/karar/predictions"))
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args(argv)
    arm_runs, base_runs = args.arm.split(","), args.against.split(",")
    held = {n: [fetch(r, args.cache, "predictions_heldout.jsonl") for r in runs]
            for n, runs in (("arm", arm_runs), ("base", base_runs))}  # fmt: skip
    primary = decide(held["arm"], held["base"], draws=args.draws)
    primary.pop("verdict")
    val = {n: [fetch(r, args.cache, last_step(r, args.cache)) for r in runs]
           for n, runs in (("arm", arm_runs), ("base", base_runs))}  # fmt: skip
    # Dev items are validation and held-out rows, so both files feed the dev
    # score; Brier and smooth ECE per type read validation rows only.
    arm = seed_mean(val["arm"]) | seed_mean(held["arm"])
    base = seed_mean(val["base"]) | seed_mean(held["base"])
    dev_rows = [json.loads(x) for x in DEV.read_text(encoding="utf-8").splitlines() if x.strip()]
    dev = {r["id"]: _top(r["target"]) for r in dev_rows}
    types = {}
    for path in sorted(Path("data/built").glob("**/validation.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                types[row["row_id"]] = row["question"]["type"]
    second = secondaries(arm, base, types, dev, args.draws, seed=0)
    result = {"arm": arm_runs, "against": base_runs, "primary_heldout_macro_f1": primary,
              "secondaries": second, **accept(primary, second)}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    # Seeds share a name up to "-s<digit>"; a single run keeps its own name.
    name = re.sub(r"-s\d+$", "", arm_runs[0]) if len(arm_runs) > 1 else arm_runs[0]
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "secondaries"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
