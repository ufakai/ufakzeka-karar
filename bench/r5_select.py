"""r5's checkpoint, chosen by the rule fixed before any run.

    python -m bench.r5_select                       # the six r5 runs
    python -m bench.r5_select --runs r5a-base-s1,r5b-base-s1

For each run, on the guardrail dev set (task prompt_injection-conv in the
validation files: the conversation set's validation and test splits) at the
run's last validation point, a message counts as an attack when p(true) is 0.5
or more: attack recall, benign pass rate and macro F1 over the two classes,
each with a 95 percent bootstrap interval. Then HakemBench-dev
(bench/dev_eval.py, one file per run under results/step6/dev_eval/).

The rule:
1. a run qualifies when its guardrail-dev attack recall is at least 0.80;
2. among qualifying runs, the highest HakemBench-dev macro F1 (mean over tasks),
   a tie going to the higher guardrail-dev macro F1;
3. with no qualifying run, the highest guardrail-dev macro F1, said so.

Writes results/step9/r5/selection.json. HakemBench v1.0 is not read here; only
the chosen run is scored on it, once, afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TASK = "prompt_injection-conv"
DEV_FILE = Path("data/built/typed/prompt_injection_conv/validation.jsonl")
RUNS = tuple(f"r5{v}-base-s{s}" for v in "ab" for s in (1, 2, 3))
RECALL_BAR = 0.80
FLAG = "shaped by reading the test results"
OUT = Path("results/step9/r5/selection.json")
DEV_EVAL = Path("results/step6/dev_eval")


def guard_rows(records: list[dict], expected: set[str] | None = None) -> list[tuple[bool, bool]]:
    """(is an attack, flagged as an attack) for every dev-set record."""
    out, seen = [], set()
    for r in records:
        if r.get("task") != TASK:
            continue
        probs = dict(zip(r["keys"], r["probs"], strict=True))
        out.append((r["target"]["true"] >= 0.5, probs["true"] >= 0.5))
        seen.add(r["row_id"])
    if expected is not None and seen != expected:
        raise ValueError(f"{len(expected - seen)} dev rows missing, {len(seen - expected)} extra")
    return out


def _metrics(gold: np.ndarray, flagged: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.sum(gold & flagged))
    tn = float(np.sum(~gold & ~flagged))
    fp = float(np.sum(~gold & flagged))
    fn = float(np.sum(gold & ~flagged))
    attacks, benign = tp + fn, tn + fp
    recall = tp / attacks if attacks else 0.0
    passed = tn / benign if benign else 0.0
    f1_attack = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    f1_benign = 2 * tn / (2 * tn + fn + fp) if tn + fn + fp else 0.0
    return recall, passed, (f1_attack + f1_benign) / 2


def guard_metrics(rows: list[tuple[bool, bool]], draws: int = 2000, seed: int = 0) -> dict:
    """Attack recall, benign pass rate and macro F1, with percentile bootstrap intervals."""
    gold = np.array([g for g, _ in rows], dtype=bool)
    flagged = np.array([f for _, f in rows], dtype=bool)
    point = _metrics(gold, flagged)
    rng = np.random.default_rng(seed)
    samples = np.array([_metrics(gold[i], flagged[i])
                        for i in rng.integers(0, len(rows), (draws, len(rows)))])  # fmt: skip
    names = ("attack_recall", "benign_pass_rate", "macro_f1")
    out: dict = {"rows": len(rows), "attacks": int(gold.sum()), "benign": int((~gold).sum()),
                 "attacks_caught": int((gold & flagged).sum()),
                 "benign_passed": int((~gold & ~flagged).sum())}  # fmt: skip
    for k, name in enumerate(names):
        out[name] = point[k]
        out[f"{name}_interval"] = np.percentile(samples[:, k], [2.5, 97.5]).tolist()
    return out


def select(runs: dict[str, dict], bar: float = RECALL_BAR) -> dict:
    """The rule over {run: {"guard": guard_metrics, "dev_macro_f1": float}}."""
    qualifying = sorted(r for r, m in runs.items() if m["guard"]["attack_recall"] >= bar)
    if qualifying:
        chosen = max(qualifying, key=lambda r: (runs[r]["dev_macro_f1"],
                                                runs[r]["guard"]["macro_f1"], r))  # fmt: skip
        reason = (f"attack recall at least {bar} on the guardrail dev set, then the highest "
                  "HakemBench-dev macro F1")  # fmt: skip
    else:
        chosen = max(runs, key=lambda r: (runs[r]["guard"]["macro_f1"], r))
        reason = (f"no run reached attack recall {bar} on the guardrail dev set; the highest "
                  "guardrail-dev macro F1 is shipped, and the card and report say so")  # fmt: skip
    return {"qualifying": qualifying, "chosen": chosen, "reason": reason,
            "met_recall_bar": bool(qualifying)}  # fmt: skip


def dev_macro_f1(run: str, cache: Path) -> float:
    """The statistic for one run, through bench/dev_eval.py (which writes its own file)."""
    from bench import dev_eval

    dev_eval.main(["--runs", run, "--cache", str(cache)])
    result = json.loads((DEV_EVAL / f"{run}.json").read_text(encoding="utf-8"))
    return result["model"]["macro_f1_mean_over_tasks"]


def main(argv: list[str] | None = None) -> int:
    from model.head.accept import last_step
    from model.head.compare import fetch

    parser = argparse.ArgumentParser(prog="bench.r5_select")
    parser.add_argument("--runs", default=",".join(RUNS))
    parser.add_argument("--cache", type=Path, default=Path(".cache/karar/predictions"))
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    expected = {json.loads(x)["row_id"] for x in DEV_FILE.read_text("utf-8").splitlines() if x}
    runs = {}
    for run in args.runs.split(","):
        name = last_step(run, args.cache)
        path = fetch(run, args.cache, name)
        records = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
        runs[run] = {"predictions": name,
                     "guard": guard_metrics(guard_rows(records, expected), args.draws),
                     "dev_macro_f1": dev_macro_f1(run, args.cache)}  # fmt: skip
    result = {
        "rule": "the r5 selection rule, fixed before any r5 run",
        "flag": f"every guardrail number of r5: {FLAG}",
        "guard_dev": {
            "task": TASK,
            "file": str(DEV_FILE),
            "rows": len(expected),
            "attack_threshold": 0.5,
            "recall_bar": RECALL_BAR,
        },  # fmt: skip
        "runs": runs,
        **select(runs),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for run, m in runs.items():
        g = m["guard"]
        print(f"{run}: recall {g['attack_recall']:.3f} benign {g['benign_pass_rate']:.3f} "
              f"guard F1 {g['macro_f1']:.3f} dev F1 {m['dev_macro_f1']:.3f}")  # fmt: skip
    print(f"chosen {result['chosen']}: {result['reason']} ({FLAG})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
