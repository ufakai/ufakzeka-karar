"""Round 2's row scores and weights, from the baseline's scored files.

Every training row is scored at every evaluation point of every baseline seed;
the scores are averaged into one distribution per row, and three rules read it:

- the confident-learning flag: a row whose target's top answer is j is flagged
  when the model gives some other answer k at least t_k, the mean probability
  the model gives k on rows whose target is k (per-class thresholds, as in
  confident learning). Per task, the flags kept are capped at the calibrated
  off-diagonal mass of the confident joint, lowest probability on the target
  first;
- trivial rows: inside each task and question type, rows whose margin is in the
  top quarter and whose top answer is the target's;
- the memorisation check: per capped task, the median soft Brier on trained
  rows against the median on rows the mix never drew.

Validation and held-out rows are never read here. The constants are ours, not
from a paper.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

FLAG_WEIGHT = 0.25
TRIVIAL_WEIGHT = 0.5
TRIVIAL_QUANTILE = 0.75
TRIVIAL_FLOOR = 0.10
MEMORISED_RATIO = 0.5


def load_scores(paths: list[Path]) -> dict[str, dict]:
    """Per row id: task, target, and the probabilities averaged over every file."""
    sums: dict[str, dict] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            probs = dict(zip(record["keys"], record["probs"], strict=True))
            fresh = {"task": record["task"], "target": record["target"],
                     "probs": defaultdict(float), "n": 0}  # fmt: skip
            row = sums.setdefault(record["row_id"], fresh)
            for key, value in probs.items():
                row["probs"][key] += value
            row["n"] += 1
    return {
        row_id: {"task": r["task"], "target": r["target"],
                 "probs": {k: v / r["n"] for k, v in r["probs"].items()}}
        for row_id, r in sums.items()
    }  # fmt: skip


def top(distribution: dict[str, float]) -> str:
    return max(distribution, key=distribution.get)


def margin(row: dict) -> float:
    """Probability on the target's top answer minus the highest other."""
    label = top(row["target"])
    others = [v for k, v in row["probs"].items() if k != label]
    return row["probs"].get(label, 0.0) - (max(others) if others else 0.0)


def soft_brier(row: dict) -> float:
    keys = set(row["probs"]) | set(row["target"])
    return sum((row["probs"].get(k, 0.0) - row["target"].get(k, 0.0)) ** 2 for k in keys)


def confident_flags(rows: dict[str, dict]) -> set[str]:
    """Row ids flagged by confident learning, pruned per cell by noise rate.

    Per task: t_j is the mean probability the model gives j on rows whose
    target's top answer is j. The confident joint counts each row once, at its
    given label i and the most probable answer k that clears t_k; each given
    label's counts are calibrated to its number of rows. Then, as cleanlab's
    prune_by_noise_rate, for every cell (i, k) with k not i, the n_ik rows given
    i with the largest margin p_k - p_i are flagged, so a large class cannot
    crowd out a small one's flags.
    """
    by_task: dict[str, list[str]] = defaultdict(list)
    for row_id, row in rows.items():
        by_task[row["task"]].append(row_id)
    flagged: set[str] = set()
    for ids in by_task.values():
        own: dict[str, list[float]] = defaultdict(list)
        given: dict[str, list[str]] = defaultdict(list)
        for row_id in ids:
            label = top(rows[row_id]["target"])
            own[label].append(rows[row_id]["probs"].get(label, 0.0))
            given[label].append(row_id)
        threshold = {k: statistics.mean(v) for k, v in own.items()}
        joint: dict[tuple[str, str], int] = defaultdict(int)
        for row_id in ids:
            row = rows[row_id]
            above = [(p, k) for k, p in row["probs"].items()
                     if k in threshold and p >= threshold[k]]  # fmt: skip
            if above:
                joint[(top(row["target"]), max(above)[1])] += 1
        counted: dict[str, int] = defaultdict(int)
        for (label, _), n in joint.items():
            counted[label] += n
        for (label, guess), n in joint.items():
            if guess == label:
                continue
            quota = round(n / counted[label] * len(given[label]))
            probs = [(rows[r]["probs"].get(guess, 0.0) - rows[r]["probs"].get(label, 0.0), r)
                     for r in given[label] if guess in rows[r]["probs"]]  # fmt: skip
            probs.sort(reverse=True)
            flagged.update(r for margin_, r in probs[:quota] if margin_ > 0)
    return flagged


def trivial_rows(rows: dict[str, dict], types: dict[str, str]) -> set[str]:
    """Top-quarter margins with the right answer, inside each task and question type."""
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row_id, row in rows.items():
        groups[(row["task"], types.get(row_id, ""))].append(row_id)
    out: set[str] = set()
    for ids in groups.values():
        margins = sorted(margin(rows[r]) for r in ids)
        cut = margins[min(len(margins) - 1, int(TRIVIAL_QUANTILE * len(margins)))]
        out.update(r for r in ids if margin(rows[r]) >= cut and margin(rows[r]) > 0)
    return out


def memorised(trained: dict[str, dict], unseen: dict[str, dict]) -> dict:
    """Per task with unseen rows: median soft Brier trained and unseen, and the verdict."""
    tasks = sorted({r["task"] for r in unseen.values()})
    per_task = {}
    for task in tasks:
        a = [soft_brier(r) for r in trained.values() if r["task"] == task]
        b = [soft_brier(r) for r in unseen.values() if r["task"] == task]
        if a and b:
            ratio = statistics.median(a) / max(statistics.median(b), 1e-9)
            per_task[task] = {"trained": statistics.median(a), "unseen": statistics.median(b),
                              "ratio": ratio}  # fmt: skip
    failing = sum(v["ratio"] < MEMORISED_RATIO for v in per_task.values())
    return {"per_task": per_task, "memorised": failing > len(per_task) / 2}


def weights(
    rows: dict[str, dict], types: dict[str, str], flags_apply: bool
) -> tuple[dict[str, float], dict]:
    """Row multipliers and what set them. Flags apply only after the owner's audit."""
    flagged = confident_flags(rows) if flags_apply else set()
    trivial = trivial_rows(rows, types) - flagged
    out = {r: FLAG_WEIGHT for r in flagged} | {r: TRIVIAL_WEIGHT for r in trivial}
    # The floor: a task keeps at least TRIVIAL_FLOOR of its weight on trivial rows.
    by_task: dict[str, list[str]] = defaultdict(list)
    for row_id, row in rows.items():
        by_task[row["task"]].append(row_id)
    for ids in by_task.values():
        total = sum(out.get(r, 1.0) for r in ids)
        easy = [r for r in ids if r in trivial]
        if easy and sum(out[r] for r in easy) < TRIVIAL_FLOOR * total:
            for r in easy:
                out[r] = 1.0
    binding = 0
    for ids in by_task.values():
        easy = [r for r in ids if r in trivial]
        binding += bool(easy) and all(out[r] == 1.0 for r in easy)
    report = {"rows": len(rows), "flagged": len(flagged), "trivial": len(trivial),
              "tasks_where_the_floor_bound": binding,
              "flagged_per_task": _count(flagged, rows),
              "trivial_per_task": _count(trivial, rows)}  # fmt: skip
    return out, report


def _count(ids: set[str], rows: dict[str, dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row_id in ids:
        counts[rows[row_id]["task"]] += 1
    return dict(sorted(counts.items()))


def audit_sample(flagged: set[str], rows: dict[str, dict], count: int = 50, seed: int = 1) -> list:
    """Flagged row ids for the owner's blind audit: a uniform draw of the flags.

    The owner's verdict decides whether the flag weight applies to every flagged
    row, so the sample mirrors where the flags fall; a draw spread evenly over
    tasks gave the support templates 88 percent of it against half the flags.
    """
    import random

    return random.Random(seed).sample(sorted(flagged), min(count, len(flagged)))


def main(argv: list[str] | None = None) -> int:
    """Weights for the next round from a round's scored runs.

    python -m model.head.scoring --runs r2-base-s1,r2-base-s2,r2-base-s3 --round b
    python -m model.head.scoring ... --round b --flags     # after the owner's audit
    """
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(prog="model.head.scoring")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--round", required=True, choices=["b", "c"])
    parser.add_argument("--flags", action="store_true", help="apply the flag weight")
    parser.add_argument("--crossfit", action="store_true", help="the runs are cross-fit folds")
    parser.add_argument("--cache", type=Path, default=Path(".cache/karar/scores"))
    parser.add_argument("--out", type=Path, default=Path("results/step6/round2"))
    args = parser.parse_args(argv)

    def fetch(run: str, name: str) -> Path:
        target = args.cache / run / name
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            # Downloaded under a temporary name, so a download cut short (a full
            # disk did this once) is never taken for a complete file.
            part = target.with_suffix(".part")
            subprocess.run(["modal", "volume", "get", "--force", "ufakzeka-karar",
                            f"round1/runs/{run}/{name}", str(part)], check=True)  # fmt: skip
            part.rename(target)
        return target

    def listed(run: str) -> list[str]:
        # The evaluation points depend on the mix's size, so the files are
        # listed rather than named (the filtered data gives 2,696 steps, not 2,704).
        out = subprocess.run(["modal", "volume", "ls", "--json", "ufakzeka-karar",
                              f"round1/runs/{run}"], check=True, capture_output=True,
                             text=True).stdout  # fmt: skip
        names = [Path(e["filename"]).name for e in json.loads(out)]
        return sorted(n for n in names if n.startswith("scores_step"))

    runs = args.runs.split(",")
    if args.crossfit:
        return crossfit(args, runs, fetch, listed)
    mixed = [json.loads(line) for line in fetch(runs[0], "train_mixed.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]  # fmt: skip
    trained_ids = {r["row_id"] for r in mixed}
    types = {r["row_id"]: r["question"]["type"] for r in mixed}

    files, last = [], []
    for run in runs:
        found = sorted(listed(run), key=lambda n: int(n.split("step")[1].split(".")[0]))
        if len(found) != 4:
            raise RuntimeError(f"{run}: expected 4 scored points, found {found}")
        files += [fetch(run, name) for name in found]
        last.append(fetch(run, found[-1]))
    scores = load_scores(files)
    trained = {k: v for k, v in scores.items() if k in trained_ids}
    # The memorisation check reads the last point only: at the first, half the
    # training rows have not been seen yet, which would bias it toward "not
    # memorised".
    final = load_scores(last)
    check = memorised({k: v for k, v in final.items() if k in trained_ids},
                      {k: v for k, v in final.items() if k not in trained_ids})  # fmt: skip
    multipliers, report = weights(trained, types, flags_apply=args.flags)
    flagged = confident_flags(trained)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"weights_{args.round}.json").write_text(json.dumps(multipliers))
    (args.out / f"audit_{args.round}.json").write_text(json.dumps(
        audit_sample(flagged, trained), indent=2))  # fmt: skip
    summary = {"runs": runs, "flags_applied": args.flags, "memorisation": check,
               "flagged_before_audit": len(flagged), **report}  # fmt: skip
    (args.out / f"weights_{args.round}_report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("per_task")},
                     indent=2, default=str))  # fmt: skip
    return 0


def crossfit(args, runs: list[str], fetch, listed) -> int:
    """Weights from out-of-sample scores: each fold run scored the fifth it never saw."""
    scores: dict[str, dict] = {}
    types: dict[str, str] = {}
    for run in runs:
        found = listed(run)
        if len(found) != 1:
            raise RuntimeError(f"{run}: expected one scored point, found {found}")
        scores |= load_scores([fetch(run, found[0])])
        for line in fetch(run, "fold.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                types[row["row_id"]] = row["question"]["type"]
    if set(scores) != set(types):
        raise RuntimeError("scored rows and fold rows differ")
    multipliers, report = weights(scores, types, flags_apply=args.flags)
    flagged = confident_flags(scores)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"weights_{args.round}.json").write_text(json.dumps(multipliers))
    (args.out / f"audit_{args.round}.json").write_text(json.dumps(
        audit_sample(flagged, scores), indent=2))  # fmt: skip
    summary = {"runs": runs, "crossfit": True, "flags_applied": args.flags,
               "flagged_before_audit": len(flagged), **report}  # fmt: skip
    (args.out / f"weights_{args.round}_report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("per_task")}, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
