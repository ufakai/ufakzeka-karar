"""The paired comparisons, from the runs' final validation predictions.

    python -m model.head.compare --pair continued      # the continuation decision
    python -m model.head.compare --pair encoders       # each encoder against ours

Each run's predictions at its last evaluation point, and on the held-out cells, are fetched from the
project volume, and model/head/decide.py's rule is applied: the mean over tasks of macro F1, a
paired bootstrap over items within each task and over seeds. Three readings per arm:

- validation: every validation task but the three general sets. For the
  continuation this one decides: the continued backbone replaces the
  original only if the whole interval lies above zero.
- general: MASSIVE, OffensEval and MiDe22 alone, to see forgetting.
- heldout: the held-out cells, which say whether a gain transfers.

The last two decide nothing, and for the encoders nothing is acted on at all.

Writes results/step4/continue/decision.json or results/step4/round1/encoders.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from model.head.decide import decide

STEP = 2704
OURS = ("r1-main-s1", "r1-main-s2", "r1-main-s3")
ARMS = {
    "continued": {"continued": ("r1-continued-s1", "r1-continued-s2", "r1-continued-s3")},
    "encoders": {
        "berturk": ("r1-berturk-lr1e-4-s1", "r1-berturk-lr1e-4-s2", "r1-berturk-lr1e-4-s3"),
        "moganbert": ("r1-moganbert-lr1e-4-s1", "r1-moganbert-lr1e-4-s2",
                      "r1-moganbert-lr1e-4-s3"),
    },
}  # fmt: skip
# The general tasks reported beside the decision and never inside it.
GENERAL = frozenset({"massive_tr", "offenseval_tr", "mide22"})
OUT = {"continued": Path("results/step4/continue/decision.json"),
       "encoders": Path("results/step4/round1/encoders.json")}  # fmt: skip


def fetch(run: str, cache: Path, file: str = f"predictions_step{STEP}.jsonl") -> Path:
    """One of the run's predictions files, downloaded once."""
    target = cache / run / file
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        remote = f"round1/runs/{run}/{file}"
        # A temporary name first, so a cut-short download is never read as whole.
        part = target.with_suffix(".part")
        subprocess.run(["modal", "volume", "get", "--force", "ufakzeka-karar", remote,
                        str(part)], check=True)  # fmt: skip
        part.rename(target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model.head.compare")
    parser.add_argument("--pair", choices=list(ARMS), required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/karar/predictions"))
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args(argv)
    ours = [fetch(run, args.cache) for run in OURS]
    ours_heldout = [fetch(run, args.cache, "predictions_heldout.jsonl") for run in OURS]
    result = {}
    for name, runs in ARMS[args.pair].items():
        theirs = [fetch(run, args.cache) for run in runs]
        theirs_heldout = [fetch(run, args.cache, "predictions_heldout.jsonl") for run in runs]
        # decide() names the first system "converted"; here it is the arm.
        readings = {
            "validation": decide(theirs, ours, draws=args.draws, exclude=GENERAL),
            "general": decide(theirs, ours, draws=args.draws, only=GENERAL),
            "heldout": decide(theirs_heldout, ours_heldout, draws=args.draws),
        }
        for reading in readings.values():
            reading.pop("verdict")  # The wording is about conversion, not this pair
            low, high = reading["interval"]
            reading["reading"] = ("ahead of ours" if low > 0 else "behind ours" if high < 0
                                  else "no difference the interval can show")  # fmt: skip
        low = readings["validation"]["interval"][0]
        verdict = {"arm": list(runs), "against": list(OURS), **readings}
        if args.pair == "continued":
            verdict["decision"] = ("the continued backbone replaces the original" if low > 0
                                   else "the original backbone is kept")  # fmt: skip
        result[name] = verdict
    OUT[args.pair].parent.mkdir(parents=True, exist_ok=True)
    OUT[args.pair].write_text(json.dumps(result, indent=2))
    print(json.dumps({arm: {part: {x: v[part][x] for x in ("difference", "interval", "reading")}
                            for part in ("validation", "general", "heldout")}
                      | ({"decision": v["decision"]} if "decision" in v else {})
                      for arm, v in result.items()}, indent=2))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
