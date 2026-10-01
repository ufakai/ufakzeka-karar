"""Which test items the owner checks, after an AI model has reviewed every one.

    python -m bench.hakembench.triage --review <folder of review answers>

Every desk unit was answered blind by an AI model: the text, the question
and the options, never the proposed answer. A unit goes to the owner when the
reviewer disagrees with the proposal or flags a question on it, or when it falls
in a seeded fifth of the agreed units (the sample that certifies the rest), or
when the owner has already checked it. Catch units (a wrong proposal on an item
of known truth) follow the same rule, so a reviewer who catches them sends them
to the owner, and one who does not is measured.

Writes the owner's desk documents (results/private/step8/desk/owner-*.json), the
private list of routed units, and a report of the reviewer against the panel per
track and question type, and against the catch items.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

DESK = Path("results/private/step8/desk")
ROUTED = Path("results/private/step8/triage.json")
REPORT = Path("results/step8/triage_report.json")
SAMPLE_SHARE = 0.2
UNITS_PER_DOC = 40


def load_units() -> list[dict]:
    return [
        u for p in sorted(DESK.glob("units-*.json")) for u in json.loads(p.read_text())["units"]
    ]


def load_catches() -> dict[tuple[str, str], dict]:
    return {(c["unit"], c["qid"]): c for p in sorted(DESK.glob("catches-*.json"))
            if p.name != "catches-sweep.json"
            for c in json.loads(p.read_text())}  # fmt: skip


def disputed(unit: dict, review: dict) -> bool:
    answers = review.get(unit["id"], {})
    for q in unit["questions"]:
        a = answers.get(q["qid"])
        if a is None or a.get("flag", "none") != "none" or a.get("answer") != q["proposed"]:
            return True
    return False


def route(units: list[dict], review: dict, checked: set[str], seed: int = 1) -> dict[str, str]:
    """Unit id to why the owner sees it: disputed, sampled or already checked."""
    rng = random.Random(seed)
    reasons = {}
    agreed = []
    for u in sorted(units, key=lambda x: x["id"]):
        if u["id"] in checked:
            reasons[u["id"]] = "checked"
        elif disputed(u, review):
            reasons[u["id"]] = "disputed"
        else:
            agreed.append(u["id"])
    for uid in rng.sample(agreed, round(SAMPLE_SHARE * len(agreed))):
        reasons[uid] = "sampled"
    return reasons


def report(units: list[dict], review: dict, catches: dict, reasons: dict) -> dict:
    agree: dict[str, list[int]] = defaultdict(list)
    caught = missed = 0
    for u in units:
        answers = review.get(u["id"], {})
        for q in u["questions"]:
            a = answers.get(q["qid"], {})
            catch = catches.get((u["id"], q["qid"]))
            if catch is not None:
                if a.get("flag", "none") == "none":
                    caught += a.get("answer") == catch["truth"]
                    missed += a.get("answer") == catch["shown"]
                continue
            if a.get("flag", "none") == "none":
                agree[f"{u['track']}:{q['type']}"].append(int(a.get("answer") == q["proposed"]))
    return {
        "units": len(units),
        "to_owner": dict(Counter(reasons.values())),
        "not_shown": len(units) - len(reasons),
        "reviewer_agrees_with_proposal": {
            k: round(sum(v) / len(v), 3) for k, v in sorted(agree.items())
        },  # fmt: skip
        "reviewer_on_catch_items": {
            "caught": caught,
            "took_the_wrong_answer": missed,
            "catch_items": len(catches),
        },  # fmt: skip
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.hakembench.triage")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--checked", type=Path, help="folder of the owner's check documents")
    args = parser.parse_args(argv)
    review = {}
    for p in sorted(args.review.glob("*.json")):
        review |= json.loads(p.read_text(encoding="utf-8"))
    checked = set()
    if args.checked:
        checked = {json.loads(p.read_text())["item"] for p in args.checked.glob("*.json")}
    units = load_units()
    missing = [u["id"] for u in units if u["id"] not in review]
    if missing:
        raise SystemExit(f"{len(missing)} units have no review answer, e.g. {missing[:3]}")
    reasons = route(units, review, checked)
    routed = [u for u in units if u["id"] in reasons]
    routed.sort(key=lambda u: (u["track"], u["order"]))
    for n, u in enumerate(routed):
        u["order"] = n
    for old in DESK.glob("owner-*.json"):
        old.unlink()
    for i in range(0, len(routed), UNITS_PER_DOC):
        (DESK / f"owner-{i // UNITS_PER_DOC}.json").write_text(
            json.dumps({"units": routed[i : i + UNITS_PER_DOC]}, ensure_ascii=False)
        )
    ROUTED.write_text(json.dumps(reasons, indent=1))
    out = report(units, review, load_catches(), reasons)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
