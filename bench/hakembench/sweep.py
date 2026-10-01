"""The second sweep: two blind AI passes and the panel decide what the owner checks.

    python -m bench.hakembench.sweep plan
    python -m bench.hakembench.sweep adjudication-input
    python -m bench.hakembench.sweep desk --adjudication <folder>

Every desk unit now has the panel's proposal and two blind answers from an AI
assistant (the first pass and a second, independent one). The
source-labelled items no reviewer had seen (the unsampled rest of the legal and
relevance splits) get the same two passes, with the source label in the
panel's place. `plan` sorts each unit:

- catch: every catch unit goes to the owner, whatever the reviewers said. A
  catch whose shown "wrong" answer all three sources gave is marked suspect:
  its source label is in doubt, the owner decides it, and it leaves the catch
  rate. A choice catch whose truth a reviewer missed shows that reviewer's
  wrong option, a plausible error rather than a random one.
- content: a pass flagged the text (illegal promotion, personal data, other).
  An adjudicator recommends drop, mask or keep; the owner confirms every drop.
- disputed: a question where the panel and the two passes are not unanimous,
  or a pass flagged it. The adjudicator answers it; the owner verifies.
- blind: 30 fact-check priority questions shown with no proposal, first on the
  desk, because the panel reads that scale differently from the reviewers and
  a systematic shift is invisible to catch items.
- sampled: a seeded fifth of the unanimous units, which certifies the rest.
- shifted: unanimous score questions shown one level off, a catch for the
  near misses an adjudicated proposal risks.
- agreed: unanimous and never shown; the shared answer becomes gold.

`desk` writes the owner's documents: the adjudicated answer is the shown
proposal on disputed questions, the shown wrong answer on catches, nothing on
blind questions and the panel's answer elsewhere.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

from bench.hakembench.desk import CANDIDATES, SPOT_SOURCES, key_of, read_jsonl, units

DESK = Path("results/private/step8/desk")
REVIEW1 = Path("results/private/step8/ai_review")
REVIEW2 = Path("results/private/step8/ai_review2")
REST = Path("results/private/step8/ai_review_rest")
CHECKS = Path("results/private/step8/checks/checks")
WORK = Path("results/private/step8/sweep")
ROUTED = Path("results/private/step8/triage.json")
REPORT = Path("results/step8/sweep_report.json")
CONTENT_FLAGS = ("illegal_promo", "personal_data", "other")
SAMPLE_SHARE = 0.2
BLIND_COUNT = 30
BLIND_CELL = ("dogrulama", "score")
SHIFTED_PER_TRACK = {"egitim": 12, "sss": 12}
UNITS_PER_DOC = 40
SEED = 120


def load_json_folder(folder: Path) -> dict:
    out: dict = {}
    for path in sorted(folder.glob("*.json")):
        out |= json.loads(path.read_text(encoding="utf-8"))
    return out


def answers_of(review: dict, uid: str) -> dict:
    """A reviewer's answers for one unit; the first pass stored them unwrapped."""
    entry = review.get(uid, {})
    return entry.get("answers", entry) if "answers" in entry else entry


def content_of(reviews: list[dict], uid: str) -> list[dict]:
    return [{"content": r[uid]["content"], "note": r[uid].get("content_note", "")}
            for r in reviews if uid in r and r[uid].get("content") in CONTENT_FLAGS]  # fmt: skip


def unanimous(proposal: str, answers: list[dict]) -> bool:
    return all(a.get("flag", "none") == "none" and a.get("answer") == proposal for a in answers)


def classify(unit: dict, reviews: list[dict], catch: dict[str, dict]) -> dict:
    """Each question's proposal, the passes' answers and whether all agree; the unit's flags."""
    questions = {}
    for q in unit["questions"]:
        answers = [answers_of(r, unit["id"]).get(q["qid"], {}) for r in reviews]
        questions[q["qid"]] = {"proposed": q["proposed"], "answers": answers,
                               "unanimous": unanimous(q["proposed"], answers),
                               "catch": q["qid"] in catch}  # fmt: skip
    return {"questions": questions, "content": content_of(reviews, unit["id"])}


def plausible_shown(catch: dict, answers: list[dict]) -> str:
    """The wrong option a reviewer picked, when one did; else the catch's own wrong answer."""
    picked = [a.get("answer") for a in answers if a.get("answer") not in (None, catch["truth"])]
    return sorted(picked)[0] if picked else catch["shown"]


def plan(desk_units: list[dict], reviews: list[dict], catches: list[dict], checked: set[str],
         previous: dict[str, str], seed: int = SEED) -> dict:  # fmt: skip
    """Status per unit, the questions to adjudicate, the blind and shifted questions."""
    rng = random.Random(seed)
    catch_of: dict[str, dict[str, dict]] = {}
    for c in catches:
        catch_of.setdefault(c["unit"], {})[c["qid"]] = c
    status, detail, agreed = {}, {}, []
    for u in sorted(desk_units, key=lambda x: x["id"]):
        info = classify(u, reviews, catch_of.get(u["id"], {}))
        detail[u["id"]] = info
        others = [q for q in info["questions"].values() if not q["catch"]]
        if u["id"] in catch_of:
            status[u["id"]] = "catch"
        elif u["id"] in checked:
            status[u["id"]] = "checked"
        elif info["content"]:
            status[u["id"]] = "content"
        elif not all(q["unanimous"] for q in others):
            status[u["id"]] = "disputed"
        else:
            status[u["id"]] = "agreed"
            agreed.append(u["id"])
    by_id = {u["id"]: u for u in desk_units}
    # Blind questions: drawn over every unit of the cell that is not a catch.
    pool = sorted((uid, q["qid"]) for uid, u in by_id.items() if u["track"] == BLIND_CELL[0]
                  and status[uid] != "catch"
                  for q in u["questions"] if q["type"] == BLIND_CELL[1])  # fmt: skip
    blind = rng.sample(pool, min(BLIND_COUNT, len(pool)))
    for uid, _ in blind:
        if status[uid] == "agreed":
            status[uid] = "blind"
            agreed.remove(uid)
    # Shifted catches: unanimous score questions of agreed units, one level off.
    shifted = []
    for track, n in sorted(SHIFTED_PER_TRACK.items()):
        candidates = sorted((uid, qid) for uid in agreed if by_id[uid]["track"] == track
                            for qid, q in detail[uid]["questions"].items()
                            if next(x for x in by_id[uid]["questions"]
                                    if x["qid"] == qid)["type"] == "score")  # fmt: skip
        for uid, qid in rng.sample(candidates, min(n, len(candidates))):
            if status[uid] != "agreed":
                continue
            q = next(x for x in by_id[uid]["questions"] if x["qid"] == qid)
            level, top = int(q["proposed"]), len(q["options"]) - 1
            steps = [s for s in (level - 1, level + 1) if 0 <= s <= top]
            shifted.append({"unit": uid, "qid": qid, "truth": q["proposed"],
                            "shown": str(rng.choice(steps)), "kind": "shifted_level"})  # fmt: skip
            status[uid] = "shifted"
            agreed.remove(uid)
    # The certification sample keeps the units sampled before that are still agreed.
    target = round(SAMPLE_SHARE * len(agreed))
    kept = [uid for uid in agreed if previous.get(uid) == "sampled"]
    fresh = [uid for uid in agreed if uid not in kept]
    extra = rng.sample(fresh, max(0, min(len(fresh), target - len(kept))))
    for uid in kept + extra:
        status[uid] = "sampled"
    # Catches: suspect when every source gave the shown answer; else a plausible wrong one.
    updated = []
    for c in catches:
        answers = detail[c["unit"]]["questions"][c["qid"]]["answers"]
        suspect = all(a.get("answer") == c["shown"] for a in answers)
        row = {**c, "suspect": suspect}
        if not suspect and c["unit"] not in checked:
            row["shown"] = plausible_shown(c, answers)
        updated.append(row)
    adjudicate = {uid: [qid for qid, q in detail[uid]["questions"].items()
                        if not q["unanimous"] and not q["catch"]]
                  for uid, s in status.items() if s in ("disputed", "content")}  # fmt: skip
    return {"status": status, "detail": detail, "blind": [list(b) for b in blind],
            "shifted": shifted, "catches": updated,
            "adjudicate": {k: v for k, v in adjudicate.items() if v},
            "content": [uid for uid, s in status.items() if s == "content"]}  # fmt: skip


def adjudication_units(result: dict, desk_units: list[dict], seed: int = SEED) -> list[dict]:
    """What the adjudicator sees: text, question, options, the distinct answers unlabelled."""
    rng = random.Random(seed + 1)
    by_id = {u["id"]: u for u in desk_units}
    out = []
    for uid in sorted(set(result["adjudicate"]) | set(result["content"])):
        u = by_id[uid]
        questions = []
        for q in u["questions"]:
            if q["qid"] not in result["adjudicate"].get(uid, []):
                continue
            seen = result["detail"][uid]["questions"][q["qid"]]
            distinct = sorted({q["proposed"], *(a.get("answer") for a in seen["answers"])} - {None})
            rng.shuffle(distinct)
            questions.append({"qid": q["qid"], "question": q["question"], "type": q["type"],
                              "options": q["options"], "candidates": distinct})  # fmt: skip
        out.append({"id": uid, "track": u["track"], "text": u["text"], "questions": questions,
                    "content_flags": result["detail"][uid]["content"]})  # fmt: skip
    return out


def owner_units(result: dict, desk_units: list[dict], adjudicated: dict) -> list[dict]:
    """The owner's desk: blind units first, then every other routed unit by track."""
    catch = {(c["unit"], c["qid"]): c for c in result["catches"] + result["shifted"]}
    blind = {tuple(b) for b in result["blind"]}
    routed = {uid for uid, s in result["status"].items() if s != "agreed"}
    rows = []
    for u in desk_units:
        if u["id"] not in routed:
            continue
        questions = []
        for q in u["questions"]:
            shown = q["proposed"]
            if (u["id"], q["qid"]) in catch:
                shown = catch[(u["id"], q["qid"])]["shown"]
            elif (u["id"], q["qid"]) in blind:
                shown = None
            elif q["qid"] in result["adjudicate"].get(u["id"], []):
                got = adjudicated.get(u["id"], {}).get("answers", {}).get(q["qid"], {})
                if got.get("answer") is None:
                    raise ValueError(f"{u['id']}:{q['qid']} has no adjudicated answer")
                shown = got["answer"]
            questions.append({**q, "proposed": shown})
        first = any((u["id"], q["qid"]) in blind for q in u["questions"])
        row = {**u, "questions": questions, "_first": first}
        if u["id"] in result["content"]:
            ruling = adjudicated.get(u["id"], {})
            why = ruling.get("content_reason", "")
            advice = ruling.get("content_ruling") or "no ruling"
            row["note"] = (
                f"Flagged content. Reviewer's view: {why} Recommended: {advice}. Press N on "
                "each question to remove this text; check it as usual to keep it."
            )
        rows.append(row)
    rows.sort(key=lambda r: (not r["_first"], r["track"], r["id"]))
    for n, r in enumerate(rows):
        r.pop("_first")
        r["order"] = n
    return rows


def rest_units() -> list[dict]:
    """Desk units for the source-labelled items never on the desk, the label as proposal."""
    on_desk = {u["id"] for u in desk_units_all()}
    metas = {m["id"]: m for p in CANDIDATES.glob("*.meta.jsonl") for m in read_jsonl(p)}
    rest = []
    for path in sorted(CANDIDATES.glob("*.jsonl")):
        if path.name.endswith(".meta.jsonl"):
            continue
        for item in read_jsonl(path):
            source = metas.get(item["id"], {}).get("source", "")
            if (
                item["id"] not in on_desk
                and item.get("gold")
                and any(s in source for s in SPOT_SOURCES)
            ):
                rest.append(item)
    made, _ = units(rest, {}, seed=1)
    # units() draws catches among known items; the rest must show their true labels.
    truth = {i["id"]: i["gold"] for i in rest}
    for u in made:
        for q in u["questions"]:
            value = truth[u["id"]][q["qid"]]
            q["proposed"] = key_of(value)
    return made


def desk_units_all() -> list[dict]:
    return [u for p in sorted(DESK.glob("units-*.json"))
            for u in json.loads(p.read_text())["units"]]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.hakembench.sweep")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    sub.add_parser("adjudication-input")
    d = sub.add_parser("desk")
    d.add_argument("--adjudication", type=Path, required=True)
    args = parser.parse_args(argv)
    WORK.mkdir(parents=True, exist_ok=True)
    desk = desk_units_all()
    rest = rest_units()
    catches = [c for p in sorted(DESK.glob("catches-*.json")) if p.name != "catches-sweep.json"
               for c in json.loads(p.read_text())]  # fmt: skip
    checked = ({json.loads(p.read_text())["item"] for p in CHECKS.glob("*.json")}
               if CHECKS.is_dir() else set())  # fmt: skip
    previous = json.loads(ROUTED.read_text()) if ROUTED.is_file() else {}
    reviews = [load_json_folder(REVIEW1), load_json_folder(REVIEW2)]
    rest_reviews = [load_json_folder(REST / "a"), load_json_folder(REST / "b")]
    for name, pool, revs in (("desk", desk, reviews), ("rest", rest, rest_reviews)):
        missing = [u["id"] for u in pool for r in revs if u["id"] not in r]
        if missing:
            raise SystemExit(f"{len(missing)} {name} answers missing, e.g. {missing[:3]}")
    result = plan(desk, reviews, catches, checked, previous)
    rest_result = plan(rest, rest_reviews, [], set(), {})
    # The rest was spot-checked through its sampled fifth on the desk: no second sample.
    for uid, s in rest_result["status"].items():
        if s in ("sampled", "blind", "shifted"):
            rest_result["status"][uid] = "agreed"
    rest_result["shifted"], rest_result["blind"] = [], []
    if args.command == "plan":
        (WORK / "plan.json").write_text(json.dumps(result, ensure_ascii=False))
        (WORK / "plan_rest.json").write_text(json.dumps(rest_result, ensure_ascii=False))
        summary = {"desk": dict(Counter(result["status"].values())),
                   "rest": dict(Counter(rest_result["status"].values())),
                   "suspect_catches": sum(c["suspect"] for c in result["catches"]),
                   "to_adjudicate": sum(len(v) for v in result["adjudicate"].values())
                   + sum(len(v) for v in rest_result["adjudicate"].values())}  # fmt: skip
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "adjudication-input":
        items = adjudication_units(result, desk) + adjudication_units(rest_result, rest)
        out = WORK / "adjudicate"
        out.mkdir(exist_ok=True)
        for i in range(0, len(items), 60):
            (out / f"batch_{i // 60:02d}.json").write_text(
                json.dumps(items[i : i + 60], ensure_ascii=False)
            )
        print(json.dumps({"units": len(items), "batches": (len(items) + 59) // 60}))
        return 0
    adjudicated = load_json_folder(args.adjudication)
    owner = owner_units(result, desk, adjudicated) + owner_units(rest_result, rest, adjudicated)
    owner.sort(key=lambda r: r["order"] if r["id"] in result["status"] else 10_000 + r["order"])
    for n, r in enumerate(owner):
        r["order"] = n
    backup = DESK / "before_sweep"
    backup.mkdir(exist_ok=True)
    for p in list(DESK.glob("owner-*.json")) + list(DESK.glob("catches-*.json")):
        shutil.copy2(p, backup / p.name)
    for p in DESK.glob("owner-*.json"):
        p.unlink()
    for i in range(0, len(owner), UNITS_PER_DOC):
        (DESK / f"owner-{i // UNITS_PER_DOC}.json").write_text(
            json.dumps({"units": owner[i : i + UNITS_PER_DOC]}, ensure_ascii=False)
        )
    # Catch records: updated shown answers and the suspect mark, plus the shifted ones.
    (DESK / "catches-sweep.json").write_text(
        json.dumps({"catches": result["catches"], "shifted": result["shifted"]}, indent=1)
    )
    rest_ids = {u["id"] for u in rest}
    routed = {uid: s for uid, s in {**result["status"], **rest_result["status"]}.items()
              if s != "agreed"}  # fmt: skip
    ROUTED.write_text(json.dumps(routed, indent=1))
    (WORK / "rest_units.json").write_text(json.dumps(rest, ensure_ascii=False))
    report = {
        "desk_units": len(desk),
        "rest_units": len(rest_ids),
        "owner_units": len(owner),
        "why": dict(Counter(routed.values())),
        "suspect_catches": sum(c["suspect"] for c in result["catches"]),
        "shifted_catches": len(result["shifted"]),
        "blind_questions": len(result["blind"]),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
