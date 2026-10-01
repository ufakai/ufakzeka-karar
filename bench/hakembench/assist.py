"""The AI-assisted check desk.

    python -m bench.hakembench.assist --research <folder of research answers>

The owner asked to see an AI model's researched answer while checking. The
model under test is never shown. For every shown question the desk now carries
the pipeline's own proposal (the panel's vote, the source label or the guardrail
writer's intent, never an AI adjudicator's answer) and the research pass's
answer with a one-sentence Turkish reason and its sources. The page shows one
answer when they agree and both, reason hidden until the pick, when they differ.

A blind control keeps the shown answers honest: a seeded quarter of the
questions on the certification-sample units, per track and type, is shown with
neither proposal nor AI answer. The owner's agreement there against the confirm
rate elsewhere is the pull of the shown answer, and the certification bound is
read on the control only. The deliberately wrong catch answers are retired;
known-truth items are scored against the truth instead (desk.collect). Units
never shown before join the desk when the research answer differs from the
agreed one.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from bench.hakembench.desk import OUT as DESK
from bench.hakembench.desk import UNITS_PER_DOC

ROUTED = Path("results/private/step8/triage.json")
REST = Path("results/private/step8/sweep/rest_units.json")
CONTROL = Path("results/private/step8/sweep/blind_control.json")
REPORT = Path("results/step8/assist_report.json")
CONTROL_SHARE = 0.25
SEED = 122
# A source that is the item's own origin is the source label again, not corroboration.
ORIGIN_DOMAINS = {"hukuk": "kararlarbilgibankasi.anayasa.gov.tr"}


def load_json_folder(folder: Path) -> dict:
    out: dict = {}
    for path in sorted(folder.glob("*.json")):
        out |= json.loads(path.read_text(encoding="utf-8"))
    return out


def proposals(desk_units: list[dict], catches: list[dict]) -> dict[tuple[str, str], str]:
    """The pipeline's own answer per question; a catch shows its truth, never a wrong one."""
    shown = {(u["id"], q["qid"]): q["proposed"] for u in desk_units for q in u["questions"]}
    for c in catches:
        if not c.get("kind"):
            shown[(c["unit"], c["qid"])] = c["truth"]
    return shown


def ai_of(research: dict, uid: str, qid: str) -> dict | None:
    got = research.get(uid, {}).get(qid)
    if not got or got.get("answer") is None or got.get("flag", "none") != "none":
        return None
    return {"answer": got["answer"], "confidence": got.get("confidence"),
            "reason": got.get("reason_tr", ""), "sources": got.get("sources", [])[:3]}  # fmt: skip


def differs(unit: dict, proposal: dict, research: dict) -> bool:
    for q in unit["questions"]:
        ai = ai_of(research, unit["id"], q["qid"])
        if ai is not None and ai["answer"] != proposal[(unit["id"], q["qid"])]:
            return True
    return False


def blind_control(owner: list[dict], routed: dict[str, str], seed: int = SEED) -> set:
    """A seeded share of the sample units' questions per track and type, shown with nothing."""
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for u in owner:
        if routed.get(u["id"]) != "sampled":
            continue
        for q in u["questions"]:
            if q["proposed"] is not None:
                strata.setdefault((u["track"], q["type"]), []).append((u["id"], q["qid"]))
    chosen = set()
    for _key, pairs in sorted(strata.items()):
        chosen |= set(rng.sample(sorted(pairs), round(CONTROL_SHARE * len(pairs))))
    return chosen


def assist(owner: list[dict], pool: list[dict], proposal: dict, research: dict,
           routed: dict[str, str], seed: int = SEED) -> tuple[list[dict], dict, set]:  # fmt: skip
    """The new desk, the new routing and the control questions."""
    routed = dict(routed)
    on_desk = {u["id"] for u in owner}
    added = [u for u in pool if u["id"] not in on_desk and u["id"] not in routed
             and differs(u, proposal, research)]  # fmt: skip
    for u in added:
        routed[u["id"]] = "research_disputed"
    control = blind_control(owner, routed, seed)
    out = []
    for u in owner + [{**u, "order": 10_000 + n} for n, u in enumerate(added)]:
        questions = []
        for q in u["questions"]:
            pair = (u["id"], q["qid"])
            if q["proposed"] is None and u["id"] in on_desk:  # the 30 blind priority questions
                questions.append({k: v for k, v in q.items() if k != "ai"})
            elif pair in control:
                questions.append({**{k: v for k, v in q.items() if k != "ai"}, "proposed": None})
            else:
                ai = ai_of(research, *pair)
                row = {**q, "proposed": proposal[pair]}
                row.pop("ai", None)
                if ai is not None:
                    row["ai"] = ai
                questions.append(row)
        out.append({**u, "questions": questions})
    out.sort(key=lambda u: u["order"])
    for n, u in enumerate(out):
        u["order"] = n
    return out, routed, control


KEEP_WHOLE = ("sampled", "checked", "content")


def settled_pairs(detail: dict, proposal: dict, research: dict) -> set[tuple[str, str]]:
    """Questions every source answered alike: the proposal, both blind passes and the research."""
    out = set()
    for uid, info in detail.items():
        for qid, seen in info["questions"].items():
            pair = (uid, qid)
            if pair not in proposal:
                continue
            want = proposal[pair]
            passes = all(a.get("flag", "none") == "none" and a.get("answer") == want
                         for a in seen["answers"])  # fmt: skip
            found = research.get(uid, {}).get(qid, {})
            if passes and found.get("flag", "none") == "none" and found.get("answer") == want:
                out.add(pair)
    return out


def prune(desk: list[dict], routed: dict[str, str], settled: set) -> tuple[list[dict], dict]:
    """Only what needs the owner: a settled question leaves the desk unless its unit is a
    certification-sample, already-checked or content unit, or the question is blind; a unit
    left with nothing leaves the routing, so its questions take the agreed answer."""
    routed = dict(routed)
    out = []
    for u in desk:
        whole = routed.get(u["id"]) in KEEP_WHOLE
        kept = [q for q in u["questions"] if whole or q["proposed"] is None
                or (u["id"], q["qid"]) not in settled]  # fmt: skip
        if kept:
            out.append({**u, "questions": kept})
        else:
            routed.pop(u["id"], None)
    for n, u in enumerate(out):
        u["order"] = n
    return out, routed


def self_sourced(research: dict, tracks: dict[str, str]) -> Counter:
    counts = Counter()
    for uid, answers in research.items():
        domain = ORIGIN_DOMAINS.get(tracks.get(uid, ""))
        for a in answers.values():
            sources = a.get("sources") or []
            if sources:
                counts["researched"] += 1
                counts["own_origin"] += bool(domain) and any(domain in s for s in sources)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.hakembench.assist")
    parser.add_argument("--research", type=Path, required=True)
    args = parser.parse_args(argv)
    research = load_json_folder(args.research)
    owner = [u for p in sorted(DESK.glob("owner-*.json"))
             for u in json.loads(p.read_text())["units"]]  # fmt: skip
    pool = [u for p in sorted(DESK.glob("units-*.json"))
            for u in json.loads(p.read_text())["units"]]  # fmt: skip
    pool += json.loads(REST.read_text())
    sweep = DESK / "catches-sweep.json"
    data = json.loads(sweep.read_text())
    proposal = proposals(pool, data["catches"])
    missing = [u["id"] for u in pool if u["id"] not in research]
    if missing:
        raise SystemExit(f"{len(missing)} units have no research answer, e.g. {missing[:3]}")
    routed = json.loads(ROUTED.read_text())
    new, routed, control = assist(owner, pool, proposal, research, routed)
    sweep_dir = ROUTED.parent / "sweep"
    detail = {}
    for name in ("plan.json", "plan_rest.json"):
        detail |= json.loads((sweep_dir / name).read_text())["detail"]
    new, routed = prune(new, routed, settled_pairs(detail, proposal, research))
    # Catches retired: nothing wrong is shown any more.
    data["catches"] = [{**c, "retired": True} for c in data["catches"]]
    data["shifted"] = [{**c, "retired": True} for c in data["shifted"]]
    sweep.write_text(json.dumps(data, indent=1))
    for p in DESK.glob("owner-*.json"):
        p.unlink()
    for i in range(0, len(new), UNITS_PER_DOC):
        (DESK / f"owner-{i // UNITS_PER_DOC}.json").write_text(
            json.dumps({"units": new[i : i + UNITS_PER_DOC]}, ensure_ascii=False)
        )
    ROUTED.write_text(json.dumps(routed, indent=1))
    CONTROL.write_text(json.dumps(sorted(control), indent=0))
    shown = Counter()
    for u in new:
        for q in u["questions"]:
            if q["proposed"] is None:
                shown["blind control" if (u["id"], q["qid"]) in control else "blind"] += 1
            elif "ai" not in q:
                shown["proposal only"] += 1
            else:
                shown["agree" if q["ai"]["answer"] == q["proposed"] else "differ"] += 1
    tracks = {u["id"]: u["track"] for u in pool}
    report = {"owner_units": len(new), "questions": sum(shown.values()), "shown": dict(shown),
              "research_disputed_added": sum(s == "research_disputed" for s in routed.values()),
              "research_sources": dict(self_sourced(research, tracks))}  # fmt: skip
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
