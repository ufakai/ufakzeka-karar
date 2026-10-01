"""AI-ensemble gold with a blind human audit.

    python -m bench.hakembench.ensemble adjudication-input --second <folder>
    python -m bench.hakembench.ensemble gold --second <folder> --adjudicated <folder>

The owner chose AI-ensemble gold over checking 1,229 questions by hand. For every
question of every text the available answers are gathered: the pipeline's
proposal (the panel's vote, the source label, or the guardrail writer's intent
with the panel's own vote beside it), the two earlier blind passes and the
research pass of one AI model family, and a blind researched pass of a
second model. When all of them agree, that answer is gold
("ai_ensemble_unanimous"). Otherwise an adjudicator of the second model sees the
distinct answers and both passes' reasons unlabelled and decides
("ai_ensemble_adjudicated").

The owner answers a stratified random sample blind (no proposal, no AI note);
those answers are gold for their questions ("owner_blind") and measure the
ensemble's error against the gold it would otherwise have had.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from bench.hakembench.assist import load_json_folder, proposals
from bench.hakembench.desk import OUT as DESK

WORK = Path("results/private/step8/ensemble")
SWEEP = Path("results/private/step8/sweep")
RESEARCH = Path("results/private/step8/research")
SECOND = Path("results/private/step8/second_pass")
GUARD_META = Path("bench/hakembench/candidates/guvenlik-gen.meta.jsonl")
SEED = 123


def pool() -> list[dict]:
    units = [
        u for p in sorted(DESK.glob("units-*.json")) for u in json.loads(p.read_text())["units"]
    ]
    return units + json.loads((SWEEP / "rest_units.json").read_text())


def panel_of_guard() -> dict[tuple[str, str], str]:
    """The panel's own answer on each generated guardrail item (its proposal is the intent)."""
    out = {}
    for line in GUARD_META.read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        mean = m.get("votes", {}).get("prompt_injection", {}).get("mean") or {}
        if mean and mean.get("true") != mean.get("false"):
            out[(m["id"], "prompt_injection")] = max(mean, key=mean.get)
    return out


def answers(units: list[dict], proposal: dict, detail: dict, research: dict, second: dict,
            guard_panel: dict) -> dict[tuple[str, str], list[str | None]]:  # fmt: skip
    """Every available answer per question; a flagged or missing answer counts as None."""

    def said(entry: dict | None) -> str | None:
        if not entry or entry.get("flag", "none") != "none":
            return None
        return entry.get("answer")

    out = {}
    for u in units:
        for q in u["questions"]:
            pair = (u["id"], q["qid"])
            got = [proposal[pair]]
            seen = detail.get(u["id"], {}).get("questions", {}).get(q["qid"])
            if seen is not None:
                got += [said(a) for a in seen["answers"]]
            if pair in guard_panel:
                got.append(guard_panel[pair])
            got += [said(research.get(u["id"], {}).get(q["qid"])),
                    said(second.get(u["id"], {}).get(q["qid"]))]  # fmt: skip
            out[pair] = got
    return out


def unanimous(got: list[str | None]) -> bool:
    return None not in got and len(set(got)) == 1


def adjudication_units(units: list[dict], votes: dict, research: dict, second: dict,
                       seed: int = SEED) -> list[dict]:  # fmt: skip
    """Text, question, options, the distinct answers and both passes' reasons, unlabelled."""
    rng = random.Random(seed)
    out = []
    for u in units:
        questions = []
        for q in u["questions"]:
            pair = (u["id"], q["qid"])
            if unanimous(votes[pair]):
                continue
            distinct = sorted({a for a in votes[pair] if a is not None})
            rng.shuffle(distinct)
            reasons = [
                r
                for r in (
                    research.get(u["id"], {}).get(q["qid"], {}).get("reason_tr"),
                    second.get(u["id"], {}).get(q["qid"], {}).get("reason_tr"),
                )
                if r
            ]
            rng.shuffle(reasons)
            questions.append({"qid": q["qid"], "question": q["question"], "type": q["type"],
                              "options": q["options"], "candidates": distinct,
                              "reasons": reasons})  # fmt: skip
        if questions:
            out.append({"id": u["id"], "track": u["track"], "text": u["text"],
                        "questions": questions})  # fmt: skip
    return out


PRIORITY = "kontrol_onceligi"
# Questions whose answer turns on subject knowledge (a student's physics answer, a lesson's
# subject, which constitutional right or which court): gold from the models' blind and
# researched passes or the source's own label; the owner's answers there are audit only.
KNOWLEDGE_QIDS = ("cevap_puani", "ders", "aym_rights", "basvuru_yeri")
AUDIT = {"adjudicated": 100, "unanimous": 70, "known_truth": 18}


def allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder split of `total` over tracks in proportion to their counts."""
    size = sum(counts.values())
    if not size:
        return {}
    exact = {t: total * n / size for t, n in counts.items()}
    got = {t: min(counts[t], int(x)) for t, x in exact.items()}
    for t in sorted(exact, key=lambda t: exact[t] - int(exact[t]), reverse=True):
        if sum(got.values()) >= min(total, size):
            break
        if got[t] < counts[t]:
            got[t] += 1
    return got


def audit_sample(votes: dict, tracks: dict[str, str], known: set, exclude: set,
                 sizes: dict[str, int] = AUDIT, seed: int = SEED) -> dict[str, list]:  # fmt: skip
    """The blind audit: adjudicated and unanimous strata spread over tracks in
    proportion, plus known-truth questions that score the auditor. Questions the owner
    has already seen, and the priority scale (audited apart), are outside the frame."""
    rng = random.Random(seed)
    frame = {pair: got for pair, got in votes.items()
             if pair not in exclude and pair[1] != PRIORITY}  # fmt: skip
    strata = {"known_truth": [pair for pair in sorted(frame) if pair in known]}
    rest = [pair for pair in sorted(frame) if pair not in known]
    strata["unanimous"] = [pair for pair in rest if unanimous(frame[pair])]
    strata["adjudicated"] = [pair for pair in rest if not unanimous(frame[pair])]
    chosen = {}
    for name, pairs in strata.items():
        by_track: dict[str, list] = {}
        for pair in pairs:
            by_track.setdefault(tracks[pair[0]], []).append(pair)
        take = allocate({t: len(v) for t, v in by_track.items()}, sizes[name])
        chosen[name] = sorted(
            q for t, v in sorted(by_track.items()) for q in rng.sample(v, take[t])
        )
    return chosen


CONTENT_QID = "__content__"
# Texts the research passes of both models flagged, in the flags file: promotion of
# illegal services, and texts too broken to judge. Each becomes one keep-or-remove question.
FLAGS = WORK / "content_flags.json"
KEEP_OPTIONS = [
    {"key": "true", "label": "Kalsın", "detail": "Metin benchmark'ta kalır."},
    {"key": "false", "label": "Çıkarılsın", "detail": "Metin benchmark'tan çıkarılır."},
]


def content_unit(unit: dict, why: str, order: int) -> dict:
    return {"id": unit["id"], "track": unit["track"], "order": order, "text": unit["text"],
            "note": why + " Recommended: remove. J confirms the removal; F then 1 keeps it.",
            "questions": [{"qid": CONTENT_QID, "question": "Bu metin HakemBench'te kalsın mı?",
                           "type": "noul", "options": KEEP_OPTIONS,
                           "proposed": "false"}]}  # fmt: skip


def audit_desk(units: list[dict], chosen: list[tuple[str, str]], priority_left: list,
               content: dict[str, str]) -> list[dict]:  # fmt: skip
    """Remaining blind priority questions first, then the audit, then content decisions."""
    by_id = {u["id"]: u for u in units}
    wanted: dict[str, list[str]] = {}
    for uid, qid in list(priority_left) + list(chosen):
        wanted.setdefault(uid, []).append(qid)
    out = []
    for uid, qids in wanted.items():
        u = by_id[uid]
        questions = [{**{k: v for k, v in q.items() if k != "ai"}, "proposed": None}
                     for q in u["questions"] if q["qid"] in qids]  # fmt: skip
        out.append({"id": uid, "track": u["track"], "order": len(out), "text": u["text"],
                    "questions": questions})  # fmt: skip
    for uid, why in sorted(content.items()):
        if uid in by_id:
            out.append(content_unit(by_id[uid], why, len(out)))
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if not n:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def audit_report(chosen: dict[str, list], owner: dict[tuple[str, str], str | None],
                 gold: dict[str, dict], stratum_sizes: dict[str, int]) -> dict:  # fmt: skip
    """The owner's blind disagreement with the AI gold per stratum, and pooled by stratum weight.
    A "can't tell" or "doesn't fit" answer marks the item ambiguous, not a disagreement."""
    out, pooled = {}, 0.0
    total = sum(stratum_sizes[s] for s in ("unanimous", "adjudicated"))
    for name, pairs in chosen.items():
        answered = [p for p in map(tuple, pairs) if owner.get(p) is not None]
        differ = sum(owner[p] != gold.get(f"{p[0]}~{p[1]}", {}).get("answer") for p in answered)
        rate = differ / len(answered) if answered else None
        unsure = sum(1 for p in map(tuple, pairs) if p in owner and owner[p] is None)
        out[name] = {
            "drawn": len(pairs),
            "answered": len(answered),
            "ambiguous": unsure,
            "differ": differ,
            "rate": rate,
            "interval": wilson(differ, len(answered)),
        }
        if name in ("unanimous", "adjudicated") and rate is not None:
            pooled += stratum_sizes[name] / total * rate
    out["pooled_by_stratum_weight"] = round(pooled, 4)
    out["stratum_sizes"] = stratum_sizes
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.hakembench.ensemble")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("adjudication-input")
    a.add_argument("--second", type=Path, required=True)
    dk = sub.add_parser("desk")
    dk.add_argument("--checks", type=Path, required=True)
    rp = sub.add_parser("report")
    rp.add_argument("--checks", type=Path, required=True)
    au = sub.add_parser("audit")
    au.add_argument("--second", type=Path, required=True)
    au.add_argument("--checks", type=Path, required=True, help="the owner's saved checks so far")
    g = sub.add_parser("gold")
    g.add_argument("--second", type=Path, required=True)
    g.add_argument("--adjudicated", type=Path, required=True)
    args = parser.parse_args(argv)
    units = pool()
    catches = json.loads((DESK / "catches-sweep.json").read_text())["catches"]
    proposal = proposals(units, catches)
    detail = {}
    for name in ("plan.json", "plan_rest.json"):
        detail |= json.loads((SWEEP / name).read_text())["detail"]
    if args.command == "desk":
        return write_desk(units, args.checks)
    if args.command == "report":
        owner = {}
        for p in args.checks.glob("*.json"):
            c = json.loads(p.read_text())
            if c["verdict"] == "label":
                owner[(c["item"], c["qid"])] = (
                    c["answer"] if c.get("flag", "none") == "none" else None
                )
        gold = json.loads((WORK / "gold.json").read_text())
        sizes = Counter(g["source"] for g in gold.values())
        report = audit_report(json.loads((WORK / "audit.json").read_text()), owner, gold,
                              {"unanimous": sizes["ai_ensemble_unanimous"],
                               "adjudicated": sizes["ai_ensemble_adjudicated"]})  # fmt: skip
        # The adjudicator read the owner's priority answers as its reference, so the priority
        # check compares the owner with the two blind researched passes instead.
        blind = [tuple(b) for b in json.loads((SWEEP / "plan.json").read_text())["blind"]]
        report["priority_blind"] = {}
        for name, folder in (("research_pass", RESEARCH), ("second_model_pass", SECOND)):
            said = load_json_folder(folder)
            as_gold = {f"{u}~{q}": {"answer": said.get(u, {}).get(q, {}).get("answer")}
                       for u, q in blind}  # fmt: skip
            report["priority_blind"][name] = audit_report(
                {"priority": blind}, owner, as_gold, {"unanimous": 1, "adjudicated": 1}
            )["priority"]
        out = Path("results/step8/audit_report.json")
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 0
    research = load_json_folder(RESEARCH)
    second = load_json_folder(args.second)
    missing = [u["id"] for u in units if u["id"] not in second]
    if missing:
        raise SystemExit(
            f"{len(missing)} units have no answer from the second model, e.g. {missing[:3]}"
        )
    votes = answers(units, proposal, detail, research, second, panel_of_guard())
    WORK.mkdir(parents=True, exist_ok=True)
    if args.command == "adjudication-input":
        items = adjudication_units(units, votes, research, second)
        out = WORK / "adjudicate"
        out.mkdir(exist_ok=True)
        for i in range(0, len(items), 60):
            (out / f"batch_{i // 60:02d}.json").write_text(
                json.dumps(items[i : i + 60], ensure_ascii=False)
            )
        counts = Counter(unanimous(v) for v in votes.values())
        print(json.dumps({"questions": len(votes), "unanimous": counts[True],
                          "to_adjudicate": counts[False], "units": len(items)}))  # fmt: skip
        return 0
    if args.command == "audit":
        seen = {(c["item"], c["qid"]) for c in (json.loads(p.read_text())
                for p in args.checks.glob("*.json"))}  # fmt: skip
        known = {(c["unit"], c["qid"]) for c in catches}
        tracks = {u["id"]: u["track"] for u in units}
        chosen = audit_sample(votes, tracks, known, seen)
        (WORK / "audit.json").write_text(json.dumps(chosen, indent=1))
        print(json.dumps({k: len(v) for k, v in chosen.items()}))
        return 0
    adjudicated = load_json_folder(args.adjudicated)
    gold, sources = {}, Counter()
    for pair, got in votes.items():
        if unanimous(got):
            gold[f"{pair[0]}~{pair[1]}"] = {"answer": got[0], "source": "ai_ensemble_unanimous"}
            sources["ai_ensemble_unanimous"] += 1
            continue
        ruled = adjudicated.get(pair[0], {}).get(pair[1], {})
        if ruled.get("answer") is None:
            sources["no gold (adjudicator gave none)"] += 1
            continue
        gold[f"{pair[0]}~{pair[1]}"] = {
            "answer": ruled["answer"],
            "source": "ai_ensemble_adjudicated",
            "confidence": ruled.get("confidence"),
        }
        sources["ai_ensemble_adjudicated"] += 1
    joint = WORK / "joint_out.json"
    if joint.is_file():
        # Fact-check items whose two gold answers contradicted each other, decided together
        # under the consistency rule (level 0 exactly when there is no checkable claim).
        for uid, decided in json.loads(joint.read_text()).items():
            for qid, ruled in decided.items():
                gold[f"{uid}~{qid}"] = {
                    "answer": ruled["answer"],
                    "source": "ai_ensemble_adjudicated",
                    "confidence": ruled.get("confidence"),
                    "joint": True,
                }
                sources["joint re-decision"] += 1
    redecided = WORK / "redecided"
    if redecided.is_dir():
        # The subjective questions re-decided under the owner's written reading.
        for uid, decided in load_json_folder(redecided).items():
            for qid, ruled in decided.items():
                # Grading a student's answer needs subject knowledge, not the owner's reading
                # of a rubric (the owner's note): the models' gold stands there.
                if ruled.get("answer") is None or qid in KNOWLEDGE_QIDS:
                    continue
                gold[f"{uid}~{qid}"] = {"answer": ruled["answer"], "source": "ai_owner_reading",
                                        "confidence": ruled.get("confidence")}  # fmt: skip
                sources["owner reading re-decision"] += 1
    # The Constitutional Court's own record of the right a case concerns is the truth for
    # that question; no AI answer overrides it.
    for line in Path("bench/hakembench/candidates/hukuk.jsonl").read_text().splitlines():
        item = json.loads(line)
        for qid, value in item.get("gold", {}).items():
            gold[f"{item['id']}~{qid}"] = {"answer": value, "source": "source_label"}
            sources["court's own label"] += 1
    (WORK / "gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=0))
    print(json.dumps(dict(sources)))
    return 0


def write_desk(units: list[dict], checks_dir: Path) -> int:
    chosen = json.loads((WORK / "audit.json").read_text())
    audit = [tuple(p) for v in chosen.values() for p in v]
    # Shuffled, so an item's place on the desk says nothing about its stratum.
    random.Random(SEED).shuffle(audit)
    seen = {(c["item"], c["qid"]) for c in (json.loads(p.read_text())
            for p in checks_dir.glob("*.json"))}  # fmt: skip
    blind = [tuple(b) for b in json.loads((SWEEP / "plan.json").read_text())["blind"]]
    left = [b for b in blind if b not in seen]
    earlier = {u["id"]: u.get("note", "") for p in DESK.glob("owner-*.json")
               for u in json.loads(p.read_text())["units"] if u.get("note")}  # fmt: skip
    flags = json.loads(FLAGS.read_text())
    content = dict.fromkeys(
        flags["promotion"], "Flagged: it promotes an illegal or grey-market service."
    )
    content |= dict.fromkeys(
        flags["broken"], "Flagged: the text is broken or too garbled to judge."
    )
    content |= {uid: note.split(" Recommended")[0] for uid, note in earlier.items()}
    desk = audit_desk(units, audit, left, content)
    for p in DESK.glob("owner-*.json"):
        p.unlink()
    for i in range(0, len(desk), 40):
        (DESK / f"owner-{i // 40}.json").write_text(json.dumps({"units": desk[i : i + 40]},
                                                               ensure_ascii=False))  # fmt: skip
    routed = {u["id"]: "audit" for u in desk}
    Path("results/private/step8/triage.json").write_text(json.dumps(routed, indent=1))
    print(
        json.dumps(
            {
                "units": len(desk),
                "questions": sum(len(u["questions"]) for u in desk),
                "priority_left": len(left),
                "audit": len(audit),
                "content": sum(u["questions"][0]["qid"] == CONTENT_QID for u in desk),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
