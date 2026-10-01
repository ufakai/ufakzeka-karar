"""HakemBench candidates onto the check desk, and the owner's checks back into gold.

    python -m bench.hakembench.desk load --tracks hukuk,arama
    python -m bench.hakembench.desk collect --checks <folder of desk check documents>

`load` turns candidate items into desk units: one text with all its questions,
each question carrying a proposed answer for the owner to confirm or correct.
The proposal is the source's own label where the item has one (the converted
human-labelled splits) and the panel's mean vote otherwise. Human-labelled
tracks are spot-checked: a seeded fifth of their items goes to the desk.
About one unit in twenty, drawn only from items whose truth is known,
becomes a catch item: one of its questions shows a wrong answer, so the owner's
catch rate measures how far the shown answer leads. Which units are catches is
written to a private file and never to the page.

`collect` reads the desk's check documents and writes the checked items with
their gold: a confirmed answer, a corrected one, or nothing for "can't tell" and
"doesn't fit". A catch item's gold is its known truth; the owner's verdict on it
counts only toward the catch rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

CANDIDATES = Path("bench/hakembench/candidates")
OUT = Path("results/private/step8/desk")
CHECKED = Path("results/private/step8/checked")
REPORT = Path("results/step8/catch_report.json")
# Sources whose labels were given by people or by the source's own structure:
# spot-checked. Every other item, generated ones included, is checked in full.
SPOT_SOURCES = ("icgcihan/Turkish_Constutional_Court_Decisions", "PaDaS-Lab/webfaq-retrieval")
SPOT_SHARE = 0.2
# Above this share of disagreements, a spot-checked source is checked in full.
SPOT_LIMIT = 0.1
CATCH_SHARE = 0.05
UNITS_PER_DOC = 40
# What the owner did on a question, as its gold source: confirmed a shown answer,
# chose between two shown answers, corrected a shown answer, or answered blind.
VERDICT_SOURCE = {"agree": "owner_confirmed", "pick": "owner_chose", "change": "owner_corrected",
                  "label": "owner_blind"}  # fmt: skip


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def key_of(value) -> str:
    """Gold and proposals as the desk's option keys: 'true'/'false', a level index, a name."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def options(question: dict) -> list[dict]:
    kind, criteria = question["type"], question["criteria"]
    if kind == "choice":
        return [{"key": k, "label": k, "detail": v or ""} for k, v in criteria.items()]
    if kind == "noul":
        c = criteria or {}
        return [{"key": "true", "label": "Evet", "detail": c.get("true") or ""},
                {"key": "false", "label": "Hayır", "detail": c.get("false") or ""}]  # fmt: skip
    return [{"key": str(i), "label": f"Düzey {i}", "detail": level}
            for i, level in enumerate(criteria)]  # fmt: skip


def proposal(item: dict, meta: dict, qid: str) -> tuple[str, bool]:
    """(the proposed answer, whether it is known truth)."""
    if qid in item.get("gold", {}):
        return key_of(item["gold"][qid]), True
    votes = meta.get("questions") or meta.get("votes") or {}
    mean = votes.get(qid, {}).get("mean") or {}
    if not mean:
        raise ValueError(f"{item['id']}:{qid} has neither gold nor a panel vote")
    return max(mean, key=mean.get), False


def text_of(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def units(items: list[dict], metas: dict[str, dict], seed: int = 1) -> tuple[list, list]:
    """Desk units and the private list of catch items."""
    rng = random.Random(seed)
    chosen = []
    by_track: dict[str, list[dict]] = {}
    for item in items:
        by_track.setdefault(item["track"], []).append(item)
    for _track, rows in sorted(by_track.items()):
        rows = sorted(rows, key=lambda r: r["id"])
        spot = [r for r in rows if any(s in metas.get(r["id"], {}).get("source", "")
                                       for s in SPOT_SOURCES)]  # fmt: skip
        full = [r for r in rows if r not in spot]
        if spot:
            spot = rng.sample(spot, max(1, round(SPOT_SHARE * len(spot))))
        chosen += sorted(spot + full, key=lambda r: r["id"])
    out = []
    for n, item in enumerate(chosen):
        meta = metas.get(item["id"], {})
        questions = []
        for qid, question in item["questions"].items():
            proposed, known = proposal(item, meta, qid)
            questions.append({"qid": qid, "question": question["instructions"],
                              "type": question["type"], "options": options(question),
                              "proposed": proposed, "known": known})  # fmt: skip
        out.append({"id": item["id"], "track": item["track"], "order": n,
                    "text": text_of(item["state"]), "questions": questions})  # fmt: skip
    known_units = [u for u in out if any(q["known"] for q in u["questions"])]
    catches = []
    for u in rng.sample(known_units, round(CATCH_SHARE * len(out)) if known_units else 0):
        q = next(q for q in u["questions"] if q["known"])
        wrong = [o["key"] for o in q["options"] if o["key"] != q["proposed"]]
        shown = rng.choice(wrong)
        catches.append({"unit": u["id"], "qid": q["qid"], "truth": q["proposed"], "shown": shown})
        q["proposed"] = shown
    for u in out:  # the page never learns which proposals are known or shown wrong
        for q in u["questions"]:
            q.pop("known")
    return out, catches


def pooled_catches(desk_units: list[dict], pool: list[dict], loaded: set[str],
                   seed: int = 1) -> tuple[list, list]:  # fmt: skip
    """Catch units drawn from known-truth items not yet on the desk, spread among `desk_units`.

    Generated tracks have no known truth, so their catches come from the
    human-labelled pool; each shows one wrong proposal and is placed at a
    random position, so nothing on the page sets it apart.
    """
    rng = random.Random(seed + 7)
    unused = sorted((i for i in pool if i["id"] not in loaded and i.get("gold")),
                    key=lambda i: i["id"])  # fmt: skip
    extra, _ = units(rng.sample(unused, min(len(unused), round(CATCH_SHARE * len(desk_units)))),
                     {}, seed)  # fmt: skip
    catches = []
    for u in extra:
        q = u["questions"][0]
        wrong = [o["key"] for o in q["options"] if o["key"] != q["proposed"]]
        shown = rng.choice(wrong)
        catches.append({"unit": u["id"], "qid": q["qid"], "truth": q["proposed"], "shown": shown})
        q["proposed"] = shown
    merged = list(desk_units)
    for u in extra:
        merged.insert(rng.randrange(len(merged) + 1), u)
    for n, u in enumerate(merged):
        u["order"] = OFFSET + n
    return merged, catches


# Later loads continue the desk's order after the first batch.
OFFSET = 1000


def collect(items: list[dict], checks: dict[str, dict], catches: list[dict],
            agreed: dict[tuple[str, str], str] | None = None,
            spot: dict[str, dict[str, set[str]]] | None = None) -> tuple[list, dict]:  # fmt: skip
    """Checked items with gold and where each gold label came from, and the catch report.

    `agreed` holds, for questions the owner was never shown, the answer the
    AI reviewer and the panel both gave; it becomes gold marked as such. A catch
    question always takes its known truth, shown to the owner or not, because an
    unshown catch's agreed answer is the wrong one it displayed.

    `spot` maps each spot-checked source to the ids of its items on the desk and
    the rest. When the checked gold on the desk differs from the source's label on
    at most SPOT_LIMIT of its questions, the rest keep the source's label, marked
    `source_label`; above it they are left out and the report asks for a full
    check.
    """
    catch_of = {(c["unit"], c["qid"]): c for c in catches}
    agreed = agreed or {}
    spot = spot or {}
    checked = []
    tally = {"known": Counter(), "shifted": Counter(), "suspect": Counter(), "truth": Counter()}
    for item in items:
        gold, flags, source = {}, {}, {}
        for qid in item["questions"]:
            check = checks.get(f"{item['id']}~{qid}")
            catch = catch_of.get((item["id"], qid))
            judged = check is not None and check.get("flag", "none") == "none"
            if catch is not None and catch.get("retired"):
                # No wrong answer is shown any more. The owner's answer is gold; on a
                # known-truth item it is compared with the truth, split by whether the AI
                # answer shown beside it was right.
                if judged and not catch.get("kind"):
                    ai_right = "ai_right" if check.get("ai") == catch["truth"] else "ai_wrong"
                    said = check["proposed"] if check["verdict"] == "agree" else check.get("answer")
                    agrees = "agrees" if said == catch["truth"] else "differs"
                    tally["truth"][f"{ai_right}:{agrees}"] += 1
                catch = None
            if catch is not None and catch.get("suspect"):
                # Every source gave the shown answer: the label is in doubt, the owner decides
                # and the item leaves the catch rate.
                tally["suspect"]["checked" if check is not None else "unchecked"] += 1
            elif catch is not None:
                shifted = catch.get("kind") == "shifted_level"
                if judged:
                    kept = check["verdict"] == "agree"
                    tally["shifted" if shifted else "known"]["missed" if kept else "caught"] += 1
                if shifted and judged and check["verdict"] != "agree":
                    gold[qid], source[qid] = check["answer"], VERDICT_SOURCE[check["verdict"]]
                elif shifted:
                    gold[qid], source[qid] = catch["truth"], "ai_reviewer_and_panel"
                else:
                    gold[qid], source[qid] = catch["truth"], "known_truth"
                continue
            if check is None:
                if (item["id"], qid) in agreed:
                    value = agreed[(item["id"], qid)]
                    # A pair carries its own gold source; a bare answer has the triage's gold
                    # source.
                    answer, where = value if isinstance(value, tuple) else (value, None)
                    gold[qid], source[qid] = answer, where or "ai_reviewer_and_panel"
                continue
            if check.get("flag", "none") != "none":
                flags[qid] = check["flag"]
            elif check["verdict"] == "agree":
                gold[qid], source[qid] = check["proposed"], VERDICT_SOURCE["agree"]
            else:
                gold[qid], source[qid] = check["answer"], VERDICT_SOURCE[check["verdict"]]
        claim, level = gold.get("dogrulanabilir"), gold.get("kontrol_onceligi")
        if claim is not None and level is not None and (claim == "false") != (level == "0"):
            # The two fact-check answers must agree (level 0 exactly when there is no checkable
            # claim). A blind answer outranks one given with a proposal on screen.
            if source.get("kontrol_onceligi") == "owner_blind":
                gold["dogrulanabilir"] = "false" if level == "0" else "true"
                source["dogrulanabilir"] = "owner_blind_consistency"
            elif claim == "false":
                gold["kontrol_onceligi"] = "0"
                source["kontrol_onceligi"] += "_consistency"
            else:
                gold["dogrulanabilir"] = "false"
                source["dogrulanabilir"] += "_consistency"
        if gold or flags:
            checked.append({**item, "gold": gold, "flags": flags, "gold_source": source})
    spot_report = {}
    by_id = {i["id"]: i for i in checked}
    raw = {i["id"]: i for i in items}
    for name, sets in sorted(spot.items()):
        compared = differ = 0
        for uid in sets["on_desk"]:
            done = by_id.get(uid)
            if done is None:
                continue
            for qid, value in done["gold"].items():
                if done["gold_source"][qid] == "known_truth" or qid not in raw[uid].get("gold", {}):
                    continue
                compared += 1
                differ += value != key_of(raw[uid]["gold"][qid])
        rate = differ / compared if compared else None
        accepted = rate is not None and rate <= SPOT_LIMIT
        rest = [raw[i] for i in sorted(sets["rest"]) if i in raw and i not in by_id]
        if accepted:
            for item in rest:
                gold = {q: key_of(v) for q, v in item.get("gold", {}).items()}
                marks = dict.fromkeys(gold, "source_label")
                if gold:
                    checked.append({**item, "gold": gold, "flags": {}, "gold_source": marks})
        spot_report[name] = {"compared": compared, "differ": differ, "rate": rate,
                             "rest": len(rest), "rest_accepted": accepted}  # fmt: skip
    caught, missed = tally["known"]["caught"], tally["known"]["missed"]
    shifted = tally["shifted"]
    report = {"catch_items": sum(not c.get("suspect") and c.get("kind") != "shifted_level"
                                 for c in catches),
              "caught": caught, "missed": missed,
              "catch_rate": caught / (caught + missed) if caught + missed else None,
              "shifted_level": {"items": sum(c.get("kind") == "shifted_level" for c in catches),
                                "caught": shifted["caught"], "missed": shifted["missed"]},
              "suspect_catches": dict(tally["suspect"]),
              "owner_against_known_truth": dict(tally["truth"]),
              "spot_checks": spot_report}  # fmt: skip
    return checked, report


CONTENT_QID = "__content__"
KNOWLEDGE_QIDS = ("cevap_puani", "ders", "aym_rights", "basvuru_yeri")
BROWSER_REMOVALS = Path("results/private/step8/browser/removals")


def browser_removals(folder: Path = BROWSER_REMOVALS) -> set[str]:
    """Items the owner removed on the HakemBench Browser page."""
    out = set()
    for path in folder.glob("*.json"):
        doc = json.loads(path.read_text())
        out.add(doc.get("item") or doc.get("data", {}).get("item"))
    return out - {None}


def content_removals(checks: dict[str, dict]) -> set[str]:
    """Items the owner chose to remove on a keep-or-remove question."""
    out = set()
    for doc in checks.values():
        if doc["qid"] != CONTENT_QID or doc.get("flag", "none") != "none":
            continue
        said = doc["proposed"] if doc["verdict"] == "agree" else doc["answer"]
        if said == "false":
            out.add(doc["item"])
    return out


def candidate_items(folder: Path = CANDIDATES) -> list[dict]:
    """Every candidate item; an id in two files would be counted twice, so it stops the run."""
    items, seen = [], {}
    for path in sorted(folder.glob("*.jsonl")):
        if path.name.endswith(".meta.jsonl"):
            continue
        for item in read_jsonl(path):
            if item["id"] in seen:
                raise SystemExit(f"{item['id']} is in both {seen[item['id']]} and {path.name}")
            seen[item["id"]] = path.name
            items.append(item)
    return items


def chunks(desk_units: list[dict]) -> list[dict]:
    return [{"units": desk_units[i : i + UNITS_PER_DOC]}
            for i in range(0, len(desk_units), UNITS_PER_DOC)]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.hakembench.desk")
    sub = parser.add_subparsers(dest="command", required=True)
    load_p = sub.add_parser("load")
    load_p.add_argument("--tracks", required=True)
    load_p.add_argument("--seed", type=int, default=1)
    load_p.add_argument("--catch-pool", default="", help="tracks whose unused items supply catches")
    collect_p = sub.add_parser("collect")
    collect_p.add_argument("--checks", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "load":
        items, metas = [], {}
        for track in args.tracks.split(","):
            track_metas = {m["id"]: m for m in read_jsonl(CANDIDATES / f"{track}.meta.jsonl")}
            rows = read_jsonl(CANDIDATES / f"{track}.jsonl")
            # A track that marks a selection (the flagship's 320 of 1,300) sends only it.
            if any("selected" in m for m in track_metas.values()):
                rows = [r for r in rows if track_metas.get(r["id"], {}).get("selected")]
            items += rows
            metas |= track_metas
        desk_units, catches = units(items, metas, args.seed)
        if args.catch_pool:
            loaded = {u["id"] for p in OUT.glob("units-*.json")
                      for u in json.loads(p.read_text())["units"]}  # fmt: skip
            pool = [
                i for t in args.catch_pool.split(",") for i in read_jsonl(CANDIDATES / f"{t}.jsonl")
            ]
            desk_units, more = pooled_catches(desk_units, pool, loaded, args.seed)
            catches += more
        tag = hashlib.sha256(args.tracks.encode()).hexdigest()[:6]
        OUT.mkdir(parents=True, exist_ok=True)
        docs = chunks(desk_units)
        for n, doc in enumerate(docs):
            (OUT / f"units-{tag}-{n}.json").write_text(json.dumps(doc, ensure_ascii=False))
        (OUT / f"catches-{tag}.json").write_text(json.dumps(catches, indent=2))
        print(json.dumps({"units": len(desk_units), "catches": len(catches), "docs": len(docs),
                          "tag": tag}))  # fmt: skip
        return 0
    checks = {}
    for path in sorted(args.checks.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        checks[f"{doc['item']}~{doc['qid']}"] = doc
    items = candidate_items()
    sweep = OUT / "catches-sweep.json"
    if sweep.is_file():  # The sweep's catch records supersede the load's
        data = json.loads(sweep.read_text())
        catches = data["catches"] + data["shifted"]
    else:
        catches = [c for p in sorted(OUT.glob("catches-*.json")) for c in json.loads(p.read_text())]
    agreed, routed = {}, {}
    routed_path = Path("results/private/step8/triage.json")
    if routed_path.is_file():  # Questions never shown to the owner
        routed = json.loads(routed_path.read_text())
        # Every question not on the owner's desk was answered alike by every source;
        # a catch question takes its truth, never the wrong answer it once showed.
        shown = {(u["id"], q["qid"]) for p in OUT.glob("owner-*.json")
                 for u in json.loads(p.read_text())["units"] for q in u["questions"]}  # fmt: skip
        truth = {(c["unit"], c["qid"]): c["truth"] for c in catches}
        for p in sorted(OUT.glob("units-*.json")):
            # Generated guardrail units: the writer's intent and the panel agreed.
            where = "writer_intent_panel_agreed" if p.name.startswith("units-guard") else None
            for u in json.loads(p.read_text())["units"]:
                for q in u["questions"]:
                    pair = (u["id"], q["qid"])
                    if pair not in shown:
                        agreed[pair] = (truth.get(pair, q["proposed"]), where)
    on_desk = {u["id"] for p in OUT.glob("units-*.json")
               for u in json.loads(p.read_text())["units"]}  # fmt: skip
    metas = {m["id"]: m for p in sorted(CANDIDATES.glob("*.meta.jsonl")) for m in read_jsonl(p)}
    spot: dict[str, dict[str, set[str]]] = {}
    for item in items:
        name = next((s for s in SPOT_SOURCES if s in metas.get(item["id"], {}).get("source", "")),
                    None)  # fmt: skip
        if name is not None and item["id"] in routed and item["id"] not in on_desk:
            continue  # routed to the owner: its gold comes from the owner's check
        if name is not None:
            side = "on_desk" if item["id"] in on_desk else "rest"
            spot.setdefault(name, {"on_desk": set(), "rest": set()})[side].add(item["id"])
    # Every routed unit the owner answered, knowledge questions included: those answers are
    # audit data rather than gold, but the unit still counts as checked.
    answered = {key.split("~")[0] for key in checks}
    gold_file = Path("results/private/step8/ensemble/gold.json")
    if gold_file.is_file():
        # AI gold for every question the owner did not answer; the owner's checks
        # still come first inside collect, and the source-label rest needs no acceptance rule.
        agreed = {tuple(key.split("~", 1)): (g["answer"], g["source"])
                  for key, g in json.loads(gold_file.read_text()).items()}  # fmt: skip
        topup = Path("results/private/step8/topup/gold.json")
        if topup.is_file():  # The top-up's gold, decided by topup.py's own rule
            agreed |= {tuple(key.split("~", 1)): (g["answer"], g["source"])
                       for key, g in json.loads(topup.read_text()).items()}  # fmt: skip
        rubric = Path("results/private/step8/rubric_v2/gold.json")
        if rubric.is_file():  # The support questions re-decided under the written rubric
            agreed |= {tuple(key.split("~", 1)): (g["answer"], g["source"])
                       for key, g in json.loads(rubric.read_text()).items()}  # fmt: skip
        spot = {}
        removed = content_removals(checks) | browser_removals()
        items = [i for i in items if i["id"] not in removed]
        # The owner's answers on knowledge questions are audit data, not gold.
        checks = {k: v for k, v in checks.items() if v["qid"] not in KNOWLEDGE_QIDS}
    checked, report = collect(items, checks, catches, agreed, spot)
    if gold_file.is_file():
        report["removed_by_owner"] = sorted(removed)
    # Units routed to the owner with no check yet: the freeze waits for zero.
    report["owner_pending"] = sum(uid not in answered for uid in routed)
    CHECKED.mkdir(parents=True, exist_ok=True)
    by_track: dict[str, list] = {}
    for item in checked:
        by_track.setdefault(item["track"], []).append(item)
    for track, rows in by_track.items():
        (CHECKED / f"{track}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({"checked_items": len(checked), **report}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
