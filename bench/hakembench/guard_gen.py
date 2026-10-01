"""Generated guardrails items for HakemBench v1.0.

    python -m bench.hakembench.guard_gen build --writers <folder of w_*.json>
    python -m bench.hakembench.guard_gen review-input
    python -m bench.hakembench.guard_gen route --review <folder>

The 48 source guardrail items are too few to report on, so an AI model
wrote new Turkish items: prompt-injection and jailbreak attempts inside a
product, and hard negatives that share their surface words. `build` removes
exact and near duplicates within the set and every item the training data or
the existing test items already hold (the candidate rules of
bench/hakembench/common.py), splits halves by hash, and writes
candidates/guvenlik-gen.jsonl with the writer's intended label in the meta file
(never as gold). The panel then labels them blind (bench/hakembench/label_items.py),
and a second AI pass rates each text's Turkish and answers blind
(`review-input`). `route` keeps what reads as natural Turkish; an item whose
intended label, blind answer and panel vote agree becomes a desk unit gold can
come from, a seeded fifth of those goes to the owner, and every other item goes
to the owner with the intended label shown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from bench.hakembench.common import half, overlapping, training_files, training_texts
from bench.hakembench.desk import OUT as DESK
from bench.hakembench.desk import options, read_jsonl
from data.decontam.ngrams import MinHashLSH, minhash_signature, tokens, whole_hash

CANDIDATES = Path("bench/hakembench/candidates")
NAME = "guvenlik-gen"
QID = "prompt_injection"
WORK = Path("results/private/step8/guard")
LICENCE = "CC BY 4.0 where rights exist, otherwise CC0; authored by ufak AI for the benchmark"
SAMPLE_SHARE = 0.2
MIN_NATURAL = 2
SEED = 121


def question() -> dict:
    """The track's question, exactly as the source items ask it."""
    first = read_jsonl(CANDIDATES / "guvenlik.jsonl")[0]
    return first["questions"][QID]


def item_id(text: str) -> str:
    return f"guvenlik-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def deduplicated(rows: list[dict]) -> tuple[list[dict], Counter]:
    """Drop exact and MinHash near duplicates within the set, keeping the first of each."""
    lsh, wholes, kept, dropped = MinHashLSH(), set(), [], Counter()
    for n, row in enumerate(rows):
        toks = tokens(row["text"])
        whole = whole_hash(toks)
        if whole in wholes:
            dropped["exact duplicate"] += 1
            continue
        signature = minhash_signature(row["text"])
        if signature is not None and lsh.query(signature):
            dropped["near duplicate"] += 1
            continue
        wholes.add(whole)
        if signature is not None:
            lsh.add(f"g:{n}", signature)
        kept.append(row)
    return kept, dropped


PLACEHOLDER = re.compile(r"\[(ad|e-posta|telefon|bağlantı|hesap no)\]")
QUOTE = re.compile(r"[\"“”'«»]")


def surface(text: str) -> tuple:
    """The surface features a classifier could key on instead of the meaning."""
    size = "short" if len(text) < 120 else "mid" if len(text) < 300 else "long"
    return (bool(PLACEHOLDER.search(text)), "?" in text, size, bool(QUOTE.search(text)))


def matched(rows: list[dict], seed: int = SEED) -> tuple[list[dict], int]:
    """Equal numbers of each label inside every surface cell, so no feature predicts it."""
    rng = random.Random(seed)
    cells: dict[tuple, dict[bool, list[dict]]] = {}
    for r in rows:
        cells.setdefault(surface(r["text"]), {True: [], False: []})[bool(r["label"])].append(r)
    kept = []
    for _cell, sides in sorted(cells.items(), key=str):
        n = min(len(sides[True]), len(sides[False]))
        for side in (True, False):
            kept += rng.sample(sorted(sides[side], key=lambda r: r["text"]), n)
    return kept, len(rows) - len(kept)


def build(writer_rows: list[dict], data_root: Path) -> tuple[list[dict], list[dict], dict]:
    rows, dropped = deduplicated(writer_rows)
    texts = {item_id(r["text"]): r["text"] for r in rows}
    existing = [(f"test:{i['id']}", i["state"]) for i in read_jsonl(CANDIDATES / "guvenlik.jsonl")]
    touched = overlapping(texts, [*existing, *training_texts(training_files(data_root), data_root)])
    q = question()
    items, metas = [], []
    for r in rows:
        uid = item_id(r["text"])
        hit = touched.get(uid)
        if hit is not None and hit["dropped"]:
            dropped["overlaps training or test: " + "+".join(hit["rules"])] += 1
            continue
        items.append({"id": uid, "track": "guvenlik", "state": r["text"], "questions": {QID: q},
                      "gold": {}})  # fmt: skip
        metas.append({"id": uid, "track": "guvenlik", "source": "generated", "licence": LICENCE,
                      "split": half(uid), "writer": "assistant", "family": r["family"],
                      "context": r["context"], "register": r["register"],
                      "intended": bool(r["label"]),
                      "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest()})  # fmt: skip
    by_text = {i["state"]: (i, m) for i, m in zip(items, metas, strict=True)}
    chosen, unmatched = matched([{"text": i["state"], "label": m["intended"]}
                                 for i, m in zip(items, metas, strict=True)])  # fmt: skip
    if unmatched:
        dropped["unmatched surface cell"] += unmatched
    items = [by_text[r["text"]][0] for r in chosen]
    metas = [by_text[r["text"]][1] for r in chosen]
    order = sorted(range(len(items)), key=lambda k: items[k]["id"])
    items, metas = [items[k] for k in order], [metas[k] for k in order]
    summary = {"written": len(writer_rows), "kept": len(items), "dropped": dict(dropped),
               "intended": dict(Counter(str(m["intended"]).lower() for m in metas)),
               "halves": dict(Counter(m["split"] for m in metas))}  # fmt: skip
    return items, metas, summary


SIBLING_JACCARD = 0.5


def template_siblings(items: list[dict]) -> dict[str, str]:
    """Source items that repeat another's template with a word swapped: id to the kept id.

    The 48 source items come from model-written sets whose texts are templates with
    a noun swapped; one item per template stays. Token Jaccard at or above
    SIBLING_JACCARD joins two items; each group keeps its smallest id.
    """
    ids = sorted(i["id"] for i in items)
    words = {i["id"]: set(tokens(i["state"])) for i in items}
    parent = {i: i for i in ids}

    def root(x: str) -> str:
        while parent[x] != x:
            x = parent[x]
        return x

    for n, a in enumerate(ids):
        for b in ids[n + 1 :]:
            shared = len(words[a] & words[b]) / max(1, len(words[a] | words[b]))
            if shared >= SIBLING_JACCARD:
                ra, rb = root(a), root(b)
                parent[max(ra, rb)] = min(ra, rb)
    return {i: root(i) for i in ids if root(i) != i}


def unit(item: dict, proposed: str, order: int) -> dict:
    q = item["questions"][QID]
    return {"id": item["id"], "track": "guvenlik", "order": order, "text": item["state"],
            "questions": [{"qid": QID, "question": q["instructions"], "type": q["type"],
                           "options": options(q), "proposed": proposed}]}  # fmt: skip


def panel_answer(meta: dict) -> str | None:
    """The panel's answer; None when it has no vote or the two judges tie."""
    mean = meta.get("votes", {}).get(QID, {}).get("mean") or {}
    if not mean or mean.get("true", 0.0) == mean.get("false", 0.0):
        return None
    return max(mean, key=mean.get)


def route(items: list[dict], metas: dict[str, dict], review: dict,
          seed: int = SEED) -> tuple[dict[str, str], Counter]:  # fmt: skip
    """Why each item goes where: dropped (unnatural), agreed, sampled or disputed."""
    rng = random.Random(seed)
    status, agreed = {}, []
    for item in sorted(items, key=lambda i: i["id"]):
        uid, meta = item["id"], metas[item["id"]]
        seen = review.get(uid, {})
        if seen.get("natural", 0) < MIN_NATURAL:
            status[uid] = "unnatural"
            continue
        intended = "true" if meta["intended"] else "false"
        blind = seen.get("answer")
        # The blind pass shares the writers' family, so it can send an item to the owner
        # but never counts as an independent vote toward agreement.
        smooth = seen.get("natural", 0) >= 3  # one oddity sends the item to the owner
        if (
            smooth
            and seen.get("flag", "none") == "none"
            and blind == intended == panel_answer(meta)
        ):
            status[uid] = "agreed"
            agreed.append(uid)
        else:
            status[uid] = "disputed"
    for uid in rng.sample(agreed, round(SAMPLE_SHARE * len(agreed))):
        status[uid] = "sampled"
    return status, Counter(status.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.hakembench.guard_gen")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--writers", type=Path, required=True)
    b.add_argument("--data", type=Path, default=Path("data/built"))
    sub.add_parser("review-input")
    sub.add_parser("collapse-source")
    r = sub.add_parser("route")
    r.add_argument("--review", type=Path, required=True)
    args = parser.parse_args(argv)
    WORK.mkdir(parents=True, exist_ok=True)
    if args.command == "build":
        rows = [x for p in sorted(args.writers.glob("w_*.json")) for x in json.loads(p.read_text())]
        items, metas, summary = build(rows, args.data)
        (CANDIDATES / f"{NAME}.jsonl").write_text(
            "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8"
        )
        (CANDIDATES / f"{NAME}.meta.jsonl").write_text(
            "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in metas), encoding="utf-8"
        )
        (WORK / "build_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "collapse-source":
        source = read_jsonl(CANDIDATES / "guvenlik.jsonl")
        siblings = template_siblings(source)
        kept = [i for i in source if i["id"] not in siblings]
        metas_src = read_jsonl(CANDIDATES / "guvenlik.meta.jsonl")
        for m in metas_src:
            if m["id"] in siblings:
                m["template_sibling_of"] = siblings[m["id"]]
        (CANDIDATES / "guvenlik.jsonl").write_text(
            "".join(json.dumps(i, ensure_ascii=False, separators=(",", ":")) + "\n" for i in kept),
            encoding="utf-8",
        )
        (CANDIDATES / "guvenlik.meta.jsonl").write_text(
            "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in metas_src), encoding="utf-8"
        )
        gold = Counter(str(i["gold"][QID]).lower() for i in kept)
        print(json.dumps({"source": len(source), "kept": len(kept), "gold": dict(gold)}))
        return 0
    items = read_jsonl(CANDIDATES / f"{NAME}.jsonl")
    metas = {m["id"]: m for m in read_jsonl(CANDIDATES / f"{NAME}.meta.jsonl")}
    if args.command == "review-input":
        # The declared register goes along, so intended typos are not read as bad Turkish.
        blind = [{"id": i["id"], "text": i["state"], "register": metas[i["id"]]["register"]}
                 for i in items]  # fmt: skip
        random.Random(SEED).shuffle(blind)
        for k in range(3):
            (WORK / f"review_{k}.json").write_text(json.dumps(blind[k::3], ensure_ascii=False))
        print(json.dumps({"items": len(blind), "files": 3}))
        return 0
    review = {}
    for p in sorted(args.review.glob("*.json")):
        review |= json.loads(p.read_text(encoding="utf-8"))
    missing = [i["id"] for i in items if i["id"] not in review]
    unvoted = [i["id"] for i in items if not metas[i["id"]].get("votes", {}).get(QID)]
    if missing or unvoted:
        raise SystemExit(f"missing review answers ({len(missing)}) or panel votes ({len(unvoted)})")
    status, counts = route(items, metas, review)
    shown = {i["id"]: "true" if metas[i["id"]]["intended"] else "false" for i in items}
    units = [unit(i, shown[i["id"]], n) for n, i in enumerate(items)
             if status[i["id"]] != "unnatural"]  # fmt: skip
    (DESK / "units-guard-0.json").write_text(json.dumps({"units": units}, ensure_ascii=False))
    routed_path = Path("results/private/step8/triage.json")
    routed = json.loads(routed_path.read_text())
    routed |= {uid: s for uid, s in status.items() if s in ("sampled", "disputed")}
    routed_path.write_text(json.dumps(routed, indent=1))
    owner = [u for u in units if status[u["id"]] in ("sampled", "disputed")]
    (WORK / "owner_units.json").write_text(json.dumps(owner, ensure_ascii=False))
    text_of = {i["id"]: i["state"] for i in items}
    unnatural = [{"id": uid, "text": text_of[uid], "review": review[uid]}
                 for uid, s in status.items() if s == "unnatural"]  # fmt: skip
    (WORK / "unnatural.json").write_text(json.dumps(unnatural, ensure_ascii=False, indent=1))
    report = {"items": len(items), "routing": dict(counts), "owner_units": len(owner)}
    Path("results/step8/guard_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
