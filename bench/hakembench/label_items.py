"""Panel labels for candidates that arrive without them, and the flagship's selection.

    python -m bench.hakembench.label_items --track dogrulama --cap-usd 12
    python -m bench.hakembench.label_items --track dogrulama --select 320

The parliament sentences (bench/hakembench/parliament.py) come with no votes.
Every (sentence, question) pair is labelled by the panel exactly as the build
does (data/label/synth.py label_one), journaled and under the step 8 cap, and
the votes are written into the track's meta file, where the check desk reads
its proposals. No model wrote these texts, so every panel judge may judge them.

`--select` then keeps the test items: a seeded draw of `n` sentences, half the
panel calls verifiable claims and half it does not, each half keeping the
candidates' party mix. The selection is marked in the meta file.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

from bench.hakembench import generate as G
from data.label import synth
from data.label.judge import JudgeSpec
from data.label.pilot import load_panel, load_relabel_judge

CANDIDATES = Path("bench/hakembench/candidates")
WORK = Path("results/private/step8/hakembench")
CLAIM_QUESTION = "dogrulanabilir"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def asks_of(items: list[dict]) -> list[synth.Ask]:
    first = items[0]
    return [synth.Ask(qid, first["track"], q) for qid, q in first["questions"].items()]


def label_text(item: dict, track: str) -> dict:
    state = item["state"]
    text = state if isinstance(state, str) else json.dumps(state)
    return {"kind": None, "text": text, "source_id": item["id"], "split": G.LABEL_SPLIT,
            "source": f"hakembench:{track}"}  # fmt: skip


def label(client, panel: list[JudgeSpec], items: list[dict], journal: Path, workers: int) -> dict:
    records = {r["key"]: r for r in read_jsonl(journal)} if journal.is_file() else {}
    jobs = []
    for ask_ in asks_of(items):
        for item in items:
            if f"{ask_.task}:{item['id']}" in records:
                continue
            jobs.append(partial(synth.label_one, client, panel, ask_,
                                label_text(item, item["track"])))  # fmt: skip
    for record in G.journaled(jobs, journal, workers):
        records[record["key"]] = record
    return records


def with_votes(metas: list[dict], items: list[dict], records: dict) -> list[dict]:
    asks = asks_of(items)
    return [{**m, "votes": {a.task: G.votes_of(records.get(f"{a.task}:{m['id']}")) for a in asks}}
            for m in metas]  # fmt: skip


def select(metas: list[dict], n: int, seed: int = 1) -> set[str]:
    """Half claims, half not, by the panel's mean vote; each half keeps the party mix."""
    rng = random.Random(seed)
    halves: dict[bool, list[dict]] = {True: [], False: []}
    for m in metas:
        mean = m.get("votes", {}).get(CLAIM_QUESTION, {}).get("mean")
        if mean:
            halves[mean.get("true", 0.0) >= 0.5].append(m)
    chosen: set[str] = set()
    for claim, rows in halves.items():
        want = n // 2 if claim else n - n // 2
        by_party: dict[str, list[dict]] = defaultdict(list)
        for m in sorted(rows, key=lambda r: r["id"]):
            by_party[m.get("party") or ""].append(m)
        total = len(rows)
        quota = {p: round(want * len(v) / total) for p, v in by_party.items()} if total else {}
        for party, members in sorted(by_party.items()):
            chosen |= {m["id"] for m in rng.sample(members, min(len(members), quota[party]))}
        leftovers = [m for m in rows if m["id"] not in chosen]
        short = want - sum(1 for m in rows if m["id"] in chosen)
        chosen |= {m["id"] for m in rng.sample(leftovers, max(0, min(short, len(leftovers))))}
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.hakembench.label_items")
    parser.add_argument("--track", required=True)
    parser.add_argument("--cap-usd", type=float, default=G.CAP_USD)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--select", type=int, default=0)
    parser.add_argument("--with-b", action="store_true",
                        help="add judge B, a third vote from the panel's two families, on texts "
                             "none of them wrote (the top-up)")  # fmt: skip
    args = parser.parse_args(argv)
    items = read_jsonl(CANDIDATES / f"{args.track}.jsonl")
    meta_path = CANDIDATES / f"{args.track}.meta.jsonl"
    metas = read_jsonl(meta_path)
    if args.select:
        chosen = select(metas, args.select)
        write_jsonl(meta_path, [{**m, "selected": m["id"] in chosen} for m in metas])
        print(json.dumps({"selected": len(chosen)}))
        return 0
    panel, _generator, _cheap = load_panel()
    if args.with_b:
        panel = [*panel, load_relabel_judge()]
    client = G.make_client(G.LEDGER_DIR, args.cap_usd)
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        records = label(client, panel, items, WORK / f"{args.track}_labels.jsonl", args.workers)
    except G.STOPPING as error:
        print(f"stopped: {error}; restart to resume", file=sys.stderr)
        return 1
    write_jsonl(meta_path, with_votes(metas, items, records))
    print(json.dumps({"pairs": len(records)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
