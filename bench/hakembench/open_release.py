"""HakemBench v1.0 as released: every redistributable item, open.

    python -m bench.hakembench.open_release

The frozen halves hold 1,198 public and 1,722 private items. The release
publishes every item whose text may be redistributed: the public half and the
1,148 private items from the same kinds of sources. The 574 private items taken
from web pages (WebFAQ passages and community Q&A, whose page text belongs to
each site) are left out and stay in the private archive.

Writes bench/hakembench/v1.0/open/: items.jsonl, provenance.jsonl, splits.json
(the frozen halves restricted to the open items; the board fits its post-hoc
temperature on part A and scores it on part B) and probes/ (every probe item
whose parent item is open). Items, ids and gold are unchanged. Every probe row carries the
canary and, per choice question, option_order: its options in the order shown, which the
criteria map also holds but some readers reorder.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from bench.hakembench.freeze import CANARY

V1 = Path("bench/hakembench/v1.0")
PRIVATE = Path("results/private/step8/v1.0")
OUT = V1 / "open"
PROBES = ("english", "paraphrase", "permutations", "slots")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def restricted(provenance: dict) -> bool:
    """Web page text the release may not redistribute."""
    source = provenance["source"].lower()
    return ("private only" in provenance["licence"] or "webfaq" in source
            or provenance["source"] == "clips/mqa")  # fmt: skip


def probe_row(row: dict) -> dict:
    """A released probe row: the canary set, and the shown order of each choice question."""
    out = {**row, "canary": CANARY}
    order = {qid: list(q["criteria"]) for qid, q in row["questions"].items()
             if q["type"] == "choice"}  # fmt: skip
    if order:
        out["option_order"] = order
    return out


def build(out: Path = OUT) -> dict:
    public = read_jsonl(V1 / "public.jsonl")
    private = read_jsonl(PRIVATE / "private.jsonl")
    provenance = {r["id"]: r for path in (V1 / "provenance.jsonl", PRIVATE / "provenance.jsonl")
                  for r in read_jsonl(path)}  # fmt: skip
    # The freeze put the canary on the public half only; every released item carries it.
    opened = [{**i, "canary": CANARY} for i in private if not restricted(provenance[i["id"]])]
    held = [i for i in private if restricted(provenance[i["id"]])]
    if any(restricted(provenance[i["id"]]) for i in public):
        raise SystemExit("a public item has a restricted source")
    items = public + opened
    ids = [i["id"] for i in items]
    if len(set(ids)) != len(ids):
        raise SystemExit("an item id repeats")
    out.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items)
    (out / "items.jsonl").write_text(lines, encoding="utf-8")
    rows = [{**provenance[i], "canary": CANARY} for i in ids]
    (out / "provenance.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    # The two parts as item files, for tools that take the halves (bench/probes.py).
    parts = Path("results/step9/open_parts")
    parts.mkdir(parents=True, exist_ok=True)
    for name, chosen in (("part_a", public), ("part_b", opened)):
        (parts / f"{name}.jsonl").write_text(
            "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in chosen), encoding="utf-8"
        )
    splits = {i["id"]: "public" for i in public} | {i["id"]: "private" for i in opened}
    (out / "splits.json").write_text(json.dumps(splits, indent=0), encoding="utf-8")
    open_ids = set(ids)
    probes = {}
    (out / "probes").mkdir(exist_ok=True)
    for kind in PROBES:
        rows, metas = [], []
        for folder, half in ((V1 / "probes", "public"), (PRIVATE / "probes", "private")):
            path = folder / f"{kind}.{half}.jsonl"
            if not path.exists():
                continue
            meta = folder / f"{kind}.{half}.meta.jsonl"
            pairs = zip(path.read_text(encoding="utf-8").splitlines(),
                        meta.read_text(encoding="utf-8").splitlines(), strict=True)  # fmt: skip
            for line, meta_line in pairs:
                row = json.loads(line)
                if row["id"].split("~")[0] in open_ids:
                    rows.append(json.dumps(probe_row(row), ensure_ascii=False,
                                           separators=(",", ":")) + "\n")  # fmt: skip
                    metas.append(meta_line + "\n")
        (out / "probes" / f"{kind}.jsonl").write_text("".join(rows), encoding="utf-8")
        (out / "probes" / f"{kind}.meta.jsonl").write_text("".join(metas), encoding="utf-8")
        probes[kind] = len(rows)
    return {"items": len(items), "from_public": len(public), "from_private": len(opened),
            "left_out": len(held), "left_out_tracks": dict(Counter(i["track"] for i in held)),
            "tracks": dict(Counter(i["track"] for i in items)), "probes": probes}  # fmt: skip


def main() -> int:
    summary = build()
    Path("results/step9/open_release.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
