"""Halves by writer unit for the generated tracks.

    python -m bench.hakembench.resplit

Generated items come in sibling groups: every spam message from one writer
request, every guardrail prompt from one attack or benign family. A hash on the
item put siblings on both sides, so a model that learned the public ones had a
head start on the private ones and the public-private gap lost its meaning
where texts are most templated. Each group now goes to one half as a whole
(common.unit_halves), stratified so both halves keep the same mix: by message
kind for spam, by attack or benign family for the guardrails. The split field
of each candidate meta is rewritten; run `desk collect` and the release audit
after it.
"""

from __future__ import annotations

import json
import sys
from collections import Counter

from bench.hakembench.common import unit_halves
from bench.hakembench.desk import CANDIDATES, read_jsonl

# meta file: (unit field, stratum of a meta row, whether ties go to the smaller half overall)
UNITS = {
    "spam": ("request", lambda m: m["kind"], False),
    "guvenlik-gen": ("family", lambda m: m["family"][0], False),
    # The authored moderation and support items; their real web rows stay private.
    "moderasyon": ("request", lambda m: m["kind"], True),
    "sss": ("request", lambda m: m["kind"], True),
}


def resplit(metas: list[dict], unit: str, stratum,
            global_ties: bool = False) -> tuple[list[dict], Counter]:  # fmt: skip
    """The metas with each generated row's split set by its unit's half, and how many moved."""
    sizes: dict[str, tuple[str, int]] = {}
    for m in metas:
        if m.get("source") != "generated":
            continue
        s, n = sizes.get(m[unit], (stratum(m), 0))
        sizes[m[unit]] = (s, n + 1)
    halves = unit_halves(sizes, global_ties=global_ties)
    moved: Counter = Counter()
    out = []
    for m in metas:
        if m.get("source") != "generated":
            out.append(m)
            continue
        new = halves[m[unit]]
        if m.get("split") != new:
            moved[f"{m.get('split')} to {new}"] += 1
        out.append({**m, "split": new})
    return out, moved


def main() -> int:
    report = {}
    for name, (unit, stratum, global_ties) in UNITS.items():
        path = CANDIDATES / f"{name}.meta.jsonl"
        metas, moved = resplit(read_jsonl(path), unit, stratum, global_ties)
        path.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in metas),
                        encoding="utf-8")  # fmt: skip
        generated = {m[unit] for m in metas if m.get("source") == "generated"}
        report[name] = {"unit": unit, "units": len(generated),
                        "halves": dict(Counter(m["split"] for m in metas)),
                        "moved": dict(moved)}  # fmt: skip
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
