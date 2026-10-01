"""The data file behind the HakemBench Browser page.

    python -m bench.hakembench.browser

Reads the collected items (results/private/step8/checked/, written by
`desk collect`), the candidates they came from and the owner's desk
removals, and writes one row per text: its track and half, the text, and every
question with its gold answer as the option's label and where the gold came
from. Texts the owner removed on the desk are listed too, marked, so the page
shows every text once.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from bench.hakembench.desk import (
    CANDIDATES,
    CHECKED,
    browser_removals,
    content_removals,
    read_jsonl,
)

OUT = Path("results/private/step8/browser/data.json")
GOLD = Path("results/private/step8/ensemble/gold.json")
LEGAL = Path("results/private/step8/legal_review.json")
CHECKS = Path("results/private/step8/checks/checks")


def label(question: dict, key) -> str | None:
    if key is None:
        return None
    if question["type"] == "noul":
        return "Evet" if str(key).lower() == "true" else "Hayır"
    if question["type"] == "score":
        levels = question["criteria"]
        k = int(key)
        return f"Düzey {k}: {levels[k]}" if 0 <= k < len(levels) else str(key)
    return str(key)


def text_of(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def rows(candidates: list[dict], metas: dict, checked: dict, removed: set,
         confidence: dict[str, str] | None = None,
         legal: dict | None = None, kept_rows: set | None = None) -> list[dict]:  # fmt: skip
    out = []
    for item in candidates:
        got = checked.get(item["id"])
        if got is None and item["id"] not in removed and item["id"] not in (kept_rows or set()):
            continue
        meta = metas.get(item["id"], {})
        questions = []
        for qid, q in item["questions"].items():
            gold = (got or {}).get("gold", {}).get(qid)
            source = (got or {}).get("gold_source", {}).get(qid)
            low = (
                (confidence or {}).get(f"{item['id']}~{qid}") == "low"
                and source
                and source.startswith("ai")
            )
            questions.append({"qid": qid, "question": q["instructions"], "answer": label(q, gold),
                              "source": source, "low": bool(low)})  # fmt: skip
        half = meta.get("half") or meta.get("split") or ""
        flag = (legal or {}).get(item["id"])
        note = f"Legal review: {flag['reason']}" if flag else ""
        out.append({"id": item["id"], "track": item["track"], "half": half,
                    "text": text_of(item["state"]), "questions": questions,
                    "removed_on_desk": item["id"] in removed, "note": note})  # fmt: skip
    out.sort(key=lambda r: (r["track"], r["id"]))
    return out


def main() -> int:
    files = [p for p in sorted(CANDIDATES.glob("*.jsonl")) if not p.name.endswith(".meta.jsonl")]
    candidates = [i for p in files for i in read_jsonl(p)]
    metas = {m["id"]: m for p in CANDIDATES.glob("*.meta.jsonl") for m in read_jsonl(p)}
    selected = {m["id"] for m in metas.values() if m.get("selected") is False}
    candidates = [i for i in candidates if i["id"] not in selected]
    checked = {i["id"]: i for p in sorted(CHECKED.glob("*.jsonl")) for i in read_jsonl(p)}
    checks = {}
    for p in CHECKS.glob("*.json"):
        doc = json.loads(p.read_text())
        checks[f"{doc['item']}~{doc['qid']}"] = doc
    gold = json.loads(GOLD.read_text()) if GOLD.is_file() else {}
    confidence = {k: g.get("confidence") for k, g in gold.items()}
    legal = json.loads(LEGAL.read_text()) if LEGAL.is_file() else {}
    # Texts removed on this page stay listed (the page marks them), so a removal can be undone.
    data = rows(candidates, metas, checked, content_removals(checks), confidence, legal,
                browser_removals())  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False))
    print(json.dumps({"texts": len(data), "questions": sum(len(r["questions"]) for r in data),
                      "removed_on_desk": sum(r["removed_on_desk"] for r in data)}))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
