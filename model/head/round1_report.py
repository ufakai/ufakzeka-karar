"""Round 1's table, from the runs' records (results/step4/round1/r1-*.json).

    python -m model.head.round1_report

Written before the last runs finished, so the rules below are fixed before
their numbers:

- each run: validation means of macro F1, Brier and log loss over task groups
  (the converted sets, the generated support templates, the generated tracks),
  the held-out cells, and MASSIVE's 59 intents;
- each encoder's rate: the one with the lower mean validation log loss over
  every task, the slice's rule;
- the ablations (no weights, no mined rows) are read against r1-main-s1 on the
  held-out cells, not on validation, whose rows also fitted the weights;
- our backbone's three seeds give a mean and a spread; so does any other arm
  with three seeds; the rest are one seed and are pilots.

Writes results/step4/round1/table.json and prints it.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path("results/step4/round1")
CONVERTED = ("massive_tr", "offenseval_tr", "mide22", "aym_rights", "facturk_verdict",
             "prompt_injection", "webfaq_relevance")  # fmt: skip
SUPPORT = ("konu", "niyet", "yanit_yeterliligi", "aciliyet", "bilgi_turu", "insan_destegi",
           "hassasiyet", "hedef_kitle")  # fmt: skip
GENERATED = ("spam", "oltalama", "egitim", "hukuk", "dogrulama")


def group_of(task: str) -> str:
    if task in CONVERTED:
        return "converted"
    head = task.split("-")[0]
    if head in SUPPORT:
        return "support"
    if head in GENERATED:
        return "generated"
    return "other"


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def summarise(record: dict) -> dict:
    last = record["evaluations"][-1]["metrics"]
    groups: dict[str, dict[str, list[float]]] = {}
    for task, m in last.items():
        g = groups.setdefault(group_of(task), {"macro_f1": [], "brier": [], "logloss": []})
        for key in g:
            g[key].append(m[key])
    held = record.get("heldout") or {}
    return {
        "backbone": record["spec"]["backbone"],
        "rate": record["spec"]["rate"],
        "seed": record["spec"]["seed"],
        "status": record["status"],
        "wall_seconds": record.get("wall_seconds"),
        "validation": {g: {k: mean(v) for k, v in vals.items()} for g, vals in groups.items()},
        "validation_logloss_all_tasks": mean([m["logloss"] for m in last.values()]),
        "converted_tasks": {
            t: {k: round(last[t][k], 4) for k in ("macro_f1", "brier")}
            for t in CONVERTED
            if t in last
        },  # fmt: skip
        "heldout_cells": {
            "macro_f1": mean([m["macro_f1"] for m in held.values()]),
            "brier": mean([m["brier"] for m in held.values()]),
        },  # fmt: skip
        "massive59_macro_f1": round(record["massive59"]["macro_f1"], 4)
        if record.get("massive59")
        else None,
    }


def main() -> int:
    runs = {p.stem: summarise(json.loads(p.read_text()))
            for p in sorted(ROOT.glob("r1-*.json"))}  # fmt: skip
    table: dict = {"runs": runs}

    def spread(group: list[dict], path) -> dict:
        values = [path(r) for r in group if path(r) is not None]
        return {"mean": mean(values),
                "sd": round(statistics.stdev(values), 4) if len(values) > 1 else None,
                "seeds": len(values)}  # fmt: skip

    def over_seeds(prefix: str) -> dict | None:
        group = [r for name, r in runs.items() if name.startswith(prefix)]
        if not group:
            return None
        return {
            "converted_macro_f1": spread(group, lambda r: r["validation"]["converted"]["macro_f1"]),
            "support_macro_f1": spread(group, lambda r: r["validation"]["support"]["macro_f1"]),
            "generated_macro_f1": spread(group, lambda r: r["validation"]["generated"]["macro_f1"]),
            "heldout_macro_f1": spread(group, lambda r: r["heldout_cells"]["macro_f1"]),
            "massive59_macro_f1": spread(group, lambda r: r["massive59_macro_f1"]),
        }

    if ours := over_seeds("r1-main-"):
        table["ours_over_seeds"] = ours
    # The rate was chosen on seed 1, where both rates were run; the later
    # seeds run only the chosen rate and must not turn the choice into a pick of
    # the luckiest seed.
    chosen = {}
    for name, r in runs.items():
        if r["backbone"] in ("ufakzeka", "continued") or not name.endswith("-s1"):
            continue
        best = chosen.get(r["backbone"])
        if (
            best is None
            or r["validation_logloss_all_tasks"] < runs[best]["validation_logloss_all_tasks"]
        ):
            chosen[r["backbone"]] = name
    table["encoder_rate_chosen"] = chosen
    # Every arm run on three seeds at one rate, beside ours.
    arms = {name.rsplit("-s", 1)[0] for name in runs if not name.startswith("r1-main-")}
    table["arms_over_seeds"] = {
        arm: over_seeds(arm + "-s")
        for arm in sorted(arms)
        if sum(name.startswith(arm + "-s") for name in runs) >= 3
    }
    main_run = runs.get("r1-main-s1")
    if main_run:
        table["ablations_on_heldout_cells"] = {
            name: {"macro_f1_minus_main": round(r["heldout_cells"]["macro_f1"]
                                                - main_run["heldout_cells"]["macro_f1"], 4),
                   "brier_minus_main": round(r["heldout_cells"]["brier"]
                                             - main_run["heldout_cells"]["brier"], 4),
                   "pilot": True}
            for name, r in runs.items() if name in ("r1-noweights-s1", "r1-nomined-s1")
        }  # fmt: skip
    (ROOT / "table.json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
