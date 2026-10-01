"""A trial of the build's cheap-judge checks on the pilots' templates and rows.

Before the full build pays for them, the template check, the scope check and
the per-template blind prior are run on what the pilots produced, where the
pre-build review named the templates and rows that should fail:

    python -m data.label.trial_checks --out results/step3/trial_checks --cap-usd 0.1

Writes checks.jsonl (one line per template or pair, with the probability) and
summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.label.build import blind_prior, check_template, in_scope, template_id, top_of
from data.label.generate import Template
from data.label.judge import LowLabelMass
from data.label.pilot import load_panel, make_client

PILOTS = ("results/step3/pilot2", "results/step3/pilot3")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data.label.trial_checks")
    parser.add_argument("--out", type=Path, default=Path("results/step3/trial_checks"))
    parser.add_argument("--cap-usd", type=float, default=0.1)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    judges, _generator, cheap_name = load_panel()
    cheap = next(j for j in judges if j.name == cheap_name)
    client = make_client(args.out, args.cap_usd)
    lines = []
    for pilot in PILOTS:
        templates = [
            Template.model_validate_json(line)
            for line in Path(pilot, "templates.jsonl").read_text("utf-8").splitlines()
        ]
        by_question = {t.question: t for t in templates}
        for template in templates:
            tid = template_id(template)
            blind = blind_prior(client, cheap, template, tid)
            lines.append(
                {
                    "kind": "template",
                    "pilot": pilot,
                    "tid": tid,
                    "task": f"{template.family}-{template.type}",
                    "question": template.question,
                    "check": round(check_template(client, cheap, template, tid), 4),
                    "blind_top": top_of(blind),
                    "blind_p": round(max(blind.values()), 4),
                }
            )
        for line in Path(pilot, "rows.jsonl").read_text("utf-8").splitlines():
            row = json.loads(line)
            template = by_question.get(row["question"]["instructions"])
            if template is None:
                continue
            try:
                p = in_scope(client, cheap, template, row["state"], row["row_id"])
            except LowLabelMass:
                continue
            lines.append(
                {
                    "kind": "scope",
                    "pilot": pilot,
                    "row_id": row["row_id"],
                    "task": row["task"],
                    "top": top_of(row["target"]),
                    "scope": round(p, 4),
                    "state": row["state"][:160],
                }
            )
    (args.out / "checks.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), "utf-8"
    )
    scopes = [x for x in lines if x["kind"] == "scope"]
    checks = [x for x in lines if x["kind"] == "template"]
    summary = {
        "templates": len(checks),
        "templates_failing_check": sum(x["check"] < 0.5 for x in checks),
        "templates_blind_confident": sum(x["blind_p"] >= 0.8 for x in checks),
        "pairs": len(scopes),
        "pairs_out_of_scope": sum(x["scope"] < 0.5 for x in scopes),
        "spent_usd": round(client.spent, 4),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
