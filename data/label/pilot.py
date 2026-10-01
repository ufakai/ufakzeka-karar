"""The labelling pilot: probe the panel, score the judges, build 500 rows.

Runs on the lab's server, where the API key lives, never on the laptop:

    python -m data.label.pilot probe  --out results/step3/pilot
    python -m data.label.pilot score  --out results/step3/pilot --per-task 75
    python -m data.label.pilot build  --out results/step3/pilot --templates 20 --per-template 25

Every command takes --cap-usd, spends through one ApiClient, and leaves the
API's per-call ledger beside its results. `--dry-run` builds every prompt and
writes the plan without a single call.

probe: one call per panel model, to confirm before anything larger that
reasoning is off and billed as zero, the pinned provider answered, and the
judges return log-probabilities with nearly all their mass on the letters.

score: the judges against human labels on the validation splits of the step 1
sets, 75 items each, under their `measure` status: each judge's
accuracy and mean probability on the gold option, and the panel's.

build: texts sampled and masked, templates written and checked, each template
applied to texts that one cheap judge says it fits, every pair labelled by the
panel, and a text-blind pass that drops rows whose answer the options alone
give away. Rows are TrainingRows; decontamination runs
on them afterwards, before any is used (rule 2).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import TypeAdapter

from data.label.client import ApiClient
from data.label.generate import FAMILIES, generator_messages, parse_template, to_question
from data.label.judge import JudgeSpec, label_row, parse_distribution
from data.label.texts import sample_mixed, write_sample
from schema.questions import Question
from schema.rows import TrainingRow, mean_vote, outcomes, text_of

QUESTION = TypeAdapter(Question)
# The API key is KARAR_API_KEY, or read from the file this names, set on the box (rule 4
# keeps the API's name out of code).
KEY_ENV = "KARAR_API_KEY"
KEY_FILE_ENV = "KARAR_API_KEY_FILE"
BLIND_STATE = "(Metin verilmedi.)"
BLIND_DROP = 0.8
# Screening: a template's first pairs are labelled before the rest.
# It is dropped when one answer takes most of its rows, since such rows teach
# the head a prior and not a decision, or when it fits too few texts to pay for.
SCREEN_PAIRS = 8
SCREEN_MIN_KEPT = 2
SCREEN_MAX_TOP_SHARE = 0.9
KINDS = ("choice", "choice", "noul", "score")
TRCOLA_QUESTION = {
    "type": "noul",
    "instructions": "Bu cümle dil bilgisi açısından doğru ve doğal bir Türkçe cümle mi?",
    "criteria": {
        "true": "Cümle kurallı ve kabul edilebilir.",
        "false": "Cümlede dil bilgisi ya da anlam bozukluğu var.",
    },
}
APPLIES_ASK = "Şu soru bu metin için anlamlı ve cevaplanabilir mi? Soru: "
APPLIES_NO = "Soru bu metinle ilgisiz ya da metin cevap için yetersiz."


def load_panel(path: Path = Path("config/panel.json")) -> tuple[list[JudgeSpec], dict, str]:
    panel = json.loads(path.read_text(encoding="utf-8"))
    judges = [
        JudgeSpec(j["name"], j["model"], j["provider"], j.get("mode", "logprobs"),
                  j.get("reasoning", "none"))
        for j in panel["judges"]
    ]  # fmt: skip
    return judges, panel["generator"], panel["cheap_judge"]


def load_relabel_judge(path: Path = Path("config/panel.json")) -> JudgeSpec:
    """Judge B, which relabels the message tasks judge A wrote."""
    j = json.loads(path.read_text(encoding="utf-8"))["relabel_judge"]
    return JudgeSpec(j["name"], j["model"], j["provider"], j.get("mode", "logprobs"),
                     j.get("reasoning", "none"))  # fmt: skip


def api_key() -> str:
    """The API key, from KARAR_API_KEY or the file KARAR_API_KEY_FILE names."""
    key, path = os.environ.get(KEY_ENV), os.environ.get(KEY_FILE_ENV)
    if key:
        return key
    if not path:
        raise SystemExit(f"set {KEY_ENV}, or {KEY_FILE_ENV} to the key's file")
    return Path(path).read_text(encoding="utf-8").strip()


def make_client(out: Path, cap_usd: float) -> ApiClient:
    return ApiClient(api_key(), cap_usd=cap_usd, ledger_path=out / "calls.jsonl")


def question_of(data: dict):
    return QUESTION.validate_python(data)


# probe ----------------------------------------------------------------------


def probe(client: ApiClient, judges: list[JudgeSpec], generator: dict) -> dict:
    question = question_of(
        {"type": "choice", "instructions": "Bu mesaj hangi konuyla ilgili?",
         "criteria": {"kargo": None, "fatura": None, "iade": None}}
    )  # fmt: skip
    state = "Siparişim üç gündür kargoya verilmedi, ne zaman gelir?"
    out = {}
    for judge in judges:
        started = time.perf_counter()
        votes = label_row(client, [judge], f"probe-{judge.name}", state, question)
        out[judge.name] = {
            "distribution": votes[0].distribution,
            "off_letter_mass": votes[0].off_letter_mass,
            "seconds": round(time.perf_counter() - started, 2),
        }
    started = time.perf_counter()
    response = client.chat(
        generator["model"], generator["provider"],
        generator_messages("konu", "choice", [f"Soru: {state}\nCevap: 2-3 iş günü içinde."] * 3),
        max_tokens=1500, temperature=0.7, forbid_reasoning_tokens=False,
        reasoning_effort=generator.get("reasoning", "none"),
    )  # fmt: skip
    content = response["choices"][0]["message"]["content"]
    parsed = parse_template(content)
    out["G"] = {
        "valid_template": parsed.template is not None,
        "problems": parsed.problems,
        "content": content[:600],
        "seconds": round(time.perf_counter() - started, 2),
    }
    return out


# score ----------------------------------------------------------------------


def scoring_items(per_task: int, seed: int = 1) -> list[dict]:
    """75 validation items per task with their human label."""
    rng = random.Random(seed)
    items = []
    for task in ("offenseval_tr", "mide22", "massive_tr"):
        lines = Path(f"data/built/typed/{task}/validation.jsonl").read_text("utf-8").splitlines()
        for line in rng.sample([x for x in lines if x.strip()], per_task):
            row = TrainingRow.model_validate_json(line)
            gold = max(row.target, key=row.target.get)
            question = row.question.model_dump(mode="json")
            item = {"task": task, "id": row.row_id, "state": text_of(row.state)}
            items.append(item | {"question": question, "gold": gold})
    # TrCOLA is scored under its `measure` status and never becomes a row.
    lines = Path("data/raw/instrument/trcola/validation.jsonl").read_text("utf-8").splitlines()
    for line in rng.sample([x for x in lines if x.strip()], per_task):
        row = json.loads(line)
        gold = "true" if row["label"] == 1 else "false"
        items.append(
            {"task": "trcola", "id": row["id"], "state": row["text"],
             "question": TRCOLA_QUESTION, "gold": gold}
        )  # fmt: skip
    return items


def score(client, judges, per_task: int, workers: int, out: Path | None = None) -> dict:
    items = scoring_items(per_task)
    kept = []

    def one(item):
        question = question_of(item["question"])
        try:
            votes = label_row(client, judges, item["id"], item["state"], question)
        except Exception as error:  # a failed item is counted, not hidden
            return item, None, repr(error)[:200]
        return item, votes, None

    per_judge = defaultdict(lambda: defaultdict(list))
    failures = Counter()
    with ThreadPoolExecutor(workers) as pool:
        for item, votes, _error in pool.map(one, items):
            if votes is None:
                failures[item["task"]] += 1
                continue
            keys = outcomes(question_of(item["question"]))
            panel = mean_vote(votes, keys)
            kept.append({"task": item["task"], "id": item["id"], "gold": item["gold"],
                         "votes": {v.judge: v.distribution for v in votes}})  # fmt: skip
            for name, dist in [(v.judge, v.distribution) for v in votes] + [("panel", panel)]:
                per_judge[item["task"]][name].append(
                    (max(dist, key=dist.get) == item["gold"], dist[item["gold"]])
                )
    if out is not None:
        (out / "score_items.jsonl").write_text(
            "".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept), "utf-8"
        )
    report = {"judges": [j.name for j in judges], "failures": dict(failures), "tasks": {}}
    for task, by_name in per_judge.items():
        report["tasks"][task] = {
            name: {"n": len(r), "accuracy": round(sum(c for c, _ in r) / len(r), 4),
                   "mean_p_gold": round(sum(p for _, p in r) / len(r), 4)}
            for name, r in by_name.items()
        }  # fmt: skip
    names = [j.name for j in judges] + ["panel"]
    report["overall"] = {
        name: round(sum(t[name]["accuracy"] for t in report["tasks"].values())
                    / len(report["tasks"]), 4)
        for name in names
    }  # fmt: skip
    return report


# build ----------------------------------------------------------------------


def write_templates(client, generator, texts, count, seed=1) -> tuple[list, Counter]:
    rng = random.Random(seed)
    families = rng.sample(list(FAMILIES) * (count // len(FAMILIES) + 1), count)
    plan = [(family, KINDS[i % len(KINDS)]) for i, family in enumerate(families)]
    templates, problems = [], Counter()
    for family, kind in plan:
        examples = [t["text"] for t in rng.sample(texts, 3)]
        response = client.chat(
            generator["model"], generator["provider"],
            generator_messages(family, kind, examples),
            max_tokens=1500, temperature=0.7, forbid_reasoning_tokens=False,
        reasoning_effort=generator.get("reasoning", "none"),
        )  # fmt: skip
        parsed = parse_template(response["choices"][0]["message"]["content"], family, kind)
        if parsed.template is None:
            for problem in parsed.problems:
                problems[problem.split(":")[0]] += 1
            continue
        templates.append(parsed.template)
    return templates, problems


def applies(client, judge: JudgeSpec, template, text: str, row_id: str) -> float:
    yes = f"Metin bu soruya uygun. ({template.applies_when})"
    check = question_of(
        {"type": "noul", "instructions": APPLIES_ASK + template.question,
         "criteria": {"true": yes, "false": APPLIES_NO}}
    )  # fmt: skip
    votes = label_row(client, [judge], f"applies-{row_id}", text, check)
    return votes[0].distribution["true"]


def screen_verdict(kept: list[TrainingRow]) -> str:
    """Whether a template's screening rows justify labelling the rest of its pairs."""
    if len(kept) < SCREEN_MIN_KEPT:
        return "rarely_applies"
    tops = Counter(max(row.target, key=row.target.get) for row in kept)
    if tops.most_common(1)[0][1] / len(kept) >= SCREEN_MAX_TOP_SHARE:
        return "low_variance"
    return "passed"


def build(
    client,
    judges,
    generator,
    cheap: str,
    out: Path,
    n_templates,
    per_template,
    workers,
    texts_path: Path | None = None,
):
    cheap_judge = next(j for j in judges if j.name == cheap)
    if texts_path is not None:
        texts = [json.loads(line) for line in texts_path.read_text("utf-8").splitlines() if line]
    else:
        texts = sample_mixed(n_templates * per_template * 2)
    write_sample(out / "texts.jsonl", texts)
    templates, template_problems = write_templates(client, generator, texts, n_templates)
    rng = random.Random(2)
    pairs = [(template, text) for template in templates
             for text in rng.sample(texts, min(per_template, len(texts)))]  # fmt: skip
    counts = Counter(templates_written=len(templates), pairs=len(pairs))

    def one(pair):
        template, text = pair
        question = question_of(to_question(template))
        draft = TrainingRow(
            track="sss", task=f"{template.family}-{template.type}", split="train",
            origin="generated", label_kind="rule", source=f"clips/mqa:{text['config']}",
            state=text["text"], question=question,
            target={k: (1.0 if i == 0 else 0.0) for i, k in enumerate(outcomes(question))},
            recipe="pilot-v3",
        )  # fmt: skip
        try:
            # Only a choice can fail to fit: its options may not cover the text.
            # A yes-or-no or score question always has an answer, and asking a
            # judge whether "is there a risk?" fits a riskless text removed the
            # "no" rows, which then made those templates look one-sided (pilot 2).
            if template.type == "choice" and (
                applies(client, cheap_judge, template, text["text"], draft.row_id) < 0.5
            ):
                return "not_applicable", None, None
            votes = label_row(client, judges, draft.row_id, text["text"], question)
            keys = outcomes(question)
            target = mean_vote(votes, keys)
            blind = label_row(client, [cheap_judge], f"blind-{draft.row_id}", BLIND_STATE, question)
            blind_dist = blind[0].distribution
            top = max(target, key=target.get)
            if max(blind_dist, key=blind_dist.get) == top and blind_dist[top] >= BLIND_DROP:
                return "answerable_blind", None, None
            row = TrainingRow(**{**draft.model_dump(mode="json"), "label_kind": "judges",
                                 "judges": [v.model_dump() for v in votes], "target": target,
                                 "row_id": ""})  # fmt: skip
            return "kept", row, votes
        except Exception as error:
            return f"failed:{type(error).__name__}", None, None

    by_template = defaultdict(list)
    for index, (template, _text) in enumerate(pairs):
        by_template[id(template)].append(index)
    screen = [i for indices in by_template.values() for i in indices[:SCREEN_PAIRS]]
    results: dict[int, tuple] = {}
    with ThreadPoolExecutor(workers) as pool:
        for index, result in zip(screen, pool.map(one, [pairs[i] for i in screen]), strict=True):
            results[index] = result
        passed, screening = [], []
        for template in templates:
            indices = by_template[id(template)]
            kept = [results[i][1] for i in indices[:SCREEN_PAIRS] if results[i][1] is not None]
            verdict = screen_verdict(kept)
            # A list, not a map by question: two templates can share a question.
            screening.append({"task": f"{template.family}-{template.type}",
                              "question": template.question, "verdict": verdict})  # fmt: skip
            if verdict == "passed":
                passed.extend(indices[SCREEN_PAIRS:])
            else:
                counts[f"template_{verdict}"] += 1
                for i in indices[:SCREEN_PAIRS]:
                    results[i] = ("screened_out", None, None)
        for index, result in zip(passed, pool.map(one, [pairs[i] for i in passed]), strict=True):
            results[index] = result

    rows, agreement, off_mass = [], [], defaultdict(list)
    for index in sorted(results):
        outcome, row, votes = results[index]
        counts[outcome.split(":")[0] if not outcome.startswith("failed") else outcome] += 1
        if row is None:
            continue
        rows.append(row)
        tops = [max(v.distribution, key=v.distribution.get) for v in votes]
        agreement.append(len(set(tops)) == 1)
        for v in votes:
            off_mass[v.judge].append(v.off_letter_mass)
    (out / "rows.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in rows), "utf-8")
    (out / "templates.jsonl").write_text(
        "".join(t.model_dump_json() + "\n" for t in templates), "utf-8"
    )
    top_by_task = defaultdict(Counter)
    for row in rows:
        top_by_task[row.task][max(row.target, key=row.target.get)] += 1
    return {
        "counts": dict(counts),
        "template_problems": dict(template_problems),
        "screening": screening,
        "texts_by_config": dict(Counter(t["config"] for t in texts)),
        "unanimous_share": round(sum(agreement) / len(agreement), 4) if agreement else None,
        "mean_off_letter_mass": {k: round(sum(v) / len(v), 4) for k, v in off_mass.items()},
        "top_option_by_task": {k: dict(v) for k, v in top_by_task.items()},
        "spent_usd": round(client.spent, 4),
        "usd_per_kept_row": round(client.spent / len(rows), 5) if rows else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data.label.pilot")
    parser.add_argument("command", choices=["probe", "score", "build"])
    parser.add_argument("--out", type=Path, default=Path("results/step3/pilot"))
    parser.add_argument("--cap-usd", type=float, default=2.0)
    parser.add_argument("--per-task", type=int, default=75)
    parser.add_argument("--templates", type=int, default=20)
    parser.add_argument("--per-template", type=int, default=25)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--judges", default="", help="restrict the panel, e.g. A,C")
    parser.add_argument("--texts", type=Path, help="a sample written by data.label.texts")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    judges, generator, cheap = load_panel()
    if args.judges:
        judges = [j for j in judges if j.name in args.judges.split(",")]
    if args.dry_run:
        plan = {
            "command": args.command,
            "judges": [j.name for j in judges],
            "cap_usd": args.cap_usd,
        }
        if args.command == "score":
            plan["items"] = Counter(i["task"] for i in scoring_items(args.per_task))
        print(json.dumps(plan, indent=2, default=dict))
        return 0
    client = make_client(args.out, args.cap_usd)
    if args.command == "probe":
        result = probe(client, judges, generator)
    elif args.command == "score":
        result = score(client, judges, args.per_task, args.workers, args.out)
    else:
        result = build(client, judges, generator, cheap, args.out, args.templates,
                       args.per_template, args.workers, args.texts)  # fmt: skip
    result["spent_usd"] = round(client.spent, 4)
    (args.out / f"{args.command}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main", "parse_distribution"]
