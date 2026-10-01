"""The full labelling build, with the fixes the pre-build review asked for.

Runs on the lab's server, where the API key lives, on a text sample written
on the laptop (data.label.texts):

    python -m data.label.build --out results/step3/build --texts texts.jsonl \
        --templates 170 --per-template 200 --cap-usd 8

What differs from the pilots (data/label/pilot.py), each for a finding of the
review:

- Every template gets an id, carried in its rows' task, so rows can be split
  and capped per template. Texts are split by source id, a fifth for
  evaluation, and a fifth of the family and type cells is held out whole
  (PLAN.md): a held-out template sees evaluation texts only, a training template's rows on
  evaluation texts are the validation split, and no training row shares a
  text with an evaluation row.
- A template is checked before it is used: the rules of data/label/generate.py,
  no near-duplicate of an earlier question, and one call to the cheap judge
  asking whether its options are distinct, answer its question, fit its family
  and, for a score, rise in order.
- Blind answerability is a property of the template, not of a row: its options
  are the same on every row. The cheap judge answers a choice template once
  per option order without a text, and the template is dropped when that blind
  answer is confident and matches its majority answer. No row is dropped for
  matching the prior, which pushed the kept labels away from it. Yes-or-no and
  score templates are not tested: without a text a judge answers "no" or the
  lowest level because nothing has the property, which says nothing about the
  options (stage one of the build dropped 26 such templates with healthy
  answer mixes before this was seen).
- A choice pair first passes a scope check that shows the judge the options:
  does one of them fit the text. A yes-or-no or score pair has none: its "no"
  or lowest level is a right answer for a text without the property, and the
  trial of the check on the pilots' rows (results/step3/trial_checks/) found
  it rejecting real customer questions with a "no" answer as readily as a
  spelling quiz. What those rows add is skew, which the stopping rule and the
  cap below handle. A row on which a judge spreads its vote evenly is kept
  apart as a no-fit row, for evaluating the abstain output later, and is not
  trained on.
- A judge whose vote misses the letter-mass rule is asked once more with the
  options in another order before the pair is given up, and the failure is
  counted per judge.
- Templates are screened on 16 pairs, then labelled in rounds of 32. A template
  stops when one answer takes too large a share of its rows or its judges
  agree too rarely, and at the end each template's most common answer is
  capped; rows over the cap are kept in a separate file, out of training.

- A template stopped for skew, or screened out for low variance, is mined for
  its rare answers: judge A alone reads more training texts and sends to both
  judges only those it leans away from the majority on. Urgent requests, for
  one, are a few percent of the texts, and without this no urgency template
  survives the screen. Mining reads training texts only, so the evaluation
  splits keep the real prior; the cap then balances the training rows.

Every pair's outcome is appended to journal.jsonl as it arrives, and a run
restarted on the same --out resumes from it without paying for a pair twice.
Errors of the network or the budget are not journaled, so a restart retries
them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from data.label.client import ApiError, BudgetExceeded, LowBalance
from data.label.generate import (
    FAMILIES,
    MAX_QUESTION_OVERLAP,
    Template,
    generator_messages,
    overlap,
    parse_template,
    to_question,
)
from data.label.judge import JudgeSpec, LowLabelMass, label_row
from data.label.pilot import BLIND_STATE, KINDS, load_panel, make_client, question_of
from data.label.texts import still_allowed
from schema.rows import TrainingRow, mean_vote, outcomes

RECIPE = "build-v1"
# Mined rows are marked so training can weight or leave them out: they were
# chosen on one judge's vote, which is also half of their target.
RECIPE_MINED = "build-v1-mined"
EVAL_TEXT_PERCENT = 20
HELDOUT_CELLS_PER_KIND = {"choice": 2, "noul": 1, "score": 1}
CHECK_MIN = 0.5
BLIND_FLAG = 0.8
SCOPE_MIN = 0.5
# A vote this flat says the judge found no option that fits.
NO_FIT_SPREAD = 0.1
SCREEN_PAIRS = 16
SCREEN_MIN_KEPT = 4
SCREEN_MAX_TOP_SHARE = 0.85
ROUND_PAIRS = 32
STOP_AFTER = 32
STOP_TOP_SHARE = {"noul": 0.75, "choice": 0.60, "score": 0.60}
MIN_UNANIMOUS_AFTER = 24
MIN_UNANIMOUS = 0.6
CAP_TOP_SHARE = {"noul": 0.65, "choice": 0.50, "score": 0.50}
# Mining for rare answers: judge A alone reads up to MINE_TEXTS more training
# texts per skewed template and sends to the panel those where it gives some
# minority answer MINE_MINORITY or more, until as many minority rows as the cap
# will keep beside the template's majority rows, at most MINE_TARGET.
MINE_TEXTS = 300
MINE_TARGET = 40
MINE_MINORITY = 0.3
MINE_MIN_ROWS = 8
# Errors that say nothing about the pair: not journaled, retried on a restart.
TRANSIENT = (ApiError, OSError, TimeoutError)


def is_transient(error: Exception) -> bool:
    """A network failure, a rate limit or a server error; not an answer about the pair.

    A provider that refuses a text answers the same way every time (an error in
    the body of a 200, or a 4xx other than 429), so that is journaled as the
    pair's outcome instead of being retried forever.
    """
    if isinstance(error, ApiError):
        status = error.status
        return status is None or status == 429 or status >= 500
    return isinstance(error, TRANSIENT)


STOPPING = (BudgetExceeded, LowBalance)

CHECK_QUESTION = {
    "type": "noul",
    "instructions": (
        "Yukarıdaki karar sorusu iyi yazılmış mı? İyi yazılmış bir soruda her seçenek "
        "başka bir şey söyler (ikisi aynı anlama gelmez), her seçenek sorunun sorduğu şeye "
        "bir cevaptır, soru verilen konu ailesine uyar ve puan düzeyleri sırayla artar."
    ),
    "criteria": {
        "true": "Soru bu koşulların hepsini sağlıyor.",
        "false": "En az bir koşul sağlanmıyor.",
    },
}
SCOPE_CHOICE = "Şu soru bu metne sorulabilir mi ve seçeneklerden biri metne uyuyor mu? Soru: "
SCOPE_OTHER = (
    'Şu soru bu metne sorulabilir mi? Cevabın "hayır" ya da en düşük düzey olması '
    "soruyu uygunsuz yapmaz; soru yalnızca metin sorunun konu aldığı türden değilse "
    "uygunsuzdur. Soru: "
)
SCOPE_NO = "Soru bu metne sorulmaz: metin sorunun konu aldığı türden değil."


def short_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=4).hexdigest()


def template_id(template: Template) -> str:
    return short_hash(template.model_dump_json())


def text_pool(source_id: str) -> str:
    return "eval" if int(short_hash(str(source_id)), 16) % 100 < EVAL_TEXT_PERCENT else "train"


def held_out_cells() -> set[str]:
    """The family and type pairs whose templates are all held out.

    Chosen by hash within each type, two of the eight choice cells and one each
    of yes-or-no and score: since choice gets half the templates, that holds
    out about a fifth of them. Holding out single templates would leave
    paraphrases of each held-out question in training, which tests rewording,
    not a new task.
    """
    cells = set()
    for kind, count in HELDOUT_CELLS_PER_KIND.items():
        ranked = sorted((f"{family}-{kind}" for family in FAMILIES), key=short_hash)
        cells.update(ranked[:count])
    return cells


def held_out(template: Template) -> bool:
    return f"{template.family}-{template.type}" in held_out_cells()


def top_of(distribution: dict[str, float]) -> str:
    return max(distribution, key=distribution.get)


# templates -----------------------------------------------------------------


def write_templates(client, generator, texts, count, seed) -> tuple[list[Template], Counter]:
    rng = random.Random(seed)
    families = rng.sample(list(FAMILIES) * (count // len(FAMILIES) + 1), count)
    plan = [(family, KINDS[i % len(KINDS)]) for i, family in enumerate(families)]
    examples_from = [t for t in texts if text_pool(t["source_id"]) == "train"]
    templates, problems = [], Counter()
    for family, kind in plan:
        examples = [t["text"] for t in rng.sample(examples_from, 3)]
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
        if any(overlap(parsed.template.question, t.question) > MAX_QUESTION_OVERLAP
               for t in templates):  # fmt: skip
            problems["near-duplicate question"] += 1
            continue
        templates.append(parsed.template)
    return templates, problems


def render(template: Template) -> str:
    """A template as the one-call check reads it."""
    lines = [f"Konu ailesi: {FAMILIES[template.family]}", f"Soru: {template.question}"]
    if template.type == "choice":
        lines.append("Seçenekler:")
        lines += [f"- {o.name}: {o.description}" if o.description else f"- {o.name}"
                  for o in template.options]  # fmt: skip
    elif template.type == "noul":
        lines += [f"Evet: {template.true}", f"Hayır: {template.false}"]
    else:
        lines.append("Puan düzeyleri, en düşükten en yükseğe:")
        lines += [f"{i}. {level}" for i, level in enumerate(template.levels)]
    return "\n".join(lines)


def check_template(client, cheap: JudgeSpec, template: Template, tid: str) -> float:
    """The cheap judge's probability that the template is well written; 0 if it cannot say."""
    for seed in (f"check-{tid}", f"check-{tid}-again"):
        try:
            votes = label_row(client, [cheap], seed, render(template),
                              question_of(CHECK_QUESTION))  # fmt: skip
            return votes[0].distribution["true"]
        except LowLabelMass:
            continue
    return 0.0


def blind_prior(client, cheap: JudgeSpec, template: Template, tid: str) -> dict[str, float]:
    """The cheap judge's answer without a text, averaged over option orders.

    Without a text a judge sometimes writes something other than a letter; such
    an order is skipped, and a template no order could answer gets a flat prior,
    which never flags it.
    """
    question = question_of(to_question(template))
    keys = outcomes(question)
    orders = 1 if template.type == "score" else max(3, len(keys))
    votes = []
    for i in range(orders):
        try:
            votes.append(label_row(client, [cheap], f"blind-{tid}-{i}", BLIND_STATE, question)[0])
        except LowLabelMass:
            continue
    return mean_vote(votes, keys) if votes else {k: 1 / len(keys) for k in keys}


# pairs ---------------------------------------------------------------------


def in_scope(client, cheap: JudgeSpec, template: Template, text: str, row_id: str) -> float:
    if template.type == "choice":
        names = "; ".join(o.name for o in template.options)
        ask = f"{SCOPE_CHOICE}{template.question} Seçenekler: {names}"
    else:
        ask = SCOPE_OTHER + template.question
    check = question_of(
        {"type": "noul", "instructions": ask,
         "criteria": {"true": f"Metin bu soruya uygun. ({template.applies_when})",
                      "false": SCOPE_NO}}
    )  # fmt: skip
    return label_row(client, [cheap], f"scope-{row_id}", text, check)[0].distribution["true"]


def no_fit(votes) -> bool:
    """A flat vote over three or more options; on two, a flat vote is only doubt."""
    return any(len(v.distribution) > 2
               and max(v.distribution.values()) - min(v.distribution.values()) <= NO_FIT_SPREAD
               for v in votes)  # fmt: skip


def label_pair(client, judges, cheap, template: Template, tid: str, text: dict, split: str,
               recipe: str = RECIPE):  # fmt: skip
    """One pair's outcome as a journal record, or None for an error worth retrying."""
    question = question_of(to_question(template))
    keys = outcomes(question)
    draft = TrainingRow(
        track="sss", task=f"{template.family}-{template.type}-{tid}", split=split,
        origin="generated", label_kind="rule", source=f"clips/mqa:{text['config']}",
        state=text["text"], question=question,
        target={k: (1.0 if i == 0 else 0.0) for i, k in enumerate(keys)}, recipe=recipe,
    )  # fmt: skip
    record = {"key": f"{tid}:{text['source_id']}", "tid": tid}
    try:
        if template.type == "choice" and (
            in_scope(client, cheap, template, text["text"], draft.row_id) < SCOPE_MIN
        ):
            return record | {"outcome": "out_of_scope"}
        for attempt in range(2):
            seed = draft.row_id if attempt == 0 else f"{draft.row_id}-again"
            try:
                votes = label_row(client, judges, seed, text["text"], question)
                break
            except LowLabelMass as error:
                if attempt == 1:
                    return record | {"outcome": "low_mass", "judge": error.judge}
                record["retried_judge"] = error.judge
        row = TrainingRow(**{**draft.model_dump(mode="json"), "label_kind": "judges",
                             "judges": [v.model_dump() for v in votes],
                             "target": mean_vote(votes, keys), "row_id": ""})  # fmt: skip
        outcome = "no_fit" if no_fit(votes) else "labelled"
        return record | {"outcome": outcome, "row": row.model_dump(mode="json")}
    except STOPPING:
        raise
    except Exception as error:
        if is_transient(error):
            return None
        return record | {"outcome": f"failed:{type(error).__name__}"}


def mine_pair(client, judges, cheap, template: Template, tid: str, text: dict, majority: str):
    """Judge A alone first; both judges only when it leans away from the majority."""
    record = {"key": f"{tid}:{text['source_id']}", "tid": tid, "mined": True}
    try:
        question = question_of(to_question(template))
        vote = label_row(client, [cheap], f"mine-{tid}-{text['source_id']}", text["text"],
                         question)[0]  # fmt: skip
    except STOPPING:
        raise
    except Exception as error:
        if is_transient(error):
            return None
        return record | {"outcome": f"failed:{type(error).__name__}"}
    if max(p for k, p in vote.distribution.items() if k != majority) < MINE_MINORITY:
        return record | {"outcome": "mined_majority"}
    result = label_pair(client, judges, cheap, template, tid, text, "train", RECIPE_MINED)
    return None if result is None else result | {"mined": True}


# decisions per template ----------------------------------------------------


def labelled_rows(records: list[dict]) -> list[TrainingRow]:
    return [TrainingRow.model_validate(r["row"]) for r in records if r["outcome"] == "labelled"]


def top_share(rows: list[TrainingRow]) -> float:
    tops = Counter(top_of(r.target) for r in rows)
    return tops.most_common(1)[0][1] / len(rows)


def unanimous_share(rows: list[TrainingRow]) -> float:
    return sum(len({top_of(v.distribution) for v in r.judges}) == 1 for r in rows) / len(rows)


def screen_verdict(rows: list[TrainingRow], blind: dict[str, float], kind: str = "choice") -> str:
    if len(rows) < SCREEN_MIN_KEPT:
        return "rarely_applies"
    if top_share(rows) >= SCREEN_MAX_TOP_SHARE:
        return "low_variance"
    majority = Counter(top_of(r.target) for r in rows).most_common(1)[0][0]
    # A blind "no" or lowest level is the null answer, not a clue in the options.
    null = {"noul": "false", "score": "0"}.get(kind)
    blind_top = top_of(blind)
    if blind_top != null and blind[blind_top] >= BLIND_FLAG and blind_top == majority:
        return "answerable_blind"
    return "passed"


def stop_reason(rows: list[TrainingRow], kind: str) -> str | None:
    if len(rows) >= MIN_UNANIMOUS_AFTER and unanimous_share(rows) < MIN_UNANIMOUS:
        return "judges_disagree"
    if len(rows) >= STOP_AFTER and top_share(rows) >= STOP_TOP_SHARE[kind]:
        return "skewed"
    return None


def balance(rows: list[TrainingRow], cap: float) -> tuple[list[TrainingRow], list[TrainingRow]]:
    """Rows kept under the cap on any one answer's share, in labelling order, and the rest."""
    tops = [top_of(r.target) for r in rows]
    limit = {}
    for answer, n in Counter(tops).items():
        others = len(rows) - n
        limit[answer] = n if n <= cap * len(rows) else int(cap * others / (1 - cap))
    kept, surplus, used = [], [], Counter()
    for row, answer in zip(rows, tops, strict=True):
        if used[answer] < limit[answer]:
            kept.append(row)
            used[answer] += 1
        else:
            surplus.append(row)
    return kept, surplus


# build ---------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def prepare_templates(client, judges, generator, cheap, texts, out, n_templates, seed) -> list:
    """Templates written, checked and given a blind prior; each step saved as it ends."""
    written = out / "generated.jsonl"
    if written.exists():
        templates = [Template.model_validate(e) for e in read_jsonl(written)]
    else:
        templates, problems = write_templates(client, generator, texts, n_templates, seed)
        written.write_text("".join(t.model_dump_json() + "\n" for t in templates), "utf-8")
        (out / "template_problems.json").write_text(json.dumps(dict(problems), indent=2))
    path = out / "templates.jsonl"
    entries = {e["tid"]: e for e in read_jsonl(path)}
    with path.open("a", encoding="utf-8") as f:
        for template in templates:
            tid = template_id(template)
            if tid in entries:
                continue
            check = check_template(client, cheap, template, tid)
            entry = {"tid": tid, "held_out": held_out(template), "check": round(check, 4),
                     "template": template.model_dump(mode="json")}  # fmt: skip
            if check < CHECK_MIN:
                entry["status"] = "failed_check"
            else:
                entry["blind"] = blind_prior(client, cheap, template, tid)
                entry["status"] = "ready"
            entries[tid] = entry
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
    return [entries[template_id(t)] for t in templates]


def pairs_for(entry: dict, texts: list[dict], per_template: int, seed: int) -> list[tuple]:
    """A template's texts in a fixed order, each with the split its row goes to."""
    if entry["held_out"]:
        eligible = [t for t in texts if text_pool(t["source_id"]) == "eval"]
    else:
        eligible = list(texts)
    # A full shuffle, so a larger --per-template on a restart extends the same order.
    random.Random(f"{seed}-{entry['tid']}").shuffle(eligible)
    chosen = eligible[:per_template]
    out = []
    for text in chosen:
        if entry["held_out"]:
            split = "heldout_task"
        else:
            split = "validation" if text_pool(text["source_id"]) == "eval" else "train"
        out.append((text, split))
    return out


def build(client, judges, generator, cheap_name, out, texts, n_templates, per_template,
          workers, seed) -> dict:  # fmt: skip
    cheap = next(j for j in judges if j.name == cheap_name)
    texts = list({t["text"]: t for t in texts}.values())  # one row per text and template
    # A sample drawn under older filters is checked again; rows already labelled
    # on a text the current filters drop are left out of every count and file.
    texts = [t for t in texts if still_allowed(t)]
    allowed = {t["source_id"] for t in texts}
    entries = prepare_templates(client, judges, generator, cheap, texts, out, n_templates, seed)
    ready = [e for e in entries if e["status"] == "ready"]
    templates = {e["tid"]: Template.model_validate(e["template"]) for e in ready}
    plans = {e["tid"]: pairs_for(e, texts, per_template, seed) for e in ready}
    journal_path = out / "journal.jsonl"
    records = {r["key"]: r for r in read_jsonl(journal_path)}
    transient = Counter()

    def run(batch: list[tuple]) -> None:
        jobs = [(tid, partial(label_pair, client, judges, cheap, templates[tid], tid, text, split))
                for tid, text, split in batch]  # fmt: skip
        run_jobs(jobs)

    def run_jobs(jobs: list[tuple]) -> None:
        stopped = None
        with ThreadPoolExecutor(workers) as pool, journal_path.open("a", encoding="utf-8") as f:
            futures = [pool.submit(job) for _tid, job in jobs]
            for (tid, _job), future in zip(jobs, futures, strict=True):
                # Every finished pair is journaled before a budget stop is raised.
                try:
                    record = future.result()
                except STOPPING as error:
                    stopped = error
                    continue
                if record is None:
                    transient[tid] += 1
                    continue
                records[record["key"]] = record
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
        if stopped is not None:
            raise stopped

    def done(tid: str) -> list[dict]:
        """A template's planned records; a mined record on a planned text is not one."""
        return [records[k] for k in (f"{tid}:{t['source_id']}" for t, _ in plans[tid])
                if k in records and not records[k].get("mined")]  # fmt: skip

    # Plans are drawn from the filtered texts, so done() never sees a dropped one.

    def of(tid: str) -> list[dict]:
        """Every record of a template, planned or mined."""
        return [r for r in records.values()
                if r["tid"] == tid and r["key"].split(":", 1)[1] in allowed]  # fmt: skip

    def todo(tid: str, limit: int) -> list[tuple]:
        pending = [(tid, t, s) for t, s in plans[tid] if f"{tid}:{t['source_id']}" not in records]
        return pending[:limit]

    status: dict[str, str] = {}
    run([p for tid in plans for p in todo(tid, max(0, SCREEN_PAIRS - len(done(tid))))])
    for tid in plans:
        screened = done(tid)[:SCREEN_PAIRS]
        verdict = screen_verdict(
            labelled_rows(screened),
            next(e["blind"] for e in ready if e["tid"] == tid),
            templates[tid].type,
        )
        status[tid] = "active" if verdict == "passed" else verdict

    while True:
        for tid in [t for t, s in status.items() if s == "active"]:
            reason = stop_reason(labelled_rows(done(tid)), templates[tid].type)
            if reason:
                status[tid] = reason
            elif not todo(tid, 1):
                status[tid] = "complete"
        batch = [p for tid, s in status.items() if s == "active" for p in todo(tid, ROUND_PAIRS)]
        if not batch:
            break
        before = len(records)
        run(batch)
        if len(records) == before:  # every pair failed transiently: stop, restart later
            break

    held = {e["tid"] for e in ready if e["held_out"]}
    train_texts = [t for t in texts if text_pool(t["source_id"]) == "train"]
    candidates = {}
    for tid, state in status.items():
        # The majority and the target come from unmined rows only, so they do not
        # move as mining adds minority rows, and a restart mines the same answer.
        rows = labelled_rows([r for r in of(tid) if not r.get("mined")])
        if state not in ("low_variance", "skewed") or tid in held or not rows:
            continue
        majority = Counter(top_of(r.target) for r in rows).most_common(1)[0][0]
        base = sum(r.split == "train" and top_of(r.target) == majority for r in rows)
        cap = CAP_TOP_SHARE[templates[tid].type]
        target = min(MINE_TARGET, max(MINE_MIN_ROWS, int(base * cap / (1 - cap))))
        order = list(train_texts)
        random.Random(f"{seed}-{tid}-mine").shuffle(order)
        candidates[tid] = (majority, target, order)

    def mined_minority(tid: str, majority: str) -> int:
        rows = labelled_rows([r for r in of(tid) if r.get("mined")])
        return sum(top_of(r.target) != majority for r in rows)

    while True:
        jobs = []
        for tid, (majority, target, order) in candidates.items():
            tried = sum(bool(r.get("mined")) for r in of(tid))
            if mined_minority(tid, majority) >= target or tried >= MINE_TEXTS:
                continue
            fresh = [t for t in order if f"{tid}:{t['source_id']}" not in records]
            room = min(ROUND_PAIRS, MINE_TEXTS - tried)
            jobs += [(tid, partial(mine_pair, client, judges, cheap, templates[tid], tid, t,
                                   majority)) for t in fresh[:room]]  # fmt: skip
        if not jobs:
            break
        before = len(records)
        run_jobs(jobs)
        if len(records) == before:
            break
    for tid, (majority, target, _order) in candidates.items():
        if mined_minority(tid, majority) >= min(MINE_MIN_ROWS, target):
            status[tid] = f"mined_{status[tid]}"

    return assemble(out, entries, templates, status, records, of, transient, client)


def assemble(out, entries, templates, status, records, of, transient, client) -> dict:
    rows_out, surplus, nofit, unused = [], [], [], []
    outcomes_count, low_mass_by_judge, retried = Counter(), Counter(), Counter()
    per_template = {}
    for tid, template in templates.items():
        records_t = of(tid)
        for r in records_t:
            outcomes_count[r["outcome"]] += 1
            if r["outcome"] == "low_mass":
                low_mass_by_judge[r.get("judge")] += 1
            if "retried_judge" in r:
                retried[r["retried_judge"]] += 1
        rows = labelled_rows(records_t)
        nofit += [TrainingRow.model_validate(r["row"]) for r in records_t
                  if r["outcome"] == "no_fit"]  # fmt: skip
        if status[tid] in ("active", "complete", "skewed") or status[tid].startswith("mined_"):
            # Only training rows are capped; evaluation keeps the real prior.
            kept, over = balance([r for r in rows if r.split == "train"],
                                 CAP_TOP_SHARE[template.type])  # fmt: skip
            rows_out += kept + [r for r in rows if r.split != "train"]
            surplus += over
        else:
            unused += rows
        per_template[tid] = {
            "task": f"{template.family}-{template.type}",
            "status": status[tid],
            "labelled": len(rows),
            "top_share": round(top_share(rows), 3) if rows else None,
            "unanimous": round(unanimous_share(rows), 3) if rows else None,
        }
    for name, rows in (("rows", rows_out), ("surplus", surplus), ("nofit", nofit),
                       ("unused", unused)):  # fmt: skip
        (out / f"{name}.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in rows),
                                           "utf-8")  # fmt: skip
    planned = sum(len(of(tid)) for tid in templates)
    return {
        "templates": dict(Counter(e["status"] for e in entries)),
        "template_status": dict(Counter(status.values())),
        "held_out_cells": sorted(held_out_cells()),
        "held_out_templates": sum(e["held_out"] for e in entries if e["status"] == "ready"),
        "pair_outcomes": dict(outcomes_count),
        "pairs_labelled_or_judged": planned,
        "low_mass_by_judge": dict(low_mass_by_judge),
        "retried_by_judge": dict(retried),
        "transient_errors": sum(transient.values()),
        "rows_by_split": dict(Counter(r.split for r in rows_out)),
        "rows": len(rows_out),
        "surplus": len(surplus),
        "nofit": len(nofit),
        "unused_rows_of_stopped_templates": len(unused),
        "per_template": per_template,
        "spent_usd": round(client.spent, 4),
        "usd_per_row": round(client.spent / len(rows_out), 5) if rows_out else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data.label.build")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--texts", type=Path, required=True,
                        help="a sample written by data.label.texts")  # fmt: skip
    parser.add_argument("--cap-usd", type=float, required=True)
    parser.add_argument("--templates", type=int, default=170)
    parser.add_argument("--per-template", type=int, default=200)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    texts = read_jsonl(args.texts)
    judges, generator, cheap = load_panel()
    client = make_client(args.out, args.cap_usd)
    try:
        result = build(client, judges, generator, cheap, args.out, texts, args.templates,
                       args.per_template, args.workers, args.seed)  # fmt: skip
    except STOPPING as error:
        print(f"stopped: {error}; restart on the same --out to resume", file=sys.stderr)
        return 1
    (args.out / "build.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != "per_template"}, indent=2,
                     ensure_ascii=False))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
