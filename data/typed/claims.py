"""FACTurk claims as typed rows: the verdict Turkish fact-checkers gave a claim.

    python -m data.typed.claims

Source: github.com/altuncu/FACTurk (MIT), 15,766 fact-checking reports from
Teyit, Doğruluk Payı, Doğrula, Malumatfuruş, Yalansavar and Günün Yalanları,
each with the claim checked, the organisation's verdict and the compiler's
normalised rating. The state is the claim text alone, masked for personal
data. The CSV also carries the reports' article text in its `content` column,
although the README says it was withheld; that text is the organisations'
writing, which the compiler's MIT licence cannot cover, and it is never read.

The source has no splits. Validation is about a tenth, carved by a hash of the
compiler's low-confidence claim cluster (claims at 0.88 similarity share one),
so a claim checked twice, by two organisations or twice by one, lands on one
side only.

Question wording. The model sees the claim and no evidence, so the question
does not ask whether the claim is true; it asks what a fact-checking
organisation would conclude on examining it, which is what the labels record.

Options. The compiler's ratings, with case and spelling normalised ("false",
"Yanliş", "Yanliþ" are one rating; "Dogru" and "Doğru" another):

- yanlış ya da yanıltıcı: every organisation's false verdict, including
  Doğruluk Payı's "Kısmen Yanlış", which the compiler maps to false, and the
  misleading verdicts the compiler split off. It split them for Teyit and
  Doğrula only, so the same kind of claim was "misleading" from one
  organisation and "false" from another; one option keeps the label
  consistent;
- karma: Teyit's "Karma", true and false parts together;
- doğru;
- sonuçlandırılamadı: Teyit's own verdict when the evidence did not settle it.

Rows left out, each counted in the report:

- Malumatfuruş pages with no verdict: the compiler's scraper writes "Unknown"
  as the verdict when the page carries no rating, so it is not a verdict of
  "unverifiable";
- Yalansavar: its scraper sets every report to false without reading one, and
  a language model wrote its claim sentences, so neither side is the
  organisation's;
- any rating the table below does not know.

Günün Yalanları's reports are all false because the site publishes only false
statements; the scraper's fixed verdict follows the site's own premise, so
they are kept.

The option names and descriptions are ours.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

from data.label.texts import mask_personal_data
from data.typed.common import (
    Item,
    carve,
    clean_splits,
    fetch_pinned,
    write_report,
    write_rows,
)
from schema.rows import TrainingRow

TASK = "facturk_verdict"
TRACK = "dogrulama"
RECIPE = "typed-facturk-verdict-v1"
REPO = "altuncu/FACTurk"
REVISION = "b85871b4dcfaa9b0f43b037237ae4b548f6a51ed"
RAW = Path("data/raw/_downloads/altuncu__FACTurk")
OUT = Path("data/built/typed") / TASK
REPORT = Path("results/step3/typed") / f"{TASK}.json"
SOURCE = f"https://github.com/{REPO}"
_FILES = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/"
PINNED = {
    "FACTurk.zip": (
        _FILES + "data/FACTurk.zip",
        "19ae5c537afec1021315ed1cf01d69cbf6cf12694fbda593847baf668e04178a",
    ),
    "LICENSE": (
        _FILES + "LICENSE",
        "82060e49c1b3e887d2007662e137b7d031ac1c76a1da28959daedb7ea919cb09",
    ),
    "README.md": (
        _FILES + "README.md",
        "19dc80b028ff8a3b529a31fd2eb15c3357c02f2e8c240b3ae4875387f488f769",
    ),
}
CSV_NAME = "FACTurk.csv"

INSTRUCTIONS = "Bir doğrulama kuruluşu bu iddiayı incelese hangi sonuca varırdı?"

FALSE = "yanlış ya da yanıltıcı"
MIXED = "karma"
TRUE = "doğru"
UNSETTLED = "sonuçlandırılamadı"

OPTIONS = {
    FALSE: (
        "İddia gerçeği yansıtmıyor ya da gerçek bir içeriği bağlamından koparıyor, "
        "çarpıtıyor, yanlış bir olayla, yerle, zamanla ilişkilendiriyor."
    ),
    MIXED: "İddiada doğru ve yanlış kısımlar bir arada.",
    TRUE: "İddia doğru.",
    UNSETTLED: "Eldeki kanıtlar iddianın doğru mu yanlış mı olduğuna karar vermeye yetmiyor.",
}

# The compiler's normalised ratings after folding case and Turkish letters.
RATINGS = {
    "false": FALSE,
    "yanlis": FALSE,
    "misleading": FALSE,
    "mixed": MIXED,
    "partially true": MIXED,
    "true": TRUE,
    "dogru": TRUE,
    "unknown": UNSETTLED,
}
# Organisations whose rows are left out, and why (see the module docstring).
LEFT_OUT = {"Yalansavar": "verdict_fixed_by_scraper"}
NO_VERDICT = "Unknown"

QUESTION = {"type": "choice", "instructions": INSTRUCTIONS, "criteria": OPTIONS}

_FOLD = str.maketrans("ıİşŞþÞğĞüÜöÖçÇ", "iissssgguuoocc")


def fold(rating: str) -> str:
    return " ".join(rating.translate(_FOLD).lower().split())


def fetch(folder: Path = RAW) -> dict[str, Path]:
    return fetch_pinned(PINNED, folder)


def read_csv(archive: Path) -> list[dict]:
    with zipfile.ZipFile(archive) as bundle, bundle.open(CSV_NAME) as handle:
        return list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")))


def items(records: list[dict], drops: Counter) -> dict[str, list[Item]]:
    """Each usable report as an item, already placed in train or validation."""
    splits: dict[str, list[Item]] = {"train": [], "validation": []}
    for record in records:
        organisation = record["organisation"].strip()
        if organisation in LEFT_OUT:
            drops[LEFT_OUT[organisation]] += 1
            continue
        if record["original_verdict"].strip() == NO_VERDICT:
            drops["no_verdict_on_page"] += 1
            continue
        label = RATINGS.get(fold(record["normalised_rating"]))
        if label is None:
            drops["unmapped_rating"] += 1
            continue
        state = mask_personal_data(" ".join(record["claim"].split()))
        if not state:
            drops["empty_claim"] += 1
            continue
        split = carve(record["claim_id_lowconf"], salt=TASK)
        splits[split].append(Item(record["report_id"], state, label))
    return splits


def make_row(item: Item, split: str) -> TrainingRow:
    target = dict.fromkeys(OPTIONS, 0.0)
    target[item.label] = 1.0
    return TrainingRow(
        track=TRACK,
        task=TASK,
        split=split,
        origin="converted",
        label_kind="human",
        source=f"{RAW.as_posix()}/",
        state=item.state,
        question=QUESTION,
        target=target,
        recipe=RECIPE,
    )


def build(raw: Path = RAW, out: Path = OUT, *, download: bool = True) -> dict:
    """Write train.jsonl and validation.jsonl and return what was written and dropped."""
    if download:
        fetch(raw)
    records = read_csv(raw / "FACTurk.zip")
    source_drops: Counter = Counter({"source_rows": len(records)})
    splits = items(records, source_drops)
    drops = {"source": source_drops, "train": Counter(), "validation": Counter()}
    splits = clean_splits(splits, drops)
    counts = write_rows(splits, make_row, out)
    return {
        "task": TASK,
        "source": SOURCE,
        "revision": REVISION,
        "splits": counts,
        "dropped": {name: dict(sorted(c.items())) for name, c in drops.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    report = build()
    write_report(report, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
