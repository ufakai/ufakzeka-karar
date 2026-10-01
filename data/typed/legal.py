"""Constitutional Court individual applications as typed rows: which right a complaint concerns.

    python -m data.typed.legal

Source: icgcihan/Turkish_Constutional_Court_Decisions (CC BY 4.0), rulings of
the Anayasa Mahkemesi on individual applications taken from the court's own
database, with the court's one-sentence summary of the complaint (`Başvuru
Konusu`), the right the ruling is filed under (`Haklar`, 21 headings) and the
outcome. The state is the summary alone, masked for personal data; the full
ruling text is never read, because it states the outcome. The outcome is not
turned into a task: predicting whether a court finds a violation has no place
in a legal-aid product.

The source's train and dev files become our train and validation. Its test
file is never downloaded (rule 2 keeps any test split out of training).

Question wording. The summary often names several rights ("kötü muamele
yasağının, özel hayata ve aile hayatına saygı hakkının ... ihlal edildiği")
while the source files each ruling under one heading, so the question asks
which right the complaint is mainly about ("öncelikle"). It says "bireysel
başvuru" so the setting is the court's individual application route, the only
one the labels come from.

Options. The 21 headings become ten, the most one training row carries,
so every row shows the same ten options and no distractor draw is needed:

- the two fair-trial headings stay apart, because civil and administrative
  cases against criminal cases is the distinction legal-aid triage needs;
- the compiler's joined headings are named by what their rows are about: the
  "Adil yargılanma hakkı (Hukuk)-Mülkiyet Hakkı" rows are property complaints
  (97 percent name "mülkiyet"), "Yaşam hakkı-Kötü muamele yasağı" keeps both
  rights in its name;
- assembly, association and trade-union rights (Anayasa articles 33, 34 and
  51) share one option, having about 230 training rows between them;
- the ten rarest headings, each under 70 training rows, become "başka bir
  hak", whose description lists them, since the head reads options blind to
  each other and "none of the others" would mean nothing to it. The heading
  the compiler calls "Adil yargılanma hakkı (Ceza)-Kişi hürriyeti ve güvenliği
  hakkı" holds complaints about the legality of crimes and penalties, and is
  described as such.

The option names and descriptions are ours.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from data.label.texts import mask_personal_data
from data.typed.common import (
    Item,
    clean_splits,
    fetch_pinned,
    hub_file_url,
    write_report,
    write_rows,
)
from schema.rows import TrainingRow

TASK = "aym_rights"
TRACK = "hukuk"
RECIPE = "typed-aym-rights-v1"
REPO = "icgcihan/Turkish_Constutional_Court_Decisions"
REVISION = "347a0f83d7e5f3bd74fcdf38d0684aa126b1920c"
RAW = Path("data/raw/_downloads/icgcihan__Turkish_Constutional_Court_Decisions")
OUT = Path("data/built/typed") / TASK
REPORT = Path("results/step3/typed") / f"{TASK}.json"
SOURCE = f"https://huggingface.co/datasets/{REPO}"

# Their split file for each of ours. test.json is deliberately absent.
SPLIT_FILES = {"train": "train.json", "validation": "dev.json"}
PINNED = {
    "train.json": "af9482ca4cf2a4b949e79ef939735c88625d1501d0e4341c0e510c435f1a576c",
    "dev.json": "91ce6078c50b7f949223f1328d107681e045ee4134ec9695458f780884b7e5b5",
    "README.md": "38ad7475a3704a761cd8324202a5d58013fe172a7d7d4106f0f99f0bf77c953f",
    "Citation for Dataset": "2046f22f562788bcf7d47c89f3bb2583e24583079cfeede5016a20acd6129143",
}

INSTRUCTIONS = "Bu bireysel başvurudaki şikâyet öncelikle hangi temel hakla ilgili?"

FAIR_CIVIL = "adil yargılanma (hukuk ve idare davaları)"
FAIR_CRIMINAL = "adil yargılanma (ceza davaları)"
PROPERTY = "mülkiyet hakkı"
LIFE = "yaşam hakkı ya da kötü muamele yasağı"
LIBERTY = "kişi özgürlüğü ve güvenliği"
PRIVATE = "özel hayatın ve aile hayatının korunması"
EXPRESSION = "ifade özgürlüğü"
INTEGRITY = "maddi ve manevi varlığın korunması"
ASSEMBLY = "toplanma, örgütlenme ve sendika hakları"
OTHER = "başka bir hak"

OPTIONS = {
    FAIR_CIVIL: (
        "Medeni hak ve yükümlülüklere ilişkin bir hukuk ya da idare davasında yargılama "
        "güvenceleri: makul sürede yargılanma, mahkemeye erişim, gerekçeli karar, "
        "kararın uygulanması."
    ),
    FAIR_CRIMINAL: (
        "Bir suç isnadıyla yürütülen ceza yargılamasında yargılama güvenceleri: makul sürede "
        "yargılanma, savunma hakkı, masumiyet karinesi, gerekçeli karar."
    ),
    PROPERTY: "Taşınmaz, kamulaştırma, alacak, vergi ya da başka bir mal varlığına müdahale.",
    LIFE: "Ölüm, yaralanma, işkence ya da kötü muamele ve bunların etkili soruşturulmaması.",
    LIBERTY: "Gözaltı, tutuklama ya da başka bir özgürlükten yoksun bırakmanın hukuka aykırılığı.",
    PRIVATE: "Özel hayat, aile hayatı, konut dokunulmazlığı ya da haberleşme özgürlüğü.",
    EXPRESSION: "Düşünceyi açıklama, basın ve sosyal medya paylaşımları nedeniyle yaptırım.",
    INTEGRITY: "Kişinin maddi ve manevi varlığı, onuru, sağlığı ya da itibarının korunması.",
    ASSEMBLY: "Toplantı ve gösteri yürüyüşü, dernek ya da sendika kurma ve bunlarda etkinlik.",
    OTHER: (
        "Eğitim hakkı, seçme ve seçilme hakkı, ayrımcılık yasağı, din ve vicdan özgürlüğü, "
        "suç ve cezaların kanuniliği, hükmün denetlenmesini isteme hakkı, etkili başvuru "
        "hakkı, zorla çalıştırma yasağı ya da bireysel başvurunun kapsamı dışında kalan "
        "bir hak."
    ),
}

# Every heading the source uses, and the option it becomes. A heading missing
# here stops the build rather than landing in an option by accident.
HEADINGS = {
    "Adil yargılanma hakkı (Medeni Hak ve Yükümlülükler)": FAIR_CIVIL,
    "Adil yargılanma hakkı (Suç İsnadı)": FAIR_CRIMINAL,
    "Adil yargılanma hakkı (Hukuk)-Mülkiyet Hakkı": PROPERTY,
    "Yaşam hakkı-Kötü muamele yasağı": LIFE,
    "Kişi özgürlüğü ve güvenliği hakkı": LIBERTY,
    "Özel hayatın ve aile hayatının korunması hakkı": PRIVATE,
    "İfade özgürlüğü": EXPRESSION,
    "Maddi ve manevi varlığın korunması hakkı": INTEGRITY,
    "Toplantı ve gösteri yürüyüşü düzenleme hakkı": ASSEMBLY,
    "Sendika hakkı": ASSEMBLY,
    "Örgütlenme özgürlüğü": ASSEMBLY,
    "Eğitim hakkı": OTHER,
    "Adil yargılanma hakkı (Ceza)-Kişi hürriyeti ve güvenliği hakkı": OTHER,
    "Seçme, seçilme ve siyasi faaliyette bulunma hakkı": OTHER,
    "Kapsam dışı haklar": OTHER,
    "Ayrımcılık yasağı": OTHER,
    "Din ve vicdan özgürlüğü": OTHER,
    "Hükmün denetlenmesini talep etme hakkı": OTHER,
    "Etkili başvuru hakkı": OTHER,
    "Zorla çalıştırma ve angarya yasağı": OTHER,
    "Bireysel başvuru hakkı": OTHER,
}

QUESTION = {"type": "choice", "instructions": INSTRUCTIONS, "criteria": OPTIONS}


def fetch(folder: Path = RAW) -> dict[str, Path]:
    return fetch_pinned(
        {name: (hub_file_url(REPO, REVISION, name), sha) for name, sha in PINNED.items()},
        folder,
    )


def items(records: list[dict], drops: Counter) -> list[Item]:
    """The summary, masked, and its heading's option, for each ruling."""
    out = []
    for record in records:
        heading = record["Haklar"].strip()
        if heading not in HEADINGS:
            raise ValueError(f"heading {heading!r} has no option; add it to HEADINGS")
        state = mask_personal_data(" ".join(record["Başvuru Konusu"].split()))
        if not state:
            drops["empty_summary"] += 1
            continue
        out.append(Item(record["Kararın Bağlantı Linki"], state, HEADINGS[heading]))
    return out


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
    drops = {split: Counter() for split in SPLIT_FILES}
    splits = {}
    for split, name in SPLIT_FILES.items():
        records = json.loads((raw / name).read_text(encoding="utf-8"))
        drops[split]["source_rows"] = len(records)
        splits[split] = items(records, drops[split])
    splits = clean_splits(splits, drops)
    counts = write_rows(splits, make_row, out)
    return {
        "task": TASK,
        "source": SOURCE,
        "revision": REVISION,
        "splits": counts,
        "dropped": {split: dict(sorted(c.items())) for split, c in drops.items()},
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
