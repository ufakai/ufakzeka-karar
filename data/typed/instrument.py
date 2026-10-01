"""Step 1 sets as typed training rows, for the week-1 slice.

Three of the five instrument sets fit the product's question types and are
used here: MASSIVE as a 59-option choice, which also exercises passes of at
most ten options; OffensEval-TR as a yes-or-no question; MiDe22 as a
three-option choice. TrCOLA is left out because part of its text was written
by a model whose terms are unchecked, and the legal NLI set because its
labels measure how close two rulings' cited articles are, not the entailment
their names claim, which is the wrong thing for the head's first run to
learn.

Only train and validation splits are written. Test splits never are: some of
them are Cetvel or TrGLUE test material (rule 2), and the rest stay the
instrument's own measurement.

A set is converted only if data/MANIFEST.yaml clears it for training. Every
instrument set starts as `measure`, which clears it for nothing, so
nothing becomes training data by default and promotion is a visible manifest
change with a dated reason.

The question wording and the Turkish option names are ours.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import yaml

from data.instrument_format import Row
from schema.rows import TrainingRow

RECIPE = "typed-instrument-v2"
# MASSIVE has 59 intents and a row carries at most ten options: the
# correct intent plus nine distractors, drawn from a seed derived from the
# source row's id, so the same source row always gets the same options.
DISTRACTORS = 9
TRAINING_USES = {"train", "both"}
INSTRUMENT_ROOT = Path("data/raw/instrument")

MASSIVE_OPTIONS = {
    "alarm_query": "alarmları sorma",
    "alarm_remove": "alarm silme",
    "alarm_set": "alarm kurma",
    "audio_volume_down": "sesi kısma",
    "audio_volume_mute": "sesi kapatma",
    "audio_volume_other": "başka bir ses ayarı",
    "audio_volume_up": "sesi açma",
    "calendar_query": "takvime bakma",
    "calendar_remove": "takvimden silme",
    "calendar_set": "takvime ekleme",
    "cooking_recipe": "yemek tarifi",
    "datetime_convert": "saat ya da tarih çevirme",
    "datetime_query": "saati ya da tarihi sorma",
    "email_addcontact": "kişilere e-posta adresi ekleme",
    "email_query": "e-postalara bakma",
    "email_querycontact": "bir kişinin bilgisini sorma",
    "email_sendemail": "e-posta gönderme",
    "general_greet": "selamlaşma",
    "general_joke": "fıkra isteme",
    "general_quirky": "başka bir sohbet",
    "iot_cleaning": "robot süpürgeyi çalıştırma",
    "iot_coffee": "kahve makinesini çalıştırma",
    "iot_hue_lightchange": "ışığın rengini değiştirme",
    "iot_hue_lightdim": "ışığı kısma",
    "iot_hue_lightoff": "ışığı kapatma",
    "iot_hue_lighton": "ışığı açma",
    "iot_hue_lightup": "ışığı artırma",
    "iot_wemo_off": "akıllı prizi kapatma",
    "iot_wemo_on": "akıllı prizi açma",
    "lists_createoradd": "liste oluşturma ya da listeye ekleme",
    "lists_query": "listeye bakma",
    "lists_remove": "listeden silme",
    "music_dislikeness": "şarkıyı beğenmediğini söyleme",
    "music_likeness": "şarkıyı beğendiğini söyleme",
    "music_query": "çalan müziği sorma",
    "music_settings": "müzik ayarları",
    "news_query": "haberleri sorma",
    "play_audiobook": "sesli kitap açma",
    "play_game": "oyun açma",
    "play_music": "müzik çalma",
    "play_podcasts": "podcast açma",
    "play_radio": "radyo açma",
    "qa_currency": "döviz kuru sorma",
    "qa_definition": "bir kelimenin tanımını sorma",
    "qa_factoid": "bilgi sorusu",
    "qa_maths": "hesap sorusu",
    "qa_stock": "hisse fiyatı sorma",
    "recommendation_events": "etkinlik önerisi isteme",
    "recommendation_locations": "mekân önerisi isteme",
    "recommendation_movies": "film önerisi isteme",
    "social_post": "sosyal medyada paylaşım yapma",
    "social_query": "sosyal medyaya bakma",
    "takeaway_order": "yemek siparişi verme",
    "takeaway_query": "sipariş ya da restoran sorma",
    "transport_query": "ulaşım bilgisi sorma",
    "transport_taxi": "taksi çağırma",
    "transport_ticket": "bilet alma",
    "transport_traffic": "trafik durumunu sorma",
    "weather_query": "hava durumunu sorma",
}

MIDE22_OPTIONS = {
    "False": ("yanlış bilgi", "Paylaşım olay hakkında yanlış ya da yanıltıcı bir iddia taşıyor."),
    "True": ("doğru bilgi", "Paylaşım olay hakkında doğru bir iddia taşıyor."),
    "Other": ("iddia yok", "Paylaşım olay hakkında doğru ya da yanlış bir iddia taşımıyor."),
}


def _question(dataset: str, labels: list[str]) -> tuple[dict, list[str]]:
    """The typed question for a set, and the target key of each label index."""
    if dataset == "massive_tr":
        missing = set(labels) - set(MASSIVE_OPTIONS)
        if missing:
            raise ValueError(f"massive_tr labels without a Turkish option name: {sorted(missing)}")
        question = {
            "type": "choice",
            "instructions": "Kullanıcı sesli asistandan ne istiyor?",
            "criteria": {MASSIVE_OPTIONS[label]: None for label in labels},
        }
        return question, [MASSIVE_OPTIONS[label] for label in labels]
    if dataset == "offenseval_tr":
        if labels != ["NOT", "OFF"]:
            raise ValueError(f"offenseval_tr labels changed: {labels}")
        question = {
            "type": "noul",
            "instructions": "Bu mesaj küfür, hakaret ya da saldırgan bir dil içeriyor mu?",
            "criteria": {
                "true": "Mesajda küfür, hakaret ya da saldırgan bir ifade var.",
                "false": "Mesajda saldırgan bir ifade yok.",
            },
        }
        return question, ["false", "true"]
    if dataset == "mide22":
        if sorted(labels) != sorted(MIDE22_OPTIONS):
            raise ValueError(f"mide22 labels changed: {labels}")
        question = {
            "type": "choice",
            "instructions": "Bu paylaşım anlattığı olay hakkında ne tür bir iddia taşıyor?",
            "criteria": {name: text for name, text in MIDE22_OPTIONS.values()},
        }
        return question, [MIDE22_OPTIONS[label][0] for label in labels]
    raise ValueError(f"{dataset} has no typed question")


def recast(question: dict, keys: list[str], correct: str, seed_text: str) -> tuple[dict, list[str]]:
    """The correct option and nine others, in a seeded order, as a new question."""
    import hashlib
    import random

    seed = int.from_bytes(hashlib.blake2b(seed_text.encode(), digest_size=8).digest(), "big")
    rng = random.Random(seed)
    others = rng.sample([key for key in keys if key != correct], DISTRACTORS)
    chosen = [correct, *others]
    rng.shuffle(chosen)
    criteria = question["criteria"]
    return {**question, "criteria": {key: criteria[key] for key in chosen}}, chosen


TRACKS = {"massive_tr": "niyet", "offenseval_tr": "moderasyon", "mide22": "dogrulama"}


def cleared_for_training(dataset: str, manifest_path: Path = Path("data/MANIFEST.yaml")) -> bool:
    """Whether the manifest clears this set's converted folder for training."""
    folder = f"{INSTRUMENT_ROOT.as_posix()}/{dataset}/"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest.get("files", manifest.get("entries", [])) or []:
        if entry.get("path") == folder:
            return entry.get("use") in TRAINING_USES
    raise KeyError(f"{folder} has no manifest entry")


def typed_rows(dataset: str, split: str, root: Path = INSTRUMENT_ROOT) -> Iterator[TrainingRow]:
    """One training row per source row. Refuses the test split outright."""
    if split not in ("train", "validation"):
        raise ValueError(f"only train and validation are converted, not {split!r}")
    folder = root / dataset
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    question, keys = _question(dataset, meta["labels"])
    for line in (folder / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = Row(**json.loads(line))
        if row.text_pair is not None:
            raise ValueError(f"{dataset}: a text pair was not expected")
        row_question, row_keys = question, keys
        if len(keys) > DISTRACTORS + 1:
            row_question, row_keys = recast(question, keys, keys[row.label], seed_text=row.id)
        target = dict.fromkeys(row_keys, 0.0)
        target[keys[row.label]] = 1.0
        yield TrainingRow(
            track=TRACKS[dataset],
            task=dataset,
            split=split,
            origin="converted",
            label_kind="human",
            source=f"{root.as_posix()}/{dataset}/",
            state=row.text,
            question=row_question,
            target=target,
            recipe=RECIPE,
        )


def build(out_root: Path, datasets: tuple[str, ...] = tuple(TRACKS)) -> dict[str, dict[str, int]]:
    """Write every cleared set's train and validation rows. Returns counts."""
    counts: dict[str, dict[str, int]] = {}
    for dataset in datasets:
        if not cleared_for_training(dataset):
            raise PermissionError(
                f"{dataset} is not cleared for training in data/MANIFEST.yaml; promoting it takes "
                "a check of its terms and a dated decision"
            )
        counts[dataset] = {}
        for split in ("train", "validation"):
            out = out_root / dataset / f"{split}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            seen: set[str] = set()
            written = 0
            with out.open("w", encoding="utf-8") as handle:
                for row in typed_rows(dataset, split):
                    if row.row_id in seen:
                        continue
                    seen.add(row.row_id)
                    handle.write(row.model_dump_json() + "\n")
                    written += 1
            counts[dataset][split] = written
    return counts
