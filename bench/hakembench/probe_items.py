"""HakemBench v1.0 probe items: option permutations, paraphrases, English, slots.

    python -m bench.hakembench.probe_items permutations --items results/private/step8/checked/*
    python -m bench.hakembench.probe_items generate --probe paraphrase --cap-usd 12
    python -m bench.hakembench.probe_items review-units --probe paraphrase
    python -m bench.hakembench.probe_items keep --probe paraphrase --answers A --equivalence E

Every probe turns gold-settled test items (bench/hakembench/desk.py collect) into
probe items. A probe item id is `<base id>~<probe>-<n>`, and `<base id>~perm.<qid>-<n>`
for a permutation, since one item can hold two choice questions; it carries only the
probed question, keeps the base item's gold and half, and is a plain harness
Item, so the harness reads probe files unchanged. What an Item cannot hold (the
base id, the half, the order shown, the English key map, the slot annotations)
goes to a `.meta.jsonl` file beside each item file, as with the candidates.

- permutations: per cell (track, choice question) a seeded draw of questions and
  one shared list of orders, every order when k! is at most 24, else 24 seeded
  ones with the identity and the reverse among them. n indexes the cell's list,
  written to results/step8/probes/permutation_orders.json. No call is made.
- paraphrase, english, slots: one generator call per drawn base item, journaled
  so a restart pays for nothing twice, under the step 8 cap counted over the
  whole step 8 ledger. n is the probed question's index in the base item.
  paraphrase and english probe one question per item, drawn per track and
  stratified by (question, gold), english first so a small track fills it;
  slots probe every gold question of every public-half item, all sharing one
  rewritten text. generate refuses while the desk's collect report has owner
  checks pending (the draws are seeded over the checked set, so a run on a
  partial set pays for calls a later run cannot reuse), and refuses a journal
  written for another draw.

Nothing generated is kept on the generator's word. review-units writes blind
review units (no proposal) and the pairs for a second pass that compares the
two texts; keep takes a candidate only when the blind answer is its gold,
unflagged, and the second pass finds the meaning the same.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import sys
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

from bench.hakembench.desk import REPORT as CATCH_REPORT
from bench.hakembench.desk import options, text_of
from bench.hakembench.generate import CAP_USD, LEDGER_DIR, journaled
from bench.harness.items import Item
from data.decontam.ngrams import tokens
from data.label.build import STOPPING, is_transient, read_jsonl
from data.label.pilot import load_panel, make_client

CHECKED = Path("results/private/step8/checked")
CANDIDATES = Path("bench/hakembench/candidates")
PUBLIC_OUT = Path("bench/hakembench/v1.0/probes")
PRIVATE_OUT = Path("results/private/step8/v1.0/probes")
WORK = Path("results/private/step8/probes")
REPORTS = Path("results/step8/probes")
SEED = 119

PROBES = ("paraphrase", "english", "slots")
# Questions per permutation cell by option count; every item when the cell has fewer.
PERM_QUESTIONS = {3: 200, 4: 50, 5: 50, 6: 50, 7: 50}
# k of 8 or more: 100 questions, 2,400 runs, because the detectable V grows with k.
PERM_QUESTIONS_WIDE = 100
MAX_ORDERS = 24
DRAW = {"paraphrase": 30, "english": 24}
KEEP = {"paraphrase": 25, "english": 20}
# Above this token Jaccard with the original, a paraphrase is a lexical echo.
JACCARD_MAX = 0.6
# Below this token Jaccard outside the slots, a slot reply rewrote more than its slots.
SLOT_JACCARD_MIN = 0.9
UNITS_PER_DOC = 40
MAX_TOKENS = 3000
TEMPERATURE = {"paraphrase": 0.8, "english": 0.2, "slots": 0.7}
SLOT_TYPES = ("person", "place", "organisation", "date", "amount")

SYSTEM = "Metinlerle çalışan dikkatli bir yazarsın. Yalnızca istenen JSON nesnesini yaz."


# items --------------------------------------------------------------------


def load_checked(paths: list[Path]) -> list[dict]:
    items, seen = [], set()
    for path in paths:
        for item in read_jsonl(path):
            if item["id"] in seen:
                raise ValueError(f"{path}: item {item['id']} is listed twice")
            seen.add(item["id"])
            items.append(item)
    return items


def load_halves(root: Path = CANDIDATES) -> dict[str, str]:
    """Item id to its half, from the candidates' meta files (`half`, or `split`)."""
    out = {}
    for path in sorted(root.glob("*.meta.jsonl")):
        for meta in read_jsonl(path):
            half = meta.get("half") or meta.get("split")
            if half is not None:
                out[meta["id"]] = half
    return out


def half_of(item: dict, halves: dict[str, str]) -> str:
    half = halves.get(item["id"])
    if half not in ("public", "private"):
        raise ValueError(f"{item['id']} has no public or private half in the candidates' meta")
    return half


def typed_gold(question: dict, key: str):
    """A desk key as the harness's gold: noul a bool, score a level index, choice the name."""
    kind = question["type"]
    if kind == "noul":
        if key not in ("true", "false"):
            raise ValueError(f"noul gold must be 'true' or 'false', got {key!r}")
        return key == "true"
    if kind == "score":
        if not str(key).isdigit():
            raise ValueError(f"score gold must be a level index, got {key!r}")
        return int(key)
    return key


def settled_qids(item: dict) -> list[str]:
    """The item's questions with gold whose source is known, in the item's order."""
    source = item.get("gold_source", {})
    return [q for q in item["questions"]
            if q in item.get("gold", {}) and source.get(q, "pending") != "pending"]  # fmt: skip


def settled(item: dict) -> bool:
    """Gold on at least one question, and no gold source pending."""
    source = item.get("gold_source", {})
    if not item.get("gold") or any(v == "pending" for v in source.values()):
        return False
    return all(source.get(q, "pending") != "pending" for q in item["gold"])


def probe_row(base: dict, qid: str, probe: str, n: int, half: str, *, state=None,
              question: dict | None = None, gold_key: str | None = None,
              **extra) -> tuple[str, str, str]:  # fmt: skip
    """(half, the Item as JSON, its meta as JSON) for one probe item; the Item is validated.

    A permutation id names its question: one item can hold two choice questions.
    """
    question = question if question is not None else base["questions"][qid]
    gold_key = gold_key if gold_key is not None else base["gold"][qid]
    tag = f"{probe}.{qid}" if probe == "perm" else probe
    item = Item.model_validate({
        "id": f"{base['id']}~{tag}-{n}", "track": base["track"],
        "state": base["state"] if state is None else state,
        "questions": {qid: question}, "gold": {qid: typed_gold(question, gold_key)},
    })  # fmt: skip
    meta = {"id": item.id, "base_id": base["id"], "probe": probe, "qid": qid, "half": half,
            **extra}  # fmt: skip
    return half, item.model_dump_json(), json.dumps(meta, ensure_ascii=False)


def write_probe(rows: list[tuple[str, str, str]], probe: str, public_dir: Path = PUBLIC_OUT,
                private_dir: Path = PRIVATE_OUT) -> dict[str, int]:  # fmt: skip
    """Item and meta files for each half, rows in id order; counts per half."""
    counts = {}
    for half, folder in (("public", public_dir), ("private", private_dir)):
        chosen = sorted((r for r in rows if r[0] == half), key=lambda r: json.loads(r[2])["id"])
        folder.mkdir(parents=True, exist_ok=True)
        stem = folder / f"{probe}.{half}"
        Path(f"{stem}.jsonl").write_text("".join(r[1] + "\n" for r in chosen), "utf-8")
        Path(f"{stem}.meta.jsonl").write_text("".join(r[2] + "\n" for r in chosen), "utf-8")
        counts[half] = len(chosen)
    return counts


# permutations -------------------------------------------------------------


def questions_for(k: int) -> int:
    return PERM_QUESTIONS.get(k, PERM_QUESTIONS[3] if k < 3 else PERM_QUESTIONS_WIDE)


def cell_orders(k: int, rng: random.Random) -> list[tuple[int, ...]]:
    """Every order when k! <= 24 (identity first), else 24 distinct: identity, reverse, drawn."""
    if math.factorial(k) <= MAX_ORDERS:
        return list(itertools.permutations(range(k)))
    identity = tuple(range(k))
    orders = [identity, identity[::-1]]
    seen = set(orders)
    while len(orders) < MAX_ORDERS:
        order = tuple(rng.sample(range(k), k))
        if order not in seen:
            seen.add(order)
            orders.append(order)
    return orders


def reordered(question: dict, order: tuple[int, ...]) -> dict:
    """The question with its options shown in `order`: position j shows option order[j]."""
    keys = list(question["criteria"])
    return {**question, "criteria": {keys[i]: question["criteria"][keys[i]] for i in order}}


def permutations(items: list[dict], halves: dict[str, str], seed: int = SEED) -> tuple[list, list]:
    """Probe rows and the cells with their shared orders."""
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in sorted(items, key=lambda i: i["id"]):
        for qid in settled_qids(item):
            if item["questions"][qid]["type"] == "choice":
                cells[(item["track"], qid)].append(item)
    rows, report = [], []
    for (track, qid), pool in sorted(cells.items()):
        ks = {len(i["questions"][qid]["criteria"]) for i in pool}
        if len(ks) != 1:
            raise ValueError(f"{track}:{qid} mixes option counts {sorted(ks)}")
        k = ks.pop()
        rng = random.Random(f"{seed}:perm:{track}:{qid}")
        drawn = sorted(rng.sample(pool, min(questions_for(k), len(pool))), key=lambda i: i["id"])
        orders = cell_orders(k, rng)
        keys = {tuple(i["questions"][qid]["criteria"]) for i in drawn}
        for item in drawn:
            half = half_of(item, halves)
            for n, order in enumerate(orders):
                question = reordered(item["questions"][qid], order)
                rows.append(probe_row(item, qid, "perm", n, half, question=question,
                                      order=list(order)))  # fmt: skip
        report.append({
            "track": track, "qid": qid, "k": k, "questions": len(drawn),
            "by_half": dict(Counter(half_of(i, halves) for i in drawn)),
            "options": list(keys.pop()) if len(keys) == 1 else None,
            "orders": [list(o) for o in orders],
        })  # fmt: skip
    return rows, report


# draws --------------------------------------------------------------------


def stratified(pairs: list[tuple[dict, str]], n: int, rng: random.Random) -> list:
    """Up to n (item, qid) pairs, one per item, taken in turn from each (qid, gold) stratum."""
    strata: dict[tuple[str, str], list] = defaultdict(list)
    for item, qid in sorted(pairs, key=lambda p: (p[0]["id"], p[1])):
        strata[(qid, str(item["gold"][qid]))].append((item, qid))
    queues = [strata[key] for key in sorted(strata)]
    for queue in queues:
        rng.shuffle(queue)
    rng.shuffle(queues)
    taken, used = [], set()
    while len(taken) < n and any(queues):
        for queue in queues:
            while queue and queue[-1][0]["id"] in used:
                queue.pop()
            if queue and len(taken) < n:
                item, qid = queue.pop()
                used.add(item["id"])
                taken.append((item, qid))
    return taken


def spec(item: dict, qids: list[str], probe: str, half: str) -> dict:
    return {"id": f"{item['id']}~{probe}", "base_id": item["id"], "track": item["track"],
            "half": half, "probe": probe, "qids": qids}  # fmt: skip


def draw(items: list[dict], halves: dict[str, str], probe: str, seed: int = SEED) -> list[dict]:
    """What a probe sends to the generator: one spec per base item."""
    if probe == "slots":
        return [spec(i, settled_qids(i), probe, "public")
                for i in sorted(items, key=lambda i: i["id"])
                if settled_qids(i) and half_of(i, halves) == "public"]  # fmt: skip
    exclude = set()
    if probe == "paraphrase":  # disjoint from the english draw, which goes first
        exclude = {s["base_id"] for s in draw(items, halves, "english", seed)}
    by_track: dict[str, list] = defaultdict(list)
    for item in items:
        if settled(item) and item["id"] not in exclude:
            by_track[item["track"]] += [(item, q) for q in settled_qids(item)]
    out = []
    for track, pairs in sorted(by_track.items()):
        rng = random.Random(f"{seed}:{probe}:{track}")
        for item, qid in stratified(pairs, DRAW[probe], rng):
            out.append(spec(item, [qid], probe, half_of(item, halves)))
    return sorted(out, key=lambda s: s["id"])


# prompts ------------------------------------------------------------------


TYPE_TR = {"choice": "seçenekli", "score": "düzeyli", "noul": "evet ya da hayır"}
TYPE_EN = {"choice": "choice", "score": "score", "noul": "yes/no"}


def shown_question(question: dict, gold_key: str, english: bool = False) -> str:
    """The question, its options and its gold, for the generator to hold fixed."""
    kind = (TYPE_EN if english else TYPE_TR)[question["type"]]
    lines = [f"({kind}) {text_of(question['instructions'])}"]
    for o in options(question):
        detail = f": {o['detail']}" if o["detail"] else ""
        key = f'"{o["key"]}"' if english else o["label"]
        lines.append(f"- {key}{detail}")
    gold = next(o for o in options(question) if o["key"] == gold_key)
    lines.append(("Correct answer: " if english else "Doğru cevap: ")
                 + (f'"{gold["key"]}"' if english else gold["label"]))  # fmt: skip
    return "\n".join(lines)


def paraphrase_prompt(base: dict, qids: list[str]) -> str:
    shown = "\n\n".join(shown_question(base["questions"][q], base["gold"][q]) for q in qids)
    return (
        "Aşağıdaki Türkçe metni anlamını koruyarak baştan yeniden yaz.\n"
        "- Açıkça farklı sözcükler ve farklı bir cümle yapısı kullan; metnin cümlelerini aynen "
        "ya da küçük değişikliklerle tekrarlama.\n"
        "- Aşağıdaki sorunun cevaplanması için gereken her bilgiyi koru; yeni metin de aynı "
        "cevabı almalı.\n"
        "- Metnin üslubunu koru: resmî ise resmî, gündelik ise gündelik; argo, küfür ya da "
        "yazım hatası varsa aynı ölçüde.\n"
        "- Yeni bir bilgi, ayrıntı ya da yorum ekleme; hiçbir bilgiyi çıkarma. Özel adları, "
        "sayıları ve tarihleri değiştirme.\n"
        '- "Soru:", "Cevap:", "Pasaj:" gibi bölüm etiketlerini ve [bağlantı] gibi köşeli '
        "ayraç içindeki yer tutucuları aynen koru.\n\n"
        f"Metin:\n<<<\n{text_of(base['state'])}\n>>>\n\n"
        "Soru ve doğru cevabı (yalnızca neyin değişmemesi gerektiğini bilmen için; metne "
        f"yazma):\n{shown}\n\n"
        'Yalnızca şu biçimde bir JSON nesnesi yaz: {"text": "..."}'
    )


def english_prompt(base: dict, qids: list[str]) -> str:
    (qid,) = qids
    return (
        "Translate the Turkish text and the decision question below into natural English, "
        "as a fluent native writer would put it.\n"
        "- Translate everything: the text, the question's instructions, the option "
        "descriptions and, for a choice question, the option names.\n"
        "- Keep every fact, the register and the section labels (translate a label such as "
        '"Soru:" as "Question:"). Keep placeholders in square brackets, such as [bağlantı], '
        "as they are.\n"
        "- For a choice question give each Turkish option name an English name; an option "
        "name that is a code (letters, digits, an abbreviation) stays as it is. For a score "
        'question keep the levels in their order; for a yes/no question keep the keys "true" '
        'and "false".\n'
        "- The English question must have the same correct answer.\n\n"
        f"Text:\n<<<\n{text_of(base['state'])}\n>>>\n\n"
        f"Question:\n{shown_question(base['questions'][qid], base['gold'][qid], True)}\n\n"
        "Reply with only a JSON object of this shape:\n"
        '{"text": "...", "question": {"instructions": "...", "criteria": ...}, '
        '"key_map": {"<Turkish option name>": "<English option name>"}}\n'
        'For a choice question "criteria" maps each English option name to its translated '
        "description (null where the Turkish has none); for a score question it is the list "
        'of translated level descriptions in order; for a yes/no question it is {"true": '
        '"...", "false": "..."}. key_map is {} for score and yes/no questions.'
    )


def slots_prompt(base: dict, qids: list[str]) -> str:
    shown = "\n\n".join(shown_question(base["questions"][q], base["gold"][q]) for q in qids)
    return (
        "Aşağıdaki Türkçe metindeki kişi adlarını, yer adlarını, kurum ve şirket adlarını, "
        "tarihleri ve tutarları başka, akla yatkın değerlerle değiştir.\n"
        "- Yeni değerleri Türkçe eklerle doğru çek: ünlü uyumu, kesme işareti, ünsüz "
        "yumuşaması.\n"
        "- Aşağıdaki soruların cevabının bağlı olduğu bir değeri asla değiştirme; onu slots "
        "listesine gold_dependent true olarak yaz ve replacement alanına aynı değeri koy.\n"
        "- Başka hiçbir şeyi değiştirme: cümleler, sözcükler, noktalama ve üslup aynı kalsın; "
        "[bağlantı] gibi yer tutucular da.\n"
        "- Değiştirilecek bir değer yoksa metni aynen yaz ve slots listesini boş bırak.\n\n"
        f"Metin:\n<<<\n{text_of(base['state'])}\n>>>\n\n"
        "Sorular ve doğru cevapları (yalnızca neyin değişmemesi gerektiğini bilmen için; "
        f"metne yazma):\n{shown}\n\n"
        "Yalnızca şu biçimde bir JSON nesnesi yaz:\n"
        '{"text": "...", "slots": [{"original": "...", "replacement": "...", "type": '
        f'"{"|".join(SLOT_TYPES)}", "gold_dependent": false}}]}}'
    )


PROMPTS = {"paraphrase": paraphrase_prompt, "english": english_prompt, "slots": slots_prompt}


# generation ---------------------------------------------------------------


def ask_one(client, generator: dict, probe: str, base: dict, s: dict) -> dict | None:
    """One generator call; the raw reply is journaled and parsed later, so a parse fix is free."""
    try:
        response = client.chat(
            generator["model"], generator["provider"],
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": PROMPTS[probe](base, s["qids"])}],
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE[probe], forbid_reasoning_tokens=False,
            reasoning_effort=generator.get("reasoning", "none"),
        )  # fmt: skip
    except STOPPING:
        raise
    except Exception as error:
        if is_transient(error):
            return None
        return {"id": s["id"], "content": None, "error": type(error).__name__}
    usage = response.get("usage") if isinstance(response, dict) else None
    cost = usage.get("cost") if isinstance(usage, dict) else None
    try:
        content = response["choices"][0]["message"]["content"] or ""
    except Exception as error:  # a malformed answer is journaled, never fatal
        return {"id": s["id"], "content": None, "error": type(error).__name__, "cost": cost}
    return {"id": s["id"], "content": content, "cost": cost}


class DrawMismatch(ValueError):
    """A journal written for another draw: its records answer other specs."""


def pending_block(report: Path) -> str | None:
    """Why generate must wait for the owner's checks, or None when the checked set is final."""
    if not report.is_file():
        return f"{report} is missing; run desk collect first"
    pending = json.loads(report.read_text(encoding="utf-8")).get("owner_pending")
    if pending != 0:
        return (f"{report} has owner_pending {pending}; the draws are seeded over the checked "
                "set, so calls made on a partial set cannot be reused")  # fmt: skip
    return None


def draw_fingerprint(specs: list[dict]) -> str:
    """sha256 of the sorted base ids drawn, one per line."""
    ids = sorted(s["base_id"] for s in specs)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def draw_file(journal: Path) -> Path:
    return journal.with_name(journal.name + ".draw.json")


def check_draw(journal: Path, specs: list[dict]) -> str:
    """Record the draw's fingerprint beside a new journal; refuse a journal of another draw."""
    fingerprint = draw_fingerprint(specs)
    sidecar = draw_file(journal)
    if sidecar.is_file():
        stored = json.loads(sidecar.read_text(encoding="utf-8")).get("fingerprint")
        if stored != fingerprint:
            raise DrawMismatch(f"{journal} was written for draw {stored}, this draw is "
                               f"{fingerprint}; move the journal aside to start over")  # fmt: skip
        return fingerprint
    if read_jsonl(journal):
        raise DrawMismatch(f"{journal} has records but no draw fingerprint in {sidecar}")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({"fingerprint": fingerprint, "base_ids": len(specs)}), "utf-8")
    return fingerprint


def reply_object(record: dict) -> tuple[dict | None, str | None]:
    """The parsed reply and its non-empty text, None where either is missing."""
    data = parse_object(record.get("content") or "")
    text = data.get("text") if data else None
    return data, text.strip() if isinstance(text, str) and text.strip() else None


def retried_file(journal: Path) -> Path:
    return journal.with_name(journal.name + ".retried")


def drop_unparsed(journal: Path) -> int:
    """Move the records with no content or no parsable text to `<journal>.retried`.

    They are asked again on the next run; the moved lines keep their cost on record.
    """
    if not journal.is_file():
        return 0
    keep, retried = [], []
    for line in journal.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            failed = record.get("content") is None or reply_object(record)[1] is None
            (retried if failed else keep).append(line)
    if retried:
        with retried_file(journal).open("a", encoding="utf-8") as f:
            f.write("".join(line + "\n" for line in retried))
        journal.write_text("".join(line + "\n" for line in keep), encoding="utf-8")
    return len(retried)


def generate(client, generator: dict, probe: str, specs: list[dict], bases: dict[str, dict],
             journal: Path, workers: int, limit: int | None = None,
             retry_unparsed: bool = False) -> dict[str, dict]:  # fmt: skip
    """Journal records by spec id; specs already journaled are not asked again.

    The journal must belong to this draw (check_draw). With retry_unparsed, the
    records with no usable reply are dropped from it first and asked again.
    """
    check_draw(journal, specs)
    if retry_unparsed:
        drop_unparsed(journal)
    done = {r["id"]: r for r in read_jsonl(journal)}
    todo = [s for s in specs if s["id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    jobs = [partial(ask_one, client, generator, probe, bases[s["base_id"]], s) for s in todo]
    for record in journaled(jobs, journal, workers):
        done[record["id"]] = record
    return done


def parse_object(content: str) -> dict | None:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def jaccard(a: str, b: str) -> float:
    x, y = set(tokens(a)), set(tokens(b))
    return len(x & y) / len(x | y) if x | y else 1.0


def outside_slots(a: str, b: str, slots: list[dict]) -> float:
    """Token Jaccard of two texts once the slots' tokens are gone.

    A token that starts with a slot token goes too, so a suffixed form such as
    "ankara'da" leaves with the slot "Ankara".
    """
    slot_tokens = {t for x in slots for t in tokens(x["original"]) + tokens(x["replacement"])}

    def rest(text: str) -> set[str]:
        return {t for t in tokens(text) if not any(t.startswith(s) for s in slot_tokens)}

    x, y = rest(a), rest(b)
    return len(x & y) / len(x | y) if x | y else 1.0


def slot_mismatch(original: str, text: str, slots: list[dict]) -> bool:
    """True when the new text does not show the slots the reply claims.

    A changed slot's replacement must occur in the new text and its original
    must not; a gold-dependent slot's original must still occur; and outside the
    slots the two texts must share at least SLOT_JACCARD_MIN of their tokens.
    """
    for x in slots:
        if x["gold_dependent"]:
            if x["original"] not in text:
                return True
        elif x["replacement"] != x["original"] and (
            x["replacement"] not in text or x["original"] in text
        ):
            return True
    return outside_slots(original, text, slots) < SLOT_JACCARD_MIN


def parse_english(question: dict, data: dict) -> tuple[dict, dict] | str:
    """(the English question, the key map) or why the reply is unusable."""
    q, key_map = data.get("question"), data.get("key_map") or {}
    if not isinstance(q, dict) or not isinstance(key_map, dict):
        return "bad translation"
    criteria = q.get("criteria")
    if question["type"] == "choice":
        keys = list(question["criteria"])
        values = [key_map.get(k) for k in keys]
        if (set(key_map) != set(keys) or not all(isinstance(v, str) and v.strip() for v in values)
                or len(set(values)) != len(values)
                or not isinstance(criteria, dict) or set(criteria) != set(values)):  # fmt: skip
            return "bad key map"
        criteria = {v: criteria[v] for v in values}
    else:
        key_map = {}
        levels = len(question["criteria"]) if question["type"] == "score" else None
        if levels is not None and (not isinstance(criteria, list) or len(criteria) != levels):
            return "bad translation"
    english = {"type": question["type"], "instructions": q.get("instructions"),
               "criteria": criteria}  # fmt: skip
    try:
        Item.model_validate({"id": "x", "track": "x", "state": "x", "questions": {"q": english}})
    except ValueError:
        return "bad translation"
    return english, key_map


def candidate(probe: str, s: dict, base: dict, record: dict | None) -> dict:
    """One parsed candidate, with `dropped` set to the reason when it goes no further."""
    out = {**s, "questions": {q: base["questions"][q] for q in s["qids"]},
           "gold": {q: base["gold"][q] for q in s["qids"]},
           "qindex": {q: list(base["questions"]).index(q) for q in s["qids"]},
           "original": text_of(base["state"]), "text": None, "dropped": None}  # fmt: skip
    if record is None:
        return out | {"dropped": "not generated"}
    out["cost"] = record.get("cost")
    if record.get("error"):
        return out | {"dropped": f"error:{record['error']}"}
    data, text = reply_object(record)
    if text is None:
        return out | {"dropped": "unparsed"}
    out["text"] = text
    if probe == "paraphrase":
        out["jaccard"] = round(jaccard(out["original"], out["text"]), 4)
        if out["jaccard"] > JACCARD_MAX:
            out["dropped"] = "lexical echo"
    elif probe == "english":
        (qid,) = s["qids"]
        parsed = parse_english(base["questions"][qid], data)
        if isinstance(parsed, str):
            out["dropped"] = parsed
        else:
            out["probe_questions"], out["key_map"] = {qid: parsed[0]}, parsed[1]
    else:
        raw = data.get("slots") if isinstance(data.get("slots"), list) else []
        slots = [{"original": str(x.get("original", "")),
                  "replacement": str(x.get("replacement", "")), "type": str(x.get("type", "")),
                  "gold_dependent": x.get("gold_dependent") is True}
                 for x in raw if isinstance(x, dict)]  # fmt: skip
        out["slots"] = slots
        if any(x["gold_dependent"] and x["replacement"] != x["original"] for x in slots):
            out["dropped"] = "gold slot changed"
        elif not any(not x["gold_dependent"] and x["replacement"] != x["original"]
                     for x in slots):  # fmt: skip
            out["dropped"] = "no slot"
        else:
            out["slot_jaccard"] = round(outside_slots(out["original"], out["text"], slots), 4)
            if slot_mismatch(out["original"], out["text"], slots):
                out["dropped"] = "slot text mismatch"
    return out


def candidates(probe: str, specs: list[dict], bases: dict[str, dict],
               records: dict[str, dict]) -> list[dict]:  # fmt: skip
    return [candidate(probe, s, bases[s["base_id"]], records.get(s["id"])) for s in specs]


# review -------------------------------------------------------------------


def probe_question(c: dict, qid: str) -> dict:
    return c.get("probe_questions", {}).get(qid) or c["questions"][qid]


def shown(question: dict, qid: str) -> dict:
    return {"qid": qid, "question": question["instructions"], "type": question["type"],
            "options": options(question)}  # fmt: skip


def review_docs(cands: list[dict]) -> tuple[list[dict], list[dict]]:
    """Blind units in the desk's shape, no proposal; and the pairs for the equivalence pass."""
    live = sorted((c for c in cands if not c["dropped"]), key=lambda c: c["id"])
    units, pairs = [], []
    for n, c in enumerate(live):
        asked = [shown(probe_question(c, q), q) for q in c["qids"]]
        units.append({"id": c["id"], "track": c["track"], "order": n, "text": c["text"],
                      "questions": asked})  # fmt: skip
        pair = {"id": c["id"], "track": c["track"], "original": c["original"], "probe": c["text"],
                "questions": [shown(c["questions"][q], q) for q in c["qids"]]}  # fmt: skip
        if "probe_questions" in c:
            pair["probe_questions"] = [shown(probe_question(c, q), q) for q in c["qids"]]
        pairs.append(pair)
    per = UNITS_PER_DOC
    return ([{"units": units[i : i + per]} for i in range(0, len(units), per)],
            [{"pairs": pairs[i : i + per]} for i in range(0, len(pairs), per)])  # fmt: skip


# keep ---------------------------------------------------------------------


def expected_key(c: dict, qid: str) -> str:
    """The gold as the probe item's key: through the key map for an English choice question."""
    return c.get("key_map", {}).get(c["gold"][qid], c["gold"][qid])


def verdict(c: dict, qid: str, answers: dict, equivalence: dict) -> str | None:
    """Why a probe question is not kept, or None when it is."""
    answer = answers.get(c["id"], {}).get(qid)
    if answer is None:
        return "no review answer"
    if answer.get("flag", "none") != "none":
        return "flagged"
    if answer.get("answer") != expected_key(c, qid):
        return "blind answer differs"
    same = equivalence.get(c["id"])
    if same is None:
        return "no equivalence verdict"
    if same.get("same") is not True:
        return "meaning differs"
    return None


def keep(probe: str, cands: list[dict], answers: dict, equivalence: dict,
         seed: int = SEED) -> tuple[list, dict]:  # fmt: skip
    """Probe rows kept by the blind and equivalence passes, capped per track, and a summary."""
    drops: dict[str, Counter] = defaultdict(Counter)
    passed: dict[str, list] = defaultdict(list)
    drawn = Counter()
    for c in sorted(cands, key=lambda c: c["id"]):
        for qid in c["qids"]:
            drawn[c["track"]] += 1
            reason = c["dropped"] or verdict(c, qid, answers, equivalence)
            if reason:
                drops[c["track"]][reason] += 1
            else:
                passed[c["track"]].append((c, qid))
    rows, kept = [], Counter()
    for track, chosen in sorted(passed.items()):
        cap = KEEP.get(probe)
        if cap is not None and len(chosen) > cap:
            random.Random(f"{seed}:keep:{probe}:{track}").shuffle(chosen)
            drops[track]["over the cap"] += len(chosen) - cap
            chosen = chosen[:cap]
        for c, qid in chosen:
            extra = {"key_map": c["key_map"]} if probe == "english" else {}
            if probe == "slots":
                extra = {"slots": c["slots"]}
            rows.append(probe_row(
                {"id": c["base_id"], "track": c["track"], "state": c["original"],
                 "questions": c["questions"], "gold": c["gold"]},
                qid, probe, c["qindex"][qid], c["half"], state=c["text"],
                question=probe_question(c, qid), gold_key=expected_key(c, qid), **extra,
            ))  # fmt: skip
            kept[track] += 1
    total = Counter()
    for counter in drops.values():
        total.update(counter)

    def shares(counts: Counter, n: int) -> dict[str, float]:
        """Each reason's share of the probe questions drawn."""
        return {reason: round(c / n, 4) for reason, c in sorted(counts.items())} if n else {}

    summary = {
        "probe": probe,
        "cap_per_track": KEEP.get(probe),
        "drawn": sum(drawn.values()),
        "kept": sum(kept.values()),
        "kept_by_half": dict(Counter(r[0] for r in rows)),
        "dropped": dict(sorted(total.items())),
        "dropped_share": shares(total, sum(drawn.values())),
        "per_track": {
            t: {
                "drawn": drawn[t],
                "kept": kept[t],
                "dropped": dict(sorted(drops[t].items())),
                "dropped_share": shares(drops[t], drawn[t]),
            }
            for t in sorted(drawn)
        },
        "spent_usd": round(sum(c.get("cost") or 0.0 for c in cands), 6),
    }
    return rows, summary


# run ----------------------------------------------------------------------


def checked_paths(given: list[Path] | None) -> list[Path]:
    return given or sorted(CHECKED.glob("*.jsonl"))


def read_folder(folder: Path) -> dict:
    out = {}
    for path in sorted(folder.glob("*.json")):
        out |= json.loads(path.read_text(encoding="utf-8"))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.hakembench.probe_items")
    sub = parser.add_subparsers(dest="command", required=True)
    perm_p = sub.add_parser("permutations")
    perm_p.add_argument("--items", type=Path, nargs="+", required=True)
    gen_p = sub.add_parser("generate")
    gen_p.add_argument("--probe", choices=PROBES, required=True)
    gen_p.add_argument("--items", type=Path, nargs="+", help="default: every checked track file")
    gen_p.add_argument("--cap-usd", type=float, default=CAP_USD,
                       help="the cap over the whole step 8 ledger")  # fmt: skip
    gen_p.add_argument("--workers", type=int, default=12)
    gen_p.add_argument("--limit", type=int, help="ask for at most this many new items")
    gen_p.add_argument("--allow-pending", action="store_true",
                       help="run although owner checks are pending (testing only)")  # fmt: skip
    gen_p.add_argument("--retry-unparsed", action="store_true",
                       help="ask again for the journaled replies with no usable text")  # fmt: skip
    units_p = sub.add_parser("review-units")
    units_p.add_argument("--probe", choices=PROBES, required=True)
    keep_p = sub.add_parser("keep")
    keep_p.add_argument("--probe", choices=PROBES, required=True)
    keep_p.add_argument("--answers", type=Path, required=True)
    keep_p.add_argument("--equivalence", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "permutations":
        rows, cells = permutations(load_checked(args.items), load_halves(CANDIDATES))
        counts = write_probe(rows, "permutations", PUBLIC_OUT, PRIVATE_OUT)
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / "permutation_orders.json").write_text(
            json.dumps({"seed": SEED, "cells": cells}, indent=1, ensure_ascii=False), "utf-8"
        )
        print(json.dumps({"items": counts, "cells": [
            {k: c[k] for k in ("track", "qid", "k", "questions")} | {"orders": len(c["orders"])}
            for c in cells]}, ensure_ascii=False))  # fmt: skip
        return 0

    work_candidates = WORK / f"{args.probe}_candidates.jsonl"
    if args.command == "generate":
        why = None if args.allow_pending else pending_block(CATCH_REPORT)
        if why is not None:
            print(f"refused: {why}; --allow-pending runs anyway, for testing", file=sys.stderr)
            return 1
        items = load_checked(checked_paths(args.items))
        specs = draw(items, load_halves(CANDIDATES), args.probe)
        bases = {i["id"]: i for i in items}
        _judges, generator, _cheap = load_panel()
        client = make_client(LEDGER_DIR, args.cap_usd)
        journal = WORK / f"{args.probe}_journal.jsonl"
        try:
            records = generate(client, generator, args.probe, specs, bases, journal,
                               args.workers, args.limit, args.retry_unparsed)  # fmt: skip
        except STOPPING as error:
            print(f"stopped: {error}; run it again to resume", file=sys.stderr)
            return 1
        except DrawMismatch as error:
            print(f"refused: {error}", file=sys.stderr)
            return 1
        cands = candidates(args.probe, specs, bases, records)
        work_candidates.write_text(
            "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cands), "utf-8"
        )
        print(json.dumps({"drawn": len(specs), "generated": sum(s["id"] in records for s in specs),
                          "dropped": dict(Counter(c["dropped"] for c in cands if c["dropped"])),
                          "spent_usd": round(client.spent, 4)}, ensure_ascii=False))  # fmt: skip
        return 0

    cands = read_jsonl(work_candidates)
    if not cands:
        raise SystemExit(f"{work_candidates} is missing or empty: run generate first")
    if args.command == "review-units":
        units, pairs = review_docs(cands)
        folder = WORK / "review"
        folder.mkdir(parents=True, exist_ok=True)
        for old in [*folder.glob(f"{args.probe}-units-*.json"),
                    *folder.glob(f"{args.probe}-pairs-*.json")]:  # fmt: skip
            old.unlink()
        for kind, docs in (("units", units), ("pairs", pairs)):
            for n, doc in enumerate(docs):
                path = folder / f"{args.probe}-{kind}-{n}.json"
                path.write_text(json.dumps(doc, ensure_ascii=False), "utf-8")
        print(json.dumps({"units": sum(len(d["units"]) for d in units), "docs": len(units)}))
        return 0
    rows, summary = keep(args.probe, cands, read_folder(args.answers),
                         read_folder(args.equivalence))  # fmt: skip
    summary["items"] = write_probe(rows, args.probe, PUBLIC_OUT, PRIVATE_OUT)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{args.probe}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), "utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
