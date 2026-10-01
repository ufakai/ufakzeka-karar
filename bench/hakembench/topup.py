"""The v1.0 top-up: new authored items for the tracks below 300.

    python -m bench.hakembench.topup plan                 # the writer requests, as agent batches
    python -m bench.hakembench.topup build --writers DIR  # screen what the agents wrote
    python -m bench.hakembench.topup passes-input         # blind batches for the two model passes
    python -m bench.hakembench.topup gold --research DIR --second DIR [--redecided DIR]

An AI model, run as local agents, writes new education exams, moderation
comments, support Q&A pairs and lay legal questions on topics, forms, sectors
and scenes that neither the training texts nor the first round used. It is
asked for the masking placeholders directly ([ad], [şirket] with its type word,
[ilçe], [bağlantı], [telefon]), never for invented names, because invented
names turned out to be real businesses.

Nothing here touches the first round's files: every output goes to
candidates/<track>-topup.jsonl and its meta, so the hand-masked texts and their
gold stay as they are. `build` masks with the regex rules only (the greeting
rule once turned "Merhaba, geçen" into "Merhaba, [ad]"), harmonises suffixes,
refuses unknown brackets and real brand names (stems anchored so "garanti"
and "getir" as plain words pass), applies texts.py's filters, drops an exam
whole when one of its four answers goes, and removes near duplicates within
the round, of the first round and of every file under data/built. Halves are
drawn by writer unit against the first round's counts, so no old item moves.

Gold (`gold`): the three panel judges vote separately and the writer's family
(the research and second passes) cannot outvote them. A question is unanimous
when all five answers agree; when the three judges agree against the family,
the judges' answer is gold and the question goes to the owner; anything else
is adjudicated and tagged writer_family_adjudicated. Support questions all go
through the owner's written reading, as the first round's did; the
knowledge questions take model gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from bench.hakembench.common import overlapping, training_files, training_texts, unit_halves
from bench.hakembench.desk import BROWSER_REMOVALS, read_jsonl
from bench.hakembench.placeholders import PLACEHOLDERS, harmonise
from data.decontam.ngrams import MinHashLSH, minhash_signature, tokens, whole_hash
from data.label import synth
from data.label import texts as T

CANDIDATES = Path("bench/hakembench/candidates")
WORK = Path("results/private/step8/topup")
CHECKED = Path("results/private/step8/checked")
LICENCE = "CC BY 4.0 where rights exist, otherwise CC0; authored by ufak AI for the benchmark"
WRITER = "assistant"
ROUND = "topup-1"
BATCH_REQUESTS = 12

# Topics, forms, sectors and scenes new to both the training texts and the first round;
# check_disjoint compares strings, and a second agent read them for overlap in meaning.
EDU_TOPICS = {
    "Türkçe": ("sıfatlar", "zarflar", "edatlar", "anlatıcı ve bakış açısı",
               "deyimler ve atasözleri"),
    "Matematik": ("oran ve orantı", "doğrusal denklemler", "üçgenler", "çember ve daire",
                  "asal sayılar ve bölünebilme"),
    "Fizik": ("düzgün doğrusal hareket", "serbest düşme", "Hooke yasası ve esneklik",
              "Doppler olayı", "yarı iletkenler"),
    "Kimya": ("tepkime denklemlerinin denkleştirilmesi", "sabun ve deterjanlar",
              "metaller ve alaşımlar", "su sertliği", "yakıtlar ve yanma"),
    "Biyoloji": ("biyolojik çeşitlilik", "mantarlar", "kan grupları",
                 "karbonhidratlar ve lipitler", "populasyon ve komünite"),
    "Tarih": ("Karahanlılar", "Gazneliler", "Anadolu beylikleri", "Kavimler Göçü",
              "Amerikan Bağımsızlık Savaşı"),
    "Coğrafya": ("karstik şekiller", "buzul şekilleri", "yeraltı suları", "okyanus akıntıları",
                 "sanayi bölgeleri"),
    "Felsefe": ("İbn Sina", "Gazali", "David Hume", "John Stuart Mill", "Jean-Paul Sartre"),
}  # fmt: skip
MODERATION_FORMS = ("sosyal medya paylaşımına yanıt", "uygulama mağazası yorumu", "kitap yorumu",
                    "spor haberine okur yorumu", "ilan sitesi yorumu")  # fmt: skip
MODERATION_PER_REQUEST = 10
SUPPORT_SECTORS = ("elektrik ve doğalgaz aboneliği", "araç kiralama", "yemek siparişi",
                   "spor salonu üyeliği", "emlak ve kiralama", "veteriner kliniği",
                   "çevrim içi bilet satışı", "mobilya ve beyaz eşya satışı")  # fmt: skip
SUPPORT_PER_REQUEST = 6
COURT_SCENES = {
    # Two scenes were written and dropped after review: a trademark dispute belongs to the
    # intellectual-property court, which is not among the options, and a dismissal in the
    # probation period gives the worker no claim, so it read as a trick question.
    "İş mahkemesi": ("iş kıyafeti parasının maaştan kesilmesi",),
    "Aile mahkemesi": ("boşandıktan sonra eski eşin soyadını kullanmak",
                       "çocuk için ödenen nafakanın artırılması"),
    "Tüketici mahkemesi ya da hakem heyeti": (
        "iptal edilen otel rezervasyonunun parasının iade edilmemesi",
        "yetkili serviste kaybolan beyaz eşya"),
    "Asliye ticaret mahkemesi": (),
    "Sulh hukuk mahkemesi": ("kiracının evi izinsiz başkasına kiralaması",
                             "ortak bahçenin kullanımı yüzünden "
                             "kat malikleri arasında anlaşmazlık"),
    "İdare mahkemesi": ("belediyenin işyerini mühürlemesi",
                        "öğretmenin norm fazlası sayılarak başka okula atanması"),
    "Asliye hukuk mahkemesi": ("arazisine yol olmayan komşunun geçit hakkı istemesi",),
    "İcra hukuk mahkemesi": ("haciz sırasında eşine ait eşyaların da haczedilmesi",),
    "Savcılığa suç duyurusu": ("adına sahte doktor raporu düzenlenmesi",),
}  # fmt: skip

PLACEHOLDER_RULE = (
    "Gerçek hiçbir kişi, marka, şirket, banka, operatör, uygulama ya da kamu kurumu adı "
    "yazma; uydurma ad da yazma. Bunların yerine yalnızca şu yer tutucuları kullan: kişi "
    "adı için [ad]; şirket, dükkân, marka ya da işletme adı için [şirket], türünü belirten "
    "sözcük varsa arkasına ekleyerek ([şirket] Eczanesi, [şirket] Sigorta, [şirket] Kargo); "
    "ilçe, semt ya da kasaba adı için [ilçe]; bağlantı için [bağlantı]; telefon numarası için "
    "[telefon]. Yer tutucudan sonra gelen eki kesme işaretiyle ayır ([şirket]'e, [ad]'ın). "
    "Başka köşeli parantez kullanma. İl adları ve genel yer adları (İstanbul, Karadeniz) "
    "olduğu gibi kalabilir."
)
CONTENT_RULE = (
    "Cinsel içerik, çocuk istismarı, şiddet tarifi, bir gruba (etnik köken, din, cinsiyet, "
    "cinsel yönelim, engellilik) yönelik düşmanlık, bahis, uyuşturucu, silah, casus yazılım "
    "ya da lisanssız finans tanıtımı yazma."
)
# An exam may name who it is about: the no-names rule is for living and private people.
HISTORY_RULE = (
    "Sınav sorusu ve cevaplar tarihî kişileri, düşünürleri, bilim insanlarını, eserleri ve "
    "tarihî devletleri adlarıyla anabilir (Farabi, Kant, Karamanoğulları gibi); bu kural "
    "yalnızca yaşayan ve özel kişiler içindir."
)
JSON_LIST = 'Yalnızca şu biçimde bir JSON dizisi yaz: [{"metin": "..."}, ...]'
JSON_QA = 'Yalnızca şu biçimde bir JSON dizisi yaz: [{"soru": "...", "cevap": "..."}, ...]'

# The brand filter of the first round, with the stems that are also plain Turkish words
# ("garanti" is a warranty, "getir" is "bring", "emniyet" is safety) anchored to the brand.
REAL_NAMES = re.compile(
    synth.REAL_NAMES.pattern.replace("garanti|", r"garanti ?bbva|")
    .replace("getir|", r"\bgetir (?:uygulama|kurye|siparis|sipariş)|")
    .replace("emniyet|", r"emniyet (?:genel )?müdürlü|"),
    re.IGNORECASE,
)
# texts.py's regex masks, writing the benchmark's own placeholder words; its greeting
# rules are left out on purpose.
MASKS = tuple(
    (pattern, {"[kimlik]": "[kimlik no]", "[iban]": "[hesap no]",
               "@[kullanıcı]": "@[kullanıcı adı]"}.get(repl, repl))
    for pattern, repl in T.MASKS
)  # fmt: skip
BRACKET = re.compile(r"\[([^\[\]\n]{1,30})\]")


def request_id(track: str, *parts: str) -> str:
    return f"tu-{track}-{hashlib.blake2b('|'.join(parts).encode(), digest_size=8).hexdigest()}"


def check_disjoint() -> None:
    """Raise if a new topic, form, sector or scene repeats one already used."""
    from bench.hakembench import generate as G

    def fold(values) -> set[str]:
        return {" ".join(v.casefold().split()) for v in values}

    pairs = {
        "egitim": ([t for v in EDU_TOPICS.values() for t in v],
                   [t for v in (*synth.TOPICS.values(), *G.EDU_TOPICS.values()) for t in v]),
        "moderasyon": (list(MODERATION_FORMS), list(G.MODERATION_FORMS)),
        "sss": (list(SUPPORT_SECTORS), list(G.SUPPORT_DOMAINS)),
        "hukuk": ([s for v in COURT_SCENES.values() for s in v],
                  [s for v in G.COURT_SCENES.values() for s in v]
                  + [s for _d, v in synth.COURTS.values() for s in v]),
    }  # fmt: skip
    for name, (new, old) in pairs.items():
        if len(fold(new)) != len(new):
            raise ValueError(f"{name}: a new entry is listed twice")
        clash = sorted(fold(new) & fold(old))
        if clash:
            raise ValueError(f"{name}: already used: {clash}")
    if set(EDU_TOPICS) != set(synth.SUBJECTS) or set(COURT_SCENES) != set(synth.COURTS):
        raise ValueError("subjects and courts must be synth's")


def plan() -> list[dict]:
    """Every writer request with its track, intended kind, unit and prompt."""
    from bench.hakembench import generate as G

    check_disjoint()
    out = []
    for subject, topics in EDU_TOPICS.items():
        for i, topic in enumerate(topics):
            prompt = (
                f"{synth.education_request(subject, topic, i)} {PLACEHOLDER_RULE} {HISTORY_RULE} "
                f"{CONTENT_RULE}"
            )
            out.append({"id": request_id("egitim", subject, topic), "track": "egitim",
                        "kind": f"ders:{subject}", "scene": topic, "shape": "exam",
                        "prompt": prompt})  # fmt: skip
    for kind in G.MODERATION_KINDS:
        for form in MODERATION_FORMS:
            prompt = (
                f"{MODERATION_PER_REQUEST} farklı Türkçe {form} yaz. Tür: {kind.brief}. Her biri "
                "gerçekçi, birbirinden farklı konuda, üslupta ve uzunlukta olsun; bazıları kısa, "
                "bazıları birkaç cümle. Türün adını metnin içinde söyleme. Metinlerin yaklaşık "
                "dörtte birinde birine adıyla ([ad]) seslen ya da ondan söz et, geri kalanında "
                f"etme. {PLACEHOLDER_RULE} {CONTENT_RULE} {JSON_LIST}"
            )
            out.append({"id": request_id("moderasyon", kind.name, form), "track": "moderasyon",
                        "kind": kind.name, "scene": form, "shape": "list",
                        "prompt": prompt})  # fmt: skip
    for kind in G.SUPPORT_KINDS:
        for sector in SUPPORT_SECTORS:
            prompt = (
                f"{SUPPORT_PER_REQUEST} farklı Türkçe müşteri hizmetleri soru-cevap çifti yaz. "
                f"Sektör: {sector}. Bir müşteri soruyor, şirketin destek ekibi cevaplıyor. "
                f"Cevabın türü: {kind.brief}. Sorular gerçekçi, birbirinden farklı konularda ve "
                "farklı uzunlukta olsun. Cevabın türünü metinde söyleme. Cevapların yaklaşık "
                "yarısında bir [telefon] ya da [bağlantı] geçsin, yarısında geçmesin. "
                f"{PLACEHOLDER_RULE} {CONTENT_RULE} {JSON_QA}"
            )
            out.append({"id": request_id("sss", kind.name, sector), "track": "sss",
                        "kind": kind.name, "scene": sector, "shape": "qa",
                        "prompt": prompt})  # fmt: skip
    for court, scenes in COURT_SCENES.items():
        desc = synth.COURTS[court][0]
        for scene in scenes:
            prompt = (
                f"{synth.legal_request(court, desc, scene, 1)} {PLACEHOLDER_RULE} {CONTENT_RULE}"
            )
            out.append({"id": request_id("hukuk", court, scene), "track": "hukuk", "kind": court,
                        "scene": scene, "shape": "list", "prompt": prompt})  # fmt: skip
    return out


# screening ------------------------------------------------------------------


def mask(text: str) -> str:
    for pattern, replacement in MASKS:
        text = pattern.sub(replacement, text)
    return harmonise(text)


def screen(text: str, shape: str) -> str | None:
    """Why a written text is dropped, or None when it is kept."""
    if not 15 <= len(text) <= 1500:
        return "length"
    if any(m.group(1).strip() not in PLACEHOLDERS for m in BRACKET.finditer(text)):
        return "unknown bracket"
    if REAL_NAMES.search(text):
        return "real name: " + REAL_NAMES.search(text).group(0).lower()
    if T.ADULT.search(text):
        return "adult"
    if T.GAMBLING.search(text):
        return "gambling"
    if shape == "qa":
        question, _, answer = text.partition("\nCevap: ")
        if T.POINTER.search(text):
            return "pointer"
        if answer and T.is_stub(question.removeprefix("Soru: "), answer):
            return "stub"
    return None


def texts_of(request: dict, output) -> tuple[list[dict], Counter]:
    """The kept texts of one request's written output, and why the others went."""
    dropped: Counter = Counter()
    rows = []
    if request["shape"] == "exam":
        if not isinstance(output, dict):
            return [], Counter({"malformed": 1})
        question = str(output.get("soru", "")).strip()
        answers = {int(a.get("seviye", -1)): str(a.get("metin", "")).strip()
                   for a in output.get("cevaplar") or [] if isinstance(a, dict)}  # fmt: skip
        if not question or sorted(answers) != [0, 1, 2, 3]:
            return [], Counter({"malformed exam": 1})
        exam = []
        for level, answer in sorted(answers.items()):
            text = mask(f"Soru: {question}\nCevap: {answer}")
            why = screen(text, "exam")
            if why:
                # One lost answer drops the exam: its siblings are graded against each other.
                return [], Counter({f"exam dropped whole ({why})": 1})
            exam.append({"text": text, "kind": f"seviye{level}", "level": level})
        return [{**r, "subject": request["kind"]} for r in exam], dropped
    for entry in output if isinstance(output, list) else []:
        if not isinstance(entry, dict):
            dropped["malformed"] += 1
            continue
        if request["shape"] == "qa":
            q, a = str(entry.get("soru", "")).strip(), str(entry.get("cevap", "")).strip()
            body = f"Soru: {q}\nCevap: {a}" if q and a else ""
        else:
            body = str(entry.get("metin", "")).strip()
        text = mask(body)
        why = screen(text, request["shape"]) if body else "empty"
        if why:
            dropped[why] += 1
            continue
        rows.append({"text": text, "kind": request["kind"]})
    return rows, dropped


def item_id(track: str, text: str) -> str:
    return f"{track}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def deduplicated(rows: list[dict], known: list[str]) -> tuple[list[dict], Counter]:
    """Drop exact and MinHash near duplicates within the round and of the first round."""
    lsh, wholes, kept, dropped = MinHashLSH(), set(), [], Counter()
    for n, text in enumerate(known):
        wholes.add(whole_hash(tokens(text)))
        signature = minhash_signature(text)
        if signature is not None:
            lsh.add(f"k:{n}", signature)
    for n, row in enumerate(rows):
        whole = whole_hash(tokens(row["text"]))
        signature = minhash_signature(row["text"])
        if whole in wholes or (signature is not None and lsh.query(signature)):
            dropped["duplicate"] += 1
            continue
        wholes.add(whole)
        if signature is not None:
            lsh.add(f"r:{n}", signature)
        kept.append(row)
    return kept, dropped


FILE_OF = {"egitim": "egitim", "moderasyon": "moderasyon", "sss": "sss", "hukuk": "hukuk-gen"}


def first_round(track: str) -> tuple[list[dict], list[dict]]:
    name = FILE_OF[track]
    return read_jsonl(CANDIDATES / f"{name}.jsonl"), read_jsonl(CANDIDATES / f"{name}.meta.jsonl")


def questions_of(track: str) -> dict:
    """The first round's questions for the track, copied from an authored item."""
    items, metas = first_round(track)
    authored = {m["id"] for m in metas if m.get("source") == "generated"}
    return next(i["questions"] for i in items if i["id"] in authored)


def stratum_of(track: str, meta: dict) -> str:
    return meta.get("subject") or meta["kind"] if track == "egitim" else meta["kind"]


def halves_for(track: str, new_units: dict[str, tuple[str, int]]) -> dict[str, str]:
    """New units to halves.

    Education and legal questions: balanced per stratum against the first round's
    authored items, which keep their halves. Moderation and support: the first round's
    authored items were all public, so the new writer's units split evenly among
    themselves and resplit.py splits the first round's by request the same way.
    """
    if track in ("moderasyon", "sss"):
        return unit_halves(new_units, global_ties=True)
    _items, metas = first_round(track)
    held: dict[str, dict[str, int]] = defaultdict(lambda: {"public": 0, "private": 0})
    for m in metas:
        if m.get("source") == "generated" and m.get("split") in ("public", "private"):
            held[stratum_of(track, m)][m["split"]] += 1
    return unit_halves(new_units, held=held, global_ties=True)


def build(requests: list[dict], outputs: dict[str, object], data_root: Path) -> tuple[dict, dict]:
    """Per track: items and metas for the new files, and a summary of what was dropped."""
    by_track: dict[str, list[dict]] = defaultdict(list)
    dropped: dict[str, Counter] = defaultdict(Counter)
    for request in requests:
        if request["id"] not in outputs:
            dropped[request["track"]]["request not written"] += 1
            continue
        rows, why = texts_of(request, outputs[request["id"]])
        dropped[request["track"]].update(why)
        by_track[request["track"]] += [{**r, "request": request["id"], "scene": request["scene"],
                                        "shape": request["shape"]} for r in rows]  # fmt: skip
    files = training_files(data_root)
    out, summary = {}, {}
    for track, rows in by_track.items():
        items_old, _ = first_round(track)
        rows, why = deduplicated(rows, [i["state"] for i in items_old])
        dropped[track].update(why)
        texts = {item_id(track, r["text"]): r["text"] for r in rows}
        touched = overlapping(texts, training_texts(files, data_root))
        lost_exams = {r["request"] for r in rows
                      if touched.get(item_id(track, r["text"]), {}).get("dropped")
                      and r["shape"] == "exam"}  # fmt: skip
        kept = []
        for r in rows:
            hit = touched.get(item_id(track, r["text"]))
            if hit is not None and hit["dropped"]:
                dropped[track]["overlaps training: " + "+".join(sorted(hit["rules"]))] += 1
            elif r["request"] in lost_exams:
                dropped[track]["exam dropped whole (a sibling overlaps training)"] += 1
            else:
                kept.append(r)
        questions = questions_of(track)
        units: dict[str, tuple[str, int]] = {}
        for r in kept:
            stratum = r.get("subject") or r["kind"]
            s, n = units.get(r["request"], (stratum, 0))
            units[r["request"]] = (s, n + 1)
        half = halves_for(track, units)
        items, metas = [], []
        for r in sorted(kept, key=lambda r: item_id(track, r["text"])):
            uid = item_id(track, r["text"])
            items.append({"id": uid, "track": track, "state": r["text"], "questions": questions,
                          "gold": {}})  # fmt: skip
            metas.append({"id": uid, "track": track, "source": "generated", "licence": LICENCE,
                          "split": half[r["request"]], "kind": r["kind"],
                          "subject": r.get("subject"), "scene": r["scene"], "writer": WRITER,
                          "request": r["request"], "cluster": r["request"], "round": ROUND,
                          "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest(),
                          })  # fmt: skip
        out[track] = (items, metas)
        summary[track] = {"written": len(rows) + sum(dropped[track].values()), "kept": len(items),
                          "dropped": dict(dropped[track]),
                          "halves": dict(Counter(m["split"] for m in metas)),
                          "kinds": dict(Counter(m["kind"] for m in metas))}  # fmt: skip
    return out, summary


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def removed() -> set[str]:
    out = set()
    for path in BROWSER_REMOVALS.glob("*.json"):
        out.add(json.loads(path.read_text(encoding="utf-8"))["item"])
    return out


# the model passes and gold --------------------------------------------------

KNOWLEDGE_QIDS = ("cevap_puani", "ders", "basvuru_yeri")
JUDGES = ("A", "B", "C")
AUDIT_OVERRULED, AUDIT_UNANIMOUS = 40, 30


def topup_items() -> tuple[list[dict], dict[str, dict]]:
    items, metas = [], {}
    for path in sorted(CANDIDATES.glob("*-topup.jsonl")):
        items += read_jsonl(path)
        metas |= {m["id"]: m for m in read_jsonl(path.with_suffix("").with_suffix(".meta.jsonl"))}
    gone = removed()
    return [i for i in items if i["id"] not in gone], metas


def units(items: list[dict]) -> list[dict]:
    """Blind units for the model passes: text, questions and options, nothing else."""
    from bench.hakembench.desk import options

    return [{"id": i["id"], "track": i["track"], "text": i["state"],
             "questions": [{"qid": qid, "question": q["instructions"], "type": q["type"],
                            "options": options(q)} for qid, q in i["questions"].items()]}
            for i in items]  # fmt: skip


def judge_answers(meta: dict, qid: str) -> list[str | None]:
    """Each panel judge's own answer (its highest-probability option), None on a tie."""
    judges = meta.get("votes", {}).get(qid, {}).get("judges") or {}
    out = []
    for name in JUDGES:
        dist = judges.get(name) or {}
        if not dist:
            out.append(None)
            continue
        best = max(dist.values())
        top = [k for k, v in dist.items() if v == best]
        out.append(top[0] if len(top) == 1 else None)
    return out


def said(passes: dict, uid: str, qid: str) -> str | None:
    entry = passes.get(uid, {}).get(qid)
    if not entry or entry.get("flag", "none") != "none":
        return None
    answer = entry.get("answer")
    return None if answer is None else str(answer)


def decide(panel: list[str | None], family: list[str | None], qid: str, track: str,
           redecided: str | None, adjudicated: str | None) -> tuple[str | None, str]:  # fmt: skip
    """The gold answer and its source under the rule, or (None, why) when not yet decided."""
    if track == "sss" and qid not in KNOWLEDGE_QIDS:
        # Every support question reads the owner's written procedure, as the first round's did.
        return (redecided, "ai_owner_reading") if redecided else (None, "needs owner reading")
    votes = panel + family
    if None not in votes and len(set(votes)) == 1:
        return votes[0], "ai_ensemble_unanimous"
    if None not in panel and len(set(panel)) == 1 and panel[0] not in family:
        # Three judges from other families agree against the writer's family: theirs stands.
        return panel[0], "llm_panel_unanimous"
    if adjudicated:
        return adjudicated, "writer_family_adjudicated"
    return None, "needs adjudication"


def read_passes(folder: Path | None) -> dict:
    out: dict = {}
    if folder is not None:
        for path in sorted(folder.glob("out_*.json")):
            out |= json.loads(path.read_text(encoding="utf-8"))
    return out


def gold(items, metas, research, second, redecided, adjudicated) -> tuple[dict, dict, list]:
    """Gold per question, the questions still open by reason, and the overruled questions."""
    out, open_, overruled = {}, defaultdict(list), []
    for item in items:
        uid = item["id"]
        for qid in item["questions"]:
            panel = judge_answers(metas[uid], qid)
            family = [said(research, uid, qid), said(second, uid, qid)]
            owners = said(redecided, uid, qid)
            answer, source = decide(panel, family, qid, item["track"], owners,
                                    said(adjudicated, uid, qid))  # fmt: skip
            if answer is None:
                open_[source].append(f"{uid}~{qid}")
                continue
            out[f"{uid}~{qid}"] = {"answer": answer, "source": source}
            majority = Counter(a for a in panel if a is not None).most_common(1)
            if majority and majority[0][1] >= 2 and majority[0][0] != answer:
                overruled.append(f"{uid}~{qid}")
    return out, dict(open_), overruled


def adjudication_units(items, metas, research, second) -> list[dict]:
    """Questions no rule settles: text, question, the distinct answers and both reasons."""
    out = []
    for u in units(items):
        questions = []
        for q in u["questions"]:
            qid, uid = q["qid"], u["id"]
            if u["track"] == "sss" and qid not in KNOWLEDGE_QIDS:
                continue
            panel = judge_answers(metas[uid], qid)
            family = [said(research, uid, qid), said(second, uid, qid)]
            answer, _ = decide(panel, family, qid, u["track"], None, None)
            if answer is not None:
                continue
            reasons = [p.get(uid, {}).get(qid, {}).get("reason_tr") for p in (research, second)]
            reasons = [r for r in reasons if r]
            questions.append({**q, "candidates": sorted({a for a in panel + family if a}),
                              "reasons": reasons})  # fmt: skip
        if questions:
            out.append({**u, "questions": questions})
    return out


def audit_sample(gold_: dict, overruled: list[str], seed: int = 127) -> dict[str, list[str]]:
    """The owner's blind audit: every overruled question up to 40, and 30 unanimous ones,
    leaving out knowledge questions (the owner's answers there are audit only)."""
    import random

    rng = random.Random(seed)
    keep = [k for k in overruled if k.split("~", 1)[1] not in KNOWLEDGE_QIDS]
    chosen_over = sorted(keep) if len(keep) <= AUDIT_OVERRULED else sorted(
        rng.sample(sorted(keep), AUDIT_OVERRULED))  # fmt: skip
    pool: dict[str, list[str]] = defaultdict(list)
    for key, g in gold_.items():
        qid = key.split("~", 1)[1]
        if (
            g["source"] in ("ai_ensemble_unanimous", "ai_owner_reading")
            and qid not in KNOWLEDGE_QIDS
            and key not in chosen_over
        ):
            pool[qid].append(key)
    unanimous: list[str] = []
    qids = sorted(pool)
    while len(unanimous) < AUDIT_UNANIMOUS and any(pool.values()):
        for qid in qids:
            if pool[qid] and len(unanimous) < AUDIT_UNANIMOUS:
                pick = rng.choice(sorted(pool[qid]))
                pool[qid].remove(pick)
                unanimous.append(pick)
    return {"overruled": chosen_over, "unanimous": sorted(unanimous)}


def batches(rows: list, n: int) -> list[list]:
    return [rows[k::n] for k in range(n)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.hakembench.topup")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    b = sub.add_parser("build")
    b.add_argument("--writers", type=Path, required=True)
    b.add_argument("--data", type=Path, default=Path("data/built"))
    b.add_argument(
        "--tracks",
        default="egitim,moderasyon,sss,hukuk",
        help="build only these; a file with panel votes is never rebuilt",
    )
    b.add_argument("--suffix", default="topup", help="'pilot' for the moderation pilot")
    pi = sub.add_parser("passes-input")
    pi.add_argument("--out", type=Path, required=True)
    pi.add_argument("--batches", type=int, default=8)
    for name in ("adjudication-input", "gold"):
        c = sub.add_parser(name)
        c.add_argument("--research", type=Path, required=True)
        c.add_argument("--second", type=Path, required=True)
        c.add_argument("--redecided", type=Path)
        c.add_argument("--adjudicated", type=Path)
        c.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command in ("passes-input", "adjudication-input", "gold"):
        return later(args)
    WORK.mkdir(parents=True, exist_ok=True)
    requests = plan()
    if args.command == "plan":
        (WORK / "plan.json").write_text(json.dumps(requests, ensure_ascii=False, indent=1))
        batches = [
            requests[k : k + BATCH_REQUESTS] for k in range(0, len(requests), BATCH_REQUESTS)
        ]
        for n, batch in enumerate(batches):
            (WORK / f"requests_{n:02d}.json").write_text(json.dumps(
                [{"id": r["id"], "shape": r["shape"], "prompt": r["prompt"]} for r in batch],
                ensure_ascii=False, indent=1))  # fmt: skip
        print(json.dumps({"requests": len(requests), "batches": len(batches),
                          "per_track": dict(Counter(r["track"] for r in requests))}))  # fmt: skip
        return 0
    outputs = {}
    for path in sorted(args.writers.glob("w_*.json")):
        outputs |= json.loads(path.read_text(encoding="utf-8"))
    tracks = set(args.tracks.split(","))
    built, summary = build([r for r in requests if r["track"] in tracks], outputs, args.data)
    for track, (items, metas) in built.items():
        meta_path = CANDIDATES / f"{FILE_OF[track]}-{args.suffix}.meta.jsonl"
        if meta_path.is_file() and any("votes" in m for m in read_jsonl(meta_path)):
            raise SystemExit(f"{meta_path} already holds panel votes; it is not rebuilt")
        write_jsonl(CANDIDATES / f"{FILE_OF[track]}-{args.suffix}.jsonl", items)
        write_jsonl(meta_path, metas)
    (WORK / f"build_summary_{args.suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


def later(args) -> int:
    """The commands that run on the built top-up files."""
    items, metas = topup_items()
    if args.command == "passes-input":
        args.out.mkdir(parents=True, exist_ok=True)
        blind = units(items)
        for n, batch in enumerate(batches(blind, args.batches)):
            (args.out / f"batch_{n:02d}.json").write_text(json.dumps(batch, ensure_ascii=False))
        sss = [u for u in blind if u["track"] == "sss"]
        for n, batch in enumerate(batches(sss, 4)):
            (args.out / f"sss_{n:02d}.json").write_text(json.dumps(batch, ensure_ascii=False))
        print(json.dumps({"units": len(blind), "sss_units": len(sss)}))
        return 0
    research, second = read_passes(args.research), read_passes(args.second)
    if args.command == "adjudication-input":
        todo = adjudication_units(items, metas, research, second)
        args.out.mkdir(parents=True, exist_ok=True)
        for n, batch in enumerate(batches(todo, 4)):
            (args.out / f"batch_{n:02d}.json").write_text(json.dumps(batch, ensure_ascii=False))
        print(json.dumps({"units": len(todo),
                          "questions": sum(len(u["questions"]) for u in todo)}))  # fmt: skip
        return 0
    redecided, adjudicated = read_passes(args.redecided), read_passes(args.adjudicated)
    decided, open_, overruled = gold(items, metas, research, second, redecided, adjudicated)
    if open_:
        print(json.dumps({"open": {k: len(v) for k, v in open_.items()}}), file=sys.stderr)
        return 1
    (WORK / "gold.json").write_text(json.dumps(decided, ensure_ascii=False, indent=1))
    sample = audit_sample(decided, overruled)
    (WORK / "audit_topup.json").write_text(json.dumps(sample, indent=1))
    print(json.dumps({"questions": len(decided),
                      "sources": dict(Counter(g["source"] for g in decided.values())),
                      "overruled": len(overruled),
                      "audit": {k: len(v) for k, v in sample.items()}}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
