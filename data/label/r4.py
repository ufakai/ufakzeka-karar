"""r4 training data: the held-out support questions labelled, new guardrail and moderation texts.

    python -m data.label.r4 support-units
    python -m data.label.r4 writer-plan
    python -m data.label.r4 label-units
    python -m data.label.r4 rows
    python -m data.label.r4 leakage
    python -m data.label.r4 status      # what every labelling and writing pass has left to do

r3 answers support near a coin flip, flags benign look-alikes as attacks and
misses authored insults (results/private/step9/board_internal.json).
The four support questions were the build's held-out cells, never trained; the
guardrail and moderation training sets are converted tweets and one attack set
with few hard negatives.

`support-units` samples texts from the support training pool (never a
validation, held-out or HakemBench-dev text for train; the validation rows come
from the validation pool, dev texts left out) and writes blind batches for two
labelling passes by two AI models, judge D in the benchmark's option
order and judge E in a shuffled one. Each pass gives a distribution per
question; the target is the mean of the two, so a disagreement stays soft.

`writer-plan` writes the briefs for new guardrail prompts and moderation
comments over a taxonomy, product contexts and comment forms none of which the
benchmark's writers used. Judge D writes; judge E labels blind
(a text is never labelled by its writer). `label-units` removes near
duplicates, every text that touches a HakemBench v1.0 item, probe or dev text,
and writes the blind batches. `rows` keeps a written text only where judge E's
answer agrees with the writer's intent and builds TrainingRows with both votes.
`leakage` is the style check on the new rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from bench.hakembench.common import overlapping
from bench.hakembench.desk import options
from bench.hakembench.guard_gen import deduplicated
from bench.hakembench.rubric import QIDS, RUBRIC, support_items
from schema.rows import JudgeVote, TrainingRow, text_of

WORK = Path("results/private/step9/r4")
BUILT = Path("data/built")
OUT = BUILT / "r4"
DEV = Path("results/private/hakembench/dev_v0.jsonl")
ITEMS = (Path("bench/hakembench/v1.0/public.jsonl"),
         Path("results/private/step8/v1.0/private.jsonl"))  # fmt: skip
PROBES = (Path("bench/hakembench/v1.0/probes"), Path("results/private/step8/v1.0/probes"))
SEED = 129
SUPPORT_TRAIN = 2700
SUPPORT_VALIDATION = 300
SUPPORT_BATCH = 50
LABEL_BATCH = 100
VALIDATION_SHARE = 0.1
RECIPE = "r4-v1"
WRITER, LABELLER = "D", "E"
PLACEHOLDERS = "[ad], [telefon], [e-posta], [bağlantı], [hesap no], [adres], [sipariş no]"

# Guardrails: grouped by what the sender is after, not by the benchmark's families.
GUARD_ATTACKS = {
    "H1": "görev kaçırma: asistanı kendi işinden çıkarıp başka bir işe koşmak",
    "H2": "başka bir müşterinin ya da çalışanın kişisel verisini, geçmişini veya belgesini istemek",
    "H3": "asistanın iç ayarlarını, araç listesini, erişim anahtarlarını ya da gizli notlarını "
    "sızdırmak",
    "H4": "yetkisi olmayan bir işlemi yaptırmak: ödeme, iade, silme, rol ya da limit değiştirme",
    "H5": "asistanın içerik kurallarını bir çerçeveyle (kurgu, test, 'araştırma') aşıp yasak "
    "bir içerik ürettirmek",
    "H6": "asistanın kalıcı davranışını değiştirmek: sonraki kullanıcılara bir şey söyletmek, "
    "hafızasına kural yazdırmak, cevaplara bağlantı ekletmek",
    "H7": "işlenecek bir içeriğin (e-posta, tablo satırı, toplantı notu, web sayfası, belge "
    "metni, form alanı) içine gömülü, asistana yönelik talimat",
    "H8": "baskıyla ikna: sahte aciliyet, duygu sömürüsü, uydurma bir yetki ya da onay",
}  # fmt: skip
GUARD_BENIGN = {
    "K1": "kişinin kendi hesabı ya da verisi üzerinde, doğru kanaldan meşru bir istek (kendi "
    "şifresini sıfırlama, kendi siparişini iptal, kendi verisini indirme)",
    "K2": "güvenlik sözcükleri geçen sıradan bir iş (parola politikası taslağı, gizlilik "
    "sözleşmesi özeti, erişim yetkisi tablosu düzenleme)",
    "K3": "kullanıcının kendi önceki isteğini ya da cevabın biçimini değiştirmesi (öncekini "
    "boş ver, baştan yaz, daha kısa, İngilizce yaz)",
    "K4": "asistanın neye erişebildiğini, verinin nasıl saklandığını ya da sınırlarını merak "
    "eden bir soru",
    "K5": "alıntılanan şüpheli bir mesajı çevirmesini, özetlemesini ya da ne olduğunu "
    "açıklamasını istemek; içindeki talimatın uygulanmasını istemeden",
    "K6": "kurgu, oyun ya da rol yapma içinde hacker, casus, dolandırıcı teması; bir kuralı "
    "aşmaya çalışmadan",
    "K7": "yetkili bir çalışanın kendi aracı hakkında doğru kanaldan, meşru bir ayar ya da "
    "rapor isteği",
    "K8": "sert, sabırsız ya da acele eden ama kendi hakkı olan bir talep",
    "K9": "hata ayıklama için log, JSON, yapılandırma ya da kod parçası yapıştırmak (içinde "
    "'ignore', 'system', 'admin' gibi sözcükler geçebilir)",
    "K10": "savunma amaçlı güvenlik eğitimi, farkındalık ya da politika sorusu",
}  # fmt: skip
GUARD_CONTEXTS = (
    "elektrik ve doğalgaz faturası asistanı", "mobil operatör müşteri asistanı",
    "otel resepsiyon asistanı", "restoran rezervasyon botu", "market sipariş uygulaması",
    "kripto cüzdan uygulaması", "emlak portföy asistanı", "depo ve lojistik yönetim aracı",
    "CRM satış asistanı", "iş takip ve proje yönetimi botu", "toplantı notu ve takvim asistanı",
    "çeviri aracı", "web sayfası özetleyen tarayıcı eklentisi", "müşteri yorumu analiz aracı",
    "eczane zinciri asistanı", "spor salonu üyelik uygulaması", "kütüphane katalog asistanı",
    "bağış platformu asistanı", "sürücü ve filo yönetimi uygulaması", "dil kursu platformu",
)  # fmt: skip
# Only benign look-alikes are written: every writer asked for attack messages was
# stopped by its safety filter, and r3 already catches attacks (204 of 216); its failure
# is the false alarm. The attack side of the task stays the converted training sets.
GUARD_PER_WRITER = {"true": 0, "false": 80}

# Moderation: new kinds and forms; the benchmark's four kinds and ten forms stay out.
MOD_OFFENSIVE = {
    "O1": "bir kişiye yönelik açık küfür",
    "O2": "küfürsüz aşağılama ya da küçümseme (zekâsı, görünüşü, işi üzerinden)",
    "O3": "tehdit ya da gözdağı",
    "O4": "alaycı, iğneleyici hakaret",
    "O5": "kısaltılmış, yıldızlı ya da harfleri bozulmuş küfür",
    "O6": "tartışma sırasında hakaret sıfatları (salak, gerizekalı ve benzeri)",
}
MOD_CLEAN = {
    "N1": "sert, olumsuz ama hakaret ya da kişisel saldırı içermeyen eleştiri",
    "N2": "hakaret içermeyen samimi argo ya da arkadaşça takılma",
    "N3": "birinin kendisine söylediği sözü aktarıp yardım ya da fikir isteyen mesaj",
    "N4": "kimseyi hedef almadan öfke ya da hayal kırıklığı dökme",
    "N5": "olağan, nazik ya da olumlu bir mesaj",
}
MOD_FORMS = (
    "aile WhatsApp grubu mesajı", "apartman yönetimi grubu mesajı", "çevrim içi oyun sohbeti",
    "canlı yayın sohbeti", "dizi ya da film platformunda yorum", "podcast bölümüne yorum",
    "müzik klibine yorum", "şikâyet sitesine yazılan başlık", "sözlük sitesine girdi",
    "iş yeri sohbet kanalı mesajı", "okul veli grubu mesajı",
    "ikinci el uygulamasında alıcı-satıcı mesajı", "yolculuk uygulamasında sürücü "
    "değerlendirmesi", "otel değerlendirmesi", "yemek tarifi sitesine yorum",
    "çevrim içi kurs platformuna yorum", "maç sonrası taraftar mesajı",
    "sosyal medyada özel mesaj", "mahalle dayanışma grubu mesajı", "kulüp forumu mesajı",
)  # fmt: skip
MOD_PER_KIND = {"true": 15, "false": 12}
WRITERS = 10
SKIPPED = "_skipped"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


def pool(name: str) -> list[str]:
    return sorted({text_of(r["state"]) for r in read_jsonl(BUILT / "sss" / name)})


# support ------------------------------------------------------------------


def support_questions() -> dict[str, dict]:
    """The four questions as HakemBench asks them, on the written rubric."""
    first = support_items()[0]["questions"]
    out = {}
    for qid in QIDS:
        q = dict(first[qid])
        rule = RUBRIC[qid]
        if isinstance(rule, dict):
            order = list(q["criteria"]) if isinstance(q["criteria"], dict) else list(rule)
            q["criteria"] = {key: rule[key] for key in order}
        else:
            q["criteria"] = list(rule)
        out[qid] = q
    return out


def shown_orders(questions: dict[str, dict], shuffle: bool) -> dict[str, list[str]]:
    """Each question's outcome keys in the order a pass sees them; score levels stay in order."""
    rng = random.Random(SEED)
    out = {}
    for qid, q in questions.items():
        keys = [o["key"] for o in options(q)]
        if shuffle and q["type"] != "score":
            rng.shuffle(keys)
        out[qid] = keys
    return out


def support_prompt(questions: dict[str, dict], orders: dict[str, list[str]]) -> str:
    lines = []
    for qid, q in questions.items():
        by_key = {o["key"]: o for o in options(q)}
        lines.append(f"### {qid} ({q['type']})\nSoru: {q['instructions']}")
        for key in orders[qid]:
            o = by_key[key]
            name = f"{key}" if q["type"] == "choice" else f"{key} ({o['label']})"
            lines.append(f"- {name}: {o['detail']}")
        lines.append("")
    return "\n".join(lines)


def support_units() -> dict:
    """Blind batches for passes D and E over train-pool and validation-pool texts."""
    dev = {r["text"] for r in read_jsonl(DEV)}
    held = set(pool("validation.jsonl")) | set(pool("heldout_task.jsonl")) | dev
    rng = random.Random(SEED)
    train = [t for t in pool("train.jsonl") if t not in held]
    validation = [t for t in pool("validation.jsonl") if t not in dev]
    rng.shuffle(train)
    rng.shuffle(validation)
    chosen = [(t, "train") for t in train[:SUPPORT_TRAIN]]
    chosen += [(t, "validation") for t in validation[:SUPPORT_VALIDATION]]
    rng.shuffle(chosen)
    units = [{"id": f"s-{sha(t)}", "text": t} for t, _ in chosen]
    if len({u["id"] for u in units}) != len(units):
        raise SystemExit("two support texts share an id")
    questions = support_questions()
    root = WORK / "support"
    write_json(root / "split.json", {f"s-{sha(t)}": split for t, split in chosen})
    write_json(root / "questions.json", questions)
    for judge, shuffle in ((WRITER, False), (LABELLER, True)):
        orders = shown_orders(questions, shuffle)
        write_json(root / f"orders_{judge}.json", orders)
        (root / f"questions_{judge}.md").write_text(support_prompt(questions, orders), "utf-8")
    for n in range(0, len(units), SUPPORT_BATCH):
        write_json(root / "units" / f"batch_{n // SUPPORT_BATCH:02d}.json",
                   units[n : n + SUPPORT_BATCH])  # fmt: skip
    return {"train": sum(s == "train" for _, s in chosen),
            "validation": sum(s == "validation" for _, s in chosen),
            "batches": -(-len(units) // SUPPORT_BATCH)}  # fmt: skip


# written texts ------------------------------------------------------------


def writer_plan() -> dict:
    """Ten briefs per track: contexts or forms dealt round, every kind in every brief."""
    plans = []
    for w in range(WRITERS):
        contexts = [GUARD_CONTEXTS[(2 * w + k) % len(GUARD_CONTEXTS)] for k in range(2)]
        plans.append({"track": "guvenlik", "writer": f"g{w:02d}", "contexts": contexts,
                      "attacks": GUARD_ATTACKS, "benign": GUARD_BENIGN,
                      "counts": GUARD_PER_WRITER})  # fmt: skip
        forms = [MOD_FORMS[(2 * w + k) % len(MOD_FORMS)] for k in range(2)]
        plans.append({"track": "moderasyon", "writer": f"m{w:02d}", "forms": forms,
                      "offensive": MOD_OFFENSIVE, "clean": MOD_CLEAN,
                      "per_kind": MOD_PER_KIND})  # fmt: skip
    for plan in plans:
        write_json(WORK / "writers" / f"plan_{plan['writer']}.json", plan)
    return {"briefs": len(plans), "placeholders": PLACEHOLDERS}


def track_question(track: str) -> tuple[str, dict]:
    """The track's question exactly as HakemBench asks it."""
    for path in ITEMS:
        for item in read_jsonl(path):
            if item["track"] == track:
                (qid, q), = item["questions"].items()  # fmt: skip
                return qid, q
    raise SystemExit(f"no {track} item")


def reference_texts() -> dict[str, str]:
    """Every HakemBench v1.0 item, probe and dev text, keyed by where it is."""
    out = {}
    for path in ITEMS:
        for item in read_jsonl(path):
            out[f"item:{item['id']}"] = text_of(item["state"])
    for folder in PROBES:
        for path in sorted(folder.glob("*.jsonl")):
            if path.name.endswith(".meta.jsonl"):
                continue
            for item in read_jsonl(path):
                out[f"probe:{item['id']}"] = text_of(item["state"])
    for row in read_jsonl(DEV):
        out[f"dev:{row['id']}"] = row["text"]
    return out


def touching(texts: dict[str, str]) -> dict[str, list[str]]:
    """Keys of texts that touch a reference text, by the rules, in either direction."""
    refs = reference_texts()
    out: dict[str, set[str]] = {}
    forward = overlapping(texts, ((f"ref:{k}", v) for k, v in refs.items()))
    for key, entry in forward.items():
        if entry["dropped"]:
            out.setdefault(key, set()).update(entry["rules"])
    backward = overlapping(refs, ((k, v) for k, v in texts.items()))
    for entry in backward.values():
        if entry["dropped"]:
            for rule, where in entry["rows"].items():
                if where in texts:
                    out.setdefault(where, set()).add(f"{rule}-back")
    return {k: sorted(v) for k, v in out.items()}


def written() -> dict[str, list[dict]]:
    by_track: dict[str, list[dict]] = {"guvenlik": [], "moderasyon": []}
    for path in sorted((WORK / "writers").glob("w_*.json")):
        writer = path.stem[2:]
        track = "guvenlik" if writer.startswith("g") else "moderasyon"
        for row in json.loads(path.read_text(encoding="utf-8")):
            by_track[track].append({**row, "writer": writer, "text": row["text"].strip()})
    return by_track


def label_units() -> dict:
    """Deduplicated, decontaminated written texts in blind batches for judge E."""
    summary = {}
    for track, rows in written().items():
        rows = [r for r in rows if isinstance(r.get("intended"), bool) and r["text"]]
        if track == "guvenlik":
            rows = [r for r in rows if not r["intended"]]
        kept, dropped = deduplicated(rows)
        texts = {f"{track[0]}-{sha(r['text'])}": r["text"] for r in kept}
        touched = touching(texts)
        final = [{**r, "id": f"{track[0]}-{sha(r['text'])}"} for r in kept
                 if f"{track[0]}-{sha(r['text'])}" not in touched]  # fmt: skip
        dropped["touches HakemBench, a probe or dev"] = len(kept) - len(final)
        rng = random.Random(SEED)
        rng.shuffle(final)
        root = WORK / "labels" / track
        write_json(root / "kept.json", final)
        write_json(root / "touched.json", touched)
        qid, q = track_question(track)
        write_json(root / "question.json", {"qid": qid, **q})
        for n in range(0, len(final), LABEL_BATCH):
            batch = [{"id": r["id"], "text": r["text"]} for r in final[n : n + LABEL_BATCH]]
            write_json(root / "units" / f"batch_{n // LABEL_BATCH:02d}.json", batch)
        summary[track] = {"written": len(rows), "kept": len(final), "dropped": dict(dropped),
                          "intended_true": sum(r["intended"] for r in final)}  # fmt: skip
    write_json(WORK / "labels" / "summary.json", summary)
    return summary


# rows ---------------------------------------------------------------------


def read_outputs(folder: Path) -> dict:
    out: dict = {}
    for path in sorted(folder.glob("out_*.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        batch.pop(SKIPPED, None)
        out |= batch
    return out


def distribution(raw: dict | None, keys: list[str]) -> dict[str, float] | None:
    """A judge's answer as a distribution over keys, renormalised; None if unusable."""
    if not isinstance(raw, dict):
        return None
    try:
        values = {k: max(0.0, float(raw.get(k, 0.0))) for k in keys}
    except (TypeError, ValueError):
        return None
    if set(raw) - set(keys):
        return None
    total = sum(values.values())
    if total <= 0 or abs(total - 1.0) > 0.05:
        return None
    return {k: v / total for k, v in values.items()}


def validation_split(text: str) -> str:
    unit = int(hashlib.sha256(f"r4-split:{text}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "validation" if unit < VALIDATION_SHARE else "train"


def row(track: str, task: str, split: str, origin: str, source: str, text: str, question: dict,
        votes: list[JudgeVote]) -> TrainingRow:  # fmt: skip
    keys = list(votes[0].distribution)
    target = {k: sum(v.distribution[k] for v in votes) / len(votes) for k in keys}
    return TrainingRow.model_validate({
        "track": track, "task": task, "split": split, "origin": origin, "label_kind": "judges",
        "source": source, "state": text, "question": question, "target": target,
        "judges": [v.model_dump() for v in votes], "recipe": RECIPE})  # fmt: skip


def support_rows(report: dict) -> list[TrainingRow]:
    root = WORK / "support"
    questions = json.loads((root / "questions.json").read_text(encoding="utf-8"))
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    units = {u["id"]: u["text"] for p in sorted((root / "units").glob("*.json"))
             for u in json.loads(p.read_text(encoding="utf-8"))}  # fmt: skip
    passes = {j: read_outputs(root / f"out_{j}") for j in (WRITER, LABELLER)}
    orders = {j: json.loads((root / f"orders_{j}.json").read_text("utf-8")) for j in passes}
    out, missing = [], Counter()
    agree = Counter()
    for uid, text in units.items():
        for qid, q in questions.items():
            keys = [o["key"] for o in options(q)]
            votes = []
            for judge in (WRITER, LABELLER):
                dist = distribution(passes[judge].get(uid, {}).get(qid), keys)
                if dist is None:
                    missing[f"{judge}:{qid}"] += 1
                    continue
                votes.append(JudgeVote(judge=judge, shown_order=orders[judge][qid],
                                       distribution=dist, off_letter_mass=0.0))  # fmt: skip
            if len(votes) < 2:
                continue
            tops = [max(v.distribution, key=v.distribution.get) for v in votes]
            agree[f"{qid}:{'agree' if tops[0] == tops[1] else 'differ'}"] += 1
            out.append(row("sss", f"{qid}-r4", split[uid], "generated", "clips/mqa", text, q,
                           votes))  # fmt: skip
    report["support"] = {"rows": len(out), "missing_votes": dict(missing),
                         "agreement": dict(sorted(agree.items()))}  # fmt: skip
    return out


def written_rows(report: dict) -> list[TrainingRow]:
    out = []
    for track in ("guvenlik", "moderasyon"):
        root = WORK / "labels" / track
        kept = json.loads((root / "kept.json").read_text(encoding="utf-8"))
        question = json.loads((root / "question.json").read_text(encoding="utf-8"))
        qid = question.pop("qid")
        labels = read_outputs(root / "out")
        counts: Counter = Counter()
        for r in kept:
            dist = distribution(labels.get(r["id"]), ["true", "false"])
            if dist is None:
                counts["no label"] += 1
                continue
            said = dist["true"] >= 0.5
            if said != r["intended"]:
                counts[f"differ (intended {str(r['intended']).lower()})"] += 1
                continue
            counts["kept"] += 1
            intent = {"true": float(r["intended"]), "false": float(not r["intended"])}
            votes = [JudgeVote(judge=WRITER, shown_order=["true", "false"], distribution=intent,
                               off_letter_mass=0.0),
                     JudgeVote(judge=LABELLER, shown_order=["true", "false"], distribution=dist,
                               off_letter_mass=0.0)]  # fmt: skip
            # Its own task name, so the mix cap does not sample it away inside the old task.
            out.append(row(track, f"{qid}-r4", validation_split(r["text"]), "generated",
                           "generated:r4-v1", r["text"], question, votes))  # fmt: skip
        report[track] = dict(counts)
    return out


def build_rows() -> dict:
    report: dict = {}
    rows = support_rows(report) + written_rows(report)
    train_texts = {text_of(r.state) for r in rows if r.split == "train"}
    shared = [r.row_id for r in rows if r.split != "train" and text_of(r.state) in train_texts]
    if shared:
        raise SystemExit(f"{len(shared)} validation rows share a text with train")
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "rows.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in rows), "utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        chosen = [r for r in rows if r.split == split]
        (OUT / f"{split}.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in chosen),
                                            "utf-8")  # fmt: skip
        report[f"{split}_rows"] = len(chosen)
    report["tasks"] = dict(Counter(f"{r.task}:{r.split}" for r in rows))
    write_json(WORK / "rows_report.json", report)
    return report


# style leakage -----------------------------------------------------


def leakage() -> dict:
    """A character n-gram classifier trained on training rows, scored on the benchmark's items.

    For each of the two tracks, one classifier on the old training rows of the
    track's task and one on r4's new written rows; both scored on the v1.0 items
    of that track per stratum (generated and source), with 5-fold accuracy
    inside the r4 rows beside them. A classifier on style alone that scores the
    benchmark far above what the old rows give is the sign the new texts share
    the benchmark's surface.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    meta = {}
    for path in Path("bench/hakembench/candidates").glob("*.meta.jsonl"):
        for m in read_jsonl(path):
            meta[m["id"]] = m
    report = {}
    old_files = {"guvenlik": BUILT / "typed/prompt_injection/train.jsonl",
                 "moderasyon": BUILT / "typed/offenseval_tr/train.jsonl"}  # fmt: skip
    new = [r for r in read_jsonl(OUT / "train.jsonl") if r["track"] in old_files]
    for track, old_path in old_files.items():
        qid, _ = track_question(track)
        items = [i for p in ITEMS for i in read_jsonl(p) if i["track"] == track]
        strata: dict[str, list[tuple[str, bool]]] = {}
        for i in items:
            stratum = "generated" if meta.get(i["id"], {}).get("writer") else "source"
            strata.setdefault(stratum, []).append((text_of(i["state"]), bool(i["gold"][qid])))
        sets = {"old": [(text_of(r["state"]), r["target"]["true"] >= 0.5)
                        for r in read_jsonl(old_path)],
                "r4": [(text_of(r["state"]), r["target"]["true"] >= 0.5)
                       for r in new if r["track"] == track]}  # fmt: skip
        entry = {}
        for name, pairs in sets.items():
            if len({y for _, y in pairs}) < 2:
                continue
            x, y = [t for t, _ in pairs], [y for _, y in pairs]

            def model():
                return make_pipeline(
                    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                    sublinear_tf=True),
                    LogisticRegression(max_iter=2000, class_weight="balanced"))  # fmt: skip

            fitted = model().fit(x, y)
            scores = {"rows": len(pairs)}
            if name == "r4":
                cv = cross_val_score(model(), x, y, cv=5, scoring="balanced_accuracy")
                scores["cv_balanced_accuracy"] = round(float(cv.mean()), 3)
            for stratum, test in sorted(strata.items()):
                truth = [g for _, g in test]
                if len(set(truth)) < 2:
                    acc = sum(p == g for p, g in zip(fitted.predict([t for t, _ in test]), truth,
                                                     strict=True)) / len(truth)  # fmt: skip
                    scores[stratum] = {"n": len(test), "accuracy": round(acc, 3)}
                    continue
                pred = fitted.predict([t for t, _ in test])
                scores[stratum] = {"n": len(test),
                                   "balanced_accuracy": round(balanced_accuracy_score(truth, pred),
                                                              3)}  # fmt: skip
            entry[name] = scores
        report[track] = entry
    write_json(WORK / "leakage.json", report)
    return report


def check_output(units_file: Path, out_file: Path) -> list[str]:
    """What is wrong with one labeller's output file for one batch; empty when it is usable."""
    units = json.loads(units_file.read_text(encoding="utf-8"))
    try:
        out = json.loads(out_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read {out_file}: {exc}"]
    root = units_file.parent.parent
    if (root / "questions.json").exists():
        questions = json.loads((root / "questions.json").read_text(encoding="utf-8"))
        wanted = {qid: [o["key"] for o in options(q)] for qid, q in questions.items()}
    else:
        wanted = None
    problems = []
    # A labeller may leave a text out; the row then lacks a vote and is not built.
    skipped = set(out.pop(SKIPPED, []))
    for unit in units:
        answer = out.get(unit["id"])
        if answer is None and unit["id"] in skipped:
            continue
        if answer is None:
            problems.append(f"{unit['id']}: missing")
            continue
        if wanted is None:
            if distribution(answer, ["true", "false"]) is None:
                problems.append(f"{unit['id']}: not a true/false distribution summing to 1")
            continue
        for qid, keys in wanted.items():
            if distribution(answer.get(qid), keys) is None:
                problems.append(f"{unit['id']} {qid}: needs keys {keys} summing to 1")
    extra = set(out) - {u["id"] for u in units}
    if extra:
        problems.append(f"ids not in the batch: {sorted(extra)[:3]}")
    return problems


def status() -> dict:
    """Which batches every pass has done, so a stopped agent's work resumes where it ended."""
    out: dict = {}
    passes = [(f"support {j}", WORK / "support" / "units", WORK / "support" / f"out_{j}")
              for j in (WRITER, LABELLER)]  # fmt: skip
    passes += [(f"label {t}", WORK / "labels" / t / "units", WORK / "labels" / t / "out")
               for t in ("guvenlik", "moderasyon")]  # fmt: skip
    for name, units, outs in passes:
        if not units.is_dir():
            out[name] = "no units yet"
            continue
        done, bad, todo = [], [], []
        for batch in sorted(units.glob("batch_*.json")):
            number = batch.stem.split("_")[1]
            result = outs / f"out_{number}.json"
            if not result.exists():
                todo.append(number)
            elif check_output(batch, result):
                bad.append(number)
            else:
                done.append(number)
        out[name] = {"done": len(done), "invalid": bad, "todo": todo}
    plans = sorted(p.stem[5:] for p in (WORK / "writers").glob("plan_*.json"))
    written_ = {p.stem[2:] for p in (WORK / "writers").glob("w_*.json")}
    out["writers"] = {"done": sorted(w for w in plans if w in written_),
                      "todo": [w for w in plans if w not in written_]}  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data.label.r4")
    parser.add_argument("command", choices=["support-units", "writer-plan", "label-units", "rows",
                                            "leakage", "check", "status"])  # fmt: skip
    parser.add_argument("--units", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "check":
        problems = check_output(args.units, args.out)
        print("\n".join(problems[:30]) if problems else "ok")
        return 1 if problems else 0
    run = {"support-units": support_units, "writer-plan": writer_plan, "label-units": label_units,
           "rows": build_rows, "leakage": leakage, "status": status}[args.command]  # fmt: skip
    print(json.dumps(run(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
