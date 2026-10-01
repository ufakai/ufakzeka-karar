"""The support rubric, written out, and the support gold re-decided under it.

    python -m bench.hakembench.rubric apply
    python -m bench.hakembench.rubric passes-input --out results/private/step8/rubric_v2/passes
    python -m bench.hakembench.rubric adjudication-input --a <folder> --b <folder> --out <folder>
    python -m bench.hakembench.rubric gold --a <folder> --b <folder> --adjudicated <folder>
    python -m bench.hakembench.rubric check

The owner's audit of the top-up found the four support questions (sensitivity,
needs a person, answer adequacy as a choice and as a score) near a coin flip
between the owner, the panel and the gold. The texts were clear; the option
wording never said how to treat a reply that only redirects ("call this number")
or whether sensitivity reads the customer's question or only the reply. The rule
is now written into each option's description, so a model under test sees the
same rule the gold follows.

`apply` rewrites the four questions in every support candidate. `passes-input`
writes blind units for two independent passes, the second with every option list
shuffled. `adjudication-input` collects the questions the passes answer
differently, with the panel's votes as outside opinions. `gold` writes
results/private/step8/rubric_v2/gold.json, which desk collect reads after the
earlier gold files; the owner's own checks still come first there. `check`
compares the new gold with every support answer the owner gave blind, which no
pass sees; the rule was drawn from those answers, so the check measures whether
the rule is applied as written.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from bench.hakembench.desk import CANDIDATES, options, read_jsonl

FILES = ("sss.jsonl", "sss-topup.jsonl")
WORK = Path("results/private/step8/rubric_v2")
CHECKS = Path("results/private/step8/checks/checks")
QIDS = ("hassasiyet-noul", "insan_destegi-choice", "yanit_yeterliligi-choice",
        "yanit_yeterliligi-score")  # fmt: skip

RUBRIC = {
    "hassasiyet-noul": {
        "true": "Konuşmanın konusu (kullanıcının sorusu dahil) kişinin kendi parası, yasal hakkı "
        "ya da yükümlülüğü, sağlığı (insan ya da hayvan) veya güvenliği üzerinde sonuç "
        "doğuruyor. Cevabın genel ya da boş olması bunu değiştirmez.",
        "false": "Konu genel bilgi ya da rutin işleyiş (çalışma saati, süre, kayıt, randevu, "
        "ürün özelliği); kişinin parası, hakkı, sağlığı ya da güvenliği üzerinde bir sonucu yok.",
    },
    "insan_destegi-choice": {
        "Acil yönlendirme": "Cevaptan sonra çözülmemiş, süren bir sorun kalıyor ve beklemek "
        "kişiye zarar veriyor: sağlık (insan ya da hayvan), güvenlik, kaybedilen para, kaçacak "
        "bir süre ya da çalışmayan bir hizmet. Bir temsilci hemen devreye girmeli.",
        "Yönlendirme gereksiz": "Cevap sorunu çözmüş ya da sorulan bilgiyi eksiksiz vermiş; "
        "bir temsilciye iş kalmıyor.",
        "İsteğe bağlı yönlendirme": "Cevap genel bilgi veriyor ya da sorunu kısmen karşılıyor; "
        "kişinin kendi durumuna özel bir ayrıntı için bir temsilci faydalı olabilir ama acil "
        "değil.",
    },
    "yanit_yeterliligi-choice": {
        "Kapsamlı çözüm": "Cevap sorulan bilgiyi, adımları ya da kararı metnin içinde eksiksiz "
        "veriyor. Yalnızca başka bir kanala (telefon, form, ekip) yönlendiren cevap kapsamlı "
        "sayılmaz; soru zaten 'nereye ya da nasıl ulaşırım' ise o kanalı vermek kapsamlıdır.",
        "Kişisel deneyim": "Doğrudan çözüm sunmak yerine benzer bir durumu veya kendi sürecini "
        "aktaran yanıt.",
        "Kısmi yönlendirme": "Cevap sorunu kendisi çözmüyor ama bu soruna özgü faydalı bir ipucu, "
        "genel bilgi ya da somut bir sonraki adım veriyor (ilgili ekip, bu konu için form, "
        "aranacak numara, getirilecek belge).",
        "İlgisiz içerik": "Cevap sorunun konusuna değinmiyor: numara ya da bağlantı içerse bile "
        "genel kurumsal kalıp (görüşleriniz değerli, 7/24 çağrı merkezimiz, sosyal medyamızı "
        "takip edin) ya da konuyu başka bir alana çeken yanıt.",
    },
    "yanit_yeterliligi-score": [
        "Soruyu yanıtlamıyor: konuya değinmeyen genel kurumsal kalıp (numara ya da bağlantı "
        "içerse bile) ya da ilgisiz içerik.",
        "Soruyu kısmen yanıtlıyor ya da yalnızca bu soruna özgü bir sonraki adımı (ekip, form, "
        "numara, belge) veriyor; cevabın kendisi eksik kalıyor.",
        "Soruyu metnin içinde doğrudan, eksiksiz ve net biçimde yanıtlıyor; soru 'nereye ya da "
        "nasıl ulaşırım' ise doğru kanalı vermek yeterlidir.",
    ],
}


def rewritten(item: dict) -> dict:
    """The item with the four support questions' option descriptions set to the rubric."""
    questions = dict(item["questions"])
    for qid in QIDS:
        q = questions[qid]
        old, rule = q["criteria"], RUBRIC[qid]
        # The same options in the item's own order: only the descriptions change.
        if isinstance(old, dict):
            if set(old) != set(rule):
                raise ValueError(f"{item['id']} {qid}: the options differ from the rubric's")
            new = {key: rule[key] for key in old}
        else:
            if len(old) != len(rule):
                raise ValueError(f"{item['id']} {qid}: the levels differ from the rubric's")
            new = list(rule)
        questions[qid] = {**q, "criteria": new}
    return {**item, "questions": questions}


def support_items() -> list[dict]:
    return [i for name in FILES for i in read_jsonl(CANDIDATES / name)]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def units(items: list[dict], shuffle_seed: int | None = None) -> list[dict]:
    """Blind units: text and the four questions; option order shuffled per question if seeded."""
    rng = random.Random(shuffle_seed)
    out = []
    for item in items:
        questions = []
        for qid in QIDS:
            q = item["questions"][qid]
            opts = options(q)
            if shuffle_seed is not None and q["type"] != "score":
                rng.shuffle(opts)
            questions.append({"qid": qid, "question": q["instructions"], "type": q["type"],
                              "options": opts})  # fmt: skip
        state = item["state"]
        text = state if isinstance(state, str) else json.dumps(state)
        out.append({"id": item["id"], "text": text, "questions": questions})
    return out


def batches(rows: list, n: int) -> list[list]:
    return [rows[k::n] for k in range(n)]


def read_passes(folder: Path | None) -> dict:
    out: dict = {}
    if folder is not None:
        for path in sorted(folder.glob("out_*.json")):
            out |= json.loads(path.read_text(encoding="utf-8"))
    return out


def said(passes: dict, uid: str, qid: str) -> str | None:
    answer = passes.get(uid, {}).get(qid, {}).get("answer")
    return None if answer is None else str(answer)


def panel_votes(meta: dict, qid: str) -> list[str]:
    """Each panel judge's top option, for the adjudicator to weigh."""
    out = []
    for dist in (meta.get("votes", {}).get(qid, {}).get("judges") or {}).values():
        if dist:
            best = max(dist.values())
            top = [k for k, v in dist.items() if v == best]
            if len(top) == 1:
                out.append(top[0])
    return out


def decide(a: str | None, b: str | None, adjudicated: str | None) -> tuple[str | None, str]:
    if a is not None and a == b:
        return a, "ai_rubric_agreed"
    if adjudicated is not None:
        return adjudicated, "ai_rubric_adjudicated"
    return None, "needs adjudication"


def gold(items: list[dict], a: dict, b: dict, adjudicated: dict) -> tuple[dict, dict]:
    out, open_ = {}, defaultdict(list)
    for item in items:
        for qid in QIDS:
            key = f"{item['id']}~{qid}"
            answer, source = decide(said(a, item["id"], qid), said(b, item["id"], qid),
                                    said(adjudicated, item["id"], qid))  # fmt: skip
            if answer is None:
                open_[source].append(key)
            else:
                out[key] = {"answer": answer, "source": source}
    return out, dict(open_)


def owner_answers(folder: Path = CHECKS) -> dict[str, str]:
    """Every support answer the owner gave on the desk, keyed item~qid."""
    out = {}
    for path in folder.glob("*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("qid") not in QIDS or doc.get("flag", "none") != "none":
            continue
        answer = doc["proposed"] if doc["verdict"] == "agree" else doc.get("answer")
        if answer is not None:
            out[f"{doc['item']}~{doc['qid']}"] = str(answer)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 3), round((c + h) / d, 3))


def check(gold_: dict, owner: dict) -> dict:
    per: dict = defaultdict(lambda: [0, 0])
    for key, answer in owner.items():
        if key in gold_:
            qid = key.split("~", 1)[1]
            per[qid][0] += gold_[key]["answer"] == answer
            per[qid][1] += 1
    k, n = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
    per_question = {q: {"agree": v[0], "of": v[1]} for q, v in sorted(per.items())}
    return {"agree": k, "of": n, "rate": round(k / n, 3) if n else None,
            "wilson": wilson(k, n), "per_question": per_question}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.hakembench.rubric")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("apply")
    pi = sub.add_parser("passes-input")
    pi.add_argument("--out", type=Path, required=True)
    pi.add_argument("--batches", type=int, default=4)
    for name in ("adjudication-input", "gold"):
        c = sub.add_parser(name)
        c.add_argument("--a", type=Path, required=True)
        c.add_argument("--b", type=Path, required=True)
        c.add_argument("--adjudicated", type=Path)
        c.add_argument("--out", type=Path)
    sub.add_parser("check")
    args = parser.parse_args(argv)
    if args.command == "apply":
        for name in FILES:
            path = CANDIDATES / name
            write_jsonl(path, [rewritten(i) for i in read_jsonl(path)])
        print(json.dumps({"items": len(support_items()), "questions": list(QIDS)}))
        return 0
    items = support_items()
    for item in items:
        for qid in QIDS:
            criteria = item["questions"][qid]["criteria"]
            if (dict(criteria) if isinstance(criteria, dict) else criteria) != RUBRIC[qid]:
                raise SystemExit(f"{item['id']} {qid} is not on the rubric yet; run apply first")
    if args.command == "passes-input":
        for tag, seed in (("a", None), ("b", 128)):
            folder = args.out / tag
            folder.mkdir(parents=True, exist_ok=True)
            for n, batch in enumerate(batches(units(items, seed), args.batches)):
                (folder / f"batch_{n:02d}.json").write_text(json.dumps(batch, ensure_ascii=False))
        print(json.dumps({"units": len(items), "questions": len(items) * len(QIDS)}))
        return 0
    if args.command == "check":
        decided = json.loads((WORK / "gold_candidate.json").read_text(encoding="utf-8"))
        report = check(decided, owner_answers())
        (WORK / "check.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1))
        return 0
    a, b = read_passes(args.a), read_passes(args.b)
    if args.command == "adjudication-input":
        meta_files = [CANDIDATES / name.replace(".jsonl", ".meta.jsonl") for name in FILES]
        metas = {m["id"]: m for path in meta_files for m in read_jsonl(path)}
        by_id = {u["id"]: u for u in units(items)}
        todo = []
        for item in items:
            qs = []
            for q in by_id[item["id"]]["questions"]:
                x, y = said(a, item["id"], q["qid"]), said(b, item["id"], q["qid"])
                if x is not None and x == y:
                    continue
                reasons = [p.get(item["id"], {}).get(q["qid"], {}).get("reason_tr") for p in (a, b)]
                outside = panel_votes(metas.get(item["id"], {}), q["qid"])
                seen = sorted({v for v in (x, y) if v is not None})
                qs.append({**q, "candidates": seen, "reasons": [r for r in reasons if r],
                           "outside_votes": outside})  # fmt: skip
            if qs:
                todo.append({**by_id[item["id"]], "questions": qs})
        args.out.mkdir(parents=True, exist_ok=True)
        for n, batch in enumerate(batches(todo, 4)):
            (args.out / f"batch_{n:02d}.json").write_text(json.dumps(batch, ensure_ascii=False))
        print(json.dumps({"units": len(todo), "questions": sum(len(u["questions"]) for u in todo)}))
        return 0
    if args.command == "gold":
        decided, open_ = gold(items, a, b, read_passes(args.adjudicated))
        if open_:
            print(json.dumps({"open": {k: len(v) for k, v in open_.items()}}), file=sys.stderr)
            return 1
        WORK.mkdir(parents=True, exist_ok=True)
        # A candidate until the owner check accepts it; desk collect reads only gold.json.
        (WORK / "gold_candidate.json").write_text(json.dumps(decided, ensure_ascii=False, indent=1))
        sources = Counter(g["source"] for g in decided.values())
        print(json.dumps({"questions": len(decided), "sources": dict(sources)}))
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
