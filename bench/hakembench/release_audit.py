"""Pre-release checks over every checked HakemBench item.

    python -m bench.hakembench.release_audit

Runs on the items `desk collect` wrote, each placed in the half the freeze would
put it in. The freeze already refuses bad gold, a wrong half and training
overlap; this looks for everything else a reader of the released files could
trip on:

- duplicates: the same normalised text twice, a MinHash near duplicate, and any
  8-word run shared by a public and a private item (a leak between the halves);
- text hygiene: mojibake, the replacement character, control and zero-width
  characters, HTML entities, text not in NFC, template leftovers, unknown
  bracket placeholders, empty or very short texts, texts too long for a
  1,024-token context;
- personal data the masking pass should have caught (e-mail, links, phone
  numbers, IBAN, Turkish identity numbers with a valid checksum, card numbers
  passing Luhn, plates, @handles);
- language: texts with too few Turkish function words;
- labels: per question, the class shares in each half, a class missing from a
  half, one class above 90 percent, and how well one surface feature alone
  (length, a placeholder, a digit, a question mark, a quote) predicts a yes/no
  gold.

The summary with counts only goes to results/step8/release_audit.json; the
item ids behind every finding go to results/private/step8/release_audit.json.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from bench.hakembench.desk import CANDIDATES, CHECKED, read_jsonl, text_of
from bench.hakembench.ensemble import wilson
from bench.hakembench.freeze import PRIVATE_ONLY_SOURCES
from bench.hakembench.placeholders import PLACEHOLDERS, unharmonised
from bench.surface_features import BRACKET, features
from data.decontam.ngrams import MinHashLSH, minhash_signature, ngram_set, tokens, whole_hash

EXCLUDE = Path("docs/PUBLIC_BUILD_EXCLUDE.txt")
SUMMARY = Path("results/step8/release_audit.json")
DETAILS = Path("results/private/step8/release_audit.json")
# Findings a person looked at and kept on purpose, {finding: {item id: reason}}; private,
# since it names private items. The freeze refuses on any finding not listed here.
ACCEPTED = Path("results/private/step8/release_audit_accepted.json")

# The masking vocabulary, plus a phone menu label that is not a mask.
KNOWN_BRACKETS = set(PLACEHOLDERS) | {"adınız"}
MOJIBAKE = re.compile(r"Ã.|Ä.|Å.|â€|Ã¼|Ã§|Ä±|ÅŸ")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
INVISIBLE = re.compile(r"[​-‏‪-‮⁠﻿]")
ENTITY = re.compile(r"&(?:[a-z]{2,8}|#\d{2,5}|#x[0-9a-f]{2,4});", re.I)
TEMPLATE = re.compile(r"\{\{?\s*[A-Za-z_]+\s*\}?\}|<[A-Z_]{3,}>|\bTODO\b|\blorem ipsum\b", re.I)
# A literal backslash-n, backslash-r or backslash-u escape; LaTeX commands starting
# with n are fine.
ESCAPED = re.compile(r"\\n(?!e\b|eq\b|eg\b|abla\b|ot\b|u\b)|\\r|\\u[0-9a-fA-F]{4}")
PII = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}"),
    "url": re.compile(r"https?://|www\.[\w-]+\.\w", re.I),
    "phone": re.compile(r"(?<!\d)(?:\+90[\s-]?|0)?5\d{2}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"),
    "iban": re.compile(r"\bTR\s?\d{2}(?:\s?\d{4}){5}\s?\d{2}\b", re.I),
    "handle": re.compile(r"(?<![\w.])@[A-Za-z_][\w.]{2,}"),
    "plate": re.compile(r"\b(?:0[1-9]|[1-7]\d|8[01])\s?[A-Z]{1,3}\s?\d{2,4}\b"),
    "landline": re.compile(
        r"(?<!\d)(?:\+90[\s-]?|0)?[2-4]\d{2}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"
    ),
    "tax_number": re.compile(
        r"vergi\w*\s*(?:kimlik\s*)?(?:no|numara\w*)?\D{0,15}\d{10}(?!\d)", re.I
    ),
    "birth_date": re.compile(r"doğum\s*tarih\w*\D{0,5}\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", re.I),
}
DIGITS11 = re.compile(r"(?<!\d)[1-9]\d{10}(?!\d)")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){15,18}\d(?!\d)")
FUNCTION_WORDS = {
    "ve", "bir", "bu", "da", "de", "ile", "için", "ne", "mi", "mı", "ama", "çok", "daha", "gibi",
    "olarak", "olan", "var", "yok", "en", "her", "şey", "ben", "sen", "o", "biz", "siz", "onlar",
    "değil", "ise", "kadar", "sonra", "önce", "nasıl", "neden", "hangi", "bana", "size", "bunu",
    "şu", "ki",
}  # fmt: skip
# "[ad]" stands for a person; an old masker also put it on the word after a greeting.
MASKED_WORD = re.compile(
    r"\[ad\] (hafta|sabah|akşam|gece|gün|ay|yıl|sezon|kez|defa|hareket\w*|ikinci|yeni)\b"
)
# "[şirket]in": a case suffix must follow an apostrophe, or harmonise cannot check it.
GLUED = re.compile(r"\[(?:ad|şirket|kurum|ilçe|ürün)\][a-zçğıöşü]")
SHORT, LONG = 15, 3500  # characters; about 1,000 tokens at the tokenizer's rate
SKEW = 0.90
SHARED = 0.30
LEAKY = 0.70


def tc_kimlik(number: str) -> bool:
    d = [int(c) for c in number]
    if len(d) != 11 or d[0] == 0:
        return False
    tenth = ((d[0] + d[2] + d[4] + d[6] + d[8]) * 7 - (d[1] + d[3] + d[5] + d[7])) % 10
    return tenth == d[9] and sum(d[:10]) % 10 == d[10]


def luhn(number: str) -> bool:
    d = [int(c) for c in number if c.isdigit()][::-1]
    total = sum(x if i % 2 == 0 else (2 * x - 9 if 2 * x > 9 else 2 * x) for i, x in enumerate(d))
    return 13 <= len(d) <= 19 and total % 10 == 0


def hygiene(text: str) -> list[str]:
    """Names of the text problems found in one text."""
    found = []
    checks = {"mojibake": MOJIBAKE, "control": CONTROL, "invisible": INVISIBLE,
              "html_entity": ENTITY, "template": TEMPLATE}  # fmt: skip
    found += [name for name, pattern in checks.items() if pattern.search(text)]
    if "�" in text:
        found.append("replacement_char")
    if unicodedata.normalize("NFC", text) != text:
        found.append("not_nfc")
    if ESCAPED.search(text):
        found.append("escaped_sequence")
    if any(m.group(1).strip().lower() not in KNOWN_BRACKETS for m in BRACKET.finditer(text)):
        found.append("unknown_bracket")
    if unharmonised(text):
        found.append("unharmonised_suffix")
    if GLUED.search(text):
        found.append("suffix_without_apostrophe")
    if MASKED_WORD.search(text):
        found.append("name_mask_on_a_word")
    if len(text.strip()) < SHORT:
        found.append("too_short")
    if len(text) > LONG:
        found.append("too_long")
    return found


def personal_data(text: str) -> list[str]:
    found = [name for name, pattern in PII.items() if pattern.search(text)]
    if any(tc_kimlik(m.group()) for m in DIGITS11.finditer(text)):
        found.append("tc_kimlik")
    if any(luhn(m.group()) for m in CARD.finditer(text)):
        found.append("card")
    return found


def turkish_share(text: str) -> float:
    words = re.findall(r"\w+", text.lower())
    return sum(w in FUNCTION_WORDS for w in words) / len(words) if words else 0.0


def not_turkish(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    return (
        len(words) >= 12 and turkish_share(text) < 0.03 and not re.search(r"[çğışöüÇĞİŞÖÜ]", text)
    )


def best_single_feature(rows: list[tuple[str, bool]]) -> tuple[str, float]:
    """The surface feature that alone predicts a yes/no gold best, by balanced accuracy."""
    yes = [t for t, g in rows if g]
    no = [t for t, g in rows if not g]
    if not yes or not no:
        return "", 0.0

    def balanced(pred) -> float:
        tpr = sum(pred(t) for t in yes) / len(yes)
        tnr = sum(not pred(t) for t in no) / len(no)
        return max((tpr + tnr) / 2, 1 - (tpr + tnr) / 2)

    scores = {name: balanced(lambda t, n=name: features(t)[n]) for name in features("")}
    lengths = sorted({len(t) for t, _ in rows})
    cuts = lengths[:: max(1, len(lengths) // 50)]
    scores["length"] = max(balanced(lambda t, c=c: len(t) > c) for c in cuts)
    name = max(scores, key=scores.get)
    return name, round(scores[name], 3)


def placed(checked: list[dict], metas: dict[str, dict]) -> list[dict]:
    """Each item with its text and the half the freeze would give it."""
    out = []
    for item in checked:
        meta = metas.get(item["id"], {})
        where = meta.get("half") or meta.get("split")
        if any(s in meta.get("source", "") for s in PRIVATE_ONLY_SOURCES):
            where = "private"
        out.append({"id": item["id"], "track": item["track"], "half": where,
                    "text": text_of(item["state"]), "item": item,
                    "party": meta.get("party"), "role": meta.get("speaker_role")})  # fmt: skip
    return out


def duplicates(rows: list[dict]) -> dict[str, list]:
    wholes: dict[int, str] = {}
    exact, near = [], []
    lsh = MinHashLSH()
    for row in rows:
        whole = whole_hash(tokens(row["text"]))
        if whole in wholes:
            exact.append([wholes[whole], row["id"]])
        else:
            wholes[whole] = row["id"]
        signature = minhash_signature(row["text"])
        if signature is not None:
            near += [[key, row["id"], round(score, 3)] for key, score in lsh.query(signature)]
            lsh.add(row["id"], signature)
    # Formulaic wording (a court summary's closing clause) shares 8-grams by nature, so a
    # cross-half pair counts only when the shared runs cover a good part of the shorter text.
    grams: dict[tuple, set[str]] = defaultdict(set)
    size = {}
    half_of = {r["id"]: r["half"] for r in rows}
    for row in rows:
        own = ngram_set(row["text"])
        size[row["id"]] = len(own)
        for gram in own:
            grams[gram].add(row["id"])
    pairs: Counter = Counter()
    for ids in grams.values():
        if 1 < len(ids) <= 20:
            for a in ids:
                for b in ids:
                    if a < b and half_of[a] != half_of[b]:
                        pairs[(a, b)] += 1
    shared = sorted([a, b, round(n / min(size[a], size[b]), 3)] for (a, b), n in pairs.items()
                    if n / min(size[a], size[b]) >= SHARED)  # fmt: skip
    return {"exact": exact, "near": near, "shared_8gram_across_halves": sorted(shared)}


def labels(rows: list[dict]) -> tuple[dict, list[dict]]:
    shares: dict[str, dict] = {}
    findings = []
    per_question: dict[tuple[str, str], dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    # Every question, one option against the rest: a choice or a score option a surface
    # feature predicts is as much a leak as a yes/no answer one does.
    golds: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        item = row["item"]
        for qid, gold in item.get("gold", {}).items():
            per_question[(row["track"], qid)][row["half"]][str(gold)] += 1
            golds[(row["track"], qid, row["half"])].append((row["text"], str(gold).lower()))
    for (track, qid), halves in sorted(per_question.items()):
        classes = set().union(*halves.values())
        entry = {}
        for where, counts in sorted(halves.items()):
            n = sum(counts.values())
            entry[where] = {"n": n, "shares": {c: round(counts[c] / n, 3) for c in sorted(classes)}}
            missing = sorted(c for c in classes if counts[c] == 0)
            if missing:
                findings.append({"track": track, "qid": qid, "half": where, "missing": missing})
            top = max(counts.values()) / n
            if top > SKEW:
                findings.append({"track": track, "qid": qid, "half": where, "skew": round(top, 3)})
        # Within each half, so two halves drawn from different sources cannot fake a leak.
        for where in sorted(halves):
            pairs = golds[(track, qid, where)]
            options = ["true"] if {g for _, g in pairs} <= {"true", "false"} else sorted(
                {g for _, g in pairs})  # fmt: skip
            best = ("", 0.0, "")
            for option in options:
                share = sum(g == option for _, g in pairs) / len(pairs)
                if share < 0.1 or share > 0.9:
                    continue  # too few on one side to read a feature from
                feature, score = best_single_feature([(t, g == option) for t, g in pairs])
                if score > best[1]:
                    best = (feature, score, option)
            entry[where]["best_single_feature"] = {
                "feature": best[0], "balanced_accuracy": best[1], "option": best[2]}  # fmt: skip
            if best[1] > LEAKY:
                findings.append({"track": track, "qid": qid, "half": where,
                                 "option": best[2], "leaky_feature": best[0],
                                 "balanced_accuracy": best[1]})  # fmt: skip
        shares[f"{track}/{qid}"] = entry
    return shares, findings


def exclusion_rules(root: Path = Path(".")) -> list[str]:
    """The rules of docs/PUBLIC_BUILD_EXCLUDE.txt: a path, or a folder ending in a slash."""
    return [line.strip() for line in (root / EXCLUDE).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]  # fmt: skip


def excluded(name: str, rules: list[str]) -> bool:
    return any(name == r or (r.endswith("/") and name.startswith(r)) for r in rules)


def kept_names(root: Path = Path(".")) -> list[str]:
    """Every git-tracked path the public build keeps, relative to the repo root."""
    rules = exclusion_rules(root)
    tracked = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True,
                             check=True).stdout.splitlines()  # fmt: skip
    return [name for name in tracked if not excluded(name, rules)]


def public_files(root: Path = Path(".")) -> list[Path]:
    """Git-tracked text files the public build keeps (docs/PUBLIC_BUILD_EXCLUDE.txt left out)."""
    return [root / name for name in kept_names(root)
            if Path(name).suffix in {".json", ".jsonl", ".md", ".txt", ".py", ".yaml", ".yml",
                                     ".html", ".csv", ".tsv", ".toml"}]  # fmt: skip


def probes(text: str, n: int = 3, width: int = 48) -> list[str]:
    """A few fixed-width pieces of a text, as a text file or a JSON string would hold them."""
    body = json.dumps(text, ensure_ascii=False)[1:-1]
    if len(body) < width:
        return [body] if len(body) >= 24 else []
    step = max(1, (len(body) - width) // max(1, n - 1))
    return [body[i : i + width] for i in range(0, len(body) - width + 1, step)][:n]


def leaks(rows: list[dict], removed: dict[str, str], files: list[Path]) -> dict[str, list]:
    """Private-half and removed texts, and private item ids, in any file the public build keeps.

    A text counts as found when two of its probes are in the same file, so a court
    summary's standard closing clause alone does not; in a file of one record per line,
    in the same record, so one summary's opening formula and another's closing formula
    are not read as a copy. An id is a prefix of its text's hash, so a private id in a
    public file lets anyone confirm which public sentence is held back.
    """
    private = {r["id"]: r["text"] for r in rows if r["half"] == "private"}
    pieces = {uid: probes(text) for uid, text in (private | removed).items()}
    found: dict[str, list] = {}
    for path in files:
        body = path.read_text(encoding="utf-8", errors="ignore")
        units = body.splitlines() if path.suffix == ".jsonl" else [body]
        hits = {
            uid
            for uid, ps in pieces.items()
            if ps and any(sum(p in unit for p in ps) >= min(2, len(ps)) for unit in units)
        }
        hits |= {f"id:{uid}" for uid in private if uid in body}
        if hits:
            found[str(path)] = sorted(hits)
    return dict(sorted(found.items()))


def parliament(rows: list[dict]) -> dict:
    """Fact-check items per party and half, and the share each party gets as checkable.

    An AI labeller can lean by party the way human annotators did on Turkish political
    text, so the card reports this table.
    """
    table: dict[str, dict] = {}
    for row in rows:
        if row["track"] != "dogrulama" or row.get("party") is None:
            continue
        cell = table.setdefault(
            row["party"], {"public": 0, "private": 0, "checkable": 0, "asked": 0}
        )
        cell[row["half"]] = cell.get(row["half"], 0) + 1
        gold = row["item"].get("gold", {}).get("dogrulanabilir")
        if gold is not None:
            cell["asked"] += 1
            cell["checkable"] += str(gold).lower() == "true"
    for cell in table.values():
        cell["checkable_share"] = (
            round(cell["checkable"] / cell["asked"], 3) if cell["asked"] else None
        )
        low, high = wilson(cell["checkable"], cell["asked"]) if cell["asked"] else (None, None)
        cell["interval"] = [low, high]
    return dict(sorted(table.items(), key=lambda kv: -(kv[1]["public"] + kv[1]["private"])))


def audit(rows: list[dict], removed: dict[str, str] | None = None,
          files: list[Path] | None = None) -> tuple[dict, dict]:  # fmt: skip
    ids: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        for name in hygiene(row["text"]):
            ids[f"hygiene/{name}"].append(row["id"])
        for name in personal_data(row["text"]):
            ids[f"personal_data/{name}"].append(row["id"])
        if not_turkish(row["text"]):
            ids["language/not_turkish"].append(row["id"])
        if row["half"] not in ("public", "private"):
            ids["half/missing"].append(row["id"])
    dups = duplicates(rows)
    shares, label_findings = labels(rows)
    leaked = leaks(rows, removed or {}, files or [])
    summary = {
        "items": len(rows),
        "per_half": dict(Counter(r["half"] for r in rows)),
        "findings": {k: len(v) for k, v in sorted(ids.items())},
        "duplicates": {k: len(v) for k, v in dups.items()},
        "public_build_leaks": {k: len(v) for k, v in leaked.items()},
        "parliament_by_party": parliament(rows),
        "label_findings": label_findings,
        "label_shares": shares,
    }
    accepted = json.loads(ACCEPTED.read_text(encoding="utf-8")) if ACCEPTED.is_file() else {}
    open_ids = {k: [i for i in v if i not in accepted.get(k, {})] for k, v in ids.items()}
    summary["findings_open"] = {k: len(v) for k, v in sorted(open_ids.items()) if v}
    # A duplicate pair a person checked and kept is listed as "a|b" in the accepted file.
    pairs = accepted.get("duplicates/shared_8gram_across_halves", {})
    open_pairs = [x for x in dups["shared_8gram_across_halves"] if f"{x[0]}|{x[1]}" not in pairs]
    summary["duplicates_open"] = {"exact": len(dups["exact"]),
                                  "shared_8gram_across_halves": len(open_pairs)}  # fmt: skip
    summary["blocking"] = blocking(summary)
    # The private half's label shares stay private until that half is retired; the public
    # summary keeps its size and how far its mix is from the public half's.
    details = {"findings": dict(sorted(ids.items())), "findings_open": open_ids,
               "duplicates": dups, "leaks": leaked, "label_findings": label_findings,
               "label_shares": shares}  # fmt: skip
    summary["label_shares"] = {q: public_view(e) for q, e in shares.items()}
    summary["label_findings"] = [f for f in label_findings if f.get("half") != "private"]
    summary["label_findings_private"] = sum(f.get("half") == "private" for f in label_findings)
    return summary, details


def public_view(entry: dict) -> dict:
    out = {k: v for k, v in entry.items() if k != "private"}
    if "private" in entry:
        out["private"] = {"n": entry["private"]["n"]}
        if "public" in entry:
            a, b = entry["public"]["shares"], entry["private"]["shares"]
            out["private"]["distance"] = round(
                sum(abs(a.get(c, 0) - b.get(c, 0)) for c in set(a) | set(b)) / 2, 3
            )
    return out


def blocking(summary: dict) -> list[str]:
    """What stops the freeze: an unaccepted finding, a duplicate, or any leak."""
    out = [f"finding {k}" for k in summary["findings_open"]]
    out += [f"duplicates {k}" for k in ("exact", "shared_8gram_across_halves")
            if summary["duplicates_open"][k]]  # fmt: skip
    out += [f"leak {k}" for k in summary["public_build_leaks"]]
    return out


def removed_texts(checked: list[dict], metas: dict[str, dict],
                  folder: Path = CANDIDATES) -> dict[str, str]:  # fmt: skip
    """Texts of candidates that are not among the checked items (removed or never selected)."""
    kept = {i["id"] for i in checked}
    out = {}
    for path in sorted(folder.glob("*.jsonl")):
        if path.name.endswith(".meta.jsonl"):
            continue
        for item in read_jsonl(path):
            if item["id"] not in kept and metas.get(item["id"], {}).get("selected") is not False:
                out[item["id"]] = text_of(item["state"])
    return out


def main() -> int:
    checked = [i for p in sorted(CHECKED.glob("*.jsonl")) for i in read_jsonl(p)]
    metas = {m["id"]: m for p in sorted(CANDIDATES.glob("*.meta.jsonl")) for m in read_jsonl(p)}
    removed = removed_texts(checked, metas)
    summary, details = audit(placed(checked, metas), removed, public_files())
    for path, body in ((SUMMARY, summary), (DETAILS, details)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("items", "per_half", "findings", "duplicates",
                                              "public_build_leaks")},
                     indent=1))  # fmt: skip
    print(json.dumps(summary["label_findings"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
