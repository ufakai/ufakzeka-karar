"""HakemBench probe items: shared orders, seeded disjoint draws, filters, resume, keep rule."""

import json
import random
import threading
from collections import Counter, defaultdict

import pytest

from bench.hakembench import probe_items as P
from bench.harness.items import load_items
from data.label.client import BudgetExceeded

COURTS = {f"mahkeme {c}": f"{c} davalarına bakan yer" for c in "abcdef"}
LEVELS = ["hiç", "az", "orta", "tam"]


def question(kind: str, k: int = 3) -> dict:
    if kind == "noul":
        return {"type": "noul", "instructions": "Bu bir dolandırıcılık mı?",
                "criteria": {"true": "Evet, öyle.", "false": "Hayır, değil."}}  # fmt: skip
    if kind == "score":
        return {"type": "score", "instructions": "Cevap ne kadar yeterli?", "criteria": LEVELS}
    keys = list(COURTS)[:k]
    return {"type": "choice", "instructions": "Hangi mahkeme bakar?",
            "criteria": {key: COURTS[key] for key in keys}}  # fmt: skip


def make_items(n: int = 80) -> tuple[list[dict], dict[str, str]]:
    """Checked items as desk collect writes them, and their halves."""
    rng = random.Random(3)
    items, halves = [], {}
    shapes = {"hukuk": {"court": ("choice", 5), "nq": ("noul", 2)},
              "sss": {"c3": ("choice", 3), "s": ("score", 4)},
              "spam": {"c4": ("choice", 4)}}  # fmt: skip
    for track, qs in shapes.items():
        for i in range(n):
            questions = {qid: question(kind, k) for qid, (kind, k) in qs.items()}
            gold, source = {}, {}
            for qid, q in questions.items():
                if q["type"] == "noul":
                    gold[qid] = rng.choice(["true", "false"])
                elif q["type"] == "score":
                    gold[qid] = str(rng.randrange(4))
                else:
                    gold[qid] = rng.choice(list(q["criteria"]))
                source[qid] = "owner" if i % 3 else "ai_reviewer_and_panel"
            if i % 10 == 8:
                source[next(iter(qs))] = "pending"
            item_id = f"{track}-{i:04d}"
            items.append({"id": item_id, "track": track,
                          "state": f"Metin {track} {i}: Ahmet Bey Ankara'da 500 lira ödedi.",
                          "questions": questions, "gold": gold, "flags": {},
                          "gold_source": source})  # fmt: skip
            halves[item_id] = "public" if i % 2 == 0 else "private"
    return items, halves


# permutations -----------------------------------------------------------------


def test_orders_are_every_order_or_24_with_identity_and_reverse():
    rng = random.Random(1)
    three, four = P.cell_orders(3, rng), P.cell_orders(4, rng)
    assert len(three) == 6 and len(set(three)) == 6 and len(four) == 24
    for orders, k in ((three, 3), (four, 4)):
        assert tuple(range(k)) in orders and tuple(range(k))[::-1] in orders
    six = P.cell_orders(6, random.Random(1))
    assert len(six) == 24 and len(set(six)) == 24
    assert six[0] == tuple(range(6)) and six[1] == tuple(range(6))[::-1]
    assert six == P.cell_orders(6, random.Random(1))
    assert [P.questions_for(k) for k in (3, 4, 5, 7, 8, 9)] == [200, 50, 50, 50, 100, 100]
    assert P.questions_for(4) == 50 and P.questions_for(3) == 200


def test_permutation_items_share_orders_keep_gold_and_validate(tmp_path):
    items, halves = make_items(80)
    rows, cells = P.permutations(items, halves)
    by_cell = {(c["track"], c["qid"]): c for c in cells}
    # Choice cells only; the k = 4 and k = 5 to 7 cells draw 50, the k = 3 cell all its settled.
    assert set(by_cell) == {("hukuk", "court"), ("sss", "c3"), ("spam", "c4")}
    assert by_cell[("spam", "c4")]["questions"] == 50
    assert by_cell[("hukuk", "court")]["questions"] == 50
    assert by_cell[("sss", "c3")]["questions"] == 72  # every tenth is pending
    assert [len(by_cell[c]["orders"]) for c in sorted(by_cell)] == [24, 24, 6]
    counts = P.write_probe(rows, "permutations", tmp_path / "pub", tmp_path / "priv")
    assert sum(counts.values()) == 50 * 24 + 72 * 6 + 50 * 24
    base = {i["id"]: i for i in items}
    seen_orders = defaultdict(lambda: defaultdict(set))
    for half in ("public", "private"):
        folder = tmp_path / ("pub" if half == "public" else "priv")
        loaded = load_items(folder / f"permutations.{half}.jsonl")
        metas = P.read_jsonl(folder / f"permutations.{half}.meta.jsonl")
        assert [m["id"] for m in metas] == [i.id for i in loaded]
        for item, meta in zip(loaded, metas, strict=True):
            original = base[meta["base_id"]]
            (qid,) = item.questions
            n = int(item.id.rsplit("-", 1)[1])
            assert item.id == f"{meta['base_id']}~perm.{qid}-{n}" and meta["half"] == half
            assert halves[meta["base_id"]] == half
            assert item.gold == {qid: original["gold"][qid]}
            keys = list(original["questions"][qid]["criteria"])
            assert list(item.questions[qid].criteria) == [keys[i] for i in meta["order"]]
            seen_orders[(item.track, qid)][n].add(tuple(meta["order"]))
    for cell, orders in seen_orders.items():
        assert all(len(s) == 1 for s in orders.values())  # one order per n in a cell
        assert [next(iter(orders[n])) for n in sorted(orders)] == [
            tuple(o) for o in by_cell[cell]["orders"]
        ]


def test_two_choice_questions_on_one_item_get_distinct_permutation_ids(tmp_path):
    """As on the support track: two choice questions asked of the same items."""
    items, halves = [], {}
    for i in range(12):
        item_id = f"destek-{i:04d}"
        questions = {"insan_destegi-choice": question("choice", 3),
                     "yanit_yeterliligi-choice": question("choice", 4)}  # fmt: skip
        items.append({"id": item_id, "track": "destek", "state": f"Metin {i}",
                      "questions": questions,
                      "gold": {q: next(iter(v["criteria"])) for q, v in questions.items()},
                      "gold_source": {q: "owner" for q in questions}})  # fmt: skip
        halves[item_id] = "public"
    rows, cells = P.permutations(items, halves)
    assert len(cells) == 2 and len(rows) == 12 * 6 + 12 * 24
    P.write_probe(rows, "permutations", tmp_path / "pub", tmp_path / "priv")
    loaded = load_items(tmp_path / "pub" / "permutations.public.jsonl")
    assert len(loaded) == len(rows) == len({i.id for i in loaded})
    assert "destek-0000~perm.insan_destegi-choice-5" in {i.id for i in loaded}
    assert "destek-0000~perm.yanit_yeterliligi-choice-23" in {i.id for i in loaded}


def test_the_permutations_command_writes_items_and_orders(tmp_path, monkeypatch):
    items, halves = make_items(20)
    checked = tmp_path / "hukuk.jsonl"
    checked.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    cands = tmp_path / "cands"
    cands.mkdir()
    (cands / "x.meta.jsonl").write_text(
        "".join(json.dumps({"id": k, "split": v}) + "\n" for k, v in halves.items())
    )
    monkeypatch.setattr(P, "CANDIDATES", cands)
    monkeypatch.setattr(P, "PUBLIC_OUT", tmp_path / "pub")
    monkeypatch.setattr(P, "PRIVATE_OUT", tmp_path / "priv")
    monkeypatch.setattr(P, "REPORTS", tmp_path / "reports")
    assert P.main(["permutations", "--items", str(checked)]) == 0
    orders = json.loads((tmp_path / "reports" / "permutation_orders.json").read_text())
    assert {c["qid"] for c in orders["cells"]} == {"court", "c3", "c4"}
    assert load_items(tmp_path / "pub" / "permutations.public.jsonl")


# draws ------------------------------------------------------------------------


def test_text_draws_are_seeded_stratified_and_disjoint():
    items, halves = make_items(80)
    para = P.draw(items, halves, "paraphrase")
    assert para == P.draw(items, halves, "paraphrase")
    assert P.draw(items, halves, "paraphrase", seed=5) != para
    english = P.draw(items, halves, "english")
    assert Counter(s["track"] for s in para) == {"hukuk": 30, "sss": 30, "spam": 30}
    assert Counter(s["track"] for s in english) == {"hukuk": 24, "sss": 24, "spam": 24}
    assert not {s["base_id"] for s in para} & {s["base_id"] for s in english}
    by_id = {i["id"]: i for i in items}
    for s in para + english:
        assert len(s["qids"]) == 1 and s["id"] == f"{s['base_id']}~{s['probe']}"
        assert P.settled(by_id[s["base_id"]])  # no pending gold source
        assert s["half"] == halves[s["base_id"]]
    # Every (question, gold) stratum of a track is represented.
    sss = [s for s in para if s["track"] == "sss"]
    assert {s["qids"][0] for s in sss} == {"c3", "s"}
    golds = {by_id[s["base_id"]]["gold"]["c3"] for s in sss if s["qids"] == ["c3"]}
    assert len(golds) == 3


def test_a_small_track_fills_its_english_draw_first():
    items, halves = make_items(48)  # as guvenlik: 48 items, 44 of them settled here
    english = P.draw(items, halves, "english")
    para = P.draw(items, halves, "paraphrase")
    assert Counter(s["track"] for s in english) == {"hukuk": 24, "sss": 24, "spam": 24}
    assert Counter(s["track"] for s in para) == {"hukuk": 20, "sss": 20, "spam": 20}
    assert not {s["base_id"] for s in para} & {s["base_id"] for s in english}


def test_the_slot_draw_is_every_public_item_with_its_settled_questions():
    items, halves = make_items(20)
    specs = P.draw(items, halves, "slots")
    public = {i["id"] for i in items if halves[i["id"]] == "public"}
    # spam's one question is pending on every tenth item, which leaves nothing to probe.
    assert {s["base_id"] for s in specs} == public - {"spam-0008", "spam-0018"}
    qids = {s["base_id"]: s["qids"] for s in specs}
    assert qids["hukuk-0000"] == ["court", "nq"]
    assert qids["hukuk-0008"] == ["nq"]  # its court gold source is pending


# prompts and parsing ----------------------------------------------------------


def test_prompts_show_the_question_and_its_gold():
    items, _ = make_items(4)
    item = items[0]
    text = P.paraphrase_prompt(item, ["court"])
    assert item["state"] in text and "Hangi mahkeme bakar?" in text
    assert f"Doğru cevap: {item['gold']['court']}" in text and '{"text": "..."}' in text
    english = P.english_prompt(item, ["nq"])
    gold = item["gold"]["nq"]
    assert f'Correct answer: "{gold}"' in english and "key_map" in english
    slots = P.slots_prompt(item, ["court", "nq"])
    assert "gold_dependent" in slots and "Hangi mahkeme bakar?" in slots


def spec_of(item: dict, probe: str, qids: list[str]) -> dict:
    return P.spec(item, qids, probe, "public")


def test_a_lexical_echo_is_dropped_and_recorded():
    items, _ = make_items(2)
    item = items[0]
    s = spec_of(item, "paraphrase", ["court"])
    echo = {"id": s["id"], "content": json.dumps({"text": item["state"] + " Evet."})}
    fresh = {"id": s["id"], "content": json.dumps({"text": "Geçen gün birisi faturayı yatırdı."})}
    dropped = P.candidate("paraphrase", s, item, echo)
    kept = P.candidate("paraphrase", s, item, fresh)
    assert dropped["dropped"] == "lexical echo" and dropped["jaccard"] > P.JACCARD_MAX
    assert kept["dropped"] is None and kept["jaccard"] <= P.JACCARD_MAX
    assert P.jaccard("a b c d", "a b c d") == 1.0 and P.jaccard("a b", "c d") == 0.0
    fenced = {"id": s["id"], "content": "```json\n" + fresh["content"] + "\n```"}
    assert P.candidate("paraphrase", s, item, fenced)["text"] == kept["text"]
    assert P.candidate("paraphrase", s, item, {"id": s["id"], "content": "yok"})["dropped"] == (
        "unparsed"
    )
    failed = {"id": s["id"], "content": None, "error": "ApiError", "cost": 0.001}
    assert P.candidate("paraphrase", s, item, failed)["dropped"] == "error:ApiError"
    assert P.candidate("paraphrase", s, item, None)["dropped"] == "not generated"


def test_a_slot_reply_with_no_swapped_slot_or_a_changed_gold_slot_is_dropped():
    items, _ = make_items(2)
    item = items[0]
    s = spec_of(item, "slots", ["court"])

    def reply(slots, text="Metin hukuk 0: Mehmet Bey Ankara'da 500 lira ödedi."):
        return {"id": s["id"], "content": json.dumps({"text": text, "slots": slots})}

    swap = {"original": "Ahmet", "replacement": "Mehmet", "type": "person",
            "gold_dependent": False}  # fmt: skip
    held = {"original": "500 lira", "replacement": "500 lira", "type": "amount",
            "gold_dependent": True}  # fmt: skip
    assert P.candidate("slots", s, item, reply([swap, held]))["dropped"] is None
    assert P.candidate("slots", s, item, reply([]))["dropped"] == "no slot"
    assert P.candidate("slots", s, item, reply([held]))["dropped"] == "no slot"
    moved = {**held, "replacement": "700 lira"}
    assert P.candidate("slots", s, item, reply([swap, moved]))["dropped"] == "gold slot changed"


def test_a_slot_reply_is_checked_against_its_text():
    items, _ = make_items(2)
    item = items[0]  # "Metin hukuk 0: Ahmet Bey Ankara'da 500 lira ödedi."
    s = spec_of(item, "slots", ["court"])
    swap = {"original": "Ahmet", "replacement": "Mehmet", "type": "person",
            "gold_dependent": False}  # fmt: skip
    place = {"original": "Ankara", "replacement": "İzmir", "type": "place",
             "gold_dependent": False}  # fmt: skip
    held = {"original": "500 lira", "replacement": "500 lira", "type": "amount",
            "gold_dependent": True}  # fmt: skip

    def dropped(text, slots):
        record = {"id": s["id"], "content": json.dumps({"text": text, "slots": slots})}
        return P.candidate("slots", s, item, record)["dropped"]

    good = "Metin hukuk 0: Mehmet Bey İzmir'de 500 lira ödedi."
    assert dropped(good, [swap, place, held]) is None  # the suffixed place leaves with its slot
    # The replacement is claimed but not in the text.
    assert dropped("Metin hukuk 0: Ahmet Bey İzmir'de 500 lira ödedi.", [swap, place, held]) == (
        "slot text mismatch"
    )
    # The original is still in the text beside its replacement.
    assert dropped("Metin hukuk 0: Mehmet Bey Ankara'da İzmir 500 lira ödedi.",
                   [swap, place, held]) == "slot text mismatch"  # fmt: skip
    # A gold-dependent value went missing.
    assert dropped("Metin hukuk 0: Mehmet Bey İzmir'de para ödedi.", [swap, place, held]) == (
        "slot text mismatch"
    )
    # More than the slots was rewritten.
    long_state = " ".join(f"sözcük{n}" for n in range(30))
    wide = {**item, "state": f"{long_state} Ahmet geldi."}
    swapped = f"{long_state} Mehmet geldi."
    ok = P.candidate("slots", s, wide, {"id": s["id"], "content": json.dumps(
        {"text": swapped, "slots": [swap]})})  # fmt: skip
    assert ok["dropped"] is None and ok["slot_jaccard"] == 1.0
    edited = swapped.replace("sözcük1 sözcük2 sözcük3 sözcük4", "başka dört yeni sözcük")
    bad = {"id": s["id"], "content": json.dumps({"text": edited, "slots": [swap]})}
    c = P.candidate("slots", s, wide, bad)
    assert c["dropped"] == "slot text mismatch" and c["slot_jaccard"] < P.SLOT_JACCARD_MIN


# generation: journal, resume, cap ---------------------------------------------


class FakeClient:
    """Answers every probe request; raises the budget stop from call `stop_at` on."""

    spent = 0.0

    def __init__(self, stop_at: int | None = None):
        self.prompts = []
        self.stop_at = stop_at
        self.lock = threading.Lock()

    def chat(self, model, provider, messages, **kwargs):
        with self.lock:
            if self.stop_at is not None and len(self.prompts) >= self.stop_at:
                raise BudgetExceeded("spent 12.0000 of a 12.0000 USD cap; no further calls")
            self.prompts.append(messages[1]["content"])
            n = len(self.prompts)
        assert messages[0]["content"] == P.SYSTEM
        text = f"Bambaşka sözcüklerle yazılmış yeni metin numara {n} burada duruyor."
        body = {"text": text, "slots": []}
        return {"choices": [{"message": {"content": json.dumps(body, ensure_ascii=False)}}],
                "usage": {"cost": 0.0005}}  # fmt: skip


GENERATOR = {"model": "vendor/g", "provider": "p", "reasoning": "low"}


def test_the_journal_resumes_and_pays_for_nothing_twice(tmp_path):
    items, halves = make_items(40)
    specs = P.draw(items, halves, "paraphrase")
    bases = {i["id"]: i for i in items}
    journal = tmp_path / "paraphrase_journal.jsonl"
    first = FakeClient()
    done = P.generate(first, GENERATOR, "paraphrase", specs, bases, journal, 4, limit=10)
    assert len(first.prompts) == 10 and len(done) == 10
    second = FakeClient()
    done = P.generate(second, GENERATOR, "paraphrase", specs, bases, journal, 4)
    assert len(second.prompts) == len(specs) - 10 and set(done) == {s["id"] for s in specs}
    third = FakeClient()
    P.generate(third, GENERATOR, "paraphrase", specs, bases, journal, 4)
    assert third.prompts == []
    assert len(P.read_jsonl(journal)) == len(specs)


def test_a_journal_of_another_draw_is_refused(tmp_path):
    items, halves = make_items(40)
    specs = P.draw(items, halves, "paraphrase")
    bases = {i["id"]: i for i in items}
    journal = tmp_path / "paraphrase_journal.jsonl"
    P.generate(FakeClient(), GENERATOR, "paraphrase", specs, bases, journal, 2, limit=3)
    stored = json.loads(P.draw_file(journal).read_text())["fingerprint"]
    assert stored == P.draw_fingerprint(specs) == P.draw_fingerprint(specs[::-1])
    other = P.draw(items[:-3], halves, "paraphrase")  # a checked set that grew or shrank
    assert P.draw_fingerprint(other) != stored
    client = FakeClient()
    with pytest.raises(P.DrawMismatch, match="another draw|was written for draw"):
        P.generate(client, GENERATOR, "paraphrase", other, bases, journal, 2)
    assert client.prompts == []
    # A journal with records and no fingerprint cannot be trusted either.
    P.draw_file(journal).unlink()
    with pytest.raises(P.DrawMismatch, match="no draw fingerprint"):
        P.generate(client, GENERATOR, "paraphrase", specs, bases, journal, 2)
    assert client.prompts == []


class FlakyClient(FakeClient):
    """Every third reply is prose instead of the JSON object."""

    def chat(self, model, provider, messages, **kwargs):
        response = super().chat(model, provider, messages, **kwargs)
        if len(self.prompts) % 3 == 0:
            response["choices"][0]["message"]["content"] = "Üzgünüm, yazamam."
        return response


def test_retry_unparsed_asks_again_and_keeps_the_dropped_lines(tmp_path):
    items, halves = make_items(40)
    specs = P.draw(items, halves, "paraphrase")[:12]
    bases = {i["id"]: i for i in items}
    journal = tmp_path / "paraphrase_journal.jsonl"
    P.generate(FlakyClient(), GENERATOR, "paraphrase", specs, bases, journal, 1)
    lost = {"id": "x", "content": None, "error": "ApiError", "cost": 0.001}
    with journal.open("a") as f:
        f.write(json.dumps(lost) + "\n")
    first = P.read_jsonl(journal)
    bad = [r for r in first if r["content"] is None or P.reply_object(r)[1] is None]
    assert len(bad) == 5  # four prose replies and the failed call
    plain = FakeClient()
    P.generate(plain, GENERATOR, "paraphrase", specs, bases, journal, 1)
    assert plain.prompts == []  # without the flag nothing is asked again
    again = FakeClient()
    done = P.generate(again, GENERATOR, "paraphrase", specs, bases, journal, 1,
                      retry_unparsed=True)  # fmt: skip
    assert len(again.prompts) == 4  # "x" is not in this draw, so it is not asked
    assert all(P.reply_object(done[s["id"]])[1] for s in specs)
    assert P.read_jsonl(P.retried_file(journal)) == bad
    assert len(P.read_jsonl(journal)) == 12


def test_generate_waits_for_the_owner_checks(tmp_path, monkeypatch, capsys):
    items, halves = make_items(20)
    cli_paths(tmp_path, monkeypatch, items, halves, owner_pending=3)
    client = FakeClient()
    monkeypatch.setattr(P, "make_client", lambda ledger_dir, cap: client)
    assert P.main(["generate", "--probe", "english"]) == 1
    assert "owner_pending 3" in capsys.readouterr().err and client.prompts == []
    P.CATCH_REPORT.unlink()
    assert P.main(["generate", "--probe", "english"]) == 1
    assert "run desk collect" in capsys.readouterr().err
    assert P.main(["generate", "--probe", "english", "--allow-pending", "--limit", "2"]) == 0
    assert len(client.prompts) == 2


def cli_paths(tmp_path, monkeypatch, items, halves, owner_pending=0):
    report = tmp_path / "catch_report.json"
    report.write_text(json.dumps({"owner_pending": owner_pending}))
    monkeypatch.setattr(P, "CATCH_REPORT", report)
    checked = tmp_path / "checked"
    checked.mkdir()
    (checked / "all.jsonl").write_text(
        "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items)
    )
    cands = tmp_path / "cands"
    cands.mkdir()
    (cands / "x.meta.jsonl").write_text(
        "".join(json.dumps({"id": k, "half": v}) + "\n" for k, v in halves.items())
    )
    for name, value in (("CHECKED", checked), ("CANDIDATES", cands), ("WORK", tmp_path / "work"),
                        ("REPORTS", tmp_path / "reports"), ("PUBLIC_OUT", tmp_path / "pub"),
                        ("PRIVATE_OUT", tmp_path / "priv")):  # fmt: skip
        monkeypatch.setattr(P, name, value)
    monkeypatch.setattr(P, "load_panel", lambda: ([], GENERATOR, "A"))


def test_a_budget_stop_journals_what_finished_and_a_rerun_resumes(tmp_path, monkeypatch, capsys):
    items, halves = make_items(40)
    cli_paths(tmp_path, monkeypatch, items, halves)
    stopping = FakeClient(stop_at=5)
    ledgers = []

    def make_client(ledger_dir, cap):
        ledgers.append((ledger_dir, cap))
        return stopping

    monkeypatch.setattr(P, "make_client", make_client)
    assert P.main(["generate", "--probe", "paraphrase", "--workers", "1"]) == 1
    assert "stopped" in capsys.readouterr().err
    assert ledgers == [(P.LEDGER_DIR, P.CAP_USD)]  # the step 8 ledger and cap
    journal = tmp_path / "work" / "paraphrase_journal.jsonl"
    assert len(P.read_jsonl(journal)) == 5
    assert not (tmp_path / "work" / "paraphrase_candidates.jsonl").exists()
    resumed = FakeClient()
    monkeypatch.setattr(P, "make_client", lambda ledger_dir, cap: resumed)
    assert P.main(["generate", "--probe", "paraphrase", "--cap-usd", "3"]) == 0
    specs = P.draw(items, halves, "paraphrase")
    assert len(resumed.prompts) == len(specs) - 5
    cands = P.read_jsonl(tmp_path / "work" / "paraphrase_candidates.jsonl")
    assert len(cands) == len(specs) and all(c["dropped"] is None for c in cands)
    assert all(c["cost"] == 0.0005 for c in cands)


# review and keep --------------------------------------------------------------


def paraphrase_candidates(n: int = 30) -> list[dict]:
    items, _ = make_items(n)
    out = []
    for item in [i for i in items if i["track"] == "hukuk"]:
        s = spec_of(item, "paraphrase", ["court"])
        record = {"id": s["id"], "content": json.dumps({"text": f"Başka bir anlatım {item['id']}"}),
                  "cost": 0.001}  # fmt: skip
        out.append(P.candidate("paraphrase", s, item, record))
    return out


def test_review_units_are_blind_in_the_desk_shape_forty_per_file():
    cands = paraphrase_candidates(50)
    cands[0]["dropped"] = "lexical echo"
    units, pairs = P.review_docs(cands)
    assert [len(d["units"]) for d in units] == [len(d["pairs"]) for d in pairs] == [40, 9]
    unit = units[0]["units"][0]
    assert set(unit) == {"id", "track", "order", "text", "questions"}
    (q,) = unit["questions"]
    assert set(q) == {"qid", "question", "type", "options"}  # no proposal
    assert [o["key"] for o in q["options"]] == list(COURTS)[:5]
    pair = pairs[0]["pairs"][0]
    assert pair["id"] == unit["id"] and pair["probe"] == unit["text"] != pair["original"]
    assert cands[0]["id"] not in {u["id"] for d in units for u in d["units"]}


def test_keep_takes_blind_gold_and_same_meaning_then_caps_per_track(tmp_path):
    cands = paraphrase_candidates(30)
    answers = {c["id"]: {"court": {"answer": c["gold"]["court"], "flag": "none"}} for c in cands}
    same = {c["id"]: {"same": True, "note": ""} for c in cands}
    wrong = next(k for k in list(COURTS)[:5] if k != cands[0]["gold"]["court"])
    answers[cands[0]["id"]]["court"]["answer"] = wrong
    answers[cands[1]["id"]]["court"]["flag"] = "unclear"
    same[cands[2]["id"]]["same"] = False
    del answers[cands[3]["id"]]
    rows, summary = P.keep("paraphrase", cands, answers, same)
    assert summary["kept"] == 25 and summary["per_track"]["hukuk"]["drawn"] == 30
    assert summary["dropped"] == {"blind answer differs": 1, "flagged": 1, "meaning differs": 1,
                                  "no review answer": 1, "over the cap": 1}  # fmt: skip
    assert summary["spent_usd"] == 0.03
    assert summary["per_track"]["hukuk"]["dropped_share"] == {
        "blind answer differs": 0.0333, "flagged": 0.0333, "meaning differs": 0.0333,
        "no review answer": 0.0333, "over the cap": 0.0333}  # fmt: skip
    assert summary["dropped_share"] == summary["per_track"]["hukuk"]["dropped_share"]
    again, _ = P.keep("paraphrase", cands, answers, same)
    assert again == rows  # the cap's order is seeded
    kept_ids = {json.loads(r[2])["base_id"] for r in rows}
    assert not kept_ids & {c["base_id"] for c in cands[:4]}
    counts = P.write_probe(rows, "paraphrase", tmp_path / "pub", tmp_path / "priv")
    loaded = load_items(tmp_path / "pub" / "paraphrase.public.jsonl")
    assert len(loaded) == counts["public"] and all("~paraphrase-0" in i.id for i in loaded)
    by_base = {c["base_id"]: c for c in cands}
    for item in loaded:
        c = by_base[item.id.split("~")[0]]
        assert item.state == c["text"] and item.gold == {"court": c["gold"]["court"]}
    # slots are not capped
    assert P.KEEP.get("slots") is None


def test_english_gold_goes_through_the_key_map(tmp_path):
    items, _ = make_items(2)
    item = items[0]
    s = {**spec_of(item, "english", ["court"]), "half": "private"}
    keys = list(item["questions"]["court"]["criteria"])
    key_map = {k: f"court {k[-1].upper()}" for k in keys}
    reply = {"text": "Mr Ahmet paid 500 lira in Ankara.",
             "question": {"instructions": "Which court hears it?",
                          "criteria": {v: f"hears {v}" for v in key_map.values()}},
             "key_map": key_map}  # fmt: skip
    c = P.candidate("english", s, item, {"id": s["id"], "content": json.dumps(reply)})
    assert c["dropped"] is None and c["key_map"] == key_map
    english_gold = key_map[item["gold"]["court"]]
    answers = {c["id"]: {"court": {"answer": english_gold, "flag": "none"}}}
    rows, summary = P.keep("english", [c], answers, {c["id"]: {"same": True, "note": ""}})
    assert summary["kept"] == 1
    P.write_probe(rows, "english", tmp_path / "pub", tmp_path / "priv")
    (loaded,) = load_items(tmp_path / "priv" / "english.private.jsonl")
    assert loaded.gold == {"court": english_gold}
    assert list(loaded.questions["court"].criteria) == [key_map[k] for k in keys]
    (meta,) = P.read_jsonl(tmp_path / "priv" / "english.private.meta.jsonl")
    assert meta["key_map"] == key_map and meta["base_id"] == item["id"]
    # The Turkish key given by a reviewer is not the gold of the English item.
    turkish = {c["id"]: {"court": {"answer": item["gold"]["court"], "flag": "none"}}}
    assert P.keep("english", [c], turkish, {c["id"]: {"same": True}})[1]["kept"] == 0
    # A key map that misses an option, or maps two onto one, is dropped.
    partial = {**reply, "key_map": dict(list(key_map.items())[1:])}
    bad = P.candidate("english", s, item, {"id": s["id"], "content": json.dumps(partial)})
    assert bad["dropped"] == "bad key map"
    # noul keys stay as they are and need no key map.
    s2 = spec_of(item, "english", ["nq"])
    noul = {"text": "Mr Ahmet paid.", "key_map": {"true": "yes"},
            "question": {"instructions": "Is it fraud?",
                         "criteria": {"true": "Yes.", "false": "No."}}}  # fmt: skip
    c2 = P.candidate("english", s2, item, {"id": s2["id"], "content": json.dumps(noul)})
    assert c2["dropped"] is None and c2["key_map"] == {}
    assert P.expected_key(c2, "nq") == item["gold"]["nq"]


def test_typed_gold_converts_desk_keys():
    assert P.typed_gold(question("noul"), "true") is True
    assert P.typed_gold(question("noul"), "false") is False
    assert P.typed_gold(question("score"), "2") == 2
    assert P.typed_gold(question("choice"), "mahkeme a") == "mahkeme a"
    with pytest.raises(ValueError):
        P.typed_gold(question("noul"), "evet")
    with pytest.raises(ValueError):
        P.half_of({"id": "x"}, {})
