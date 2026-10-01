"""HakemBench test items: new texts only, the right judges, halves by hash, overlaps dropped."""

import json
import math
import threading
from collections import Counter
from pathlib import Path

import pytest

from bench.dev import text_hash
from bench.hakembench import generate as G
from bench.harness.items import load_items
from data.label import synth
from data.label.judge import LETTERS, JudgeSpec
from data.label.pilot import load_panel, load_relabel_judge

LETTERS_ONLY = str.maketrans("0123456789", "ghijklmnop")


def tag(prompt: str) -> str:
    """A letters-only mark per request, so no mask mistakes it for a number."""
    return synth.short_id(prompt).translate(LETTERS_ONLY)


def fake_texts(prompt: str) -> str:
    """What the fake writer answers to one request, fixed by the prompt."""
    t = tag(prompt)
    if "sınav sorusu" in prompt:
        return json.dumps({"soru": f"{t} konusu nedir ve neden önemlidir?",
                           "cevaplar": [{"seviye": i, "metin": f"{t} cevabı düzey {i}: şöyle ki"}
                                        for i in range(4)]}, ensure_ascii=False)  # fmt: skip
    n = int(prompt.split(" ", 1)[0])
    if '"soru"' in prompt:
        return json.dumps([{"soru": f"{t} {j} siparişim ne zaman gelir acaba?",
                            "cevap": f"{t} {j} siparişiniz iki gün içinde teslim edilir."}
                           for j in range(n)], ensure_ascii=False)  # fmt: skip
    return json.dumps([{"metin": f"{t} {j} için bir örnek metin ve kısa bir açıklama."}
                       for j in range(n)], ensure_ascii=False)  # fmt: skip


class FakeClient:
    """Writes fixed texts; every judge puts 0.8 on the first option it is shown."""

    spent = 0.0

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def chat(self, model, provider, messages, **kwargs):
        with self.lock:
            self.calls.append((model, messages[1]["content"]))
        if messages[0]["content"] == synth.GENERATOR_SYSTEM:
            return {"choices": [{"message": {"content": fake_texts(messages[1]["content"])}}]}
        lines = messages[1]["content"].split("Seçenekler:\n")[1].split("\n\n")[0].split("\n")
        rest = 0.2 / (len(lines) - 1)
        top = [{"token": LETTERS[i], "logprob": math.log(0.8 if i == 0 else rest)}
               for i in range(len(lines))]  # fmt: skip
        return {"choices": [{"logprobs": {"content": [{"token": "A", "top_logprobs": top}]}}]}


PANEL = [JudgeSpec("A", "vendor/a", "p"), JudgeSpec("C", "vendor/c", "p")]
JUDGE_B = JudgeSpec("B", "vendor/b", "p")
GENERATOR = {"model": "vendor/g", "provider": "p"}
WRITERS = {"A": "vendor/a", "G": "vendor/g"}


# new texts only -------------------------------------------------------------


def test_new_topics_and_scenes_are_disjoint_from_training():
    G.check_disjoint()
    clash = {**G.EDU_TOPICS, "Fizik": (synth.TOPICS["Fizik"][0], *G.EDU_TOPICS["Fizik"][1:])}
    with pytest.raises(ValueError, match="already in the training"):
        G.check_disjoint(topics=clash)
    scenes = {**G.MESSAGE_SCENES, "islem": ("Kargo  Teslimatı", *G.MESSAGE_SCENES["islem"][1:])}
    with pytest.raises(ValueError, match="already in the training"):
        G.check_disjoint(scenes=scenes)
    court = next(iter(G.COURT_SCENES))
    old = synth.COURTS[court][1][0]
    courts = {**G.COURT_SCENES, court: (old, *G.COURT_SCENES[court][1:])}
    with pytest.raises(ValueError, match="already in the training"):
        G.check_disjoint(courts=courts)
    short = {**G.COURT_SCENES, court: G.COURT_SCENES[court][1:]}
    with pytest.raises(ValueError, match="120 scenes"):
        G.check_disjoint(courts=short)


def test_the_plan_writes_the_planned_texts_with_the_right_writers():
    plan = G.generation_plan()
    texts = Counter()
    for p in plan:
        texts[p["track"]] += p["n"]
    assert texts == {"egitim": 160, "spam": 320, "hukuk": 120, "moderasyon": 160, "sss": 160}
    assert len({p["id"] for p in plan}) == len({p["prompt"] for p in plan}) == len(plan)
    writers = {t: {p["writer"] for p in plan if p["track"] == t} for t in G.TRACKS}
    assert writers == {"egitim": {"G"}, "spam": {"A"}, "hukuk": {"G"}, "moderasyon": {"A"},
                       "sss": {"G"}}  # fmt: skip
    kinds = Counter(p["kind"] for p in plan if p["track"] == "spam")
    assert set(kinds) == {k.name for k in synth.MESSAGE_KINDS} and set(kinds.values()) == {8}


def test_each_request_asks_for_its_own_scene_in_synth_words():
    plan = G.generation_plan()
    exam = next(p for p in plan if p["shape"] == "exam")
    assert exam["scene"] in exam["prompt"] and "dört öğrencinin cevabını" in exam["prompt"]
    message = next(p for p in plan if p["track"] == "spam")
    assert message["prompt"].startswith("8 farklı Türkçe") and message["scene"] in message["prompt"]
    legal = next(p for p in plan if p["track"] == "hukuk")
    assert legal["kind"] in legal["prompt"] and legal["scene"] in legal["prompt"]
    support = next(p for p in plan if p["track"] == "sss")
    assert support["shape"] == "qa" and '"soru"' in support["prompt"]
    assert all(p.get("brand_check") for p in plan if p["track"] in ("spam", "moderasyon", "sss"))


def test_a_question_and_answer_list_becomes_masked_pairs():
    request = {"id": "r", "kind": "tam", "shape": "qa", "prompt": "x"}
    content = json.dumps([{"soru": "Kargom nerede kaldı acaba?",
                           "cevap": "Takip için www.ornek-kargo.test adresine bakın lütfen."},
                          {"soru": "", "cevap": "boş soru"}], ensure_ascii=False)  # fmt: skip
    texts = G.texts_of(request, content)
    assert len(texts) == 1
    assert texts[0]["text"].startswith("Soru: Kargom") and "[bağlantı]" in texts[0]["text"]


def test_the_moderation_question_is_offensevals_word_for_word():
    from data.typed.instrument import _question

    assert _question("offenseval_tr", ["NOT", "OFF"])[0] == G.OFFENSIVE
    assert G.OFFENSIVE["instructions"] == (
        "Bu mesaj küfür, hakaret ya da saldırgan bir dil içeriyor mu?"
    )


def test_one_held_out_template_per_cell_from_the_build():
    from data.label.build import held_out_cells

    chosen = G.held_out_templates()
    assert set(chosen) == held_out_cells()
    for cell, entry in chosen.items():
        assert entry["held_out"] and entry["status"] == "ready"
        assert f"{entry['template']['family']}-{entry['template']['type']}" == cell
    asks = G.track_asks(chosen)
    assert [a.task for a in asks["sss"]] == sorted(chosen)


# judges ---------------------------------------------------------------------


def test_messages_are_judged_by_b_and_c_and_nothing_by_its_writer():
    asks = G.track_asks(G.held_out_templates())
    names = lambda js: sorted(j.name for j in js)  # noqa: E731
    message = {"writer": "A"}
    for ask in asks["spam"]:
        assert names(G.judges_for(ask, message, PANEL, JUDGE_B, WRITERS)) == ["B", "C"]
    for ask in asks["sss"]:
        assert names(G.judges_for(ask, {"writer": "G"}, PANEL, JUDGE_B, WRITERS)) == ["A", "C"]
        assert names(G.judges_for(ask, {"writer": None}, PANEL, JUDGE_B, WRITERS)) == ["A", "C"]
    with pytest.raises(ValueError, match="judged by B and C"):
        G.judges_for(asks["spam"][0], {"writer": "G"}, PANEL, JUDGE_B, WRITERS)


@pytest.mark.skipif(
    not Path("config/panel.json").is_file(),
    reason="the labelling panel's config is kept out of the public repository",
)
def test_with_the_real_panel_no_writer_judges_its_own_text():
    judges, generator, _ = load_panel()
    judge_b = load_relabel_judge()
    a = next(j for j in judges if j.name == "A")
    writers = {"A": a.model, "G": generator["model"]}
    asks = G.track_asks(G.held_out_templates())
    for request in G.generation_plan():
        for ask in asks[request["track"]]:
            chosen = G.judges_for(ask, request, judges, judge_b, writers)
            assert writers[request["writer"]] not in {j.model for j in chosen}
            if ask.task in synth.WRITER_TASKS:
                assert sorted(j.name for j in chosen) == ["B", "C"]


# halves, filters, overlap ---------------------------------------------------


def text(track, kind, body, **extra):
    sha = text_hash(body)
    return {"track": track, "kind": kind, "text": body, "source": "generated", "sha": sha,
            "source_id": sha[:24], "writer": "G", **extra}  # fmt: skip


def test_halves_split_each_kind_evenly_by_hash_and_keep_an_exam_together():
    texts = []
    for subject in ("ders:Fizik", "ders:Tarih"):
        for e in range(5):
            for level in range(4):
                texts.append(text("egitim", f"seviye{level}", f"{subject} {e} cevap {level}",
                                  exam=f"{subject}-{e}", subject=subject))  # fmt: skip
    texts += [text("spam", k, f"{k} mesaj {i}") for k in ("islem", "kisisel") for i in range(7)]
    texts += [text("moderasyon", "olagan", "genel yorum"),
              {**text("moderasyon", None, "Soru: a\nCevap: b"), "source": "clips/mqa"}]  # fmt: skip
    G.assign_halves(texts)
    by_exam = {}
    for t in texts:
        if "exam" in t:
            by_exam.setdefault(t["exam"], set()).add(t["split"])
    assert all(len(s) == 1 for s in by_exam.values())
    for subject in ("ders:Fizik", "ders:Tarih"):
        sides = Counter(next(iter(s)) for e, s in by_exam.items() if e.startswith(subject))
        assert sides == {"public": 3, "private": 2}
    for kind in ("islem", "kisisel"):
        sides = Counter(t["split"] for t in texts if t["kind"] == kind)
        assert sides == {"public": 4, "private": 3}
    assert [t["split"] for t in texts if t["track"] == "moderasyon"] == ["public", "private"]
    again = [dict(t) for t in reversed(texts)]
    G.assign_halves(again)
    assert {t["sha"]: t["split"] for t in again} == {t["sha"]: t["split"] for t in texts}


def test_texts_py_rules_drop_adult_gambling_and_stub_texts():
    assert G.drop_reason(text("spam", "istenmeyen", "Bu akşam canlı casino fırsatı sizi bekliyor"))
    assert G.drop_reason(text("spam", "x", "Yeni üyelere deneme bonusu var")) == "gambling"
    assert G.drop_reason(text("spam", "x", "Bu gece live sex yayını için tıklayın")) == "adult"
    stub = text("sss", "tam", "Soru: Kargom nerede kaldı acaba bir bakar mısınız?\nCevap: "
                "Kargom nerede kaldı acaba bir bakar mısınız? teşekkürler", shape="qa")  # fmt: skip
    assert G.drop_reason(stub) == "stub"
    assert (
        G.drop_reason(text("spam", "kisisel", "Akşam sohbet için bize gel, çay demledim")) is None
    )


def test_a_text_already_in_a_training_file_is_dropped_and_counted(tmp_path):
    whole = text("spam", "islem", "Siparişiniz yola çıktı, yarın teslim edilecek, iyi günler.")
    inside = text("hukuk", "İş mahkemesi", "Patronum üç aydır maaşımı yatırmıyor, ne yapmalıyım, "
                  "işten çıkarılmaktan da korkuyorum açıkçası")  # fmt: skip
    fresh = text("spam", "kisisel", "Yarın akşam annemlere gidiyoruz, sen de gelir misin?")
    train = tmp_path / "train.jsonl"
    rows = [{"row_id": "r1", "state": whole["text"]},
            {"row_id": "r2", "state": "Uzun bir giriş cümlesi burada. " * 30 + inside["text"]},
            {"row_id": "r3", "state": "Hiç ilgisi olmayan, tamamen farklı bir metin."}]  # fmt: skip
    train.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    kept, drops, hits = G.prepare([whole, inside, fresh], [], [train])
    assert [t["sha"] for t in kept] == [fresh["sha"]]
    assert drops == {"spam:overlap": 1, "hukuk:overlap": 1}
    assert hits[whole["source_id"]] == [f"{train}:r1"]
    assert hits[inside["source_id"]] == [f"{train}:r2"]


def test_the_private_halves_leave_out_every_text_already_used(tmp_path):
    used = "Soru: Kargom gelmedi, ne yapmalıyım?\nCevap: Takip numaranızla kargo sayfasına bakın."
    built = tmp_path / "built" / "sss"
    built.mkdir(parents=True)
    (built / "train.jsonl").write_text(json.dumps({"row_id": "x", "state": used}) + "\n")
    build_texts = tmp_path / "texts.jsonl"
    build_texts.write_text(json.dumps({"source_id": "seen", "text": "başka"}) + "\n")
    dev_text = "Soru: Faturamı nasıl öderim?\nCevap: Uygulamadaki ödeme ekranından ödeyebilirsiniz."
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps({"text_sha256": [text_hash(dev_text)]}))

    def sample(n, *, config, per_domain, seed):
        rows = [
            (used, "a"),
            ("Soru: Neden?\nCevap: Çünkü bu ürün stokta yok, yakında gelir.", "seen"),
            (dev_text, "b"),
            ("Soru: Bonus?\nCevap: Deneme bonusu ile casino oyna.", "c"),
        ]  # noqa: E501
        rows += [(f"Soru: {config} {seed} soru {i} nedir?\nCevap: Bunun cevabı şöyle açıklanır.",
                  f"{config}-{seed}-{i}") for i in range(4)]  # fmt: skip
        return [{"source_id": sid, "domain": "d", "text": t, "config": config} for t, sid in rows]

    rows = G.sample_private(tmp_path / "built", count=3, sample=sample, build_texts=build_texts,
                            dev_manifest=dev)  # fmt: skip
    assert Counter(r["track"] for r in rows) == {"moderasyon": 3, "sss": 3}
    texts = [r["text"] for r in rows]
    assert used not in texts and dev_text not in texts and len(set(texts)) == len(texts)
    assert not any("bonus" in t or "stokta" in t for t in texts)
    assert {r["config"] for r in rows if r["track"] == "moderasyon"} == {"tr-cqa-question"}


# run and output -------------------------------------------------------------


def small_plan():
    plan, seen = [], set()
    for p in G.generation_plan():
        if p["track"] not in seen:
            seen.add(p["track"])
            plan.append(p)
    return plan


def test_a_run_writes_items_and_meta_with_the_panels_votes_and_resumes(tmp_path):
    plan = small_plan()
    spam = next(p for p in plan if p["track"] == "spam")
    overlap = json.loads(fake_texts(spam["prompt"]))[0]["metin"]
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"row_id": "t", "state": overlap}, ensure_ascii=False) + "\n")
    work, out = tmp_path / "work", tmp_path / "candidates"
    work.mkdir()
    private = [{"track": "moderasyon", "mqa_id": "m1", "domain": "d", "config": "tr-cqa-question",
                "text": "Soru: Komşum gece geç saatte gürültü yapıyor, ne yapabilirim?\nCevap: "
                        "Önce konuşmayı deneyin, olmazsa yönetime yazın."},
               {"track": "sss", "mqa_id": "s1", "domain": "d", "config": "tr-faq-question",
                "text": "Soru: İade süresi kaç gündür?\nCevap: Ürünü 14 gün içinde iade "
                        "edebilirsiniz."}]  # fmt: skip
    (work / "private_texts.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in private), "utf-8"
    )
    client = FakeClient()
    templates = G.held_out_templates()
    result = G.run(client, PANEL, GENERATOR, JUDGE_B, work=work, out=out, files=[train],
                   templates=templates, workers=4, plan=plan)  # fmt: skip
    assert result["dropped"] == {"spam:overlap": 1}
    tracks = result["tracks"]
    assert tracks["spam"]["items"] == 7 and tracks["spam"]["pairs"] == 14
    assert tracks["egitim"]["items"] == 4 and tracks["sss"]["items"] == 6
    assert tracks["moderasyon"]["by_source"] == {"generated": 8, "clips/mqa": 1}
    assert tracks["sss"]["outcomes"] == {"labelled": 24}

    for track in G.TRACKS:
        name = G.FILE_OF.get(track, track)
        items = load_items(out / f"{name}.jsonl")
        metas = [json.loads(x) for x in (out / f"{name}.meta.jsonl").read_text().splitlines()]
        assert [i.id for i in items] == [m["id"] for m in metas]
        for item, meta in zip(items, metas, strict=True):
            assert item.gold == {} and item.track == track
            assert meta["text_sha256"] == text_hash(item.state)
            assert set(meta["votes"]) == set(item.questions)
            judges = {tuple(sorted(v["judges"])) for v in meta["votes"].values()}
            writer_is_a = meta["writer"] == "A"
            assert judges == ({("B", "C")} if writer_is_a else {("A", "C")})
            if meta["source"] == "generated":
                assert meta["licence"] == G.LICENCE["generated"]
            else:
                assert meta["split"] == "private" and meta["licence"].startswith("private only")
    metas = [json.loads(x) for x in (out / "egitim.meta.jsonl").read_text().splitlines()]
    subject = plan[0]["kind"].removeprefix("ders:")
    assert {m["votes"]["ders"]["expected"] for m in metas} == {subject}
    assert {m["votes"]["cevap_puani"]["expected"] for m in metas} == {"0", "1", "2", "3"}
    sss = [json.loads(x) for x in (out / "sss.meta.jsonl").read_text().splitlines()]
    assert Counter(m["split"] for m in sss) == {"public": 5, "private": 1}

    calls = len(client.calls)
    again = G.run(client, PANEL, GENERATOR, JUDGE_B, work=work, out=out, files=[train],
                  templates=templates, workers=4, plan=plan)  # fmt: skip
    assert len(client.calls) == calls
    assert again["tracks"] == result["tracks"]


def test_a_run_refuses_without_training_files_to_check(tmp_path):
    with pytest.raises(ValueError, match="no training files"):
        G.run(FakeClient(), PANEL, GENERATOR, JUDGE_B, work=tmp_path, out=tmp_path, files=[],
              templates=G.held_out_templates(), workers=1, plan=[])  # fmt: skip


def test_the_dry_run_prints_the_plan_and_its_cost_without_a_call(capsys, monkeypatch):
    monkeypatch.setattr(G, "make_client", lambda *a, **k: pytest.fail("no client in a dry run"))
    assert G.main(["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["requests"] == 252 and plan["pairs"] == 2680 and plan["texts"] == 1240
    assert plan["tracks"]["spam"]["pairs_by_judges"] == {"B+C": 640}
    assert plan["tracks"]["sss"]["pairs_by_judges"] == {"A+C": 1280}
    assert plan["calls_by_role"]["B"] == 800 and plan["calls_by_role"]["G"] == 192
    assert 0 < plan["estimate_usd"] <= plan["estimate_with_retries_usd"] < plan["cap_usd"]
    assert all(v["options"] <= 10 for v in plan["held_out_templates"].values())
