"""Generated tracks: the plan, parsing what comes back, splits, pairing and labelling."""

import json
import math

from data.label import synth as S
from data.label.judge import LETTERS, JudgeSpec


def test_the_plan_asks_for_every_kind_in_batches():
    plan = S.generation_plan(1)
    kinds = {p["kind"] for p in plan}
    assert {k.name for k in S.MESSAGE_KINDS} <= kinds
    assert set(S.COURTS) <= kinds
    assert sum(p["shape"] == "exam" for p in plan) == S.EDU_QUESTIONS
    assert len({p["id"] for p in plan}) == len(plan)


def test_a_list_answer_becomes_masked_texts_of_its_kind():
    request = {"id": "r", "kind": "dolandirici", "shape": "list", "prompt": "x"}
    content = "```json\n" + json.dumps([
        {"metin": "Kargonuz gümrükte bekliyor, ücreti ödemek için tıklayın: https://x.co/a"},
        {"metin": "kısa"}, "bozuk"]) + "\n```"  # fmt: skip
    texts = S.texts_from(request, content)
    assert len(texts) == 1 and texts[0]["kind"] == "dolandirici"
    assert "https://" not in texts[0]["text"]
    assert S.texts_from(request, "not json") == []


def test_an_exam_answer_needs_all_four_levels_and_keeps_them_together():
    request = {"id": "e", "kind": "ders:Fizik", "shape": "exam", "prompt": "x"}
    good = {
        "soru": "Sürtünme kuvveti nedir?",
        "cevaplar": [{"seviye": i, "metin": f"cevap {i}"} for i in range(4)],
    }
    texts = S.texts_from(request, json.dumps(good))
    assert [t["kind"] for t in texts] == ["seviye0", "seviye1", "seviye2", "seviye3"]
    assert len({t["exam"] for t in texts}) == 1
    bad = {**good, "cevaplar": good["cevaplar"][:3]}
    assert S.texts_from(request, json.dumps(bad)) == []


def test_every_answer_to_one_exam_question_lands_in_one_split():
    exam = "a" * 20
    splits = {S.split_of(exam) for _ in range(4)}
    assert len(splits) == 1


def test_pairs_ask_each_question_of_the_right_texts():
    texts = [
        {"kind": "dolandirici", "text": "t1", "source_id": "1", "split": "train"},
        {"kind": "seviye3", "exam": "x", "subject": "ders:Fizik", "text": "t2",
         "source_id": "2", "split": "train"},
        {"kind": "seviye1", "exam": "x", "subject": "ders:Fizik", "text": "t3",
         "source_id": "3", "split": "train"},
        {"kind": "facturk", "text": "t4", "source_id": "4", "split": "validation"},
    ]  # fmt: skip
    got = sorted((a.task, t["source_id"], t["kind"]) for a, t in S.pairs(texts))
    assert got == [
        ("cevap_puani", "2", "seviye3"), ("cevap_puani", "3", "seviye1"),
        ("ders", "2", "ders:Fizik"), ("dogrulanabilir", "4", "facturk"),
        ("mesaj_turu", "1", "dolandirici"), ("oltalama", "1", "dolandirici"),
    ]  # fmt: skip


class FakeClient:
    """Both judges pick the option whose line mentions the expected answer, if any."""

    spent = 0.0

    def chat(self, model, provider, messages, **kwargs):
        content = messages[1]["content"]
        lines = content.split("Seçenekler:\n")[1].split("\n\n")[0].split("\n")
        pick = next((i for i, line in enumerate(lines) if "dolandırıcılık" in line.lower()
                     or "Evet" in line), 0)  # fmt: skip
        rest = 0.2 / (len(lines) - 1)
        top = [{"token": LETTERS[i], "logprob": math.log(0.8 if i == pick else rest)}
               for i in range(len(lines))]  # fmt: skip
        return {"choices": [{"logprobs": {"content": [{"token": "A", "top_logprobs": top}]}}]}


def test_a_pair_is_labelled_by_the_panel_and_journaled_with_its_intent(tmp_path):
    judges = [JudgeSpec("A", "vendor/a", "p"), JudgeSpec("C", "vendor/c", "p")]
    ask = next(a for a in S.asks() if a.task == "oltalama")
    text = {"kind": "dolandirici", "text": "Hesabınız askıya alındı, kodu paylaşın.",
            "source_id": "z", "split": "train", "source": S.SOURCE}  # fmt: skip
    record = S.label_one(FakeClient(), judges, ask, text)
    assert record["outcome"] == "labelled" and record["expected"] == "true"
    assert record["row"]["task"] == "oltalama-oltalama" and record["row"]["track"] == "oltalama"
    report = S.assemble(tmp_path, [text], {record["key"]: record}, FakeClient())
    assert report["tasks"]["oltalama"]["judges_agree_with_intended_kind"] == 1.0


def test_every_message_kind_has_one_writer_and_other_tracks_the_generator():
    plan = S.generation_plan(1)
    writers = {p["kind"]: p.get("writer", "G") for p in plan}
    assert {writers[k.name] for k in S.MESSAGE_KINDS} == {"A"}
    assert {writers[c] for c in S.COURTS} == {"G"}


def test_a_writer_placeholder_is_dropped_and_the_masks_are_kept():
    request = {"id": "r", "kind": "dolandirici", "shape": "list", "prompt": "x"}
    content = json.dumps([
        {"metin": "Sayın müşterimiz, [marka] hesabınız askıya alındı, hemen giriş yapın."},
        {"metin": "Hesabınız askıya alındı, [bağlantı] adresinden hemen giriş yapın lütfen."},
    ])  # fmt: skip
    texts = S.texts_from(request, content)
    assert [("[marka]" in t["text"]) for t in texts] == [False]


def test_check_worthiness_takes_both_labels_from_both_sources():
    ask = next(a for a in S.asks() if a.task == "dogrulanabilir")
    assert {"facturk", "iddia", "mqa_soru"} <= set(ask.only)
    assert ask.expected["facturk"] == ask.expected["iddia"] == "true"
    assert all(ask.expected[k.name] == "false" for k in S.NON_CLAIM_KINDS)
    assert "mqa_soru" not in ask.expected


def test_every_exam_request_has_its_own_prompt_and_a_topic_split():
    exams = [p for p in S.generation_plan(1) if p["shape"] == "exam"]
    assert len({p["prompt"] for p in exams}) == len(exams) == S.EDU_QUESTIONS
    assert {p["split"] for p in exams} == {"train", "validation"}


def test_one_scene_lands_in_one_split_and_every_kind_has_validation():
    plan = [p for p in S.generation_plan(1) if p["shape"] == "list"]
    by_prompt = {}
    for p in plan:
        by_prompt.setdefault(p["prompt"], set()).add(p["split"])
    assert all(len(splits) == 1 for splits in by_prompt.values())
    for kind in {p["kind"] for p in plan}:
        assert "validation" in {p["split"] for p in plan if p["kind"] == kind}, kind


def test_a_message_naming_a_real_bank_or_body_is_dropped():
    request = {"id": "r", "kind": "dolandirici", "shape": "list", "brand_check": True,
               "split": "train", "prompt": "x"}  # fmt: skip
    content = json.dumps(
        [
            {"metin": "Ziraat Bankası hesabınız askıya alındı, [bağlantı] girin."},
            {"metin": "Nova Bank hesabınız askıya alındı, [bağlantı] girin lütfen."},
        ]
    )
    texts = S.texts_from(request, content)
    assert [t["text"].startswith("Nova") for t in texts] == [True]
    assert texts[0]["split"] == "train"


def test_a_malformed_exam_answer_is_dropped_not_fatal():
    request = {"id": "e", "kind": "ders:Fizik", "shape": "exam", "prompt": "x"}
    bad = {"soru": "Soru?", "cevaplar": [{"seviye": None, "metin": "a"},
                                         {"seviye": "iki", "metin": "b"}]}  # fmt: skip
    assert S.texts_from(request, json.dumps(bad)) == []


def test_the_writer_s_vote_is_replaced_on_its_own_texts(tmp_path):
    judges = [JudgeSpec("A", "vendor/a", "p"), JudgeSpec("C", "vendor/c", "p")]
    ask = next(a for a in S.asks() if a.task == "oltalama")
    text = {"kind": "dolandirici", "text": "Hesabınız askıya alındı, kodu paylaşın.",
            "source_id": "z", "split": "train", "source": S.SOURCE}  # fmt: skip
    record = S.label_one(FakeClient(), judges, ask, text)
    records = {record["key"]: record}
    judge_b = JudgeSpec("B", "vendor/b", "p")
    out = S.relabel_without_writer(FakeClient(), judge_b, records, tmp_path / "j.jsonl", 2)
    new = out[record["key"]]
    assert new["relabelled"] and sorted(v["judge"] for v in new["row"]["judges"]) == ["B", "C"]
    again = S.relabel_without_writer(FakeClient(), judge_b, out, tmp_path / "j.jsonl", 2)
    assert again[record["key"]] is new  # nothing relabelled twice
