"""The full build: splits by text and template, screening, stopping, the cap, resuming."""

import hashlib
import json
import math

import pytest

from data.label import build as B
from data.label.generate import Template, to_question
from data.label.judge import LETTERS, JudgeSpec
from schema.rows import JudgeVote, TrainingRow, outcomes

CHOICE = Template.model_validate({
    "family": "konu", "type": "choice", "question": "Metin hangi konuyla ilgili?",
    "applies_when": "Bir ürün ya da hizmet sorusu.",
    "options": [{"name": "Kargo"}, {"name": "Fatura"}, {"name": "Teknik sorun"}],
})  # fmt: skip
NOUL = Template.model_validate({
    "family": "insan_destegi", "type": "noul", "question": "Bir temsilci gerekiyor mu?",
    "applies_when": "Bir müşteri talebi.", "true": "Temsilci gerekir.",
    "false": "Kendi başına çözülür.",
})  # fmt: skip


SPLIT = Template.model_validate({**NOUL.model_dump(), "question": "Konu acil mi sence?",
                                 "family": "aciliyet"})  # fmt: skip


RARE = Template.model_validate({**NOUL.model_dump(), "question": "Bu talep çok acil mi?",
                                "family": "aciliyet"})  # fmt: skip


def row(top: str, keys=("a", "b", "c"), state="metin") -> TrainingRow:
    """A row both judges voted for `top`."""
    target = {k: (0.8 if k == top else 0.2 / (len(keys) - 1)) for k in keys}
    votes = [JudgeVote(judge=name, shown_order=list(keys), distribution=target,
                       off_letter_mass=0.0) for name in ("A", "C")]  # fmt: skip
    return TrainingRow(
        track="sss", task="t", split="train", origin="generated", label_kind="judges",
        source="s", state=f"{state}-{top}", question={"type": "choice", "instructions": "?",
        "criteria": dict.fromkeys(keys)}, target=target, judges=votes, recipe="r",
    )  # fmt: skip


def test_balance_caps_the_most_common_answer_and_keeps_the_rest_apart():
    rows = [row("a", state=str(i)) for i in range(8)] + [row("b", state=str(i)) for i in range(2)]
    kept, surplus = B.balance(rows, 0.5)
    assert sorted(B.top_of(r.target) for r in kept) == ["a", "a", "b", "b"]
    assert len(surplus) == 6
    even = [row(k, state=str(i)) for i, k in enumerate("abcabc")]
    assert B.balance(even, 0.5) == (even, [])


def test_screen_drops_a_template_whose_blind_answer_is_its_majority():
    rows = [row("a", state=str(i)) for i in range(6)] + [row("b", state=str(i)) for i in range(4)]
    assert B.screen_verdict(rows, {"a": 0.9, "b": 0.05, "c": 0.05}) == "answerable_blind"
    assert B.screen_verdict(rows, {"a": 0.05, "b": 0.9, "c": 0.05}) == "passed"
    assert B.screen_verdict(rows[:3], {"a": 0.4, "b": 0.3, "c": 0.3}) == "rarely_applies"
    assert B.screen_verdict(rows[:6], {"a": 0.4, "b": 0.3, "c": 0.3}) == "low_variance"


def test_a_blind_null_answer_does_not_drop_a_yes_or_no_template():
    no = [row("false", keys=("true", "false"), state=str(i)) for i in range(6)]
    yes = [row("true", keys=("true", "false"), state=str(i)) for i in range(4)]
    assert B.screen_verdict(no + yes, {"true": 0.02, "false": 0.98}, "noul") == "passed"
    # A confident blind "yes" that is also the majority is a real clue.
    assert B.screen_verdict(yes * 2 + no[:3], {"true": 0.95, "false": 0.05},
                            "noul") == "answerable_blind"  # fmt: skip
    rows = [row("a", state=str(i)) for i in range(6)] + [row("b", state=str(i)) for i in range(4)]
    assert B.screen_verdict(rows, {"a": 0.9, "b": 0.05, "c": 0.05}, "choice") == "answerable_blind"


def test_a_skewed_template_stops_once_it_has_enough_rows():
    skewed = [row("a", state=str(i)) for i in range(26)] + [row("b", state=str(i))
                                                           for i in range(6)]  # fmt: skip
    assert B.stop_reason(skewed[:20], "choice") is None
    assert B.stop_reason(skewed, "choice") == "skewed"


def texts(n=200):
    text = "Soru: bu ürün {i} için ne zaman gelir?\nCevap: sipariş {i} iki gün içinde kargoda."
    return [{"source_id": f"id{i}", "text": text.format(i=i),
             "config": "tr-faq-question"} for i in range(n)]  # fmt: skip


def test_held_out_cells_cover_every_type():
    cells = B.held_out_cells()
    assert sorted(cell.rsplit("-", 1)[1] for cell in cells) == ["choice", "choice", "noul", "score"]
    held = {cell.rsplit("-", 1)[0] for cell in cells if cell.endswith("-choice")}
    for family in ("konu", "niyet", "aciliyet", "bilgi_turu", "hedef_kitle", "hassasiyet"):
        template = Template.model_validate({**CHOICE.model_dump(), "family": family})
        assert B.held_out(template) == (family in held)


def test_no_training_row_shares_a_text_with_an_evaluation_row():
    sample = texts()
    heldout = {"tid": "h", "held_out": True}
    trained = {"tid": "t", "held_out": False}
    for text, split in B.pairs_for(heldout, sample, 100, 1):
        assert split == "heldout_task" and B.text_pool(text["source_id"]) == "eval"
    splits = {}
    for text, split in B.pairs_for(trained, sample, 200, 1):
        splits.setdefault(split, set()).add(text["source_id"])
        assert (split == "validation") == (B.text_pool(text["source_id"]) == "eval")
    assert splits["train"] and splits["validation"]
    assert not splits["train"] & splits["validation"]


class FakeClient:
    """Judges that agree: "Evet" to every check, an option by a hash of the text otherwise."""

    def __init__(self):
        self.calls = 0
        self.spent = 0.0

    def chat(self, model, provider, messages, **kwargs):
        self.calls += 1
        content = messages[1]["content"]
        lines = content.split("Seçenekler:\n")[1].split("\n\n")[0].split("\n")
        state = content.split("Metin:\n")[1].split("\n\nSoru:")[0]
        asked = content.split("\n\nSoru: ")[1].split("\n")[0]
        names = [line[3:].split(":")[0] for line in lines]
        seen = int(hashlib.md5(state.encode()).hexdigest(), 16)
        if asked == NOUL.question:
            wanted = ("Evet", "Hayır")[seen % 2]
        elif asked == RARE.question:  # "yes" on one text in ten
            wanted = "Evet" if seen % 10 == 0 else "Hayır"
        elif asked == SPLIT.question:  # the two judges answer independently
            wanted = ("Evet", "Hayır")[
                int(hashlib.md5((state + model).encode()).hexdigest(), 16) % 2
            ]
        elif "Evet" in names:  # the scope and template checks
            wanted = "Evet"
        else:
            wanted = sorted(names)[seen % len(names)]
        pick = names.index(wanted)
        # Judge C is less sure on the split question, so its disagreements do not tie.
        sure = 0.7 if asked == SPLIT.question and model.endswith("/c") else 0.8
        rest = (1 - sure) / (len(lines) - 1)
        top = [{"token": LETTERS[i], "logprob": math.log(sure if i == pick else rest)}
               for i in range(len(lines))]  # fmt: skip
        return {"choices": [{"logprobs": {"content": [{"token": "A", "top_logprobs": top}]}}]}


def test_the_build_labels_splits_and_resumes_without_paying_twice(tmp_path):
    judges = [JudgeSpec("A", "vendor/a", "p"), JudgeSpec("C", "vendor/c", "p")]
    entries = []
    for template in (CHOICE, NOUL):
        tid = B.template_id(template)
        entries.append(
            {
                "tid": tid,
                "held_out": False,
                "check": 1.0,
                "status": "ready",
                "blind": {"x": 1.0},
                "template": template.model_dump(mode="json"),
            }
        )
    for entry in entries:  # a blind prior over the template's own keys
        keys = outcomes(B.question_of(to_question(Template.model_validate(entry["template"]))))
        entry["blind"] = {k: 1 / len(keys) for k in keys}
    (tmp_path / "templates.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (tmp_path / "generated.jsonl").write_text(CHOICE.model_dump_json() + "\n"
                                              + NOUL.model_dump_json() + "\n")  # fmt: skip
    client = FakeClient()
    result = B.build(client, judges, {}, "A", tmp_path, texts(), 0, 60, 4, 1)
    assert result["rows"] > 0
    assert set(result["rows_by_split"]) <= {"train", "validation"}
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert all(r["task"].rsplit("-", 1)[1] in {e["tid"] for e in entries} for r in rows)
    calls = client.calls
    again = B.build(client, judges, {}, "A", tmp_path, texts(), 0, 60, 4, 1)
    assert client.calls == calls
    assert again["rows"] == result["rows"]


def test_a_restart_with_a_longer_plan_never_resumes_a_stopped_template(tmp_path):
    judges = [JudgeSpec("A", "vendor/a", "p"), JudgeSpec("C", "vendor/c", "p")]
    entry = {
        "tid": B.template_id(SPLIT),
        "held_out": False,
        "check": 1.0,
        "status": "ready",
        "blind": {"true": 0.5, "false": 0.5},
        "template": SPLIT.model_dump(mode="json"),
    }
    (tmp_path / "templates.jsonl").write_text(json.dumps(entry) + "\n")
    (tmp_path / "generated.jsonl").write_text(SPLIT.model_dump_json() + "\n")
    first = B.build(FakeClient(), judges, {}, "A", tmp_path, texts(400), 0, 120, 4, 1)
    status = first["per_template"][entry["tid"]]["status"]
    assert status == "judges_disagree"
    labelled = first["per_template"][entry["tid"]]["labelled"]
    assert labelled > B.SCREEN_PAIRS  # past the screen, where a negative limit once bit
    client = FakeClient()
    B.build(client, judges, {}, "A", tmp_path, texts(400), 0, 300, 4, 1)
    assert client.calls == 0


def test_a_rare_answer_is_mined_from_training_texts_only(tmp_path):
    judges = [JudgeSpec("A", "vendor/a", "p"), JudgeSpec("C", "vendor/c", "p")]
    entry = {
        "tid": B.template_id(RARE),
        "held_out": False,
        "check": 1.0,
        "status": "ready",
        "blind": {"true": 0.5, "false": 0.5},
        "template": RARE.model_dump(mode="json"),
    }
    (tmp_path / "templates.jsonl").write_text(json.dumps(entry) + "\n")
    (tmp_path / "generated.jsonl").write_text(RARE.model_dump_json() + "\n")
    result = B.build(FakeClient(), judges, {}, "A", tmp_path, texts(600), 0, 60, 4, 1)
    assert result["per_template"][entry["tid"]]["status"].startswith("mined_")
    journal = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    mined = [r for r in journal if r.get("mined")]
    assert mined and all(r["outcome"] in ("mined_majority", "labelled", "no_fit") for r in mined)
    assert all(r["row"]["split"] == "train" for r in mined if "row" in r)
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    train = [r for r in rows if r["split"] == "train"]
    yes = sum(r["target"]["true"] > 0.5 for r in train)
    assert yes >= B.MINE_MIN_ROWS
    assert yes / len(train) >= 1 - B.CAP_TOP_SHARE["noul"] - 1e-9
    assert {r["recipe"] for r in rows if r["split"] == "train"} == {B.RECIPE, B.RECIPE_MINED}
    assert all(r["recipe"] == B.RECIPE for r in rows if r["split"] != "train")
    # A restart finds nothing to do and reaches the same statuses.
    client = FakeClient()
    again = B.build(client, judges, {}, "A", tmp_path, texts(600), 0, 60, 4, 1)
    assert client.calls == 0
    assert again["per_template"] == result["per_template"]


def test_promotion_splits_rows_and_refuses_a_text_shared_across_splits(tmp_path):
    from data.label.promote import promote

    train, validation = row("a", state="bu ürün için bir"), row("b", state="bu kargo için iki")
    rows = tmp_path / "rows.jsonl"
    rows.write_text(train.model_dump_json() + "\n"
                    + validation.model_copy(update={"split": "validation"}).model_dump_json()
                    + "\n")  # fmt: skip
    counts = {"train": 1, "validation": 1, "dropped_by_current_filters": 0}
    assert promote(rows, tmp_path / "out") == counts
    leak = row("a", state="bu ürün için bir").model_copy(update={"split": "validation"})
    rows.write_text(train.model_dump_json() + "\n" + leak.model_dump_json() + "\n")
    with pytest.raises(ValueError):
        promote(rows, tmp_path / "leak")


def test_a_refused_text_is_journaled_and_a_server_error_is_retried():
    from data.label.client import ApiError

    assert not B.is_transient(ApiError("refused", status=200))
    assert not B.is_transient(ApiError("bad request", status=400))
    assert B.is_transient(ApiError("rate limited", status=429))
    assert B.is_transient(ApiError("server", status=503))
    assert B.is_transient(ApiError("no answer", status=None))
    assert B.is_transient(TimeoutError())
    assert not B.is_transient(ValueError("parse"))
