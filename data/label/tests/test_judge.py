"""The judge prompt, the log-probability parse and the vote rows, with no network."""

import math

import pytest
from pydantic import TypeAdapter

from data.label.judge import (
    LETTERS,
    SYSTEM_PROMPT,
    JudgeSpec,
    LowLabelMass,
    NoLogprobs,
    judge_messages,
    label_row,
    option_lines,
    parse_distribution,
    shown_orders,
)
from schema.questions import Question
from schema.rows import TrainingRow, mean_vote, outcomes

QUESTION = TypeAdapter(Question)

CHOICE = QUESTION.validate_python(
    {
        "type": "choice",
        "instructions": "Bu şikâyet hangi birime yönlendirilmeli?",
        "criteria": {
            "fatura": None,
            "teknik": "**Bağlantı**  ve\n  cihaz  sorunları",
            "iptal": "Aboneliği `sonlandırma` talepleri",
        },
    }
)
NOUL = QUESTION.validate_python(
    {
        "type": "noul",
        "instructions": "Metin bir şikâyet mi?",
        "criteria": {"true": "Müşteri bir _sorun_ bildiriyor", "false": None},
    }
)
SCORE = QUESTION.validate_python(
    {
        "type": "score",
        "instructions": "Talep ne kadar acil?",
        "criteria": ["Acil değil", "# Orta\n- birkaç gün içinde", "Hemen [bakılmalı](http://x)"],
    }
)
JUDGES = [JudgeSpec("A", "vendor/a", "prov-a"), JudgeSpec("B", "vendor/b", "prov-b")]
JUDGES3 = [*JUDGES, JudgeSpec("C", "vendor/c", "prov-c")]
STATE = "Faturam iki kez kesildi, param iade edilmedi."


def response(top):
    """A chat response whose first token has these (token, probability) alternatives."""
    return {
        "choices": [
            {
                "message": {"content": top[0][0]},
                "logprobs": {
                    "content": [
                        {
                            "token": top[0][0],
                            "logprob": math.log(top[0][1]),
                            "top_logprobs": [
                                {"token": token, "logprob": math.log(p)} for token, p in top
                            ],
                        }
                    ]
                },
            }
        ],
        "usage": {"cost": 0.0001},
    }


# option_lines


def test_choice_lines_use_letters_and_strip_markdown():
    assert option_lines(CHOICE, ["iptal", "fatura", "teknik"]) == [
        "A) iptal: Aboneliği sonlandırma talepleri",
        "B) fatura",
        "C) teknik: Bağlantı ve cihaz sorunları",
    ]


def test_noul_lines_read_evet_and_hayir():
    assert option_lines(NOUL, ["false", "true"]) == [
        "A) Hayır",
        "B) Evet: Müşteri bir sorun bildiriyor",
    ]
    bare = QUESTION.validate_python({"type": "noul", "instructions": "Olumlu mu?"})
    assert option_lines(bare, ["true", "false"]) == ["A) Evet", "B) Hayır"]


def test_score_lines_name_the_level():
    assert option_lines(SCORE, ["0", "1", "2"]) == [
        "A) Düzey 0: Acil değil",
        "B) Düzey 1: Orta birkaç gün içinde",
        "C) Düzey 2: Hemen bakılmalı",
    ]


def test_underscores_inside_a_word_are_kept():
    question = QUESTION.validate_python(
        {"type": "choice", "instructions": "?", "criteria": {"kod_a": "__vurgu__", "b": None}}
    )
    assert option_lines(question, ["kod_a", "b"])[0] == "A) kod_a: vurgu"


def test_an_order_that_is_not_the_options_is_refused():
    with pytest.raises(ValueError):
        option_lines(CHOICE, ["fatura", "teknik"])
    with pytest.raises(ValueError):
        option_lines(CHOICE, ["fatura", "teknik", "teknik"])


def test_messages_follow_the_turkish_template():
    system, user = judge_messages(STATE, NOUL, ["true", "false"])
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert user["role"] == "user"
    assert user["content"] == (
        f"Metin:\n{STATE}\n\nSoru: Metin bir şikâyet mi?\n\nSeçenekler:\n"
        "A) Evet: Müşteri bir sorun bildiriyor\nB) Hayır\n\nCevap (yalnızca harf):"
    )


# parse_distribution


def test_token_variants_map_to_letters_and_add_up():
    top = [(" A", 0.5), ("a", 0.2), ("(B", 0.15), ("C.", 0.1), ("Evet", 0.05)]
    distribution, off = parse_distribution(response(top), ["x", "y", "z"])
    assert off == pytest.approx(0.05)
    assert distribution == pytest.approx({"x": 0.7 / 0.95, "y": 0.15 / 0.95, "z": 0.1 / 0.95})
    assert math.fsum(distribution.values()) == pytest.approx(1.0)


def test_letters_map_to_keys_through_the_shown_order():
    top = [("A", 0.9), ("B", 0.1)]
    distribution, _ = parse_distribution(response(top), ["teknik", "fatura", "iptal"])
    assert distribution == pytest.approx({"teknik": 0.9, "fatura": 0.1, "iptal": 0.0})


def test_letters_beyond_the_options_are_ignored():
    top = [("A", 0.6), ("B", 0.35), ("C", 0.04), ("D", 0.01)]
    distribution, off = parse_distribution(response(top), ["true", "false"])
    assert set(distribution) == {"true", "false"}
    assert off == pytest.approx(0.05)
    assert distribution["true"] == pytest.approx(0.6 / 0.95)


def test_dotless_i_does_not_count_as_letter_i():
    order = [f"k{i}" for i in range(10)]
    top = [("I", 0.5), ("ı", 0.45), ("A", 0.05)]
    with pytest.raises(LowLabelMass):
        parse_distribution(response(top), order)


def test_low_letter_mass_is_rejected():
    top = [("A", 0.5), ("B", 0.3), ("Bence", 0.2)]
    with pytest.raises(LowLabelMass) as caught:
        parse_distribution(response(top), ["x", "y"])
    assert caught.value.mass == pytest.approx(0.8)


@pytest.mark.parametrize(
    "broken",
    [
        {"choices": [{"message": {"content": "A"}}]},
        {"choices": [{"logprobs": None}]},
        {"choices": [{"logprobs": {"content": []}}]},
        {"choices": [{"logprobs": {"content": [{"token": "A", "top_logprobs": []}]}}]},
        {"choices": []},
        {},
    ],
)
def test_missing_logprobs_are_rejected(broken):
    with pytest.raises(NoLogprobs):
        parse_distribution(broken, ["x", "y"])


# shown_orders


def test_choice_orders_rotate_across_judges_and_repeat_for_the_same_row():
    keys = outcomes(CHOICE)
    orders = shown_orders(CHOICE, keys, JUDGES3, "row-123")
    assert orders == shown_orders(CHOICE, keys, JUDGES3, "row-123")
    assert len({tuple(order) for order in orders.values()}) == 3
    base = orders["A"]
    assert orders["B"] == base[1:] + base[:1]
    assert orders["C"] == base[2:] + base[:2]
    assert all(sorted(order) == sorted(keys) for order in orders.values())


def test_the_row_id_sets_the_shuffle():
    keys = [f"secenek-{i}" for i in range(8)]
    question = QUESTION.validate_python(
        {"type": "choice", "instructions": "?", "criteria": dict.fromkeys(keys)}
    )
    firsts = {tuple(shown_orders(question, keys, JUDGES, f"row-{i}")["A"]) for i in range(20)}
    assert len(firsts) > 1


def test_noul_orders_differ_between_two_judges():
    orders = shown_orders(NOUL, ["true", "false"], JUDGES, "row-9")
    assert orders["A"] != orders["B"]


def test_score_levels_stay_in_natural_order():
    orders = shown_orders(SCORE, ["2", "0", "1"], JUDGES3, "row-123")
    assert orders == {name: ["0", "1", "2"] for name in "ABC"}


def test_keys_that_are_not_the_outcomes_are_refused():
    with pytest.raises(ValueError):
        shown_orders(CHOICE, ["fatura", "teknik"], JUDGES, "row")
    with pytest.raises(ValueError):
        shown_orders(CHOICE, outcomes(CHOICE), [JUDGES[0], JUDGES[0]], "row")


def test_judge_names_are_letters_a_to_d():
    with pytest.raises(ValueError):
        JudgeSpec("E", "m", "p")


# label_row


class FakeClient:
    """Answers each judge by model id; the answer is a map from key to probability."""

    def __init__(self, answers, fail_model=None):
        self.answers = answers
        self.fail_model = fail_model
        self.calls = []

    def chat(self, model, provider, messages, **kwargs):
        self.calls.append((model, provider, messages, kwargs))
        if model == self.fail_model:
            raise RuntimeError("judge unavailable")
        order = self._order_from(messages)
        top = [(LETTERS[i], self.answers[model][key]) for i, key in enumerate(order)]
        return response([(token, p) for token, p in top if p > 0])

    @staticmethod
    def _order_from(messages):
        lines = messages[1]["content"].split("Seçenekler:\n")[1].split("\n\n")[0].split("\n")
        texts = [line[3:] for line in lines]
        by_text = {"iptal": "iptal", "fatura": "fatura", "teknik": "teknik"}
        return [by_text[text.split(":")[0]] for text in texts]


ANSWERS = {
    "vendor/a": {"fatura": 0.7, "teknik": 0.2, "iptal": 0.1},
    "vendor/b": {"fatura": 0.4, "teknik": 0.6, "iptal": 0.0},
    "vendor/c": {"fatura": 0.1, "teknik": 0.1, "iptal": 0.8},
}


def test_label_row_builds_votes_whose_mean_is_the_arithmetic_mean():
    client = FakeClient(ANSWERS)
    votes = label_row(client, JUDGES3, "row-42", STATE, CHOICE)
    assert [vote.judge for vote in votes] == ["A", "B", "C"]
    for vote, model in zip(votes, ["vendor/a", "vendor/b", "vendor/c"], strict=True):
        assert vote.distribution == pytest.approx(ANSWERS[model])
        assert vote.off_letter_mass == pytest.approx(0.0, abs=1e-9)
    keys = outcomes(CHOICE)
    mean = mean_vote(votes, keys)
    for key in keys:
        assert mean[key] == pytest.approx(sum(ANSWERS[m][key] for m in ANSWERS) / 3)
    assert votes[0].shown_order != votes[1].shown_order
    row = TrainingRow(
        track="triage",
        task="abonelik-sikayet",
        split="train",
        origin="generated",
        label_kind="judges",
        source="authored",
        state=STATE,
        question=CHOICE,
        target=mean,
        judges=votes,
        recipe="label-test",
    )
    assert row.target == pytest.approx(mean)


def test_label_row_calls_with_one_token_logprobs_and_reasoning_off():
    client = FakeClient(ANSWERS)
    label_row(client, JUDGES, "row-42", STATE, CHOICE)
    assert [(model, provider) for model, provider, _, _ in client.calls] == [
        ("vendor/a", "prov-a"),
        ("vendor/b", "prov-b"),
    ]
    for _, _, _, kwargs in client.calls:
        assert kwargs == {
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 10,
            "reasoning_effort": "none",
            "temperature": 0.0,
        }


def test_label_row_reraises_when_one_judge_fails():
    client = FakeClient(ANSWERS, fail_model="vendor/b")
    with pytest.raises(RuntimeError, match="judge unavailable"):
        label_row(client, JUDGES3, "row-42", STATE, CHOICE)


def test_label_row_reraises_a_low_mass_answer():
    answers = {**ANSWERS, "vendor/c": {"fatura": 0.5, "teknik": 0.3, "iptal": 0.0}}
    with pytest.raises(LowLabelMass):
        label_row(FakeClient(answers), JUDGES3, "row-42", STATE, CHOICE)


def test_stated_probabilities_are_parsed_renormalised_and_checked():
    import pytest as _pytest

    from data.label.judge import LowLabelMass, NoStatedDistribution, parse_stated

    def reply(text):
        return {"choices": [{"message": {"content": text}}]}

    order = ["kargo", "fatura", "iade"]
    dist, missing = parse_stated(reply('```json\n{"A": 0.7, "B": 0.25}\n```'), order)
    assert dist["kargo"] == _pytest.approx(0.7 / 0.95) and dist["iade"] == 0.0
    assert missing == _pytest.approx(0.05)
    assert parse_stated(reply('{"a": 0.5, "(B)": 0.5}'), order)[0]["fatura"] == 0.5
    for bad in ("hiç JSON yok", '{"D": 1.0}', '{"A": -0.1, "B": 1.1}', '{"A": "yüksek"}', "{}"):
        with _pytest.raises(NoStatedDistribution):
            parse_stated(reply(bad), order)
    with _pytest.raises(LowLabelMass):
        parse_stated(reply('{"A": 0.3, "B": 0.2}'), order)
