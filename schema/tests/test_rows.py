"""A training row: its target matches its question, and its labels add up."""

import pytest

from schema.rows import JudgeVote, TrainingRow, mean_vote, outcomes

CHOICE = {
    "type": "choice",
    "instructions": "Bu şikâyet hangi birime yönlendirilmeli?",
    "criteria": {"fatura": None, "teknik": None, "iptal": "Aboneliği sonlandırma talepleri"},
}


def base(**overrides):
    row = {
        "track": "triage",
        "task": "abonelik-sikayet",
        "split": "train",
        "origin": "generated",
        "label_kind": "rule",
        "source": "authored",
        "state": "Faturam iki kez kesildi.",
        "question": CHOICE,
        "target": {"fatura": 1.0, "teknik": 0.0, "iptal": 0.0},
        "recipe": "gen-v1",
    }
    return {**row, **overrides}


def vote(judge, dist, order=("fatura", "teknik", "iptal"), off=0.0):
    return JudgeVote(judge=judge, shown_order=list(order), distribution=dist, off_letter_mass=off)


def test_targets_are_keyed_like_the_answer_contract():
    assert outcomes(TrainingRow(**base()).question) == ["fatura", "teknik", "iptal"]
    score = TrainingRow(
        **base(
            question={
                "type": "score",
                "instructions": "Aciliyet?",
                "criteria": ["düşük", "orta", "yüksek"],
            },
            target={"0": 0.1, "1": 0.2, "2": 0.7},
        )
    )
    assert outcomes(score.question) == ["0", "1", "2"]
    noul = TrainingRow(
        **base(
            question={"type": "noul", "instructions": "Acil mi?"},
            target={"true": 0.3, "false": 0.7},
        )
    )
    assert outcomes(noul.question) == ["true", "false"]
    with pytest.raises(ValueError, match="keys do not match"):
        TrainingRow(**base(target={"fatura": 1.0}))
    with pytest.raises(ValueError, match="sum to"):
        TrainingRow(**base(target={"fatura": 0.5, "teknik": 0.1, "iptal": 0.1}))


def test_a_judge_row_is_the_arithmetic_mean_of_its_votes():
    votes = [
        vote("A", {"fatura": 0.9, "teknik": 0.1, "iptal": 0.0}),
        vote(
            "B", {"fatura": 0.5, "teknik": 0.5, "iptal": 0.0}, order=("iptal", "fatura", "teknik")
        ),
        vote(
            "C", {"fatura": 1.0, "teknik": 0.0, "iptal": 0.0}, order=("teknik", "iptal", "fatura")
        ),
    ]
    target = mean_vote(votes, ["fatura", "teknik", "iptal"])
    assert target["fatura"] == pytest.approx(0.8)
    row = TrainingRow(**base(label_kind="judges", judges=votes, target=target))
    assert len(row.judges) == 3
    # A target that is not the mean, such as the sharper geometric mean, is refused.
    sharp = {"fatura": 0.9, "teknik": 0.1, "iptal": 0.0}
    with pytest.raises(ValueError, match="arithmetic mean"):
        TrainingRow(**base(label_kind="judges", judges=votes, target=sharp))
    with pytest.raises(ValueError, match="at least two"):
        TrainingRow(**base(label_kind="judges", judges=votes[:1], target=votes[0].distribution))
    with pytest.raises(ValueError, match="appears twice"):
        TrainingRow(
            **base(label_kind="judges", judges=[votes[0], votes[0]], target=votes[0].distribution)
        )
    bad_order = vote("C", {"fatura": 1.0, "teknik": 0.0, "iptal": 0.0}, order=("fatura", "teknik"))
    with pytest.raises(ValueError, match="shown_order"):
        TrainingRow(**base(label_kind="judges", judges=[votes[0], bad_order], target=target))


def test_labels_and_their_kind_agree():
    with pytest.raises(ValueError, match="carries no judge votes"):
        TrainingRow(**base(judges=[vote("A", {"fatura": 1.0, "teknik": 0.0, "iptal": 0.0})]))
    with pytest.raises(ValueError, match="one-hot"):
        TrainingRow(**base(label_kind="human", target={"fatura": 0.6, "teknik": 0.4, "iptal": 0.0}))
    assert TrainingRow(**base(label_kind="human")).label_kind == "human"


def test_the_row_id_is_a_content_hash_that_ignores_the_labels():
    first = TrainingRow(**base())
    relabelled = TrainingRow(**base(target={"fatura": 0.0, "teknik": 1.0, "iptal": 0.0}))
    assert first.row_id == relabelled.row_id, "the same state and question are one row"
    assert TrainingRow(**base(state="Başka bir metin.")).row_id != first.row_id
    with pytest.raises(ValueError, match="content hash"):
        TrainingRow(**base(row_id="deadbeef"))
    assert TrainingRow(**base(row_id=first.row_id)).row_id == first.row_id


def test_a_row_round_trips_through_json():
    row = TrainingRow(**base())
    assert TrainingRow.model_validate_json(row.model_dump_json()) == row


def test_a_training_row_carries_at_most_ten_options():
    eleven = {f"secenek {i}": None for i in range(11)}
    target = dict.fromkeys(eleven, 0.0)
    target["secenek 0"] = 1.0
    with pytest.raises(ValueError, match="at most 10 options"):
        TrainingRow(
            **base(
                question={"type": "choice", "instructions": "?", "criteria": eleven}, target=target
            )
        )
