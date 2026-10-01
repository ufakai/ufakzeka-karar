"""The second sweep: routing by the panel and two blind passes, and the owner's desk."""

import pytest

from bench.hakembench import sweep as S

OPTS = [{"key": "true", "label": "Evet", "detail": ""}, {"key": "false", "label": "Hayır",
        "detail": ""}]  # fmt: skip
LEVELS = [{"key": str(k), "label": f"Düzey {k}", "detail": ""} for k in range(4)]


def unit(uid, track="spam", proposed="true", kind="noul"):
    options = OPTS if kind == "noul" else LEVELS
    return {"id": uid, "track": track, "order": 0, "text": f"metin {uid}",
            "questions": [{"qid": "q", "question": "?", "type": kind, "options": options,
                           "proposed": proposed}]}  # fmt: skip


def said(answer, flag="none"):
    return {"q": {"answer": answer, "flag": flag}}


def reviews(first: dict, second: dict) -> list[dict]:
    """The first pass stores answers unwrapped, the second under "answers" with content."""
    return [first, {uid: {"answers": a, "content": "none"} for uid, a in second.items()}]


def test_units_route_by_unanimity_and_every_catch_goes_to_the_owner():
    units = [unit("u1"), unit("u2"), unit("u3")]
    r = reviews({"u1": said("true"), "u2": said("false"), "u3": said("false")},
                {"u1": said("true"), "u2": said("true"), "u3": said("false")})  # fmt: skip
    catches = [{"unit": "u3", "qid": "q", "truth": "false", "shown": "true"}]
    got = S.plan(units, r, catches, set(), {})
    assert got["status"]["u2"] == "disputed" and got["status"]["u3"] == "catch"
    assert got["adjudicate"] == {"u2": ["q"]}
    assert got["catches"][0]["suspect"] is False


def test_a_catch_every_source_answered_with_the_shown_answer_is_suspect():
    r = reviews({"u1": said("true")}, {"u1": said("true")})
    catches = [{"unit": "u1", "qid": "q", "truth": "false", "shown": "true"}]
    assert S.plan([unit("u1")], r, catches, set(), {})["catches"][0]["suspect"] is True


def test_a_choice_catch_shows_the_wrong_option_a_reviewer_picked():
    catch = {"truth": "a", "shown": "c"}
    assert S.plausible_shown(catch, [{"answer": "a"}, {"answer": "b"}]) == "b"
    assert S.plausible_shown(catch, [{"answer": "a"}, {"answer": "a"}]) == "c"


def test_content_flags_and_reviewer_flags_route_the_unit():
    units = [unit("u1"), unit("u2")]
    r = [{"u1": said("true"), "u2": said("true", "cant_tell")},
         {"u1": {"answers": said("true"), "content": "illegal_promo"},
          "u2": {"answers": said("true"), "content": "none"}}]  # fmt: skip
    got = S.plan(units, r, [], set(), {})
    assert got["status"] == {"u1": "content", "u2": "disputed"}
    assert got["content"] == ["u1"]


def test_the_sample_keeps_earlier_sampled_units_that_are_still_agreed():
    units = [unit(f"u{i:02d}") for i in range(50)]
    both = {u["id"]: said("true") for u in units}
    got = S.plan(units, reviews(both, both), [], set(), {"u07": "sampled", "u08": "disputed"})
    sampled = {uid for uid, s in got["status"].items() if s == "sampled"}
    assert "u07" in sampled and len(sampled) == 10


def test_shifted_catches_move_one_level_on_agreed_score_questions(monkeypatch):
    monkeypatch.setattr(S, "SHIFTED_PER_TRACK", {"egitim": 3})
    units = [unit(f"e{i}", "egitim", "0", "score") for i in range(5)]
    both = {u["id"]: said("0") for u in units}
    got = S.plan(units, reviews(both, both), [], set(), {})
    assert len(got["shifted"]) == 3
    assert all(c["shown"] == "1" and c["kind"] == "shifted_level" for c in got["shifted"])


def test_the_owner_desk_shows_adjudicated_catch_and_blind_answers_blind_first(monkeypatch):
    monkeypatch.setattr(S, "SHIFTED_PER_TRACK", {})
    units = [unit("s1"), unit("s2"), unit("d1", "dogrulama", "2", "score")]
    r = reviews({"s1": said("false"), "s2": said("true"), "d1": said("2")},
                {"s1": said("false"), "s2": said("true"), "d1": said("2")})  # fmt: skip
    catches = [{"unit": "s2", "qid": "q", "truth": "true", "shown": "false"}]
    got = S.plan(units, r, catches, set(), {})
    assert got["blind"] == [["d1", "q"]]
    adjudicated = {"s1": {"answers": {"q": {"answer": "false", "confidence": "high"}}}}
    desk = S.owner_units(got, units, adjudicated)
    shown = {u["id"]: u["questions"][0]["proposed"] for u in desk}
    assert shown == {"d1": None, "s1": "false", "s2": "false"}
    assert desk[0]["id"] == "d1" and [u["order"] for u in desk] == [0, 1, 2]
    with pytest.raises(ValueError, match="no adjudicated answer"):
        S.owner_units(got, units, {})
