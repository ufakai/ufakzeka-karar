"""The AI-assisted desk: proposals, AI answers, the blind control and new disputes."""

from bench.hakembench import assist as A


def unit(uid, proposed="true", track="spam", order=0):
    return {"id": uid, "track": track, "order": order, "text": "metin",
            "questions": [{"qid": "q", "question": "?", "type": "noul", "options": [],
                           "proposed": proposed}]}  # fmt: skip


def said(answer, flag="none"):
    return {"q": {"answer": answer, "flag": flag, "confidence": "high", "reason_tr": "çünkü",
                  "sources": []}}  # fmt: skip


def test_catches_show_their_truth_and_shifted_ones_keep_the_agreed_answer():
    pool = [unit("c1", "false"), unit("s1", "2")]
    catches = [
        {"unit": "c1", "qid": "q", "truth": "true", "shown": "false"},
        {"unit": "s1", "qid": "q", "truth": "2", "shown": "3", "kind": "shifted_level"},
    ]
    shown = A.proposals(pool, catches)
    assert shown == {("c1", "q"): "true", ("s1", "q"): "2"}


def test_the_desk_carries_the_panel_proposal_and_the_ai_answer():
    owner = [unit("d1", "false", order=0)]  # the adjudicated answer was on the desk
    proposal = {("d1", "q"): "true"}
    new, _, _ = A.assist(owner, [unit("d1")], proposal, {"d1": said("false")}, {"d1": "disputed"})
    q = new[0]["questions"][0]
    assert q["proposed"] == "true" and q["ai"]["answer"] == "false"


def test_blind_questions_stay_blind_and_a_quarter_of_the_sample_becomes_control():
    owner = [unit("b1", None, order=0)] + [unit(f"s{i}", order=i + 1) for i in range(8)]
    routed = {"b1": "blind"} | {f"s{i}": "sampled" for i in range(8)}
    proposal = {(u["id"], "q"): "true" for u in owner}
    research = {u["id"]: said("true") for u in owner}
    new, _, control = A.assist(owner, owner, proposal, research, routed)
    assert len(control) == 2
    blind = [u for u in new if u["questions"][0]["proposed"] is None]
    assert len(blind) == 3 and all("ai" not in u["questions"][0] for u in blind)


def test_an_unshown_unit_joins_when_the_research_answer_differs():
    owner = [unit("d1", order=0)]
    pool = [unit("d1"), unit("a1"), unit("a2")]
    proposal = {(u["id"], "q"): "true" for u in pool}
    research = {"d1": said("true"), "a1": said("false"), "a2": said("true")}
    new, routed, _ = A.assist(owner, pool, proposal, research, {"d1": "disputed"})
    assert routed["a1"] == "research_disputed" and "a2" not in routed
    assert [u["id"] for u in new] == ["d1", "a1"]


def test_a_flagged_research_answer_is_not_shown():
    assert A.ai_of({"x": said("true", "cant_tell")}, "x", "q") is None


def test_settled_questions_leave_the_desk_but_sample_and_blind_ones_stay():
    both = unit("d1")
    both["questions"].append({"qid": "r", "question": "?", "type": "noul", "options": [],
                              "proposed": "true"})  # fmt: skip
    desk = [both, unit("s1", order=1), unit("x1", order=2), unit("b1", None, order=3)]
    routed = {"d1": "disputed", "s1": "sampled", "x1": "shifted", "b1": "blind"}
    settled = {("d1", "r"), ("s1", "q"), ("x1", "q")}
    out, left = A.prune(desk, routed, settled)
    assert [(u["id"], [q["qid"] for q in u["questions"]]) for u in out] == [
        ("d1", ["q"]), ("s1", ["q"]), ("b1", ["q"])]  # fmt: skip
    assert "x1" not in left and left["s1"] == "sampled"


def test_a_question_is_settled_only_when_every_source_agrees():
    detail = {
        "u": {
            "questions": {
                "q": {"answers": [{"answer": "a"}, {"answer": "a"}]},
                "r": {"answers": [{"answer": "a"}, {"answer": "b"}]},
            }
        }
    }
    proposal = {("u", "q"): "a", ("u", "r"): "a"}
    research = {"u": {"q": {"answer": "a"}, "r": {"answer": "a"}}}
    assert A.settled_pairs(detail, proposal, research) == {("u", "q")}
