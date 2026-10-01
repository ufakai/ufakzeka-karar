"""The check desk loader: spot checks, catch items kept off the page, and gold from checks."""

import json

import pytest

from bench.hakembench.desk import candidate_items, collect, content_removals, units


def item(i, track="arama", gold=True):
    q = {
        "type": "noul",
        "instructions": "İlgili mi?",
        "criteria": {"true": "evet", "false": "hayır"},
    }
    return {"id": f"{track}-{i}", "track": track, "state": f"metin {i}", "questions": {"q": q},
            "gold": {"q": gold} if gold is not None else {}}  # fmt: skip


def test_spot_tracks_send_a_fifth_and_catches_stay_off_the_page():
    items = [item(i) for i in range(100)] + [item(i, "spam", None) for i in range(20)]
    metas = {f"spam-{i}": {"votes": {"q": {"mean": {"true": 0.8, "false": 0.2}}}}
             for i in range(20)}  # fmt: skip
    metas |= {f"arama-{i}": {"source": "https://huggingface.co/datasets/PaDaS-Lab/webfaq-retrieval"}
              for i in range(100)}  # fmt: skip
    desk, catches = units(items, metas)
    assert sum(u["track"] == "arama" for u in desk) == 20
    assert sum(u["track"] == "spam" for u in desk) == 20
    assert len(catches) == round(0.05 * len(desk))
    assert all("known" not in q for u in desk for q in u["questions"])
    shown = {(u["id"], q["qid"]): q["proposed"] for u in desk for q in u["questions"]}
    for c in catches:
        assert c["unit"].startswith("arama") and shown[(c["unit"], c["qid"])] == c["shown"]
        assert c["shown"] != c["truth"]


def test_gold_comes_from_checks_and_catch_items_keep_their_truth():
    items = [item(0), item(1), item(2)]
    catches = [{"unit": "arama-2", "qid": "q", "truth": "true", "shown": "false"}]
    checks = {
        "arama-0~q": {"verdict": "agree", "proposed": "true", "answer": "true", "flag": "none"},
        "arama-1~q": {"verdict": "change", "proposed": "true", "answer": "false", "flag": "none"},
        "arama-2~q": {"verdict": "agree", "proposed": "false", "answer": "false", "flag": "none"},
    }
    checked, report = collect(items, checks, catches)
    gold = {c["id"]: c["gold"]["q"] for c in checked}
    assert gold == {"arama-0": "true", "arama-1": "false", "arama-2": "true"}
    assert (report["catch_items"], report["caught"], report["missed"]) == (1, 0, 1)
    assert report["catch_rate"] == 0.0 and report["spot_checks"] == {}


def test_pooled_catches_come_from_unused_known_items_and_are_spread():
    from bench.hakembench.desk import pooled_catches

    desk = [{"id": f"spam-{i}", "track": "spam", "order": i, "text": "t",
             "questions": [{"qid": "q", "question": "?", "type": "noul", "options": [
                 {"key": "true", "label": "Evet", "detail": ""},
                 {"key": "false", "label": "Hayır", "detail": ""}], "proposed": "true"}]}
            for i in range(100)]  # fmt: skip
    pool = [item(i) for i in range(50)]
    merged, catches = pooled_catches(desk, pool, loaded={"arama-0", "arama-1"})
    assert len(catches) == 5 and len(merged) == 105
    assert not {c["unit"] for c in catches} & {"arama-0", "arama-1"}
    positions = [n for n, u in enumerate(merged) if u["track"] == "arama"]
    assert positions != list(range(100, 105))  # spread, not appended


def test_unshown_agreed_questions_become_gold_marked_as_such():
    checked, _ = collect([item(0), item(1)], {}, [], agreed={("arama-0", "q"): "true"})
    assert [(c["id"], c["gold"], c["gold_source"]) for c in checked] == [
        ("arama-0", {"q": "true"}, {"q": "ai_reviewer_and_panel"})
    ]


def test_an_unshown_catch_takes_its_truth_not_the_wrong_answer_it_showed():
    catches = [{"unit": "arama-0", "qid": "q", "truth": "true", "shown": "false"}]
    checked, report = collect([item(0)], {}, catches, agreed={("arama-0", "q"): "false"})
    assert checked[0]["gold"] == {"q": "true"}
    assert checked[0]["gold_source"] == {"q": "known_truth"}
    assert report["caught"] == report["missed"] == 0


def spot_check(differ):
    items = [item(i) for i in range(30)]
    checks = {f"arama-{i}~q": {"item": f"arama-{i}", "qid": "q", "proposed": "true",
                               "verdict": "change" if i < differ else "agree",
                               "answer": "false" if i < differ else None, "flag": "none"}
              for i in range(10)}  # fmt: skip
    spot = {"webfaq": {"on_desk": {f"arama-{i}" for i in range(10)},
                       "rest": {f"arama-{i}" for i in range(10, 30)}}}  # fmt: skip
    return collect(items, checks, [], spot=spot)


def test_a_clean_spot_check_keeps_the_rest_with_the_source_label():
    checked, report = spot_check(differ=1)
    assert len(checked) == 30
    rest = [c for c in checked if int(c["id"].split("-")[1]) >= 10]
    assert all(c["gold"] == {"q": "true"} and c["gold_source"] == {"q": "source_label"}
               for c in rest)  # fmt: skip
    assert report["spot_checks"]["webfaq"]["rate"] == 0.1
    assert report["spot_checks"]["webfaq"]["rest_accepted"] is True


def test_a_spot_check_above_the_limit_leaves_the_rest_out():
    checked, report = spot_check(differ=2)
    assert len(checked) == 10
    assert report["spot_checks"]["webfaq"] == {"compared": 10, "differ": 2, "rate": 0.2,
                                               "rest": 20, "rest_accepted": False}  # fmt: skip


def check(i, verdict, answer=None, proposed="true", flag="none"):
    return {f"arama-{i}~q": {"item": f"arama-{i}", "qid": "q", "proposed": proposed,
                             "verdict": verdict, "answer": answer, "flag": flag}}  # fmt: skip


def test_a_suspect_catch_takes_the_owners_answer_and_leaves_the_rate():
    catches = [{"unit": "arama-0", "qid": "q", "truth": "true", "shown": "false", "suspect": True}]
    checked, report = collect([item(0)], check(0, "agree", proposed="false"), catches)
    assert checked[0]["gold"] == {"q": "false"} and checked[0]["gold_source"] == {
        "q": "owner_confirmed"
    }
    assert report["caught"] == report["missed"] == 0 and report["catch_items"] == 0
    assert report["suspect_catches"] == {"checked": 1}


def test_a_shifted_level_catch_is_scored_apart_and_keeps_the_agreed_answer_when_missed():
    catches = [{"unit": "arama-0", "qid": "q", "truth": "true", "shown": "false",
                "kind": "shifted_level"},
               {"unit": "arama-1", "qid": "q", "truth": "true", "shown": "false",
                "kind": "shifted_level"}]  # fmt: skip
    checks = check(0, "agree", proposed="false") | check(1, "change", answer="true",
                                                         proposed="false")  # fmt: skip
    checked, report = collect([item(0), item(1)], checks, catches)
    gold = {c["id"]: (c["gold"]["q"], c["gold_source"]["q"]) for c in checked}
    assert gold == {
        "arama-0": ("true", "ai_reviewer_and_panel"),
        "arama-1": ("true", "owner_corrected"),
    }
    assert report["shifted_level"] == {"items": 2, "caught": 1, "missed": 1}
    assert report["catch_items"] == 0


def test_a_blind_answer_is_marked_as_such():
    checked, _ = collect([item(0)], check(0, "label", answer="false", proposed=None), [])
    assert checked[0]["gold_source"] == {"q": "owner_blind"}


def test_an_agreed_pair_carries_its_own_gold_source():
    agreed = {("arama-0", "q"): ("false", "writer_intent_panel_agreed")}
    checked, _ = collect([item(0)], {}, [], agreed=agreed)
    assert checked[0]["gold"] == {"q": "false"}
    assert checked[0]["gold_source"] == {"q": "writer_intent_panel_agreed"}


def test_a_retired_catch_takes_the_owners_answer_and_is_scored_against_the_truth():
    catches = [
        {"unit": "arama-0", "qid": "q", "truth": "true", "shown": "false", "retired": True},
        {"unit": "arama-1", "qid": "q", "truth": "true", "shown": "false", "retired": True},
    ]
    checks = check(0, "agree", proposed="true") | check(1, "pick", answer="false")
    checks["arama-0~q"]["ai"] = "true"
    checks["arama-1~q"]["ai"] = "false"
    checked, report = collect([item(0), item(1)], checks, catches)
    gold = {c["id"]: (c["gold"]["q"], c["gold_source"]["q"]) for c in checked}
    assert gold == {"arama-0": ("true", "owner_confirmed"), "arama-1": ("false", "owner_chose")}
    assert report["owner_against_known_truth"] == {"ai_right:agrees": 1, "ai_wrong:differs": 1}
    assert report["caught"] == report["missed"] == 0


def test_a_confirmed_or_chosen_removal_drops_the_text():
    checks = {
        "a~__content__": {"item": "a", "qid": "__content__", "proposed": "false",
                          "verdict": "agree", "answer": "false", "flag": "none"},
        "b~__content__": {"item": "b", "qid": "__content__", "proposed": "false",
                          "verdict": "change", "answer": "true", "flag": "none"},
        "c~q": {"item": "c", "qid": "q", "proposed": "false", "verdict": "agree",
                "answer": "false", "flag": "none"},
    }  # fmt: skip
    assert content_removals(checks) == {"a"}


def test_candidate_items_refuses_an_id_in_two_files(tmp_path):
    row = json.dumps({"id": "moderasyon-1", "track": "moderasyon"}) + "\n"
    (tmp_path / "moderasyon-topup.jsonl").write_text(row)
    (tmp_path / "moderasyon-topup.meta.jsonl").write_text(row)
    assert [i["id"] for i in candidate_items(tmp_path)] == ["moderasyon-1"]
    (tmp_path / "moderasyon-pilot.jsonl").write_text(row)
    with pytest.raises(SystemExit, match="moderasyon-1"):
        candidate_items(tmp_path)
