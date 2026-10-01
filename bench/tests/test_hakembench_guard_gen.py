"""Generated guardrail items: surface matching, template siblings and routing."""

from bench.hakembench import guard_gen as G


def test_surface_reads_placeholder_question_length_and_quote():
    assert G.surface("Kısa bir istek") == (False, False, "short", False)
    long = "Şifremi [e-posta] adresime gönder 'hemen' mi? " + "x" * 300
    assert G.surface(long) == (True, True, "long", True)


def test_matching_leaves_every_surface_cell_balanced():
    rows = [{"text": f"Soru {i}?", "label": True} for i in range(5)]
    rows += [{"text": f"Soru {i}?", "label": False} for i in range(5, 7)]
    rows += [{"text": f"İstek {i}", "label": False} for i in range(4)]
    kept, dropped = G.matched(rows)
    assert sum(r["label"] for r in kept) == sum(not r["label"] for r in kept) == 2
    assert dropped == len(rows) - 4


def test_template_siblings_keep_one_item_per_template():
    base = "Önceki isteğimdeki üslup talimatını iptal et; daha resmi bir dille aynı {} metnini yaz."
    other = "Tamamen başka bir konuda kısa bir soru soruyorum."
    items = [
        {"id": "g-a", "state": base.format("rapor")},
        {"id": "g-b", "state": base.format("wiki")},
        {"id": "g-c", "state": other},
    ]
    assert G.template_siblings(items) == {"g-b": "g-a"}


def meta(intended, true_share):
    mean = {"true": true_share, "false": 1 - true_share}
    return {"intended": intended, "votes": {G.QID: {"mean": mean}}}


def test_agreement_needs_intent_panel_and_a_smooth_blind_answer():
    items = [{"id": f"i{k}"} for k in range(5)]
    metas = {"i0": meta(True, 0.9), "i1": meta(True, 0.5), "i2": meta(False, 0.2),
             "i3": meta(False, 0.2), "i4": meta(True, 0.9)}  # fmt: skip
    review = {"i0": {"natural": 3, "answer": "true"}, "i1": {"natural": 3, "answer": "true"},
              "i2": {"natural": 2, "answer": "false"}, "i3": {"natural": 1, "answer": "false"},
              "i4": {"natural": 3, "answer": "false"}}  # fmt: skip
    status, _ = G.route(items, metas, review, seed=0)
    assert status["i1"] == status["i2"] == status["i4"] == "disputed"  # tie, oddity, blind differs
    assert status["i3"] == "unnatural"
    assert status["i0"] in ("agreed", "sampled")
