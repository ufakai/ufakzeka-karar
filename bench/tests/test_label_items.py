"""Labels for vote-less candidates, and the flagship's balanced, party-keeping selection."""

from bench.hakembench.label_items import select, with_votes


def meta(i, claim, party):
    mean = {"true": 0.8, "false": 0.2} if claim else {"true": 0.2, "false": 0.8}
    return {"id": f"d{i}", "party": party, "votes": {"dogrulanabilir": {"mean": mean}}}


def test_selection_is_half_claims_and_keeps_the_party_mix():
    metas = [meta(i, i % 3 == 0, "A" if i % 4 else "B") for i in range(600)]
    chosen = select(metas, 100)
    by_id = {m["id"]: m for m in metas}
    claims = sum(by_id[i]["votes"]["dogrulanabilir"]["mean"]["true"] >= 0.5 for i in chosen)
    assert len(chosen) == 100 and claims == 50
    share_b = sum(by_id[i]["party"] == "B" for i in chosen) / 100
    assert abs(share_b - 0.25) < 0.05
    assert chosen == select(metas, 100)


def test_votes_are_written_per_question():
    items = [{"id": "d0", "track": "dogrulama", "state": "x",
              "questions": {"dogrulanabilir": {"type": "noul"}}}]  # fmt: skip
    records = {"dogrulanabilir:d0": {"outcome": "labelled", "row": {
        "judges": [{"judge": "A", "distribution": {"true": 1.0, "false": 0.0}}],
        "target": {"true": 1.0, "false": 0.0}}}}  # fmt: skip
    out = with_votes([{"id": "d0"}], items, records)
    assert out[0]["votes"]["dogrulanabilir"]["mean"] == {"true": 1.0, "false": 0.0}
