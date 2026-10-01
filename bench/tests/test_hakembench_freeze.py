"""The HakemBench freeze: typed gold, drops, the halves, the refusals, the training check."""

import json

import pytest

from bench.hakembench import freeze
from bench.harness.items import load_items

TEXT = "Kargom üç gündür gelmedi, takip numarası da çalışmıyor. Siparişim nerede, iade olur mu?"
NOUL = {"type": "noul", "instructions": "Şikâyet mi?"}
SCORE = {"type": "score", "instructions": "Kızgın mı?", "criteria": ["sakin", "kızgın", "çok"]}
CHOICE = {"type": "choice", "instructions": "Hangi birim?",
          "criteria": {"kargo": None, "iade": None}}  # fmt: skip


def checked(uid, track="spam", text=TEXT, gold=None, flags=None, questions=None):
    questions = questions or {"n": NOUL, "s": SCORE, "c": CHOICE}
    gold = {"n": "true", "s": "2", "c": "iade"} if gold is None else gold
    return {"id": uid, "track": track, "state": f"{uid}: {text}", "questions": questions,
            "gold": gold, "flags": flags or {},
            "gold_source": dict.fromkeys(gold, "owner")}  # fmt: skip


def meta(uid, half, licence="CC BY 4.0, authored by ufak AI", key="split"):
    return {"id": uid, key: half, "licence": licence, "source": "authored"}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Every path the freeze reads or writes under tmp_path, and one clean training file."""
    for name, rel in [("CHECKED", "checked"), ("CATCH_REPORT", "catch_report.json"),
                      ("CANDIDATES", "candidates"), ("PUBLIC_DIR", "public"),
                      ("PRIVATE_DIR", "private"), ("MANIFEST_DIR", "manifest"),
                      ("REPORT", "decontam/hakembench-v1.json"),
                      ("LEDGER", "decontam/flagged_ids.txt"), ("AUDIT", "release_audit.json"),
                      ("CHECKS", "checks"), ("BROWSER_REMOVALS", "removals"),
                      ("HMAC_KEY", "private/key"),
                      ("REMOVED", "private/removed.json")]:  # fmt: skip
        monkeypatch.setattr(freeze, name, tmp_path / rel)
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"row_id": "t0", "state": "Hiç ilgisi olmayan bir eğitim metni."})
                     + "\n", encoding="utf-8")  # fmt: skip
    monkeypatch.setattr(freeze, "train_files", lambda data: [train])
    (tmp_path / "checked").mkdir()
    (tmp_path / "candidates").mkdir()
    return tmp_path


def write(repo, items, metas, pending=0, blocking=()):
    # In the order the real run makes them: candidates, then desk collect, then the audit.
    lines = "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in metas)
    (repo / "candidates/spam.meta.jsonl").write_text(lines, encoding="utf-8")
    lines = "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items)
    (repo / "checked/spam.jsonl").write_text(lines, encoding="utf-8")
    (repo / "catch_report.json").write_text(json.dumps({"owner_pending": pending}))
    (repo / "release_audit.json").write_text(json.dumps({"blocking": list(blocking)}))


def test_desk_keys_become_typed_gold():
    kept, dropped = freeze.convert([checked("a")])
    assert kept[0]["item"]["gold"] == {"n": True, "s": 2, "c": "iade"}
    assert kept[0]["gold_source"] == {"n": "owner", "s": "owner", "c": "owner"}
    assert freeze.gold_value(NOUL, "false") is False
    assert dropped["questions"] == {} and dropped["items"] == {}
    with pytest.raises(freeze.FreezeError):
        freeze.gold_value(NOUL, "evet")
    with pytest.raises(freeze.FreezeError):  # a level past the rubric fails the item's check
        freeze.convert([checked("b", gold={"n": "true", "s": "5", "c": "iade"})])


def test_a_flagged_question_is_dropped_and_an_item_with_none_left_too():
    items = [checked("a", gold={"n": "true"}, flags={"s": "cant_tell"}),
             checked("b", gold={}, flags={"n": "no_fit", "s": "no_fit"})]  # fmt: skip
    kept, dropped = freeze.convert(items)
    assert [k["item"]["id"] for k in kept] == ["a"]
    assert list(kept[0]["item"]["questions"]) == ["n"]
    assert dropped["questions"] == {"flagged cant_tell": 1, "flagged no_fit": 2, "no gold": 2}
    assert dropped["items"] == {"no question left": 1}


def test_the_halves_go_to_their_files(repo):
    write(repo, [checked("b"), checked("a", track="arama"), checked("c")],
          [meta("a", "public", key="half"), meta("b", "private"), meta("c", "public")])  # fmt: skip
    assert freeze.main([]) == 0
    public = load_items(repo / "public/v1.0/public.jsonl")
    assert [i.id for i in public] == ["a", "c"]  # track, then id
    assert [i.id for i in load_items(repo / "private/v1.0/private.jsonl")] == ["b"]
    lines = (repo / "public/v1.0/provenance.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["gold_source"] == {"n": "owner", "s": "owner", "c": "owner"}
    manifest = json.loads((repo / "manifest/hakembench_v1.0.json").read_text())
    assert manifest["version"] == "1.0" and manifest["draft"] is False
    assert manifest["files"]["private"]["items"] == 1
    assert manifest["per_half"]["public"]["questions"] == {"arama": 3, "spam": 3}
    assert manifest["questions_per_type"] == {"choice": 3, "noul": 3, "score": 3}
    assert manifest["gold_source"]["all"] == {"owner": 9}
    assert len(manifest["text_sha256"]) == 2 and len(manifest["private_text_hmac_sha256"]) == 1
    assert manifest["training_rows_removed"] == 0 and manifest["training_rows_matching"] == 0


def test_a_private_only_licence_in_the_public_half_is_refused(repo, capsys):
    write(repo, [checked("a")], [meta("a", "public", licence="private only: clips (CC0-1.0)")])
    assert freeze.main([]) == 1
    assert "private only" in capsys.readouterr().err
    assert not (repo / "public").exists()


def test_an_item_without_a_half_is_refused(repo, capsys):
    write(repo, [checked("a")], [{"id": "a", "licence": "CC-BY-4.0"}])
    assert freeze.main([]) == 1
    assert "no public or private half" in capsys.readouterr().err


def test_pending_owner_checks_refuse_the_freeze_or_write_a_draft(repo, capsys):
    write(repo, [checked("a"), checked("b")], [meta("a", "public"), meta("b", "private")], 3)
    assert freeze.main([]) == 1
    assert "3 units" in capsys.readouterr().err
    assert freeze.main(["--allow-pending"]) == 0
    assert not (repo / "public/v1.0").exists()
    assert (repo / "public/v1.0-draft/public.jsonl").is_file()
    assert (repo / "private/v1.0-draft/private.jsonl").is_file()
    manifest = json.loads((repo / "manifest/hakembench_v1.0-draft.json").read_text())
    assert manifest["draft"] is True and manifest["version"] == "1.0-draft"


def test_a_training_row_carrying_a_test_text_is_removed(repo, monkeypatch):
    write(repo, [checked("a"), checked("b")], [meta("a", "public"), meta("b", "private")])
    train = repo / "carrying.jsonl"
    rows = [{"row_id": "keep", "state": "Bambaşka bir konu, hiçbir ortak cümle yok burada."},
            {"row_id": "drop", "state": f"b: {TEXT}"}]  # fmt: skip
    train.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    monkeypatch.setattr(freeze, "train_files", lambda data: [train])
    assert freeze.main([]) == 0
    assert [json.loads(x)["row_id"] for x in train.read_text().splitlines()] == ["keep"]
    report = json.loads((repo / "decontam/hakembench-v1.json").read_text())
    assert report["reference"] == "hakembench-v1.0"
    assert [(r["row_id"], r["rule"], r["refs"]) for r in report["per_row"]] == [
        ("drop", "whole", ["b"])
    ]
    assert (repo / "decontam/flagged_ids.txt").read_text() == "drop\n"
    manifest = json.loads((repo / "manifest/hakembench_v1.0.json").read_text())
    assert manifest["training_rows_removed"] == 1


def test_a_match_left_after_removal_refuses_the_freeze(repo, monkeypatch, capsys):
    write(repo, [checked("a")], [meta("a", "public")])
    monkeypatch.setattr(freeze, "training_overlap", lambda rows, files: [
        {"file": str(files[0]), "row_id": "t0", "rule": "whole", "refs": ["a"]}])  # fmt: skip
    assert freeze.main([]) == 1
    assert "still carry" in capsys.readouterr().err
    assert not (repo / "public").exists()


def test_web_faq_passages_are_always_private():
    from bench.hakembench.freeze import split

    kept = [{"item": {"id": "arama-1", "track": "arama"}, "gold_source": {}}]
    metas = {
        "arama-1": {
            "half": "public",
            "licence": "CC-BY-4.0",
            "source": "https://huggingface.co/datasets/PaDaS-Lab/webfaq-retrieval",
        }
    }
    halves = split(kept, metas)
    assert [e["item"]["id"] for e in halves["private"]] == ["arama-1"] and not halves["public"]


def test_the_freeze_waits_for_a_fresh_collect_and_a_clean_audit(repo):
    write(repo, [checked("a")], [meta("a", "public")], blocking=["leak x"])
    assert freeze.main([]) == 1
    write(repo, [checked("a")], [meta("a", "public")])
    later = (repo / "checked/spam.jsonl").stat().st_mtime + 10
    removal = repo / "removals/a.json"
    removal.parent.mkdir()
    removal.write_text(json.dumps({"item": "a"}))
    import os

    os.utime(removal, (later, later))
    assert freeze.main([]) == 1
    removal.unlink()
    assert freeze.main([]) == 0


def test_public_items_carry_the_canary_and_private_texts_a_keyed_hash(repo):
    write(repo, [checked("a"), checked("b")], [meta("a", "public"), meta("b", "private")])
    assert freeze.main([]) == 0
    public = [
        json.loads(line) for line in (repo / "public/v1.0/public.jsonl").read_text().splitlines()
    ]
    private = [
        json.loads(line) for line in (repo / "private/v1.0/private.jsonl").read_text().splitlines()
    ]
    assert public[0]["canary"] == freeze.CANARY and "canary" not in private[0]
    manifest = json.loads((repo / "manifest/hakembench_v1.0.json").read_text())
    plain = freeze.text_hash(private[0]["state"])
    assert (
        plain not in manifest["text_sha256"] and plain not in manifest["private_text_hmac_sha256"]
    )
    assert len(manifest["private_text_hmac_sha256"]) == 1
