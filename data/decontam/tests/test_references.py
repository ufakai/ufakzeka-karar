"""The reference registry, fetch with fake transports, and the cache readers."""

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from data.decontam import references
from data.decontam.references import REFERENCE_SETS, RefSpec
from data.decontam.tests.fixtures import PARAGRAPH

INSTRUMENT_SETS = ["massive_tr", "offenseval_tr", "trcola", "mide22", "legal_nli_tr"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to reach the network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def test_registry_is_well_formed():
    for name, spec in REFERENCE_SETS.items():
        assert ":" not in name
        assert spec.fields, name
        if spec.source == "hub":
            assert spec.config, name
            assert re.fullmatch(r"[0-9a-f]{40}", spec.revision or ""), name
        if spec.source == "github_squad":
            # Pinned to a commit, not a branch.
            assert re.search(r"/[0-9a-f]{40}/", spec.repo), name
    local = {name for name, spec in REFERENCE_SETS.items() if spec.source == "local"}
    assert local == {f"{name}_test" for name in INSTRUMENT_SETS}
    for name in INSTRUMENT_SETS:
        spec = REFERENCE_SETS[f"{name}_test"]
        assert spec.repo == f"data/raw/instrument/{name}/test.jsonl"
        assert spec.fields == ("text", "text_pair")


@pytest.mark.skipif(not Path("data/raw").is_dir(), reason="data/raw is not present")
def test_instrument_entries_point_at_local_files():
    for name in INSTRUMENT_SETS:
        path = Path(REFERENCE_SETS[f"{name}_test"].repo)
        assert path.is_file(), path
        with path.open(encoding="utf-8") as handle:
            first = json.loads(handle.readline())
        assert {"id", "text", "text_pair"} <= set(first), path


def test_text_joins_non_empty_string_fields():
    row = {"a": " bir ", "b": None, "c": "", "d": ["liste"], "e": "iki"}
    assert references.text_from(row, ("a", "b", "c", "d", "e", "missing")) == "bir\niki"


def test_text_reads_nested_fields_and_joins_lists_only_when_asked():
    row = {
        "question": {"stem": "Başkent?", "choices": {"text": ["Ankara", " ", "İzmir", 3]}},
        "translation": {"tr": "merhaba", "en": "hello"},
        "tokens": ["Club", "Brugge'de", "oynuyor", "."],
    }
    fields = ("question.stem", "question.choices.text[]", "translation.tr", "missing.key")
    assert references.text_from(row, fields) == "Başkent?\nAnkara İzmir\nmerhaba"
    assert references.text_from(row, ("tokens[]",)) == "Club Brugge'de oynuyor ."
    assert references.text_from(row, ("tokens", "translation", "question.stem[]")) == ""


def fake_hub(total: int, sha: str = "f" * 40):
    import pyarrow as pa
    import pyarrow.parquet as pq

    calls = []
    rows = [{"premise": f"öncül {i}", "hypothesis": "" if i % 2 else "h"} for i in range(total)]
    halves = [rows[: total // 2], rows[total // 2 :]]

    def get_json(url: str):
        calls.append(url)
        if "/tree/" in url:
            return [{"path": f"tr/test/{i:04d}.parquet", "size": -1} for i in range(2)]
        return {"sha": sha}

    def download(url: str, target):
        calls.append(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(halves[int(url[-12:-8])]), target)

    return get_json, download, calls


def test_fetch_reads_the_parquet_export_through_the_resolver(tmp_path, monkeypatch):
    spec = RefSpec(
        benchmark="deneme",
        source="hub",
        repo="org/set",
        config="tr",
        split="test",
        fields=("premise", "hypothesis"),
    )
    monkeypatch.setitem(REFERENCE_SETS, "fake_hub", spec)
    get_json, download, calls = fake_hub(150)
    path = references.fetch("fake_hub", tmp_path, get_json=get_json, download=download)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 150
    assert lines[0] == {"id": "0", "text": "öncül 0\nh"}
    assert lines[1] == {"id": "1", "text": "öncül 1"}
    assert lines[75]["id"] == "75"  # indices run on across files
    assert sum("/resolve/refs%2Fconvert%2Fparquet/tr/test/" in url for url in calls) == 2
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["items"] == 150 and meta["revision_read"] == "f" * 40


def test_fetch_reads_a_squad_file_and_keeps_ids_unique(tmp_path, monkeypatch):
    spec = RefSpec(
        benchmark="deneme",
        source="github_squad",
        repo="https://example.invalid/dev.json",
        split="validation",
        fields=("context", "question"),
        id_field="id",
    )
    monkeypatch.setitem(REFERENCE_SETS, "fake_squad", spec)
    document = {
        "data": [
            {
                "paragraphs": [
                    {
                        "context": "Ankara Türkiye'nin başkentidir.",
                        "qas": [
                            {"id": "q1", "question": "Başkent neresi?"},
                            {"id": "q1", "question": "Hangi ülke?"},
                        ],
                    }
                ]
            }
        ]
    }
    path = references.fetch("fake_squad", tmp_path, get_json=lambda url: document)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["id"] for line in lines] == ["q1", "q1#1"]
    assert lines[0]["text"] == "Ankara Türkiye'nin başkentidir.\nBaşkent neresi?"


def test_local_fetch_load_and_build(tmp_path, monkeypatch):
    source = tmp_path / "test.jsonl"
    rows = [
        {"id": "a", "text": PARAGRAPH, "text_pair": None},
        {"id": "b", "text": "sessiz", "text_pair": "ol"},
        {"id": "c", "text": "", "text_pair": None},
    ]
    source.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    spec = RefSpec(
        benchmark="deneme",
        source="local",
        repo=str(source),
        split="test",
        fields=("text", "text_pair"),
        id_field="id",
    )
    monkeypatch.setattr(references, "REFERENCE_SETS", {"yerel": spec})
    cache = tmp_path / "cache"
    assert references.missing(cache) == ["yerel"]
    references.fetch("yerel", cache)
    assert references.missing(cache) == []

    items = references.load_all(cache)
    assert items == [("yerel:a", PARAGRAPH), ("yerel:b", "sessiz\nol")]

    index = references.build_index(cache)
    assert index.overlap(PARAGRAPH) == (1.0, ["yerel:a"])
    assert index.overlap("Sessiz ol!") == (1.0, ["yerel:b"])

    lsh = references.build_lsh(cache)
    assert len(lsh) == 1  # "sessiz ol" is under five tokens and has no shingle
