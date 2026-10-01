"""The check command: per-row flags, the clean file and the 2 percent set-level rule."""

import json
import random

from data.decontam import cli
from data.decontam.ngrams import MinHashLSH, NgramIndex, minhash_signature
from data.decontam.tests.fixtures import PARAGRAPH, UNRELATED, make_row, turkish_upper

WORDS = [
    "ev", "okul", "yol", "deniz", "dağ", "orman", "nehir", "köprü", "şehir", "köy", "tarla",
    "bahçe", "kitap", "kalem", "masa", "kapı", "pencere", "çatı", "duvar", "sokak", "meydan",
    "cami", "çarşı", "pazar", "liman", "gemi", "tren", "istasyon", "otobüs", "uçak", "bulut",
    "yağmur", "güneş", "rüzgâr", "kar", "buz", "ateş", "toprak", "taş", "kum",
]  # fmt: skip


def sentences(seed: int, count: int, length: int = 12) -> list[str]:
    """Shuffled word lists: two of them share an 8-gram with negligible chance."""
    rng = random.Random(seed)
    return [" ".join(rng.sample(WORDS, length)) for _ in range(count)]


ALFA = sentences(1, 10)
BETA = sentences(2, 59, length=20)
FILLER = " ".join(sentences(3, 1, 40))
OTHER_FILLER = " ".join(sentences(4, 1, 40))


def write_indexes(tmp_path):
    items = [(f"alfa:{i}", text) for i, text in enumerate(ALFA)]
    items += [(f"beta:{i}", text) for i, text in enumerate(BETA)]
    items.append(("beta:paragraph", PARAGRAPH))
    index = NgramIndex.build(items)
    lsh = MinHashLSH()
    for ref_id, text in items:
        lsh.add(ref_id, minhash_signature(text))
    index_path = tmp_path / "index.pkl"
    index.save(index_path)
    lsh.save(cli.lsh_path_for(index_path))
    return index_path


def write_rows(path, states):
    rows = [make_row(state) for state in states]
    path.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
    return rows


def run_check(tmp_path, states, *extra):
    index_path = write_indexes(tmp_path)
    rows_path = tmp_path / "rows.jsonl"
    rows = write_rows(rows_path, states)
    out = tmp_path / "report.json"
    code = cli.main(
        ["check", "--rows", str(rows_path), "--index", str(index_path), "--out", str(out), *extra]
    )
    return code, json.loads(out.read_text(encoding="utf-8")), rows, rows_path


def test_clean_rows_pass(tmp_path):
    code, report, _, rows_path = run_check(tmp_path, [UNRELATED, "merhaba", FILLER])
    assert code == 0
    assert report["flagged"] == 0 and report["contaminated_sets"] == []
    assert report["sets"]["alfa"] == {
        "items": 10,
        "overlapped": 0,
        "share": 0.0,
        "over_limit": False,
    }
    assert report["sets"]["beta"]["items"] == 60


def test_row_flags_and_the_clean_file(tmp_path):
    copied = turkish_upper(ALFA[0]) + "."
    embedded = FILLER + " " + ALFA[1] + " " + OTHER_FILLER
    edited = PARAGRAPH.replace("simit", "poğaça")
    states = [copied, embedded, edited, UNRELATED]
    code, report, rows, rows_path = run_check(tmp_path, states, "--write-clean")

    by_id = {record["row_id"]: record for record in report["per_row"]}
    first, second, third, fourth = (by_id[row.row_id] for row in rows)

    assert first["reasons"] == ["ngram", "covers_reference", "minhash"]
    assert first["ngram_fraction"] == 1.0 and first["matched_refs"] == ["alfa:0"]

    # A whole test item inside a long row: the row's own share is low, the item's is not.
    assert second["ngram_fraction"] < 0.5
    assert second["reasons"] == ["covers_reference"]
    assert second["covers_refs"] == ["alfa:1"]

    assert "minhash" in third["reasons"]
    assert third["minhash"][0][0] == "beta:paragraph" and third["minhash"][0][1] >= 0.8

    assert not fourth["flagged"]

    clean = (tmp_path / "rows.clean.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["row_id"] for line in clean] == [rows[3].row_id]
    assert report["clean_rows"] == 1 and report["flagged"] == 3

    # Two of ten alfa items overlapped is 20 percent, over the 2 percent limit.
    assert report["sets"]["alfa"]["overlapped"] == 2
    assert report["contaminated_sets"] == ["alfa"]
    assert code == 1


def test_set_level_rule_sits_at_two_percent(tmp_path):
    # One beta item of sixty is 1.7 percent: flagged row, but the set passes.
    code, report, _, _ = run_check(tmp_path, [BETA[5], UNRELATED])
    assert report["flagged"] == 1
    assert report["sets"]["beta"]["overlapped"] == 1
    assert report["sets"]["beta"]["over_limit"] is False
    assert code == 0


def test_set_level_rule_triggers_a_non_zero_exit(tmp_path):
    # Two of sixty is 3.3 percent. Each item is split over two rows, tokens 0
    # to 9 in one and 9 to 18 in the other, so no row covers more than half
    # of any item and none is flagged, yet together they cover 19 of its 20
    # tokens: the set-level rule still fails the file.
    states = []
    for item in (BETA[7], BETA[8]):
        words = item.split()
        states.append(FILLER + " " + " ".join(words[:10]))
        states.append(" ".join(words[9:19]) + " " + UNRELATED)
    code, report, _, _ = run_check(tmp_path, states)
    assert report["flagged"] == 0
    assert report["sets"]["beta"]["overlapped"] == 2
    assert report["contaminated_sets"] == ["beta"]
    assert code == 1


def test_check_refuses_without_the_minhash_index(tmp_path):
    index_path = write_indexes(tmp_path)
    cli.lsh_path_for(index_path).unlink()
    rows_path = tmp_path / "rows.jsonl"
    write_rows(rows_path, [UNRELATED])
    out = tmp_path / "report.json"
    argv = ["check", "--rows", str(rows_path), "--index", str(index_path), "--out", str(out)]
    assert cli.main(argv) == 2
    assert not out.exists()
