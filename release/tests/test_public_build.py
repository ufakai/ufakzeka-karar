"""The step 10 public build on a tiny fake repo."""

import gzip
import json
import shutil
import subprocess
from pathlib import Path

from bench.hakembench import release_audit as A
from bench.hakembench.freeze import CANARY, keyed
from release import public_build as B

IDENTITY = ("Test Owner", "owner@example.invalid")
PUBLIC_TEXT = "Kampanya kodunu girerek bu hafta tüm siparişlerde yüzde yirmi indirim kazanın."
PRIVATE_TEXT = (
    "Kargom üç gündür dağıtım merkezinde bekliyor, müşteri hizmetleri de "
    "telefonu açmıyor; ne yapmam gerekiyor acaba?"
)
OPENED_TEXT = (
    "Siparişim yarın teslim edilecek mi, kargo takip numarası henüz gelmedi ama ödeme "
    "hesabımdan çekildi görünüyor."
)
REMOVED_TEXT = (
    "Bu ileti kaldırılan bir metindir ve hiçbir yayımlanan dosyada görünmemesi "
    "gereken uzunlukta bir cümledir."
)
PUB, PRIV, GONE = "spam-0123456789abcdef", "spam-fedcba9876543210", "spam-00112233aabbccdd"
# A former private item the open release publishes; PRIV stays left out (restricted).
OPENED = "spam-aaaabbbbccccdddd"
KEY = "11" * 32
# Built at run time, so this file's own text holds no address or number the scan would flag.
MAIL = "ali" + "@" + "ornek.com"
PHONE = "0532 12" + "3 45 67"
FAKE_BENCH = ["bench/__init__.py", "bench/harness/items.py"]
GENERATED = "CC BY 4.0 where rights exist, otherwise CC0; authored by ufak AI for the benchmark"
# Training rows whose text is not ours to publish (data/built/, never copied into a tree).
WEB_TEXT = (
    "Soru: faturamı nasıl öderim?\nCevap: fatura ödemesini mobil uygulamadan, internet "
    "şubesinden ya da en yakın şubeden yapabilirsiniz, işlem ücretsizdir."
)
TWEET = "bu maçtan sonra hakemi kimse savunamaz"
OUR_TEXT = (
    "Bu metin bizim yazdığımız bir örnektir ve eğitim verisinde yer alsa da yayımlanabilir "
    "çünkü kaynağı bizim."
)


def item(uid: str, text: str, canary: bool = False) -> dict:
    return {
        **({"canary": CANARY} if canary else {}),
        "id": uid,
        "track": "spam",
        "state": text,
        "questions": {"spam": {"type": "noul", "instructions": "Spam mı?"}},
        "gold": {"spam": True},
    }


def jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def prov(uid: str, source: str = "generated", canary: bool = True) -> dict:
    licence = GENERATED if source == "generated" else "private only: " + source
    return {
        "id": uid,
        "track": "spam",
        "source": source,
        "licence": licence,
        "gold_source": {"spam": "owner"},
        **({"canary": CANARY} if canary else {}),
    }


def make_names() -> str:
    return json.dumps(
        {
            "terms": ["vendorx", "strictco"],
            "always_blocking": ["strictco"],
            "config_files": ["config/panel.json"],
            "contacts": ["hello@ufakai.com"],
            "allowed": [
                {"path": "model/", "terms": ["vendorx"], "category": "technical"},
                {
                    "path": "docs/tools.md",
                    "terms": ["acmeco"],
                    "line_contains": "acmeco-cli",
                    "category": "tool",
                },
            ],
        }
    )


def make_repo(
    root: Path, extra: dict | None = None, source: str = "generated", opened_canary: bool = True
) -> Path:
    files = {
        "README.md": "# fake\n\nThe private working repo.\n",
        "release/public_readme.md": "# fake\n\nThe public readme.\n",
        "SESSION.md": "instructions for the session\n",
        "secret/notes.md": "held back\n",
        "docs/PUBLIC_BUILD_EXCLUDE.txt": "# rules\nSESSION.md\nconfig/panel.json\n"
        "config/release_names.json\nresults/private/\nbench/hakembench/candidates/\nsecret/\n",
        "config/panel.json": json.dumps(
            {
                "judges": [{"model": "acmeco/acme-7b", "provider": "acme-cloud"}],
                "generator": {"model": "acmeco/acme-9b", "provider": "together"},
            }
        ),
        "config/release_names.json": make_names(),
        "data/MANIFEST.yaml": "entries:\n  - path: bench/hakembench/candidates/spam.jsonl\n"
        "    source: authored\n    attribution: ufak AI.\n",
        "bench/hakembench/v1.0/open/items.jsonl": jsonl(
            [item(PUB, PUBLIC_TEXT, True), item(OPENED, OPENED_TEXT, opened_canary)]
        ),
        "bench/hakembench/v1.0/open/provenance.jsonl": jsonl(
            [prov(PUB, source), prov(OPENED, canary=opened_canary)]
        ),
        "bench/hakembench/v1.0/open/splits.json": json.dumps({PUB: "public", OPENED: "private"}),
        "bench/hakembench/v1.0/open/probes/paraphrase.jsonl": jsonl(
            [item(PUB + "~paraphrase-1", PUBLIC_TEXT + " Hemen katılın.")]
        ),
        "bench/hakembench/candidates/spam.jsonl": jsonl(
            [
                item(PUB, PUBLIC_TEXT),
                item(PRIV, PRIVATE_TEXT),
                item(OPENED, OPENED_TEXT),
                item(GONE, REMOVED_TEXT),
            ]
        ),
        "bench/hakembench/candidates/spam.meta.jsonl": jsonl(
            [
                {"id": PUB, "half": "public", "source": source},
                {"id": PRIV, "half": "private", "source": "clips/mqa"},
                {"id": OPENED, "half": "private", "source": "generated"},
                {"id": GONE, "half": "public", "source": "generated"},
            ]
        ),
        "LICENSE": "Apache License, Version 2.0\n",
        "bench/__init__.py": "",
        "bench/harness/items.py": "# The harness's items, byte for byte.\n",
        "results/private/step8/v1.0/private.jsonl": jsonl(
            [item(PRIV, PRIVATE_TEXT), item(OPENED, OPENED_TEXT)]
        ),
        "results/private/step8/v1.0/provenance.jsonl": jsonl(
            [prov(PRIV, "clips/mqa", canary=False), prov(OPENED, canary=False)]
        ),
        "results/private/step8/text_hmac.key": KEY,
        "results/private/step8/checked/spam.jsonl": jsonl(
            [item(PUB, PUBLIC_TEXT), item(PRIV, PRIVATE_TEXT), item(OPENED, OPENED_TEXT)]
        ),
        "data/built/sss/train.jsonl": jsonl(
            [
                {"row_id": "r1", "source": "clips/mqa:tr-faq-question", "state": WEB_TEXT},
                {"row_id": "r2", "source": "generated:synth-v1", "state": OUR_TEXT},
                # The open set's own text in a training row never blocks.
                {"row_id": "r3", "source": "clips/mqa", "state": PUBLIC_TEXT},
            ]
        ),
        "data/built/typed/mide22/train.jsonl": jsonl(
            [{"row_id": "t1", "source": "data/raw/instrument/mide22/", "state": TWEET}]
        ),
        **(extra or {}),
    }
    # data/built/ is not tracked in the real repo either: written after the commit.
    built = {k: v for k, v in files.items() if k.startswith("data/built/")}
    for name, body in files.items():
        if name in built:
            continue
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(body, bytes):
            (root / name).write_bytes(body)
        else:
            (root / name).write_text(body, encoding="utf-8")
    env = [
        "-c",
        f"user.name={IDENTITY[0]}",
        "-c",
        f"user.email={IDENTITY[1]}",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "commit.gpgsign=false",
    ]
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    subprocess.run(["git", *env, "add", "-A", "-f"], cwd=root, check=True)
    subprocess.run(["git", *env, "commit", "-q", "-m", "fake"], cwd=root, check=True)
    for name, body in built.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(body, encoding="utf-8")
    return root


def run(tmp_path: Path, bench=FAKE_BENCH, **kw) -> tuple[dict, Path]:
    # The venv and test run need the network and minutes; check_bench is tested apart.
    (tmp_path / "repo").mkdir()
    out = tmp_path / "public-build"
    return B.build(make_repo(tmp_path / "repo", **kw), out, run_tests=False, bench_files=bench), out


def test_a_clean_repo_builds_two_one_commit_trees_without_excluded_paths(tmp_path):
    report, out = run(tmp_path)
    assert report["ok"], report["blocking_summary"]
    code = out / B.CODE_TREE
    assert (code / "docs/PUBLIC_BUILD_EXCLUDE.txt").exists()
    assert (code / ".gitattributes").read_text().endswith("* -text\n")
    # The public README replaces the private one and is not shipped a second time.
    assert (code / "README.md").read_text() == "# fake\n\nThe public readme.\n"
    assert not (code / "release/public_readme.md").exists()
    for gone in (
        "SESSION.md",
        "secret/notes.md",
        "config/panel.json",
        "results/private",
        "bench/hakembench/candidates",
        "config/release_names.json",
    ):
        assert not (code / gone).exists(), gone
    data = out / B.DATA_TREE
    assert sorted(
        p.relative_to(data).as_posix()
        for p in data.rglob("*")
        if p.is_file() and ".git" not in p.parts
    ) == [
        ".gitattributes",
        ".gitignore",
        "ATTRIBUTION.md",
        "LICENSE",
        "README.md",
        "bench/__init__.py",
        "bench/harness/items.py",
        "data/v1.0/items.jsonl",
        "data/v1.0/probes/paraphrase.jsonl",
        "data/v1.0/provenance.jsonl",
        "data/v1.0/splits.json",
        "pyproject.toml",
    ]
    # The open set ships in both trees; the items left out never do.
    assert (code / "bench/hakembench/v1.0/open/items.jsonl").is_file()
    assert report["trees"][B.DATA_TREE]["splits"] == {"public": 1, "private": 1}
    repo = tmp_path / "repo"
    assert (data / "bench/harness/items.py").read_bytes() == (
        repo / "bench/harness/items.py"
    ).read_bytes()
    assert 'name = "hakembench"' in (data / "pyproject.toml").read_text()
    assert report["trees"][B.DATA_TREE]["readme_placeholder"]
    for tree in (code, data):
        count = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=tree,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert count.strip() == "1"
    for tree in report["trees"].values():
        assert tree["git"]["author"] == "Test Owner <owner@example.invalid>"
        assert not tree["git"]["co_author_line"]
    attribution = (data / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert "- Source: Written by various LLMs to ufak AI's briefs" in attribution
    assert "- Credit: ufak AI, which releases these texts under the licence above" in attribution
    assert "- Licence: CC BY 4.0 where rights exist, otherwise CC0\n" in attribution
    assert "authored" not in attribution
    readme = (data / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# HakemBench v1.0\n\nPlaceholder:") and "(2 items" in readme
    assert json.loads((out / "scan_report.json").read_text())["ok"]
    # A rebuild wipes the old trees first.
    (out / B.CODE_TREE / "stale.txt").write_text("x")
    B.build(tmp_path / "repo", out, run_tests=False, bench_files=FAKE_BENCH)
    assert not (out / B.CODE_TREE / "stale.txt").exists()


def test_the_build_refuses_an_out_folder_it_would_wipe_with_other_work(tmp_path):
    (tmp_path / "repo").mkdir()
    repo = make_repo(tmp_path / "repo")
    other = tmp_path / "other"
    other.mkdir()
    (other / B.CODE_TREE).mkdir()
    (other / "notes.txt").write_text("keep")
    for out, why in [
        (tmp_path, "holds the repo"),
        (tmp_path.parent, "holds the repo"),
        (repo / "public-build", "inside the repo"),
        (other, "holds other files"),
    ]:
        try:
            B.build(repo, out, run_tests=False, bench_files=FAKE_BENCH)
        except B.BuildError as error:
            assert why in str(error)
        else:
            raise AssertionError(f"built into {out}")
    assert (other / "notes.txt").read_text() == "keep" and (repo / "README.md").exists()


def test_a_bench_file_the_public_build_would_not_keep_blocks_and_is_not_copied(tmp_path):
    report, out = run(tmp_path, bench=[*FAKE_BENCH, "secret/notes.md", "bench/nowhere.py"])
    missing = {f["file"] for f in report["blocking"]["bench_files_missing"]}
    assert missing == {"hakembench/secret/notes.md", "hakembench/bench/nowhere.py"}
    assert not (out / B.DATA_TREE / "secret").exists() and not report["ok"]


def test_failing_bench_tests_in_their_own_venv_fail_the_build(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "lock_bench", lambda tree: {"ok": True})
    monkeypatch.setattr(
        B, "check_bench", lambda tree: {"ok": False, "step": "pytest", "summary": "1 failed"}
    )
    (tmp_path / "repo").mkdir()
    report = B.build(make_repo(tmp_path / "repo"), tmp_path / "out", bench_files=FAKE_BENCH)
    assert not report["ok"]
    assert "hakembench: tests failed in its own venv (pytest)" in report["blocking_summary"]


def test_exclusion_rules_match_paths_and_folders():
    rules = ["SESSION.md", "results/private/"]
    assert A.excluded("SESSION.md", rules) and A.excluded("results/private/a/b.json", rules)
    assert not A.excluded("docs/SESSION.md", rules) and not A.excluded("results/step8/x", rules)


def test_a_left_out_or_removed_id_hash_or_text_blocks_but_an_opened_item_does_not(tmp_path):
    held_hash = keyed(PRIVATE_TEXT, bytes.fromhex(KEY))
    leak = {
        "docs/notes.md": f"Held back: {PRIV[5:]} and a hash {PRIV[5:]}9abc0d.\n",
        "results/step8/rows.jsonl": jsonl([{"id": "x", "state": PRIVATE_TEXT}]),
        "results/step8/gone.txt": f"removed {GONE} here: {REMOVED_TEXT}\n",
        "results/step8/manifest.json": json.dumps({"private_text_hmac_sha256": [held_hash]}),
        # A former private item the open release publishes: its id and text are fine.
        "results/step8/opened.jsonl": jsonl([{"id": OPENED, "state": OPENED_TEXT}]),
    }
    report, _ = run(tmp_path, extra=leak)
    assert not report["ok"]
    found = report["blocking"]
    assert {(f["file"], f["line"], f["id"]) for f in found["restricted_ids"]} == {
        ("ufakzeka-karar-public/docs/notes.md", 1, PRIV)
    }
    assert [(f["file"], f["line"], f["id"]) for f in found["restricted_texts"]] == [
        ("ufakzeka-karar-public/results/step8/rows.jsonl", 1, PRIV)
    ]
    assert [(f["file"], f["id"]) for f in found["restricted_hashes"]] == [
        ("ufakzeka-karar-public/results/step8/manifest.json", PRIV)
    ]
    assert report["scan"]["overlap"] == {"restricted_and_open": [], "removed_and_open": []}
    assert [f["id"] for f in found["removed_ids"]] == [GONE]
    assert [f["id"] for f in found["removed_texts"]] == [GONE]


def test_an_open_item_from_a_private_only_source_is_listed_not_dropped(tmp_path):
    report, out = run(tmp_path, source="clips/mqa")
    found = report["blocking"]["private_only_sources"]
    assert {f["id"] for f in found} == {PUB} and not report["ok"]
    assert {f["why"] for f in found} == {
        "private-only source in its provenance",
        "private-only source in its candidate meta",
    }
    public = (out / B.DATA_TREE / "data/v1.0/items.jsonl").read_text(encoding="utf-8")
    assert PUB in public


def test_an_open_item_or_provenance_row_without_the_canary_blocks(tmp_path):
    report, _ = run(tmp_path, opened_canary=False)
    missing = report["blocking"]["canary_missing"]
    assert {(f["file"], f["id"]) for f in missing} == {
        ("hakembench/data/v1.0/items.jsonl", OPENED),
        ("hakembench/data/v1.0/provenance.jsonl", OPENED),
    }
    # The probe rows carry none either; they are counted, not blocking.
    assert report["trees"][B.DATA_TREE]["canary_missing_probe_rows"] == 1
    assert report["blocking_summary"] == ["canary_missing: 2"]


def test_private_only_rows_in_a_kept_row_file_block_even_gzipped(tmp_path):
    rows = [
        {"row": {"source": "generated:v1", "state": "a"}},
        {"row": {"source": "clips/mqa:tr-faq-question", "state": "b"}},
    ]
    extra = {
        "results/step3/journal.jsonl.gz": gzip.compress(jsonl(rows).encode()),
        "docs/data.md": "Web passages come from clips/mqa and stay private.\n",
    }
    report, _ = run(tmp_path, extra=extra)
    assert report["blocking"]["private_only_rows"] == [
        {
            "file": "ufakzeka-karar-public/results/step3/journal.jsonl.gz",
            "rows": 1,
            "lines": [2],
            "sources": {"clips/mqa:tr-faq-question": 1},
        }
    ]
    assert report["scan"]["binary_not_scanned"] == []


def test_denylist_names_block_except_tested_models_on_result_rows(tmp_path):
    extra = {
        "docs/report.md": "The labels came from\nacme-7b, sadly.\n",
        "results/step9/runs/vendorx-public.jsonl": jsonl([{"model": "VendorX-2", "id": PUB}]),
        "results/step9/runs/strictco-public.jsonl": jsonl([{"model": "strictco-5"}]),
        "bench/adapters/vendorx.py": "MODEL = 'vendorx-2'\n",
        "docs/team.md": f"We worked together on this; mail hello@ufakai.com or {MAIL}.\n",
        "model/arch.py": "# the vendorx architecture class\n",
        "docs/tools.md": "Run acmeco-cli here.\nacmeco judged it.\n",
        "model/judge.py": "JUDGE = 'acme-7b'\n",
        "model/strict.py": "# strictco\n",
    }
    report, _ = run(tmp_path, extra=extra)
    blocking = {
        (f["file"], f["term"], tuple(f["lines"])) for f in report["blocking"]["names_blocking"]
    }
    assert ("ufakzeka-karar-public/docs/report.md", "acme-7b", (2,)) in blocking
    rows = "ufakzeka-karar-public/results/step9/runs/strictco-public.jsonl"
    assert (rows, "strictco", (0, 1)) in blocking
    # An allow rule covers only its terms and lines; an always-blocking term is never allowed.
    assert ("ufakzeka-karar-public/docs/tools.md", "acmeco", (2,)) in blocking
    assert ("ufakzeka-karar-public/model/judge.py", "acme-7b", (1,)) in blocking
    assert ("ufakzeka-karar-public/model/strict.py", "strictco", (1,)) in blocking
    allowed = {(f["file"], f["term"], f["category"]) for f in report["allowed_names"]}
    assert allowed == {
        ("ufakzeka-karar-public/model/arch.py", "vendorx", "technical"),
        ("ufakzeka-karar-public/docs/tools.md", "acmeco", "tool"),
    }
    competitor = {(f["file"], f["term"]) for f in report["competitor_names"]}
    assert competitor == {
        ("ufakzeka-karar-public/results/step9/runs/vendorx-public.jsonl", "vendorx"),
        ("ufakzeka-karar-public/bench/adapters/vendorx.py", "vendorx"),
    }
    assert report["scan"]["provider_words_not_scanned"] == ["together"]
    assert [f["match"] for f in report["blocking"]["emails"]] == [MAIL]


def test_em_dashes_and_phone_numbers_block_but_numbers_in_results_do_not(tmp_path):
    extra = {
        "docs/style.md": "Plain line.\nA line \u2014 with a dash.\n",
        "docs/call.md": f"Arayın: {PHONE}. Örnek: 0555 000 00 00.\n",
        "results/step8/scores.json": json.dumps({"x": 0.5321234567, "h": "ab5321234567cd"}),
    }
    report, _ = run(tmp_path, extra=extra)
    assert [(f["file"], f["line"]) for f in report["blocking"]["em_dashes"]] == [
        ("ufakzeka-karar-public/docs/style.md", 2)
    ]
    phones = [f["file"] for f in report["blocking"]["phones"]]
    assert phones == ["ufakzeka-karar-public/docs/call.md"]


def test_the_text_index_finds_what_release_audit_leaks_finds(tmp_path):
    # A shared opening formula alone is not a copy of the text.
    boiler = "Anayasa Mahkemesi başvurunun kabul edilebilir olduğuna karar verdi"
    texts = {
        "a": PRIVATE_TEXT,
        "b": REMOVED_TEXT,
        "d": "kısa",
        "c": boiler + "; başvurucu yargılamanın uzun sürdüğünü ve tazminat istediğini yazdı.",
    }
    files = {
        "one.md": f"önce {PRIVATE_TEXT} sonra",
        "two.jsonl": jsonl([{"s": boiler + ", ikinci dava."}, {"s": REMOVED_TEXT}]),
    }
    for name, body in files.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    rows = [{"id": k, "half": "private", "text": v} for k, v in texts.items()]
    # leaks also reports ids; one-letter ids are in every file, and ids are scanned apart.
    expected = {
        path: [h for h in hits if not h.startswith("id:")]
        for path, hits in A.leaks(rows, {}, [tmp_path / n for n in files]).items()
    }
    index = B.TextIndex(texts)
    got = {}
    for name, body in files.items():
        units = body.splitlines() if name.endswith(".jsonl") else [body]
        hits = set().union(*(index.hits(u) for u in units))
        if hits:
            got[str(tmp_path / name)] = sorted(hits)
    assert got == expected == {str(tmp_path / "one.md"): ["a"], str(tmp_path / "two.jsonl"): ["b"]}


def test_internal_blocks_are_cut_with_their_markers():
    body = (
        "# Log\n\nKept one.\n\n<!-- internal:start -->\nHeld back.\n\nAlso held back.\n"
        "<!-- internal:end -->\n\n## Next\n\n"
        "Kept <!-- internal:start -->inline<!-- internal:end --> two.\n"
    )
    text, problems = B.strip_internal(body)
    assert problems == []
    assert text == "# Log\n\nKept one.\n\n## Next\n\nKept  two.\n"
    assert B.strip_internal("<!--internal:start-->\nx\n<!--  internal:end -->\nKept.\n") == (
        "Kept.\n",
        [],
    )


def test_unbalanced_internal_markers_are_reported_and_fail_closed():
    text, problems = B.strip_internal("Kept.\n<!-- internal:start -->\nHeld back.\n")
    assert "Held back" not in text and "internal:start" in text
    assert problems == [{"line": 2, "why": "start with no end"}]
    text, problems = B.strip_internal("Kept.\n<!-- internal:end -->\n")
    assert problems == [{"line": 2, "why": "end with no start"}]
    text, problems = B.strip_internal(
        "<!-- internal:start -->\na\n<!-- internal:start -->\nb\n<!-- internal:end -->\nc\n"
    )
    assert text == "c\n" and problems == [{"line": 3, "why": "start inside an open block"}]


def test_a_repo_with_internal_blocks_builds_and_an_unbalanced_marker_blocks(tmp_path):
    held = "A paragraph the owner keeps out of public text."
    extra = {
        "docs/NOTES.md": f"## One\n\nPublic.\n\n<!-- internal:start -->\n{held}\n"
        "<!-- internal:end -->\n\n## Two\n\nPublic too.\n",
        "bench/notes.py": "# <!-- internal:start --> is only cut from Markdown files\n",
    }
    report, out = run(tmp_path, extra=extra)
    assert report["ok"], report["blocking_summary"]
    shipped = (out / B.CODE_TREE / "docs/NOTES.md").read_text(encoding="utf-8")
    assert shipped == "## One\n\nPublic.\n\n## Two\n\nPublic too.\n"
    assert held not in shipped

    (tmp_path / "again").mkdir()
    extra = {"docs/NOTES.md": f"## One\n\n<!-- internal:start -->\n{held}\n"}
    report = B.build(
        make_repo(tmp_path / "again", extra=extra),
        tmp_path / "out2",
        run_tests=False,
        bench_files=FAKE_BENCH,
    )
    assert not report["ok"] and "internal_markers: 2" in report["blocking_summary"]
    assert {(f["file"], f["line"], f["why"]) for f in report["blocking"]["internal_markers"]} == {
        ("ufakzeka-karar-public/docs/NOTES.md", 3, "start with no end"),
        ("ufakzeka-karar-public/docs/NOTES.md", 3, "marker left in the tree"),
    }
    assert held not in (tmp_path / "out2" / B.CODE_TREE / "docs/NOTES.md").read_text()


def test_the_build_refuses_a_repo_without_the_public_readme(tmp_path):
    (tmp_path / "repo").mkdir()
    repo = make_repo(tmp_path / "repo")
    subprocess.run(["git", "rm", "-q", "release/public_readme.md"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=a", "-c", "user.email=a@example.invalid", "commit", "-qm", "x"],
        cwd=repo,
        check=True,
    )
    try:
        B.build(repo, tmp_path / "out", run_tests=False, bench_files=FAKE_BENCH)
    except B.BuildError as error:
        assert "public_readme.md" in str(error)
    else:
        raise AssertionError("built without the public README")


# Python fixtures are joined from pieces, so no line of this file is a marker line itself.
START, END = "# internal:" + "start", "# internal:" + "end"


def test_python_internal_blocks_are_cut_whole_lines_and_blank_runs_merge():
    body = "\n".join(
        [
            "import os",
            START,
            "SECRET = 1",
            END,
            "",
            "",
            "def kept():",
            "    x = 1",
            "    " + START,
            "    if x:",
            "        return SECRET",
            "    " + END,
            "    return x",
            "",
            "",
            START,
            "def held():",
            "    return os.sep",
            "",
            "",
            END,
            "",
            "",
            "def last():",
            "    return 2",
            "",
        ]
    )
    text, problems = B.strip_internal_py(body)
    assert problems == []
    assert text == (
        "import os\n\n\ndef kept():\n    x = 1\n    return x\n\n\ndef last():\n    return 2\n"
    )
    # A comment that only mentions a marker is kept, and a file without blocks is unchanged.
    same = "x = 1  " + START + "\n# see " + END + " below\n"
    assert B.strip_internal_py(same) == (same, [])


def test_unbalanced_python_markers_and_a_broken_cut_are_reported():
    text, problems = B.strip_internal_py("a = 1\n" + START + "\nb = 2\n")
    assert "b = 2" not in text and START in text
    assert problems == [{"line": 2, "why": "start with no end"}]
    text, problems = B.strip_internal_py("a = 1\n" + END + "\n")
    assert text == "a = 1\n" + END + "\n"
    assert problems == [{"line": 2, "why": "end with no start"}]
    body = "\n".join([START, "a = 1", START, "b = 2", END, "c = 3", ""])
    text, problems = B.strip_internal_py(body)
    assert text == "c = 3\n" and problems == [{"line": 3, "why": "start inside an open block"}]
    # Cutting the only statement of a body leaves code that does not parse.
    body = "\n".join(["def f():", "    " + START, "    return 1", "    " + END, ""])
    _, problems = B.strip_internal_py(body)
    assert [p["why"] for p in problems] == ["does not parse once cut"]


def test_the_build_cuts_python_blocks_in_both_trees_and_blocks_a_left_marker(tmp_path):
    runner = "\n".join(
        [
            "MODELS = {'open': 1}",
            START,
            "MODELS['held-back'] = 2",
            END,
            "",
        ]
    )
    extra = {"bench/harness/items.py": runner}
    report, out = run(tmp_path, extra=extra)
    assert report["ok"], report["blocking_summary"]
    for tree in (B.CODE_TREE, B.DATA_TREE):
        assert (out / tree / "bench/harness/items.py").read_text() == "MODELS = {'open': 1}\n"
    # The private repo keeps its file whole.
    assert "held-back" in (tmp_path / "repo/bench/harness/items.py").read_text()

    (tmp_path / "again").mkdir()
    extra = {"bench/harness/items.py": "MODELS = {}\n" + START + "\nMODELS['x'] = 1\n"}
    report = B.build(
        make_repo(tmp_path / "again", extra=extra),
        tmp_path / "out2",
        run_tests=False,
        bench_files=FAKE_BENCH,
    )
    found = {(f["file"], f["why"]) for f in report["blocking"]["internal_markers"]}
    for tree in (B.CODE_TREE, B.DATA_TREE):
        assert (f"{tree}/bench/harness/items.py", "start with no end") in found
        assert (f"{tree}/bench/harness/items.py", "marker left in the tree") in found
    assert not report["ok"]


def test_an_allow_rule_may_name_one_tree_by_its_folder(tmp_path):
    card = "# Card\n\nBoard row: vendorx-2 0.8.\n"
    extra = {
        "release/public_readme.md": "# fake\n\nvendorx is named here.\n",
        "release/cards/hakembench/README.md": card,
        "config/release_names.json": json.dumps(
            {
                "terms": ["vendorx", "strictco"],
                "always_blocking": ["strictco"],
                "contacts": ["hello@ufakai.com"],
                "allowed": [
                    {
                        "path": "hakembench/README.md",
                        "terms": ["vendorx", "strictco"],
                        "category": "competitor",
                    }
                ],
            }
        ),
    }
    report, out = run(tmp_path, extra=extra)
    assert (out / B.DATA_TREE / "README.md").read_text() == card
    allowed = {(f["file"], f["term"], f["category"]) for f in report["allowed_names"]}
    assert allowed == {("hakembench/README.md", "vendorx", "competitor")}
    # The code tree's README.md is not the card, and the card's own copy in the code tree has
    # another path: the rule names neither, so both still block.
    blocking = {(f["file"], f["term"]) for f in report["blocking"]["names_blocking"]}
    assert blocking == {
        ("ufakzeka-karar-public/README.md", "vendorx"),
        ("ufakzeka-karar-public/release/cards/hakembench/README.md", "vendorx"),
    }


def test_a_training_text_that_is_not_ours_blocks_whatever_its_key(tmp_path):
    escaped = json.dumps({"x": WEB_TEXT}, ensure_ascii=True)
    extra = {
        # The audit_b case: rows keyed "task" and "text", no "source" key.
        "results/step6/audit_items.jsonl": jsonl([{"task": "sss-noul", "text": WEB_TEXT}]),
        "results/step6/chunk.json": escaped,
        "results/step6/tweets.jsonl.gz": gzip.compress(jsonl([{"t": TWEET}]).encode()),
        "docs/notes.md": f"An example:\n\n{WEB_TEXT.split(chr(10))[1]}\n",
        # Our own texts and the open set's texts never block.
        "results/step6/ours.jsonl": jsonl([{"text": OUR_TEXT}, {"text": PUBLIC_TEXT}]),
        "docs/PUBLIC_TEXT_FILES.txt": "results/step6/ours.jsonl our own texts\n"
        "results/step6/audit_items.jsonl reviewed, and still blocked by the overlap scan\n",
    }
    report, _ = run(tmp_path, extra=extra)
    found = {f["file"]: f for f in report["blocking"]["training_texts"]}
    prefix = "ufakzeka-karar-public/"
    assert set(found) == {
        prefix + "results/step6/audit_items.jsonl",
        prefix + "results/step6/chunk.json",
        prefix + "results/step6/tweets.jsonl.gz",
        prefix + "docs/notes.md",
    }
    assert found[prefix + "results/step6/audit_items.jsonl"]["sources"] == {"clips/mqa": 1}
    assert found[prefix + "results/step6/tweets.jsonl.gz"]["sources"] == {"mide22": 1}
    assert found[prefix + "docs/notes.md"]["lines"] == [3]
    assert report["scan"]["overlap_index"]["by_source"] == {
        "clips/mqa": 2,
        "mide22": 1,
        "left out": 1,
        "removed": 1,
    }
    assert report["scan"]["overlap_index"]["runs_dropped_as_open_set_text"] > 0


def test_the_build_refuses_to_scan_without_the_training_rows(tmp_path):
    (tmp_path / "repo").mkdir()
    repo = make_repo(tmp_path / "repo")
    shutil.rmtree(repo / "data/built")
    try:
        B.build(repo, tmp_path / "out", run_tests=False, bench_files=FAKE_BENCH)
    except B.BuildError as error:
        assert "data/built" in str(error)
    else:
        raise AssertionError("scanned without the training rows")


def test_text_fields_under_results_block_unless_reviewed_and_rows_stay_on_open_items(tmp_path):
    words = " ".join(["kelime"] * 9)
    long_note = " ".join(["uzun"] * 30)
    row = {"row_schema": 1, "item_id": PUB, "answer": {}, "path_used": long_note}
    extra = {
        "results/step7/texts.jsonl": jsonl([{"id": "a", "passage": words}]),
        "results/step7/note.json": json.dumps({"summary": {"why": long_note}}),
        "results/step7/short.json": json.dumps({"text": "kısa bir metin"}),
        "results/step7/listed/a.jsonl": jsonl([{"state": words}]),
        "results/step9/open_parts/part_a.jsonl": jsonl([{"state": words}]),
        "results/step9/runs/m-public.jsonl": jsonl([row, {**row, "item_id": OPENED + "~p-1"}]),
        "results/step9/runs/m-private.jsonl": jsonl([{**row, "item_id": PRIV}]),
        "results/step7/broken.jsonl": "{not json\n",
        "docs/PUBLIC_TEXT_FILES.txt": "# reviewed\nresults/step7/listed/ texts we wrote\n"
        "results/gone.json a file that is not there\n",
    }
    report, _ = run(tmp_path, extra=extra)
    prefix = "ufakzeka-karar-public/results/"
    fields = {f["file"]: f for f in report["blocking"]["text_fields"]}
    assert set(fields) == {
        prefix + "step7/texts.jsonl",
        prefix + "step7/note.json",
        prefix + "step7/broken.jsonl",
    }
    assert fields[prefix + "step7/texts.jsonl"]["keys"] == {"passage": 1}
    assert fields[prefix + "step7/note.json"]["keys"] == {"why": 1}
    off = report["blocking"]["rows_off_open_set"]
    assert [(f["file"], f["first"]) for f in off] == [
        (prefix + "step9/runs/m-private.jsonl", [PRIV])
    ]
    assert report["scan"]["reviewed_text_files_unused"] == ["results/gone.json"]


def test_a_reviewed_text_file_needs_a_reason(tmp_path):
    extra = {"docs/PUBLIC_TEXT_FILES.txt": "results/step7/a.json\n"}
    (tmp_path / "repo").mkdir()
    repo = make_repo(tmp_path / "repo", extra=extra)
    try:
        B.build(repo, tmp_path / "out", run_tests=False, bench_files=FAKE_BENCH)
    except B.BuildError as error:
        assert "has no reason" in str(error)
    else:
        raise AssertionError("a reviewed file without a reason was accepted")


def test_a_name_after_an_escape_is_found_and_an_always_blocking_term_needs_a_line_rule(tmp_path):
    story = "Değerli [ad],\n\nStrictco Teknoloji'de bir iş teklifi var."
    names = json.loads(make_names())
    names["allowed"].append(
        {
            "path": "results/step3/generated.jsonl",
            "terms": ["strictco"],
            "line_contains": "Strictco Teknoloji'de bir iş",
            "always_blocking_line": True,
            "category": "generated text",
        }
    )
    extra = {
        "config/release_names.json": json.dumps(names),
        # Escaped newline right before a name: the raw text has "nVendorx".
        "results/step3/rows.jsonl": jsonl([{"s": "bir\n\nVendorx iki"}]),
        "model/code.py": 'MESSAGE = "one\\nvendorx two"\n',
        "results/step3/generated.jsonl": jsonl([{"text": story}]),
        # The same rule does not cover a second mention on its line, nor another file.
        "results/step3/other.jsonl": jsonl([{"text": story + " strictco"}]),
        "results/step3/plain.jsonl": jsonl([{"text": "Strictco judged it."}]),
    }
    report, _ = run(tmp_path, extra=extra)
    blocking = {(f["file"], f["term"]) for f in report["blocking"]["names_blocking"]}
    prefix = "ufakzeka-karar-public/"
    assert (prefix + "results/step3/rows.jsonl", "vendorx") in blocking
    assert (prefix + "results/step3/other.jsonl", "strictco") in blocking
    assert (prefix + "results/step3/plain.jsonl", "strictco") in blocking
    assert (prefix + "results/step3/generated.jsonl", "strictco") not in blocking
    allowed = {(f["file"], f["term"], f["category"]) for f in report["allowed_names"]}
    assert (prefix + "results/step3/generated.jsonl", "strictco", "generated text") in allowed
    # model/ allows vendorx as a technical name, escaped or not.
    assert (prefix + "model/code.py", "vendorx", "technical") in allowed
    assert B.unescaped('a\\nb\\u00e7\\ud83d\\"') == 'a bç "'


def test_the_justfile_loses_its_internal_recipes(tmp_path):
    justfile = "\n".join(
        ["default:", "    @just --list", "", START, "private:", "    cat results/private/x", END,
         "", "test:", "    uv run pytest", ""]
    )  # fmt: skip
    report, out = run(tmp_path, extra={"justfile": justfile})
    assert report["ok"], report["blocking_summary"]
    shipped = (out / B.CODE_TREE / "justfile").read_text()
    assert shipped == "default:\n    @just --list\n\ntest:\n    uv run pytest\n"
    (tmp_path / "again").mkdir()
    report = B.build(
        make_repo(tmp_path / "again", extra={"justfile": "a:\n" + START + "\n    x\n"}),
        tmp_path / "out2",
        run_tests=False,
        bench_files=FAKE_BENCH,
    )
    assert not report["ok"] and "internal_markers: 2" in report["blocking_summary"]
