"""The license gate fails closed.

Each test builds a small repo in a temporary folder, so the real manifest
is never touched. test_the_repo_manifest_passes checks the real one.
"""

import datetime as dt
from pathlib import Path

import pytest
import yaml

from data.licenses import ALLOWED_LICENSE_IDS, deny_markers
from data.manifest import check, main

TODAY = dt.date(2026, 9, 19)
REPO_ROOT = Path(__file__).resolve().parents[2]

GOOD_ENTRY = {
    "path": "data/raw/xcopa-tr/",
    "source": "https://huggingface.co/datasets/cambridgeltl/xcopa",
    "upstream": "original release by the authors, license read on the dataset card",
    "license_as_shown": "cc-by-4.0",
    "license_id": "CC-BY-4.0",
    "verified": TODAY,
    "attribution": "Ponti et al. 2020, XCOPA",
    "use": "both",
}


def make_repo(tmp_path: Path, entries: list[dict], files: list[str]) -> Path:
    manifest = {"version": 1, "roots": ["data/raw", "bench/items"], "entries": entries}
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "data/MANIFEST.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    return tmp_path


def errors_for(tmp_path, entries, files, **kwargs) -> list[str]:
    return check(make_repo(tmp_path, entries, files), today=TODAY, **kwargs).errors


def test_a_complete_allowed_entry_passes(tmp_path):
    files = ["data/raw/xcopa-tr/test.jsonl", "data/raw/xcopa-tr/val/part-0.jsonl"]
    report = check(make_repo(tmp_path, [GOOD_ENTRY], files), today=TODAY)
    assert report.ok, report.errors
    assert report.files_checked == 2


# The done-when of step 0: the manifest check fails on a fake file.
def test_a_fake_file_without_an_entry_fails(tmp_path):
    files = ["data/raw/xcopa-tr/test.jsonl", "data/raw/scraped-tweets.jsonl"]
    errors = errors_for(tmp_path, [GOOD_ENTRY], files)
    assert errors == ["data/raw/scraped-tweets.jsonl: data file without a valid manifest entry"]


def test_a_fake_file_fails_through_the_command_line(tmp_path, capsys):
    repo = make_repo(tmp_path, [GOOD_ENTRY], ["bench/items/fake.jsonl"])
    assert main(["--repo-root", str(repo)]) == 1
    assert "bench/items/fake.jsonl" in capsys.readouterr().err


def test_a_file_entry_does_not_cover_its_neighbours(tmp_path):
    entry = {**GOOD_ENTRY, "path": "data/raw/a.jsonl"}
    errors = errors_for(tmp_path, [entry], ["data/raw/a.jsonl", "data/raw/a.jsonl.bak"])
    assert errors == ["data/raw/a.jsonl.bak: data file without a valid manifest entry"]


@pytest.mark.parametrize(
    ("license_id", "as_shown"),
    [
        ("CC-BY-SA-4.0", "cc-by-sa-4.0"),
        ("CC-BY-NC-4.0", "cc-by-nc-4.0"),
        ("CC-BY-NC-ND-4.0", "cc-by-nc-nd-4.0"),
        ("CC-BY-NC-SA-4.0", "CC BY-NC-SA 4.0"),
        ("GPL-3.0-only", "GPL-3.0"),
        ("unknown", "no license shown"),
        ("cc-by-4.0", "cc-by-4.0"),  # ids are matched exactly, case included
    ],
)
def test_a_license_outside_the_allow_list_fails(tmp_path, license_id, as_shown):
    entry = {**GOOD_ENTRY, "license_id": license_id, "license_as_shown": as_shown}
    errors = errors_for(tmp_path, [entry], ["data/raw/xcopa-tr/test.jsonl"])
    assert any("not on the allow-list" in e for e in errors)
    # The rejected entry covers nothing, so its file is reported as well.
    assert any("without a valid manifest entry" in e for e in errors)


@pytest.mark.parametrize(
    "as_shown",
    [
        "cc-by-nc-4.0",
        "CC BY-SA 4.0",
        "Creative Commons Attribution-NonCommercial 4.0",
        "Attribution Non Commercial",
        "CC BY-ND",
        "free for general research use",
        "other",
    ],
)
def test_an_allowed_id_cannot_cover_a_denied_license_string(tmp_path, as_shown):
    entry = {**GOOD_ENTRY, "license_as_shown": as_shown}
    errors = errors_for(tmp_path, [entry], [])
    assert any("denied terms" in e for e in errors)


@pytest.mark.parametrize(
    "as_shown",
    [
        "cc-by-4.0",
        "CC BY 2.0",
        "Creative Commons Attribution 4.0 International",
        "MIT",
        "apache-2.0",
        "BSD 3-Clause",
        "CC0 1.0 Universal",
        "Public domain under FSEK article 31",
        "CC BY 4.0, authored by ufak AI",
    ],
)
def test_allowed_license_strings_carry_no_denied_terms(as_shown):
    assert deny_markers(as_shown) == []


def test_the_allow_list_holds_no_restricted_license():
    for license_id in ALLOWED_LICENSE_IDS:
        assert deny_markers(license_id) == [], license_id


@pytest.mark.parametrize("key", ["source", "license_as_shown", "verified", "attribution", "use"])
def test_a_missing_field_fails(tmp_path, key):
    entry = {k: v for k, v in GOOD_ENTRY.items() if k != key}
    assert any(f"missing {key}" in e for e in errors_for(tmp_path, [entry], []))


def test_a_blank_field_counts_as_missing(tmp_path):
    errors = errors_for(tmp_path, [{**GOOD_ENTRY, "attribution": "  "}], [])
    assert any("missing attribution" in e for e in errors)


def test_a_misspelled_key_fails(tmp_path):
    errors = errors_for(tmp_path, [{**GOOD_ENTRY, "licence": "MIT"}], [])
    assert any("unknown keys ['licence']" in e for e in errors)


def test_a_future_verification_date_fails(tmp_path):
    entry = {**GOOD_ENTRY, "verified": TODAY + dt.timedelta(days=1)}
    assert any("in the future" in e for e in errors_for(tmp_path, [entry], []))


def test_a_verification_date_must_be_a_date(tmp_path):
    entry = {**GOOD_ENTRY, "verified": "last week"}
    assert any("YYYY-MM-DD" in e for e in errors_for(tmp_path, [entry], []))


def test_a_source_must_be_a_url_unless_authored(tmp_path):
    entry = {**GOOD_ENTRY, "source": "a colleague's drive"}
    assert any("source must be the URL" in e for e in errors_for(tmp_path, [entry], []))

    authored = {
        **GOOD_ENTRY,
        "path": "bench/items/smoke.jsonl",
        "source": "authored",
        "license_id": "LicenseRef-ufakai-authored",
        "license_as_shown": "CC BY 4.0, authored by ufak AI",
    }
    assert errors_for(tmp_path / "b", [authored], ["bench/items/smoke.jsonl"]) == []


def test_an_authored_claim_on_a_downloaded_source_fails(tmp_path):
    entry = {**GOOD_ENTRY, "license_id": "LicenseRef-ufakai-authored"}
    assert any("authored data must have source" in e for e in errors_for(tmp_path, [entry], []))


@pytest.mark.parametrize("path", ["/etc/passwd", "data/raw/../../secrets", "models/weights/"])
def test_an_entry_outside_the_data_roots_fails(tmp_path, path):
    errors = errors_for(tmp_path, [{**GOOD_ENTRY, "path": path}], [])
    assert any("outside the data roots" in e or "relative to the repo root" in e for e in errors)


def test_overlapping_entries_fail(tmp_path):
    inner = {**GOOD_ENTRY, "path": "data/raw/xcopa-tr/test.jsonl"}
    errors = errors_for(tmp_path, [GOOD_ENTRY, inner], [])
    assert any("overlap" in e for e in errors)


def test_an_absent_file_passes_by_default_and_fails_when_required(tmp_path):
    repo = make_repo(tmp_path, [GOOD_ENTRY], [])
    relaxed = check(repo, today=TODAY)
    assert relaxed.ok
    assert relaxed.absent == ["data/raw/xcopa-tr/"]
    strict = check(repo, today=TODAY, require_present=True)
    assert any("not on disk" in e for e in strict.errors)


def test_os_litter_is_ignored_and_nothing_else(tmp_path):
    files = ["data/raw/.DS_Store", "data/raw/.gitkeep", "data/raw/.hidden.jsonl"]
    errors = errors_for(tmp_path, [], files)
    assert errors == ["data/raw/.hidden.jsonl: data file without a valid manifest entry"]


def test_a_missing_or_broken_manifest_fails(tmp_path):
    assert not check(tmp_path, today=TODAY).ok
    (tmp_path / "data").mkdir()
    (tmp_path / "data/MANIFEST.yaml").write_text("version: 1\nroots: [", encoding="utf-8")
    assert not check(tmp_path, today=TODAY).ok


def test_the_repo_manifest_passes():
    report = check(REPO_ROOT)
    assert report.ok, report.errors


def test_odc_by_is_allowed_and_the_share_alike_database_licence_is_not():
    """ODC-By 1.0 is attribution-only; ODbL is share-alike and stays out."""
    from data.licenses import ALLOWED_LICENSE_IDS, deny_markers

    assert "ODC-By-1.0" in ALLOWED_LICENSE_IDS
    # The id itself must not trip the consistency check that reads the licence
    # string as the source shows it.
    assert deny_markers("ODC-By-1.0") == []
    assert deny_markers("Open Data Commons Attribution License v1.0") == []
    # ODbL shares alike, so both its id and its spelled-out name are refused.
    assert "ODbL-1.0" not in ALLOWED_LICENSE_IDS
    assert deny_markers("ODbL-1.0") == ["odbl"]
    assert "sa" in deny_markers("ODC-BY-SA")
