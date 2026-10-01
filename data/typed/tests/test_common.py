"""Pinned downloads are checked by hash, and a changed file stops the build."""

import pytest

from data.typed.common import fetch_pinned, hub_file_url, sha256_of


def test_a_file_whose_hash_changed_is_refused(tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("kaynak", encoding="utf-8")
    expected = sha256_of(good)
    calls = []

    def fake_download(url, target):
        calls.append(url)
        target.write_text("kaynak", encoding="utf-8")

    folder = tmp_path / "raw"
    paths = fetch_pinned({"a.txt": ("https://example.org/a", expected)}, folder,
                         download=fake_download)  # fmt: skip
    assert paths["a.txt"].read_text(encoding="utf-8") == "kaynak" and len(calls) == 1
    fetch_pinned({"a.txt": ("https://example.org/a", expected)}, folder, download=fake_download)
    assert len(calls) == 1, "a present file is downloaded again"

    (folder / "a.txt").write_text("değişti", encoding="utf-8")
    with pytest.raises(ValueError, match="is not the pinned"):
        fetch_pinned({"a.txt": ("https://example.org/a", expected)}, folder,
                     download=fake_download)  # fmt: skip


def test_hub_urls_are_pinned_and_quoted():
    revision = "a" * 40
    url = hub_file_url("org/set", revision, "Citation for Dataset")
    base = "https://huggingface.co/datasets/org/set/resolve"
    assert url == f"{base}/{revision}/Citation%20for%20Dataset"
