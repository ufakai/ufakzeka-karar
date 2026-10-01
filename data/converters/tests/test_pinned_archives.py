"""The command-line path refuses an archive that is not the one that was read."""

import pytest

from data.converters import massive_tr, offenseval_tr


def _never(url, dest):
    raise AssertionError("no download expected")


@pytest.mark.parametrize(
    ("module", "archive_name"),
    [(massive_tr, massive_tr.ARCHIVE_NAME), (offenseval_tr, offenseval_tr.ARCHIVE_NAME)],
)
def test_a_different_archive_is_refused(tmp_path, module, archive_name):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / archive_name).write_bytes(b"not the archive that was read on 2026-09-19")
    with pytest.raises(RuntimeError, match="not the pinned archive"):
        module.convert(
            tmp_path / "out", cache, fetch=_never, expected_sha256=module.EXPECTED_SHA256
        )
    assert not (tmp_path / "out").exists()


def test_the_pins_are_full_sha256_digests():
    for module in (massive_tr, offenseval_tr):
        assert len(module.EXPECTED_SHA256) == 64
        int(module.EXPECTED_SHA256, 16)
