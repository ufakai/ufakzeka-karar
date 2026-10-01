"""The license gate: `just check-licenses`.

data/MANIFEST.yaml lists every data file with its source, license, date
verified and attribution. This module fails the build when

  1. a file exists under a data root and no entry covers it,
  2. an entry is incomplete, malformed or carries an unknown key,
  3. an entry's license id is not on the allow-list (data/licenses.py),
  4. the license string as shown on the source contains a denied term,
  5. an entry lies outside the data roots, or two entries cover one path.

Downloaded and built data stay out of git, so on a fresh clone or in CI an
entry's file may be absent. That is reported and passes by default. Dataset
builds and training runs pass --require-present, which makes it an error.

An entry path is a file, or a folder written with a trailing slash, which
covers everything beneath it (a dataset snapshot with many shards). There
are no globs: a pattern that matches too much would defeat the check.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from data.licenses import ALLOWED_LICENSE_IDS, AUTHORED_LICENSE_ID, deny_markers

MANIFEST_PATH = Path("data/MANIFEST.yaml")
MANIFEST_VERSION = 1

REQUIRED_KEYS = (
    "path",
    "source",
    "upstream",
    "license_as_shown",
    "license_id",
    "verified",
    "attribution",
    "use",
)
OPTIONAL_KEYS = ("sha256", "notes")
# measure: used to take a measurement (the instrument). It clears a set for neither
# the release model's training data nor the benchmark; promoting it takes a new check.
USES = ("train", "bench", "both", "measure")

# Files the operating system or git conventions leave behind. Nothing else is skipped.
IGNORED_NAMES = frozenset({".DS_Store", ".gitkeep"})


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)
    files_checked: int = 0
    entries_checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def _is_folder_entry(path: str) -> bool:
    return path.endswith("/")


def _covers(entry_path: str, file_path: str) -> bool:
    if _is_folder_entry(entry_path):
        return file_path.startswith(entry_path)
    return file_path == entry_path


def _inside(path: str, roots: list[str]) -> bool:
    return any(path == root or path.startswith(root + "/") for root in roots)


def _check_entry(index: int, entry: object, roots: list[str], today: dt.date) -> list[str]:
    where = f"entry {index}"
    if not isinstance(entry, dict):
        return [f"{where}: must be a mapping"]
    if isinstance(entry.get("path"), str):
        where = f"entry {index} ({entry['path']})"

    errors: list[str] = []
    unknown = sorted(set(entry) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS))
    if unknown:
        errors.append(f"{where}: unknown keys {unknown}")
    for key in REQUIRED_KEYS:
        value = entry.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"{where}: missing {key}")
    if errors:
        return errors

    path, license_id = entry["path"], entry["license_id"]
    for key in ("path", "source", "upstream", "license_as_shown", "license_id", "attribution"):
        if not isinstance(entry[key], str):
            errors.append(f"{where}: {key} must be text")
    if errors:
        return errors

    if path.startswith("/") or ".." in Path(path).parts:
        errors.append(f"{where}: path must be relative to the repo root, without '..'")
    elif not _inside(path.rstrip("/"), roots):
        errors.append(f"{where}: path is outside the data roots {roots}")

    if license_id not in ALLOWED_LICENSE_IDS:
        errors.append(f"{where}: license id {license_id!r} is not on the allow-list")
    markers = deny_markers(entry["license_as_shown"])
    if markers:
        errors.append(
            f"{where}: license as shown {entry['license_as_shown']!r} contains "
            f"denied terms {markers}"
        )

    source = entry["source"]
    if license_id == AUTHORED_LICENSE_ID:
        if source != "authored":
            errors.append(f"{where}: authored data must have source 'authored'")
    elif not source.startswith(("https://", "http://")):
        errors.append(f"{where}: source must be the URL the data was taken from")

    verified = entry["verified"]
    # PyYAML reads an unquoted 2026-09-19 as a date. A datetime is not accepted.
    if not isinstance(verified, dt.date) or isinstance(verified, dt.datetime):
        errors.append(f"{where}: verified must be a date written as YYYY-MM-DD")
    elif verified > today:
        errors.append(f"{where}: verified date {verified} is in the future")

    if entry["use"] not in USES:
        errors.append(f"{where}: use must be one of {list(USES)}")
    return errors


def _data_files(repo_root: Path, roots: list[str]) -> list[str]:
    found: list[str] = []
    for root in roots:
        base = repo_root / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            # A symlink counts as a file and is not followed out of the tree.
            if path.is_dir() and not path.is_symlink():
                continue
            if path.name in IGNORED_NAMES:
                continue
            found.append(path.relative_to(repo_root).as_posix())
    return found


def check(
    repo_root: Path,
    *,
    require_present: bool = False,
    today: dt.date | None = None,
) -> Report:
    report = Report()
    today = today or dt.date.today()
    manifest_file = repo_root / MANIFEST_PATH
    if not manifest_file.is_file():
        report.errors.append(f"{MANIFEST_PATH} does not exist")
        return report
    try:
        manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        report.errors.append(f"{MANIFEST_PATH} is not valid YAML: {exc}")
        return report

    if not isinstance(manifest, dict) or manifest.get("version") != MANIFEST_VERSION:
        report.errors.append(
            f"{MANIFEST_PATH}: expected a mapping with version: {MANIFEST_VERSION}"
        )
        return report
    unknown = sorted(set(manifest) - {"version", "roots", "entries"})
    if unknown:
        report.errors.append(f"{MANIFEST_PATH}: unknown top-level keys {unknown}")
    roots = manifest.get("roots")
    if not isinstance(roots, list) or not roots or not all(isinstance(r, str) for r in roots):
        report.errors.append(f"{MANIFEST_PATH}: roots must be a non-empty list of folders")
        return report
    roots = [r.rstrip("/") for r in roots]
    entries = manifest.get("entries") or []
    if not isinstance(entries, list):
        report.errors.append(f"{MANIFEST_PATH}: entries must be a list")
        return report

    valid_paths: list[str] = []
    for index, entry in enumerate(entries):
        report.entries_checked += 1
        problems = _check_entry(index, entry, roots, today)
        report.errors.extend(problems)
        # A rejected entry covers nothing, so its files are also reported as
        # uncovered. One bad entry cannot hide the files beneath it.
        if not problems:
            valid_paths.append(entry["path"])

    for i, a in enumerate(valid_paths):
        for b in valid_paths[i + 1 :]:
            if a == b or _covers(a, b) or _covers(b, a):
                report.errors.append(f"entries overlap: {a!r} and {b!r}")

    files = _data_files(repo_root, roots)
    report.files_checked = len(files)
    for file_path in files:
        if not any(_covers(entry_path, file_path) for entry_path in valid_paths):
            report.errors.append(f"{file_path}: data file without a valid manifest entry")

    for entry_path in valid_paths:
        target = repo_root / entry_path
        present = target.is_dir() if _is_folder_entry(entry_path) else target.is_file()
        if not present:
            report.absent.append(entry_path)
            if require_present:
                report.errors.append(f"{entry_path}: listed in the manifest but not on disk")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="License gate for data/MANIFEST.yaml")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--require-present",
        action="store_true",
        help="fail when a listed file is not on disk (dataset builds and training runs)",
    )
    args = parser.parse_args(argv)

    report = check(args.repo_root, require_present=args.require_present)
    for path in report.absent:
        if not args.require_present:
            print(f"absent (not downloaded here): {path}")
    for error in report.errors:
        print(f"FAIL {error}", file=sys.stderr)
    summary = f"{report.entries_checked} entries, {report.files_checked} data files on disk"
    if report.ok:
        print(f"license check passed: {summary}")
        return 0
    print(f"license check failed with {len(report.errors)} problems: {summary}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
