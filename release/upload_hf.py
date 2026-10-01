"""Upload the model and HakemBench to the Hugging Face Hub, after the owner's review.

    python -m release.upload_hf --dataset-tree DIR [--model-dir DIR]            # dry run
    uv run --group baselines python -m release.upload_hf --dataset-tree DIR --model-dir DIR --go

Lists exactly what would go where, and uploads only with --go:

- the model repo ufakai/ufakzeka-karar: every file of the folder release/convert.py wrote
  (checked against its SHA256SUMS first), plus release/cards/model/README.md as README.md
  when that card exists and passes the public build's text scans (names, dashes, e-mail
  addresses and phone numbers, left-out ids, hashes and texts, internal markers), which the
  card, sent outside the two trees, would otherwise skip;
- the dataset repo ufakai/HakemBench: the HakemBench tree's data/ folder, its README.md
  (the card), croissant.json when present, and ATTRIBUTION.md, which the data licences ask
  for. The code stays on GitHub.

With --go the scan report of the public build must say ok and must describe what goes up:
the dataset tree's git HEAD must be the commit the report scanned for it, with no change on
top, and this repository's HEAD must be the report's repo_head. --dataset-tree is always
given; no folder is guessed. Nothing is skipped quietly: a file of the model folder that its
SHA256SUMS does not list (a __pycache__ folder or a hidden file included) is refused with
what to do about it. Each repo is made private when it does not exist yet (the owner makes it
public), and each upload is one upload_folder call over exactly the listed files, never git.
Nothing on the Hub is deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from release import public_build

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "ufakai/ufakzeka-karar"
DATASET_REPO = "ufakai/HakemBench"
MODEL_CARD = ROOT / "release/cards/model/README.md"
# The path the card is scanned under, whatever file is passed: the allow rules name it.
CARD_PATH = "release/cards/model/README.md"
BUILD = ROOT.parent / "public-build"
REPORT = BUILD / "scan_report.json"
DATA_TREE = "hakembench"
DATASET_TOP = ("README.md", "croissant.json", "ATTRIBUTION.md")
SUMS = "SHA256SUMS"


class UploadError(RuntimeError):
    pass


def _unwanted(path: Path, root: Path) -> str | None:
    """Why a file found in a folder must not be there, with the fix, or None."""
    parts = path.relative_to(root).parts
    if "__pycache__" in parts or path.suffix == ".pyc":
        return (
            f"{path.relative_to(root)} is Python bytecode, written when karar.py ran from the "
            f"folder: remove it (rm -rf {root}/__pycache__) and run from the folder with "
            "PYTHONDONTWRITEBYTECODE=1 set"
        )
    if any(part.startswith(".") for part in parts):
        return f"{path.relative_to(root)} is a hidden file: remove it (rm -rf {path})"
    return None


def git_head(tree: Path) -> tuple[str | None, str]:
    """A folder's git HEAD and its uncommitted changes, or (None, why)."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=tree, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=tree, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, f"{tree} is not a git repository ({exc})"
    return head, dirty


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def card_blocked(card: Path, repo: Path = ROOT) -> str | None:
    """Why the model card may not go up, or None: the public build's scans, run on the card."""
    try:
        result = public_build.scan_text(repo, card, CARD_PATH)
    except (public_build.BuildError, OSError, KeyError, ValueError) as exc:
        return f"the model card could not be scanned: {exc}"
    if result["ok"]:
        return None
    hits = []
    for key, rows in result["blocking"].items():
        for r in rows:
            at = r.get("lines", r.get("line"))
            extra = r.get("term") or r.get("match") or r.get("id") or r.get("why") or ""
            hits.append(f"{key} line {at} {extra}".rstrip())
    return f"the model card has blocking scan hits: {'; '.join(hits[:20])}"


def model_files(model_dir: Path, card: Path = MODEL_CARD, repo: Path = ROOT) -> dict[str, Path]:
    """Path in the model repo to local file; the folder must match its own SHA256SUMS, and
    the card must pass the public build's scans."""
    if not model_dir.is_dir():
        raise UploadError(f"{model_dir} is not a folder")
    found = [p for p in sorted(model_dir.rglob("*")) if p.is_file()]
    if unwanted := [why for p in found if (why := _unwanted(p, model_dir))]:
        raise UploadError("; ".join(unwanted))
    files = {p.relative_to(model_dir).as_posix(): p for p in found}
    if SUMS not in files:
        raise UploadError(f"{model_dir} has no {SUMS}; build it with release/convert.py")
    listed = {}
    for line in files[SUMS].read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        listed[name.strip()] = digest
    unlisted = sorted(set(files) - set(listed) - {SUMS})
    if unlisted:
        raise UploadError(f"files not in {SUMS}: {', '.join(unlisted)}")
    for name, digest in listed.items():
        if name not in files:
            raise UploadError(f"{name} is in {SUMS} but not in the folder")
        if sha256(files[name]) != digest:
            raise UploadError(f"{name} does not match its {SUMS} line")
    if "README.md" in files:
        raise UploadError("the model folder has its own README.md; the card comes from release/")
    if "LICENSE" not in files:
        raise UploadError(f"{model_dir} has no LICENSE; build it again with release/convert.py")
    if card.is_file():
        if why := card_blocked(card, repo):
            raise UploadError(why)
        files["README.md"] = card
    return files


def dataset_files(tree: Path) -> dict[str, Path]:
    """Path in the dataset repo to local file: data/, the card, croissant.json, attribution."""
    data = tree / "data"
    if not data.is_dir():
        raise UploadError(f"{tree} has no data/ folder")
    found = [p for p in sorted(data.rglob("*")) if p.is_file()]
    if unwanted := [why for p in found if (why := _unwanted(p, tree))]:
        raise UploadError("; ".join(unwanted))
    files = {p.relative_to(tree).as_posix(): p for p in found}
    for name in DATASET_TOP:
        if (tree / name).is_file():
            files[name] = tree / name
    for name in ("README.md", "ATTRIBUTION.md"):
        if name not in files:
            raise UploadError(f"{tree} has no {name}")
    return files


def listing(repo_id: str, kind: str, files: dict[str, Path]) -> list[str]:
    total = sum(p.stat().st_size for p in files.values())
    lines = [f"{kind} {repo_id}: {len(files)} files, {total / 1e6:.1f} MB"]
    lines += [
        f"  {name}  ({path.stat().st_size:,} bytes, from {path})" for name, path in files.items()
    ]
    return lines


def report_ok(report: Path, tree: Path, repo: Path = ROOT) -> str | None:
    """Why the public build's scan does not clear this upload, or None when it does: the
    report must say ok, have scanned the dataset tree at its current commit with nothing on
    top, and have been built from this repository's current HEAD."""
    if not report.is_file():
        return f"{report} is missing; run python -m release.public_build first"
    body = json.loads(report.read_text(encoding="utf-8"))
    if not body.get("ok"):
        return f"{report} is not ok: {'; '.join(body.get('blocking_summary', []))}"
    scanned = ((body.get("trees") or {}).get(DATA_TREE) or {}).get("git", {}).get("commit")
    head, dirty = git_head(tree)
    if head is None:
        return dirty
    if head != scanned:
        return (
            f"the dataset tree {tree} is at commit {head}, the scan report scanned {scanned}; "
            "pass the tree the report describes, or build and scan again (just release)"
        )
    if dirty:
        return f"the dataset tree {tree} has changes on top of its commit: {dirty}"
    repo_head, _ = git_head(repo)
    if repo_head != body.get("repo_head"):
        return (
            f"this repository is at {repo_head}, the scan report was built at "
            f"{body.get('repo_head')}; build and scan again (just release)"
        )
    return None


def stage(files: dict[str, Path], where: Path) -> Path:
    """The listed files under their repo paths, hard-linked where the volume allows."""
    for name, src in files.items():
        out = where / name
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, out)
        except OSError:
            shutil.copy2(src, out)
    return where


def upload(repo_id: str, kind: str, files: dict[str, Path], message: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type=kind, private=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="upload-", dir=ROOT.parent) as tmp:
        info = api.upload_folder(
            repo_id=repo_id,
            repo_type=kind,
            folder_path=stage(files, Path(tmp)),
            allow_patterns=sorted(files),
            commit_message=message,
        )
    return str(info.commit_url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="release.upload_hf")
    parser.add_argument("--model-dir", type=Path, help="the folder release/convert.py wrote")
    parser.add_argument(
        "--dataset-tree", type=Path, required=True, help="the HakemBench tree the report scanned"
    )
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--go", action="store_true", help="upload; without it nothing is sent")
    args = parser.parse_args(argv)
    tree = args.dataset_tree
    plans = []
    try:
        if args.model_dir:
            plans.append((MODEL_REPO, "model", model_files(args.model_dir)))
        plans.append((DATASET_REPO, "dataset", dataset_files(tree)))
    except UploadError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    if not args.model_dir:
        print(f"model {MODEL_REPO}: no --model-dir given, nothing listed")
    for repo_id, kind, files in plans:
        print("\n".join(listing(repo_id, kind, files)))
    blocked = report_ok(args.report, tree)
    if not args.go:
        print(
            "dry run: nothing uploaded"
            + (f" (and --go would refuse: {blocked})" if blocked else "")
        )
        return 0
    if blocked:
        print(f"refused: {blocked}", file=sys.stderr)
        return 1
    for repo_id, kind, files in plans:
        print(f"{kind} {repo_id}: {upload(repo_id, kind, files, 'ufak AI release')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
