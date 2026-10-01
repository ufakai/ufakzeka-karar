"""What release/upload_hf.py would send where, on fake trees (no network)."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from release import upload_hf as U
from release.tests.test_public_build import MAIL, make_repo


def write(root: Path, files: dict[str, str]) -> Path:
    for name, body in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(body, encoding="utf-8")
    return root


def model_dir(tmp_path: Path, extra: dict[str, str] | None = None) -> Path:
    files = {
        "model.safetensors": "weights",
        "config.json": "{}",
        "karar.py": "# loader\n",
        "LICENSE": "Apache License\n",
    }
    root = write(tmp_path / "model", files)
    sums = "".join(
        f"{hashlib.sha256(body.encode()).hexdigest()}  {name}\n" for name, body in files.items()
    )
    write(root, {U.SUMS: sums, **(extra or {})})
    return root


def fake_repo(tmp_path: Path, allowed: list[dict] | None = None) -> Path:
    """The public build's fake repo, with its own name list (vendorx, always-blocking strictco)."""
    names = {
        "terms": ["vendorx", "strictco"],
        "always_blocking": ["strictco"],
        "contacts": ["hello@ufakai.com"],
        "allowed": allowed or [],
    }
    (tmp_path / "repo").mkdir(parents=True)
    return make_repo(tmp_path / "repo", extra={"config/release_names.json": json.dumps(names)})


def test_the_model_repo_gets_the_converted_folder_and_the_card(tmp_path):
    repo = fake_repo(tmp_path)
    card = write(tmp_path / "cards", {"README.md": "# card\n"}) / "README.md"
    files = U.model_files(model_dir(tmp_path), card, repo)
    assert sorted(files) == [
        "LICENSE",
        "README.md",
        U.SUMS,
        "config.json",
        "karar.py",
        "model.safetensors",
    ]
    assert files["README.md"] == card
    assert sorted(U.model_files(model_dir(tmp_path / "b"), tmp_path / "none.md", repo)) == [
        "LICENSE",
        U.SUMS,
        "config.json",
        "karar.py",
        "model.safetensors",
    ]


def test_bytecode_hidden_files_and_a_missing_licence_are_refused_with_the_fix(tmp_path):
    root = model_dir(tmp_path, {"__pycache__/karar.cpython-313.pyc": "x"})
    with pytest.raises(U.UploadError, match="Python bytecode") as refused:
        U.model_files(root)
    assert "rm -rf" in str(refused.value) and "PYTHONDONTWRITEBYTECODE=1" in str(refused.value)
    with pytest.raises(U.UploadError, match="hidden file"):
        U.model_files(model_dir(tmp_path / "b", {".DS_Store": "x"}))
    root = model_dir(tmp_path / "c")
    (root / "LICENSE").unlink()
    sums = [x for x in (root / U.SUMS).read_text().splitlines() if not x.endswith("LICENSE")]
    (root / U.SUMS).write_text("\n".join(sums) + "\n")
    with pytest.raises(U.UploadError, match="no LICENSE"):
        U.model_files(root)


def test_a_model_card_with_a_blocking_scan_hit_is_refused(tmp_path):
    board = {"path": U.CARD_PATH, "terms": ["vendorx", "strictco"], "category": "competitor"}
    repo = fake_repo(tmp_path, [board])
    folder = model_dir(tmp_path)
    # A board name the allow rule names passes; an always-blocking term never does.
    card = write(tmp_path / "ok", {"README.md": "# card\n\nBoard: vendorx-2 0.8.\n"})
    assert U.model_files(folder, card / "README.md", repo)["README.md"] == card / "README.md"
    for body, hit in [
        ("Labels by strictco-5.\n", "names_blocking line [1] strictco"),
        ("A range \u2014 here.\n", "em_dashes line 1"),
        ("A range 1\u20132.\n", "en_dashes line 1"),
        (f"Write to {MAIL}.\n", f"emails line 1 {MAIL}"),
        ("Held back: fedcba9876543210.\n", "restricted_ids line 1 spam-fedcba9876543210"),
        ("<!-- internal:start -->\nx\n", "internal_markers line 1"),
    ]:
        card = write(tmp_path / "bad", {"README.md": body}) / "README.md"
        with pytest.raises(U.UploadError, match="blocking scan hits") as refused:
            U.model_files(folder, card, repo)
        assert hit in str(refused.value), (body, str(refused.value))
    # Without its allow rule a board name blocks too, and a card that cannot be scanned is
    # refused rather than sent.
    card = write(tmp_path / "ok", {"README.md": "Board: vendorx-2.\n"}) / "README.md"
    with pytest.raises(U.UploadError, match="names_blocking"):
        U.model_files(folder, card, fake_repo(tmp_path / "strict"))
    with pytest.raises(U.UploadError, match="could not be scanned"):
        U.model_files(folder, card, tmp_path / "no-repo")


def test_a_model_folder_that_does_not_match_its_sums_is_refused(tmp_path):
    root = model_dir(tmp_path)
    (root / "config.json").write_text('{"changed": true}')
    with pytest.raises(U.UploadError, match="config.json does not match"):
        U.model_files(root)
    root = model_dir(tmp_path / "b", {"notes.txt": "stray"})
    with pytest.raises(U.UploadError, match="files not in SHA256SUMS: notes.txt"):
        U.model_files(root)


def test_the_dataset_repo_gets_data_card_croissant_and_attribution_never_code(tmp_path):
    tree = write(
        tmp_path / "hakembench",
        {
            "data/v1.0/public.jsonl": "{}\n",
            "data/v1.0/probes/slots.public.jsonl": "{}\n",
            "README.md": "# card\n",
            "croissant.json": "{}",
            "ATTRIBUTION.md": "# Attribution\n",
            "LICENSE": "Apache\n",
            "pyproject.toml": "[project]\n",
            "bench/harness/cli.py": "# code\n",
            ".venv/lib/x.py": "",
            ".git/HEAD": "ref\n",
        },
    )
    assert sorted(U.dataset_files(tree)) == [
        "ATTRIBUTION.md",
        "README.md",
        "croissant.json",
        "data/v1.0/probes/slots.public.jsonl",
        "data/v1.0/public.jsonl",
    ]
    (tree / "ATTRIBUTION.md").unlink()
    with pytest.raises(U.UploadError, match="no ATTRIBUTION.md"):
        U.dataset_files(tree)


def committed(tree: Path) -> str:
    config = ["-c", "user.name=a", "-c", "user.email=a@example.invalid"]
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tree, check=True)
    subprocess.run(["git", *config, "add", "-A"], cwd=tree, check=True)
    subprocess.run(["git", *config, "commit", "-qm", "x"], cwd=tree, check=True)
    return U.git_head(tree)[0]


def test_dry_run_lists_and_go_needs_a_clean_scan(tmp_path, capsys):
    tree = write(
        tmp_path / "hakembench",
        {"data/v1.0/public.jsonl": "{}\n", "README.md": "# c\n", "ATTRIBUTION.md": "# a\n"},
    )
    report = tmp_path / "scan_report.json"
    report.write_text(json.dumps({"ok": False, "blocking_summary": ["names_blocking: 1"]}))
    args = ["--dataset-tree", str(tree), "--report", str(report)]
    assert U.main(args) == 0
    out = capsys.readouterr().out
    assert "dataset ufakai/HakemBench: 3 files" in out and "data/v1.0/public.jsonl" in out
    assert "dry run: nothing uploaded" in out and "names_blocking: 1" in out
    assert U.main([*args, "--go"]) == 1
    with pytest.raises(SystemExit):
        U.main(["--report", str(report)])


def test_go_needs_the_scanned_dataset_tree_and_repo_head(tmp_path):
    tree = write(
        tmp_path / "hakembench",
        {"data/v1.0/public.jsonl": "{}\n", "README.md": "# c\n", "ATTRIBUTION.md": "# a\n"},
    )
    head = committed(tree)
    repo = write(tmp_path / "repo", {"a.txt": "a\n"})
    repo_head = committed(repo)
    report = tmp_path / "scan_report.json"

    def scanned(tree_commit: str, repo_commit: str) -> Path:
        body = {
            "ok": True,
            "repo_head": repo_commit,
            "trees": {"hakembench": {"git": {"commit": tree_commit}}},
        }
        report.write_text(json.dumps(body))
        return report

    assert U.report_ok(scanned(head, repo_head), tree, repo) is None
    why = U.report_ok(scanned("0" * 40, repo_head), tree, repo)
    assert why and "scan report scanned " + "0" * 40 in why
    why = U.report_ok(scanned(head, "1" * 40), tree, repo)
    assert why and "scan report was built at " + "1" * 40 in why
    (tree / "data/v1.0/public.jsonl").write_text("{}\n{}\n")
    why = U.report_ok(scanned(head, repo_head), tree, repo)
    assert why and "changes on top of its commit" in why
    assert "not a git repository" in U.report_ok(report, tmp_path / "nowhere", repo)
