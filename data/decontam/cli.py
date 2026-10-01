"""Decontamination commands: fetch the reference sets, build the indexes, check a row file.

    python -m data.decontam.cli fetch [NAME ...]
    python -m data.decontam.cli build --index data/built/decontam/index.pkl
    python -m data.decontam.cli check --rows ROWS.jsonl --index data/built/decontam/index.pkl \\
        --out REPORT.json [--write-clean [CLEAN.jsonl]]

`check` reads a jsonl of TrainingRow and flags a row when

- more than half of its tokens sit in 8-grams shared with the references, or
  a text under eight tokens equals a reference whole ("ngram");
- the row by itself covers more than half of some reference's tokens, which
  catches a short test item pasted inside a long row ("covers_reference");
- a reference is a MinHash near-duplicate at an estimated Jaccard of 0.8 or
  more ("minhash").

The row's state and its question instructions are checked as two separate
texts, so our own question wording cannot dilute a copied state.

Set-level rule (Tülu 3, arXiv 2411.15124): a reference item is overlapped when
the rows together cover more than half of its tokens, or equal it whole, or
hold a near-duplicate of it. If more than 2 percent of any set's items are
overlapped, `check` exits 1. The rule is applied to the rows as given; to
confirm a cleaned file passes, run `check` on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from data.decontam import references
from data.decontam.ngrams import (
    JACCARD_THRESHOLD,
    NGRAM,
    OVERLAP_THRESHOLD,
    MinHashLSH,
    NgramIndex,
    gram_hashes,
    minhash_signature,
    set_name,
    tokens,
    whole_hash,
)
from schema.rows import TrainingRow, text_of

# More than this share of a reference set's items overlapped fails the build.
SET_LIMIT = 0.02
# Per-row lists of matched references are cut to this many in the report.
MAX_LISTED = 50


def lsh_path_for(index_path: Path) -> Path:
    return index_path.with_name(index_path.stem + ".lsh.pkl")


def default_clean_path(rows_path: Path) -> Path:
    return rows_path.with_name(rows_path.stem + ".clean.jsonl")


def row_texts(row: TrainingRow) -> list[str]:
    """The texts of a row that could carry test material: the state and the instructions."""
    texts = [text_of(row.state), text_of(row.question.instructions)]
    return [text for text in texts if text.strip()]


def _check_row(
    row: TrainingRow, index: NgramIndex, lsh: MinHashLSH
) -> tuple[dict[str, Any], set[int], set[int], set[str]]:
    fraction = 0.0
    matched: set[str] = set()
    covers: set[str] = set()
    near: dict[str, float] = {}
    grams: set[int] = set()
    wholes: set[int] = set()
    for text in row_texts(row):
        part_fraction, part_refs = index.overlap(text)
        fraction = max(fraction, part_fraction)
        matched.update(part_refs)
        toks = tokens(text)
        hashes = gram_hashes(toks, index.n)
        grams.update(hashes)
        wholes.add(whole_hash(toks))
        if hashes:
            covers.update(index.covered_references(set(hashes), part_refs))
        signature = minhash_signature(text)
        if signature is not None:
            for ref_id, jaccard in lsh.query(signature):
                near[ref_id] = max(near.get(ref_id, 0.0), jaccard)
    reasons = []
    if fraction > OVERLAP_THRESHOLD:
        reasons.append("ngram")
    if covers:
        reasons.append("covers_reference")
    if near:
        reasons.append("minhash")
    matched_sorted = sorted(matched)
    record = {
        "row_id": row.row_id,
        "ngram_fraction": round(fraction, 4),
        "matched_ref_count": len(matched_sorted),
        "matched_refs": matched_sorted[:MAX_LISTED],
        "covers_refs": sorted(covers)[:MAX_LISTED],
        "minhash": [[ref_id, round(j, 4)] for ref_id, j in sorted(near.items())][:MAX_LISTED],
        "flagged": bool(reasons),
        "reasons": reasons,
    }
    return record, grams, wholes, set(near)


def check(
    rows_path: Path,
    index: NgramIndex,
    lsh: MinHashLSH,
    clean_path: Path | None = None,
) -> dict[str, Any]:
    """Check every row and the set-level rule; write the clean file when asked."""
    lines = [line for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    per_row: list[dict[str, Any]] = []
    train_grams: set[int] = set()
    train_wholes: set[int] = set()
    near_refs: set[str] = set()
    clean: list[str] = []
    for line in lines:
        row = TrainingRow.model_validate_json(line)
        record, grams, wholes, near = _check_row(row, index, lsh)
        per_row.append(record)
        train_grams |= grams
        train_wholes |= wholes
        near_refs |= near
        if not record["flagged"]:
            clean.append(line)

    coverage = index.reference_coverage(train_grams, train_wholes)
    overlapped = {
        ref_id
        for ref_id, share in zip(index.ref_ids, coverage, strict=True)
        if share > OVERLAP_THRESHOLD
    } | near_refs
    counts = Counter(set_name(ref_id) for ref_id in overlapped)
    sets = {}
    for name, size in sorted(index.set_sizes().items()):
        share = counts[name] / size if size else 0.0
        sets[name] = {
            "items": size,
            "overlapped": counts[name],
            "share": round(share, 6),
            "over_limit": share > SET_LIMIT,
        }
    if clean_path is not None:
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        clean_path.write_text("".join(line + "\n" for line in clean), encoding="utf-8")
    return {
        "rows_file": str(rows_path),
        "rows": len(per_row),
        "flagged": sum(record["flagged"] for record in per_row),
        "clean_file": str(clean_path) if clean_path is not None else None,
        "clean_rows": len(clean),
        "thresholds": {
            "ngram": NGRAM,
            "row_overlap_above": OVERLAP_THRESHOLD,
            "minhash_jaccard_at_least": JACCARD_THRESHOLD,
            "set_share_above": SET_LIMIT,
        },
        "sets": sets,
        "contaminated_sets": [name for name, entry in sets.items() if entry["over_limit"]],
        "per_row": per_row,
    }


def _cmd_fetch(args: argparse.Namespace) -> int:
    names = args.names or list(references.REFERENCE_SETS)
    unknown = [name for name in names if name not in references.REFERENCE_SETS]
    if unknown:
        print(f"unknown reference sets: {unknown}", file=sys.stderr)
        return 2
    for name in names:
        path = references.fetch(name, args.cache)
        print(f"{name}: {path}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    absent = references.missing(args.cache)
    if absent and not args.allow_missing:
        print(f"not fetched yet: {absent}; fetch them or pass --allow-missing", file=sys.stderr)
        return 2
    index = references.build_index(args.cache)
    index.save(args.index)
    lsh_path = args.lsh or lsh_path_for(args.index)
    references.build_lsh(args.cache).save(lsh_path)
    print(f"index: {len(index)} references -> {args.index}; lsh -> {lsh_path}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    lsh_path = args.lsh or lsh_path_for(args.index)
    if not lsh_path.is_file():
        print(f"no MinHash index at {lsh_path}; run `build` first", file=sys.stderr)
        return 2
    index = NgramIndex.load(args.index)
    lsh = MinHashLSH.load(lsh_path)
    clean_path = None
    if args.write_clean is not None:
        clean_path = Path(args.write_clean) if args.write_clean else default_clean_path(args.rows)
    report = check(args.rows, index, lsh, clean_path)
    report["index"] = str(args.index)
    report["lsh"] = str(lsh_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"{report['rows']} rows, {report['flagged']} flagged; "
        f"contaminated sets: {report['contaminated_sets'] or 'none'}"
    )
    return 1 if report["contaminated_sets"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m data.decontam.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="download or copy reference sets into the cache")
    fetch.add_argument("names", nargs="*", help="registered set names; all when omitted")
    fetch.add_argument("--cache", type=Path, default=references.CACHE_DIR)
    fetch.set_defaults(run=_cmd_fetch)

    build = commands.add_parser("build", help="build the 8-gram index and the MinHash LSH")
    build.add_argument("--cache", type=Path, default=references.CACHE_DIR)
    build.add_argument("--index", type=Path, required=True)
    build.add_argument("--lsh", type=Path, default=None)
    build.add_argument("--allow-missing", action="store_true")
    build.set_defaults(run=_cmd_build)

    check_cmd = commands.add_parser("check", help="check a TrainingRow jsonl against the index")
    check_cmd.add_argument("--rows", type=Path, required=True)
    check_cmd.add_argument("--index", type=Path, required=True)
    check_cmd.add_argument("--lsh", type=Path, default=None)
    check_cmd.add_argument("--out", type=Path, required=True)
    check_cmd.add_argument(
        "--write-clean",
        nargs="?",
        const="",
        default=None,
        help="write the unflagged rows; defaults to ROWS with a .clean.jsonl suffix",
    )
    check_cmd.set_defaults(run=_cmd_check)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
