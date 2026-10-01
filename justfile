# ufakzeka-karar commands. Run `just` to list them.

set shell := ["bash", "-euo", "pipefail", "-c"]

# list the recipes
default:
    @just --list --unsorted

# manifest and allow-list gate
check-licenses:
    uv run python -m data.manifest

# Compared with results/step9/board_public.json; about 5 CPU-hours at 2,000 draws. The
# board never overwrites its file: pass a new out path.
# the public board again, from public files only
board-public workers="8" out="results/step9/board_public_check.json":
    uv run --group instrument python -m bench.board $(uv run python -m release.public_board args --out {{ out }}) --workers {{ workers }}
    uv run python -m release.public_board check {{ out }}

# unit tests (no model weights, no network)
test:
    uv run pytest

# adapter tests that need torch (fake model, no download)
test-local:
    uv run --group local pytest bench/tests/test_local.py

# everything, including tests that need torch and the instrument group
test-instrument:
    uv run --group local --group instrument pytest

# lint and format check
lint:
    uv run ruff check .
    uv run ruff format --check .

# what CI runs. test-instrument, not test: the metrics, the selection rule
# and the training loop are the product, and a green run that skipped them
# would protect nothing. Linux resolves torch from the CPU index, so this
# pulls no CUDA wheels.
# lint, every test and the licence gate, as CI runs them
ci: lint test-instrument check-licenses

# step 0 smoke: one question, one local model, one results row
smoke *args:
    uv run --group local python -m bench.harness.cli smoke {{ args }}
