# results/step9

The HakemBench v1.0 board on the open set, the probe statistics and the result rows
behind them.

- board_public.json is the published board. Its `inputs` list the files it was computed
  from under their original paths: each model's part B rows and the human answers were
  read at the time from the lab's private archive (paths under results/private/), where they
  sat beside the answers on the 574 left-out items. Those paths are a record of the run; they
  are not in this repository. probes_public.json lists its inputs the same way. Its part B
  probe rows are not in this repository, so the probe statistics cannot be recomputed from
  public files; the board can. probes_public.json's inputs.monolingual list was split on
  spaces when it was recorded; the models the English probe does not apply to are those
  whose english.status says "not applicable".
- release/public_board.py recomputes the board from public files only: each model's part A
  rows (runs/<model>-public.jsonl), its part B rows on open items
  (runs/<model>-open-partb.jsonl), the surface baseline's rows on open items
  (runs/surface-baseline-open.jsonl) and the human answers on the open questions of the
  human check (owner_check_open_answers.json). `just board-public` runs it and writes
  board_public_check.json, and `python -m release.public_board check` compares that file
  with board_public.json number by number.
