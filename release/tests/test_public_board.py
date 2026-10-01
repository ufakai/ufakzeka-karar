import json

from release import public_board


def test_args_list_every_model_in_the_board_order(tmp_path):
    args = public_board.board_args(tmp_path / "out.json")
    rows = args[1 : args.index("--splits")]
    assert rows[0] == "results/step9/runs/ufakzeka-karar-r5-public.jsonl"
    assert rows[1] == "results/step9/runs/ufakzeka-karar-r5-open-partb.jsonl"
    assert rows[2] == "results/step9/runs/surface-baseline-open.jsonl"
    assert len(rows) == 2 * len(public_board.MODELS) - 1
    assert not any(a.startswith("results/private") for a in args)
    assert "--only-listed" in args
    assert args[-2:] == ["--out", str(tmp_path / "out.json")]


def _board(tmp_path, name, value, **extra):
    body = {
        "created_utc": extra.get("created", "a"),
        "git_commit": "c",
        "inputs": {"rows": [name]},
        "ranking": ["m"],
        "models": {"m": {"composite": {"value": value, "interval": [0.1, 0.9]}}},
    }
    path = tmp_path / name
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_check_ignores_the_run_record_only(tmp_path):
    a = _board(tmp_path, "a.json", 0.5, created="x")
    b = _board(tmp_path, "b.json", 0.5, created="y")
    assert public_board.check(a, b)["identical"]
    c = _board(tmp_path, "c.json", 0.5 + 1e-15)
    result = public_board.check(c, b)
    assert not result["identical"]
    assert result["differences"] == 1 and result["all_differences_are_floats"]
    assert 0 < result["largest_float_difference"] < 1e-14
    assert result["same_within_relative_1e-12"]
    d = _board(tmp_path, "d.json", 0.51)
    assert not public_board.check(d, b)["same_within_relative_1e-12"]
