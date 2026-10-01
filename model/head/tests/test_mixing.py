"""Round 1's mix: each task capped, the same rows every time, nothing lost below the cap."""

import json

from model.head.mixing import mix


def test_each_task_is_capped_and_the_mix_is_repeatable(tmp_path):
    rows = [{"task": "big", "i": i} for i in range(50)] + [{"task": "small", "i": i}
                                                            for i in range(5)]  # fmt: skip
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    first = mix([path], cap=10, seed=1)
    tasks = [json.loads(x)["task"] for x in first]
    assert tasks.count("big") == 10 and tasks.count("small") == 5
    assert mix([path], cap=10, seed=1) == first
    assert len(set(first)) == len(first)


def test_unseen_rows_are_the_capped_tasks_rows_the_mix_left_out(tmp_path):
    import json

    from model.head.mixing import mix, unseen

    path = tmp_path / "train.jsonl"
    lines = [json.dumps({"task": "big", "i": i}) for i in range(50)]
    lines += [json.dumps({"task": "small", "i": i}) for i in range(5)]
    path.write_text("\n".join(lines) + "\n")
    drawn = set(mix([path], cap=20))
    left = unseen([path], cap=20, per_task=10)
    assert len(left) == 10 and not drawn & set(left)
    assert all(json.loads(line)["task"] == "big" for line in left)


def test_every_template_of_one_text_lands_in_the_same_fold():
    from model.head.mixing import FOLDS, fold_of

    lines = [json.dumps({"task": f"t{i % 7}", "state": f"metin {i % 40}"}) for i in range(400)]
    by_text = {}
    for line in lines:
        by_text.setdefault(json.loads(line)["state"], set()).add(fold_of(line))
    assert all(len(f) == 1 for f in by_text.values())
    assert {f for s in by_text.values() for f in s} == set(range(FOLDS))


def _rows(task, count):
    return [json.dumps({"task": task, "row_id": f"{task}-{i}", "state": f"{task} metin {i}"})
            for i in range(count)]  # fmt: skip


def test_a_task_fraction_halves_one_task_by_stratum_and_leaves_the_rest():
    from model.head.mixing import take_fractions

    lines = _rows("r4", 40) + _rows("other", 25)
    # Four kinds of ten rows, each over two contexts.
    strata = {"r4": {f"r4 metin {i}": f"K{i % 4}|c{i % 8 // 4}" for i in range(40)}}
    cut = take_fractions(lines, {"r4": 0.5}, strata)
    kept = [json.loads(x) for x in cut if json.loads(x)["task"] == "r4"]
    assert len(kept) == 20
    kinds = {}
    for row in kept:
        label = strata["r4"][row["state"]]
        kinds[label] = kinds.get(label, 0) + 1
    # Eight cells of five rows: each keeps two or three.
    assert len(kinds) == 8 and set(kinds.values()) <= {2, 3}
    by_kind = {}
    for label, n in kinds.items():
        by_kind[label.split("|")[0]] = by_kind.get(label.split("|")[0], 0) + n
    assert set(by_kind.values()) == {5}
    assert [x for x in cut if json.loads(x)["task"] == "other"] == _rows("other", 25)
    assert take_fractions(lines, {"r4": 0.5}, strata) == cut
    assert take_fractions(lines, {"r4": 0.5}, strata, seed=7) != cut


def test_a_zero_fraction_drops_the_task_and_a_bad_fraction_is_refused():
    import pytest

    from model.head.mixing import take_fractions

    lines = _rows("r4", 10) + _rows("other", 3)
    assert take_fractions(lines, {"r4": 0.0}) == _rows("other", 3)
    assert take_fractions(lines, {"r4": 1.0}) == lines
    with pytest.raises(ValueError, match="outside"):
        take_fractions(lines, {"r4": 1.5})


def test_strata_come_from_the_writers_records(tmp_path):
    from model.head.mixing import load_strata, strata_from_records

    path = tmp_path / "kept.json"
    path.write_text(json.dumps([{"text": "a", "kind": "K1", "context": "banka", "x": 1}]))
    assert strata_from_records(path, ("kind", "context")) == {"a": "K1|banka"}
    folder = tmp_path / "strata"
    folder.mkdir()
    (folder / "r4.json").write_text(json.dumps({"a": "K1|banka"}))
    assert load_strata(folder) == {"r4": {"a": "K1|banka"}}
    assert load_strata(tmp_path / "missing") == {}


def test_the_r5_specs_stay_out_of_default_runs_and_carry_their_filter():
    import pytest

    pytest.importorskip("modal")
    from model.head.round1 import R5_GPU, specs

    r5 = {s["name"]: s for s in specs("x") if s["name"].startswith("r5")}
    assert sorted(r5) == [f"r5{v}-base-s{s}" for v in "ab" for s in (1, 2, 3)]
    assert {s["task_fraction"]["prompt_injection-r4"] for s in r5.values()} == {0.5, 0.0}
    assert all(s["gpu"] == R5_GPU == "H200" for s in r5.values())
    assert all(s["prior_exclude"] == ["prompt_injection-conv"] for s in r5.values())
    r4 = next(s for s in specs("x") if s["name"] == "r4-base-s1")
    extra = {"name", "seed", "task_fraction", "gpu", "timeout", "prior_exclude"}
    assert {k: v for k, v in r5["r5a-base-s1"].items() if k not in extra} == {
        k: v for k, v in r4.items() if k not in extra}  # fmt: skip
    assert all("gpu" not in s for s in specs("x") if not s["name"].startswith("r5"))
