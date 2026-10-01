def test_units_split_whole_and_balanced_within_each_stratum():
    from bench.hakembench.common import unit_halves

    units = {f"a{i}": ("attack", 10 + i) for i in range(10)}
    units |= {f"b{i}": ("benign", 20) for i in range(8)}
    out = unit_halves(units)
    assert set(out) == set(units) and set(out.values()) == {"public", "private"}
    for stratum in ("attack", "benign"):
        held = {h: sum(units[u][1] for u in out if units[u][0] == stratum and out[u] == h)
                for h in ("public", "private")}  # fmt: skip
        assert abs(held["public"] - held["private"]) <= 20
    assert unit_halves(units) == out
