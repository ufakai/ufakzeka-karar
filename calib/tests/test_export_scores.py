"""The step 7 scores on a hand-checked sample."""

from calib.export import scores


def test_scores_on_a_small_sample():
    meta = [{"task": "t", "keys": ["a", "b"], "target": {"a": 1.0, "b": 0.0}},
            {"task": "t", "keys": ["a", "b"], "target": {"a": 0.0, "b": 1.0}}]  # fmt: skip
    got = scores([[0.9, 0.1], [0.6, 0.4]], meta)
    assert got["accuracy"] == 0.5
    assert abs(got["brier"] - (0.02 + 0.72) / 2) < 1e-9
    assert got["questions"] == 2
    # One class right, one wrong: F1 for "a" is 2/3, for "b" 0.
    assert abs(got["macro_f1_mean_over_tasks"] - (2 / 3) / 2) < 1e-4


def test_the_scorer_product_is_found_by_its_width_by_one_weight():
    import numpy as np
    import pytest

    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    from calib.export import head_matmuls

    def weight(name, shape):
        return numpy_helper.from_array(np.zeros(shape, dtype=np.float32), name)

    nodes = [helper.make_node("MatMul", ["x", "w_body"], ["h"], name="body"),
             helper.make_node("MatMul", ["h", "w_score"], ["s"], name="score"),
             helper.make_node("MatMul", ["h", "w_abstain"], ["a"], name="abstain"),
             helper.make_node("MatMul", ["h", "h2"], ["q"], name="activation_product")]  # fmt: skip
    graph = helper.make_graph(
        nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4]),
                     helper.make_tensor_value_info("h2", TensorProto.FLOAT, [4, 1])],
        [helper.make_tensor_value_info(o, TensorProto.FLOAT, None) for o in ("s", "a", "q")],
        [weight("w_body", (4, 4)), weight("w_score", (4, 1)), weight("w_abstain", (4, 1))],
    )  # fmt: skip
    # Both width-by-one products are found; an activation-by-activation product is not.
    assert sorted(head_matmuls(onnx.helper.make_model(graph))) == ["abstain", "score"]


def test_down_projections_are_the_products_that_take_in_the_widest_width():
    import numpy as np
    import pytest

    pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    from calib.export import down_projections

    shapes = {"q": (8, 8), "k": (8, 2), "gate": (8, 32), "down": (32, 8), "score": (8, 1)}
    nodes = [helper.make_node("MatMul", ["x", f"w_{n}"], [f"y_{n}"], name=n) for n in shapes]
    inits = [numpy_helper.from_array(np.zeros(s, dtype=np.float32), f"w_{n}")
             for n, s in shapes.items()]  # fmt: skip
    graph = helper.make_graph(
        nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info(f"y_{n}", TensorProto.FLOAT, None) for n in shapes], inits,
    )  # fmt: skip
    assert down_projections(helper.make_model(graph)) == ["down"]
