"""A small fixed scoring model and test set, built in code so the tests need no stored model file.

The model is a logistic regression over four features, written with onnx.helper: one output,
``probabilities``, with the probability of class 0 and of class 1 per row."""

import io

FEATURES = ["feat1", "feat2", "feat3", "feat4"]
WEIGHTS = [0.08, -0.5, 0.03, 1.2]
BIAS = -1.5


def scoring_model_bytes() -> bytes:
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, len(FEATURES)])
    out = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, 2])
    nodes = [
        helper.make_node("MatMul", ["X", "W"], ["xw"]),
        helper.make_node("Add", ["xw", "B"], ["z"]),
        helper.make_node("Sigmoid", ["z"], ["p"]),
        helper.make_node("Sub", ["one", "p"], ["q"]),
        helper.make_node("Concat", ["q", "p"], ["probabilities"], axis=1),
    ]
    inits = [
        helper.make_tensor("W", TensorProto.FLOAT, [len(FEATURES), 1], WEIGHTS),
        helper.make_tensor("B", TensorProto.FLOAT, [1], [BIAS]),
        helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0]),
    ]
    graph = helper.make_graph(nodes, "scoring", [x], [out], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def scoring_rows() -> list[dict]:
    """24 rows: four features and the true class ``target``."""
    rows = []
    for i in range(24):
        row = {"feat1": float((i * 7) % 40), "feat2": float((i * 3) % 10) / 2,
               "feat3": float((i * 11) % 50), "feat4": i % 2}
        # the true class mostly follows the model, with a few rows against it
        z = sum(w * row[f] for w, f in zip(WEIGHTS, FEATURES)) + BIAS
        row["target"] = int(z > 0) if i % 5 else int(z <= 0)
        rows.append(row)
    return rows


def scoring_dataset_csv() -> bytes:
    import pandas as pd

    buf = io.StringIO()
    pd.DataFrame(scoring_rows()).to_csv(buf, index=False)
    return buf.getvalue().encode()


def config() -> dict:
    return {
        "frequency": "",
        "window_size": "",
        "date_feature": None,
        "target_feature": "target",
        "features": [{"name": f, "min": 0.0, "max": 100.0, "type": "Float"} for f in FEATURES[:3]]
        + [{"name": "feat4", "min": 0.0, "max": 1.0, "type": "Integer"},
           {"name": "target", "min": 0.0, "max": 1.0, "type": "Integer"}],
    }


def measures(results: dict) -> dict:
    """The measures of a run without their time stamps, for comparing two runs."""
    return {name: [{k: v for k, v in m.items() if k != "time"} for m in values]
            for name, values in results.items()}
