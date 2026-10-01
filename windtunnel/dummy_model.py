"""Stand-in model that honours the ONNX contract, for pipeline testing.

The prediction is masked uniform flow: u = U0 * (1 - mask), v = 0, p = 0.
It is useless as physics and deliberately so: it lets surrogate.py, the app
and evaluate.py run before a trained model exists, and gives Abhay a
reference ONNX file with the exact input/output names, shapes and dynamic
batch axis that his export must match.

    python -m windtunnel.dummy_model models/dummy.onnx   # needs the onnx package
"""

import sys

import numpy as np

from windtunnel import contract as C


class DummySurrogate:
    """numpy version, same interface as surrogate.load_surrogate()."""

    is_dummy = True
    path = None

    def predict_batch(self, inputs):
        inputs = np.asarray(inputs, np.float32)
        fluid = 1.0 - inputs[:, C.IN_MASK]
        out = np.zeros((inputs.shape[0], C.N_OUT, C.NY, C.NX), np.float32)
        out[:, C.OUT_U] = C.U0 * fluid
        return out

    def predict(self, inputs):
        return self.predict_batch(np.asarray(inputs)[None])[0]


def export_dummy_onnx(path):
    """Write the dummy as an ONNX graph (opset ONNX_OPSET). Needs onnx (train image)."""
    import onnx
    from onnx import TensorProto, helper

    def const(name, values, dtype=TensorProto.FLOAT):
        arr = np.asarray(values)
        return helper.make_node("Constant", [], [name], value=helper.make_tensor(
            name + "_t", dtype, arr.shape, arr.flatten().tolist()))

    nodes = [
        const("starts", [C.IN_MASK], TensorProto.INT64),
        const("ends", [C.IN_MASK + 1], TensorProto.INT64),
        const("axes", [1], TensorProto.INT64),
        const("one", 1.0),
        const("u0", C.U0),
        const("zero", 0.0),
        helper.make_node("Slice", [C.ONNX_INPUT, "starts", "ends", "axes"], ["mask"]),
        helper.make_node("Sub", ["one", "mask"], ["fluid"]),
        helper.make_node("Mul", ["fluid", "u0"], ["u"]),
        helper.make_node("Mul", ["mask", "zero"], ["v"]),
        helper.make_node("Mul", ["mask", "zero"], ["p"]),
        helper.make_node("Concat", ["u", "v", "p"], [C.ONNX_OUTPUT], axis=1),
    ]
    shape_in = [C.ONNX_BATCH_AXIS, C.N_IN, C.NY, C.NX]
    shape_out = [C.ONNX_BATCH_AXIS, C.N_OUT, C.NY, C.NX]
    graph = helper.make_graph(
        nodes, "dummy_windtunnel",
        [helper.make_tensor_value_info(C.ONNX_INPUT, TensorProto.FLOAT, shape_in)],
        [helper.make_tensor_value_info(C.ONNX_OUTPUT, TensorProto.FLOAT, shape_out)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", C.ONNX_OPSET)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


if __name__ == "__main__":
    export_dummy_onnx(sys.argv[1] if len(sys.argv) > 1 else "models/dummy.onnx")
