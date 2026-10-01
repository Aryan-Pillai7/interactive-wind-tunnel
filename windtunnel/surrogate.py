"""onnxruntime loader for the exported model. (Owner: Aryan)

The app and evaluation use only this module, never torch.
"""

import os

import numpy as np

from windtunnel import contract as C


class Surrogate:
    def __init__(self, path, threads=None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.path = path
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        ins = [i.name for i in self.session.get_inputs()]
        outs = [o.name for o in self.session.get_outputs()]
        if ins != [C.ONNX_INPUT] or outs != [C.ONNX_OUTPUT]:
            raise ValueError(f"{path}: expected input {C.ONNX_INPUT!r} and output "
                             f"{C.ONNX_OUTPUT!r}, got {ins} -> {outs}")
        self.is_dummy = False

    def predict_batch(self, inputs):
        """inputs [N, 3, NY, NX] raw float32 -> fields [N, 3, NY, NX]."""
        x = np.ascontiguousarray(inputs, np.float32)
        if x.ndim != 4 or x.shape[1:] != (C.N_IN, C.NY, C.NX):
            raise ValueError(f"expected [N, {C.N_IN}, {C.NY}, {C.NX}], got {x.shape}")
        return self.session.run([C.ONNX_OUTPUT], {C.ONNX_INPUT: x})[0]

    def predict(self, inputs):
        """inputs [3, NY, NX] -> fields [3, NY, NX]."""
        return self.predict_batch(np.asarray(inputs)[None])[0]


def load_surrogate(path=C.MODEL_PATH, allow_dummy=False, threads=None):
    """Load models/model.onnx (see contract ONNX_*).

    Returns an object with:
        predict(inputs [3, NY, NX] float32 raw) -> fields [3, NY, NX]
        predict_batch(inputs [N, 3, NY, NX]) -> fields [N, 3, NY, NX]
        is_dummy: True if the file was missing and allow_dummy gave the
            masked-uniform-flow DummySurrogate instead.
    """
    if not os.path.exists(path):
        if not allow_dummy:
            raise FileNotFoundError(path)
        from windtunnel.dummy_model import DummySurrogate

        return DummySurrogate()
    return Surrogate(path, threads=threads)
