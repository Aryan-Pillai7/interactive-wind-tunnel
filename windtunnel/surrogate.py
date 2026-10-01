"""onnxruntime loader for the exported model. (Owner: Aryan)

The app and evaluation use only this module, never torch.
"""


def load_surrogate(path):
    """Load models/model.onnx (see contract ONNX_*).

    Returns an object with:
        predict(inputs [3, NY, NX] float32 raw) -> fields [3, NY, NX]
        predict_batch(inputs [N, 3, NY, NX]) -> fields [N, 3, NY, NX]
    """
    raise NotImplementedError
