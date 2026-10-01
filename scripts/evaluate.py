"""Evaluate the surrogate and baselines on test and OOD. (Owner: Aryan)

Writes DATA_DIR/results/results.json and figures. Measures relative L2 per
channel, pressure drag/lift error, latency through onnxruntime, and the
solver-vs-surrogate speedup on this machine.
"""

if __name__ == "__main__":
    raise NotImplementedError("evaluation session")
