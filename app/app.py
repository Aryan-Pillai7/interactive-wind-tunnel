"""Streamlit virtual wind tunnel. (Owner: Aryan)

Shape picker, sliders (size, aspect, angle, Re), and two modes:
- LBM solver: runs solve() (capped at MAX_ITERS) and draws its field.
- Neural surrogate: models/model.onnx through onnxruntime, with a "run real
  solver" button for a side-by-side comparison and error map.
Streamlines, pressure map, derived metrics and the compute time are shown.

Run: docker compose up app  ->  http://localhost:8501
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from windtunnel import contract as C  # noqa: E402
from windtunnel import geometry, lbm, metrics, viz  # noqa: E402
from windtunnel.dataset import build_inputs  # noqa: E402
from windtunnel.surrogate import load_surrogate  # noqa: E402

st.set_page_config(page_title="Virtual wind tunnel", layout="wide")


@st.cache_resource
def get_model():
    return load_surrogate(C.MODEL_PATH, allow_dummy=True)


@st.cache_data(show_spinner=False, max_entries=32)
def run_solver(kind, size, aspect, angle, re):
    mask = geometry.make_mask(kind, size, aspect, angle)
    return lbm.solve(mask, re)


model = get_model()

st.title("Virtual wind tunnel")
st.caption("A neural surrogate for steady 2D flow in a **simulated** 128 x 64 channel "
           "(solid top and bottom walls, flow left to right). Ground truth is our own "
           "lattice-Boltzmann solver. 2D only; Re 10-40 only, where the flow is steady.")
with st.sidebar:
    modes = ["LBM solver", "Neural surrogate"]
    mode = st.radio("Flow from", modes, index=0 if model.is_dummy else 1)
    if mode == modes[1] and model.is_dummy:
        st.warning("No trained model at models/model.onnx: the surrogate is a **dummy** "
                   "(uniform flow around the shape).")
    st.header("Obstacle")
    kind = st.selectbox("Shape", list(C.TRAIN_KINDS) + ["triangle (out of distribution)"])
    kind = kind.split()[0]
    size = st.slider("Height across the flow D (cells)", C.D_MIN, C.D_MAX, 12.0, 0.5)
    aspect, angle = 1.0, 0.0
    if kind != "circle":
        lo, hi = {"ellipse": (1.2, 2.5), "rectangle": (1.0, 3.0), "triangle": (0.8, 1.6)}[kind]
        aspect = st.slider("Aspect (length / thickness)", lo, hi, (lo + hi) / 2, 0.05)
        angle = st.slider("Angle (degrees, counter-clockwise)", -90.0, 90.0, 0.0, 1.0)
    re = st.slider("Reynolds number", C.RE_MIN, C.RE_MAX, 20.0, 0.5)
    if kind == "triangle":
        st.info("Triangles were never seen in training: expect larger errors.")

mask = geometry.make_mask(kind, size, aspect, angle)
d = geometry.obstacle_height(mask)
if mode == modes[0]:
    with st.spinner("Solving..."):
        res = run_solver(kind, size, aspect, angle, re)
    pred = np.stack([res["u"], res["v"], res["p"]])
    ms = res["seconds"] * 1000
    source = f"LBM solver ({res['iterations']} steps)"
    if not res["converged"]:
        st.info(f"Solver stopped at the {res['iterations']}-step cap before reaching steady "
                f"state (last relative change {res['residual']:.1e} per {C.CONV_EVERY} steps): "
                "the near field is developed, the far wake may still be adjusting.")
else:
    x = build_inputs(mask, re)
    t0 = time.perf_counter()
    pred = model.predict(x)
    ms = (time.perf_counter() - t0) * 1000
    source = "dummy surrogate" if model.is_dummy else "neural surrogate"
pm = metrics.field_metrics(pred, mask)

col_fig, col_num = st.columns([3, 1])
with col_fig:
    fig = viz.plot_fields(pred, mask, f"{source}: {kind}, D = {d:.0f}, Re = {re:.1f}")
    st.pyplot(fig)
    plt.close(fig)
with col_num:
    st.subheader("From the predicted field")
    st.metric("Pressure drag C_D,p", f"{pm['cd_p']:.3f}")
    st.metric("Pressure lift C_L,p", f"{pm['cl_p']:.3f}")
    st.metric("Peak stagnation pressure", f"{pm['stagnation_p']:.2f}")
    st.metric("Wake vorticity |w| D/U0", f"{pm['wake_vorticity']:.3f}")
    st.metric("Compute time" if mode == modes[0] else "Inference time",
              f"{ms / 1000:.2f} s" if ms >= 1000 else f"{ms:.1f} ms")
    st.caption(f"Blockage D/H = {d / C.CHANNEL_HEIGHT:.0%}. Pressure is a coefficient "
               "relative to the outlet; forces are pressure-only surface integrals.")

if mode == modes[0]:
    st.stop()

st.divider()
st.subheader("Check against the real solver")
st.caption("Runs the lattice-Boltzmann solver for this exact shape (a few seconds on CPU).")
if st.button("Run real solver"):
    with st.spinner("Solving..."):
        res = run_solver(kind, size, aspect, angle, re)
    if not np.isfinite(res["u"]).all():
        st.error("The solver diverged for this case; no ground truth.")
    else:
        true = np.stack([res["u"], res["v"], res["p"]])
        tm = metrics.field_metrics(true, mask)
        rel = metrics.relative_l2(pred, true, mask)
        fig = viz.plot_comparison(pred, true, mask)
        st.pyplot(fig)
        plt.close(fig)
        c = st.columns(5)
        c[0].metric("Velocity rel. L2 error", f"{rel['vel']:.1%}")
        c[1].metric("Pressure rel. L2 error", f"{rel['p']:.1%}")
        c[2].metric("C_D,p (solver field)", f"{tm['cd_p']:.3f}",
                    f"{pm['cd_p'] - tm['cd_p']:+.3f} surrogate", delta_color="off")
        c[3].metric("Solver time", f"{res['seconds']:.1f} s", f"{res['iterations']} steps",
                    delta_color="off")
        c[4].metric("Speed-up", f"{res['seconds'] * 1000 / max(ms, 1e-3):,.0f}x")
        st.caption(f"Solver total C_D (momentum exchange, pressure + viscous): "
                   f"{res['cd_total']:.3f}. Not comparable with the pressure-only C_D,p.")
