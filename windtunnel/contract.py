"""Shared contract for both halves of the project.

Every constant that crosses the boundary between the data/eval half (Aryan)
and the model half (Abhay) lives here. Import from this module; never
hard-code these values elsewhere. Changing anything here is a contract change
and needs agreement from both sides.

Conventions
-----------
- Arrays are channel-first: [C, H, W] = [C, NY, NX] = [C, 64, 128];
  batches are [N, C, H, W].
- Row index j is y (j = 0 is the bottom wall), column index i is x
  (i = 0 is the inlet). Flow goes left to right, so u > 0 is downstream and
  v > 0 is upward. Plot with origin="lower".
- Everything is in lattice units (cell size 1, time step 1) unless stated.
- u, v: velocity in lattice units (the inlet is u = U0, v = 0).
- p: gauge pressure coefficient, p = (rho - rho_ref) / 3 / (0.5 * U0**2),
  with rho_ref = mean density over the fluid cells of the outlet column.
- Inside the obstacle all three target channels (u, v, p) are exactly 0.
- SDF: Euclidean distance in cells to the obstacle surface, positive in the
  fluid, negative inside the obstacle.
- re_norm = (Re - RE_MIN) / (RE_MAX - RE_MIN), in [0, 1].
"""

import os

# --- Grid -------------------------------------------------------------------
NX = 128  # cells along the flow (width)
NY = 64   # cells across the flow (height); rows 0 and NY-1 are solid walls
GRID_SHAPE = (NY, NX)  # (H, W)

# Obstacle placement. CY sits exactly between the two walls so symmetric
# shapes at zero angle have zero lift by construction.
OBSTACLE_CX = 32.0
OBSTACLE_CY = (NY - 1) / 2.0  # 31.5

# Characteristic length D = obstacle height across the flow, in cells.
# D_MAX = 16 keeps blockage D / (NY - 2) under ~26%. Blockage is stored per
# sample in meta.json.
D_MIN = 10.0
D_MAX = 16.0
CHANNEL_HEIGHT = NY - 2  # fluid rows between the two wall rows

# --- Flow -------------------------------------------------------------------
U0 = 0.05  # inlet velocity, lattice units
RE_MIN = 10.0
RE_MAX = 40.0
TAU_MIN = 0.51  # samples with tau = 3 * nu + 0.5 below this are rejected

# Steady-state criterion: relative change of u below CONV_TOL over
# CONV_EVERY steps. MAX_ITERS is the cap; non-converged samples are dropped.
CONV_TOL = 1e-5
CONV_EVERY = 100
MAX_ITERS = 40_000  # provisional; to be confirmed when the solver is verified

# --- Shapes -----------------------------------------------------------------
# NACA airfoils are deferred until the solver is shown to resolve them cleanly.
TRAIN_KINDS = ("circle", "ellipse", "rectangle")
OOD_KINDS = ("triangle",)  # never used for training, validation or tuning

# --- Channels ---------------------------------------------------------------
INPUT_CHANNELS = ("sdf", "mask", "re_norm")
TARGET_CHANNELS = ("u", "v", "p")
N_IN = len(INPUT_CHANNELS)
N_OUT = len(TARGET_CHANNELS)
IN_SDF, IN_MASK, IN_RE = 0, 1, 2
OUT_U, OUT_V, OUT_P = 0, 1, 2

# --- Splits -----------------------------------------------------------------
# Fractions of the in-distribution samples. The OOD set is separate.
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_KEYS = ("idx_train", "idx_val", "idx_test", "idx_ood")
# The split is stratified: the fractions above are applied, with a seeded
# shuffle, within each (kind, Re bucket). Bucket b covers
# [RE_BUCKET_EDGES[b], RE_BUCKET_EDGES[b+1]).
RE_BUCKET_EDGES = (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0001)

# --- Files ------------------------------------------------------------------
# DATA_DIR is a bind mount (see .env); inside the containers it is /data.
DATA_DIR = os.environ.get("DATA_DIR", "/data")
DATASET_FILE = "dataset.npz"
META_FILE = "meta.json"
NORM_FILE = "norm.json"
RESULTS_DIR = "results"            # relative to DATA_DIR
CHECKPOINT_DIR = "checkpoints"     # relative to DATA_DIR
TRAIN_LOG_FILE = "train_log.csv"   # in DATA_DIR/results
RESULTS_FILE = "results.json"      # in DATA_DIR/results

# Committed model handoff, relative to the repo root.
MODEL_PATH = "models/model.onnx"
MODEL_CARD_PATH = "models/model_card.json"
MODEL_MAX_COMMIT_MB = 20

# Keys of dataset.npz.
NPZ_KEYS = ("inputs", "targets", "cd_total", "cl_total") + SPLIT_KEYS

# --- ONNX handoff -----------------------------------------------------------
ONNX_OPSET = 17
ONNX_INPUT = "inputs"   # float32 [N, 3, 64, 128], RAW (unnormalised) inputs
ONNX_OUTPUT = "fields"  # float32 [N, 3, 64, 128], physical units, masked
ONNX_BATCH_AXIS = "batch"
TORCH_ONNX_MAX_ABS_DIFF = 1e-4  # on the full validation set


def re_to_norm(re):
    """Map a Reynolds number in [RE_MIN, RE_MAX] to [0, 1]."""
    return (re - RE_MIN) / (RE_MAX - RE_MIN)


def norm_to_re(re_norm):
    """Inverse of re_to_norm."""
    return RE_MIN + re_norm * (RE_MAX - RE_MIN)


def data_path(*parts):
    """Path under DATA_DIR, e.g. data_path(RESULTS_DIR, RESULTS_FILE)."""
    return os.path.join(DATA_DIR, *parts)
