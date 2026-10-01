# --- base: lean image for solver, generator, eval, app, tests (no torch) ---
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    MPLBACKEND=Agg
WORKDIR /app
COPY requirements-base.txt .
RUN pip install -r requirements-base.txt
CMD ["bash"]

# --- train: base + torch + onnx (Abhay only; Aryan never builds this) ---
FROM base AS train
# CPU wheels by default; set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 for CUDA.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --index-url ${TORCH_INDEX_URL} torch==2.7.1
COPY requirements-train.txt .
RUN pip install -r requirements-train.txt
