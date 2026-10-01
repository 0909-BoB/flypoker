# flypoker -- FastAPI/WebSocket poker server with a fly-connectome brain.
#
# Single-stage, no Node/npm step: the frontend is plain HTML/CSS/vanilla JS
# with Three.js already vendored into frontend/vendor/, served as static
# files straight from this image.
#
# Installed with `pip install -e .` (editable), matching local dev -- server.py
# locates frontend/, data/ and runs/ relative to its own file path, which only
# resolves correctly when the package stays inside this source tree instead of
# being copied into site-packages by a normal (non-editable) install.
FROM python:3.12-slim

WORKDIR /app

# System build deps for scipy/numpy wheels on slim images that don't ship one.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY frontend ./frontend
# data/ and runs/ are copied via .dockerignore's allowlist (see that file) --
# only the handful of files the server actually reads at runtime, not the
# 970MB training dataset or any locally-learned state.
COPY data ./data
COPY runs ./runs

# Not baked into the image, never needed at runtime (only scripts/build_*.py,
# run by hand on a developer machine, touch neuPrint) -- nothing here reads
# NEUPRINT_TOKEN or any other secret.
ENV PYTHONUNBUFFERED=1
EXPOSE 8420

# Each connection's game state (and the shared learning hub) lives in this
# one process's memory -- do not run this with multiple worker processes
# behind this same command, or players would land on different, inconsistent
# copies of the fly brain. Scale by running more *separate* instances behind
# a load balancer with sticky sessions instead, if that's ever needed.
CMD ["python", "-m", "flypoker.server"]
