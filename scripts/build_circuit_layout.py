"""Precomputes a 2D layout for the real connectome subgraph in
data/circuit.npz, for the web UI's live circuit visualization
(server.py streams this once per connection; frontend/main.js draws it as
a static "wiring diagram" and animates spikes over it).

There's no real spatial position for these neurons in circuit_meta.json
(neuPrint gives connectivity, not the coordinates we pulled) -- see
build_circuit.py. So this computes a spectral layout instead: eigenvectors
of the graph Laplacian, which is a standard technique for embedding a large
sparse graph in 2D such that strongly-connected neurons end up near each
other. It's derived from the *real* synaptic connectivity, not an
arbitrary/decorative one, even though it's not the neurons' literal 3D
position in the fly's head.

Each role's nodes then get pulled slightly toward a per-role centroid so
the six input populations / output population / relay pool form
recognizable, separated regions on screen (a readability aid; the
within-role structure from the spectral embedding is preserved).

Usage:
    python scripts/build_circuit_layout.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MATRIX_PATH = os.path.join(DATA_DIR, "circuit.npz")
META_PATH = os.path.join(DATA_DIR, "circuit_meta.json")
OUT_PATH = os.path.join(DATA_DIR, "circuit_layout.json")

N_EDGES_FOR_RENDER = 9000  # cap so the browser isn't asked to draw 134k lines
ROLE_PULL = 0.35  # 0 = pure spectral layout, 1 = fully collapsed to role centroids


def main():
    W = sp.load_npz(MATRIX_PATH).tocsr()
    with open(META_PATH) as f:
        meta = json.load(f)
    nodes = meta["nodes"]
    n = len(nodes)
    print(f"{n} nodes, {W.nnz} directed synapse entries")

    # Symmetrize + compress dynamic range for a layout-friendly weight (a
    # few huge synapse counts shouldn't totally dominate the embedding).
    # Sign is excitatory/inhibitory, not relevant to "how connected" two
    # neurons are for layout purposes, so use magnitude.
    W_abs = W.copy()
    W_abs.data = np.abs(W_abs.data)
    A = W_abs.maximum(W_abs.T)
    A.data = np.log1p(A.data)

    print("computing spectral layout (this takes a bit for ~6k nodes)...")
    L = laplacian(A, normed=True).astype(np.float64)
    # This subgraph isn't fully connected (180 components, mostly one giant
    # one + many tiny fragments -- expected for an induced neuPrint
    # subgraph, not a parsing bug), which makes the Laplacian singular at
    # eigenvalue 0 with multiplicity > 1. Shift-invert mode (the usual way
    # to ask ARPACK for the smallest eigenvalues) needs to factor the
    # matrix at that singular point and fails outright, so this asks for
    # the smallest *algebraic* eigenvalues directly instead -- slower, but
    # doesn't require that factorization.
    vals, vecs = eigsh(L, k=4, which="SA")
    order = np.argsort(vals)
    xy = vecs[:, order[1:3]]  # eigenvectors 2 and 3

    # Raw eigenvector values are unusable directly: this graph has a lot of
    # low-degree "pendant" neurons (real -- an induced subgraph naturally
    # has many partially-captured neighbors, not a data bug), and those
    # dominate the smallest eigenvectors' *magnitude*, so a handful of them
    # land far out while the well-connected bulk of the network crams into
    # a tiny needle-thin region near the center. Rank-normalizing each axis
    # (every node's coordinate becomes its percentile position among all
    # nodes on that axis) fixes that: it keeps the real relative
    # ordering/clustering from the spectral embedding but guarantees an
    # even spread across the canvas instead of a hairball with spikes.
    def rank_normalize(v):
        ranks = np.argsort(np.argsort(v))
        return (ranks / (len(v) - 1)) * 2 - 1  # -> [-1, 1]

    xy = np.column_stack([rank_normalize(xy[:, 0]), rank_normalize(xy[:, 1])])

    roles = [m["role"] for m in nodes]
    role_list = sorted(set(roles))
    # deterministic centroid per role, spread around a circle
    centroids = {}
    for i, role in enumerate(role_list):
        angle = 2 * np.pi * i / len(role_list)
        centroids[role] = np.array([np.cos(angle), np.sin(angle)]) * 1.15

    pulled = np.array([
        (1 - ROLE_PULL) * xy[i] + ROLE_PULL * centroids[roles[i]]
        for i in range(n)
    ])

    layout_nodes = [
        {"index": int(m["index"]), "role": m["role"], "x": round(float(pulled[i, 0]), 4),
         "y": round(float(pulled[i, 1]), 4)}
        for i, m in enumerate(nodes)
    ]

    # Edge sample for rendering: strongest synapses first, capped. Upper
    # triangle only -- A is symmetric, so the full matrix has each
    # undirected pair twice and would otherwise waste half the budget
    # drawing the same line twice.
    coo = sp.triu(A, k=1).tocoo()
    keep = coo.data.argsort()[::-1][:N_EDGES_FOR_RENDER]
    edges = [
        {"source": int(coo.row[i]), "target": int(coo.col[i]), "weight": round(float(coo.data[i]), 3)}
        for i in keep
    ]
    print(f"kept {len(edges)} of {coo.nnz} unique undirected edges for rendering")

    with open(OUT_PATH, "w") as f:
        json.dump({"nodes": layout_nodes, "edges": edges}, f)
    print(f"saved layout to {OUT_PATH}")


if __name__ == "__main__":
    main()
