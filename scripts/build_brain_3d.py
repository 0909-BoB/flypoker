"""Builds the 3D brain assets for the web UI from the same neuPrint dataset
(male-cns:v1.0) the circuit was extracted from, so the mesh and the neuron
positions share one coordinate frame:

  frontend/brain.stl            decimated brain surface (central brain + both
                                optic lobes, real ROI meshes), centered and
                                scaled so its longest side is 100 units
  data/circuit_positions.json   {"positions": {node_index: [x, y, z]}} in the
                                same normalized frame, one entry per circuit
                                node. Soma location when the neuron has one;
                                otherwise the centroid of its dominant ROI
                                (plus a little jitter so they don't stack).

Needs NEUPRINT_TOKEN (.env) and `pip install fast-simplification`.

    .venv/bin/python scripts/build_brain_3d.py
"""
from __future__ import annotations

import json
import os
import struct

import fast_simplification
import numpy as np
from dotenv import load_dotenv
from neuprint import Client, NeuronCriteria as NC, fetch_neurons

ROOT = os.path.join(os.path.dirname(__file__), "..")
load_dotenv(os.path.join(ROOT, ".env"))
META_PATH = os.path.join(ROOT, "data", "circuit_meta.json")
POS_OUT = os.path.join(ROOT, "data", "circuit_positions.json")
STL_OUT = os.path.join(ROOT, "frontend", "brain.stl")

BRAIN_MESHES = ["CentralBrain", "Optic(L)", "Optic(R)"]
TARGET_FACES = 90_000
SIZE = 100.0


def parse_obj(data: bytes):
    verts, faces = [], []
    for line in data.decode().splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(p.split("/")[0]) - 1 for p in line.split()[1:4]])
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def write_binary_stl(path, verts, faces):
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    with open(path, "wb") as f:
        f.write(b"flypoker brain (male-cns:v1.0 ROI meshes)".ljust(80, b" "))
        f.write(struct.pack("<I", len(faces)))
        for n, t in zip(normals, tri):
            f.write(struct.pack("<12fH", *n, *t.reshape(-1), 0))


def main():
    client = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=os.environ["NEUPRINT_TOKEN"])

    all_v, all_f, offset = [], [], 0
    for roi in BRAIN_MESHES:
        v, f = parse_obj(client.fetch_roi_mesh(roi))
        print(f"{roi}: {len(v)} verts, {len(f)} faces")
        all_v.append(v)
        all_f.append(f + offset)
        offset += len(v)
    verts, faces = np.vstack(all_v), np.vstack(all_f)

    center = (verts.min(0) + verts.max(0)) / 2
    scale = SIZE / (verts.max(0) - verts.min(0)).max()
    dv, df = fast_simplification.simplify(verts.astype(np.float32), faces.astype(np.int32),
                                          target_reduction=1 - TARGET_FACES / len(faces))
    dv = (dv.astype(np.float64) - center) * scale
    write_binary_stl(STL_OUT, dv, df)
    print(f"wrote {STL_OUT}: {len(df)} faces ({os.path.getsize(STL_OUT) / 1e6:.1f} MB)")

    nodes = json.load(open(META_PATH))["nodes"]
    ids = [n["bodyId"] for n in nodes]
    soma: dict[int, list[float]] = {}
    dominant_roi: dict[int, str] = {}
    for i in range(0, len(ids), 400):
        df_n, roi_counts = fetch_neurons(NC(bodyId=ids[i:i + 400]))
        for _, row in df_n.iterrows():
            if row["somaLocation"] is not None:
                soma[int(row["bodyId"])] = list(row["somaLocation"])
        if len(roi_counts):
            best = roi_counts.assign(w=roi_counts["pre"] + roi_counts["post"]).sort_values("w", ascending=False)
            for bid, grp in best.groupby("bodyId"):
                dominant_roi[int(bid)] = grp.iloc[0]["roi"]
        print(f"neurons {min(i + 400, len(ids))}/{len(ids)}", flush=True)

    roi_centroid: dict[str, np.ndarray] = {}
    def centroid(roi):
        if roi not in roi_centroid:
            try:
                roi_centroid[roi] = parse_obj(client.fetch_roi_mesh(roi))[0].mean(axis=0)
            except Exception:
                roi_centroid[roi] = center
        return roi_centroid[roi]

    rng = np.random.default_rng(0)
    positions, n_soma = {}, 0
    for n in nodes:
        bid = n["bodyId"]
        if bid in soma:
            p = np.array(soma[bid], dtype=np.float64)
            n_soma += 1
        else:
            p = centroid(dominant_roi.get(bid, "CentralBrain")) + rng.normal(scale=600, size=3)
        positions[n["index"]] = [round(float(x), 3) for x in (p - center) * scale]
    print(f"positions: {n_soma} from soma, {len(nodes) - n_soma} from ROI centroid")
    with open(POS_OUT, "w") as f:
        json.dump({"positions": positions}, f)
    print(f"wrote {POS_OUT}")


if __name__ == "__main__":
    main()
