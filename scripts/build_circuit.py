"""Extracts a real MaleCNS connectome subgraph via the neuPrint API and
writes it in the schema flypoker.brain.LIFNetwork expects (data/circuit.npz
+ data/circuit_meta.json), replacing the synthetic placeholder.

Channel -> real neuron type assignment (see README for the honest framing:
these are invented sensory "ports", not literal biological correlates of
poker concepts):

    in:equity          LC4, LPLC1, LPLC2      (visual looming/threat detectors)
    in:pot_odds        LC9, LC12, LC17        (other visual projection neurons)
    in:spr             PFNa, PFNp_a, PFNp_b   (protocerebral bridge -> fan-shaped body)
    in:street          ER2_c, ER4d, ER5       (ellipsoid body ring neurons)
    in:position        FC1D, FC2B, FC2C       (fan-shaped body columnar neurons)
    in:opp_aggression  JO-EV1, JO-EV3, JO-FV  (Johnston's organ mechanosensory)
    out:action         DNg03, DNg06, DNg07, DNg08, DNge091, DNge094 (descending neurons)

Requires NEUPRINT_TOKEN in the environment / a .env file. Get one by
creating a free account at https://neuprint.janelia.org and visiting your
Account page.

Usage:
    python scripts/build_circuit.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import scipy.sparse as sp
from dotenv import load_dotenv
from neuprint import Client, NeuronCriteria as NC, fetch_adjacencies, fetch_neurons

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_MATRIX_PATH = os.path.join(DATA_DIR, "circuit.npz")
OUT_META_PATH = os.path.join(DATA_DIR, "circuit_meta.json")

NEUPRINT_SERVER = "https://neuprint.janelia.org"
NEUPRINT_DATASET = "male-cns:v1.0"

SEED_TYPES_BY_ROLE = {
    "in:equity": ["LC4", "LPLC1", "LPLC2"],
    "in:pot_odds": ["LC9", "LC12", "LC17"],
    "in:spr": ["PFNa", "PFNp_a", "PFNp_b"],
    "in:street": ["ER2_c", "ER4d", "ER5"],
    "in:position": ["FC1D", "FC2B", "FC2C"],
    "in:opp_aggression": ["JO-EV1", "JO-EV3", "JO-FV"],
    "out:action": ["DNg03", "DNg06", "DNg07", "DNg08", "DNge091", "DNge094"],
}

MIN_SYNAPSES = 8
MAX_NEIGHBORHOOD = 4000  # safety cap so the LIF sim stays fast
INHIBITORY_NT = {"gaba", "glutamate"}


def sign_from_nt(nt) -> int:
    return -1 if str(nt).lower() in INHIBITORY_NT else 1


def main():
    load_dotenv()
    token = os.environ["NEUPRINT_TOKEN"]
    client = Client(NEUPRINT_SERVER, dataset=NEUPRINT_DATASET, token=token)

    role_by_body: dict[int, str] = {}
    for role, types in SEED_TYPES_BY_ROLE.items():
        neurons, _ = fetch_neurons(NC(type=types), client=client)
        for bid in neurons["bodyId"]:
            role_by_body[int(bid)] = role
        print(f"{role}: {len(neurons)} seed neurons ({', '.join(types)})")

    seed_ids = list(role_by_body.keys())
    print(f"Total seed neurons: {len(seed_ids)}")

    print("Fetching 1-hop neighborhood (outgoing)...")
    _, out_edges = fetch_adjacencies(
        sources=NC(bodyId=seed_ids), targets=None, omit_rois=True,
        min_total_weight=MIN_SYNAPSES, properties=["type", "predictedNt"], client=client,
    )
    print("Fetching 1-hop neighborhood (incoming)...")
    _, in_edges = fetch_adjacencies(
        sources=None, targets=NC(bodyId=seed_ids), omit_rois=True,
        min_total_weight=MIN_SYNAPSES, properties=["type", "predictedNt"], client=client,
    )

    partner_ids = set(out_edges["bodyId_post"]) | set(in_edges["bodyId_pre"])
    partner_ids -= set(seed_ids)
    print(f"Direct partners (weight >= {MIN_SYNAPSES}): {len(partner_ids)}")

    if len(partner_ids) > MAX_NEIGHBORHOOD:
        weight_by_partner: dict[int, int] = {}
        for _, row in out_edges.iterrows():
            b = int(row["bodyId_post"])
            if b in partner_ids:
                weight_by_partner[b] = weight_by_partner.get(b, 0) + int(row["weight"])
        for _, row in in_edges.iterrows():
            b = int(row["bodyId_pre"])
            if b in partner_ids:
                weight_by_partner[b] = weight_by_partner.get(b, 0) + int(row["weight"])
        top = sorted(weight_by_partner.items(), key=lambda kv: -kv[1])[:MAX_NEIGHBORHOOD]
        partner_ids = {b for b, _ in top}
        print(f"Capped to top {MAX_NEIGHBORHOOD} partners by total synaptic weight")

    node_ids = np.array(sorted(set(seed_ids) | partner_ids), dtype=np.int64)
    index_of = {int(b): i for i, b in enumerate(node_ids)}
    n = len(node_ids)
    print(f"Total nodes: {n}")

    print("Fetching induced subgraph edges...")
    _, induced = fetch_adjacencies(
        sources=NC(bodyId=node_ids.tolist()), targets=NC(bodyId=node_ids.tolist()),
        omit_rois=True, min_total_weight=MIN_SYNAPSES, client=client,
    )
    print(f"Induced subgraph edges: {len(induced)}")

    print("Fetching predictedNt for all nodes (for excitatory/inhibitory polarity)...")
    node_info, _ = fetch_neurons(NC(bodyId=node_ids.tolist()), client=client)
    nt_by_body = dict(zip(node_info["bodyId"], node_info["predictedNt"]))

    rows = induced["bodyId_post"].map(index_of).to_numpy()
    cols = induced["bodyId_pre"].map(index_of).to_numpy()
    signs = induced["bodyId_pre"].map(lambda b: sign_from_nt(nt_by_body.get(int(b)))).to_numpy()
    data = induced["weight"].to_numpy() * signs

    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    os.makedirs(DATA_DIR, exist_ok=True)
    sp.save_npz(OUT_MATRIX_PATH, W)

    nodes_meta = [
        {"index": i, "bodyId": int(bid), "role": role_by_body.get(int(bid), "relay")}
        for i, bid in enumerate(node_ids)
    ]
    with open(OUT_META_PATH, "w") as f:
        json.dump({"source": "male-cns:v1.0", "min_synapses": MIN_SYNAPSES,
                    "seed_types_by_role": SEED_TYPES_BY_ROLE, "nodes": nodes_meta}, f)

    print(f"Saved {OUT_MATRIX_PATH} ({n} nodes, {W.nnz} directed edges)")
    print(f"Saved {OUT_META_PATH}")

    from collections import Counter
    print("\nRole counts:")
    for role, count in Counter(m["role"] for m in nodes_meta).most_common():
        print(f"  {role}: {count}")


if __name__ == "__main__":
    main()
