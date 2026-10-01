"""Adds the stack-aware input channels to an already-built circuit without
re-querying neuPrint: relabels a deterministic subset (every k-th neuron by
bodyId) of two existing real input populations as new channels. Wiring
(circuit.npz) is untouched, so these are still real MaleCNS neurons.

    in:pot_odds (LC9/LC12/LC17)  -> in:pot_odds + in:commit   (half each)
    in:spr      (PFNa/PFNp_*)    -> in:spr + in:depth + in:rel_stack (thirds)

Idempotent: does nothing if in:commit already exists. Rerun
scripts/build_circuit_layout.py afterwards to refresh the UI layout.
"""
import json
import os

META = os.path.join(os.path.dirname(__file__), "..", "data", "circuit_meta.json")

with open(META) as f:
    meta = json.load(f)

if any(n["role"] == "in:commit" for n in meta["nodes"]):
    print("already split; nothing to do")
    raise SystemExit

def split(source_role, new_roles):
    members = sorted((n for n in meta["nodes"] if n["role"] == source_role), key=lambda n: n["bodyId"])
    k = len(new_roles) + 1
    for i, node in enumerate(members):
        slot = i % k
        if slot > 0:
            node["role"] = new_roles[slot - 1]

split("in:pot_odds", ["in:commit"])
split("in:spr", ["in:depth", "in:rel_stack"])
meta["stack_channels_note"] = "in:commit/in:depth/in:rel_stack relabeled from pot_odds/spr populations by scripts/split_input_roles.py"

with open(META, "w") as f:
    json.dump(meta, f)

from collections import Counter
for role, c in sorted(Counter(n["role"] for n in meta["nodes"]).items()):
    print(f"  {role}: {c}")
