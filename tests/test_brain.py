import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flypoker import synthetic_circuit
from flypoker.brain import LIFNetwork


def test_synthetic_circuit_responds_to_input(tmp_path):
    synthetic_circuit.build(seed=1, out_dir=str(tmp_path))
    net = LIFNetwork(
        matrix_path=str(tmp_path / "circuit.npz"),
        meta_path=str(tmp_path / "circuit_meta.json"),
    )
    assert net.n > 0
    assert set(net.input_roles()) == set(synthetic_circuit.INPUT_CHANNELS)
    assert set(net.output_roles()) == set(synthetic_circuit.OUTPUT_CHANNELS)

    net.reset()
    baseline = net.run({}, n_steps=100)
    net.reset()
    driven = net.run({r: 5.0 for r in net.input_roles()}, n_steps=100)
    assert sum(driven.values()) >= sum(baseline.values())


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        test_synthetic_circuit_responds_to_input(Path(d))
    print("all tests passed")
