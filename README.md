# flypoker

A subgraph of a real fly brain (MaleCNS v1.0 connectome, Janelia/Google, 2026)
plays heads-up no-limit Texas Hold'em, via a spiking neural network simulation
with a small trainable readout layer on top.

## Honest framing, up front

A fly's brain has no concept of pot odds, stack-to-pot ratio, or bluffing —
it evolved for flight control, vision, and courtship, not card games. This
project does **not** claim otherwise. What it actually does:

- **The connectome wiring is real.** `scripts/build_circuit.py` pulls an
  induced subgraph of real neurons and real synapse weights from the MaleCNS
  dataset via the neuPrint API, centered on a handful of neuron types (visual
  projection neurons, central-complex neurons, mechanosensory neurons,
  descending neurons).
- **The sensory/motor mapping is invented.** There's no "pot odds neuron."
  Game-state numbers (hand equity, pot odds, stack-to-pot ratio, street,
  position, opponent aggression) are injected as current into six named
  neuron populations, exactly the same spirit as the "fly plays Doom"
  projects mapping pixel brightness to photoreceptors — a consistent but
  arbitrary sensory code, not a biological claim.
- **The bluffing/stack-awareness comes from a trained decoder, not the
  connectome.** The connectome's spiking dynamics are fixed and never
  updated. A small softmax layer reads the spike pattern of the output
  ("descending neuron") population and is trained via REINFORCE across
  thousands of self-play hands against a baseline bot. *That* layer is what
  learns to fold weak hands, raise strong ones, and occasionally bluff —
  the connectome is a large, fixed, biologically-real feature extractor
  sitting underneath it, not a card player in its own right.

If you want a system that *only* uses unmodified real connectome dynamics
with no trainable layer (closer to the "fly reflex" spirit of the
flappy-fly project), skip `train.py` and read `play.py --decoder` behavior
with an untrained (random-weight) decoder — expect it to play close to
randomly, same as a real fly would.

`brain.py`'s leaky integrate-and-fire update rule is adapted from
[naginagiyev/flappy-fly](https://github.com/naginagiyev/flappy-fly)'s
`backend/brain.py`, generalized from one visual-input/one DN-output
population to arbitrary named input/output populations.

## Project layout

```
src/flypoker/
  poker.py        heads-up NLHE engine: hand evaluation, betting, discretized actions
  brain.py         leaky integrate-and-fire simulator over the connectome subgraph
  synthetic_circuit.py   placeholder random circuit (same file schema) for dev without a token
  encoder.py       game state -> current injection into named input populations
  decoder.py       trainable softmax readout: spike pattern -> action
  agents.py        FlyAgent (brain+decoder) and EquityBot (rule-based baseline opponent)
  train.py         self-play REINFORCE training loop
  play.py          CLI to watch a match, or play against the fly yourself
scripts/
  build_circuit.py  pulls the real MaleCNS subgraph via neuPrint (needs NEUPRINT_TOKEN)
data/               circuit.npz + circuit_meta.json (gitignored, built locally)
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

### Circuit data

Either:

- **Synthetic placeholder** (no token needed, for development):
  ```bash
  .venv/bin/python -m flypoker.synthetic_circuit
  ```
- **Real MaleCNS subgraph** (needs a free neuPrint account + API token from
  https://neuprint.janelia.org, saved as `NEUPRINT_TOKEN` in `.env`):
  ```bash
  .venv/bin/pip install neuprint-python
  .venv/bin/python scripts/build_circuit.py
  ```
  This pulls ~2,200 seed neurons (see the type list in `build_circuit.py`'s
  docstring) plus their strongest direct synaptic partners, capped at 4,000
  partners so the simulation stays fast, and writes `data/circuit.npz` /
  `data/circuit_meta.json`.

## Running it

Watch the fly (untrained, random decoder) play a few hands against the
baseline bot:

```bash
.venv/bin/python -m flypoker.play --hands 5
```

Train the decoder (a few thousand hands takes a few minutes on the
synthetic/real subgraph sizes used here):

```bash
.venv/bin/python -m flypoker.train --hands 5000 --out runs/decoder.npz
```

Watch the trained fly play:

```bash
.venv/bin/python -m flypoker.play --hands 10 --decoder runs/decoder.npz
```

Play against it yourself (terminal):

```bash
.venv/bin/python -m flypoker.play --human --decoder runs/decoder.npz
```

### Web UI

A browser poker table with a live neuron-activity panel (FastAPI + WebSocket
backend, plain HTML/CSS/JS frontend, same pattern as flappy-fly's
backend/frontend split but served from one process):

```bash
.venv/bin/python -m flypoker.server
```

Then open http://127.0.0.1:8420. It automatically loads `runs/decoder.npz`
if present (falls back to an untrained decoder otherwise). Each browser tab
gets its own hand of heads-up NLHE against the fly, dealt in a background
thread; the side panel shows real-time spike counts per named neuron
population as the fly decides each action.

## Tuning / known rough edges

- REINFORCE with a single hand-level reward per decision is high-variance;
  early training can collapse into an overly aggressive policy before it
  settles. Lowering the learning rate in `decoder.py` (`reinforce_update`'s
  `lr` argument) or training for more hands helps.
- `train.py` logs a running "bluff rate" (aggressive actions taken with
  Monte Carlo equity < 0.35) as a rough signal that mixed-strategy bluffing
  is emerging, alongside average bb/hand won.
- The LIF simulator resets membrane state before every decision (see
  `FlyAgent.reset_brain_each_decision`), so each decision is an independent
  ~150-step simulation rather than one continuous hand-long trace. That's a
  simplification, not a biological claim.
