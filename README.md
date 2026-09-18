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
  hand_history_parser.py  parses the real IRC Poker Database into (state, action) examples
  pretrain.py      supervised behavior-cloning warm-start from real hand histories
  train.py         self-play REINFORCE training loop (optionally warm-started from pretrain.py)
  play.py          CLI to watch a match, or play against the fly yourself
scripts/
  build_circuit.py  pulls the real MaleCNS subgraph via neuPrint (needs NEUPRINT_TOKEN)
data/               circuit.npz + circuit_meta.json (gitignored, built locally)
                    hand_histories/IRCdata.tgz (gitignored, downloaded -- see below)
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

### Hand history data (optional, for pretraining)

`pretrain.py` warm-starts the decoder from real players' actions instead of
starting `train.py`'s self-play REINFORCE from random weights. The data is
the **IRC Poker Database**: over 10 million hands logged from IRC poker
channels in 1995-2001, maintained by the University of Alberta Computer
Poker Research Group and a long-standing reference dataset in poker AI
research. There's no formal license attached -- the source page's only
usage statement is "may be useful to poker programming researchers and
hobbyists" -- so this is used here in that spirit (research/hobbyist), not
under a claimed open-source license.

```bash
.venv/bin/python scripts/fetch_hand_history_dataset.py
```

It's a ~970MB download. `hand_history_parser.py`'s module docstring has the
full raw format (verified against the real files, not just the source
site's documentation) and, importantly, explains a real structural
limitation: a folded hand never reveals hole cards, so this data can only
ever supply *non-fold* examples. `pretrain.py` accounts for that -- it
never updates the decoder's FOLD weights at all, so it can only make
non-fold calibration better, never worse (see "Tuning / known rough edges"
below for why that specifically mattered here).

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

Optionally warm-start that from real hand histories first (see "Hand
history data" above for getting `IRCdata.tgz`):

```bash
.venv/bin/python -m flypoker.pretrain --hands 10000 --out runs/decoder.pretrained.npz
.venv/bin/python -m flypoker.train --hands 6000 \
    --decoder-in runs/decoder.pretrained.npz --out runs/decoder.npz
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

Raising isn't limited to two fixed presets: clicking "加注" opens a
bet-sizing panel (quick 1/2-pot / pot / 2x-pot / all-in buttons, plus a
free-drag slider and a number input) like a normal poker client, and sends
the exact chosen total to the engine (`poker.py`'s `_betting_round` accepts
an explicit "raise to" amount from any agent, not just a preset).

When both players are all-in with no more decisions left, the remaining
streets are dealt out one at a time with a short pause between each (see
`HeadsUpHand.play`'s `on_street_dealt` callback), rather than jumping
straight to the showdown -- the brain panel keeps updating through the
run-out too (cosmetic only: no real decision is being made, so it doesn't
touch the decoder).

## Tuning / known rough edges

- REINFORCE with a single hand-level reward per decision is high-variance,
  and this decoder is small (a linear layer over ~90 spike-count features).
  In practice it doesn't converge to a smoothly-calibrated policy -- it
  reliably converges to an *overconfident* one, and which specific verdict
  it locks onto is sensitive to training details. Across several retraining
  attempts while chasing this, the same shape of failure showed up
  repeatedly, just pointed at different actions:
  - **Fold-collapse:** `train.py` originally used one global scalar baseline
    (a running average of `payoff_bb` across every decision ever made).
    That silently miscredits variance -- a *correct* call with AA gets
    blamed whenever that hand later loses to a bad runout, while folding is
    never blamed for anything because it ends the trajectory immediately
    and never sees the runout. Over thousands of hands this reliably taught
    the decoder to fold to almost any bet regardless of equity (verified:
    folded pocket aces to a raise ~99.9% of the time). Fix: bucket the
    baseline by the equity at decision time, so a call with AA is judged
    against other strong-hand outcomes, not diluted by the mass of weak
    preflop folds.
  - **Input-independent collapse:** even after the baseline fix, longer runs
    or a higher-variance training opponent could still drive the decoder to
    an overconfident, near-saturated softmax -- at which point *whichever*
    action it saturated on (fold, or in one run, call) got picked almost
    identically regardless of hand strength, bet size, or street. Entropy
    regularization alone doesn't reliably prevent this: its gradient
    vanishes exactly as probabilities approach 0/1, i.e. right when it's
    needed most. Two backstops now guard against it:
    1. `decoder.py`'s `reinforce_update` applies weight decay to *both* `W`
       and `b` (decaying only `W` just moved the problem -- an undecayed
       bias became the new escape hatch the decoder used to ignore input
       entirely), which bounds how extreme the logits can get.
    2. `SoftmaxDecoder.act` mixes a hard `ACTION_FLOOR` (12%) of uniform
       probability into every decision, so no legal action's probability
       can reach exactly 0 or 1 no matter what the trained weights say.
       This is a permanent, training-independent guarantee -- whatever a
       future retrain converges to, the game can never lock into "always
       folds" (or "always calls") again.
  - **Self-play still overwrites a good starting point.** Warm-starting
    `train.py` from `pretrain.py`'s behavior-cloned weights (see "Hand
    history data" above) was expected to sidestep this -- give REINFORCE a
    sensible, equity-responsive starting policy instead of random noise.
    It didn't: 6000 hands of self-play against `EquityBot` reliably steered
    even that good starting point back into the same input-independent
    fold-collapse (verified: after warm-starting, AA facing a raise still
    ended up folding ~90% of the time regardless of bet size or street,
    barely above the hard floor). So `runs/decoder.npz` shipped here is
    `pretrain.py`'s output *without* any further self-play -- genuinely
    calibrated from real players' actions on the four non-fold actions
    (verified responsive across varied equity/bet-size scenarios, e.g.
    favoring RAISE_BIG with AA facing an all-in shove rather than folding),
    but its FOLD weights are literally untouched since birth (pretraining
    is designed to never move them -- see `hand_history_parser.py`), so
    fold happens to sit near the floor rate everywhere rather than tracking
    equity. That's an accepted, disclosed trade-off, not an oversight: it
    reliably avoids both collapse modes above, at the cost of folding too
    rarely rather than too often.
  - This points at `train.py`'s self-play loop itself -- not just
    initialization -- as the actual remaining problem. Likely needs
    per-decision (not per-hand) credit assignment, e.g. a small
    value-function baseline, or a training opponent less exploitable than
    `EquityBot` by a fold-everything-except-the-nuts strategy, rather than
    more hyperparameter tweaking on vanilla REINFORCE.
- `train.py` logs a running "bluff rate" (aggressive actions taken with
  Monte Carlo equity < 0.35) and a "fold_rate(eq>0.65,faced bet)" (folding a
  strong hand facing a bet) as rough signals of policy health, alongside
  average bb/hand won. Both are cumulative over the whole run, so they lag
  behind the current policy and can look fine while the live policy has
  already collapsed (or vice versa) -- spot-check actual behavior directly,
  e.g. with the scenario probes used while debugging this, rather than
  trusting the trend line alone.
- The LIF simulator resets membrane state before every decision (see
  `FlyAgent.reset_brain_each_decision`), so each decision is an independent
  ~150-step simulation rather than one continuous hand-long trace. That's a
  simplification, not a biological claim.
