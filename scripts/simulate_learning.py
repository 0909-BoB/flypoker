"""Does the online learner stay stable? Plays simulated hands against the
equity bot (random stack depths) with LearningHub updating after every hand,
in a throwaway directory, and prints behavior in windows so drift toward
over-folding / over-shoving would show.

    .venv/bin/python scripts/simulate_learning.py 1500
"""
import os
import random
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["FLYPOKER_LEARN_DIR"] = tempfile.mkdtemp()

from flypoker import train
from flypoker.agents import EquityBot, FlyAgent, feature_dim
from flypoker.brain import LIFNetwork
from flypoker.decoder import SoftmaxDecoder
from flypoker.learning import LearningHub
from flypoker.poker import Action

n_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
hub = LearningHub()
net = LIFNetwork()
dec = SoftmaxDecoder(n_features=feature_dim(net))
fly = FlyAgent(net, decoder=dec, seed=5)
bot = EquityBot(seed=6)
rng = random.Random(21)

win = {"bb": [], "fold": [], "shove": [], "strong": [0, 0]}
for i in range(1, n_hands + 1):
    dec.W, dec.b = hub.snapshot()
    payoff, _, _ = train.run_hand(fly, bot, rng, button_seat=i % 2)
    bb = payoff / train.BIG_BLIND
    hub.learn_from_hand(fly.trajectory, bb)
    hub.hands += 1
    win["bb"].append(bb)
    for st in fly.trajectory:
        win["fold"].append(st["action_idx"] == Action.FOLD)
        win["shove"].append(st["action_idx"] == Action.ALL_IN)
        if st["to_call"] > 0 and st["equity"] > 0.65:
            win["strong"][0] += 1
            win["strong"][1] += st["action_idx"] == Action.FOLD
    if i % 250 == 0:
        s = hub.status()
        print(f"hands {i:5d}: avg_bb={np.mean(win['bb']):+.2f}  fold={np.mean(win['fold']):.0%}  "
              f"shove={np.mean(win['shove']):.0%}  strong-hand fold={win['strong'][1] / max(win['strong'][0], 1):.0%}  "
              f"drift={s['drift']:.3f}  updates={s['updates']} skipped={s['skipped']} healthy={s['healthy']}", flush=True)
        win = {"bb": [], "fold": [], "shove": [], "strong": [0, 0]}
