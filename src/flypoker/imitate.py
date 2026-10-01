"""Distills a stack-aware teacher policy into the decoder, reading the real
fly brain's spike features. REINFORCE against a noisy bot kept collapsing
(over-folding, over-shoving) once stack depth varied, so instead: sample many
varied game states, run each through the (fixed) connectome, and fit the
decoder's softmax to the teacher's action *distribution* for that state. The
brain still does all the perceiving -- only the linear readout is fit.

Teacher (teacher_probs): fold/call by risk-adjusted equity (a call that eats
more of my stack needs more equity than pot odds alone), raise more with
strength, push-or-fold when short.

Usage:
    .venv/bin/python -m flypoker.imitate --examples 16000 --out runs/decoder.v3.npz
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import random
import time

import numpy as np

from .agents import N_SIM_STEPS, brain_features
from .brain import LIFNetwork
from .decoder import N_ACTIONS, SoftmaxDecoder
from .encoder import encode
from .poker import Action, Observation, STREETS, make_deck

BB = 20
N_BOARD = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def legal_mask_for(to_call: int, my_stack: int, big_blind: int) -> np.ndarray:
    mask = np.ones(N_ACTIONS)
    if to_call == 0:
        mask[Action.FOLD] = 0.0
    if to_call >= my_stack or my_stack <= big_blind:
        mask[Action.RAISE_SMALL] = 0.0
        mask[Action.RAISE_BIG] = 0.0
    return mask


def teacher_probs(equity: float, pot: int, to_call: int, my_stack: int, big_blind: int) -> np.ndarray:
    depth = my_stack / big_blind
    p = np.zeros(N_ACTIONS)
    if to_call == 0:
        r = min(max((equity - 0.5) / 0.35, 0.0), 1.0)
        p[Action.ALL_IN] = 0.02 + 0.10 * r ** 3
        p[Action.RAISE_BIG] = 0.04 + 0.34 * r ** 2
        p[Action.RAISE_SMALL] = 0.08 + 0.30 * r
        if depth < 12:  # short: push or check, sized raises make no sense
            p[Action.ALL_IN] += 0.60 * r
            p[Action.RAISE_BIG] *= 0.3
            p[Action.RAISE_SMALL] *= 0.3
        p[Action.CHECK_CALL] = max(1.0 - p.sum(), 0.15)
    else:
        pot_odds = to_call / (pot + to_call)
        commit = min(to_call / my_stack, 1.0)
        # The stack-risk premium only applies to stacks deep enough to have
        # something to protect: a short stack is pot-committed, so it should
        # call closer to plain pot odds instead of folding into every shove.
        ramp = min(max((depth - 8) / 30.0, 0.0), 1.0)
        need = pot_odds + (0.30 * commit ** 1.5 + 0.05) * ramp
        p_fold = _sigmoid(-(equity - need) / 0.06)
        cont = 1.0 - p_fold
        r2 = min(max((equity - 0.62) / 0.30, 0.0), 1.0) * (1.0 - commit)
        raise_total = cont * (0.03 + 0.45 * r2)
        p[Action.FOLD] = p_fold
        p[Action.RAISE_BIG] = raise_total * 0.4
        p[Action.RAISE_SMALL] = raise_total * 0.6
        shove = cont * (0.02 + 0.35 * r2 ** 2)
        if depth < 15 and equity > 0.55:
            shove += cont * 0.35
        p[Action.ALL_IN] = min(shove, cont - raise_total) if cont > raise_total else 0.0
        p[Action.CHECK_CALL] = max(cont - p[Action.RAISE_BIG] - p[Action.RAISE_SMALL] - p[Action.ALL_IN], 0.0)
    p *= legal_mask_for(to_call, my_stack, big_blind)
    return p / p.sum()


_net = None


def _init_worker():
    global _net
    _net = LIFNetwork()


def sample_example(seed: int):
    rng = random.Random(seed)
    street = rng.choice(STREETS)
    deck = make_deck()
    rng.shuffle(deck)
    hole = deck[:2]
    board = deck[2:2 + N_BOARD[street]]

    logu = lambda lo, hi: math.exp(rng.uniform(math.log(lo), math.log(hi)))
    n_opp = rng.randint(1, 3)
    my_stack = int(logu(4, 200) * BB)
    opp_stacks = [int(logu(4, 200) * BB) for _ in range(n_opp)]
    pot = max(int(logu(1.5, 80) * BB), BB)
    to_call = 0 if rng.random() < 0.45 else max(int(pot * rng.uniform(0.15, 1.5)), BB)
    to_call = min(to_call, my_stack)
    actions = []
    if to_call > 0 and rng.random() < 0.75:
        actions = [Action.ALL_IN if to_call >= my_stack else
                   (Action.RAISE_SMALL if to_call < 0.6 * pot else Action.RAISE_BIG)]
    elif to_call == 0 and rng.random() < 0.4:
        actions = [Action.CHECK_CALL]  # someone checked before me this street
    # (an empty list while facing a bet is the preflop-vs-blinds case)

    obs = Observation(seat=0, hole=hole, board=board, street=street, pot=pot, to_call=to_call,
                      my_stack=my_stack, opp_stack=sum(opp_stacks), is_button=rng.random() < 0.5,
                      street_actions=actions, my_bet=0, active_opponents=n_opp,
                      big_blind=BB, max_opp_stack=max(opp_stacks))
    inputs, equity = encode(obs)
    _net.reset()
    _net.run(inputs, n_steps=N_SIM_STEPS, record=True)
    features = brain_features(_net)
    mask = legal_mask_for(to_call, my_stack, BB)
    return features, mask, teacher_probs(equity, pot, to_call, my_stack, BB), equity, to_call / my_stack if my_stack else 1.0


def fit(features, masks, targets, weights=None, epochs=3000, lr=0.05, l2=1e-4, seed=0):
    """Full-batch Adam on masked-softmax cross-entropy against soft targets."""
    dec = SoftmaxDecoder(n_features=features.shape[1], seed=seed)
    X = np.tanh(features / 5.0)
    W, b = dec.W.copy(), dec.b.copy()
    mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
    n = len(X)
    w_s = np.ones(n) if weights is None else weights
    for step in range(1, epochs + 1):
        logits = X @ W.T + b
        logits = np.where(masks > 0, logits, -1e9)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        g = (p - targets) * (w_s / w_s.sum())[:, None]
        gW, gb = g.T @ X + l2 * W, g.sum(axis=0)
        for arr, grad, m, v in ((W, gW, mW, vW), (b, gb, mb, vb)):
            m *= 0.9; m += 0.1 * grad
            v *= 0.999; v += 0.001 * grad ** 2
            arr -= lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
    dec.W, dec.b = W, b
    return dec


def evaluate(dec, features, masks, targets):
    X = np.tanh(features / 5.0)
    logits = np.where(masks > 0, X @ dec.W.T + dec.b, -1e9)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
    kl = float(np.mean(np.sum(targets * (np.log(targets + 1e-9) - np.log(p + 1e-9)), axis=1)))
    return kl, float(np.mean(np.argmax(p, 1) == np.argmax(targets, 1))), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=16000)
    ap.add_argument("--out", type=str, default="runs/decoder.v3.npz")
    ap.add_argument("--data", type=str, default="runs/imitation_data.npz")
    ap.add_argument("--reuse-data", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.reuse_data:
        d = np.load(args.data)
        F, M, T, E, C = d["F"], d["M"], d["T"], d["E"], d["C"]
    else:
        t0 = time.time()
        with mp.Pool(mp.cpu_count(), initializer=_init_worker) as pool:
            rows = []
            for i, row in enumerate(pool.imap(sample_example, range(args.seed * 10**6, args.seed * 10**6 + args.examples), chunksize=20), 1):
                rows.append(row)
                if i % 2000 == 0:
                    print(f"generated {i}/{args.examples}  ({i / (time.time() - t0):.0f}/s)", flush=True)
        F = np.array([r[0] for r in rows]); M = np.array([r[1] for r in rows])
        T = np.array([r[2] for r in rows]); E = np.array([r[3] for r in rows]); C = np.array([r[4] for r in rows])
        np.savez(args.data, F=F, M=M, T=T, E=E, C=C)

    n_val = len(F) // 8
    # Strong hands matter most to get right (folding one is a real blunder),
    # so weight them up in the fit.
    weights = 1.0 + 2.0 * (E > 0.6)
    dec = fit(F[n_val:], M[n_val:], T[n_val:], weights=weights[n_val:], seed=args.seed)
    kl, acc, _ = evaluate(dec, F[:n_val], M[:n_val], T[:n_val])
    print(f"held-out: KL(teacher||decoder)={kl:.3f}  argmax-agreement={acc:.1%}")
    dec.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
