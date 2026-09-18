"""Downloads the IRC Poker Database to data/hand_histories/IRCdata.tgz, for
pretrain.py to build real-player training examples from (see
hand_history_parser.py).

Source: University of Alberta Computer Poker Research Group
(http://poker.cs.ualberta.ca/irc_poker_database.html) -- over 10 million
hands logged from IRC poker channels, 1995-2001. A long-standing reference
dataset in poker AI research.

License note (read before using): the source page states no formal
license, only "may be useful to poker programming researchers and
hobbyists." This script and pretrain.py use it in that spirit; it is not
released under an open-source license and shouldn't be treated as one.

The download is ~970MB.

Usage:
    python scripts/fetch_hand_history_dataset.py
"""
from __future__ import annotations

import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "hand_histories")
OUT_PATH = os.path.join(DATA_DIR, "IRCdata.tgz")
URL = "http://poker.cs.ualberta.ca/IRC/IRCdata.tgz"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(OUT_PATH):
        print(f"{OUT_PATH} already exists, skipping download")
        return

    print(f"Downloading {URL} -> {OUT_PATH} (~970MB, this takes a while)")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            print(f"\r  {pct}% ({downloaded / 1e6:.0f}MB / {total_size / 1e6:.0f}MB)", end="", flush=True)

    urllib.request.urlretrieve(URL, OUT_PATH, reporthook=_progress)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
