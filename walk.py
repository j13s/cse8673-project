#!/usr/bin/env python3
# Generate a random walk

import argparse
import sys

import numpy as np

from util import randwalk

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--number-of-players",
        help="Number of Ludo players",
        type=int,
        default=4
    )

    parser.add_argument(
        "--number-of-pieces",
        help="Number of pieces per Ludo player",
        type=int,
        default=4
    )
    args = parser.parse_args()

    walk = randwalk(
        number_of_players=args.number_of_players,
        number_of_pieces=args.number_of_pieces,
    )

    history = walk.get_hist()
    np.save(sys.stdout.buffer, history)

if __name__ == "__main__":
    main()

