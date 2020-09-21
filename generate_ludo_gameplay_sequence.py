import argparse
import os
import stat

import numpy as np

import ludopy

def set_history_read_only(history=None):
    mode = os.stat(history).st_mode
    ro_mask = 0o777 ^ (stat.S_IWRITE | stat.S_IWGRP | stat.S_IWOTH)
    return os.chmod(history, mode & ro_mask)

def randwalk():
    # Generate a saved Numpy array representing a 2-player Ludo game sequence.
    # This is taken from the `test/randomwalk.py` in LUDOpy

    g = ludopy.Game()
    there_is_a_winner = False

    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = g.get_observation()

        if len(move_pieces):
            piece_to_move = \
                move_pieces[np.random.randint(0, len(move_pieces))]
        else:
            piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

    return g

def save_history(game=None, filename_prefix=None, n=None, padding=None):
    # Save the play history to a file
    try:
        filename = f"{filename_prefix}.{str(n).zfill(padding)}.npy"
        game.save_hist(filename)

        # Set history readonly to prevent accidentally overwriting old
        # histories
        #
        # From https://stackoverflow.com/a/51262451/374681

    except Exception as e:
        raise e

    return filename


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--number-of-histories",
        help="Number of Ludo game histories to generate (default 50)",
        type=int,
        default=50
    )

    parser.add_argument(
        "--filename-prefix",
        help="Prefix for the saved history filename (ludo_historyXX.npy)",
        type=str,
        default="ludo_history"
    )

    parser.add_argument(
        '--read-only',
        dest='read_only',
        action='store_true',
        help="Set histories to read-only to prevent accidental overwrites.",
    )

    parser.add_argument(
        '--read-write',
        dest='read_only',
        action='store_false',
        help="Do not set histories to read-only",
    )
    parser.set_defaults(read_only=True)
    args = parser.parse_args()

    # Start counting from 0, so if 1000 histories, only need 000-999. Not
    # 0000-0999.
    padding = len(str(args.number_of_histories - 1))

    for i in range(0, args.number_of_histories):
        game = randwalk()

        filename_of_saved_history = save_history(
            game=game,
            filename_prefix=args.filename_prefix,
            n=i,
            padding=padding,
        )

        if args.read_only:
            set_history_read_only(history=filename_of_saved_history)

    return


if '__main__' == __name__:
    main()

