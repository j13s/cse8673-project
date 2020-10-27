import numpy as np

import ludopy

def policy(number_of_players=4, number_of_pieces=4):
    """
    Generate a saved Numpy array representing a 2-player Ludo game sequence.

    This is taken from the `test/randomwalk.py` in LUDOpy

    :param number_of_players: Number of Ludo players.
    :type number_of_players: `int`

    :param number_of_pieces: Number of pieces per player.
    :type number_of_pieces: `int`
    """

    # `ghost_players` is a LUDOpy specific way to specify the number of
    # players. So, if we want 2 players, the code below will generate a list:
    #
    #     [3, 2, 1, 0]
    #
    # and slice it so it omits players 2 and 3.
    #
    #     [3, 2, 1, 0][:2] == [3, 2]

    g = ludopy.Game(
        ghost_players=list(reversed(range(0, 4)))[:-number_of_players],
        number_of_pieces=number_of_pieces
    )
    there_is_a_winner = False

    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = g.get_observation()

        if len(move_pieces):
            piece_to_move = \
                move_pieces[policyinator(dice, move_pieces, player_pieces, enemy_pieces)]
        else:
            piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

    return g


def policyinator(dice, move_pieces, player_pieces, enemy_pieces):
    # Here's where we do all that machine learning stuff
    return np.random.randint(0, len(move_pieces))
