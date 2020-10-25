import tempfile
import unittest

import numpy as np

from util import randwalk


class TestRandomWalkParameters(unittest.TestCase):
    def test_2_player_game(self):
        game = randwalk(number_of_players=2)
        history = game.get_hist()

        self.assertEqual(2, len(history[-1][0]))
        self.assertEqual(4, len(history[-1][0][0]))

    def test_2_player_game_with_2_pieces(self):
        game = randwalk(number_of_players=2, number_of_pieces=2)
        history = game.get_hist()

        self.assertEqual(2, len(history[-1][0][0]))

    def test_4_player_game(self):
        game = randwalk()
        history = game.get_hist()

        self.assertEqual(4, len(history[-1][0]))
        self.assertEqual(4, len(history[-1][0][0]))

    def test_4_player_game_with_2_pieces(self):
        game = randwalk(number_of_pieces=2)
        history = game.get_hist()

        self.assertEqual(4, len(history[-1][0]))
        self.assertEqual(2, len(history[-1][0][0]))
