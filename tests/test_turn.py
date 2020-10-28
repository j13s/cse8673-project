import tempfile
import unittest

import numpy as np

from policy_util import policy
from util import Turn


class TestTurn(unittest.TestCase):
    def setUp(self):
        history = np.load(
            "tests/fixtures/ludo_history_2_players_4_pieces.npy",
            allow_pickle=True
        )

        self.turn1 = Turn(frames=history[:2])
        self.turn2 = Turn(frames=history[2:4])
        self.turn3 = Turn(frames=history[4:6])

        return

    def test_turn1(self):
        self.assertEqual(0, self.turn1.player())
        self.assertEqual(None, self.turn1.action())
        self.assertEqual(5, self.turn1.die_roll())

    def test_turn2(self):
        self.assertEqual(0, self.turn2.player())
        self.assertEqual(2, self.turn2.action())
        self.assertEqual(6, self.turn2.die_roll())

    def test_turn3(self):
        self.assertEqual(1, self.turn3.player())
        self.assertEqual(0, self.turn3.action())
        self.assertEqual(6, self.turn3.die_roll())

