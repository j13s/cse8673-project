import tempfile
import unittest

import numpy as np

from policy_util import policy
from util import History


class TestHistory(unittest.TestCase):
    def setUp(self):
        self.history = np.load(
            "tests/fixtures/ludo_history_2_players_4_pieces.npy",
            allow_pickle=True
        )

        return

    def test_number_of_rounds(self):
        history = History(frames=self.history)

        self.assertEqual(71, len(history.rounds()))

        round1 = history.rounds()[0]
        self.assertEqual(3, len(round1.turns()))

        turn1 = round1.turns()[0]
        self.assertEqual(5, turn1.die_roll())
        self.assertEqual(0, turn1.player())
        self.assertEqual(None, turn1.action())

        turn2 = round1.turns()[1]
        self.assertEqual(6, turn2.die_roll())
        self.assertEqual(0, turn2.player())
        self.assertEqual(2, turn2.action())

        turn3 = round1.turns()[2]
        self.assertEqual(6, turn3.die_roll())
        self.assertEqual(1, turn3.player())

        return

