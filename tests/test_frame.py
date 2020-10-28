import tempfile
import unittest

import numpy as np

from policy_util import policy
from util import Round


class TestRound(unittest.TestCase):
    def setUp(self):
        history = np.load(
            "tests/fixtures/ludo_history_2_players_4_pieces.npy",
            allow_pickle=True
        )

        self.round = Round(frames=history[:6])

        return

