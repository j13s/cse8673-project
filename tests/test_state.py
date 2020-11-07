import copy
import unittest

import numpy as np

import ludopy
from util import State

class TestState(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Seed the random number generator to a known state
        cls.initial_np_state = np.random.get_state()

    def setUp(self):
        np.random.seed(0)
        self.__g = ludopy.Game(ghost_players=[3, 2], number_of_pieces=2)

    def test_initial_state(self):
        # Test against two-players, two pieces per player
        obs = self.__g.get_observation()

        state = State(observation=obs[0], for_player=obs[-1])

        self.assertEqual(5, state.die())
        self.assertEqual(0, state.whose_turn_is_it())
        np.testing.assert_equal([], state.actions())

        self.__test_initial_state(state)

        state = State(
            observation=self.__g.answer_observation(0), for_player=obs[-1]
        )

        self.assertEqual(5, state.die())

        self.__test_initial_state(state)

    def test_second_move(self):
        # Test against two-players, two pieces per player
        self.__g.get_observation()
        self.__g.answer_observation(0)
        obs = self.__g.get_observation()

        state = State(observation=obs[0], for_player=obs[-1])

        self.assertEqual(6, state.die())
        self.assertEqual(0, state.whose_turn_is_it())
        np.testing.assert_equal([0, 1], state.actions())

        self.__test_initial_board_state_for(state, player=0)
        self.__test_initial_board_state_for(state, player=1)

        for action in state.actions():
            g = copy.deepcopy(self.__g)
            resultant_state = State(
                observation=g.answer_observation(action),
                for_player=state.whose_turn_is_it(),
            )
            self.assertEqual(
                [
                    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 
                ],
                resultant_state.board_for(player=0)
            )
            self.__test_initial_board_state_for(resultant_state, player=1)

    def tearDown(self):
        del self.__g

    @classmethod
    def tearDownClass(cls):
        np.random.set_state(cls.initial_np_state)

    def __test_initial_state(self, state):
        self.__test_initial_board_state_for(state, player=0)
        self.__test_initial_board_state_for(state, player=1)

        self.assertIsNone(state.who_won())
        self.assertFalse(state.is_there_a_winner())
        self.assertEqual(0, state.whose_turn_is_it())

    def __test_initial_board_state_for(self, state, player=None):
        self.assertEqual([
            2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        ], state.board_for(player=player))
