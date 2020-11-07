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
        self.__g = ludopy.Game(ghost_players=[3, 2], number_of_pieces=4)

    def test_first_move(self):
        # Test against two-players, two pieces per player
        obs = self.__g.get_observation()

        state = State(observation=obs[0], for_player=obs[-1])

        self.assertEqual(5, state.die())
        self.assertEqual(0, state.whose_turn_is_it())
        self.assertIsNone(state.who_won())
        np.testing.assert_equal([], state.actions())

        self.__test_initial_state(state)

        state = State(
            observation=self.__g.answer_observation(0), for_player=obs[-1]
        )

        self.assertEqual(5, state.die())

        self.__test_initial_state(state)

        obs = self.__g.get_observation()

        state = State(observation=obs[0], for_player=obs[-1])

        self.assertEqual(6, state.die())
        self.assertEqual(0, state.whose_turn_is_it())
        np.testing.assert_equal([0, 1, 2, 3], state.actions())

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
                    3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 
                ],
                resultant_state.board_for(player=0)
            )
            self.__test_initial_board_state_for(resultant_state, player=1)

    def test_second_move(self):
        # Test against two-players, two pieces per player
        self.__g.get_observation()
        self.__g.answer_observation(0)
        self.__g.get_observation()
        self.__g.answer_observation(0)

        # Now it's player 1's turn
        obs = self.__g.get_observation()
        state = State(observation=obs[0], for_player=obs[-1])

        self.assertEqual(1, state.die())
        self.assertEqual(1, state.whose_turn_is_it())
        np.testing.assert_equal([], state.actions())

        self.assertEqual(
            [
                3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            ],
            state.board_for(player=0)
        )

        self.assertEqual(
            [
                4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            ],
            state.board_for(player=1)
        )

        # Can't take an action, so just do something.
        state = State(
            observation=self.__g.answer_observation(0),
            for_player=state.whose_turn_is_it(),
        )

        self.assertEqual(1, state.die())
        self.assertEqual(1, state.whose_turn_is_it())
        np.testing.assert_equal([], state.actions())

        self.assertEqual(
            [
                3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            ],
            state.board_for(player=0)
        )

        self.assertEqual(
            [
                4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            ],
            state.board_for(player=1)
        )

        ######################################################################

        for i in range(3):
            obs = self.__g.get_observation()
            state = State(observation=obs[0], for_player=obs[-1])

            self.assertEqual(4, state.die())
            self.assertEqual(1, state.whose_turn_is_it())
            np.testing.assert_equal([], state.actions())

            self.assertEqual(
                [
                    3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 
                ],
                state.board_for(player=0)
            )

            self.assertEqual(
                [
                    4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 
                ],
                state.board_for(player=1)
            )

            # Can't take an action, so just do something.
            state = State(
                observation=self.__g.answer_observation(0),
                for_player=state.whose_turn_is_it(),
            )

            self.assertEqual(4, state.die())
            self.assertEqual(1, state.whose_turn_is_it())
            np.testing.assert_equal([], state.actions())

            self.assertEqual(
                [
                    3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 
                ],
                state.board_for(player=0)
            )

            self.assertEqual(
                [
                    4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 
                ],
                state.board_for(player=1)
            )

        ######################################################################

        obs = self.__g.get_observation()
        state = State(observation=obs[0], for_player=obs[-1])

        self.assertEqual(2, state.die())
        self.assertEqual(0, state.whose_turn_is_it())
        np.testing.assert_equal([0], state.actions())

        self.assertEqual(
            [
                3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            ],
            state.board_for(player=0)
        )

        self.assertEqual(
            [
                4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            ],
            state.board_for(player=1)
        )

        state = State(
            observation=self.__g.answer_observation(0),
            for_player=state.whose_turn_is_it(),
        )

        self.assertEqual(2, state.die())
        self.assertEqual(0, state.whose_turn_is_it())
        np.testing.assert_equal([0], state.actions())

        self.assertEqual(
            [
                3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            ],
            state.board_for(player=0)
        )

        self.assertEqual(
            [
                4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            ],
            state.board_for(player=1)
        )


    def test_winner(self):
        there_is_not_a_winner = True

        while there_is_not_a_winner:
            obs = self.__g.get_observation()
            state = State(observation=obs[0], for_player=obs[-1])

            action = 0
            actions = state.actions()
            if actions.any():
                action = actions[np.random.randint(0, len(actions))]

            state = State(
                observation=self.__g.answer_observation(action),
                for_player=state.whose_turn_is_it(),
            )

            there_is_not_a_winner = not state.is_there_a_winner()

        self.assertTrue(state.is_there_a_winner())
        self.assertEqual(0, state.who_won())

        self.assertEqual(
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 
            ],
            state.board_for(player=0)
        )

        self.assertEqual(
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 
            ],
            state.board_for(player=1)
        )


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
            4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        ], state.board_for(player=player))
