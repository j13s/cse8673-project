import copy
import unittest

import numpy as np

import ludopy
from util import State

from tests.util import test_state_attributes

class TestStateWith2Players(unittest.TestCase):
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

        test_state_attributes(
            test=self,
            state=state,
            die=5,
            whose_turn=0,
            who_won=None,
            actions=[],
            player_1_board=[0, 0, 0, 0],
            player_2_board=[0, 0, 0, 0],
            player_3_board=[0, 0, 0, 0],
            player_4_board=[0, 0, 0, 0],
        )

        state = State(
            observation=self.__g.answer_observation(0), for_player=obs[-1]
        )

        test_state_attributes(
            test=self,
            state=state,
            die=5,
            whose_turn=0,
            who_won=None,
            actions=[],
            player_1_board=[0, 0, 0, 0],
            player_2_board=[0, 0, 0, 0],
        )

        obs = self.__g.get_observation()

        state = State(observation=obs[0], for_player=obs[-1])

        test_state_attributes(
            test=self,
            state=state,
            die=6,
            whose_turn=0,
            who_won=None,
            actions=[0, 1, 2, 3],
            player_1_board=[0, 0, 0, 0],
            player_2_board=[0, 0, 0, 0],
        )

        g = copy.deepcopy(self.__g)
        resultant_state = State(
            observation=g.answer_observation(0),
            for_player=state.whose_turn_is_it(),
        )

        test_state_attributes(
            test=self,
            state=resultant_state,
            die=6,
            whose_turn=0,
            who_won=None,
            actions=[0, 1, 2, 3],
            player_1_board=[1, 0, 0, 0],
            player_2_board=[0, 0, 0, 0],
        )

        g = copy.deepcopy(self.__g)
        resultant_state = State(
            observation=g.answer_observation(1),
            for_player=state.whose_turn_is_it(),
        )

        test_state_attributes(
            test=self,
            state=resultant_state,
            die=6,
            whose_turn=0,
            who_won=None,
            actions=[0, 1, 2, 3],
            player_1_board=[0, 1, 0, 0],
            player_2_board=[0, 0, 0, 0],
        )


    def test_second_move(self):
        # Test against two-players, two pieces per player
        self.__g.get_observation()
        self.__g.answer_observation(0)
        self.__g.get_observation()
        self.__g.answer_observation(0)

        # Now it's player 1's turn
        obs = self.__g.get_observation()
        state = State(observation=obs[0], for_player=obs[-1])

        test_state_attributes(
            test=self,
            state=state,
            die=1,
            whose_turn=1,
            who_won=None,
            actions=[],
            player_1_board=[1, 0, 0, 0],
            player_2_board=[0, 0, 0, 0],
        )

        # Can't take an action, so just do something.
        state = State(
            observation=self.__g.answer_observation(0),
            for_player=state.whose_turn_is_it(),
        )

        test_state_attributes(
            test=self,
            state=state,
            die=1,
            whose_turn=1,
            who_won=None,
            actions=[],
            player_1_board=[1, 0, 0, 0],
            player_2_board=[0, 0, 0, 0],
        )

        ######################################################################

        for i in range(3):
            obs = self.__g.get_observation()
            state = State(observation=obs[0], for_player=obs[-1])

            test_state_attributes(
                test=self,
                state=state,
                die=4,
                whose_turn=1,
                who_won=None,
                actions=[],
                player_1_board=[1, 0, 0, 0],
                player_2_board=[0, 0, 0, 0],
            )

            # Can't take an action, so just do something.
            state = State(
                observation=self.__g.answer_observation(0),
                for_player=state.whose_turn_is_it(),
            )

            test_state_attributes(
                test=self,
                state=state,
                die=4,
                whose_turn=1,
                who_won=None,
                actions=[],
                player_1_board=[1, 0, 0, 0],
                player_2_board=[0, 0, 0, 0],
            )


        ######################################################################

        obs = self.__g.get_observation()
        state = State(observation=obs[0], for_player=obs[-1])

        test_state_attributes(
            test=self,
            state=state,
            die=2,
            whose_turn=0,
            who_won=None,
            actions=[0],
            player_1_board=[1, 0, 0, 0],
            player_2_board=[0, 0, 0, 0],
        )

        state = State(
            observation=self.__g.answer_observation(0),
            for_player=state.whose_turn_is_it(),
        )

        test_state_attributes(
            test=self,
            state=state,
            die=2,
            whose_turn=0,
            who_won=None,
            actions=[0],
            player_1_board=[3, 0, 0, 0],
            player_2_board=[0, 0, 0, 0],
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

        test_state_attributes(
            test=self,
            state=state,
            die=4,
            whose_turn=0,
            who_won=0,
            actions=[],
            player_1_board=[59, 59, 59, 59],
            player_2_board=[59, 54, 53, 26],
        )

    def tearDown(self):
        del self.__g

    @classmethod
    def tearDownClass(cls):
        np.random.set_state(cls.initial_np_state)

