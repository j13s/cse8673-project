import numpy as np

def test_state_attributes(
    test=None,
    state=None,
    die=None,
    whose_turn=None,
    who_won=None,
    actions=None,
    player_1_board=None,
    player_2_board=None,
    player_3_board=None,
    player_4_board=None,):

    test.assertEqual(die, state.die())
    test.assertEqual(whose_turn, state.whose_turn_is_it())
    test.assertEqual(who_won, state.who_won())
    np.testing.assert_equal(actions, state.actions())
    np.testing.assert_equal(player_1_board, state.board_for(player=0))
    np.testing.assert_equal(player_2_board, state.board_for(player=1))

    if None is not player_3_board:
        np.testing.assert_equal(player_3_board, state.board_for(player=2))
    if None is not player_4_board:
        np.testing.assert_equal(player_4_board, state.board_for(player=3))

    return

