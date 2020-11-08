import copy

import numpy as np

import ludopy

def randwalk(number_of_players=4, number_of_pieces=4):
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
                move_pieces[np.random.randint(0, len(move_pieces))]
        else:
            piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

    return g


class State:
    """Represents the state of the ludo board."""

    board = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        ]

    def __init__(self, observation=None, for_player=None):
        self.__player_turn = for_player

        self.__die = observation[0]

        # If there is a winner...
        if observation[-1]:
            # Remember which player won
            self.__winner = for_player
        else:
            self.__winner = None

        self.__actions = observation[1]

        self.__boards = [
            self.__compute_board_for(
                player_pieces=observation[2],
                enemy_pieces=observation[3],
                for_player=0,
            ),
            self.__compute_board_for(
                player_pieces=observation[2],
                enemy_pieces=observation[3],
                for_player=1,
            ),
            self.__compute_board_for(
                player_pieces=observation[2],
                enemy_pieces=observation[3],
                for_player=2,
            ),
            self.__compute_board_for(
                player_pieces=observation[2],
                enemy_pieces=observation[3],
                for_player=3,
            ),
        ]

        return

    def die(self):
        return self.__die

    def whose_turn_is_it(self):
        return self.__player_turn

    def is_there_a_winner(self):
        return self.__winner is not None

    def who_won(self):
        return self.__winner

    def actions(self):
        return self.__actions

    def board_for(self, player=None):
        return copy.copy(self.__boards[player])

    def __compute_board_for(
        self,
        player_pieces=None,
        enemy_pieces=None,
        for_player=None):
        board = copy.copy(__class__.board)

        turn = self.whose_turn_is_it()
        if 0 == turn:
            if 0 == for_player:
                for position in player_pieces:
                    board[position] += 1
            elif 1 == for_player:
                for position in enemy_pieces[0]:
                    board[position] += 1
            elif 2 == for_player:
                for position in enemy_pieces[1]:
                    board[position] += 1
            elif 3 == for_player:
                for position in enemy_pieces[2]:
                    board[position] += 1
        elif 1 == turn:
            if 0 == for_player:
                for position in enemy_pieces[2]:
                    board[position] += 1
            elif 1 == for_player:
                for position in player_pieces:
                    board[position] += 1
            elif 2 == for_player:
                for position in enemy_pieces[0]:
                    board[position] += 1
            elif 3 == for_player:
                for position in enemy_pieces[1]:
                    board[position] += 1
        elif 2 == turn:
            if 0 == for_player:
                for position in enemy_pieces[1]:
                    board[position] += 1
            elif 1 == for_player:
                for position in enemy_pieces[2]:
                    board[position] += 1
            elif 2 == for_player:
                for position in player_pieces:
                    board[position] += 1
            elif 3 == for_player:
                for position in enemy_pieces[0]:
                    board[position] += 1
        elif 3 == turn:
            if 0 == for_player:
                for position in enemy_pieces[0]:
                    board[position] += 1
            elif 1 == for_player:
                for position in enemy_pieces[1]:
                    board[position] += 1
            elif 2 == for_player:
                for position in enemy_pieces[2]:
                    board[position] += 1
            elif 3 == for_player:
                for position in player_pieces:
                    board[position] += 1


        return board


BOARD = 0
DIE = -3
PLAYER = -2
ROUND = -1
class History:
    def __init__(self, frames=None):
        self.__rounds = self.__compute_rounds(frames=frames)

        return

    def __compute_rounds(self, frames=[]):
        rounds = []

        for i in range(1, frames[-1][ROUND] + 1):
            rounds.append(Round())
            current_round = list(filter(lambda f: f[ROUND] == i, frames))

            for j in range(0, current_round[-1][PLAYER] + 1):
                turn_frames = list(
                    filter(lambda f: f[PLAYER] == j, current_round)
                )

                for i in range(0, len(turn_frames), 2):
                    before = turn_frames[i]
                    after = turn_frames[i + 1]
                    t = Turn(frames=[before, after])
                    rounds[-1].append_turn(t)

        return rounds

    def rounds(self):
        return copy.copy(self.__rounds)


class Round:
    def __init__(self):
        self.__turns = []

        return

    def append_turn(self, turn=None):
        if isinstance(turn, Turn):
            self.__turns.append(turn)

    def turns(self):
        return copy.copy(self.__turns)


class Turn:
    def __init__(self, frames=[]):
        self.__frames = copy.copy(frames)

        self.__player = frames[0][PLAYER]

        self.__states = self.__compute_states(frames=frames)

        self.__num_pieces = len(self.__states[0])

        self.__action = self.__compute_action(frames=frames)

        self.__die_roll = frames[0][DIE]

        return

    def player(self):
        return self.__player

    def action(self):
        return self.__action

    def die_roll(self):
        return self.__die_roll

    def state(self):
        return copy.copy(self.__states[0])

    def __compute_action(self, frames=[]):
        before = self.__states[-2]
        after = self.__states[-1]

        action = None
        for i in range(self.__num_pieces):
            if before[i] != after[i]:
                action = i
                break
        
        return action

    def __compute_states(self, frames=[]):
        states = []

        for frame in frames:
            states.append(frame[BOARD][self.player()])

        return states

class Action:
    def __init__(self,num_players,num_of_pieces,reward):
        self.num_players = num_players
        self.num_of_pieces = num_of_pieces
        self.reward = reward
        
    def getAction(self,PG="",
               enemy_pieces="",
               player_pieces="",
               move_pieces="",
               dice=""
                 ):
        
        self.PG = PG
        self.enemy_pieces = enemy_pieces
        self.player_pieces = player_pieces
        self.move_pieces = move_pieces
        self.dice = dice
        if PG:
            return self.__policyAction()
        else:
            return self.__randomAction()
            
    def __policyAction(self):
        if not len(self.move_pieces):
            return None,False
        random = False
        if self.num_players != 4:
            enemy_pieces = self.__getEnemy()
        else:
            enemy_pieces = self.enemy_pieces
        observation = np.vstack((self.player_pieces[:,np.newaxis],\
                                enemy_pieces[0][:,np.newaxis]))
        observation = (observation + 1)/60
        observation = np.vstack((observation,self.dice/6))
        move_pieces = np.ones(4)*-1
        if len(self.move_pieces):
            move_pieces[self.move_pieces] = 1
        observation = np.vstack((observation, move_pieces[:,np.newaxis]))

        observation = observation.reshape([(self.num_players*\
                                            self.num_of_pieces)+5,])
        action = self.PG.choose_action(observation)
        if not len(self.move_pieces):
            if action == 4:
                self.PG.store_transition(observation, action, self.reward)
                action == None
            else:
                random = True
                reward = 2*self.reward
                self.PG.store_transition(observation, action, reward)
                action = None
        else:
            if action not in self.move_pieces:
                reward = 2*self.reward
                self.PG.store_transition(observation, action, reward)
                action = self.__randomAction()
                random = True
            else:
                self.PG.store_transition(observation, action, self.reward)
        return action, random
    
    def __getEnemy(self):
        enemy = self.enemy_pieces[np.any(self.enemy_pieces != 0, axis=1)]
        if enemy.size <= 0:
            return np.array([self.enemy_pieces[0]])
        return enemy
    
    
    def __randomAction(self):        
        if len(self.move_pieces):
            piece_to_move = \
                self.move_pieces[np.random.randint(0, len(self.move_pieces))]
        else:
            piece_to_move = -1
            
        return piece_to_move
        
