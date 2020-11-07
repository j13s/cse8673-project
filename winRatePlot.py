import gym
from policy_gradient import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np
import time
import ludopy
import pdb
from collections import defaultdict
from matplotlib import pyplot as plt
import util
import tensorflow as tf

def winRate(load_path,episodes,player_num):
    tf.reset_default_graph()
    number_of_players=2
    number_of_pieces=4
    reward = -1000
    EPISODES = episodes
    ghost_players = list(reversed(range(0, 4)))[:-number_of_players]
    players = list(reversed(range(0, 4)))[-number_of_players:]
    winner = None
    act = util.Action(number_of_players,
                number_of_pieces,
                reward)
    winnerCount = defaultdict(int)
    print(load_path,"---")
    PG = PolicyGradient(
        n_x = (number_of_players*number_of_pieces) + 5,   #input layer size
        n_y = 5,   #ouput layer size
        learning_rate=0.02,
        reward_decay=0.99,
        load_path=load_path,
        save_path=None,
        player_num=player_num
    )
    preds = list()
    for episode in range(EPISODES):
        g = ludopy.Game(ghost_players=ghost_players,\
             number_of_pieces=number_of_pieces)

        there_is_a_winner = False
        winner = None
        totalMoves,wrongPred = 0,0
        while True:
            for i in range(number_of_players):
                (dice, move_pieces, player_pieces, enemy_pieces, \
                         player_is_a_winner,there_is_a_winner),\
                                 player_i = g.get_observation()
            
                if player_i == 1:
                    action,random = act.getAction(PG,
                                           enemy_pieces,
                                           player_pieces,
                                           move_pieces,
                                           dice)
                    totalMoves += 1
                    if random:
                        wrongPred += 1
                else:
                    action = act.getAction(move_pieces=move_pieces)

                _, _, _, _, _, there_is_a_winner = g.answer_observation(action)

                if there_is_a_winner:
                    if episode%1000 == 0 and 0:
                        print("saving the game--",episode)
                    winner = player_i
                    winnerCount[player_i] += 1
                    break
            if there_is_a_winner:
                preds.append([wrongPred,totalMoves])
                break
    return winnerCount,preds


