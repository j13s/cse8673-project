import gym
from policy_gradient import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np
import time
import ludopy
import pdb
from collections import defaultdict
import util
import tensorflow as tf

def train(episode,rewardType=None):
    tf.reset_default_graph()
    number_of_players=2
    number_of_pieces=4
    # Load checkpoint
    load_version =11
    save_version = load_version+1
    #load_path = "output/weights/ludo/{}/ludo-v2.ckpt".format(load_version)
    load_path = None
    save_path = "/content/drive/My Drive/cse8673_project/output/weights/ludo/{}/ludo-v2.ckpt".format(rewardType)
    PG_dict = {}
    reward = -1000
    act = util.Action(number_of_players,
                number_of_pieces,
                reward)
    PG = PolicyGradient(
        n_x = (number_of_players*number_of_pieces) + 5,   #input layer size
        n_y = 5,   #ouput layer size
        learning_rate=0.02,
        reward_decay=0.99,
        load_path=load_path,
        save_path=save_path,
        player_num = 0,
        rewardType = rewardType
    )
    EPISODES = episode
    ghost_players = list(reversed(range(0, 4)))[:-number_of_players]
    players = list(reversed(range(0, 4)))[-number_of_players:]
    winner = None
    winnerCount = defaultdict(int)
    for episode in range(EPISODES):
        if episode%500 == 0 :
            print("episode : ", episode)
        g = ludopy.Game(ghost_players=ghost_players,\
             number_of_pieces=number_of_pieces)
                
        episode_reward = 0

        there_is_a_winner = False
        winner = None
        count = 0
        while True:
            count += 1
            for i in range(number_of_players):
                if i == 0:
                    (dice, move_pieces, player_pieces, enemy_pieces,player_is_a_winner,
                                     there_is_a_winner),player_i = g.get_observation()

                    action,random = act.getAction(PG,
                                           enemy_pieces,
                                           player_pieces,
                                           move_pieces,
                                           dice)

                    _, _, _, _, _, there_is_a_winner = g.answer_observation(action)
                else:
                    action = act.getAction(move_pieces=move_pieces)
                
                if there_is_a_winner:
                    winner = player_i
                    winnerCount[player_i] += 1
                    break

            #this is where the agents are leanring
            if there_is_a_winner:
                if winner == 0:
                    PG.episode_rewards = [i+2000 if i == -1000 else i for i in PG.episode_rewards]

                discounted_episode_rewards_norm = PG.learn(episode,0,winner)
    return winnerCount,save_path
