import sys
sys.path.append("openai/")

import gym
from policy_gradient import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np
import time
import ludopy
import pdb

if __name__ == "__main__":
    number_of_players=2
    number_of_pieces=4
    # Load checkpoint
    load_version = 8
    save_version = load_version + 1
    load_path = "openai/output/weights/ludo/{}/ludo-v2.ckpt".format(load_version)
    load_path = None
    save_path = "openai/output/weights/ludo/{}/ludo-v2.ckpt".format(save_version)

    PG_dict = {}
    reward = 0
    for i in range(number_of_players):
        pg = PolicyGradient(
            n_x = (number_of_players*number_of_pieces) + 1,   #input layer size
            n_y = 4,   #ouput layer size
            learning_rate=0.02,
            reward_decay=0.99,
            load_path=load_path,
            save_path=save_path,
            player_num = i
        )
    
        PG_dict[i] = pg
    EPISODES = 10
    ghost_players = list(reversed(range(0, 4)))[:-number_of_players]
    players = list(reversed(range(0, 4)))[-number_of_players:]
    
    for episode in range(EPISODES):
        
        g = ludopy.Game(ghost_players=ghost_players,\
             number_of_pieces=number_of_pieces)
                
        episode_reward = 0

        there_is_a_winner = False
        
        while True:
            for i in range(number_of_players):
                PG = PG_dict[i]
                (dice, move_pieces, player_pieces, enemy_pieces, \
                     player_is_a_winner,there_is_a_winner),\
                     player_i = g.get_observation()
		#How to get the right enemy_pieces
                if player_i == 0:
                    observation = np.vstack((player_pieces[:,np.newaxis],\
                                            enemy_pieces[-1][:,np.newaxis]))
                elif player_i == 1:
                    observation = np.vstack((player_pieces[:,np.newaxis],\
                                            enemy_pieces[0][:,np.newaxis]))
                    
                observation = np.vstack((observation,dice))
                observation = observation.reshape([(number_of_players*number_of_pieces)+1,])
          
                # 1. Choose an action based on observation
                action = PG.choose_action(observation)
                #pdb.set_trace()
                if len(move_pieces):
                    if action not in move_pieces:
                        action = move_pieces[np.random.randint(0, len(move_pieces))]
                        reward = -10
                else:
                    action = None

                try:
                    _, _, _, _, _, there_is_a_winner = g.answer_observation(action)
                except e:
                    print(e)
                    #pdb.set_trace()

                PG.store_transition(observation, action, reward)
                reward = 0
                
                if there_is_a_winner:
                    break

            if there_is_a_winner:

                for i in range(number_of_players):
                    # 5. Train neural network
                    #pdb.set_trace()
                    PG = PG_dict[i]
                    discounted_episode_rewards_norm = PG.learn()
                break


