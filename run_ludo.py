import sys
sys.path.append("openai/")

import gym
from policy_gradient import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np
import time
import ludopy

if __name__ == "__main__":
    number_of_players=2
    number_of_pieces=2
    # Load checkpoint
    load_version = 8
    save_version = load_version + 1
    load_path = "output/weights/LunarLander/{}/LunarLander-v2.ckpt".format(load_version)
    save_path = "output/weights/LunarLander/{}/LunarLander-v2.ckpt".format(save_version)

    PG_dict = {}
    
    for i in range(number_of_players:
        pg = PolicyGradient(
            n_x = (number_of_players*number_of_pieces) + 1,   #input layer size
            n_y = 5,   #ouput layer size
            learning_rate=0.02,
            reward_decay=0.99,
            load_path=load_path,
            save_path=save_path
        )
    
        PG_dict[i] = pg
    
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
                if player_i == 0:
                    observation = np.vstack(player_pieces[:,np.newaxis],\
                                            enemy_pieces[-1][:,np.newaxis])
                elif player_i == 1:
                    observation = np.vstack(player_pieces[:,np.newaxis],\
                                            enemy_pieces[0][:,np.newaxis])
                    
                observation = np.vstack((observation,dice))

                # 1. Choose an action based on observation
                action = PG.choose_action(observation)

                _, _, _, _, _, there_is_a_winner = g.answer_observation(action)

                PG.store_transition(observation, action, reward)
                
                if there_is_a_winner:
                    break

            if there_is_a_winner:

                for i in range(number_of_players):
                    # 5. Train neural network
                    PG = PG_dict[i]
                    discounted_episode_rewards_norm = PG.learn()
                break


