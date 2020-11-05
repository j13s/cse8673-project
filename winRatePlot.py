import gym
from policy_gradient import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np
import time
import ludopy
import pdb
from collections import defaultdict
from matplotlib import pyplot as plt
import Action

number_of_players=2
number_of_pieces=4
# Load checkpoint
load_version =12
load_path = "output/weights/ludo/{}/ludo-v2.ckpt".format(load_version)+str(19000)
PG_dict = {}
reward = -1000



EPISODES = 100
ghost_players = list(reversed(range(0, 4)))[:-number_of_players]
players = list(reversed(range(0, 4)))[-number_of_players:]
winner = None
player = 0

def winRate(PG):
    act = Action.Action(number_of_players,
                number_of_pieces,
                reward)
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
        invalidMoves = 0
        totalMoves = 0
        validMoves = 0
        notPossible = 0
        while True:
            count += 1
            for i in range(number_of_players):
                (dice, move_pieces, player_pieces, enemy_pieces, \
                         player_is_a_winner,there_is_a_winner),\
                                 player_i = g.get_observation()
            
                if player_i == 1:
                    
                    action,random = 
                    
                    
                    
                    
                    totalMoves += 1
                    observation = np.vstack((player_pieces[:,np.newaxis],\
                                            enemy_pieces[0][:,np.newaxis]))
                    observation = (observation + 1)/60
                    observation = np.vstack((observation,dice))
                    observation = observation.reshape([(number_of_players*number_of_pieces)+1,])

                    action = PG.choose_action(observation)
                    if len(move_pieces):
                        if action not in move_pieces:
                            invalidMoves += 1
                            action = move_pieces[np.random.randint(0, len(move_pieces))]
                        else:
                            validMoves += 1
                    else:
                        notPossible += 1
                        action = None
                else:
                    if len(move_pieces):
                        action = \
                        move_pieces[np.random.randint(0, len(move_pieces))]
                    else:
                        action = None
                        #now there will be no None as the output from the policy


                try:
                    _, _, _, _, _, there_is_a_winner = g.answer_observation(action)
                except Exception as e:
                    print(e)
                    break
                
                if there_is_a_winner:
                    if episode%1000 == 0:
                        print("saving the game--",episode)
                        g.save_hist_video("videos/"+str(episode)+"game.avi")
                    winner = player_i
                    winnerCount[player_i] += 1
                    break
            if there_is_a_winner:
                break
    return winnerCount,invalidMoves,totalMoves,validMoves,notPossible

PG = PolicyGradient(
    n_x = (number_of_players*number_of_pieces) + 1,   #input layer size
    n_y = 4,   #ouput layer size
    learning_rate=0.02,
    reward_decay=0.99,
    load_path=load_path,
    save_path="",
    player_num = player
)
for i in range(1):
    print(winRate(PG))