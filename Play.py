import util
from collections import defaultdict
import tensorflow as tf
import os
import pickle
number_of_players=2
number_of_pieces=4
ghost_players = list(reversed(range(0, 4)))[:-number_of_players]
players = list(reversed(range(0, 4)))[-number_of_players:]
player = util.Play()
print(ghost_players)
print(players)
episodeStart = -1
load_path = None
save_path = "output/weights/ludo/{}/ludo-v2.ckpt"
player_num = 0
randomMovesDict = defaultdict(dict)
episodes = 30001
model2keep = int(episodes/1000)
policyPlayer = [players[0]]
random_player = [players[1]] 
itr = ["18Nov/13"]
for i in itr:
    print("-----------{}--------------".format(i))
    new_path = save_path.format(i)
    print(new_path)
    tf.reset_default_graph()
    player.play(
        ghost_players = ghost_players,
        policyPlayers = policyPlayer,
        randomPlayers = random_player,
        load_path=None,
        save_path= new_path,
        episodes=episodes,
        episodeStart=episodeStart,
        training=True,
        n_x=13,
        n_y=4,
        learning_rate=0.02,
        reward_decay=0.99,
        player_num = 0,
        number_of_players=2,
        number_of_pieces=4,
        reward=-1000,
        rewardType = "monte",
        model2keep = model2keep,
        inputBoardType = "fullBoard"
    )
"""
path = "output/weights/ludo/{}/checkpoint"
modelPath = "output/weights/ludo/{}/{}"
modelType = "reward4CorrectMove"
graphDataPath = "output/grpahdata/{}.pkl".format(modelType)
util.PlotGraphs(graphDataPath,path,modelPath,modelType,1,0,players
,ghost_players,policyPlayer,random_player,False,n_x=22)
"""