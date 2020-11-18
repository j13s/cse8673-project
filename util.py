import copy
import numpy as np
import ludopy
import copy
from policy_gradient import PolicyGradient
from collections import defaultdict
import pickle
from matplotlib import pyplot as plt
import os 
import tensorflow as tf
import time
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

        board = None

        turn = self.whose_turn_is_it()
        if 0 == turn:
            if 0 == for_player:
                board = copy.copy(player_pieces)
            elif 1 == for_player:
                board = copy.copy(enemy_pieces[0])
            elif 2 == for_player:
                board = copy.copy(enemy_pieces[1])
            elif 3 == for_player:
                board = copy.copy(enemy_pieces[2])
        elif 1 == turn:
            if 0 == for_player:
                board = enemy_pieces[2]
            elif 1 == for_player:
                board = player_pieces
            elif 2 == for_player:
                board = enemy_pieces[0]
            elif 3 == for_player:
                board = enemy_pieces[1]
        elif 2 == turn:
            if 0 == for_player:
                board = enemy_pieces[1]
            elif 1 == for_player:
                board = enemy_pieces[2]
            elif 2 == for_player:
                board = player_pieces
            elif 3 == for_player:
                board = enemy_pieces[0]
        elif 3 == turn:
            if 0 == for_player:
                board = enemy_pieces[0]
            elif 1 == for_player:
                board = enemy_pieces[1]
            elif 2 == for_player:
                board = enemy_pieces[2]
            elif 3 == for_player:
                board = player_pieces

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
    def __init__(self,reward):
        self.reward = reward
        
    def action(self,play,state,n_y,playerPool=None,currPlayer=None,data=None,PG=None,train=True):
        if PG:
            return self.__action(play,state,playerPool,currPlayer,PG,data,train,n_y)
        else:
            return self.__randomAction(state)
        
    def __action(self,play,state,playerPool,currPlayer,PG,data,train,n_y):
        random = False
        action = prev = -1
        punish = 1
        count = 0
        onehot = None
        enemies = [i for i in playerPool if i != currPlayer]
        pBoard = state.board_for(currPlayer)
        actions = np.zeros(n_y)
        actions[state.actions()] = 1
        for enemy in enemies:
            pBoard = np.hstack((pBoard,state.board_for(enemy)))
        pBoard = (pBoard)/59
        pBoard= np.hstack((pBoard,actions))
        modelInput = np.hstack((pBoard,state.die()/6))
        action,prob = PG.choose_action(modelInput,state.actions())
        onehot = np.eye(n_y)[action]
        data.store(onehot,modelInput,self.reward,prob)
        return action

    def __randomAction(self,state):        
        if len(state.actions()):
            action = \
                state.actions()[np.random.randint(0, len(state.actions()))]
        else:
            action = -1

        return action
    
class StoreTrainingData:
    def __init__(self,outputLSize):
        self.__initData()
        self.outputL = outputLSize
        
    def store(self,a,modelInput,reward,prob=None):
        self.actions.append(a)
        self.input.append(modelInput)
        self.rewards.append(reward)
        if prob is not None:
            self.probs.append(prob)
        
    def render(self,clearData):
        action =  np.array(self.actions)
        obs = np.array(self.input)
        rewards = np.array(self.rewards)
        if clearData:
            self.__initData()
        return action,obs,rewards
    
    def __initData(self):
        self.actions = []
        self.input = []
        self.rewards = []
        self.probs = []
        
class Play:
    def play(
        self,
        policyPlayers,
        randomPlayers,
        load_path,
        save_path,
        episodes,
        episodeStart,
        training,
        ghost_players,
        model2keep,
        n_x=125,
        n_y=5,
        learning_rate=0.02,
        reward_decay=0.99,
        player_num = 0,
        number_of_players=2,
        number_of_pieces=4,
        reward=-1000,
        rewardType = "monte",
        inputBoardType = "fullBoard"
    ):
        totalPlayers = len(policyPlayers)+len(randomPlayers)
        playerPool = policyPlayers+randomPlayers
        data = dict()
        for i in policyPlayers:
            data[i] = StoreTrainingData(n_y) 
        act = Action(reward)
        PG = PolicyGradient(
            n_x = n_x,   #input layer size
            n_y = n_y,   #ouput layer size
            learning_rate=learning_rate,
            reward_decay=reward_decay,
            load_path=load_path,
            save_path=save_path,
            player_num = player_num,
            rewardType = rewardType,
            toKeep = model2keep
        )
        timeInterval = 50
        winCount = defaultdict(int)
        preds = list()
        startTime = time.time()
        for episode in range(episodeStart+1,episodeStart+episodes):
            g = ludopy.Game(ghost_players=ghost_players,\
             number_of_pieces=number_of_pieces)
            while True:
                obs,currPlayer = g.get_observation()
                state = State(obs,currPlayer)
                if currPlayer in policyPlayers and len(state.actions()) > 0:
                    action= act.action(self,state,n_y,playerPool,currPlayer,data[currPlayer],PG,training)
                elif currPlayer in randomPlayers:
                    action = act.action(self,state,n_y)
                _, _, _, _, _, there_is_a_winner = g.answer_observation(action)
                if int(time.time() - startTime) > timeInterval:
                    print("episode: {} running for {}".format(episode,time.time() - startTime))
                    timeInterval += 50
                if there_is_a_winner:
                    winCount[currPlayer] += 1
                    if not training:
                        print("saving history")
                        with open("history.pkl","wb") as file:
                            pickle.dump(data,file)
                    if episode%1000 == 0:
                        print("wincount: {}".format(winCount))
                        print("time take for this epoch is {}".format(time.time() - startTime))
                        startTime = time.time()
                        timeInterval = 50
                        winCount = defaultdict(int)
                        g.save_hist_video("videos/gameabc{}.avi".format(episode))
                    if training:
                        self.__train(PG,data,episode,currPlayer)
                    break
        return winCount

    def __train(self,PG,data,episode,winner):
        a,o,r = np.empty((0,4)),np.array([]),np.array([])
        for player in data:
            if episode%1000 == 0:
                print("gathering data for player {}".format(player))
            actions, modelInput,rew = data[player].render(True)
            if player == winner:
                rew[np.where(rew == -1000)] = 1000
            
            if len(a):
                actions = np.vstack((actions,a))
                np.vstack((modelInput,o))
                np.hstack((rew,r))
            else:
                a = actions
                o = modelInput
                r = rew
        PG.episode_actions = a
        PG.episode_observations = o
        PG.episode_rewards = r
        PG.learn(episode,player,winner)
            
            
class PlotGraphs:
    def __init__(self,
                 graphDataPath,
                 path,
                 modelPath,
                 modelType,
                 e,
                 episodeStart,
                 players,
                 ghost_players,
                 policyPlayers,
                 randomPlayers,
                 dataPresent,
                 n_x=125,
                 n_y=4,
                 player_num=0
                ):
        print("mode input is {}".format(n_x))
        self.graphDataPath = graphDataPath
        self.player_num = player_num
        self.episode = e
        print(dataPresent)
        if not dataPresent:
            print("playing")
            self.__play(path,
                        modelPath,
                        modelType,
                        e,
                        episodeStart,
                        players,
                        n_x,
                        n_y,
                        ghost_players,
                        policyPlayers,
                        randomPlayers
                       )
        self.__plotGraphs()
        
    def __getEpisodeNum(self,modelName):
        i = len(modelName)-1
        while modelName[i].isdigit():
            i -= 1
        return modelName[i+1:]
    
    def __play(self,
               path,
               modelPath,
               modelType,
               episodes,
               episodeStart,
               players,
               n_x,
               n_y,
               ghost_players,
               policyPlayers,
               randomPlayers,
               player_num=0
              ):
        player = Play()
        randomMovesDict = defaultdict(dict)
        for i in [modelType]:
            print("-----------{}--------------".format(i))
            new_path = path.format(i)
            new_modelPath = modelPath.format(i,"{}")
            with open(new_path) as checkpoints:
                for cp in checkpoints:
                    print("---------------------------------")
                    cp = cp.split(":")[1]
                    _,modelName = os.path.split(cp)
                    modelName = modelName.strip(' "\'\t\r\n')
                    newModelPath = new_modelPath.format(modelName)
                    episodeNum = self.__getEpisodeNum(modelName)
                    tf.reset_default_graph()
                    randomMovesDict[i][episodeNum] = player.play(
                            policyPlayers = policyPlayers,
                            randomPlayers = randomPlayers,
                            load_path=newModelPath,
                            save_path= None,
                            episodes=episodes,
                            episodeStart=episodeStart,
                            training=False,
                            n_x=n_x,
                            n_y=n_y,
                            learning_rate=0.02,
                            reward_decay=0.99,
                            player_num = 0,
                            number_of_players=2,
                            number_of_pieces=4,
                            reward=-1000,
                            rewardType = "monte",
                            model2keep = 0,
                            ghost_players=ghost_players
                    )
            

            with open(self.graphDataPath,"wb") as file:
                pickle.dump(randomMovesDict,file)
                
    def __plotGraphs(self,random=False,winRate = False):
        with open(self.graphDataPath,"rb") as file:
            randomMovesDict = pickle.load(file)
        
        if random:
            self.__plotRandom(randomMovesDict)
        if winRate:
            self.__plotWinRate(randomMovesDict)
            
    def __plotRandom(self,randomMovesDict):
        fig,ax = plt.subplots(nrows=1,ncols=1)
        plt.xlabel('Epochs')
        plt.ylabel('Randomness')
        plt.title('Randomness VS Epochs')
        count = 0
        for _,d in randomMovesDict.items():
            x = np.array([[0.5,0.5]])
            for keys,values in d.items():
                values = np.array(values[1])
                ratio = sum(values[:,0])/sum(values[:,1])
                x = np.vstack((x,np.array([int(keys),ratio])))
            x = x[x[:,0].argsort()]
            x = np.delete(x,0,axis=0)
            ax.plot(x[:,0],x[:,1])
            count += 1
        print(np.mean(x[:,0]))
        
    def __plotWinRate(self,randomMovesDict):
        ax = None
        fig,ax = plt.subplots(nrows=1,ncols=1)
        plt.xlabel('Epochs')
        plt.ylabel('Winrate')
        plt.title('Winrate Change Over Training')
        count = 0
        print()
        for keys,values in randomMovesDict.items():
            x = np.array([[1,1]])
        #for keys,values in d.items():
            x = np.vstack((x,np.array([int(keys),values[self.player_num]])))
            x = x[x[:,0].argsort()]
            x = np.delete(x,0,axis=0)
            ax.plot(x[:,0],x[:,1]/self.episode)
            count += 1
        print(np.mean(x[:,1]/self.episode))