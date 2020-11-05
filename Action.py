import numpy as np
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
            return None
        random = False
        if self.num_players != 4:
            enemy_pieces = self.__getEnemy()
        else:
            enemy_pieces = self.enemy_pieces
        observation = np.vstack((self.player_pieces[:,np.newaxis],\
                                enemy_pieces[0][:,np.newaxis]))
        observation = (observation + 1)/60
        observation = np.vstack((observation,self.dice/6))
        observation = observation.reshape([(self.num_players*\
                                            self.num_of_pieces)+1,])
        action = self.PG.choose_action(observation)
        if action not in self.move_pieces:
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
        