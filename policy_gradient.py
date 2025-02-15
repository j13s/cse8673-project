"""
Policy Gradient Reinforcement Learning
Uses a 3 layer neural network as the policy network

"""
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

class PolicyGradient:
    def __init__(
        self,
        n_x,
        n_y,
        learning_rate=0.01,
        reward_decay=0.95,
        load_path=None,
        save_path=None,
        player_num = "",
        rewardType = "monte",
        toKeep = 100
    ):

        self.n_x = n_x
        self.n_y = n_y
        self.lr = learning_rate
        self.gamma = reward_decay

        self.save_path = None
        self.rewardType = rewardType
        if save_path is not None:
            self.save_path = save_path

        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []
        
        self.build_network(player_num)

        self.cost_history = []

        self.sess = tf.Session()

        # $ tensorboard --logdir=logs
        # http://0.0.0.0:6006/
        tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver(max_to_keep=toKeep)

        # Restore model
        if load_path is not None:
            self.load_path = load_path
            self.saver.restore(self.sess, self.load_path)

    def store_transition(self, s, a, r):
        """
            Store play memory for training

            Arguments:
                s: observation
                a: action taken
                r: reward after action
        """
        self.episode_observations.append(s)
        self.episode_rewards.append(r)

        # Store actions as list of arrays
        # e.g. for n_y = 2 -> [ array([ 1.,  0.]), array([ 0.,  1.]), array([ 0.,  1.]), array([ 1.,  0.]) ]
        action = np.zeros(self.n_y)
        action[a] = 1
        self.episode_actions.append(action)


    def choose_action(self, observation, actions,punish=1):
        """
            Choose action based on observation

            Arguments:
                observation: array of state, has shape (num_features)

            Returns: index of action we want to choose
        """
        # Reshape observation to (num_features, 1)
        observation = observation[:, np.newaxis]
        # Run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.outputs_softmax, feed_dict = {self.X: observation})
        prob = prob_weights[0]
        mask = np.ones(prob.shape,dtype=bool)
        mask[actions] = False
        prob[mask] = 0
        if np.sum(prob) != 0:
            prob = prob/np.sum(prob)
            action = np.random.choice(range(len(prob.ravel())), p=prob.ravel())
        else:
            action = np.random.choice(actions)
        return action,prob_weights

    def learn(self,episode,player,winner,singleReward=None):
        # Discount and normalize episode reward
        if singleReward is None:
            discounted_episode_rewards_norm =self.discount_and_norm_rewards(player,winner)
        # Train on episode
        else:
            discounted_episode_rewards_norm = [singleReward]
        self.sess.run(self.train_op, feed_dict={
             self.X: np.vstack(self.episode_observations).T,
             self.Y: np.vstack(np.array(self.episode_actions)).T,
             self.discounted_episode_rewards_norm: discounted_episode_rewards_norm,
        })

        # Reset the episode data
        self.episode_observations, self.episode_actions, self.episode_rewards  = [], [], []

        # Save checkpoint
        if self.save_path is not None and episode%1000 == 0 and singleReward is None:
            save_path = self.saver.save(self.sess, 
                                        self.save_path+str(episode))
            print("Model saved in file: %s" % save_path)

        return discounted_episode_rewards_norm

    def discount_and_norm_rewards(self,player,winner):
        discounted_episode_rewards = np.zeros_like(self.episode_rewards)
        cumulative = 0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.gamma + self.episode_rewards[t]
            discounted_episode_rewards[t] = cumulative
        discounted_episode_rewards = np.float_(discounted_episode_rewards)
        d = discounted_episode_rewards
        try:
            discounted_episode_rewards -= np.mean(discounted_episode_rewards)
            discounted_episode_rewards /= np.std(discounted_episode_rewards)
        except:
            import pdb; pdb.set_trace()
            pass
        if player != winner:
            d = d*-1
            d = np.ones(d.shape)*-0.2
        else:
            d = np.ones(d.shape)
        return d
        #return discounted_episode_rewards
    def build_network(self,player_num):
        # Create placeholders
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=(self.n_x, None), name="X")
            self.Y = tf.placeholder(tf.float32, shape=(self.n_y, None), name="Y")
            self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # Initialize parameters
        units_layer_1 = 30
        units_layer_2 = 30
        units_layer_3 = 30
        units_layer_4 = 10
        units_layer_5 = 10
        units_layer_6 = 10
        units_output_layer = self.n_y
        with tf.name_scope('parameters'):
            W1 = tf.get_variable("W1"+str(player_num), [units_layer_1, self.n_x], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b1 = tf.get_variable("b1"+str(player_num), [units_layer_1, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))

            W2 = tf.get_variable("W2"+str(player_num), [units_layer_2, units_layer_1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b2 = tf.get_variable("b2"+str(player_num), [units_layer_2, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            
            W3 = tf.get_variable("W3"+str(player_num), [units_layer_3, units_layer_2], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b3 = tf.get_variable("b3"+str(player_num), [units_layer_3, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            """
            W4 = tf.get_variable("W4"+str(player_num), [units_layer_4, units_layer_3], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b4 = tf.get_variable("b4"+str(player_num), [units_layer_4, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            
            W5 = tf.get_variable("W5"+str(player_num), [units_layer_5, units_layer_4], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b5 = tf.get_variable("b5"+str(player_num), [units_layer_5, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            
            W6 = tf.get_variable("W6"+str(player_num), [units_layer_6, units_layer_5], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b6 = tf.get_variable("b6"+str(player_num), [units_layer_6, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            """
            ####connection from the fourth layer 
            W7 = tf.get_variable("W7"+str(player_num), [self.n_y, units_layer_3], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b7 = tf.get_variable("b7"+str(player_num), [self.n_y, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))

        # Forward prop
        with tf.name_scope('layer_1'):
            Z1 = tf.add(tf.matmul(W1,self.X), b1)
            A1 = tf.nn.relu(Z1)
        with tf.name_scope('layer_2'):
            Z2 = tf.add(tf.matmul(W2, A1), b2)
            A2 = tf.nn.relu(Z2)
        with tf.name_scope('layer_3'):
            Z3 = tf.add(tf.matmul(W3, A2), b3)
            A3 = tf.nn.relu(Z3)
        """
        with tf.name_scope('layer_4'):
            Z4 = tf.add(tf.matmul(W4, A3), b4)
            A4 = tf.nn.relu(Z4)
        
        with tf.name_scope('layer_5'):
            Z5 = tf.add(tf.matmul(W5, A4), b5)
            A5 = tf.nn.relu(Z5)
        with tf.name_scope('layer_6'):
            Z6 = tf.add(tf.matmul(W6, A5), b6)
            A6 = tf.nn.relu(Z6)
        """
        with tf.name_scope('layer_7'):
            Z7 = tf.add(tf.matmul(W7, A3), b7)
            A7 = tf.nn.softmax(Z7)

        # Softmax outputs, we need to transpose as tensorflow nn functions expects them in this shape
        logits = tf.transpose(Z7)
        labels = tf.transpose(self.Y)
        self.outputs_softmax = tf.nn.softmax(logits, name='A7')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_norm)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def plot_cost(self):
        import matplotlib
        matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()

