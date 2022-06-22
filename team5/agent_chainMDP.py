# TEAM : 5
# - HyeokJin Kwon
# - Seongmin Seo
# - Sungyoung Lee

import numpy as np
import tensorflow as tf
from tensorflow import keras

from collections import deque
import numpy as np
from tqdm import tqdm

import os

class Qfunction(keras.Model):

    def __init__(self, obssize, actsize, hidden_dims):

        super(Qfunction, self).__init__()

        initializer = keras.initializers.RandomUniform(minval=-1e-2, maxval=1e-2)

        self.input_layer = keras.layers.InputLayer(input_shape=(obssize,))

        self.hidden_layers = []
        for hidden_dim in hidden_dims:
            layer = keras.layers.Dense(hidden_dim, activation='relu',
                                       kernel_initializer=initializer)
            self.hidden_layers.append(layer)
        self.output_layer = keras.layers.Dense(actsize)

    @tf.function
    def call(self, states):
        x = self.input_layer(states)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        return self.output_layer(x)


# Wrapper class for training Qfunction and updating weights (target network)
class DQN(object):

    def __init__(self, obssize, actsize, hidden_dims, optimizer):
        self.qfunction = Qfunction(obssize, actsize, hidden_dims)
        self.optimizer = optimizer
        self.obssize = obssize
        self.actsize = actsize

    def _predict_q(self, states, actions):
        q = []
        for j in range(len(actions)):
            q.append(self.qfunction(states)[j][actions[j]])
        return tf.convert_to_tensor(q, dtype=tf.float32)

    def _loss(self, Qpreds, targets):
        l = tf.math.reduce_mean(tf.square(Qpreds - targets))
        return l

    def compute_Qvalues(self, states):
        inputs = np.atleast_2d(states.astype('float32'))
        return self.qfunction(inputs)

    def train(self, states, actions, targets):
        with tf.GradientTape() as tape:
            Qpreds = self._predict_q(states, actions)
            loss = self._loss(Qpreds, targets)
        variables = self.qfunction.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def update_weights(self, from_network):

        from_var = from_network.qfunction.trainable_variables
        to_var = self.qfunction.trainable_variables

        # soft assign
        for v1, v2 in zip(from_var, to_var):
            v2.assign(0.8 * v1 + 0.2 * v2)


# Implement replay buffer
class ReplayBuffer(object):

    def __init__(self, maxlength):
        self.buffer = deque()
        self.number = 0
        self.maxlength = maxlength

    def append(self, experience):
        self.buffer.append(experience)
        self.number += 1
        if (self.number > self.maxlength):
            self.pop()

    def pop(self):
        while self.number > self.maxlength:
            self.buffer.popleft()
            self.number -= 1

    def sample(self, batchsize):
        inds = np.random.choice(len(self.buffer), batchsize, replace=False)
        return [self.buffer[idx] for idx in inds]


### DQN implementation ###
class agent():

    def __init__(self):
        self.hidden_dim = [8, 4]
        self.lr = 5e-3
        self.ensemble_num = 3
        print("chainMDP agent made.")


    def action(self, s):
        self.s = s
        voting_paper = np.zeros(2)

        if np.random.rand() < self.eps:
            return np.random.choice([0, 1])
        else:
            for n in range(self.ensemble_num):
                Q = self.Qs[n][0].compute_Qvalues(np.array(self.s))
                action = np.argmax(Q)  # always max action choose
                # voting_paper[action] += 1
                voting_paper[action] += (Q[0][action] - np.mean(Q[0])) / np.std(Q[0])
            return np.argmax(voting_paper)


    def load_weights(self):
        print("loading existing model..")
        self.Qs = []
        for i in range(self.ensemble_num):
            Qprin = DQN(10, 2, self.hidden_dim, optimizer = keras.optimizers.Adam(learning_rate=self.lr))
            Qtarg = DQN(10, 2, self.hidden_dim, optimizer = keras.optimizers.Adam(learning_rate=self.lr))
            Qprin.qfunction = keras.models.load_model(os.getcwd()+"/model/chainMDP"+'/Qprin_'+str(i))
            Qtarg.qfunction = keras.models.load_model(os.getcwd()+"/model/chainMDP"+'/Qtarg_'+str(i))

            self.Qs.append([Qprin, Qtarg])
        
        self.eps = 0


    def train(self, episodes, env, no_render):

        self.env = env
        self.state = self.env.reset()
        self.episode_length = episodes


        ### For Q value training ###
        self.eps = 1

        self.bernoulli_prob = 0.3

        self.Qprin = DQN(self.env.n, self.env.action_space.n, self.hidden_dim,
                         optimizer=keras.optimizers.Adam(learning_rate=self.lr))
        self.Qtarg = DQN(self.env.n, self.env.action_space.n, self.hidden_dim,
                         optimizer=keras.optimizers.Adam(learning_rate=self.lr))
        self.Qs = []
        for _ in range(self.ensemble_num):
            self.Qs.append([self.Qprin, self.Qtarg])
        totalstep = 0
        initialize = self.env.n * 30
        eps = 1
        eps_minus = 1e-2
        tau = self.env.n * 10
        gamma = 0.99
        batchsize = 64
        buff_max_size = 10000
        buffer = ReplayBuffer(buff_max_size)
        ############################

        self.r_record = []
        # === For additional rewards for less visited states === #
        state_visit_history = np.zeros(self.env.observation_space.n)
        # ====================================================== #
        for iter in tqdm(range(self.episode_length)):
            self.state = self.env.reset()
            done = False
            rsum = 0

            while not done:
                totalstep += 1

                #####################
                ### Epsilon Decay ###
                # #####################
                # if np.sum(np.array(self.r_record) >= 1) >= 1:
                if eps > 0.05 and totalstep > initialize:
                    eps -= eps_minus
                elif eps < 0.05 and totalstep > initialize:
                    eps = 0

                ##################
                ### Get Action ###
                ##################
                if np.random.rand() < eps or totalstep <= initialize:
                    action = np.random.choice([0, 1])
                else:
                    voting_paper = np.zeros(self.env.action_space.n)

                    for n in range(self.ensemble_num):
                        Q = self.Qs[n][0].compute_Qvalues(np.array(self.state))
                        action = np.argmax(Q)  # always max action choose
                        # voting_paper[action] += 1
                        voting_paper[action] += (Q[0][action] - np.mean(Q[0])) / np.std(Q[0])

                    action = np.argmax(voting_paper)
                ##################

                ##################
                ###  ONE STEP  ###
                ##################
                curr_state = self.state
                next_state, reward, done, _ = self.env.step(action)
                rsum += reward
                ##################
                # ===== additional reward for less visited states ===== #
                pos = np.count_nonzero(next_state)
                reward += np.exp(-(state_visit_history[pos-1]//self.env.n))
                state_visit_history[pos-1] += 1
                #print(state_visit_history, eps, iter)
                # ===================================================== #

                # === ensemble DQN === #
                heads = np.random.binomial(1, self.bernoulli_prob, self.ensemble_num)
                if np.sum(heads) == 0:
                    heads[np.random.randint(0, self.ensemble_num)] = 1
                # ==================== #

                #####################
                ### Update Buffer ###
                #####################
                buffer.append((curr_state, action, reward, next_state, done, heads))
                #####################

                #############################
                ### N Samples from Buffer ###
                ###         and           ###
                ### Update theta of Qprin ###
                #############################
                if totalstep > initialize:

                    # sample
                    s = buffer.sample(batchsize)

                    d = []
                    for j in range(len(s)):  # for each s[j]s
                        ## if head 0 : append 0 at K, head 1 : append value of k to K
                        ## iterate for all samples
                        K = []
                        cS = s[j][0];
                        A = s[j][1];
                        R = s[j][2];
                        nS = s[j][3];
                        DONE = s[j][4];
                        heads = s[j][5];
                        if not DONE:
                            for n in range(len(heads)):
                                if heads[n] == 0:
                                    k = 0
                                else:
                                    k = R + gamma * np.max(self.Qs[n][1].compute_Qvalues(nS))  # Qtarg_n
                                K.append(k)
                        elif DONE:
                            for n in range(len(heads)):
                                if heads[n] == 0:
                                    k = 0
                                else:
                                    k = R
                                K.append(k)
                        d.append(K)

                    # update Qprins
                    for n in range(self.ensemble_num):
                        set_of_S = np.array([s[x][0] for x in range(len(s)) if s[x][5][n] == 1])
                        set_of_A = np.array([s[x][1] for x in range(len(s)) if s[x][5][n] == 1])
                        D = [d[x][n] for x in range(len(s)) if s[x][5][n] == 1]

                        self.Qs[n][0].train(set_of_S, set_of_A, tf.convert_to_tensor(D, dtype=tf.float32))  # Qprin
                #############################

                #############################
                ### Update theta of Qtarg ###
                #############################
                if totalstep % tau == 0:
                    for n in range(self.ensemble_num):
                        self.Qs[n][1].update_weights(self.Qs[n][0])
                #############################

                pass

            #########################
            ### Sample Efficiency ###
            #########################
            self.r_record.append(rsum)

        self.eps = eps

        # for i in range(len(self.Qs)):
        #     self.Qs[i][0].qfunction.save(os.getcwd()+"/model/chainMDP"+'/Qprin_'+str(i))
        #     self.Qs[i][1].qfunction.save(os.getcwd()+"/model/chainMDP"+'/Qtarg_'+str(i))

        return self.r_record