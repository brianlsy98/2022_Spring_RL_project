# TEAM : 5
# - HyeokJin Kwon
# - Seongmin Lee
# - Sungyoung Lee
# Code by Sungyoung Lee

import tensorflow as tf
from tensorflow import keras

from collections import deque
import numpy as np


class Qfunction(keras.Model):
    
    def __init__(self, obssize, actsize, hidden_dims):
        super(Qfunction, self).__init__()

        # Layer weight initializer
        initializer = keras.initializers.RandomUniform(minval=-1., maxval=1.)

        # Input Layer
        self.input_layer = keras.layers.InputLayer(input_shape=(obssize,))
        
        # Hidden Layer
        self.hidden_layers = []
        for hidden_dim in hidden_dims:
            layer = keras.layers.Dense(hidden_dim, activation='relu',
                                      kernel_initializer=initializer)
            self.hidden_layers.append(layer) 
        
        # Output Layer : 
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
        """
        obssize: dimension of state space
        actsize: dimension of action space
        optimizer: 
        """
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
            v2.assign(0.8*v1+0.2*v2)

# Implement replay buffer
class ReplayBuffer(object):
    
    def __init__(self, maxlength):

        self.buffer = deque()
        self.number = 0
        self.maxlength = maxlength
    
    def append(self, experience):

        self.buffer.append(experience)
        self.number += 1
        if(self.number > self.maxlength):
            self.pop()
        
    def pop(self):

        while self.number > self.maxlength:
            self.buffer.popleft()
            self.number -= 1
    
    def sample(self, batchsize):

        inds = np.random.choice(len(self.buffer), batchsize, replace=False)
        return [self.buffer[idx] for idx in inds]






### ensembleDQN implementation ###
class agent():
    
    def __init__(self, e, head_num):

        self.env = e
        self.state = self.env.reset()

        ### For Q value training ###        
        self.episode_length = 100
        self.hidden_dim = [8, 4]
        self.lr = 1e-3

        self.bernoulli_prob = 0.9
        self.ensemble_num = head_num


        self.Qs = []
        for _ in range(self.ensemble_num):
            Qprin = DQN(10, e.action_space.n, self.hidden_dim, optimizer = keras.optimizers.Adam(learning_rate=self.lr))
            Qtarg = DQN(10, e.action_space.n, self.hidden_dim, optimizer = keras.optimizers.Adam(learning_rate=self.lr))
            self.Qs.append([Qprin, Qtarg])

        return
    

    # Action :
    # - train : head fixed for each epoch
    # - eval : vote
    def action(self, s):
        voting_paper = np.zeros(self.env.action_space.n)
        
        for n in range(self.ensemble_num):
            Q = self.Qs[n][0].compute_Qvalues(np.array(s))
            action = np.argmax(Q)   # always max action choose
            # voting_paper[action] += 1
            voting_paper[action] += Q[0][action] - np.mean(Q[0])

        return np.argmax(voting_paper)


    def train(self):

        ### For Q value training ###
        totalstep = 0
        initialize = 500
        eps = 1; eps_minus = 5e-4
        tau = 100
        gamma = 0.99
        batchsize = 64
        buff_max_size = 10000
        buffer = ReplayBuffer(buff_max_size)
        ############################

        r_record = []

        for iter in range(self.episode_length):
            self.state = self.env.reset()
            done = False
            rsum = 0

            # Action :
            # - train : head fixed for each epoch
            # - eval : vote
            head4action_train = np.random.randint(0, self.ensemble_num)

            while not done:
                totalstep += 1

                if eps > 0.05 and totalstep > initialize: eps -= eps_minus
                elif eps < 0.05 and totalstep > initialize: eps = 0.05

                ##################
                ### Get Action ###
                ##################
                if np.random.rand() < eps or totalstep <= initialize:
                    action = np.random.choice([0, 1])
                else:
                    Q = self.Qs[head4action_train][0].compute_Qvalues(np.array(self.state)) # Qprin
                    action = np.argmax(Q)   # always max action choose
                ##################

                ##################
                ###  ONE STEP  ###
                ##################
                curr_state = self.state
                next_state, reward, done, _ = self.env.step(action)
                rsum += reward
                ##################
                
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
                    for j in range(len(s)):     # for each s[j]s
                        ## if head 0 : append 0 at K, head 1 : append value of k to K
                        ## iterate for all samples
                        K = []
                        cS = s[j][0]; A = s[j][1]; R = s[j][2]; nS = s[j][3]; DONE = s[j][4]; heads = s[j][5]
                        if not DONE:
                            for n in range(len(heads)):
                                if heads[n] == 0 : k = 0
                                else : k = R + gamma*np.max(self.Qs[n][1].compute_Qvalues(np.array(nS)))    # Qtarg
                                K.append(k)
                        elif DONE:
                            for n in range(len(heads)):
                                if heads[n] == 0 : k = 0
                                else : k = R
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
                    print("")
                    print("epsilon : ", eps)
                    print("target updated, totalstep : ", totalstep)
                    for n in range(self.ensemble_num):
                        self.Qs[n][1].update_weights(self.Qs[n][0])
                #############################

                pass
            

            r_record.append(rsum)
            if iter % 10 == 0:
                print('iteration {} ave reward {}'.format(iter, np.mean(r_record[-10:])))

        return