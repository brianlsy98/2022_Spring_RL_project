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
        """
        obssize: dimension of state space
        actsize: dimension of action space
        hidden_dims: list containing output dimension of hidden layers 
        """
        super(Qfunction, self).__init__()

        # Layer weight initializer
        initializer = keras.initializers.RandomUniform(minval=-1., maxval=1.)

        # Input Layer
        self.input_layer = keras.layers.InputLayer(input_shape=(obssize,))
        
        # Hidden Layer
        self.hidden_layers = []
        for hidden_dim in hidden_dims:
            # TODO: define each hidden layers
            layer = keras.layers.Dense(hidden_dim, activation='relu',
                                      kernel_initializer=initializer)
            self.hidden_layers.append(layer) 
        
        # Output Layer : 
        # TODO: Define the output layer.
        self.output_layer = keras.layers.Dense(actsize) 

    @tf.function
    def call(self, states):
        ########################################################################
        # TODO: You SHOULD implement the model's forward pass
        x = self.input_layer(states)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        return self.output_layer(x)
        ########################################################################

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
        """
        states represent s_t
        actions represent a_t
        """
        ########################################################################
        # TODO: Define the logic for calculate  Q_\theta(s,a)
        q = []
        for j in range(len(actions)):
           q.append(self.qfunction(states)[j][actions[j]])
        return tf.convert_to_tensor(q, dtype=tf.float32)
        ########################################################################
        

    def _loss(self, Qpreds, targets):
        """
        Qpreds represent Q_\theta(s,a)
        targets represent the terms E[r+gamma Q] in Bellman equations

        This function is OBJECTIVE function
        """
        l = tf.math.reduce_mean(tf.square(Qpreds - targets))
        return l

    
    def compute_Qvalues(self, states):
        """
        states: numpy array as input to the neural net, states should have
        size [numsamples, obssize], where numsamples is the number of samples
        output: Q values for these states. The output should have size 
        [numsamples, actsize] as numpy array
        """
        inputs = np.atleast_2d(states.astype('float32'))
        return self.qfunction(inputs)


    def train(self, states, actions, targets):
        """
        states: numpy array as input to compute loss (s)
        actions: numpy array as input to compute loss (a)
        targets: numpy array as input to compute loss (Q targets)
        """
        with tf.GradientTape() as tape:
            Qpreds = self._predict_q(states, actions)
            loss = self._loss(Qpreds, targets)
        variables = self.qfunction.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def update_weights(self, from_network):
        """
        We need a subroutine to update target network 
        i.e. to copy from principal network to target network. 
        This function is for copying  theta -> theta target 
        """
        
        from_var = from_network.qfunction.trainable_variables
        to_var = self.qfunction.trainable_variables
        
        # soft assign
        for v1, v2 in zip(from_var, to_var):
            v2.assign(0.8*v1+0.2*v2)

# Implement replay buffer
class ReplayBuffer(object):
    
    def __init__(self, maxlength):
        """
        maxlength: max number of tuples to store in the buffer
        if there are more tuples than maxlength, pop out the oldest tuples
        """
        self.buffer = deque()
        self.number = 0
        self.maxlength = maxlength
    
    def append(self, experience):
        """
        this function implements appending new experience tuple
        experience: a tuple of the form (s,a,r,s^\prime)
        """
        self.buffer.append(experience)
        self.number += 1
        if(self.number > self.maxlength):
            self.pop()
        
    def pop(self):
        """
        pop out the oldest tuples if self.number > self.maxlength
        """
        while self.number > self.maxlength:
            self.buffer.popleft()
            self.number -= 1
    
    def sample(self, batchsize):
        """
        this function samples 'batchsize' experience tuples
        batchsize: size of the minibatch to be sampled
        return: a list of tuples of form (s,a,r,s^\prime)
        """
        inds = np.random.choice(len(self.buffer), batchsize, replace=False)
        return [self.buffer[idx] for idx in inds]






### DQN implementation ###
class agent():
    
    def __init__(self, e):

        self.env = e
        self.state = self.env.reset()

        ### For Q value training ###        
        self.episode_length = 1000
        self.hidden_dim = [8, 4]
        self.lr = 5e-4
        self.Qprin = DQN(10, e.action_space.n, [8, 4], optimizer = keras.optimizers.Adam(learning_rate=self.lr))
        self.Qtarg = DQN(10, e.action_space.n, [8, 4], optimizer = keras.optimizers.Adam(learning_rate=self.lr))
        ############################

        return
    
    def action(self):
        
        Q = self.Qprin.compute_Qvalues(np.array(self.state))
        action = np.argmax(Q)   # always max action choose

        return action

    def train(self):

        ### For Q value training ###
        totalstep = 0
        initialize = 500
        eps = 1; eps_minus = 1e-4
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
                    Q = self.Qprin.compute_Qvalues(np.array(self.state))
                    action = np.argmax(Q)   # always max action choose
                ##################

                ##################
                ###  ONE STEP  ###
                ##################
                curr_state = self.state
                next_state, reward, done, _ = self.env.step(action)
                rsum += reward
                ##################
                
                #####################
                ### Update Buffer ###
                #####################
                buffer.append((curr_state, action, reward, next_state, done))
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
                    for j in range(len(s)):
                        cS = s[j][0]; A = s[j][1]; R = s[j][2]; nS = s[j][3]; DONE = s[j][4]
                        if not DONE:
                            k = R + gamma*np.max(self.Qtarg.compute_Qvalues(np.array(nS)))
                        elif DONE:
                            k = R
                        d.append(k)
                    
                    set_of_S = np.array([s[x][0] for x in range(len(s))])
                    set_of_A = np.array([s[x][1] for x in range(len(s))])
                    loss = self.Qprin.train(set_of_S, set_of_A, tf.convert_to_tensor(d, dtype=tf.float32))
                #############################


                #############################
                ### Update theta of Qtarg ###
                #############################
                if totalstep % tau == 0:
                    print("")
                    print("epsilon : ", eps)
                    print("target updated, totalstep : ", totalstep)
                    self.Qtarg.update_weights(self.Qprin)
                #############################

                pass
            

            r_record.append(rsum)
            if iter % 10 == 0:
                print('iteration {} ave reward {}'.format(iter, np.mean(r_record[-10:])))

        return