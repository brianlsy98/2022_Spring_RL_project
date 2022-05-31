import tensorflow as tf
from tensorflow import keras

from collections import deque
import numpy as np
import matplotlib.pyplot as plt

import gym
import lava_grid as lava


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
        for hidden in hidden_dims:
            linear = keras.layers.Dense(hidden, activation='linear',
                                        kernel_initializer=initializer)
            layer = keras.layers.Dense(hidden, activation='relu',
                                      kernel_initializer=initializer)
            
            self.hidden_layers.append(linear)
            self.hidden_layers.append(layer)
            
        
        # Output Layer : 
        # TODO: Define the output layer.
        self.output_layer = keras.layers.Dense(actsize, activation='linear',
                                               kernel_initializer=initializer)


    @tf.function
    def call(self, states):
        ########################################################################
        # TODO: You SHOULD implement the model's forward pass

          o=self.input_layer(states)
          for layer in self.hidden_layers:
            o=layer(o)
          output=self.output_layer(o)

          return output
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
        # TODO: Define the logic for calculate  Q_\theta(s,a

        q_s=self.compute_Qvalues(states)
        q_sa=[]
        for i in range(len(states)):
          q_sa.append(q_s[i][actions[i]])
        q_sa=tf.convert_to_tensor(q_sa,dtype=tf.float32)

        return q_sa
        ########################################################################
        

    def _loss(self, Qpreds, targets):
        """
        Qpreds represent Q_\theta(s,a)
        targets represent the terms E[r+gamma Q] in Bellman equations

        This function is OBJECTIVE function            self.hidden_layers.append(layer)

        """
        return tf.math.reduce_mean(tf.square(Qpreds - targets))

    
    def compute_Qvalues(self, states):
        """
        states: numpy array as input to the neural net, states should have
        size [numsamples, obssize], where numsamples is the number of samples
        output: Q values for these states. The output should have size 
        [numsamples, actsize] as numpy array
        """
        inputs = np.atleast_2d(states)
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
        This function is for copying  𝜃←𝜃target 
        """
        
        from_var = from_network.qfunction.trainable_variables
        to_var = self.qfunction.trainable_variables
        
        for v1, v2 in zip(from_var, to_var):
            v2.assign(v1)

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

"""Now that we have all the ingredients for DQN, we can write the main procedure to train DQN on a given environment. The implementation is straightforward if you follow the pseudocode. Refer to the pseudocode pdf for details."""

################################################################################
lr = 0.001  # learning rate for gradient update 
batchsize = 64  # batchsize for buffer sampling
maxlength = 2000  # max number of tuples held by buffer
tau = 100  # time steps for target update
episodes = 3000  # number of episodes to run
initialize = 1000  # initial time steps before start updating
epsilon = 1.0  # constant for exploration
decay=0.999
e_min=0.01
e_mid=0.4
gamma = .95  # discount
hidden_dims=[128,64,16] # hidden dimensions

max_steps=100
stochasticity=0
no_render=True
################################################################################

# initialize environment
env = lava.ZigZag6x10(max_steps=max_steps, act_fail_prob=stochasticity, goal=(5, 9), numpy_state=False)
obssize = env.observation_space.n
actsize = env.action_space.n

# optimizer
optimizer = keras.optimizers.Adam(learning_rate=lr)

# initialize networks
Qprincipal = DQN(obssize, actsize, hidden_dims, optimizer)
Qtarget = DQN(obssize, actsize, hidden_dims, optimizer)

# initialization of buffer
buffer = ReplayBuffer(maxlength)

################################################################################
# TODO: Complete the main iteration
# CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.

step_list = []
rrecord = []
totalstep = 0
for ite in range(episodes):
    obs = env.reset()
    obs = np.eye(env.observation_space.n)[0]
    done = False
    rsum = 0.0

    while not done:
      totalstep+=1
      state=obs

      if np.random.rand() < epsilon:
          action = env.action_space.sample()
      else:
          Q = Qprincipal.compute_Qvalues(obs)
          action = np.argmax(Q[0])

      obs, reward, done, info = env.step(action)
      rsum += reward
      exp=(state,action,reward,obs,done)
      buffer.append(exp)

      if (totalstep>initialize):
        if (epsilon>e_mid):
            epsilon*=decay
        
        samples=buffer.sample(batchsize)
        states=[]
        actions=[] 
        d=[]
        for i in range(len(samples)):
          (s,a,r,s_p,term) = samples[i]
          if not term:
            d_i=r+gamma*np.max(Qtarget.compute_Qvalues(s_p)[0])
          else:
            d_i=r
          states.append(s)
          actions.append(a)
          d.append(d_i)
        states=np.stack([states[i] for i in range(len(states))],axis=0)
        actions=np.asarray(actions)
        
        d=tf.convert_to_tensor(d, dtype=tf.float32)

        Qprincipal.train(states, actions, d)

      if (totalstep%tau==0):
        Qtarget.update_weights(Qprincipal)

################################################################################
    step_list.append(env._steps)
    ## DO NOT CHANGE THIS PART!
    rrecord.append(rsum)
    if ite % 10 == 0:
        if np.mean(rrecord[-10:])>0:
            epsilon=max(epsilon*decay, e_min)
        print('iteration {} ave reward {} / ave step {}'.format(ite, np.mean(rrecord[-10:]),np.mean(step_list[-10:])))
        print('final state : \n{}\n'.format(obs.reshape(6,10)))
    
    ave100 = np.mean(rrecord[-100:])   
    if  ave100 > 0.0:
        print("Solved after %d episodes."%ite)
        break

# plot [episode, reward] history
x = [i+1 for i in range(len(rrecord))]
plt.plot(x, rrecord)
plt.title('episode rewards')
plt.xlabel('episodes')
plt.ylabel('rewards')
plt.show()