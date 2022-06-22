# TEAM : 5
# - HyeokJin Kwon
# - Seongmin Seo
# - Sungyoung Lee

# Referred to the example code from keras.io (cartpole ppo example)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import scipy.signal
import os
from matplotlib import pyplot as plt

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage .
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:

    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize he advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer)
        )
        self.advantage_buffer = (self.advantage_buffer -advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )



class agent():      # PPO agent

    def __init__(self):
        print("Lava agent made.")

    def mlp(self, x, sizes, activation=tf.tanh, output_activation=None):
        # Build a feedforward neural network
        for size in sizes[:-1]:
            x = layers.Dense(units=size, activation=activation)(x)
        return layers.Dense(units=sizes[-1], activation=output_activation)(x)


    def logprobabilities(self, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.num_actions) * logprobabilities_all, axis=1
        )
        return logprobability


    # Sample action from actor
    @tf.function
    def sample_action(self, observation):
        logits = self.actor(observation)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action


    # Train the policy by maximizing the PPO-Clip objective
    @tf.function
    def train_policy(self,
        observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
    ):

        with tf.GradientTape() as tape:
            ratio = tf.exp(
                self.logprobabilities(self.actor(observation_buffer), action_buffer)
                - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        kl = tf.reduce_mean(
            logprobability_buffer
            - self.logprobabilities(self.actor(observation_buffer), action_buffer)
        )
        kl = tf.reduce_sum(kl)
        return kl


    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(self, observation_buffer, return_buffer):
        with tf.GradientTape() as tape: # Record operations for automatic differentiation
            value_loss = tf.reduce_mean((return_buffer - self.critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))


    def action(self, s):
        if isinstance(s, np.int64):
            shape = self.actor.layers[0].input_shape[0]
            obs = np.zeros(shape[1])
            obs[s] += 1
            obs = obs.reshape(1,-1)
            _, action = self.sample_action(obs)
        else:
            obs = s
            obs = obs.reshape(1,-1)
            _, action = self.sample_action(obs)
        return action[0].numpy()

    def load_weights(self):
        print("loading existing model..")
        self.actor = keras.models.load_model(os.getcwd()+"/model/lava"+'/lava_actor')    # load policy
        self.critic = keras.models.load_model(os.getcwd()+"/model/lava"+'/lava_critic')  # load value-func


    def train(self, episodes, env, no_render):
        
        result = []
        self.epochs = episodes
        self.buffer_size = 4000

        # === Hyperparameters of the PPO algorithm === #
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.policy_learning_rate = 3e-3
        self.value_function_learning_rate = 1e-3
        self.hidden_sizes = (128, 128)

        self.train_policy_iterations = 80
        self.train_value_iterations = 80
        self.lam = 0.97
        self.target_kl = 0.05
        # ============================================ #


        # ==== Initialize ==== #
        self.observation_dimensions = env.observation_space.n
        self.num_actions = env.action_space.n

        self.buffer = Buffer(self.observation_dimensions, self.buffer_size)

        self.observation_input = keras.Input(shape=(self.observation_dimensions,), dtype=tf.float32)
        self.logits = self.mlp(self.observation_input, list(self.hidden_sizes) + [self.num_actions], tf.tanh, None)
        self.actor = keras.Model(inputs=self.observation_input, outputs=self.logits)
        self.value = tf.squeeze(
            self.mlp(self.observation_input, list(self.hidden_sizes) + [1], tf.tanh, None), axis=1
        )
        self.critic = keras.Model(inputs=self.observation_input, outputs=self.value)

        self.policy_optimizer = keras.optimizers.Adam(learning_rate=self.policy_learning_rate)
        self.value_optimizer = keras.optimizers.Adam(learning_rate=self.value_function_learning_rate)
        # ============================================ #

        observation, epoch_return = env.reset(), 0
        
        epoch = 0
        min_reward = 0
        
        while True:
            
            num_episodes = 0

            obs_only_state = np.zeros((env._shape[0]*env._shape[1],))

            prev_t = 0
            goal_reached = 0    # Coarse Control at least 1 time every buffer size(4000) steps
            for t in range(self.buffer_size):
                if not no_render:
                    env.render()

                # ====== observation accumulation ===== #
                obs_only_state[observation] += 1
                # ===================================== #  

                # obs : one-hot array
                # observation : integer
                obs = np.zeros((env._shape[0]*env._shape[1],))
                obs[observation] += 1
                obs = obs.reshape(1,-1)
                logits, action = self.sample_action(obs)
                observation_new, reward, done, _ = env.step(action[0].numpy())

                epoch_return += reward

                # ======= reward -100 if reached to same place 3 times after action ========= #
                # ======= *********** key idea for training time decrease *********** ======= #
                ########### COARSE CONTROL ###########
                if reward < min_reward : min_reward = reward
                if done != True and goal_reached == 0: reward = 0
                if goal_reached == 0:   # reward implementation until first goal reach
                    for state in obs_only_state :
                        if state > 2 : reward -= 1*abs(min_reward); done = True
                        elif state > 1 : reward -= state*abs(min_reward)/len(obs_only_state)
                        elif state == 1 : reward += abs(min_reward)*max(1/len(obs_only_state), 0.01)   # 1/60 : this should be higher than 0.01
                
                    if np.where(observation_new == 1)[0] == env.goal[0]*env._shape[1]+env.goal[1]:
                        # 2*(env._shape[0]+env._shape[1]) : longest dist to goal
                        reward += abs(min_reward)*np.exp(3*(2*(env._shape[0]+env._shape[1]) - np.count_nonzero(obs_only_state))/len(obs_only_state))
                        goal_reached = 1
                # =========================================================================== #
                # if goal reached, reward is only given by env.step function

                value_t = self.critic(obs)
                logprobability_t = self.logprobabilities(logits, action)

                observation = np.where(observation_new == 1)[0]
                # observation : integer
                # observation_new : np.array

                self.buffer.store(obs, action, reward, value_t, logprobability_t)

                terminal = done
                if terminal or (t == self.buffer_size - 1):

                    obs_only_state = np.zeros((env._shape[0]*env._shape[1],))

                    last_value = 0 if done else self.critic(obs)
                    self.buffer.finish_trajectory(last_value)
                    num_episodes += 1

                    # goal reached at past
                    # and goal reached at current
                    if goal_reached == 1 and epoch_return > 0:
                        epoch += 1
                        _return = round(epoch_return / num_episodes, 2)
                        result.append(_return)
                        print(
                            "- Epoch "+str(epoch)+". Avg Return: "+str(_return)
                        )
                        epoch_return = 0
                        num_episodes = 0
                        prev_t = t
                    
                    if epoch >= self.epochs : break
                    observation = env.reset()
                
                # goal reached at past
                # but goal not reached at current
                if goal_reached == 1 and epoch_return < 0 and (t-prev_t) % env.max_steps == env.max_steps - 1:
                    num_episodes += 1
                    epoch += 1
                    _return = round(epoch_return / num_episodes, 2)
                    result.append(_return)
                    print(
                        "- Epoch "+str(epoch)+". Avg Return: "+str(_return)
                    )
                    epoch_return = 0
                    num_episodes = 0
                    if epoch >= self.epochs : break
                    observation = env.reset()

                if goal_reached == 0 and t % env.max_steps == env.max_steps - 1:
                    num_episodes += 1
                    epoch += 1
                    _return = round(epoch_return / num_episodes, 2)
                    result.append(_return)
                    print(
                        "- Epoch "+str(epoch)+". Avg Return: "+str(_return)
                    )
                    epoch_return = 0
                    num_episodes = 0
                    if epoch >= self.epochs : break
                    observation = env.reset()

            # information about epoch : (one epoch can have several episodes)
            # - if goal not reached at current episode : epoch += 1 when agent steps 100 steps
            # - if goal reached at current episode : epoch += 1 when episode ends
            if epoch >= self.epochs : break

            (
                observation_buffer,
                action_buffer,
                advantage_buffer,
                return_buffer,
                logprobability_buffer,
            ) = self.buffer.get()


            for _ in range(self.train_policy_iterations):
                kl = self.train_policy(
                    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
                )
                if kl > 1.5 * self.target_kl:
                    break

            for _ in range(self.train_value_iterations):
                self.train_value_function(observation_buffer, return_buffer)

        return result