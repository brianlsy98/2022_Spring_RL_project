# TEAM : 5
# - HyeokJin Kwon
# - Seongmin Lee
# - Sungyoung Lee
# Code by Sungyoung Lee

# Referred to the example code from keras.io (cartpole ppo example)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers

import gym
import scipy.signal
import time
import os

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

        # ==== Train or Eval mode select ==== #
        print("")
        self.mode = int(input("[train from the beginning](1) or [load weights & eval](2) : "))
        if self.mode == 1:
            print("train from the beginning..")
            self.initialize_mode = int(input("initialization for layers. Select int from 1 to 10 : "))
        elif self.mode == 2:
            print("loading existing model..")
            self.actor = keras.models.load_model(os.getcwd()+"/model/lava"+'/lava_actor')    # load policy
            self.critic = keras.models.load_model(os.getcwd()+"/model/lava"+'/lava_critic')  # load value-func
        # =================================== #


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

    
    def train(self, env):
        
        # ==== Training length ==== #
        self.steps_per_epoch = 4000
        self.epochs = 50
        # ========================= #


        # === Hyperparameters of the PPO algorithm === #
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.policy_learning_rate = 3e-4
        self.value_function_learning_rate = 1e-3
        self.hidden_sizes = (128, 128)

        self.train_policy_iterations = 80
        self.train_value_iterations = 80
        self.lam = 0.97
        self.target_kl = 0.01
        # ============================================ #


        # ==== Initialize ==== #
        self.observation_dimensions = env.observation_space.n
        self.num_actions = env.action_space.n

        self.buffer = Buffer(self.observation_dimensions, self.steps_per_epoch)

        self.observation_input = keras.Input(shape=(self.observation_dimensions,), dtype=tf.float32)
        self.logits = self.mlp(self.observation_input, list(self.hidden_sizes) + [self.num_actions], tf.tanh, None)
        self.actor = keras.Model(inputs=self.observation_input, outputs=self.logits)
        self.value = tf.squeeze(
            self.mlp(self.observation_input, list(self.hidden_sizes) + [1], tf.tanh, None), axis=1
        )
        self.critic = keras.Model(inputs=self.observation_input, outputs=self.value)

        self.policy_optimizer = keras.optimizers.Adam(learning_rate=self.policy_learning_rate)
        self.value_optimizer = keras.optimizers.Adam(learning_rate=self.value_function_learning_rate)

        # initializing weights with seed range 1~10 (int)
        np.random.seed(self.initialize_mode)
        a_w = self.actor.get_weights()
        c_w = self.critic.get_weights()
        a_w = [np.random.permutation(w.flat).reshape(w.shape) for w in a_w]
        c_w = [np.random.permutation(w.flat).reshape(w.shape) for w in c_w]
        self.actor.set_weights(a_w); self.critic.set_weights(c_w)
        # ===================== #


        observation, episode_return, episode_length = env.reset(), 0, 0
        reward_for_print = 0

        for epoch in range(self.epochs):
            
            sum_return = 0
            sum_length = 0
            num_episodes = 0

            print_once = 0
            goal_reached = 0

            # == only for visualizing = #
            obs_with_pit_goal = np.zeros((env._shape[0]*env._shape[1],))
            obs_only_state = np.zeros((env._shape[0]*env._shape[1],))
            for pit in env._pits:
                obs_with_pit_goal[pit] -= 100                                   # visualize pit as -100
            obs_with_pit_goal[env.goal[0]*env._shape[1]+env.goal[1]] = 1000     # visuailze goal as 1000
            # ========================= #


            for t in range(self.steps_per_epoch):

                # == trajectory visualizing == #
                obs_with_pit_goal[observation] += 1
                # ============================ #
                # ====== observation cum ===== #
                obs_only_state[observation] += 1
                # ============================ #  

                # obs : one-hot array
                # observation : integer
                obs = np.zeros((env._shape[0]*env._shape[1],))
                obs[observation] += 1
                obs = obs.reshape(1,-1)
                logits, action = self.sample_action(obs)
                observation_new, reward, done, _ = env.step(action[0].numpy())

                reward_for_print += reward

                # ======= reward -100 if reached to same place 3 times after action ========= #
                # ======= *********** key idea for training time decrease *********** ======= #
                if goal_reached == 0:   # reward implementation until first goal reach
                    for state in obs_only_state :
                        if state > 2 : reward -= 1; done = True; reward_for_print -= 1
                        elif state > 1 : reward -= state/len(obs_only_state)
                        elif state == 1 : reward += max(1/len(obs_only_state), 0.01)   # 1/60 : this should be higher than 0.01
                
                    if np.where(observation_new == 1)[0] == env.goal[0]*env._shape[1]+env.goal[1]:
                        # 2*(env._shape[0]+env._shape[1]) : longest dist to goal
                        reward += np.exp(3*(2*(env._shape[0]+env._shape[1]) - np.count_nonzero(obs_only_state))/len(obs_only_state))
                        goal_reached = 1
                # =========================================================================== #
                # if goal reached, reward is only given by env.step function


                episode_return += reward
                episode_length += 1

                value_t = self.critic(obs)
                logprobability_t = self.logprobabilities(logits, action)

                observation = np.where(observation_new == 1)[0]
                # observation : integer
                # observation_new : np.array

                self.buffer.store(obs, action, reward, value_t, logprobability_t)


                terminal = done
                if terminal or (t == self.steps_per_epoch - 1):
                    # == only for visualizing == #
                    if print_once == 0 and t > self.steps_per_epoch - 0.5*env.max_steps:
                        print_once = 1
                        print("")
                        print(f"trajectory : at epoch {epoch+1}")
                        print(obs_with_pit_goal.reshape(env._shape))
                        if goal_reached == 1: print("goal_reached")
                    # ========================== #
                    # == only for visualizing = #
                    obs_with_pit_goal = np.zeros((env._shape[0]*env._shape[1],))
                    obs_only_state = np.zeros((env._shape[0]*env._shape[1],))
                    for pit in env._pits:
                        obs_with_pit_goal[pit] -= 100                                   # visualize pit as -100
                    obs_with_pit_goal[env.goal[0]*env._shape[1]+env.goal[1]] = 1000     # visuailze goal as 1000
                    # ========================= #

                    last_value = 0 if done else self.critic(obs)
                    self.buffer.finish_trajectory(last_value)
                    sum_return += reward_for_print
                    sum_length += episode_length
                    num_episodes += 1
                    observation, episode_return, episode_length = env.reset(), 0, 0
                    reward_for_print = 0
                    

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


            print(
                " Epoch: "+str(epoch + 1)+". Mean Return: "+str(sum_return / num_episodes)+". Mean Length: "+str(sum_length / num_episodes)
            )

        self.actor.save(os.getcwd()+"/model/lava"+'/lava_actor')
        self.critic.save(os.getcwd()+"/model/lava"+'/lava_critic')