import gym
from lava_grid import ZigZag6x10
from agent_lava import agent
import random
import numpy as np

import os

# default setting
max_steps = 100
stochasticity = 0.2 # probability of the selected action failing and instead executing any of the remaining 3
no_render = True

env = ZigZag6x10(no_goal_rew=False, percentage_reward=False, max_steps=max_steps, 
                act_fail_prob=stochasticity, goal=(5,11), numpy_state=False)
s = env.reset()
done = False
cum_reward = 0.0

""" Your agent"""
agent = agent(env)
print("Training start.")
agent.train(no_render)
print("")