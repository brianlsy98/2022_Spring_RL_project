import gym
from lava_grid import ZigZag6x10
from agent_lava import agent
import random
import numpy as np

# default setting
max_steps = 100
stochasticity = 0 # probability of the selected action failing and instead executing any of the remaining 3
no_render = True

env = ZigZag6x10(max_steps=max_steps, act_fail_prob=stochasticity, goal=(5, 9), numpy_state=False)
s = env.reset()
done = False
cum_reward = 0.0

""" Your agent"""
agent = agent(env, 5)

# agent.train()       # train policy 

# moving costs -0.01, falling in lava -1, reaching goal +1
# final reward is number_of_steps / max_steps

s = env.reset()
obs_ = np.zeros((env._shape[0]*env._shape[1],))
while not done:
    action = agent.action(s)
    obs_ += s
    print(obs_.reshape(env._shape))
    ns, reward, done, _ = env.step(action)          # by step : we can observe current pos of agent in env & reward

    cum_reward += reward
    s = ns
print(f"total reward: {cum_reward}")