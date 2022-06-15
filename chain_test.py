# TEAM : 5
# - HyeokJin Kwon
# - Seongmin Lee
# - Sungyoung Lee
# Code by Sungyoung Lee

# from chain_mdp import ChainMDP
# from agent_chainMDP import agent


# # recieve 1 at rightmost state and recieve small reward at leftmost state
# env = ChainMDP(10)
# s = env.reset()


# """ Your agent"""
# agent = agent(env, 5)     # agent call (env, head_num for ensemble DQN)


# ##### eval code #####
# done = False
# cum_reward = 0.0

# s = env.reset()
# while not done:
#     action = agent.action(s)
#     ns, reward, done, _ = env.step(action)
#     print(s)
#     cum_reward += reward
#     s = ns

# print(f"total reward : {cum_reward}")
# #####################


from chain_mdp import ChainMDP
from agent_chainMDP import agent
import numpy as np
import time

start = time.time()
n = 10 # chain mdp state num

# recieve 1 at rightmost stae and recieve small reward at leftmost state
env = ChainMDP(n)
s = env.reset()

""" Your agent"""
agent = agent(env)  # <-- what's added

#######################################
# === For training from beginning === #
if agent.mode == 1 :
    AUC = agent.train() #     <-- what's added
    print(AUC)
    print("AUC: ", np.sum(AUC))
# =================================== #
#######################################

print("training time : ", time.time() - start)

done = False
cum_reward = 0.0
# always move right left: 0, right: 1
env = ChainMDP(n)
s = env.reset()

while not done:    
    action = agent.action(s)
    print("state: ", s)
    print("action: ", action)
    ns, reward, done, _ = env.step(action)
    cum_reward += reward
    s = ns
print(f"total reward: {cum_reward}")