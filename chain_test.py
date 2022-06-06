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


# recieve 1 at rightmost stae and recieve small reward at leftmost state
env = ChainMDP(10)
s = env.reset()

""" Your agent"""
agent = agent(env.observation_space.n, env.action_space.n)  # <-- what's added

#######################################
# === For training from beginning === #
if agent.mode == 1 : agent.train(env) #     <-- what's added
# =================================== #
#######################################

done = False
cum_reward = 0.0
# always move right left: 0, right: 1
action = 1
while not done:    
    action = agent.action(s)
    ns, reward, done, _ = env.step(action)
    cum_reward += reward
    s = ns
print(f"total reward: {cum_reward}")