# TEAM : 5
# - HyeokJin Kwon
# - Seongmin Lee
# - Sungyoung Lee
# Code by Sungyoung Lee

from chain_mdp import ChainMDP
from agent_chainMDP import agent


# recieve 1 at rightmost state and recieve small reward at leftmost state
env = ChainMDP(10)
s = env.reset()


""" Your agent"""
agent = agent(env, 5)     # agent call (env, head_num for ensemble DQN)

# ==== Agent Training Part ==== #
agent.train()             # train policy of the agent
# ============================= #


##### eval code #####
done = False
cum_reward = 0.0

s = env.reset()
while not done:
    action = agent.action(s)
    ns, reward, done, _ = env.step(action)
    print(s)
    cum_reward += reward
    s = ns

print(f"total reward : {cum_reward}")
#####################