import numpy as np

def calculate_performance(episodes, env, agent):

    episodic_returns = []

    for epi in range(episodes):
        
        s = env.reset()

        done = False
        cum_reward = 0.0

        while not done:    
            action = agent.action(s)
            ns, reward, done, _ = env.step(action)
            cum_reward += reward
            s = ns
        
        episodic_returns.append(cum_reward)
    result = np.sum(episodic_returns)
    print((f"--Return for this seed : {round(result,3)}--"))
    return result


def calculate_sample_efficiency(episodes, env, agent):

    episodic_returns = agent.train(episodes, env, no_render=True)        
                    
    return np.sum(episodic_returns)

