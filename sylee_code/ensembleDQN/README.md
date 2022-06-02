# Results for executing ensembleDQN code

What's different from DQN :
- 5 pairs of Qprin, Qtarg in self.Qs
- add head infos in replay buffer
- head is made randomly in bernoulli trial (binomial result)
- update Qprin & Qtarg only when head[i] is 1
- train action selection != eval action selection
    (train : fix one head )
    (eval : for 5 pairs of Qprin/Qtarg, get argmax(Q) and vote 5 times)

* Added Algorithm by S.Y. LEE :
when voting (in eval action selection), add weights, not just 1




Finished after

- trial 1. 100 iter -> evaluation reward = 10
- trial 2. 100 iter -> evaluation reward = 10
- trial 3. 
- trial 4. 
- trial 5.

iteration
