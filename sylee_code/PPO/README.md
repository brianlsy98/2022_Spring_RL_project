# Results for executing PPO code

PPO : only for LAVA
- converges very early
- problem : local optimum ---> solved!

Main Idea
- Coarse Control + Fine Control
- Coarse (goal_reached = 0) : complex rewards added (refer to agent_lava.py in PPO file)
- Fine (goal_reached = 1) : reward --> move -0.01, lava -1, goal +1
