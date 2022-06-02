# Lava 문제
## code 분석
* percentage_reward = True : 느리게 goal에 도달할수록 reward가 작아짐
* no_goal_rew = True : goal에서 reward 안 줌
* dense_reward = True : reward가 100배씩 scale 커짐
* numpy_state = True : state를 넘파이 배열로 return
* xy_state = True : state가 one-hot이 아닌 scalar 형태
* P : 전이 확률
* isd : 초기 state에 대한 분포. 현재는 무조건 (0,0)에서 시작하게 설정되어있음  
* _check_bounds : grid 안 벗어나게 제한 해줌
* act_fail_prob : 내가 취한 행동이 실패할 확률. 나머지 3개의 행동의 확률이 됨.

## DQN
* 06/01 - 02:00 am
```python
lr = 0.01           # learning rate for gradient update 
batchsize = 64      # batchsize for buffer sampling
maxlength = 2000    # max number of tuples held by buffer
tau = 100           # time steps for target update
episodes = 3000     # number of episodes to run
initialize = 1000   # initial time steps before start updating
epsilon = 1.0       # constant for exploration
decay = 0.999
e_min = 0.01
e_mid = 0.2
gamma = .95         # discount
hidden_dims = [32,32]   # hidden dimensions
```
  * 350 episode 근방에서 goal에 도달하기 시작
  * 850 episodes를 train하는 동안의 reward 결과
  * ![0601_02:00am](https://user-images.githubusercontent.com/81223817/171236269-09a4fafe-fca0-4afd-ab6f-f2847f1c640e.png)
  * ![0601_02:00am_screenshot](https://user-images.githubusercontent.com/81223817/171553883-23d800a5-4035-4e92-91f0-01268ba88bb8.png)



# Link
* [Bootstrapped DQN](https://joungheekim.github.io/2020/12/06/code-review/)
* [Rainbow DQN](https://velog.io/@isseebx/Noisy-networks-for-exploration)

