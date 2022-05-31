# Lava 문제
## code 분석
### env
* percentage_reward = True : 느리게 goal에 도달할수록 reward가 작아짐
* no_goal_rew = True : goal에서 reward 안 줌
* dense_reward = True : reward가 100배씩 scale 커짐
* numpy_state = True : state를 넘파이 배열로 return
* xy_state = True : state가 one-hot이 아닌 scalar 형태
* P : 전이 확률 (잘 모르겠음. 아마 DP 같은 곳에서 쓰던 확률인듯)
* isd : 초기 state에 대한 분포. 현재는 무조건 (0,0)에서 시작하게 설정되어있음  
* _check_bounds : grid 안 벗어나게 제한 해줌
* act_fail_prob : 내가 취한 행동이 실패할 확률. 나머지 3개의 행동의 확률이 됨.


## Link
* [Bootstrapped DQN](https://joungheekim.github.io/2020/12/06/code-review/)
* [Rainbow DQN](https://velog.io/@isseebx/Noisy-networks-for-exploration)

