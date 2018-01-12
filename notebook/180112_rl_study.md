## 180112


### Intro
오늘은 멀캠에서 "파이썬과 오픈소스를 이용한 딥러닝" 수업을 듣는데 수강생 30명 넘는 분들이 다들 식사 후에 앉아서 쉬고 있는데 앉아서 나몰라라하고 떠들었네요. 정말 부끄러운 날이었습니다. ㅠ.ㅠ 여튼 오늘은 점심시간이 짧아서 1시간 이내로 마쳤구요,
Q-Network를 구현한 frozen lake 소스를 실제 뉴럴넷을 이용한 Q 함수(이전엔 테이블값 참조한 리턴이었죠)와
손실함수, 그리고 최적화를 하는 걸 정리했습니다. 오늘 한 내용은요,, 참석자가 총 4분이라 다시 제대로 설명을 하겠습니다. 물론,,
스터디 환경이 너무 안좋았습니다. 너무 주변 신경이 쓰여서요... 테더링도 중간에 끊기고 ㅜ.ㅜ


### Q&A
여튼, 동석프로님 질문이 DQN뒤의 다른 강화학습 알고리즘 문의를 주셨는데요,
DQN은 Value based 계열의 알고리즘이고, 이 후에는 Policy based(state Env(observation) -> action (정책망))의
알고리즘들을 볼거에요 REINFORCE, DDPG(Derterministic Distributed Policy Gradient) 그리고 Hybrid(value + policy)인
Actor/Critic Network을 가진 A2C, A3C 알고리즘을 다룰겁니다. 물론 김성훈 교수님 강의가 아주 쉬운데, DQN까지 밖에 없어요..

### DQN 이후
그래서 향후 교육자료는 다른 자료를 가지고 하게 될건데 여러 책과 온라인 자료를 봤는데
우선 간추려서 한번 투표하겠습니다. 대략 영어 4권, 한국어 2권 중에.. 물론 영어는 온라인 동영상 컨텐츠도 같이 있는게 있답니다.

### 이벤트
각 알고리즘 끝낼때 마다,중간중간 재미를 위해서 Game competition을 할거에요, 물론 초기에는 OpenAI gym을 이용해서 우리가 계속 건드리고 있는 프로즌레이크나 브레이크아웃 제공된 환경을 쓰는게 맞겠지만요, 나중에는 실제 세상(예로 들면, 주식시장, 가상화폐)같은데  강화학습을 접목을할려면.. 우리가 우리가 직접!! 환경을 만들어야되요.  물론 이것도 의렴수렴해서 뭐를 할지 해나가보자구요,

뭐 예를 들면, 주식/가상화폐를 할려면 게임과 비교해서 이렇게 되겠죠..
게임은 score가 제공되어 이걸로 리워드를 책정하지만, 우리가 별도로 만드는 주식환경이라면
수익율을 직접 우리가 계산해주는 걸 만들어야겠죠? 물론 계산을 우리가 짜도 되지만. 아니면..백테스팅 오픈소스가 있어서 수익율 뿐만 아니라 계좌관리등 다 가능 합니다. 실제로 호가에 체결량, 매수/매도 잔량등도 다 시뮬레이션하거나 가져올 수 있으니까요

### 목표
여튼 우리가 쭈욱 잘해나간다면 이렇게 우리는 성장하리라 봅니다.
#### Step #1
OpenAI 열심히하고 강화학습 막 돌려바..
#### Step #2
환경을 내가 직접 구축하고 싶은 욕망이 생겨요..
#### Step #3
만들고 더 어려운 세계로 나가고..
#### Step #4
그다음에 단계가..우리가 배운 value, policy기반의 모델/알고리즘 변형을 해서 섞어서 막 돌려보면서 빠른 시간에 score를 높이고 최종 누적 reward를 높이고 싶은 욕망이 생겨요
#### Step #5
다음에 이거저거 막 적용해보고 싶죠..
#### Step #6
사업을 일으키거나,, 부자가 되겠죠..
#### Step #7
그 다음에 꿈 실현이에요... 그리고 퇴사... 회사 건물 소유주..

### 스터디
오늘 소스 보고 설명한겁니다. 한번 훑어보세요~
```
"""
아래 가보면 주피터 노트북으로 강화학습 알고리즘 구현해놔씀~~
https://github.com/hunkim/DeepRL-Agents
"""
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0') ## slippery = True

# 환경에서 제공하는 observation과 action의 사이즈를 가져와서 입력/출력의 사이즈 지정
input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1  ## hyper parameter로써 learning Rate는 다양하게 줘본다.


Y =  f(X)

X가 들어가서 Y가 나오잖아요
Y = Wx + b 라는 식에요... a,b를 X,Y의 pair data set 넣어서 a,b 추측을


Qpred === predict value of Y
Qpred가 엉뚱한게 Mean Square Error
즉 잘못되값이 10 잘된값이 1 10-1 squre처리하면 81 --> 잘못이 클수록 더 빨리 고쳐라
loss function : 절대값을 취할수도있고 mse
잘못된값이 절대값(10 - 1 )

텐서플로우가 알아서 손실함수를 넣어주고 learning rate정해주면 알아서 W를 최적화 해서 근사한다.


shape matrix형태가 어떻게 생겼냐, 32비트 부동소수점 실수형!!
# 입력을 주기 위해서 아래와 같이 placeholder를 준다. 그리고 one hotencoding으로 16을 준다.
X = tf.placeholder(shape=[1, input_size], dtype=tf.float32)  # state input


# 뉴럴넷의 가중치 영역이구요 웨이트고.. 뉴럴넷에서 값을 X르 가공해서 Y로 나오게 해주는 값으로써
#이걸 학습을 해야 되는거죠..
# 그리고 학습해야하는 대상이 되는것은 tf.Variable이라는걸 사용해서 넣는다.
# 0~1의 소수로 값을 랜덤으로 샘플링(뽑아서) 셋팅을 하는 부분
W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))  # weight

# matmul : matrix multifly 행렬곱
# 이건 메트릭스의 곱(즉 뉴럴넷)으로 구성되는 값의 결과(즉 뉴럴넷 결과)
Qpred = tf.matmul(X, W)  # Out Q prediction

#X랑 W(가중치)를 막 곱해서 처음에는 W가 말도 안되게 초기화가 랜덤으로 되있기때문에 엉뚱한값을 뱉습니다.
# Qpred -> Q함수의 결과값

# Q Table라는걸 가지고 Q 함수를 만들었으면
# Q Network를 가지고 Table를 대체하는거

# 이것도 마찬가지로 주어지는 즉 실제 값
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)  # Y label

# 이건 손실함수 MSE(Mean Square Error)
# Q 값이 1  100 100 - 1만큼 잘못되었어!!!!
# tf.abs(Y-Qpred) 절대값 , 차이를 얘기해야 되기때문에  제곱을 취하는 이유는 많이 틀리면 손실을 더 크게 하겠다라는 의미
loss = tf.reduce_sum(tf.square(Y - Qpred))

# 손실함수를 최소화하는 옵티마이저
# W를 잘못된 값을 나오게 했으니 예를 최적화시켜라
# 한줄로 뉴럴넷이 자동 학습
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
```



### 마치며
안녕하세요 2018년 활기차게 시작하고 계신가요??

저는 이번주부터 모두의 연구소에 Deep Learning College를 시작했고요,
아주 빡신 한주 였지만,  강화학습으로 잘나가는 이웅원/양학렬씨가 강화학습을 담당, 파이선 심화 프로그래밍은 조용래님이
수학은 이대 수학과 교수님이, 그리고 모두의 연구소의 잘나가는 분들하고 네트워킹이 쌓는 좋은시간이었습니다.
