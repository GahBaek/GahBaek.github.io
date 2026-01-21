---
layout: posts
title: "title"
categories: ["Andrew.Ng"]
---

**Regularization**

machin learning 에서 모델이 훈련 데이터에 과도하게 맞춰져 새로운 데이터에 대한 성능이 떨어지는 것을 막기 위해 사용하는 기술.

모델의 가중치 (Weight) 에 제약 조건을 추가하여 학습된 모델이 새로운 데이터에 대해서도 잘 일반화할 수 있게 한다.

## Regularization

**L2 Regularization**

: 모델이 훈련 데이터에 과하게 맞춰 가중치가 너무 커지는 것을 막기 위해 큰 가중치에 벌점을 주는 정규화 방법이다.

<img width="1143" height="437" alt="image" src="https://github.com/user-attachments/assets/00a9a59f-4a27-427b-8142-319374cd5d01" />

**정확히 맞추는 것(J)**과 **가중치를 작게 유지하는 것($\sum w^2$)** 을 같이 최적화

→ 가중치가 커지면 모델 출력이 입력 변화에 아주 민감해져서 훈련 데이터의 잡음까지 따라가며 암기하기 쉽다.

$w←(1−αλ)w−α(∂w/∂J)$여기서 $(1-\alpha\lambda)$ 때문에 매 스텝마다 가중치가 **조금씩 “감쇠(decay)”** 돼서 weight decay라고 부른다.

<img width="1365" height="859" alt="image" src="https://github.com/user-attachments/assets/48273ede-0265-41d9-81eb-de19c4700c68" />

- L1 Regularization 의 경우에는 일부 가중치를 정확히 0 으로 만들기 쉬워서 희소해진다.
- L2 Regularization 의 경우에는 가중치를 줄이지만 완전히 0으로 만들기보단 작은 값으로 남긴다.

<img width="961" height="793" alt="image" src="https://github.com/user-attachments/assets/a9a777a3-a0ce-4328-8696-83f3da00eade" />

Bias: 평균적으로 얼마나 빗나가 있나

→ the model doesn’t fit the training data effectively

solution: use a more complicated model

Variance: 데이터가 바뀔 때 예측이 얼마나 흔들리나

→ the models can fit the training data but doesn’t fit the test data

solution: use a less complicated model

## Understanding Dropout

**Intuition**: Can’t rely on any one feature, so have to spread out weights.

Dropout

: 학습 중에 뉴런을 랜덤으로 꺼서( = 출력을 0으로) 학습시키는 정규화 기법

Making prediction at test time

- test 시에는 dropout 을 사용하지 않고 모든 unit 을 사용해야한다.
- dropout 을 진행 시 결과가 계속 바뀐다.
- 이는 예측 성능에 noise 를 더할 뿐이다.

Dropout 은 L2 정규화 같이 과적합을 막을 수 있다.

**`L2 정규화`**에서는 다른 가중치는 다르게 취급한다. 그 가중치에 곱해지는 활성화의 크기에 따라 다르다.

**`keep_prob`**

: 각 층에 해당 유닛을 유지할 확률

→ layer 마다 keep_prob 을 달리 할 수 있다.

overfitting 의 확률이 높으면 keep_prob 을 낮게 설정해야한다.

<img width="958" height="695" alt="image" src="https://github.com/user-attachments/assets/2779568c-f34a-4cdc-a1c8-8eb278db3d2f" />

⇒ 7개 뉴런 → 7개 뉴런 완전 연결: 가중치 행렬 W 가 7 x 7 이 되어 파라미터(자유도)가 확 늘어난다.

가중치: 7 x 7

bias: 다음 층 뉴런의 수  + 7

⇒ 56개의 파라미터가 이 구간 하나에만 생긴다.

Dropout의 단점

: cost function J 를 더 이상 사용할 수 없다.

## Other Regularization Methods

Data Augmentation

Early Stopping

<img width="1625" height="702" alt="image" src="https://github.com/user-attachments/assets/50e7bcbd-7e2c-4360-8e8f-46ff5628ff6c" />

valid set error 가 증가하는 시점에서 epoch 을 중지하는 것이 좋다. 이후에는 과적합이 진행되기 때문

early stopping의 단점

- optimize cost function, not overfitting 문제를 단독으로 풀 수 없다.
- 두 가지의 문제가 존재할 대 각각 해결하여 성능을 더 높일 수 있는데 학습을 이른 시점에 종료한다면 두 가지를 단독으로 해결할 수는 없다.

⇒ L2 regularization을 사용하는 게 낫다.

---

**Normalization**

: 입력 데이터의 크기나 분포를 일정한 범위로 조정하여 학습을 안정화시키기 위해 사용된다. 주로 데이터 전처리나 모델 내부에서 수행된다.

학습 안정화는 결국 학습 속도 개선에 영향을 미치게되며 과적합도 간접적으로 방지할 수 있게 된다. 또한, 학습 시 출력값을 정규화하기 때문에 가중치 조기값에 대한 민감도를 줄이며 기울기 소실 문제도 일부 완화하 수 있다.

## Normalizing Inputs

훈련을 빠르게 하는 방법 중 input 을 정규화하는 것도 있다.

1. 평균을 빼는 것. 즉 0으로 만드는 것.
2. 분산을 정규화하기 → test set 를 정규화할 때도 같은 u, 감마를 사용해야한다.

**why normalize inputs ?**

<img width="1683" height="835" alt="image" src="https://github.com/user-attachments/assets/ed2daa32-c536-4522-a723-0ad573dc3b0a" />

왼쪽 사진과 같은 cost function 을 사용한다면 매우 작은 학습률을 사용해야할 것이다.

오른쪽과 같은 cost function 을 사용한다면 왔다갔다 하지 않아도 큰 step 으로 전진할 수 있다.

## Vanishing/Exploding Gradients

<img width="1615" height="613" alt="image" src="https://github.com/user-attachments/assets/3eff6f95-6ed2-426f-b74b-c7b013663b5f" />

→ 층 수가 크면 forward 에서도 곱이 많이 생기고, backprop 에서도 곱이 더 많이 생긴다.

## Weight Initilization in a Deep Network

: Vanishing / Exploding 문제를 해결할 수 있는 방법 (완벽히 해결하는 것은 아니지만, 굉장한 효과가 있다.)

- Weight Initialization randomly * np.sqrt(1/n[l-1])
- Activation Function 이 relu 인 경우, np.sqrt(2/n[l-1]) 가 더 잘 작동한다.

⇒ 각각의 가중치 행렬 w 를 1보다 너무 커지거나 너무 작아지지 않게 설정해서 너무 빨리 폭발하거나 소실되지 않게한다.

tanh activation function → **`Xavier initialization`**

## Numertical Approximation of Gradients

경사 검사를 구현해 역전파의 구현이 맞는지 확인하는 방법

기울기 계산에 있어서 two-sided difference 또한 세타에서 미분값과 근사함을 확인할 수 있다. → 미분학적 기울기의 정의로 보았을 때 대략적인 오차를 구할 수 있다.

## Gradient Checking

: 현재 gradient 가 정확하다면 엡실론 크기만큼 변화를 주었을 때 생성되는 approimate gradient 와 기존의 gradient 의 차이 또한 엡실론과 비슷해야한다.

## Mini Batch Gradient Descent

: 신경망을 더 빠르게 학습시키는 최적화 방법

- 참고
    - Vectorization 이 m 개의 샘플에 대한 계산을 효율적으로 만들어준다.

⇒ 그러나, 모든 훈련 예제에 대해 기울기를 계산하기 전에 일부 데이터에 대해서 gradient descent 를 수행하는 것이 더 빠른 학습이 가능하다는 것이 밝혀졌다.

## Understanding Mini-Batch Gradient Descent

<img width="755" height="645" alt="image" src="https://github.com/user-attachments/assets/aa49e4c1-add3-4be2-9d78-75e4eb73ea24" />

→ Batch gradient descent 에서는 이런식으로 하강만 해야한다. 만일 한번이라고 상승세가 나타난다면 이는 learning rate 가 너무 크다던가 하는 문제가 있는 것이다.

<img width="858" height="628" alt="image" src="https://github.com/user-attachments/assets/d5504132-a724-44e5-8c75-f46227cb2714" />

→ 그러나, mini-batch gradient descent 를 사용한다면 전체적인 학습은 하강하지만 이런식으로 noise 가 있을 수 있다.

노이즈가 발생하는 이유: $x^{t}$, $y^{t}$ 의 비용보다 $x^{t+1}$, $y^{t+1}$ 의 비용이 더 클 수도 있기 때문 (잘못 표기했다던지 등)

전체 dataset 이 m 일 때,

mini batch size 가 

- m 일때: 그냥 Gradient Descent 와 다를 바 없다.
- 1 일때: 확률적 경사 하강법 (pick 1 examples randomly) → 벡터화에서 얻을 수 있는 속도 향상을 잃을 수 있다.

<img width="1839" height="1028" alt="image" src="https://github.com/user-attachments/assets/f4153d1d-196a-41f2-b820-c052e299a6ef" />

1 ~ m 사이의 mini batch size

- 많은 vectorization 을 얻을 수 있다.
- make progress without waiting process

작은 훈련 세트: 배치 경사 하강법

큰 훈련 세트: mini-batch size (64, 128, 256, 512)

## Exponentially Weighted Averages

지수 가중 평균

: 최근 값에 더 큰 가중치를 주는 이동평균

V(t) = $a * V_{t-1} + (1-a) * v_{t}$

- V(t) : current output
- V(t-1) : prev output
- v(t) : current input

: a 가 커질수록 선이 더 부드러워진다. → 더 많은 x 값들의 평균을 사용하기 때문에 곡선이 더 부드러워진다.

⇒ 지수가중평균

<img width="1011" height="504" alt="image" src="https://github.com/user-attachments/assets/c52abf6b-6f47-4938-a048-fd18de03202c" />

<img width="1201" height="856" alt="image" src="https://github.com/user-attachments/assets/870767eb-c1cc-4520-bb5e-ae5e1d709c08" />

## Understanding Exponentially Weighted Averages

(1) 학습 곡선 smoothing

→ 미니 배치 때문에 loss 가 들쑥날쑥 하니, EWA 로 매끈하게 만들어 추세를 보기 쉽게 한다.

(2) momentum

→ 기울기의 EWA 를 속도처럼 누적해 업데이트한다.

(3) RMSProp / Adam

실수 하나만을 선언하고 가장 최근에 얻은 값을 이 식에 계속 update 하기만 하면 된다.

→ 메모리 사용량이 적다.

## Bias Correction of Exponentially Weighted Averages

편향 보정

<img width="1368" height="677" alt="image" src="https://github.com/user-attachments/assets/e4a3d339-2cb2-4f60-a6b5-19f2255ab5e0" />

B 가 0.98 일 때, 실제로는 초록색 곡선이 아니라, 보라색 곡선을 얻을 수 있다. 또한, 보라색 곡선은 매우 낮은 temperature 에서부터 시작한다는 점을 알 수 있다.

추정의 초기 단계에서 초기화를 잘해야한다 → 항을 추가함으로써 초반부에 대한 bias 를 수정할 수 있다.

<img width="929" height="465" alt="image" src="https://github.com/user-attachments/assets/67363098-8b67-440d-ba84-66bc8f29fd76" />

t 가 더 커질수록 B^t 는 0에 가까워진다.

## Gradient Descent with Momentum

경사에 대한 지수가중평균을 얻고 그 가중치를 업데이트하는 것.

Momentum: Gradient Descent 보다 빠른 알고리즘 → 지수 가중 평균을 산출하는 것.

<img width="1821" height="564" alt="image" src="https://github.com/user-attachments/assets/d00f86b4-69fe-4564-ada4-990ea2ec9a96" />

수평으로는 빠른 학습을 원하지만, 수직 방향으로는 느린 학습을 원한다.

<img width="1849" height="1034" alt="image" src="https://github.com/user-attachments/assets/ecc5db89-2d7f-49b4-8493-0372cf704175" />

현재 미니배치의 기울기 dW, db 를 그대로 사용하지 않고, 기울기의 지수이동평균을 속도 v 로 누적해서 업데이트를 더 안정적/빠르게 만든다.

**`SGD`** 는 미니배치 때문에 gradient 가 지그재그로 흔들린다.

**`momentum`** 은 과거 방향을 누적해서 일관된 방향으로는 더 빨리 가속, 좌우로 흔들리는 노이즈는 평균으로 상쇄 → 그래서 더 빠르게 수렴하거나 더 안정적으로 내려간다.

Momentum 은 학습시에 변화량에 있어 이전 변화량을 고려하므로 W, b 각각에 대해 변화율을 다르게 할 수 있다.

## RMSProp

: 파라미터마다 학습률을 자동으로 다르게 조절해서, 지그재그로 느리게 가는 문제를 줄이고 더 빨리 골짜기 중심으로 가게 만든다.

<img width="1776" height="482" alt="image" src="https://github.com/user-attachments/assets/dc901aee-92bf-4ffd-bb7c-825537250aaa" />

**`general SGD`** 는 동일한 learning rate 를 모든 방향에 쓰니까

- 가파른 축에서는 계속 튀며 진동
- 완만한 축으로는 천천히 이돈

⇒ 결과적으로 지그재그로 느리게 수렴한다.

RMSProp: 최근 제곱 gradient 의 평균을 나눠서 step 을 조절

$W:=W−α{\frac{dW}{\sqrt {S_{db}} + \gamma}}$

$b:= b - \alpha \frac{db}{\sqrt{S_{db}} + \gamma}$

$\gamma$  = $10^{-8}$

**Intuition**

- 어떤 파라미터는 gradient 가 자주 크면 S 가 커짐
- 어떤 파라미터는 gradient 가 작으면 S 가 커짐

## Adam Optimization Algorithm

: RMSProp + Momentum

- Momentum 의 V_{dw} 와 같은 지수 평균 이동 기법
- RMSProp 의 Gradient 에 따른 변수 별 update 변화량 할당

이 두 가지를 적용한 기법

**Hyperparameter**

1. $\alpha$ : need to be turn
2. $\beta_{1}$ (momentum) : 0.9 (dw)
3. $\beta_{2}$ (RMSprop) : 0.999(dw^2)
4. $\epsilon$ : $10^{-8}$

## Learning Rate Decay

학습 알고리즘의 속도를 높이는 한 가지 방법은 시간에 따라 학습률을 천천히 줄이는 것이다.

→ 학습 초기에는 큰 폭으로 learning rate 를 설정하고, 학습이 수렴할수록 학습률이 느려져 작은 스텝으로 학습한다.

$learning rate = ({\frac {1}{(1+decayrate * epochnum)}}) * learning rate)$

## The Problem of Local Optima

→ Weight 별 학습량을 다르게 하여 plateaus 구간을 빠르게 빠져나올 수 있도록 Momentum, RMSprop, Adam 과 같은 알고리즘이 사용된다.

## Tuning Process

Neural Network 에서는 정해야하는 하이퍼파라미터가 매우 많다

**Try Random Values**

- 일단 무작위로 조합을 설정 후 여러 개를 테스트
- 성능이 좋은 지점을 주변으로 subset 지역을 할당하여 다시 설정된 범위 내에서 여러 개를 테스트

<img width="1376" height="908" alt="image" src="https://github.com/user-attachments/assets/b3bf7ce7-c700-48ad-a745-5f14a033fbdd" />

## Using an Appropriate Scale

선형 척도 대신로그 척도를 사용하는 것이 낫다

<img width="1626" height="715" alt="image" src="https://github.com/user-attachments/assets/44389476-509e-496f-821a-8acb29400307" />

**`선형 균일 샘플링`**: 구간 길이가 큰 0.1 ~ 1 에 90% 확률로 몰림

**`로그 스케일 샘플링`**: 각 자릿수 구간을 공평하게 탐색

**Hyperparameters for exponentially weighted averages**

## Hyperparameter Tuning in Practice

하이퍼파라미터가 모델에, dataset 에 잘 적용이 되는지 몇달에 한 번씩 확인해보는 것이 좋다.

- Babysitting one model: 한 모델에서 파라미터를 변경해보면서 관찰하기
- Training many models in parallel: 동시에 여러 모델 학습시키기

## Normalizing Activations in a Network

입력 변수들을 정규화하면 학습이 빨라진다.

<img width="1874" height="1054" alt="image" src="https://github.com/user-attachments/assets/1d9a5a08-9ce4-409a-b1ca-8dda255f7566" />

logistic regression 에서는 사진 속 식으로 정규화를 했다.

그럼 아래 사진과 같은 Neural Network 에서는 어떻게 할까 ?

→ batch norm

**$Z$: 어떤 층에서 활성화 함수를 통과하기 직전의 값, 즉 pre-activation (선형 결합 결과)**

BatchNorm 은 보통 각 층의 $Z^{[l]}$을 정규화한다.

미니배치에 대해

- 평균 $μ,$ 분산 $\sigma^2$를 $Z$로 계산하고

$Z_{{norm}}^{(i)}=\frac{Z^{(i)}-\mu}{\sqrt{\sigma^2+\epsilon}}$

그 다음 학습 가능한 파라미터로 다시 스케일/시프트:

$\tilde{Z}^{(i)} = \gamma Z_{\text{norm}}^{(i)} + \beta$

그리고 activation은 보통 이렇게:

$A=g(\tilde{Z})$

즉, **정규화된 $~\tilde{Z}$** 를 activation에 넣는 구조야.

## Fitting Batch Norm Into Neural Networks

<img width="1699" height="506" alt="image" src="https://github.com/user-attachments/assets/eed4c61c-37c4-4660-acea-6901ff7f6870" />

각 층에서 Z = WA +  b 를 만든 다음, 그 Z 를 바로 activation 에 넣지 말고 BatchNorm 으로 정규화한 $\tilde Z$를 activation 에 넣는다.

```python
tf.nn.batch_normalization() # 으로 구현 가능하다.
```

BatchNorm 은 보통 mini-batch 에 적용된다. 각 mini-batch 마다 BN 이 따로 계산된다.

- **$W,b,γ,β$** 는 미니배치가 바뀌어도 같은 값(공유)인데
- BN에서 쓰는 **$μ,\sigma^2$** 는 **현재 미니배치 데이터로 계산**되기 때문에
    - 1번 미니배치 통계 $(\mu_{\{1\}}, \sigma^2_{\{1\}})$
    - 2번 미니배치 통계 $(\mu_{\{2\}}, \sigma^2_{\{2\}})$
        
        가 **서로 달라질 수 있다.**
        

같은 네트워크라도 미니배치가 달라지면 BN 정규화 결과가 조금씩 달라진다.

→ BN 이 약간의 noise 를 만들어서 정규화/일반화에 도움된다.

train 때 BN vs test 때 BN

- 훈련(train): 미니배치의 $\mu, \sigma^2$ 사용
- 테스트(test)/추론(inference): 미니배치가 없거나 1개 샘플일 수도 있으니
    
    훈련 중 쌓아둔 running average(이동평균) 의 $\mu_{\text{running}}, \sigma^2_{\text{running}}$ 사용
    

**train: batch 통계**

**test: running 통계**

## Why Does Batch Norm Work?

**Learning on shifting input distribution**

covariate shift

x 의 분포가 바뀐다면 모델을 다시 학습시켜야한다.

<img width="1849" height="686" alt="image" src="https://github.com/user-attachments/assets/f81fa2e2-605a-4fff-a5d4-638ae5abc4e0" />

<img width="1866" height="776" alt="image" src="https://github.com/user-attachments/assets/2e444f2a-0b99-4235-bde8-a104f99b2ce1" />

- 3번째 층이 받는 입력은 보통 A[2]A^{[2]}A[2] 또는 Z[3]Z^{[3]}Z[3]로 이어지는데,
- 학습 중에 앞 층 파라미터가 계속 바뀌니까 **3번째 층이 보는 입력 분포가 계속 변함**
    
    → internal covariate shift”
    

mini batch를 이용할 경우 Noramlize가 일어나는 과정에서 계산되는 평균,분산은 mini-batch 내의 평균,분산이므로 전체 데이터셋에 비해 약간의 noise가 추가되어 있는 상태이다.

규제. ⇒ 전체 데이터에 대해 계산한 것과 비교해서 mini-batch 로 계산한 평균과 분산은 다소 noise 가 있다. (사대적으로 작은 데이터에 대해 추정한 것이기 때문이다.)

BatchNorm 은 약간의 normalization 효과가 있다. → 하나의 은닉층에 너무 의존하지 않게끔 하기 때문이다. (normalization 효과가 많이 크지는 않다.)

→ dropout + batchNorm 을 함께 사용하면 더 큰 일반화 효과가 나타난다.

(dropout + )큰 미니배치를 사용하면 잡음이 줄고 일반화 효과도 줄어든다.

---

요즘 BatchNorm 을 보는 시각

- scale 을 안정화해서 최적화가 쉬워진다.
- Gradient 흐름이 좋아진다.
- 학습률을 더 크게 써도 안정적이다.

## BatchNorm at Test Time

BatchNorm 은 테스트할 때는 미니배치의 평균과 분산을 사용하면 안된다.

train 시에는 각 미니배치로 평균/분산을 계산해서 정규화

test 시에는 $\alpha ^ 2$ 와 $\mu$ 의 추정치를 사용하면 된다. (훈련 동안 모아둔 평균/분산을 사용해서 정규화)

<img width="1859" height="942" alt="image" src="https://github.com/user-attachments/assets/52a59858-da35-40d2-88c7-5e9d8140ba11" />

→ 지수가중평균이 그 은닉층의 z  값 평균의 추정치가 되는 것이다.

test 시에는 한장씩 넣기도 하고 배치 구성이 계속 바뀌기도 한다.

→ 추론 결과가 불안정 / 비결정적이 된다.

→ **test 때**는 현재 배치 통계를 사용하지 않고 **고정된 통계를 사용**해야한다.

## Softmax Regression

logistic regression 을 일반화한 softmax regression

<img width="1859" height="986" alt="image" src="https://github.com/user-attachments/assets/cef6b72d-a7cc-42ad-a375-53152dbcd645" />

softmax activation function’s input and outputs are vector value. to normalize

<img width="1879" height="675" alt="image" src="https://github.com/user-attachments/assets/d04b90bf-f5a0-45a9-9213-5bf93a3fdc2b" />

<img width="1859" height="450" alt="image" src="https://github.com/user-attachments/assets/5103e02e-a536-43e3-ae8f-b1e6c68ec3e2" />

## Training Softmax Classifier

hard max:

Z 의 값을 보고 가장 가까운 원소를 1 로 하고 다른 원소들은 0

가장 큰 값을 가지는 것을 1, 나머지는 0으로 치환

softmax:

확률을 모두 나타내므로 soft 방식

Softmax regression generalized logistic regression to C classes

**Loss Function**

loss 를 줄이기 위해서는 $\hat y$ 을 증가시켜야 한다.

<img width="1877" height="1007" alt="image" src="https://github.com/user-attachments/assets/8da86dce-abba-4fc0-8e17-fe42df7b331f" />

Backpropagation

**$dz^{L}$ $= \hat y - y$**

---

## Exercise 1

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/content/sample_data/ex1data1.txt", delimiter=",")
x, y = data[:, 0], data[:, 1]

m = y.size
X = np.stack([np.ones(m), x], axis = 1)

# theta0 = b
# theta1 = W

def computeCost(X, y, theta):
    m = y.size
    # 예측값
    pred = theta[0] * X[:, 0] + theta[1] * X[:, 1]
    # cost
    err = pred - y
    return (err @ err) / (2 * m) 

def gradientDescent(X, y, theta, learningrate, num_iters):
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        pred = theta[0] * X[:, 0] + theta[1] * X[:, 1]
        err = pred - y

        theta0 = theta[0] - learningrate * (1/m) * np.sum(err * X[:, 0])
        theta1 = theta[1] - learningrate * (1/m) * np.sum(err * X[:, 1])
        theta = np.array([theta0, theta1])

        J_history.append(computeCost(X, y, theta))
    return theta, J_history

def gradientDescentMulti(X, y, theta, learningRate, num_iters):
    theta = theta.copy()
    J_history = []

    for _ in range(num_iters):
        # X.dot(theta) = pred
        err = (X.dot(theta)) - y               
        grad = np.sum(X * err[:, None], axis=0) / m  
        theta = theta - learningRate * grad

        J_history.append(np.sum(err**2) / (2*m))

    return theta, J_history

def featureNormalize(X):
    X_norm = X.copy().astype(float)
    mu = np.mean(X_norm, axis=0)
    sigma = np.std(X_norm, axis=0, ddof=0)
    X_norm = (X_norm - mu) / sigma
    return X_norm, mu, sigma

def nomalEqn(X, y):
    theta = np.zeros(size(X, 2),1)
    return theta

theta0 = np.zeros(2)
theta, J_hist = gradientDescent(X, y, theta0, learningrate=0.01, num_iters=1500)
print(theta, J_hist[-1])
```

## Exercise 2

```python
import os
import numpy as np
import matplotlib.pyplot as plt

# load data
data = np.loadtxt("/content/ex2data2.txt", delimiter=",")
X = data[:, :2]
y = data[:, 2]

# X 를 6차 다항식 형태로 변환하여 28개의 특성으로 늘린다.
def mapFeature(X1, X2, degree=6):
    if X1.ndim > 0:
        out = [np.ones(X1.shape[0])]
    else:
        out = [np.ones(1)]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))

    if X1.ndim > 0:
        return np.stack(out, axis=1)
    else:
        return np.array(out)

# plt 위에 뿌릴 X, y 데이터
def plotData(X, y):
    pos = y == 1
    neg = y == 0
    plt.plot(X[pos, 0], X[pos, 1], 'k+', lw=2, ms=10)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)

# 등고선
def plotDecisionBoundary(plotData, theta, X, y):
    plotData(X[:, 1:3], y)

    if X.shape[1] > 3:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))

        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                z[i, j] = np.dot(mapFeature(np.array([ui]), np.array([vj])), theta)

        z = z.T 
        plt.contour(u, v, z, levels=[0], linewidths=2, colors='g')
        plt.title('Decision Boundary')

def predict(theta, X):
    p_prob = sigmoid(X @ theta)
    return (p_prob >= 0.5).astype(int)

# 2 dim -> 28 dim
X_mapped = mapFeature(X[:, 0], X[:, 1]) 

# sigmoid
def sigmoid(z):
    z = np.array(z)
    g = np.zeros(z.shape)
    g = 1 / (1 + np.exp(-z))
    return g

# Cost Function and Regression
def costFunctionReg(theta, X, y, lambda_):
    m = y.size
    J = 0
    grad = np.zeros(theta.shape)
    h = sigmoid(X @ theta)
    term1 = y*np.log(h)
    term2 = (1-y)*np.log(1-h)
    reg_term = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
    error = h - y
    J = (-1 / m) * np.sum(term1 + term2) + reg_term
    grad = (1 / m) * (X.T @ error)
    grad[1:] = grad[1:] + (lambda_ / m) * theta[1:]
    return J, grad

initial_theta = np.zeros(X_mapped.shape[1])
lambda_ = 1 # 정규화 매개변수

cost, grad = costFunctionReg(initial_theta, X_mapped, y, lambda_)

initial_theta = np.zeros(X_mapped.shape[1])

lambda_ = 50
options= {'maxiter': 100}

res = optimize.minimize(costFunctionReg,
                        initial_theta,
                        (X_mapped, y, lambda_),
                        jac=True,
                        method='TNC',
                        options=options)

cost = res.fun

theta = res.x

plotDecisionBoundary(plotData, theta, X_mapped, y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0'])
plt.grid(False)
plt.title('lambda = %0.2f' % lambda_)

p = predict(theta, X_mapped)
plt.show()
```

## Exercise 3

```python
# Exercise 3 - Multi-class Classification
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.io import loadmat

input_layer_size = 400
num_labels = 10

data = loadmat("/content/ex3data1.mat")
X, y = data['X'], data['y'].ravel()

# 인덱싱 때문에 10으로 저장된 0 숫자를 다시 0으로 변경한다.
y[y==10] = 0
m = y.size

# 무작위로 100개 추출
rand_indices = np.random.choice(m, 100, replace = False)
# 추출된 인덱스의 이미지 데이터만 추출
sel = X[rand_indices, :]

# test 용 파라미터
theta_t = np.array([-2, -1, 1, 2], dtype = float)
X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order = 'F')/10.0], axis = 1)
# test 용 정답 y 와 상수 (lambda_t)
y_t = np.array([1, 0, 1, 0, 1])
lambda_t = 3

def sigmoid(z):
    z = np.array(z)
    g = np.zeros(z.shape)
    g = 1 / (1 + np.exp(-z))
    return g

def lrCostFunction(theta, X, y, lambda_):
    m = y.size
    if y.dtype == bool:
        y = y.astype(float)
    
    J = 0
    grad = np.zeros(theta.shape)
    h = sigmoid(X @ theta)
    term1 = y*np.log(h)
    term2 = (1-y)*np.log(1-h)
    reg_term = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
    error = h - y
    J = (-1 / m) * np.sum(term1 + term2) + reg_term
    grad = (1 / m) * (X.T @ error)
    grad[1:] = grad[1:] + (lambda_ / m) * theta[1:]
    return J, grad

J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

print('Cost         : {:.6f}'.format(J))
print('Expected cost: 2.534819')
print('-----------------------')
print('Gradients:')
print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
print('Expected gradients:')
print(' [0.146561, -0.548558, 0.724722, 1.398003]')

def oneVsAll(X, y, num_labels, lambda_):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1)) 

    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    for c in range(num_labels):
        initial_theta = np.zeros(n + 1)
        options = {'maxiter': 50}
        
        res = optimize.minimize(lrCostFunction, 
                        initial_theta, 
                        (X, (y == c), lambda_), 
                        jac=True, 
                        method='TNC', 
                        options={'maxfun': 50})
        all_theta[c, :] = res.x

    return all_theta

lambda_ = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_)

```

```python
# Exercise 3 - Neural Network

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# display
def displayData(X, example_width=None, figsize=(10, 10)):
    """
    Displays 2D data stored in X in a nice grid.
    """
    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')

def sigmoid(z):
    z = np.array(z)
    g = np.zeros(z.shape)
    g = 1 / (1 + np.exp(-z))
    return g

data = loadmat("/content/ex3data1.mat")
X, y = data['X'], data['y'].ravel()

y[y == 10] = 0
m = y.size
indices = np.random.permutation(m)
rand_indices = np.random.choice(m, 100, replace = False)
sel = X[rand_indices, :]

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

weights = loadmat("/content/ex3weights.mat")
Theta1, Theta2 = weights['Theta1'], weights['Theta2']
Theta2 = np.roll(Theta2, 1, axis = 0)

# Feedforward Propagation and Prediction
def predict(Theta1, Theta2, X):
    if X.ndim == 1:
        X = X[None]
    
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    p = np.zeros(m)
    
    # layer 1
    # bias 추가 X.shape = (m, 401)
    a1 = np.concatenate([np.ones((m, 1)), X], axis=1)
    
    # layer 2
    z2 = a1 @ Theta1.T
    # activation 함수 a2.shape = (m, 26)
    a2 = sigmoid(z2)
    a2 = np.concatenate([np.ones((m, 1)), a2], axis=1)

    # layer 3
    z3 = a2 @ Theta2.T
    # 최종 activation func, a3.shape = (m,10) 
    a3 = sigmoid(z3)

    # 예측값 결정
    p = np.argmax(a3, axis=1)
    return p

pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: {:.1f}%'.format(np.mean(pred == y) * 100))

if indices.size > 0:
    i, indices = indices[0], indices[1:]
    displayData(X[i, :], figsize=(4, 4))
    pred = predict(Theta1, Theta2, X[i, :])
    print('Neural Network Prediction: {}'.format(*pred))
else:
    print('No more images to display!')
```
