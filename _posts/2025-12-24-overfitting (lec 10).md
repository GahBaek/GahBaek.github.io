---
layout: posts
title: "overfitting (lec 10)"
categories: ["DeepLearning"]
---

### Neural Network dropout and model ensemble

**Overfitting**

- more training data
- **regularization**

**Regularization**

```python
// Regularization
l2reg = 0.001 * tf.reduce_sum(tf.square(W))
```

**Dropout**

: a simple way to prevent Neural Networks from Overfitting

<img width="639" height="348" alt="image" src="https://github.com/user-attachments/assets/f977377b-bfe9-4e69-9176-067ee59269b0" />

→ 학습할 때 random 하게 몇몇의 뉴런의 연결을 끊어내자.

```python
dropout_rate = tf.placeholder("float")
_L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
L1 = tf.nn.dropout(_L1, dropout_rate)

// 학습하는 동안에만 dropout
TRAIN:
	dropout_rate: 0.7

EVALUATIOM:
	dropout_rate: 1
```

**Ensemble**

: dataset 이 많을 때 사용할 수 있는 방법

서로 다르게 학습된 여러 모델의 예측을 결합 (Combiner) 하여 최종 예측을 만드는 방법이다.

<img width="638" height="399" alt="image" src="https://github.com/user-attachments/assets/2c24538c-174a-44e5-a6d6-3995cd93ec8d" />

**핵심 효과**

: 분산 감소 → 과적합 완화 → 일반화 성능 향상

2 ~ 4, 5% 의 성능 향상이 이루어진다.

- dropout 은 마치 여러 모델을 앙상블하는 것과 **`유사한 효과`**를 내어, 모델의 성능을 개선한다.
    - 직관적으로 “수많은 얇은 모델을 번갈아 학습시키는 것”처럼 보이고,
    - 테스트 시에는 dropout 을 끈채로 “수 많은 서브 모델의 평균 효과”를 근사한다고 해석할 수 있다.
- dropout 과 중요한 차이
    - 앙상블: 여러 모델을 실제로 따로 학습하고, 추론 때도 여러 번 계산해서 결합함 (고비용)
    - dropout: 모델 1개로 학습하면서 내부적으로 다양한 서브 네트워크를 샘플링함 (추론 비용은 거의 그대로)

딥러닝에서 모델을 학습시킬 때 모든 것을 생각하는 것보다 떄로는 몇몇은 배제한 채(Dropout), 혹은 예측 시스템을 별도로 여러 개 만들어(Ensemble) 학습하는 것이 훨씬 균형 잡히고 효율적인 모델을 만들어 낸다.

**Neural Network 의 구조**

Input

Hidden

Output

### NN Lego Play

**feedforward neural network**

- Fast Forward
    
    <img width="718" height="175" alt="image" src="https://github.com/user-attachments/assets/3d54c16a-c2c2-4acc-97b9-2c9e62434a8a" />

    바로 다음의 node 에 output 을 input 으로 주는 것이 아닌 2칸 띄어서 줄 수 있다.
    
    ⇒ He 이론 = ResNet의 아이디어
    
    앞쪽으로 이동할 때 하나의 레이어가 아닌 2개 이상의 레이어씩 이동하는 방법으로 Dropout 처럼 난수를 사용하여 다양한 결과를 통합한다. 
    
- Split & Merge
    
    <img width="750" height="149" alt="image" src="https://github.com/user-attachments/assets/23f0bace-f7d5-49de-aa4f-b4439631a3d7" />

    <img width="735" height="250" alt="image" src="https://github.com/user-attachments/assets/e5e9df48-1fa4-43ae-84c1-76b4fa02acd4" />

    <img width="791" height="331" alt="image" src="https://github.com/user-attachments/assets/7647b5ce-0dad-4292-9cd4-2581647a150a" />

    ⇒ convolutional network
    
    이미지 전체를 보는 것이 아닌 부분을 보아 합치는 구조이다. CNN 은 MNIST 모델을 구축한 Neural Network이다.
    

- Recurrent network
    
    <img width="379" height="370" alt="image" src="https://github.com/user-attachments/assets/a67a817d-588d-4491-b8c0-1fc151110922" />

    ⇒ RNN
    
    음성, 문자 등 순차적으로 등장하는 데이터 처리에 적합한 모델로 한 방향으로 진행하는 네트워크를 옆으로 진행해 순환 구조를 이루고 있다.
    
    RNN 에서도 입력을 나누고 결과를 병합하는 과정이 있다. 
    
    출처: https://childult-programmer.tistory.com/45
    

### 실습 - 시각화

```python
%matplotlib inline
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt  # 시각화 라이브러리 추가
plt.ion()

# 데이터 설정
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 가중치 설정 (Wide & Deep)
W1 = tf.Variable(tf.random.normal([2, 10]))
b1 = tf.Variable(tf.random.normal([10]))
W2 = tf.Variable(tf.random.normal([10, 1]))
b2 = tf.Variable(tf.random.normal([1]))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# 손실 값을 저장할 리스트
cost_history = []

for step in range(5001):
    with tf.GradientTape() as tape:
        l1 = tf.sigmoid(tf.matmul(x_data, W1) + b1)
        hypothesis = tf.sigmoid(tf.matmul(l1, W2) + b2)
        cost = -tf.reduce_mean(y_data * tf.math.log(hypothesis + 1e-7) + 
                              (1 - y_data) * tf.math.log(1 - hypothesis + 1e-7))

    gradients = tape.gradient(cost, [W1, b1, W2, b2])
    optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2]))
    
    # 리스트에 현재 cost 저장
    cost_history.append(cost.numpy())

    if step % 1000 == 0:
        print(f"Step: {step:5} | Cost: {cost.numpy():.4f}")

# 그래프 그리기
plt.plot(cost_history)
plt.title('Model Cost over Steps')
plt.xlabel('Step')
plt.ylabel('Cost')
plt.show()
```

<img width="626" height="590" alt="image" src="https://github.com/user-attachments/assets/90b4e7d4-9f6a-4693-b747-898551b5e0f5" />

### 실습2 - MNIST NN 학습하기

```python
import tensorflow as tf
import numpy as np
import random

# 1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(100).shuffle(10000)

# 2. 가중치 초기화 (Xavier/He 초기화 대신 간단히 정규분포 사용)
# 구조: 입력(784) -> Hidden1(256) -> Hidden2(128) -> 출력(10)
W1 = tf.Variable(tf.random.normal([784, 256], stddev=0.01))
b1 = tf.Variable(tf.zeros([256]))

W2 = tf.Variable(tf.random.normal([256, 128], stddev=0.01))
b2 = tf.Variable(tf.zeros([128]))

W3 = tf.Variable(tf.random.normal([128, 10], stddev=0.01))
b3 = tf.Variable(tf.zeros([10]))

variables = [W1, b1, W2, b2, W3, b3]

# 3. 최적화 도구 (Adam이 보통 SGD보다 수렴이 빠릅니다)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

print("--- 학습 시작 ---")

for epoch in range(15):
    total_cost = 0
    step = 0
    
    for batch_xs, batch_ys in train_dataset:
        with tf.GradientTape() as tape:
            # 레이어 1 (ReLU 활성화 함수 추가)
            L1 = tf.nn.relu(tf.matmul(batch_xs, W1) + b1)
            # 레이어 2
            L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
            # 레이어 3 (최종 출력층 - logits)
            logits = tf.matmul(L2, W3) + b3
            
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=batch_ys, logits=logits))
        
        # 모든 변수에 대해 그레디언트 계산 및 업데이트
        grads = tape.gradient(cost, variables)
        optimizer.apply_gradients(zip(grads, variables))
        
        total_cost += cost
        step += 1

    print(f"Epoch: {epoch + 1}, Avg Cost: {total_cost / step:.4f}")

print("--- 학습 완료 ---")
# 1. 테스트 데이터 중 랜덤하게 하나를 뽑습니다.
random_idx = random.randint(0, len(x_test) - 1)
selected_image = x_test[random_idx]
selected_label = y_test[random_idx]

# 2. 모델에게 물어봅니다 (3단 레이어 통과 과정 재현)
# (1, 784) 형태로 모양을 맞춰서 순전파(Forward) 진행
x_input = selected_image.reshape(1, 784)

# 학습 때와 동일한 순서로 연산합니다.
L1 = tf.nn.relu(tf.matmul(x_input, W1) + b1)
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
logits_final = tf.matmul(L2, W3) + b3

# 확률로 변환 (Softmax)
prediction_prob = tf.nn.softmax(logits_final)
predicted_label = tf.argmax(prediction_prob, 1).numpy()[0]
actual_label = tf.argmax(selected_label).numpy()

# 3. 시각적으로 나타내기 (이미지와 확률 그래프)
plt.figure(figsize=(10, 4))

# (왼쪽) 숫자 이미지 출력
plt.subplot(1, 2, 1)
plt.imshow(selected_image.reshape(28, 28), cmap='Greys')
plt.title(f"Predicted: {predicted_label} / Actual: {actual_label}")
plt.axis('off')

# (오른쪽) 0~9까지의 확률분포 막대그래프 출력
plt.subplot(1, 2, 2)
plt.bar(range(10), prediction_prob.numpy()[0], color='skyblue')
plt.xticks(range(10))
plt.ylim([0, 1])
plt.xlabel("Digit")
plt.ylabel("Probability")
plt.title("Prediction Probability")

plt.tight_layout()
plt.show()

# 4. 텍스트 결과 출력
print(f"방금 본 이미지의 예측 결과: {predicted_label}")
print(f"실제 정답: {actual_label}")
print(f"숫자별 확률: \n{np.round(prediction_prob.numpy(), 3)}")"
```

<img width="879" height="480" alt="image" src="https://github.com/user-attachments/assets/4d4e560d-9301-4ad9-9f9f-0fcf60dfb214" />

- Xavier 을 사용하여 가중치 초기화
    - random.normal 을 사용하면 층이 깊어질수록 출력값이 너무 커지거나 작아져서 학습이 잘 이루어지지 않는다.
    - 또한, cost 가 처음부터 낮게 잘 initialized 된다.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

y_train_one_hot = tf.one_hot(y_train, depth=10)
y_test_one_hot = tf.one_hot(y_test, depth=10)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_one_hot)).batch(100).shuffle(10000)

# 2. 가중치 초기화
initializer = tf.keras.initializers.GlorotNormal()

# 2. 가중치 초기화 적용
W1 = tf.Variable(initializer(shape=[784, 256]), name='weight1')
b1 = tf.Variable(tf.zeros([256]), name='bias1')

W2 = tf.Variable(initializer(shape=[256, 128]), name='weight2')
b2 = tf.Variable(tf.zeros([128]), name='bias2')

W3 = tf.Variable(initializer(shape=[128, 10]), name='weight3')
b3 = tf.Variable(tf.zeros([10]), name='bias3')

variables = [W1, b1, W2, b2, W3, b3]
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

print("--- 학습 시작 ---")

for epoch in range(15):
    total_cost = 0
    total_acc = 0
    step = 0
    
    for batch_xs, batch_ys in train_dataset:
        with tf.GradientTape() as tape:
            # 순전파 (Forward Pass)
            L1 = tf.nn.relu(tf.matmul(batch_xs, W1) + b1)
            L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
            logits = tf.matmul(L2, W3) + b3
            
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=batch_ys, logits=logits))
        
        # 역전파 (Backpropagation)
        grads = tape.gradient(cost, variables)
        optimizer.apply_gradients(zip(grads, variables))
        
        # 정확도 계산 (현재 배치)
        prediction = tf.argmax(logits, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(batch_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        total_cost += cost
        total_acc += accuracy
        step += 1

    # Epoch 결과 출력 (Cost와 Accuracy)
    avg_cost = total_cost / step
    avg_acc = total_acc / step
    print(f"Epoch: {epoch + 1:2d}, Avg Cost: {avg_cost:.4f}, Acc: {avg_acc:.4f}")

print("--- 학습 완료 ---")
# 1. 테스트 데이터 중 랜덤하게 하나를 뽑습니다.
random_idx = random.randint(0, len(x_test) - 1)
selected_image = x_test[random_idx]
selected_label = y_test[random_idx]

# 2. 모델에게 물어봅니다 (3단 레이어 통과 과정 재현)
# (1, 784) 형태로 모양을 맞춰서 순전파(Forward) 진행
x_input = selected_image.reshape(1, 784)

# 학습 때와 동일한 순서로 연산합니다.
L1 = tf.nn.relu(tf.matmul(x_input, W1) + b1)
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
logits_final = tf.matmul(L2, W3) + b3

# 확률로 변환 (Softmax)
prediction_prob = tf.nn.softmax(logits_final)
predicted_label = tf.argmax(prediction_prob, 1).numpy()[0]
actual_label = tf.argmax(selected_label).numpy()

# 3. 시각적으로 나타내기 (이미지와 확률 그래프)
plt.figure(figsize=(10, 4))

# (왼쪽) 숫자 이미지 출력
plt.subplot(1, 2, 1)
plt.imshow(selected_image.reshape(28, 28), cmap='Greys')
plt.title(f"Predicted: {predicted_label} / Actual: {actual_label}")
plt.axis('off')

# (오른쪽) 0~9까지의 확률분포 막대그래프 출력
plt.subplot(1, 2, 2)
plt.bar(range(10), prediction_prob.numpy()[0], color='skyblue')
plt.xticks(range(10))
plt.ylim([0, 1])
plt.xlabel("Digit")
plt.ylabel("Probability")
plt.title("Prediction Probability")

plt.tight_layout()
plt.show()

# 4. 텍스트 결과 출력
print(f"방금 본 이미지의 예측 결과: {predicted_label}")
print(f"실제 정답: {actual_label}")
print(f"숫자별 확률: \n{np.round(prediction_prob.numpy(), 3)}")
```

<img width="630" height="461" alt="image" src="https://github.com/user-attachments/assets/18692146-e6ec-4e61-8f45-82db1f7a7a44" />

→ 초기값만 바뀌었는데 정확도와 cost 가 크게 차이난다.

다만, Deep NN for MNIST 는 성능 차이가 크지 않다

⇒ Overfitting

### 실습 - dropout (overfitting 의 해결 방법 for NN)

```python
# Deep NN for MNIST
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

y_train_one_hot = tf.one_hot(y_train, depth=10)
y_test_one_hot = tf.one_hot(y_test, depth=10)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_one_hot)).batch(100).shuffle(10000)

# 2. 가중치 초기화
initializer = tf.keras.initializers.GlorotNormal()

# 2. 가중치 초기화 적용
W1 = tf.Variable(initializer(shape=[784, 512]), name='weight1')
b1 = tf.Variable(tf.zeros([512]), name='bias1')

W2 = tf.Variable(initializer(shape=[512, 512]), name='weight2')
b2 = tf.Variable(tf.zeros([512]), name='bias2')

W3 = tf.Variable(initializer(shape=[512, 10]), name='weight3')
b3 = tf.Variable(tf.zeros([10]), name='bias3')

variables = [W1, b1, W2, b2, W3, b3]
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

print("--- 학습 시작 ---")

for epoch in range(15):
    total_cost = 0
    total_acc = 0
    step = 0
    
    for batch_xs, batch_ys in train_dataset:
        with tf.GradientTape() as tape:
            # 순전파 (Forward Pass)
            L1 = tf.nn.relu(tf.matmul(batch_xs, W1) + b1)
            L1 = tf.nn.dropout(L1, rate = 0.3)
            L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
            L2 = tf.nn.dropout(L2, rate = 0.3)
            logits = tf.matmul(L2, W3) + b3
            
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=batch_ys, logits=logits))
        
        # 역전파 (Backpropagation)
        grads = tape.gradient(cost, variables)
        optimizer.apply_gradients(zip(grads, variables))
        
        # 정확도 계산 (현재 배치)
        prediction = tf.argmax(logits, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(batch_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        total_cost += cost
        total_acc += accuracy
        step += 1

    # Epoch 결과 출력 (Cost와 Accuracy)
    avg_cost = total_cost / step
    avg_acc = total_acc / step
    print(f"Epoch: {epoch + 1:2d}, Avg Cost: {avg_cost:.4f}, Acc: {avg_acc:.4f}")

print("--- 학습 완료 ---")
# 1. 테스트 데이터 중 랜덤하게 하나를 뽑습니다.
# 1. 테스트 데이터 중 랜덤하게 하나를 뽑습니다.
random_idx = random.randint(0, len(x_test) - 1)
selected_image = x_test[random_idx]
selected_label = y_test_one_hot[random_idx] # 또는 y_test[random_idx]

# 2. 모델 예측 (순전파)
x_input = selected_image.reshape(1, 784)
L1 = tf.nn.relu(tf.matmul(x_input, W1) + b1)
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
logits_final = tf.matmul(L2, W3) + b3

prediction_prob = tf.nn.softmax(logits_final)
predicted_label = tf.argmax(prediction_prob, 1).numpy()[0]

# --- 이 부분이 수정되었습니다 ---
actual_label = np.argmax(selected_label) 
# ------------------------------

# 3. 결과 출력
print(f"예측: {predicted_label}, 실제: {actual_label}")

plt.imshow(selected_image.reshape(28, 28), cmap='Greys')
plt.title(f"Pred: {predicted_label} / Actual: {actual_label}")
plt.show()

# 4. 텍스트 결과 출력
print(f"방금 본 이미지의 예측 결과: {predicted_label}")
print(f"실제 정답: {actual_label}")
print(f"숫자별 확률: \n{np.round(prediction_prob.numpy(), 3)}")
```

<img width="507" height="640" alt="image" src="https://github.com/user-attachments/assets/40f8956f-2294-44da-8238-a1ea416c946b" />

- Optimizer -> Adam 이 좋다.
