# 경사하강법(Gradient Descent)

import numpy as np
import matplotlib.pyplot as plt

# 100개의 행과 2개의 열을 가지는 가상의 데이터 생성
X = np.random.rand(100, 2)

# y는 X와 선형 관계인 종속 변수 y = ax + b
y = 2 * X + 1 + 0.1 * np.random.randn(100, 2)

# 0.1 * np.random.randn(100, 2) ---> 실제 데이터와 유사하도록 노이즈 추가

# 모델 파라미터 초기화
# 가중치(weight) 초기화
weight = np.random.rand(2, 1)

# 편향(bias) 초기화
bias = np.random.rand(100, 1)
 
print("가중치:", weight)
print("편향:", bias)
"""
가중치: [[0.72955936]
 [0.04488185]]
편향: [[0.51215691]
 [0.19448464]
 [0.02407914]
 ...
 [0.73387266]
 [0.38978429]
 [0.51741   ]]
(총 100개)
"""

# 학습률 설정
learning_rate = 0.01

# 반복 횟수 설정
num_iterations = 1000

# 손실 및 매개변수 변화 기록을 위한 리스트 생성(for 시각화)
loss_history = []
weight_history = []
bias_history = []

for iteration in range(num_iterations):
    # 예측 계산
    predictions = X.dot(weight) + bias
    
    # 손실 함수 계산
    loss = np.mean((predictions - y) ** 2)          # 예측값과 실제값 간의 오차를 계산한다. 각 오차를 제곱하여 양수로 변환한다. * 음수 오차와 양수 오차를 동일하게 다루기 위함
    loss_history.append(loss)

    # 기울기 계산
    gradient_weight = 2/len(X) * X.T.dot(predictions - y)   # 전치 행렬로 만들어 행렬 곱에 필요한 형태로 만든다.
    gradient_bias = 2/len(X) * np.sum(predictions - y, axis=0)
    
    # 매개변수 업데이트
    weight = weight - learning_rate * gradient_weight
    bias = bias - learning_rate * gradient_bias

    # 가중치 기록에 추가
    weight_history.append(weight.copy())
    bias_history.append(bias.copy())

print("최종 가중치:", weight)
print("최종 편향:", bias)
"""
최종 가중치: [[1.8691233  0.31093415]
 [0.13602856 1.68588525]]

최종 편향: [[1.27233648 1.28165777]
 [0.557106   0.56642729]
 [1.20110651 1.2104278 ]
...
 [1.30671851 1.3160398 ]
 [0.82747664 0.83679793]
 [1.14273828 1.15205957]]
(총 100개)
"""

# 시각화하여 손실 함수와 모델 매개변수가 어떻게 변화하는지 이해해보자.

# 손실 값의 변화를 시각화
plt.figure(figsize=(12, 4))
plt.plot(range(num_iterations), loss_history)
plt.xlabel('Iterations')
plt.ylabel('loss')
plt.title('Change in Gradient Descent Loss')
plt.show()

# 가중치의 변화를 시각화
plt.figure(figsize=(12, 4))
weight_history = np.array(weight_history)
plt.plot(range(len(weight_history)), weight_history[:, 0, 0], label='Weight 1') # 첫 번째 특성에 대한 가중치
plt.plot(range(len(weight_history)), weight_history[:, 0, 1], label='Weight 2') # 두 번째 특성에 대한 가중치
plt.xlabel('Iterations')
plt.ylabel('Weight')
plt.title('Change in Weights')
plt.legend()

# 편향의 변화를 시각화
plt.figure(figsize=(12, 4))
bias_history = np.array(bias_history)
plt.plot(range(len(bias_history)), bias_history[:, 0, 0], label='Bias 1') # 첫 번째 특성에 대한 편향
plt.plot(range(len(bias_history)), bias_history[:, 0, 1], label='Bias 2') # 두 번째 특성에 대한 편향
plt.xlabel('Iterations')
plt.ylabel('Bias')
plt.title('Change in Biases')
plt.legend()

plt.show()
