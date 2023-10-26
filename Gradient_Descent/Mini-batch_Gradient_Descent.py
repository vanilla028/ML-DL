# 미니배치 경사 하강법(Mini-batch Gradient Descent)
# 데이터를 미니배치로 나누어 학습하는 방법. GD와 SGD의 사이의 절충안.

import numpy as np

# 100개의 행과 1개의 열을 가지는 가상의 데이터 생성
X = np.random.rand(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# 모델 파라미터 초기화
theta = np.random.rand(2, 1)

# 학습률 설정
learning_rate = 0.01

# 미니배치 크기 설정
batch_size = 10

# 반복 횟수 설정
num_iterations = 1000

for iteration in range(num_iterations):
    # 임의의 미니배치 선택
    random_indices = np.random.choice(len(X), batch_size)
    xi = X[random_indices]
    yi = y[random_indices]
    
    # 예측 계산
    predictions = xi.dot(theta)
    
    # 손실 함수 계산
    loss = np.mean((predictions - yi) ** 2)
    
    # 기울기 계산
    gradient = 2/batch_size * xi.T.dot(predictions - yi)
    
    # 매개변수 업데이트
    theta = theta - learning_rate * gradient

print("최종 매개변수:", theta)
