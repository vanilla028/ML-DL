# 확률적 경사하강법(Stochastic Gradient Descent)
# 무작위로 선택한 데이터(미니 배치)를 사용하여 파라미터를 업데이트하는 방식
# 대용량 데이터에 용이. 빠른 수렴(학습)을 돕지만, 일부 노이즈가 발생할 수 있다. ---> 최적화 알고리즘

import numpy as np


# 100개의 행과 1개의 열을 가지는 가상의 데이터 생성
X = np.random.rand(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# 모델 파라미터 초기화
theta = np.random.rand(2, 1)

# 학습률 설정
learning_rate = 0.01

# 반복 횟수 설정
num_iterations = 1000

for iteration in range(num_iterations):
    # 랜덤한 샘플 선택
    random_index = np.random.randint(0, len(X))
    xi = X[random_index:random_index+1]
    yi = y[random_index:random_index+1]
    
    # 예측 계산
    prediction = xi.dot(theta)
    
    # 손실 함수 계산
    loss = (prediction - yi) ** 2
    
    # 기울기 계산
    gradient = 2 * xi.T.dot(prediction - yi)
    
    # 매개변수 업데이트
    theta = theta - learning_rate * gradient

print("최종 매개변수:", theta)
