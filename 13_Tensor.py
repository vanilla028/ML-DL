import math
import numpy as np

# 0차원 텐서
x = np.array(3) # 1개짜리 스칼라 값, 방향성이 없다.
print(x) # 3
print(x.shape) # ()
print(np.ndim(x)) # 0 ==> 차원이 없다.

# 벡터(1차원 텐서)
a = np.array([1,2,3,4])
b = np.array([5,6,7,8])

# 벡터(1차원 텐서)의 연산: 덧셈
c = a + b
print(c) # [ 6  8 10 12]
print(c.shape) # (4,)
print(np.ndim(c)) # 1 ==> 1차원이다.

# 벡터(1차원 텐서)의 연산: 곱셈
c = a * b
print(c) # [ 5 12 21 32]
print(c.shape) # (4,)
print(np.ndim(c)) # 1 ==> 1차원이다.

#스칼라와 벡터의 곱. 일괄적용에 쓰인다. 예) 물품 항목 * 1.04(물가 0.4% 상승 반영)
a = np.array(10) # 스칼라(0차원 텐서)
b = np.array([1,2,3]) # 벡터(1차원 텐서)
c = a * b

print(c) # [10 20 30]

# 전치행렬(행과 열을 바꾼 배열의 형태)
# 2차원 텐서

A = np.array([[1,3,4], [4,5,6]])
print('A\n', A)
print('A.shape\n', A.shape)
print('--------------------')

A_ = A.T
print('A_\n', A_)
print('A_.shape\n', A_.shape)
print('--------------------')

"""
A
 [[1 3 4]
 [4 5 6]]
A.shape
 (2, 3)
--------------------
A_
 [[1 4]
 [3 5]
 [4 6]]
A_.shape
 (3, 2)
--------------------
"""