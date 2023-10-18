import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
iris = load_iris()
print(iris.DESCR)
"""
**Data Set Characteristics:**

    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica

    :Summary Statistics:

    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
    ============== ==== ==== ======= ===== ====================

    ===> #종류에 따라 3개: Setosa, Iris-Versicolour, Iris-Virginica
"""
data = iris.data
label = iris.target
columns = iris.feature_names

# data frame 만들기
data = pd.DataFrame(data, columns=columns)
print(data.head())
"""
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
3                4.6               3.1                1.5               0.2
4                5.0               3.6                1.4               0.2
"""

print(data.shape)
# (150, 4)

# 데이터의 준비 0.2(20%)는 테스트용 데이터
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data,
    label,
    test_size=0.2, random_state=2022) 

# Classification: Logistic Regression 회귀 라인을 가지고 분류한다. 선형 회귀 모델의 변형 모델.
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

# 학습하기
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Regression 문제에서는 결정계수(R2)로 확인했다면, Classification 문제에서는 정확도(accuracy)로 판별
from sklearn.metrics import accuracy_score

# 테스트용 데이터의 실제값, 예측값
print('로지스틱 회귀, 정확도: {:.2f}'.format(accuracy_score(y_test, y_pred)))
# 로지스틱 회귀, 정확도: 0.97

