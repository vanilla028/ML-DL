import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
iris = load_iris()

data = iris.data
label = iris.target
columns = iris.feature_names

# data frame 만들기
data = pd.DataFrame(data, columns=columns)

# 데이터의 준비 0.2(20%)는 테스트용 데이터
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data,
    label,
    test_size=0.2, random_state=2022) 

# 랜덤 포레스트(Random Forest) 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=5)

# Regression 문제에서는 결정계수(R2)로 확인했다면, Classification 문제에서는 정확도(accuracy)로 판별
from sklearn.metrics import accuracy_score

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('랜덤 포레스트, 정확도: {:.2f}'.format(accuracy_score(y_test, y_pred)))
# 랜덤 포레스트, 정확도: 0.97
