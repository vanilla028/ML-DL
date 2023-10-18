import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_wine
wine = load_wine()

print(wine.DESCR)
"""
**Data Set Characteristics:**

    :Number of Instances: 178
    :Number of Attributes: 13 numeric, predictive attributes and the class
    :Attribute Information:
                - Alcohol
                - Malic acid
                - Ash
                - Alcalinity of ash
                - Magnesium
                - Total phenols
                - Flavanoids
                - Nonflavanoid phenols
                - Proanthocyanins
                - Color intensity
                - Hue
                - OD280/OD315 of diluted wines
                - Proline

    - class:
            - class_0
            - class_1
            - class_2

    :Summary Statistics:

    ============================= ==== ===== ======= =====
                                   Min   Max   Mean     SD
    ============================= ==== ===== ======= =====
    Alcohol:                      11.0  14.8    13.0   0.8
    Malic Acid:                   0.74  5.80    2.34  1.12
    Ash:                          1.36  3.23    2.36  0.27
    Alcalinity of Ash:            10.6  30.0    19.5   3.3
    Magnesium:                    70.0 162.0    99.7  14.3
    Total Phenols:                0.98  3.88    2.29  0.63
    Flavanoids:                   0.34  5.08    2.03  1.00
    Nonflavanoid Phenols:         0.13  0.66    0.36  0.12
    Proanthocyanins:              0.41  3.58    1.59  0.57
    Colour Intensity:              1.3  13.0     5.1   2.3
    Hue:                          0.48  1.71    0.96  0.23
    OD280/OD315 of diluted wines: 1.27  4.00    2.61  0.71
    Proline:                       278  1680     746   315
    ============================= ==== ===== ======= =====
"""

data = wine.data
label = wine.target
columns = wine.feature_names

data = pd.DataFrame(data, columns=columns)
print(data.head())
"""
   alcohol  malic_acid   ash  alcalinity_of_ash  magnesium  total_phenols  flavanoids  nonflavanoid_phenols  proanthocyanins  color_intensity   hue  od280/od315_of_diluted_wines  proline
0    14.23        1.71  2.43               15.6      127.0           2.80        3.06                  0.28             2.29             5.64  1.04                          3.92   1065.0
1    13.20        1.78  2.14               11.2      100.0           2.65        2.76                  0.26             1.28             4.38  1.05                          3.40   1050.0
2    13.16        2.36  2.67               18.6      101.0           2.80        3.24                  0.30             2.81             5.68  1.03                          3.17   1185.0
3    14.37        1.95  2.50               16.8      113.0           3.85        3.49                  0.24             2.18             7.80  0.86                          3.45   1480.0
4    13.24        2.59  2.87               21.0      118.0           2.80        2.69                  0.39             1.82             4.32  1.04                          2.93    735.0
"""

# Unsupervised Learning: K-Means

# 데이터 전처리 필요 -> MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

print(data.shape)
# (178, 13)

# 13차원에서는 패턴을 찾기 힘드므로 차원을 축소한다
# PCA(차원의 축소)
# PCA (차원의 축소)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data = pca.fit_transform(data)
print(data)
"""
[[-7.06335756e-01 -2.53192753e-01]
 [-4.84976802e-01 -8.82289142e-03]
 [-5.21172266e-01 -1.89187222e-01]
 ...
 [ 5.72991102e-01 -4.25516087e-01]  
 [ 7.01763997e-01 -5.13504983e-01]]

==> 13차원의 데이터가 2차원으로 변환되었다. 전처리 끝.
"""

# Unsupervised learning: k-Means clustering

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3) # 세 종류이므로 3의 값을 넣어준다.
kmeans.fit(data)
cluster = kmeans.predict(data)
print(cluster)
"""
[2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 0 1 1 1 1 2 1 1 1 1 2 1 2
 2 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 2 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
"""

# 데이터 시각화하기
plt.scatter(data[:,0],data[:,1], c=cluster, edgecolor='black',linewidth=1)
plt.show()

# Silhouette
from sklearn.metrics import silhouette_score

best_n = 1
best_score = -1

for n_cluster in range(2,11):
  kmeans = KMeans(n_clusters=n_cluster)
  kmeans.fit(data)
  cluster = kmeans.predict(data)
  score = silhouette_score(data, cluster)

  print('클러스터의 수: {} 실루엣 점수:{:.2f}'.format(n_cluster, score))

  if score > best_score:
    best_n = n_cluster
    best_score = score

print('가장 높은 실루엣 점수를 가진 클러스터 수: {}, 실루엣 점수 {:.2f}'.format(best_n, best_score))
"""
클러스터의 수: 2 실루엣 점수:0.49
클러스터의 수: 3 실루엣 점수:0.57
클러스터의 수: 4 실루엣 점수:0.49
클러스터의 수: 5 실루엣 점수:0.46
클러스터의 수: 6 실루엣 점수:0.43
클러스터의 수: 7 실루엣 점수:0.40
클러스터의 수: 8 실루엣 점수:0.40
클러스터의 수: 9 실루엣 점수:0.38
클러스터의 수: 10 실루엣 점수:0.38
가장 높은 실루엣 점수를 가진 클러스터 수: 3, 실루엣 점수 0.57
"""


