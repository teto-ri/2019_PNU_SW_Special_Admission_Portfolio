
# 데이터 통계 라이브러리
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 설문조사 데이터 읽기
df = pd.read_csv('score.csv',
               names = ["Nightstudy", "Sleep", "Game", "Academy", "Score"])

dataset = df.values

xs = np.array(dataset[:,1], dtype=np.float32)
ys = np.array(dataset[:,3], dtype=np.float32)
zs = np.array(dataset[:,4], dtype=np.float32)

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs)
ax.set_xlabel('Sleep')
ax.set_ylabel('Academy')
ax.set_zlabel('Score')
ax.view_init(15, 15)

plt.show()


