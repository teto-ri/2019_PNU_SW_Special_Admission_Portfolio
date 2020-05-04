import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('score.csv',
                 names = ["Nightstudy", "Sleep", "Game", "Academy", "Score"])

print(df.info())
print(df.describe())
print(df)

plt.figure(figsize=(12,12))

sns.heatmap(df.corr(), linewidths = 0.1, vmax = 0.5, cmap = plt.cm.gist_heat,
            linecolor = 'white', annot = True)

plt.show()
