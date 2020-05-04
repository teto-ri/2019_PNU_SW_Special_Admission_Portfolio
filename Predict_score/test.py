import numpy as np
from tqdm import tqdm_notebook
import pandas as pd

df = pd.read_csv('score.csv')

x_data = np.array(df[0:4], dtype=np.float32)
y_data = np.array(df[:,4], dtype=np.float32)

print(x_data)
