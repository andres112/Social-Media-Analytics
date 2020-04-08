import random
import pandas as pd
import numpy as np

data = pd.read_csv("task1.txt", names=["Edges"])
data["p(v,w)"] = pd.Series(np.random.random(len(data.index)))
print(data)

data_3 = pd.read_csv("task3.txt", names=["Nodes"])
data_3["threshold θ"] = pd.Series(np.random.random(len(data_3.index)))
print(data_3)

