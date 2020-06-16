import random
import pandas as pd
import numpy as np

data = pd.read_csv("task1.txt", names=["Edges"])
data["p(v,w)"] = pd.Series(np.random.random(len(data.index)))
print(data)

data_3 = pd.read_csv("task3.txt", names=["Nodes"])
data_3["threshold θ"] = pd.Series(np.random.random(len(data_3.index)))
print(data_3)

# Task 2
adjacency_matrix = pd.read_csv("task_2_adjacency.csv", sep=";", header=None)
degree_vector = pd.read_csv("task_2_degree.csv", sep=";", header=None)
age_vector = pd.read_csv("task_2_age.csv", sep=";", header=None)
indicator_matrix = pd.read_csv("indicator_matrix.csv", header=None)
degree_vector = degree_vector.T
age_vector = age_vector.T
print(adjacency_matrix)
print(degree_vector)
print(age_vector)


def calculateB():
    b_matrix = pd.DataFrame()
    for i in adjacency_matrix.columns:
        b_row = pd.Series([])
        for j in adjacency_matrix.columns:
            item = round(adjacency_matrix[i][j] -
                         (degree_vector[j]*degree_vector[i])/44, 3)
            b_row = b_row.append(item, ignore_index=True)

        b_matrix[i] = b_row

    return b_matrix


B_matrix = calculateB()
print(B_matrix)
modularity = indicator_matrix.transpose().dot(B_matrix.dot(indicator_matrix)).values

print(f'Modularity = {(modularity/44).sum()}')

def calculateXX():
    b_matrix = pd.DataFrame()
    for i in age_vector.columns:
        b_row = pd.Series([])
        for j in age_vector.columns:
            item = age_vector[i]*age_vector[j]
            b_row = b_row.append(item, ignore_index=True)
        b_matrix[i] = b_row
    return b_matrix

BXX_matrix = calculateXX()    
multiplicationResults = B_matrix.mul(BXX_matrix)
print(multiplicationResults)

agebyage = age_vector**2

adjbyage = pd.DataFrame()
for i in adjacency_matrix.columns:
    adjbyage[i] = agebyage[i].values * adjacency_matrix[i]

adjbyage = adjbyage.T

def calculateD():
    b_matrix = pd.DataFrame()
    for i in age_vector.columns:
        b_row = pd.Series([])
        for j in age_vector.columns:
            item = (degree_vector[i]*degree_vector[j]/44) * (age_vector[i]*age_vector[j])
            b_row = b_row.append(item, ignore_index=True)
        b_matrix[i] = b_row
    return b_matrix

D_matrix = calculateD()
print(D_matrix)

D_matrix.to_csv('results.txt', sep=';', encoding='utf-8', index=True, header=True)
