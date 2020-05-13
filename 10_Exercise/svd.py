import numpy as np
import pandas as pd

X = np.genfromtxt('senate_votes.txt')
# Get the U matrix, s Singular Values and V transposed matrix
U, s, VT = np.linalg.svd(X, full_matrices=False)

# calculate the total sum of the squared singular values
total_sum = (s**2).sum()

#################
# Energy Method
current_sum = 0 # Handle de current sum of the squares
threshold_item = 0 # singular value where the 0.8 of total sum is reached
while(current_sum < (0.85*total_sum)):
    current_sum += s[threshold_item]**2
    threshold_item += 1

s_1 = s.copy()
s_1[threshold_item:] = 0 # remove the aditional singular values

#################
# Entropy-Based Method
f= (s**2)/total_sum
s_f = -np.multiply(f, np.log2(f)).sum()
E = s_f/np.log2(len(s))

current_sum = 0 # Handle de current sum of relative contribution values σk
threshold_item = 0 # where the condition of Sumation of smallest values of f is < than E.
while(current_sum < E): 
    current_sum += f[threshold_item]  
    threshold_item += 1      

s_2 = s.copy()
s_2[threshold_item:]= 0 # remove the aditional singular values

#################
# voting matrix after dimensionality reduction
Sigma = np.diag(s_1)
new_X = np.dot(U, np.dot(Sigma, VT))
print("Energy Method Reduction\n",pd.DataFrame(new_X))

# voting matrix after dimensionality reduction
Sigma = np.diag(s_2)
new_X = np.dot(U, np.dot(Sigma, VT))
print("\nEntropy Method Reduction\n",pd.DataFrame(new_X))
