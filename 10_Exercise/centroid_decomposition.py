import numpy as np
from numpy import linalg as LA

def SSV(x,n,m):
	#Calculate maximizing sign vector z
	return z
	
#Calculate centroid decomposition 
def centroid_decomposition(x):
	n = x.shape[0]
	m = x.shape[1]
	L = R = []	
	for i in range (0, m):
		z = SSV(x,n,m)
		r = (np.transpose(x)*z)/LA.norm(np.transpose(x)*z)
		l = x*r
		if i==0:
			R=r
			L=l
		else:
			R=np.c_[R,r]
			L=np.c_[L,l]		
		x = x - l*np.transpose(r)		
	return L,R
	
	
input_matrix = []
with open("./senate_votes.txt") as f:
	for line in f:
		current_row=[int(x) for x in line.strip().split(' ')]
		input_matrix.append(current_row)
		
input_matrix = np.matrix(input_matrix)

       
L,R = centroid_decomposition(input_matrix)

print("L= " + str(L))
print("R=" + str(R))
