import numpy as np
from numpy import linalg as LA


def SSV(x, n, m):
    sign = -1
    while True:
        # Check if sign is negative
        if sign == -1:
            z = np.matrix(np.ones((n, 1)), dtype=np.float)
            z = np.ones((n, 1))
        else:
            z[sign] = z[sign]*-1  # Change the sign of the z element

        # Calculate the vector v
        v = np.zeros([n, 1])
        for i in range(0, n):
            v[i] = np.dot(np.dot(x[i], x.T), z) - \
                np.dot(z[i], np.dot(x[i], x.T[:, i]))

        # Find the greatest absolute element with diferent sign than z element
        val = 0
        sign = -1
        for i in range(0, n):
            if z[i]*v[i] < 0:
                if abs(v[i]) > abs(val):
                    val = v[i]
                    sign = i
                # The sign works as a flag in this step, to stop the iterations for z maximize
        if sign == -1:
            break
    return z

# Calculate centroid decomposition


def centroid_decomposition(x):
    n = x.shape[0]
    m = x.shape[1]
    L = R = []
    for i in range(0, m):
        z = SSV(x, n, m)
        r = np.dot(np.transpose(x), z)/LA.norm(np.dot(np.transpose(x), z))
        l = np.dot(x, r)
        if i == 0:
            R = r
            L = l
        else:
            R = np.c_[R, r]
            L = np.c_[L, l]
        x = x - np.dot(l, r.T)
        print(f"Iteration {i+1} of {m}")
    return L, R


input_matrix = []
with open("./senate_votes.txt") as f:
    for line in f:
        current_row = [int(x) for x in line.strip().split(' ')]
        input_matrix.append(current_row)

input_matrix = np.array(input_matrix)

L, R = centroid_decomposition(input_matrix)

print("L= " + str(L))
print("R=" + str(R))
