import numpy
import pandas as pd


def matrix_factorization(R, U, V, K, max_iter=5000, alpha=0.001, lambda_=0.02):
    '''
    :param R: user(row)-item(column) matrix. R similar to UxV^T
    :param U: |user| x k matrix. Each row of U represents the associations between a user and the features
    :param V: |item| x k matrix. Each row of V represents the associations between an item and the features
    :param K: number of features
    :param max_iter: number of iterations
    :param alpha: learning rate
    :param lambda_: regularization term for U and V
    :return: updated matrices U and V
    '''
    V = V.T  # V factor matrix Transposed
    for current_iter in range(max_iter):
        for i in range(len(R)):  # R rows iterator
            for j in range(len(R[i])):  # R column iterator
                if R[i][j] > 0:  # indicator function
                    # error computation
                    eij = R[i][j] - numpy.dot(U[i, :], V[:, j])
                    for k in range(K):
                        # Computing the partial derivative w.r.t. U
                        Ueij = -(eij)*V[k][j] + lambda_ * U[i][k]
                        # Computing the partial derivative w.r.t. V
                        Veij = -(eij)*U[i][k] + lambda_ * V[k][j]
                        # Update U
                        U[i][k] = U[i][k] - alpha*Ueij
                        # Update V
                        V[k][j] = V[k][j] - alpha*Veij

        # This is the predicted Rating matrix after dot product between the 2 factor matrices
        current_Rating = numpy.dot(U, V)

        error = 0
        counter = 0

        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    error = error + (R[i][j] - numpy.dot(U[i, :], V[:, j]))**2
                    # compute the overall error (objective loss function)
                    for k in range(K):
                        error = error + (lambda_/2)*(U[i][k]**2 + V[k][j]**2)
                    counter = counter + 1

        average_error = error / counter
        print("{} iteration: average error {}, Total error {}".format(
            current_iter + 1, average_error, error))

        # stop criteria
        if error < 0.4:
            break

    return U, V.T


def main():
    R_observed = [
        [4, 4, 5, 3, 5],
        [5, 5, 3, 0, 4],
        [5, 0, 2, 5, 3],
        [5, 4, 3, 4, 0],
        [4, 3, 0, 3, 5],
        [4, 5, 4, 5, 5],
    ]

    R_observed = numpy.array(R_observed)

    rows_number = len(R_observed)  # number of rows (users)
    colums_number = len(R_observed[0])  # number of columns (items)
    K = 3

    # how to obtain U and V: initialize the two matrices with some values, calculate how 'different' their product is to r, and then try to minimize this difference iteratively (gradient descent).
    U = numpy.random.rand(rows_number, K)
    V = numpy.random.rand(colums_number, K)

    new_U, new_V = matrix_factorization(R_observed, U, V, K)
    R_predicted = numpy.dot(new_U, new_V.T)

    movies = ["Forrest Gump", "Intouchables",
              "Fight Club", "Lion King", "Pulp Fiction"]
    users = ["Michael", "Paul", "Ann", "Julie", "Pierre", "Sophie"]

    R_predicted = pd.DataFrame(R_predicted, columns=movies, index=users).round(0)
    print(R_predicted)


if __name__ == "__main__":
    main()
