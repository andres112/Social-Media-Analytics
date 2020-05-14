import logging
import time
import numpy as np
import copy

from support import evaluation_metrics as em

# Logging configuration
logging.basicConfig(format='%(message)s', level=logging.INFO)
# Adding a file to save the results
results_pmf = open('results/results_pmf1', 'a+')


class PMF:
    """Probabilistic Matrix Factorization"""

    def __init__(self, rating_matrix, lambda_u, lambda_v, latent_dim, learn_rate, momentum, num_iters, seed):
        """Initializing the parameters"""
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.momentum = momentum
        self.rating_matrix = rating_matrix
        from numpy.random import RandomState
        self.random_state = RandomState(seed)
        self.iterations = num_iters
        self.learn_rate = learn_rate

        # Creating the Identity matrix
        self.I = copy.deepcopy(self.rating_matrix)
        self.I[self.I != 0] = 1
        # Creating arrays of shape NxK, MxK and populating them with random samples from the uniform distribution [0, 1]
        self.U = 0.1 * self.random_state.rand(np.size(rating_matrix, 0), latent_dim)
        self.V = 0.1 * self.random_state.rand(np.size(rating_matrix, 1), latent_dim)

    def train(self, train_data, validation_data):
        """Train tne model using the train data and fine tuning to eliminate over fitting using the validation data"""
        prev_validation_rmse = None

        # Momentum helps accelerate gradients vectors in the right directions leading to faster converging.
        momentum_U = np.zeros(self.U.shape)
        momentum_V = np.zeros(self.V.shape)

        for iter in range(self.iterations):
            time_start = time.time()
            # Computing the partial derivative w.r.t. U
            derv_U = np.dot(self.I * (self.rating_matrix - np.dot(self.U, self.V.T)), -self.V) + self.lambda_u * self.U

            # Computing the partial derivative w.r.t. V
            derv_V = np.dot((self.I * (self.rating_matrix - np.dot(self.U, self.V.T))).T,
                            -self.U) + self.lambda_v * self.V

            # Updating U and V accordingly
            momentum_U = (self.momentum * momentum_U) + self.learn_rate * derv_U
            momentum_V = (self.momentum * momentum_V) + self.learn_rate * derv_V
            self.U = self.U - momentum_U
            self.V = self.V - momentum_V

            # Calculating the loss function
            train_loss = self.loss()

            # Predicting the validation data and calculating the RMSE error
            validation_pred = self.predict(validation_data)
            validation_rmse = em.RMSE(validation_data[:, 2], validation_pred)

            print('Iteration:{: d}, Loss:{: f}, Validation RMSE:{: f}, Time:{: f} seconds'
                  .format(iter + 1, train_loss, validation_rmse, time.time() - time_start))
            # Saving the iterations to the file
            #print('Iteration:{: d}, Loss:{: f}, Validation RMSE:{: f}, Time:{: f} seconds'
             #     .format(iter + 1, train_loss, validation_rmse, time.time() - time_start), file=results_pmf)

            # If the current iteration has an higher or equal error to the previous iteration, it means that the
            # local minima is at the previous iteration.
            if prev_validation_rmse and (prev_validation_rmse - validation_rmse) <= 0:
                print('The model converged at iteration: {: d}'.format(iter + 1))
                # Saving the iterations to the file
                print('The model converged at iteration: {: d}'.format(iter + 1), file = results_pmf)
                break
            else:
                prev_validation_rmse = validation_rmse

        return self.U, self.V

    def loss(self):
        """Loss function for the PMF"""
        loss = np.sum(self.I * (self.rating_matrix - np.dot(self.U, self.V.T)) ** 2) + self.lambda_u * np.sum(
            np.square(self.U)) + self.lambda_v * np.sum(np.square(self.V))
        return loss

    def predict(self, data):
        """Predicting the new values"""
        # Getting the Users and Items from the Validation data
        index_data = np.array([[int(ele[0]), int(ele[1])] for ele in data], dtype=int)
        # Pick the data that matches the users and items  from U and V matrices (from the index_data)
        user_features = self.U.take(index_data.take(0, axis=1), axis=0)
        item_features = self.V.take(index_data.take(1, axis=1), axis=0)
        predictions = np.sum(user_features * item_features, 1)
        return predictions
