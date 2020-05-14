import numpy as np
import time
import logging

from support import data_loading_analysis as dla
import model_based_cf as mbcf
from support import evaluation_metrics as em

# Logging configuration
logging.basicConfig(format='%(message)s', level=logging.INFO)

# Parameters required
lambda_u = 0.02
lambda_v = 0.02
latent_dims = (5, 50)
learn_rate = 0.005
num_iters = 10000
bounds = (1, 5)

# Adding a file to save the results
results_pmf = open('results/results_pmf1', 'a+')

# Loading the dataset
path = 'Ciao-DVD-Datasets/movie-ratings.txt'
data = dla.load_dataset(path, pmf=True)
dla.get_information(data)
# Remove this columns, is not required in analysis
data = data.drop('genreID', axis=1) 

# Splitting the data into train(60%), validation(20%) and test(20%) sets
train_data, validation_data, test_data = dla.train_validate_test_split_pmf(data)

# Number of Users, Items(Movies)
num_users = data.userId.unique().shape[0]
num_items = data.movieId.unique().shape[0]

# Creating the rating matrix
rating_matrix = np.zeros([num_users, num_items])
for ele in train_data:
    rating_matrix[int(ele[0]), int(ele[1])] = float(ele[2])

print('Parameters for the model are: \nlambda_u: {:f}, \nlambda_v: {:f}, '
      '\nLearning rate: {:f}, \nNumber of iterations: {:d}'.format(lambda_u, lambda_v, learn_rate,
                                                                   num_iters))
# Saving the parameters to a file
print('Parameters for the model are: \nlambda_u: {:f}, \nlambda_v: {:f}, \nLearning rate: {:f},'
      ' \nNumber of iterations: {:d}'.format(lambda_u, lambda_v, learn_rate, num_iters), file=results_pmf)

for latent_dim in range(latent_dims[0], latent_dims[1]+1, 1):
    # Constructing the PMF model
    logging.info('\nBuilding the PMF model with {:d} latent dimensions....'.format(latent_dim))
    # Saving the latent dimensions to a file
    print('\nPMF model with {:d} latent dimensions....'.format(latent_dim), file=results_pmf)

    time_start = time.time()
    pmf_model = mbcf.PMF(rating_matrix=rating_matrix, lambda_u=lambda_u, lambda_v=lambda_v, latent_dim=latent_dim,
                         learn_rate=learn_rate, momentum=0.9, num_iters=num_iters, seed=1)
    U, V = pmf_model.train(train_data=train_data, validation_data=validation_data)
    time_elapsed = time.time() - time_start
    logging.info('Completed model building in {0:.5f} seconds'.format(time_elapsed))

    # Saving the build time to a file
    print('Time to build model: {0:.5f} seconds'.format(time_elapsed), file=results_pmf)

    logging.info('Testing the PMF model with {:d} latent dimensions....'.format(latent_dim))
    time_start = time.time()
    predictions = pmf_model.predict(data=test_data)
    time_elapsed = time.time() - time_start
    logging.info('Completed model testing in {0:.5f} seconds'.format(time_elapsed))
    # Saving the test time to a file
    print('Time to test model: {0:.5f} seconds'.format(time_elapsed), file=results_pmf)

    # Transforming the data to be with in the bounds
    low, high = bounds
    predictions[predictions < low] = low
    predictions[predictions > high] = high

    # Calculating the RMSE and MAE between the test data and the predicted data
    test_rmse = em.RMSE(test_data[:, 2], predictions)
    test_mae = em.MAE(test_data[:, 2], predictions)

    print('RMSE on test data: {:f}'.format(test_rmse))
    print('MAE on test data: {:f}'.format(test_mae))

    # Saving the errors to a file
    print('RMSE on test data: {:f}'.format(test_rmse), file=results_pmf)
    print('MAE on test data: {:f}'.format(test_mae), file=results_pmf)
