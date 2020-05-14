from support import data_loading_analysis as dla
from memory_based_cf import MeB
from support import evaluation_metrics as em

import time
import logging
import warnings
import pandas as pd

warnings.filterwarnings("ignore")  # ignore warnings in logs

logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO)  # Logging configuration

# Loading Dataset
path = 'Ciao-DVD-Datasets/movie-ratings.txt'
data_set = dla.load_dataset(path)

#dataset information
print('Original Dataset Information')
dla.get_information(data_set)

# Plots
ratings = dla.ratings_analysis(data_set, True)
genres = dla.genre_analysis(data_set, True)
users_avg = dla.user_analysis(data_set, True)
# dla.sparsity_analysis(data_set)

# User and Movie treshold to be considered as relevant for dataset (this helps to reduce the dataset size)
p_user = 1 # threshold to consider a user relevant
p_movie = 100 # threshold to consider a movie popular
p_rnd = 1 # percentage of data to prune 
how = 'm' # Prune by user 'u', by movie 'm' or randomly 'r'

# Here It is got the pruned dataset acording previous parameters
start = time.time()
p_data_set = dla.prune_dataset(data_set, users_avg, ratings, p_user, p_movie, p_rnd, how)
logging.info("Process done in: {0:.5f} seconds".format(time.time() - start))
p_data_set.shape
# New dataset information
print('Pruned Dataset Information')
dla.get_information(p_data_set)

# Rating matrix (only the user ratings per movie)
start = time.time()
R_matrix = p_data_set.pivot_table(index = 'userId', columns ='movieId', values = 'movieRating')
logging.info("Rating Matrix done in: {0:.5f} seconds".format(time.time() - start))
index_names = R_matrix.index # Save the users Id
columns_names = R_matrix.columns # Save the movies Id
R_matrix = R_matrix.values # Get only the ratings
print("Rating matrix shape: ",R_matrix.shape)

# Split Rating matrix into training and testing datasets
start = time.time()
train,test = dla.split_train_test_custom(R_matrix,0.2)
logging.info("Spliting dataset done in: {0:.5f} seconds".format(time.time() - start))

########################################################################################
## User based Collaborative Filtering Recommender with pearson correlation coefficient##
# Memory Based RS instance.
bounds = (1, 5)  # max and min boundaries
u2u_pc = MeB(train, index_names, columns_names, bounds)
# user average list
u2u_pc.get_user_avgs()

# Pearson Correlation Coeficient
print("Pearson Correlation step")
start = time.time()
u2u_pc.pearson_correlation()
logging.info("Pearson Correlation done in: {0:.5f} seconds".format(time.time() - start))

# Neighborhood selection based on k size
def get_neighborhood(k_size):
    k = k_size  # Define the k neighbors size
    start = time.time()
    u2u_pc.neighborhood(k)  
    end = time.time() - start
    logging.info(f"Neighborhood for k-size= {k} done in: {end:.5f} seconds")
    return end

# Prediction Computing: Mean-Centering normalization
def get_prediction():
    start = time.time()
    prediction = u2u_pc.predict()
    end = time.time() - start
    logging.info(f"Prediction for k-size= {k} done in: {end:.5f} seconds")
    return prediction, end

# RMSE and MAE metrics evaluation
logging.info('Metrics evaluation')
def get_metrics(prediction):
    start = time.time()
    rmse = em.RMSE(test,prediction)
    mae = em.MAE(test,prediction)
    print('RMSE: ',rmse)
    print('MAE: ',mae)
    end = time.time() - start
    logging.info(f"Metrics for k-size= {k} size done in: {end:.5f} seconds")
    return rmse, mae

# Implementation of loop for get the results for different k-size values
n_times = {} ### Handle neighborhood calculation time
p_times = {} ### Handle prediction calculation time
metrics = {} ### Handle metrics calculation time
for k in range(5, 50, 10):
   n_time = get_neighborhood(k)
   n_times[k] = n_time
   prediction, p_time = get_prediction()
   p_times[k] = p_time
   rmse, mae = get_metrics(prediction)
   metrics[k] = (rmse,mae)

# External storing of results
# pd.DataFrame(metrics, index=['rmse','mae']).T.to_csv('results/metrics.csv',float_format='%.5f')
# pd.DataFrame(n_times, index=['time']).T.to_csv('results/neighborhood_time.csv',float_format='%.2f')
# pd.DataFrame(p_times, index=['time']).T.to_csv('results/prediction_time.csv',float_format='%.2f')