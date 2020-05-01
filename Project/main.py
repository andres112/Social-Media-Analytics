import pandas as pd
import numpy as np
import logging
import warnings
import math
import asyncio
import time

import plots
import splitData
import metrics

warnings.filterwarnings("ignore")  # ignore warnings in logs

logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO)  # Logging configuration

## Setting variables
k = 5 # Define the k neighbors
bounds = (1, 5) # max and min boundaries
threshold = 0 # Threshold for similarity neighborhood

# SupportFunctions

# This function receives the correlation values and a list of ratings, if  the rating is different than Nan
# the we multiply  the value of the correlations by the rating otherwise no. The sum of all the operations is returned.


def sumproduct(correlations, ratings, user_mean):
    sum = 0
    sum = np.nansum(correlations*(ratings - user_mean))
    return sum

# This function receives the correlation values and a list of ratings, if  the rating is different than Nan
# the we sum up the value of the correlations, otherwise no. The value of the sum is returned


def sumif(correlations, ratings):
    # sum all correlation values for users who have rated the item before
    sum = 0
    ratings_notnan = np.isfinite(ratings)
    sum = np.nansum(correlations * ratings_notnan)
    return sum

# Implementation of the Normalized score rating function, this is implemented  using the correlation
# the ratings and the mean values of the users


def prediction_normalized(correlations, ratings, k, user_mean=np.zeros(k), ru_mean=0):
    ratings = np.array(ratings, dtype=float)
    # k is the neigborhood size
    if correlations.shape[1] > 0 and sumif(correlations.values.reshape(k), ratings) > 0:
        return ru_mean + (sumproduct(correlations.values.reshape(k), ratings, user_mean) / sumif(correlations.values.reshape(k), ratings))
    return np.nan

# ***************** Main Function **********************


def get_movie_matrix():
    # Load data set
    logging.info('Loading Data Set')
    headers = ['userId', 'movieId', 'movie_categoryId',
               'reviewId', 'movieRating', 'reviewDate']
    columns = ['userId', 'movieId', 'movie_categoryId', 'movieRating']
    data_set = pd.read_csv('Dataset/movie-ratings_test.txt',
                           sep=',', names=headers, usecols=columns, dtype={"userId": "str", "movieId": "str"})

    num_users = data_set.userId.unique().shape[0]
    num_items = data_set.movieId.unique().shape[0]
    sparsity = 1 - len(data_set) / (num_users * num_items)
    print(f"Users: {num_users}\nMovies: {num_items}\nSparsity: {sparsity}")

    logging.info('Getting mean value of movie rating by user')
    # average rating for each movie
    ratings = pd.DataFrame(data_set.groupby('movieId')['movieRating'].mean())

    print(ratings.describe().T)

    # quantity of ratings per movie
    ratings['ratings_per_movie'] = data_set.groupby(
        'movieId')['movieRating'].count()

    # sorted by number of ratings
    print(ratings.sort_values('ratings_per_movie', ascending=False).head(10))

    settings = {'axisX':'movieRating', 'axisY': 'ratings_per_movie', 'topic': 'Movie'}
    plots.scatterPlot(ratings, settings)

    logging.info('Getting Train and Test matrices')
    train_data, test_data = splitData.split_train_test(data_set, 0.2)

    print("Train data\n",train_data)
    print("Test data\n",test_data)
    return train_data, test_data


def get_user_mean(movie_matrix):
    logging.info('User mean values')
    # We get the average of all the movie ratings for each user
    user_mean = movie_matrix.mean(axis=0, skipna=True)
    # Print the user ratings mean value
    print(user_mean)

    # Plot number of movies per rating
    plots.avg_ratings_per_user(user_mean)
    return user_mean


def get_pearson_correlation(movie_matrix, user):
    # correlation between the movies: indicates the extent to which two or more variables fluctuate together
    # high correlation coefficient are the movies that are most similar to each other

    # ************** Pearson Coefficient Correlation ********************
    logging.info('Getting Pearson correlation for user {}'.format(user))
    # Pearson correlation coefficient: this will lie between -1 and 1
    pearson_corr = movie_matrix.corrwith(movie_matrix[user], method='pearson')
    # Here below the pearson correlation computation from scratch is commented because 
    # the performance is lower than the pandas's corrwith method used above
    """
    pearson_corr = pd.DataFrame(columns = movie_matrix.columns)
    for i in movie_matrix.columns:
        user_1 = movie_matrix[user][:]
        user_2 = movie_matrix[i][:]
        nans = ~np.logical_or(np.isnan(user_1), np.isnan(user_2))
        user_1 = np.compress(nans.values, user_1.values)
        user_2 = np.compress(nans.values, user_2.values)
        
        # Numerator 
        user1_product = user_1 - user_1.mean()
        user2_product = user_2 - user_2.mean()
        top = (user1_product*user2_product).sum()
        
        # Denominator 
        user1_sqr = np.sqrt((user1_product**2).sum())
        user2_sqr = np.sqrt((user2_product**2).sum())
        bottom = user1_sqr * user2_sqr
        
        correl = top/bottom
        pearson_corr[i] = [correl]"""

    return pearson_corr

##*****************************************************************##

def get_prediction(movie_matrix, pearson_corr, user, k=1):
    # The k similar users for user, the highest the correlation value, the more similar. Observe that the dataframe
    # has been sliced from the index 1, since in the index 0 the value will be 1.00 (self-correlation)
    logging.info('Getting k neighbors to user {}'.format(user))
    corr_top = pearson_corr.sort_values(ascending=False)

    # Neighborhood selection: Based on
    # Threshold and no NaN correlation values
    # TODO: this validation should be keep?
    if 0 < len(corr_top) and ~pd.isnull(corr_top).all():
        top = set(corr_top[(corr_top.iloc[0:] < threshold)].index.values)
        top.add(user)
        corr_top = corr_top.drop(top)[:k].to_frame().T
    else:
        a = np.empty(len(movie_matrix))
        a[:] = np.nan
        return a  # if there is not neighbors or the correlation is nan return a vector of nan

    print(f"Neighborhood size: {corr_top.count().sum()}\n",corr_top)

    logging.info('Getting the ratings of the k neighbors to user')
    # This list is basically to select the rating of the k users
    selection_labels = corr_top.columns.tolist()
    # Here we are using the previous list to select the ratings
    rating_top = movie_matrix.loc[:, selection_labels]
    # print(rating_top)

    logging.info('Getting the mean of the k neighbors to user')
    # Taking the average ratings only for the top k neighbors
    mean_top = user_mean[selection_labels]
    # print(mean_top)

    logging.info('Getting predicted scores for user {}'.format(user))
    # Taking the average rating for the target user
    ru_mean = user_mean[user]

    prediction_results = pd.DataFrame()

    # We iterate over the rows of the top k similar users ratings
    for item, row in rating_top.iterrows():
        # Getting the rating values for the movies of each user of the top 
        ratings_row = row[selection_labels].values
        # Computing the prediction values, using the normalized model. We call this function sending as parameters
        # the correlation values of the k users and the ratings they have assigned to the items

        # with normalization
        pred_value = prediction_normalized(
            corr_top, ratings_row, k, mean_top, ru_mean)

        # limit the results to the min and max bounds
        pred_value = bounds[0] if pred_value < bounds[0] else (
            bounds[1] if pred_value > bounds[1] else pred_value)

        prediction_results = prediction_results.append(pd.Series({user: round(pred_value,2)}, name=item))
    return prediction_results


def main(movie_matrix, user, k):
    pearson_corr = get_pearson_correlation(movie_matrix, user)
    prediction = get_prediction(movie_matrix, pearson_corr, user, k)
    logging.info(f'Prediction for user {user} is done!')
    return prediction


if __name__ == "__main__":
    # Matrix of movie ratings
    train_data, test_data = get_movie_matrix()

    # Get all user mean values
    user_mean = get_user_mean(train_data)

    start = time.time()
    prediction_matrix = pd.DataFrame(index=train_data.index)
    for name, data in train_data.iteritems():
        # name = user ###TODO: delete and replace for N random user
        prediction_matrix[name] = main(train_data, name, k)
        # break
    logging.info("Process done in: {0:.2f} seconds".format(
        time.time() - start))

    small_pred = prediction_matrix.dropna(axis=1, how='all').dropna(how='all')
    print("Prediction Matrix \n", small_pred)
    print("Test Matrix \n", test_data.dropna(axis=1, how='all').dropna(how='all'))

    logging.info('\nMetric Calculations RMSE and MAE')
    rmse_value = metrics.rmse(test_data, prediction_matrix)
    print(f'RMSE:\t{rmse_value}')

    mae_value = metrics.mae(test_data, prediction_matrix)
    print(f'MAE:\t{mae_value}')