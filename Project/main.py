import pandas as pd
import numpy as np
import logging
import warnings
import math
import asyncio
import time

import plots
import splitData
import rmse

warnings.filterwarnings("ignore")  # ignore warnings in logs

logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO)  # Logging configuration

# Define the k neighbors
k = 3

bounds = (1, 5)

# SupportFunctions

# This function receives the correlation values and a list of ratings, if  the rating is different than Nan
# the we multiply  the value of the correlations by the rating otherwise no. The sum of all the operations is returned.


def sumproduct(correlations, ratings, user_mean):
    sum = 0
    for i in range(len(list(correlations))):
        if not math.isnan(ratings[i]):
            sum = sum+(correlations[i]*(ratings[i] - user_mean[i]))
    return sum

# This function receives the correlation values and a list of ratings, if  the rating is different than Nan
# the we sum up the value of the correlations, otherwise no. The value of the sum is returned


def sumif(correlations, ratings):
    ratings_isnan = np.isfinite(ratings)
    sum = (correlations * ratings_isnan).sum()
    return sum

# Implementation of the Normalized score rating function, this is implemented  using the correlation
# the ratings and the mean values of the users


def prediction_normalized(correlations, ratings, k, user_mean=np.zeros(k), ru_mean=0):
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

    print(ratings.describe())
    print(ratings.sort_values('movieRating', ascending=False).head(10))

    # quantity of ratings per movie
    ratings['ratings_per_movie'] = data_set.groupby(
        'movieId')['movieRating'].count()

    # sorted by number of ratings
    print(ratings.sort_values('ratings_per_movie', ascending=False).head(10))

    # Plot number of movies per rating
    plots.scatterPlot(ratings)

    logging.info('Getting movie matrix')
    # correlation matrix, between movies and users
    movie_matrix = data_set.pivot_table(
        index='movieId', columns="userId", values="movieRating")

    print(movie_matrix)
    return movie_matrix


def get_user_mean(movie_matrix):
    logging.info('User mean values')
    # We get the average of all the movie ratings for each user
    user_mean = movie_matrix.mean(axis=0, skipna=True)
    # Print the user ratings mean value
    print(user_mean)

    # Plot number of movies per rating
    plots.avg_ratings_per_user(user_mean)
    return user_mean


async def get_pearson_correlation(movie_matrix, user):
    # TODO: is this assumption correct?
    # The reason to fill nan with 0 is to reduce the sparcity
    movie_matrix = movie_matrix.fillna(0)

    # correlation between the movies: indicates the extent to which two or more variables fluctuate together
    # high correlation coefficient are the movies that are most similar to each other

    # ************** Pearson Coefficient Correlation ********************
    logging.info('Getting Pearson correlation for user {}'.format(user))
    # Pearson correlation coefficient: this will lie between -1 and 1
    pearson_corr = movie_matrix.corrwith(movie_matrix[user], method='pearson')
    # print(f"Correlation for user{user}\n{pearson_corr}")

    return pearson_corr


async def get_prediction(movie_matrix, pearson_corr, user, k=1):
    # The k similar users for user, the highest the correlation value, the more similar. Observe that the dataframe
    # has been sliced from the index 1, since in the index 0 the value will be 1.00 (self-correlation)
    logging.info('Getting k neighbors to user {}'.format(user))
    corr_top = pearson_corr.sort_values(ascending=False)

    # TODO: this validation should be keep?
    if 0 < len(corr_top) > corr_top.isnull().sum():
        corr_top = corr_top.drop([user])[:k].to_frame().T
    else:
        a = np.empty(len(movie_matrix))
        a[:] = np.nan
        return a  # if there is not neighbors or the correlation is nan return a vector of nan
    # print(corr_top)

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

        # TODO: This section set limits to the predicted data, is it required?
        # pred_value = bounds[0] if pred_value < bounds[0] else (
        #     bounds[1] if pred_value > bounds[1] else pred_value)

        prediction_results = prediction_results.append(pd.Series({user: pred_value}, name=item))
    return prediction_results


async def main(movie_matrix, user, k):
    pearson_corr = await get_pearson_correlation(movie_matrix, user)
    prediction = await get_prediction(movie_matrix, pearson_corr, user, k)
    logging.info(f'Prediction for user {user} is done!')
    return prediction


if __name__ == "__main__":
    # Matrix of movie ratings
    movie_matrix = get_movie_matrix()

    # Get all user mean values
    user_mean = get_user_mean(movie_matrix)

    # Split dataset: 80% training 20% testing
    logging.info(f'Spliting Dataset between training and testing data')
    start = time.time()
    train_data, test_data = splitData.split_train_test(movie_matrix, 0.2)
    logging.info("Process done in: {0:.2f} seconds".format(
        time.time() - start))

    start = time.time()
    prediction_matrix = pd.DataFrame(index=train_data.index)
    for name, data in train_data.iteritems():
        prediction_matrix[name] = asyncio.run(main(train_data, name, k))
        # # TODO: this validation should be keep?
        # if(not train_data[name].isnull().all()):
        #     prediction_matrix[name] = asyncio.run(main(train_data, name, k))
        # else:
        #     a = np.empty(len(train_data))
        #     a[:] = np.nan
        #     prediction_matrix[name] = a
    logging.info("Process done in: {0:.2f} seconds".format(
        time.time() - start))

    print("train Matrix \n", train_data)
    print("test Matrix \n", test_data)
    print("Prediction Matrix \n", prediction_matrix)

    # prediction_matrix.to_csv('results/prediction_matrix.txt', sep=';',
    #                          encoding='utf-8', index=True, header=True, float_format='%.2f')

    rmse_value = rmse.get_rmse(test_data, prediction_matrix)
    print(f'RMSE:\t{rmse_value}')
