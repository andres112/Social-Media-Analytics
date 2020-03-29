import pandas as pd
import numpy as np
import logging
import plots
import warnings
import math

warnings.filterwarnings("ignore")  # ignore warnings in logs

logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO)  # Logging configuration

# User target and k neighbors
user = "1"
k = 20

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
    sum = 0
    for i in range(len(list(correlations))):
        if not math.isnan(ratings[i]):
            sum = sum+correlations[i]
    return sum

# Implementation of the Normalized score rating function, this is implemented  using the correlation
# the ratings and the mean values of the users


def prediction_normalized(correlations, ratings, k, user_mean=np.zeros(k), ru_mean = 0):
    # k is the neigborhood size
    if sumif(correlations.values.reshape(k), ratings) > 0:
        return ru_mean + (sumproduct(correlations.values.reshape(k), ratings, user_mean) / sumif(correlations.values.reshape(k), ratings))
    return np.nan

# ***************** Main Function **********************


def main(user, k=1):
    # Load data set
    logging.info('Loading Data Set')
    headers = ['userId', 'movieId', 'movie_categoryId',
               'reviewId', 'movieRating', 'reviewDate']
    columns = ['userId', 'movieId', 'movie_categoryId', 'movieRating']
    data_set = pd.read_csv('Dataset/movie-ratings.txt',
                           sep=',', names=headers, usecols=columns, dtype={"userId": "str", "movieId": "str"})

    print(data_set.info(verbose=False))

    logging.info('Getting mean value of movie rating by user')
    # average rating for each movie
    ratings = pd.DataFrame(data_set.groupby('movieId')['movieRating'].mean())

    print(ratings.sort_values('movieRating', ascending=False).head(10))

    # quantity of ratings per movie
    ratings['ratings_per_movie'] = data_set.groupby(
        'movieId')['movieRating'].count()

    # sorted by number of ratings
    print(ratings.sort_values('ratings_per_movie', ascending=False).head(10))

    # Plot
    plots.scatterPlot(ratings)

    logging.info('Getting movie matrix')
    # correlation matrix, between movies and users
    movie_matrix = data_set.pivot_table(
        index='movieId', columns="userId", values="movieRating")
    print(movie_matrix)

    logging.info('User mean values')
    # We get the average of all the movie ratings for each user
    user_mean = movie_matrix.mean(axis=0, skipna=True)
    # Print the user ratings mean value
    print(user_mean)

    # correlation between the movies: indicates the extent to which two or more variables fluctuate together
    # high correlation coefficient are the movies that are most similar to each other

    logging.info('Getting Pearson correlation for user {}'.format(user))
    # Pearson correlation coefficient: this will lie between -1 and 1
    # corr_matrix = movie_matrix.corr(method='pearson')
    corr_matrix = movie_matrix.corrwith(movie_matrix[user], method='pearson')
    print(corr_matrix)

    # The k similar users for user, the highest the correlation value, the more similar. Observe that the dataframe
    # has been sliced from the index 1, since in the index 0 the value will be 1.00 (self-correlation)
    logging.info('Getting k neighbors to user {}'.format(user))
    corr_top = (corr_matrix.sort_values(
        ascending=False)).drop([user]).dropna()[:k].to_frame().T
    
    print(corr_top)
    k = len(corr_top.columns) # Modify the neighbors if there are users with NaN similarities

    logging.info('Getting the mean of the k neighbors to user')
    # Taking the average ratings only for the top 5 neighbors of Arielle
    mean_top = []
    for namen in corr_top.columns:
        mean_top.append(user_mean[namen])
    print(mean_top)

    logging.info('Getting the ratings of the k neighbors to user')
    # This list is basically to select the rating of the k users
    selection_labels = corr_top.columns.tolist()
    # Here we are using the previous list to select the ratings
    rating_top = movie_matrix.loc[:, selection_labels]
    print(rating_top)

    logging.info('Predicted scores for user {}'.format(user))
    # Taking the average rating for the target user (Arielle)
    ru_mean = user_mean[user]
    prediction_results = []

    # We iterate over the rows of the top5 similar users ratings
    for index, row in rating_top.iterrows():
        # Getting the rating values for the movies of each user of the top 5
        ratings_row = row[selection_labels].values
        # Computing the prediction values, using the normalized model. We call this function sending as parameters
        # the correlation values of the k users and the ratings they have assigned to the items

        # with normalization
        pred_value = prediction_normalized(
            corr_top, ratings_row, k, user_mean, ru_mean)

        # without normalization
        # pred_value = prediction_normalized(
        #     corr_top, ratings_row, k)
        # List with the results of the prediction, we add a new result in each iteration, one per each item
        prediction_results.append(pred_value)

    # vector of predicted values for the user
    print(pd.DataFrame(prediction_results,
                       index=movie_matrix.index, columns=["Prediction"]).dropna() )


if __name__ == "__main__":
    
    main(user,k)
